from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from arc_tactic3.language_fastlearn_benchmark import set_global_seed
from arc_tactic3.language_nanochat_actual_compare import _train_candidate
from arc_tactic3.language_realtext_microbench import RealTextConfig, TokenBlockDataset, _build_train_batch_schedule
from arc_tactic3.language_recurrent_memory_rewrites import ChunkedTokenMemoryPartialLM
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM, _top_token_ids


@dataclass(frozen=True, slots=True)
class LearnedCompressorCandidateConfig:
    cache_path: Path
    tokenizer_name: str = "gpt2"
    train_blocks: int = 1024
    val_blocks: int = 64
    sequence_length: int = 127
    batch_size: int = 16
    eval_batch_size: int = 32
    train_steps: int = 16
    eval_interval: int = 8
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 13
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    tensor_batching: bool = False
    cache_dataset_on_device: bool = False
    compute_val_bpb: bool = False
    paired_train_batches: bool = True
    reseed_per_model: bool = True
    train_schedule_seed: int | None = None
    optimizer_recipe: str = "default"
    warmup_steps: int = 0
    lr_schedule: str = "none"
    min_lr_scale: float = 1.0
    partial_untied_tokens: int = 1024
    recurrent_embedding_dim: int = 144
    recurrent_hidden_dim: int = 288
    recurrent_memory_dim: int = 144
    dropout: float = 0.1
    chunk_size: int = 8


def _load_cached_datasets(
    config: LearnedCompressorCandidateConfig,
) -> tuple[TokenBlockDataset, TokenBlockDataset, int]:
    payload = torch.load(config.cache_path, map_location="cpu", weights_only=False)
    block_size = config.sequence_length + 1
    train_blocks = payload["train_tokens"].long().view(-1, block_size)
    val_blocks = payload["val_tokens"].long().view(-1, block_size)
    train_dataset = TokenBlockDataset(
        train_blocks[: config.train_blocks, :-1].contiguous(),
        train_blocks[: config.train_blocks, 1:].contiguous(),
    )
    val_dataset = TokenBlockDataset(
        val_blocks[: config.val_blocks, :-1].contiguous(),
        val_blocks[: config.val_blocks, 1:].contiguous(),
    )
    return train_dataset, val_dataset, int(payload["vocab_size"])


def _shared_realtext_config(config: LearnedCompressorCandidateConfig) -> RealTextConfig:
    return RealTextConfig(
        seed=config.seed,
        sequence_length=config.sequence_length,
        train_steps=config.train_steps,
        eval_interval=config.eval_interval,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        use_amp=config.use_amp,
        pin_memory=config.pin_memory,
        use_fused_adamw=config.use_fused_adamw,
        tensor_batching=config.tensor_batching,
        cache_dataset_on_device=config.cache_dataset_on_device,
        paired_train_batches=config.paired_train_batches,
        reseed_per_model=config.reseed_per_model,
        train_schedule_seed=config.train_schedule_seed,
        optimizer_recipe=config.optimizer_recipe,
        warmup_steps=config.warmup_steps,
        lr_schedule=config.lr_schedule,
        min_lr_scale=config.min_lr_scale,
    )


class LearnedChunkWriterPartialLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        memory_dim: int,
        dropout: float,
        max_length: int,
        chunk_size: int,
        untied_token_ids: torch.Tensor,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.chunk_value_proj = nn.Linear(hidden_dim, embedding_dim)
        self.partial_head = nn.Linear(embedding_dim, untied_token_ids.numel(), bias=True)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.write_proj = nn.Linear(hidden_dim, hidden_dim)
        self.write_score = nn.Linear(hidden_dim, 1)
        self.retain_gate = nn.Linear(hidden_dim, 1)
        self.read_gate = nn.Linear(hidden_dim, 1)
        self.memory_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("untied_token_ids", untied_token_ids.long(), persistent=False)
        num_chunks = (max_length + chunk_size - 1) // chunk_size
        chunk_ends = torch.arange(num_chunks, dtype=torch.long) * chunk_size + (chunk_size - 1)
        self.register_buffer("_chunk_end_positions", chunk_ends.clamp_max(max_length - 1), persistent=False)

    def _encode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        return embeddings, self.dropout(states)

    def _chunk_view(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        seq_len = states.size(1)
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        total_len = num_chunks * self.chunk_size
        pad_len = total_len - seq_len
        if pad_len > 0:
            states = F.pad(states, (0, 0, 0, pad_len))
        chunked_states = states.reshape(states.size(0), num_chunks, self.chunk_size, states.size(-1))
        valid_mask = (torch.arange(total_len, device=states.device) < seq_len).view(1, num_chunks, self.chunk_size)
        return chunked_states, valid_mask, seq_len, num_chunks

    def _compress_chunks(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        chunked_states, valid_mask, seq_len, num_chunks = self._chunk_view(states)
        write_hidden = torch.tanh(self.write_proj(chunked_states))
        write_logits = self.write_score(write_hidden).squeeze(-1)
        write_logits = write_logits.masked_fill(~valid_mask, torch.finfo(write_logits.dtype).min)
        write_weights = torch.softmax(write_logits, dim=-1)
        write_weights = write_weights * valid_mask
        write_weights = write_weights / write_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        chunk_states = (chunked_states * write_weights.unsqueeze(-1)).sum(dim=2)
        retain = torch.sigmoid(self.retain_gate(chunk_states))
        chunk_features = self.chunk_value_proj(chunk_states) * retain
        return chunk_states, chunk_features, retain, write_weights, seq_len, num_chunks

    def inspect_chunk_writer(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            _, states = self._encode(input_ids)
            chunk_states, chunk_features, retain, write_weights, _, _ = self._compress_chunks(states)
            _, valid_mask, _, _ = self._chunk_view(states)
            return {
                "chunk_states": chunk_states,
                "chunk_features": chunk_features,
                "retain_gates": retain,
                "write_weights": write_weights,
                "valid_mask": valid_mask,
            }

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, states = self._encode(input_ids)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        base_partial = self.partial_head(base_features)

        chunk_states, chunk_features, _, _, seq_len, num_chunks = self._compress_chunks(states)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(chunk_states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        target_positions = torch.arange(seq_len, device=input_ids.device).view(1, -1, 1)
        causal_mask = self._chunk_end_positions[:num_chunks].view(1, 1, -1).to(input_ids.device) < target_positions
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        chunk_partial = self.partial_head(chunk_features)
        memory_partial = torch.matmul(attention.to(chunk_partial.dtype), chunk_partial)
        read_gate = torch.sigmoid(self.read_gate(states))
        total_partial = base_partial + self.memory_scale.to(base_partial.dtype) * read_gate.to(base_partial.dtype) * memory_partial

        full_partial = torch.zeros_like(base_logits)
        index = self.untied_token_ids.view(1, 1, -1).expand(input_ids.size(0), input_ids.size(1), -1)
        full_partial.scatter_add_(2, index, total_partial)
        return base_logits + full_partial


def _build_models(
    config: LearnedCompressorCandidateConfig,
    *,
    vocab_size: int,
    partial_token_ids: torch.Tensor,
) -> dict[str, nn.Module]:
    common = {
        "vocab_size": vocab_size,
        "embedding_dim": config.recurrent_embedding_dim,
        "hidden_dim": config.recurrent_hidden_dim,
        "memory_dim": config.recurrent_memory_dim,
        "dropout": config.dropout,
        "max_length": config.sequence_length,
    }
    return {
        "partial_untied": PartialUntiedAssociativeLM(untied_token_ids=partial_token_ids, **common),
        "chunk_mean_token_memory": ChunkedTokenMemoryPartialLM(
            untied_token_ids=partial_token_ids,
            chunk_size=config.chunk_size,
            **common,
        ),
        "learned_chunk_writer": LearnedChunkWriterPartialLM(
            untied_token_ids=partial_token_ids,
            chunk_size=config.chunk_size,
            **common,
        ),
    }


def run_learned_compressor_benchmark(config: LearnedCompressorCandidateConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(config)
    partial_token_ids = _top_token_ids(
        train_dataset,
        count=config.partial_untied_tokens,
        vocab_size=vocab_size,
    )
    shared_cfg = _shared_realtext_config(config)
    schedule_seed = config.seed if config.train_schedule_seed is None else config.train_schedule_seed
    batch_schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=shared_cfg.batch_size,
        steps=shared_cfg.train_steps,
        seed=schedule_seed,
        drop_last=True,
    )
    tokenizer = None
    if config.compute_val_bpb:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True, local_files_only=True)

    results: dict[str, dict[str, Any]] = {}
    models = _build_models(config, vocab_size=vocab_size, partial_token_ids=partial_token_ids)
    total = len(models)
    for index, (model_name, model) in enumerate(models.items(), start=1):
        if config.reseed_per_model:
            set_global_seed(config.seed)
        report = _train_candidate(
            model,
            train_dataset,
            val_dataset,
            model_name=model_name,
            tokenizer=tokenizer,
            config=shared_cfg,
            compute_val_bpb=config.compute_val_bpb,
            batch_schedule=batch_schedule,
        )
        results[model_name] = report
        print(
            json.dumps(
                {
                    "progress": f"{index}/{total}",
                    "model": model_name,
                    "final_val_loss": report["final_val_loss"],
                    "train_tok_per_sec": report["train_tok_per_sec"],
                    "pure_train_tok_per_sec": report["pure_train_tok_per_sec"],
                    "peak_vram_mb": report["peak_vram_mb"],
                    "parameter_count": report["parameter_count"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        model_name: {
            "final_val_loss": report["final_val_loss"],
            "train_tok_per_sec": report["train_tok_per_sec"],
            "pure_train_tok_per_sec": report["pure_train_tok_per_sec"],
            "peak_vram_mb": report["peak_vram_mb"],
            "parameter_count": report["parameter_count"],
        }
        for model_name, report in results.items()
    }
    incumbent = summary["partial_untied"]
    deltas = {
        model_name: {
            "loss_minus_partial_untied": model["final_val_loss"] - incumbent["final_val_loss"],
            "train_tok_per_sec_minus_partial_untied": model["train_tok_per_sec"] - incumbent["train_tok_per_sec"],
            "pure_train_tok_per_sec_minus_partial_untied": model["pure_train_tok_per_sec"]
            - incumbent["pure_train_tok_per_sec"],
        }
        for model_name, model in summary.items()
        if model_name != "partial_untied"
    }
    return {
        "benchmark": "language_candidate_learned_compressor",
        "config": {
            "cache_path": str(config.cache_path),
            "tokenizer_name": config.tokenizer_name,
            "train_blocks": config.train_blocks,
            "val_blocks": config.val_blocks,
            "sequence_length": config.sequence_length,
            "batch_size": config.batch_size,
            "eval_batch_size": config.eval_batch_size,
            "train_steps": config.train_steps,
            "eval_interval": config.eval_interval,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "device": config.device,
            "use_amp": config.use_amp,
            "pin_memory": config.pin_memory,
            "use_fused_adamw": config.use_fused_adamw,
            "tensor_batching": config.tensor_batching,
            "cache_dataset_on_device": config.cache_dataset_on_device,
            "compute_val_bpb": config.compute_val_bpb,
            "paired_train_batches": config.paired_train_batches,
            "reseed_per_model": config.reseed_per_model,
            "train_schedule_seed": config.train_schedule_seed,
            "optimizer_recipe": config.optimizer_recipe,
            "warmup_steps": config.warmup_steps,
            "lr_schedule": config.lr_schedule,
            "min_lr_scale": config.min_lr_scale,
            "partial_untied_tokens": config.partial_untied_tokens,
            "recurrent_embedding_dim": config.recurrent_embedding_dim,
            "recurrent_hidden_dim": config.recurrent_hidden_dim,
            "recurrent_memory_dim": config.recurrent_memory_dim,
            "dropout": config.dropout,
            "chunk_size": config.chunk_size,
        },
        "results": results,
        "summary": summary,
        "delta_vs_partial_untied": deltas,
        "architecture_summary": {
            "candidate": "learned_chunk_writer",
            "question": "Can the model learn what to keep and discard for long-term memory?",
            "design": [
                "A GRU backbone produces token states and a ReLU^2 readout produces base token logits.",
                "Tokens are grouped into chunks, but long-term memory is not mean-pooled.",
                "A learned per-token write scorer chooses which token states to keep inside each chunk via a masked softmax.",
                "A learned retain gate can suppress whole chunk summaries before they are written into long-term memory.",
                "Queries read chunk summaries causally and add their memory contribution only through a partial untied token head.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cheap learned-compressor probes.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--train-blocks", type=int, default=1024)
    parser.add_argument("--val-blocks", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--partial-untied-tokens", type=int, default=1024)
    parser.add_argument("--recurrent-embedding-dim", type=int, default=144)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=288)
    parser.add_argument("--recurrent-memory-dim", type=int, default=144)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-fused-adamw", action="store_true")
    parser.add_argument("--tensor-batching", action="store_true")
    parser.add_argument("--cache-on-device", action="store_true")
    parser.add_argument("--compute-val-bpb", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_learned_compressor_benchmark(
        LearnedCompressorCandidateConfig(
            cache_path=args.cache_path,
            tokenizer_name=args.tokenizer_name,
            train_blocks=args.train_blocks,
            val_blocks=args.val_blocks,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            train_steps=args.train_steps,
            eval_interval=args.eval_interval,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            use_amp=not args.no_amp,
            pin_memory=not args.no_pin_memory,
            use_fused_adamw=not args.no_fused_adamw,
            tensor_batching=args.tensor_batching,
            cache_dataset_on_device=args.cache_on_device,
            compute_val_bpb=args.compute_val_bpb,
            partial_untied_tokens=args.partial_untied_tokens,
            recurrent_embedding_dim=args.recurrent_embedding_dim,
            recurrent_hidden_dim=args.recurrent_hidden_dim,
            recurrent_memory_dim=args.recurrent_memory_dim,
            dropout=args.dropout,
            chunk_size=args.chunk_size,
        )
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
