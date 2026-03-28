from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from arc_tactic3.language_fastlearn_benchmark import count_parameters, set_global_seed
from arc_tactic3.language_nanochat_actual_compare import _load_cached_datasets, _shared_realtext_config, _train_candidate
from arc_tactic3.language_realtext_microbench import _build_train_batch_schedule
from arc_tactic3.language_recurrent_nano_tricks import (
    PartialUntiedAssociativeLM,
    ReLU2HeadAssociativeLM,
    RecurrentNanoTricksConfig,
    _top_token_ids,
)


@dataclass(frozen=True, slots=True)
class RecurrentMemoryRewriteConfig:
    cache_path: Path
    train_blocks: int = 1024
    val_blocks: int = 64
    sequence_length: int = 127
    train_steps: int = 16
    eval_interval: int = 8
    seed: int = 13
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_val_bpb: bool = False
    partial_untied_tokens: int = 1024
    recurrent_embedding_dim: int = 144
    recurrent_hidden_dim: int = 288
    recurrent_memory_dim: int = 144
    stride: int = 4
    chunk_size: int = 8
    use_amp: bool = torch.cuda.is_available()
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    tensor_batching: bool = False
    cache_dataset_on_device: bool = False


class FeatureRetrievalAssociativeLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        memory_dim: int,
        dropout: float,
        max_length: int,
        source_mode: str = "full",
        stride: int = 4,
        chunk_size: int = 8,
    ) -> None:
        super().__init__()
        if source_mode not in {"full", "stride", "chunk"}:
            raise ValueError(f"Unsupported source_mode: {source_mode}")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.source_mode = source_mode
        self.stride = stride
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.mix_gate = nn.Linear(hidden_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        if source_mode in {"full", "stride"}:
            positions = torch.arange(max_length, dtype=torch.long)
            source_positions = positions if source_mode == "full" else positions[::stride]
            target_positions = positions.view(1, -1, 1)
            causal_mask = source_positions.view(1, 1, -1) < target_positions
            self.register_buffer("_source_positions", source_positions, persistent=False)
            self.register_buffer("_causal_mask", causal_mask, persistent=False)
        else:
            num_chunks = (max_length + chunk_size - 1) // chunk_size
            chunk_ends = torch.arange(num_chunks, dtype=torch.long) * chunk_size + (chunk_size - 1)
            self.register_buffer("_chunk_end_positions", chunk_ends.clamp_max(max_length - 1), persistent=False)

    def _build_sources(self, states: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = states.size(1)
        device = states.device
        if self.source_mode == "full":
            source_states = states
            source_values = values
            source_positions = self._source_positions[:seq_len]
            causal_mask = self._causal_mask[:, :seq_len, :seq_len]
            return source_states, source_values, source_positions.to(device), causal_mask
        if self.source_mode == "stride":
            source_states = states[:, :: self.stride]
            source_values = values[:, :: self.stride]
            source_positions = self._source_positions[: source_states.size(1)]
            causal_mask = self._causal_mask[:, :seq_len, : source_states.size(1)]
            return source_states, source_values, source_positions.to(device), causal_mask

        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad_len = num_chunks * self.chunk_size - seq_len
        if pad_len > 0:
            states = F.pad(states, (0, 0, 0, pad_len))
            values = F.pad(values, (0, 0, 0, pad_len))
        chunked_states = states.reshape(states.size(0), num_chunks, self.chunk_size, states.size(-1)).mean(dim=2)
        chunked_values = values.reshape(values.size(0), num_chunks, self.chunk_size, values.size(-1)).mean(dim=2)
        source_positions = self._chunk_end_positions[:num_chunks]
        target_positions = torch.arange(seq_len, device=device).view(1, -1, 1)
        causal_mask = source_positions.view(1, 1, -1).to(device) < target_positions
        return chunked_states, chunked_values, source_positions.to(device), causal_mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)

        source_states, source_values, _, causal_mask = self._build_sources(states, embeddings)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(source_states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        retrieved_values = torch.matmul(attention.to(source_values.dtype), self.value_proj(source_values))
        mixed_features = base_features + torch.sigmoid(self.mix_gate(states)) * retrieved_values
        return F.linear(mixed_features, self.embedding.weight, self.output_bias)


class ChunkedFeaturePartialUntiedLM(nn.Module):
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
        self.value_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.mix_gate = nn.Linear(hidden_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.partial_head = nn.Linear(embedding_dim, untied_token_ids.numel(), bias=True)
        self.register_buffer("untied_token_ids", untied_token_ids.long(), persistent=False)
        num_chunks = (max_length + chunk_size - 1) // chunk_size
        chunk_ends = torch.arange(num_chunks, dtype=torch.long) * chunk_size + (chunk_size - 1)
        self.register_buffer("_chunk_end_positions", chunk_ends.clamp_max(max_length - 1), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)

        seq_len = states.size(1)
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad_len = num_chunks * self.chunk_size - seq_len
        padded_states = states
        padded_embeddings = embeddings
        if pad_len > 0:
            padded_states = F.pad(padded_states, (0, 0, 0, pad_len))
            padded_embeddings = F.pad(padded_embeddings, (0, 0, 0, pad_len))
        chunked_states = padded_states.reshape(states.size(0), num_chunks, self.chunk_size, states.size(-1)).mean(dim=2)
        chunked_values = padded_embeddings.reshape(
            embeddings.size(0), num_chunks, self.chunk_size, embeddings.size(-1)
        ).mean(dim=2)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(chunked_states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        target_positions = torch.arange(seq_len, device=input_ids.device).view(1, -1, 1)
        causal_mask = self._chunk_end_positions[:num_chunks].view(1, 1, -1).to(input_ids.device) < target_positions
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        retrieved_values = torch.matmul(attention.to(chunked_values.dtype), self.value_proj(chunked_values))
        mixed_features = base_features + torch.sigmoid(self.mix_gate(states)) * retrieved_values
        base_logits = F.linear(mixed_features, self.embedding.weight, self.output_bias)
        partial_logits = self.partial_head(mixed_features)
        full_partial = torch.zeros_like(base_logits)
        index = self.untied_token_ids.view(1, 1, -1).expand(input_ids.size(0), input_ids.size(1), -1)
        full_partial.scatter_add_(2, index, partial_logits)
        return base_logits + full_partial


class ChunkedTokenMemoryPartialLM(nn.Module):
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
        self.register_buffer("untied_token_ids", untied_token_ids.long(), persistent=False)
        num_chunks = (max_length + chunk_size - 1) // chunk_size
        chunk_ends = torch.arange(num_chunks, dtype=torch.long) * chunk_size + (chunk_size - 1)
        self.register_buffer("_chunk_end_positions", chunk_ends.clamp_max(max_length - 1), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)

        seq_len = states.size(1)
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad_len = num_chunks * self.chunk_size - seq_len
        padded_states = states
        if pad_len > 0:
            padded_states = F.pad(padded_states, (0, 0, 0, pad_len))
        chunked_states = padded_states.reshape(states.size(0), num_chunks, self.chunk_size, states.size(-1)).mean(dim=2)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(chunked_states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        target_positions = torch.arange(seq_len, device=input_ids.device).view(1, -1, 1)
        causal_mask = self._chunk_end_positions[:num_chunks].view(1, 1, -1).to(input_ids.device) < target_positions
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        base_partial = self.partial_head(base_features)
        chunk_partial = self.partial_head(self.chunk_value_proj(chunked_states))
        memory_partial = torch.matmul(attention.to(chunk_partial.dtype), chunk_partial)
        total_partial = base_partial + memory_partial

        full_partial = torch.zeros_like(base_logits)
        index = self.untied_token_ids.view(1, 1, -1).expand(input_ids.size(0), input_ids.size(1), -1)
        full_partial.scatter_add_(2, index, total_partial)
        return base_logits + full_partial


def _build_models(
    config: RecurrentNanoTricksConfig,
    *,
    vocab_size: int,
    partial_token_ids: torch.Tensor,
    stride: int,
    chunk_size: int,
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
        "recurrent_relu2_legacy": ReLU2HeadAssociativeLM(**common),
        "partial_untied": PartialUntiedAssociativeLM(untied_token_ids=partial_token_ids, **common),
        "feature_retrieval_full": FeatureRetrievalAssociativeLM(**common, source_mode="full"),
        "feature_retrieval_stride4": FeatureRetrievalAssociativeLM(**common, source_mode="stride", stride=stride),
        "feature_retrieval_chunk8": FeatureRetrievalAssociativeLM(**common, source_mode="chunk", chunk_size=chunk_size),
        "chunk8_partial_hybrid": ChunkedFeaturePartialUntiedLM(
            **common,
            chunk_size=chunk_size,
            untied_token_ids=partial_token_ids,
        ),
        "chunk8_token_partial_memory": ChunkedTokenMemoryPartialLM(
            **common,
            chunk_size=chunk_size,
            untied_token_ids=partial_token_ids,
        ),
        "recurrent_champion": ChunkedTokenMemoryPartialLM(
            **common,
            chunk_size=chunk_size,
            untied_token_ids=partial_token_ids,
        ),
    }


def run_memory_rewrite_benchmark(config: RecurrentMemoryRewriteConfig) -> dict[str, Any]:
    base_cfg = RecurrentNanoTricksConfig(
        cache_path=config.cache_path,
        train_blocks=config.train_blocks,
        val_blocks=config.val_blocks,
        sequence_length=config.sequence_length,
        train_steps=config.train_steps,
        eval_interval=config.eval_interval,
        seed=config.seed,
        device=config.device,
        compute_val_bpb=config.compute_val_bpb,
        partial_untied_tokens=config.partial_untied_tokens,
        recurrent_embedding_dim=config.recurrent_embedding_dim,
        recurrent_hidden_dim=config.recurrent_hidden_dim,
        recurrent_memory_dim=config.recurrent_memory_dim,
        window_size=config.stride,
        use_amp=config.use_amp,
        pin_memory=config.pin_memory,
        use_fused_adamw=config.use_fused_adamw,
        tensor_batching=config.tensor_batching,
        cache_dataset_on_device=config.cache_dataset_on_device,
    )
    set_global_seed(config.seed)
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(base_cfg)
    partial_token_ids = _top_token_ids(train_dataset, count=base_cfg.partial_untied_tokens, vocab_size=vocab_size)
    shared_cfg = _shared_realtext_config(base_cfg)
    batch_schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=shared_cfg.batch_size,
        steps=shared_cfg.train_steps,
        seed=shared_cfg.seed if shared_cfg.train_schedule_seed is None else shared_cfg.train_schedule_seed,
        drop_last=True,
    )
    models = _build_models(
        base_cfg,
        vocab_size=vocab_size,
        partial_token_ids=partial_token_ids,
        stride=config.stride,
        chunk_size=config.chunk_size,
    )
    results: dict[str, dict[str, Any]] = {}
    total = len(models)
    for index, (model_name, model) in enumerate(models.items(), start=1):
        set_global_seed(config.seed)
        report = _train_candidate(
            model,
            train_dataset,
            val_dataset,
            model_name=model_name,
            tokenizer=None,
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
            "pure_train_tok_per_sec_minus_partial_untied": model["pure_train_tok_per_sec"]
            - incumbent["pure_train_tok_per_sec"],
            "train_tok_per_sec_minus_partial_untied": model["train_tok_per_sec"] - incumbent["train_tok_per_sec"],
        }
        for model_name, model in summary.items()
        if model_name != "partial_untied"
    }
    return {
        "benchmark": "language_recurrent_memory_rewrites",
        "config": {
            "cache_path": str(config.cache_path),
            "train_blocks": config.train_blocks,
            "val_blocks": config.val_blocks,
            "sequence_length": config.sequence_length,
            "train_steps": config.train_steps,
            "eval_interval": config.eval_interval,
            "seed": config.seed,
            "device": config.device,
            "compute_val_bpb": config.compute_val_bpb,
            "partial_untied_tokens": config.partial_untied_tokens,
            "recurrent_embedding_dim": config.recurrent_embedding_dim,
            "recurrent_hidden_dim": config.recurrent_hidden_dim,
            "recurrent_memory_dim": config.recurrent_memory_dim,
            "stride": config.stride,
            "chunk_size": config.chunk_size,
            "use_amp": config.use_amp,
            "pin_memory": config.pin_memory,
            "use_fused_adamw": config.use_fused_adamw,
            "tensor_batching": config.tensor_batching,
            "cache_dataset_on_device": config.cache_dataset_on_device,
        },
        "results": results,
        "summary": summary,
        "delta_vs_partial_untied": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cheap recurrent memory-rewrite probes.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--train-blocks", type=int, default=1024)
    parser.add_argument("--val-blocks", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument("--train-steps", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--recurrent-embedding-dim", type=int, default=144)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=288)
    parser.add_argument("--recurrent-memory-dim", type=int, default=144)
    parser.add_argument("--partial-untied-tokens", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-fused-adamw", action="store_true")
    parser.add_argument("--tensor-batching", action="store_true")
    parser.add_argument("--cache-on-device", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_memory_rewrite_benchmark(
        RecurrentMemoryRewriteConfig(
            cache_path=args.cache_path,
            train_blocks=args.train_blocks,
            val_blocks=args.val_blocks,
            sequence_length=args.sequence_length,
            train_steps=args.train_steps,
            eval_interval=args.eval_interval,
            seed=args.seed,
            device=args.device,
            recurrent_embedding_dim=args.recurrent_embedding_dim,
            recurrent_hidden_dim=args.recurrent_hidden_dim,
            recurrent_memory_dim=args.recurrent_memory_dim,
            partial_untied_tokens=args.partial_untied_tokens,
            stride=args.stride,
            chunk_size=args.chunk_size,
            use_amp=not args.no_amp,
            pin_memory=not args.no_pin_memory,
            use_fused_adamw=not args.no_fused_adamw,
            tensor_batching=args.tensor_batching,
            cache_dataset_on_device=args.cache_on_device,
        )
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
