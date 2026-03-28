from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from arc_tactic3.language_fastlearn_benchmark import count_parameters, set_global_seed
from arc_tactic3.language_realtext_microbench import (
    RealTextConfig,
    TokenBlockDataset,
    _build_optimizer,
    _dataset_tensors,
    _iter_tensor_batches,
    _loss_and_tokens,
    _move_batch,
    AssociativeRecurrentLM,
)


@dataclass(frozen=True, slots=True)
class ThroughputCandidateConfig:
    cache_path: Path
    tokenizer_name: str = "gpt2"
    train_blocks: int = 1024
    val_blocks: int = 128
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
    embedding_dim: int = 96
    hidden_dim: int = 192
    memory_dim: int = 96
    dropout: float = 0.1
    window_size: int = 32


class GRUOnlyLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_to_embedding = nn.Linear(hidden_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        return F.linear(self.hidden_to_embedding(states), self.embedding.weight, self.output_bias)


class SharedProjectionAssociativeLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        memory_dim: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.memory_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.hidden_to_embedding = nn.Linear(hidden_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.memory_scale = nn.Parameter(torch.tensor(6.0))
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=-1).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        base_logits = F.linear(self.hidden_to_embedding(states), self.embedding.weight, self.output_bias)
        memory_states = self.memory_proj(states)
        scores = torch.matmul(memory_states, memory_states.transpose(1, 2)) / math.sqrt(memory_states.size(-1))
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        gate = torch.sigmoid(self.gate(states))
        gated_attention = (attention * (gate * self.memory_scale)).to(base_logits.dtype)
        base_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits


class WindowedAssociativeLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        memory_dim: int,
        dropout: float,
        max_length: int,
        window_size: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.hidden_to_embedding = nn.Linear(hidden_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.memory_scale = nn.Parameter(torch.tensor(6.0))
        full_mask = torch.tril(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=-1)
        if window_size > 0:
            row_index = torch.arange(max_length).unsqueeze(1)
            col_index = torch.arange(max_length).unsqueeze(0)
            full_mask &= (row_index - col_index) <= window_size
        self.register_buffer("_causal_mask", full_mask.unsqueeze(0), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        base_logits = F.linear(self.hidden_to_embedding(states), self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / math.sqrt(query_keys.size(-1))
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        gate = torch.sigmoid(self.gate(states))
        gated_attention = (attention * (gate * self.memory_scale)).to(base_logits.dtype)
        base_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits


class DecayedVoteAssociativeLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_to_embedding = nn.Linear(hidden_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.read_gate = nn.Linear(hidden_dim, 1)
        self.write_gate = nn.Linear(hidden_dim, 1)
        self.decay_gate = nn.Linear(hidden_dim, 1)
        self.memory_scale = nn.Parameter(torch.tensor(4.0))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        base_logits = F.linear(self.hidden_to_embedding(states), self.embedding.weight, self.output_bias)
        batch_size, sequence_length, vocab_size = base_logits.shape
        memory_logits = torch.zeros((batch_size, vocab_size), dtype=base_logits.dtype, device=base_logits.device)
        outputs: list[torch.Tensor] = []
        for index in range(sequence_length):
            state = states[:, index, :]
            read_gate = torch.sigmoid(self.read_gate(state)) * self.memory_scale
            outputs.append(base_logits[:, index, :] + read_gate * memory_logits)
            decay = torch.sigmoid(self.decay_gate(state))
            write_gate = torch.sigmoid(self.write_gate(state))
            memory_logits = memory_logits * decay
            memory_logits.scatter_add_(1, input_ids[:, index : index + 1], write_gate.to(memory_logits.dtype))
        return torch.stack(outputs, dim=1)


def _load_cached_datasets(config: ThroughputCandidateConfig) -> tuple[TokenBlockDataset, TokenBlockDataset, int]:
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


def _shared_realtext_config(config: ThroughputCandidateConfig) -> RealTextConfig:
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
    )


def _build_candidates(config: ThroughputCandidateConfig, *, vocab_size: int) -> dict[str, nn.Module]:
    return {
        "baseline": AssociativeRecurrentLM(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            memory_dim=config.memory_dim,
            dropout=config.dropout,
            max_length=config.sequence_length,
        ),
        "gru_only": GRUOnlyLM(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        ),
        "shared_projection": SharedProjectionAssociativeLM(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            memory_dim=config.memory_dim,
            dropout=config.dropout,
            max_length=config.sequence_length,
        ),
        "windowed_32": WindowedAssociativeLM(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            memory_dim=config.memory_dim,
            dropout=config.dropout,
            max_length=config.sequence_length,
            window_size=config.window_size,
        ),
        "decayed_vote": DecayedVoteAssociativeLM(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        ),
    }


def _estimate_target_bytes(dataset: TokenBlockDataset, *, tokenizer) -> int:
    total_bytes = 0
    for row in dataset.targets:
        text = tokenizer.decode(row.tolist(), clean_up_tokenization_spaces=False)
        total_bytes += len(text.encode("utf-8"))
    return max(total_bytes, 1)


def _peak_vram_mb(device: str) -> float | None:
    if device != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _train_candidate(
    model: nn.Module,
    train_dataset: TokenBlockDataset,
    val_dataset: TokenBlockDataset,
    *,
    tokenizer,
    config: RealTextConfig,
    compute_val_bpb: bool,
) -> dict[str, Any]:
    device = torch.device(config.device)
    model.to(device)
    optimizer = _build_optimizer(model, config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=config.use_amp and device.type == "cuda")
    use_amp = config.use_amp and device.type == "cuda"
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]

    if config.tensor_batching:
        train_source = _dataset_tensors(
            train_dataset,
            device=device,
            cache_on_device=config.cache_dataset_on_device,
            pin_memory=config.pin_memory,
        )
        val_source = _dataset_tensors(
            val_dataset,
            device=device,
            cache_on_device=config.cache_dataset_on_device,
            pin_memory=config.pin_memory,
        )
        train_iterator = _iter_tensor_batches(
            train_source[0],
            train_source[1],
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            device=device,
            non_blocking=config.pin_memory and device.type == "cuda",
        )
    else:
        pin_memory = config.pin_memory and device.type == "cuda"
        train_source = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=pin_memory,
            num_workers=0,
            persistent_workers=False,
        )
        val_source = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=0,
            persistent_workers=False,
        )
        train_iterator = cycle(train_source)

    initial_val_loss = _evaluate_candidate(model, val_source, device=device, use_amp=use_amp, config=config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history: list[dict[str, float]] = [
        {
            "step": 0.0,
            "sequences_seen": 0.0,
            "tokens_seen": 0.0,
            "train_loss": float("nan"),
            "val_loss": float(initial_val_loss),
        }
    ]
    step_times: list[float] = []
    tokens_seen = 0
    sequences_seen = 0
    start = time.perf_counter()
    for step in range(1, config.train_steps + 1):
        if config.tensor_batching:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = _iter_tensor_batches(
                    train_source[0],
                    train_source[1],
                    batch_size=config.batch_size,
                    shuffle=True,
                    drop_last=True,
                    device=device,
                    non_blocking=config.pin_memory and device.type == "cuda",
                )
                batch = next(train_iterator)
        else:
            batch = _move_batch(next(train_iterator), device)
        step_start = time.perf_counter()
        model.train()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch["input_ids"])
            loss, token_count = _loss_and_tokens(logits, batch["targets"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameter_list, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_start)
        tokens_seen += token_count
        sequences_seen += batch["input_ids"].size(0)
        if step % config.eval_interval == 0 or step == config.train_steps:
            val_loss = _evaluate_candidate(model, val_source, device=device, use_amp=use_amp, config=config)
            history.append(
                {
                    "step": float(step),
                    "sequences_seen": float(sequences_seen),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": float(loss.item()),
                    "val_loss": float(val_loss),
                }
            )
    total_time = time.perf_counter() - start
    final_val_loss = float(history[-1]["val_loss"])
    target_bytes = _estimate_target_bytes(val_dataset, tokenizer=tokenizer) if compute_val_bpb and tokenizer is not None else None
    total_val_tokens = int(val_dataset.targets.numel())
    total_val_bits = final_val_loss * total_val_tokens / math.log(2.0)
    return {
        "parameter_count": count_parameters(model),
        "initial_val_loss": float(initial_val_loss),
        "final_val_loss": final_val_loss,
        "val_bits_per_token": final_val_loss / math.log(2.0),
        "val_bpb": (total_val_bits / target_bytes) if target_bytes is not None else None,
        "train_tokens_seen": tokens_seen,
        "train_tok_per_sec": tokens_seen / max(total_time, 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "peak_vram_mb": _peak_vram_mb(config.device),
        "history": history,
        "total_training_time_seconds": total_time,
    }


def _evaluate_candidate(
    model: nn.Module,
    eval_source,
    *,
    device: torch.device,
    use_amp: bool,
    config: RealTextConfig,
) -> float:
    model.eval()
    loss_sum = 0.0
    token_total = 0
    batch_iterator = eval_source
    if config.tensor_batching:
        input_ids, targets = eval_source
        batch_iterator = _iter_tensor_batches(
            input_ids,
            targets,
            batch_size=config.eval_batch_size,
            shuffle=False,
            drop_last=False,
            device=device,
            non_blocking=config.pin_memory and device.type == "cuda",
        )
    with torch.inference_mode():
        for batch in batch_iterator:
            if not config.tensor_batching:
                batch = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch["input_ids"])
                loss, tokens = _loss_and_tokens(logits, batch["targets"])
            loss_sum += float(loss.item()) * tokens
            token_total += tokens
    return loss_sum / max(token_total, 1)


def run_throughput_candidates(config: ThroughputCandidateConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(config)
    tokenizer = None
    if config.compute_val_bpb:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True, local_files_only=True)
    shared_config = _shared_realtext_config(config)
    reports: dict[str, dict[str, Any]] = {}
    for candidate_name, model in _build_candidates(config, vocab_size=vocab_size).items():
        reports[candidate_name] = _train_candidate(
            model,
            train_dataset,
            val_dataset,
            tokenizer=tokenizer,
            config=shared_config,
            compute_val_bpb=config.compute_val_bpb,
        )
    return {
        "benchmark": "language_throughput_candidates",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
        },
        "results": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run throughput-oriented recurrent architecture candidates.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--train-blocks", type=int, default=1024)
    parser.add_argument("--val-blocks", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = ThroughputCandidateConfig(
        cache_path=args.cache_path,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        device=args.device,
    )
    payload = run_throughput_candidates(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
