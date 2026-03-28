from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from arc_tactic3.language_fastlearn_benchmark import set_global_seed
from arc_tactic3.language_nanochat_actual_compare import (
    _load_cached_datasets,
    _shared_realtext_config,
    _train_candidate,
)
from arc_tactic3.language_realtext_microbench import _build_train_batch_schedule
from arc_tactic3.language_recurrent_nano_tricks import (
    PartialUntiedAssociativeLM,
    _top_token_ids,
)


@dataclass(frozen=True, slots=True)
class LocalGlobalMemoryCompareConfig:
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
    recurrent_embedding_dim: int = 144
    recurrent_hidden_dim: int = 288
    recurrent_memory_dim: int = 144
    dropout: float = 0.1
    paired_train_batches: bool = True
    reseed_per_model: bool = True
    train_schedule_seed: int | None = None
    optimizer_recipe: str = "default"
    warmup_steps: int = 0
    lr_schedule: str = "none"
    min_lr_scale: float = 1.0
    partial_untied_tokens: int = 2048
    local_window: int = 32
    older_chunk_size: int = 8


class LocalGlobalTokenPartialMemoryLM(nn.Module):
    """Two-timescale recurrent memory.

    - exact token memory over a recent local window
    - compressed chunk summaries for older context
    - learned softmax gate that mixes local exact and global compressed memory
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        memory_dim: int,
        dropout: float,
        max_length: int,
        untied_token_ids: torch.Tensor,
        local_window: int,
        older_chunk_size: int,
    ) -> None:
        super().__init__()
        if local_window < 1:
            raise ValueError("local_window must be >= 1.")
        if older_chunk_size < 1:
            raise ValueError("older_chunk_size must be >= 1.")
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.partial_head = nn.Linear(embedding_dim, untied_token_ids.numel(), bias=True)

        self.local_query_proj = nn.Linear(hidden_dim, memory_dim)
        self.local_key_proj = nn.Linear(hidden_dim, memory_dim)
        self.global_query_proj = nn.Linear(hidden_dim, memory_dim)
        self.global_key_proj = nn.Linear(hidden_dim, memory_dim)
        self.global_value_proj = nn.Linear(hidden_dim, embedding_dim)
        self.global_partial_head = nn.Linear(embedding_dim, untied_token_ids.numel(), bias=False)
        self.memory_mix = nn.Linear(hidden_dim, 2)
        self.local_memory_scale = nn.Parameter(torch.tensor(6.0))
        self.global_memory_scale = nn.Parameter(torch.tensor(4.0))

        self.local_window = local_window
        self.older_chunk_size = older_chunk_size
        self.register_buffer("untied_token_ids", untied_token_ids.long(), persistent=False)

        positions = torch.arange(max_length)
        query_positions = positions.view(max_length, 1)
        key_positions = positions.view(1, max_length)
        local_mask = (key_positions < query_positions) & (key_positions >= (query_positions - local_window))
        self.register_buffer("_local_mask", local_mask.unsqueeze(0), persistent=False)

    def _base_logits(self, states: torch.Tensor) -> torch.Tensor:
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        partial_logits = self.partial_head(base_features)
        partial_full = torch.zeros_like(base_logits)
        partial_index = self.untied_token_ids.view(1, 1, -1).expand(states.size(0), states.size(1), -1)
        partial_full.scatter_add_(2, partial_index, partial_logits)
        return base_logits + partial_full

    def _local_exact_logits(
        self,
        input_ids: torch.Tensor,
        states: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sequence_length = input_ids.size(1)
        queries = self.local_query_proj(states)
        keys = self.local_key_proj(states)
        scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(queries.size(-1))
        local_mask = self._local_mask[:, :sequence_length, :sequence_length]
        scores = scores.masked_fill(~local_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * local_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        local_logits = torch.zeros(
            input_ids.size(0),
            sequence_length,
            self.vocab_size,
            device=input_ids.device,
            dtype=dtype,
        )
        value_index = input_ids.unsqueeze(1).expand(-1, sequence_length, -1)
        local_logits.scatter_add_(2, value_index, attention.to(dtype))
        return local_logits

    def _global_chunk_logits(
        self,
        states: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = states.shape
        chunk_size = self.older_chunk_size
        chunk_count = (sequence_length + chunk_size - 1) // chunk_size
        pad = chunk_count * chunk_size - sequence_length
        if pad > 0:
            padded_states = torch.cat((states, states.new_zeros(batch_size, pad, hidden_dim)), dim=1)
            padded_mask = torch.cat(
                (
                    torch.ones(sequence_length, device=states.device, dtype=states.dtype),
                    torch.zeros(pad, device=states.device, dtype=states.dtype),
                ),
                dim=0,
            )
        else:
            padded_states = states
            padded_mask = torch.ones(sequence_length, device=states.device, dtype=states.dtype)

        chunk_states = padded_states.view(batch_size, chunk_count, chunk_size, hidden_dim)
        chunk_mask = padded_mask.view(1, chunk_count, chunk_size, 1)
        chunk_counts = chunk_mask.sum(dim=2).clamp_min(1.0)
        chunk_summary = (chunk_states * chunk_mask).sum(dim=2) / chunk_counts

        global_queries = self.global_query_proj(states)
        global_keys = self.global_key_proj(chunk_summary)
        global_values = self.global_value_proj(chunk_summary)
        scores = torch.matmul(global_queries, global_keys.transpose(1, 2)) / math.sqrt(global_queries.size(-1))

        chunk_end_positions = (
            (torch.arange(chunk_count, device=states.device) + 1) * chunk_size - 1
        ).clamp(max=sequence_length - 1)
        query_positions = torch.arange(sequence_length, device=states.device)
        older_cutoff = query_positions - self.local_window
        global_mask = chunk_end_positions.view(1, 1, -1) < older_cutoff.view(1, sequence_length, 1)

        scores = scores.masked_fill(~global_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * global_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        global_context = torch.matmul(attention.to(global_values.dtype), global_values)
        partial_logits = self.global_partial_head(global_context)

        global_logits = torch.zeros(
            batch_size,
            sequence_length,
            self.vocab_size,
            device=states.device,
            dtype=dtype,
        )
        partial_index = self.untied_token_ids.view(1, 1, -1).expand(batch_size, sequence_length, -1)
        global_logits.scatter_add_(2, partial_index, partial_logits.to(dtype))
        return global_logits

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)

        base_logits = self._base_logits(states)
        local_logits = self._local_exact_logits(input_ids, states, dtype=base_logits.dtype)
        global_logits = self._global_chunk_logits(states, dtype=base_logits.dtype)

        mix = torch.softmax(self.memory_mix(states), dim=-1).to(base_logits.dtype)
        local_weight = mix[..., :1] * self.local_memory_scale.to(base_logits.dtype)
        global_weight = mix[..., 1:] * self.global_memory_scale.to(base_logits.dtype)
        return base_logits + local_weight * local_logits + global_weight * global_logits


def _build_models(
    config: LocalGlobalMemoryCompareConfig,
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
        "partial_untied": PartialUntiedAssociativeLM(
            untied_token_ids=partial_token_ids,
            **common,
        ),
        "local_global_partial_memory": LocalGlobalTokenPartialMemoryLM(
            untied_token_ids=partial_token_ids,
            local_window=config.local_window,
            older_chunk_size=config.older_chunk_size,
            **common,
        ),
    }


def run_local_global_memory_compare(config: LocalGlobalMemoryCompareConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(config)
    partial_token_ids = _top_token_ids(
        train_dataset,
        count=config.partial_untied_tokens,
        vocab_size=vocab_size,
    )

    tokenizer = None
    if config.compute_val_bpb:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True, local_files_only=True)

    shared_config = _shared_realtext_config(config)
    schedule_seed = config.seed if config.train_schedule_seed is None else config.train_schedule_seed
    batch_schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=shared_config.batch_size,
        steps=shared_config.train_steps,
        seed=schedule_seed,
        drop_last=True,
    )

    reports: dict[str, dict[str, Any]] = {}
    for model_name, model in _build_models(config, vocab_size=vocab_size, partial_token_ids=partial_token_ids).items():
        if config.reseed_per_model:
            set_global_seed(config.seed)
        reports[model_name] = _train_candidate(
            model,
            train_dataset,
            val_dataset,
            model_name=model_name,
            tokenizer=tokenizer,
            config=shared_config,
            compute_val_bpb=config.compute_val_bpb,
            batch_schedule=batch_schedule,
        )

    return {
        "benchmark": "language_candidate_local_global_memory",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
        },
        "fairness": {
            "same_dataset": True,
            "same_tokenizer": True,
            "paired_batch_schedule": True,
            "reseed_per_model": bool(config.reseed_per_model),
            "candidate_design": "exact_local_window_plus_compressed_older_partial_memory",
        },
        "results": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a two-timescale local/global recurrent memory candidate against partial_untied."
    )
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--train-blocks", type=int, default=1024)
    parser.add_argument("--val-blocks", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dataset-on-device", action="store_true")
    parser.add_argument("--local-window", type=int, default=32)
    parser.add_argument("--older-chunk-size", type=int, default=8)
    parser.add_argument("--partial-untied-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = LocalGlobalMemoryCompareConfig(
        cache_path=args.cache_path,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        sequence_length=args.sequence_length,
        seed=args.seed,
        device=args.device,
        cache_dataset_on_device=args.cache_dataset_on_device,
        local_window=args.local_window,
        older_chunk_size=args.older_chunk_size,
        partial_untied_tokens=args.partial_untied_tokens,
    )
    payload = run_local_global_memory_compare(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
