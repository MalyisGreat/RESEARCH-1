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

from arc_tactic3.language_fastlearn_benchmark import count_parameters, set_global_seed
from arc_tactic3.language_nanochat_actual_compare import _load_cached_datasets, _train_candidate
from arc_tactic3.language_realtext_microbench import RealTextConfig, TokenBlockDataset, _build_train_batch_schedule
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM


@dataclass(frozen=True, slots=True)
class DynamicTokenBasisConfig:
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
    partial_untied_tokens: int = 512
    token_basis_rank: int = 48
    routing_experts: int = 4
    routing_top_k: int = 2


def _top_token_ids(dataset: TokenBlockDataset, *, count: int, vocab_size: int) -> torch.Tensor:
    flat_tokens = dataset.targets.reshape(-1)
    histogram = torch.bincount(flat_tokens, minlength=vocab_size)
    top_k = min(max(count, 1), vocab_size)
    return torch.topk(histogram, k=top_k, largest=True, sorted=False).indices


def _shared_realtext_config(config: DynamicTokenBasisConfig) -> RealTextConfig:
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


def _topk_router(logits: torch.Tensor, *, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if top_k >= logits.size(-1):
        return torch.softmax(logits, dim=-1)
    top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
    sparse_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
    sparse_logits.scatter_(-1, top_indices, top_values)
    return torch.softmax(sparse_logits, dim=-1)


class DynamicTokenBasisAssociativeLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        memory_dim: int,
        dropout: float,
        max_length: int,
        basis_rank: int,
        routing_experts: int,
        routing_top_k: int,
    ) -> None:
        super().__init__()
        if basis_rank <= 0:
            raise ValueError("basis_rank must be positive.")
        if routing_experts <= 0:
            raise ValueError("routing_experts must be positive.")
        if routing_top_k <= 0:
            raise ValueError("routing_top_k must be positive.")
        self.routing_top_k = min(routing_top_k, routing_experts)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.exact_gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.route_proj = nn.Linear(hidden_dim, routing_experts)
        self.base_coeff_weight = nn.Parameter(torch.empty(routing_experts, embedding_dim, basis_rank))
        self.base_coeff_bias = nn.Parameter(torch.zeros(routing_experts, basis_rank))
        self.memory_coeff_weight = nn.Parameter(torch.empty(routing_experts, hidden_dim, basis_rank))
        self.memory_coeff_bias = nn.Parameter(torch.zeros(routing_experts, basis_rank))
        self.basis_gate = nn.Linear(hidden_dim, basis_rank)
        self.basis_up = nn.Linear(basis_rank, embedding_dim, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.exact_memory_scale = nn.Parameter(torch.tensor(6.0))
        self.basis_memory_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=-1).unsqueeze(0),
            persistent=False,
        )
        nn.init.xavier_uniform_(self.base_coeff_weight)
        nn.init.xavier_uniform_(self.memory_coeff_weight)

    def _routed_coefficients(
        self,
        inputs: torch.Tensor,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor,
        route: torch.Tensor,
    ) -> torch.Tensor:
        projected = torch.einsum("btd,gdr->btgr", inputs, weight)
        projected = projected + bias.view(1, 1, bias.size(0), bias.size(1))
        return torch.sum(projected * route.unsqueeze(-1), dim=2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)

        route = _topk_router(self.route_proj(states), top_k=self.routing_top_k)
        base_basis = self._routed_coefficients(
            base_features,
            weight=self.base_coeff_weight,
            bias=self.base_coeff_bias,
            route=route,
        )

        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / math.sqrt(query_keys.size(-1))
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        memory_states = torch.matmul(attention, states)
        memory_basis = self._routed_coefficients(
            memory_states,
            weight=self.memory_coeff_weight,
            bias=self.memory_coeff_bias,
            route=route,
        )
        basis_gate = torch.sigmoid(self.basis_gate(states))
        dynamic_delta = self.basis_up(base_basis + self.basis_memory_scale * (basis_gate * memory_basis))
        base_logits = base_logits + F.linear(dynamic_delta, self.embedding.weight)

        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        exact_gate = torch.sigmoid(self.exact_gate(states))
        gated_attention = (attention * (exact_gate * self.exact_memory_scale)).to(base_logits.dtype)
        base_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits


def _build_models(
    config: DynamicTokenBasisConfig,
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
            **common,
            untied_token_ids=partial_token_ids,
        ),
        "dynamic_token_basis": DynamicTokenBasisAssociativeLM(
            **common,
            basis_rank=config.token_basis_rank,
            routing_experts=config.routing_experts,
            routing_top_k=config.routing_top_k,
        ),
    }


def run_dynamic_token_basis_compare(config: DynamicTokenBasisConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(config)
    partial_token_ids = _top_token_ids(
        train_dataset,
        count=config.partial_untied_tokens,
        vocab_size=vocab_size,
    )
    shared_config = _shared_realtext_config(config)
    batch_schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=shared_config.batch_size,
        steps=shared_config.train_steps,
        seed=config.seed if config.train_schedule_seed is None else config.train_schedule_seed,
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
            tokenizer=None,
            config=shared_config,
            compute_val_bpb=config.compute_val_bpb,
            batch_schedule=batch_schedule,
        )
    dynamic_params = reports["dynamic_token_basis"]["parameter_count"]
    partial_params = reports["partial_untied"]["parameter_count"]
    return {
        "benchmark": "language_candidate_dynamic_token_basis",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
        },
        "compare_target": "partial_untied",
        "architecture_summary": {
            "candidate": "dynamic_token_basis",
            "delta_head": "adaptive routed low-rank basis in embedding space",
            "memory_path": "retrieval-conditioned basis coefficients plus exact token replay",
            "replaces_fixed_top_tokens": True,
        },
        "results": reports,
        "parameter_delta_vs_partial_untied": {
            "absolute": int(dynamic_params - partial_params),
            "relative": float((dynamic_params - partial_params) / max(partial_params, 1)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cheap paired compare for a dynamic token-basis recurrent candidate against partial_untied."
    )
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--train-blocks", type=int, default=1024)
    parser.add_argument("--val-blocks", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--partial-untied-tokens", type=int, default=512)
    parser.add_argument("--token-basis-rank", type=int, default=48)
    parser.add_argument("--routing-experts", type=int, default=4)
    parser.add_argument("--routing-top-k", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = DynamicTokenBasisConfig(
        cache_path=args.cache_path,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        device=args.device,
        partial_untied_tokens=args.partial_untied_tokens,
        token_basis_rank=args.token_basis_rank,
        routing_experts=args.routing_experts,
        routing_top_k=args.routing_top_k,
    )
    payload = run_dynamic_token_basis_compare(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
