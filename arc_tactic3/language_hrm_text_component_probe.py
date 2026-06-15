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


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


@dataclass(frozen=True, slots=True)
class HRMTextComponentProbeConfig:
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
    latent_refine_steps: int = 2
    high_update_interval: int = 2


class MagicNormPartialUntiedAssociativeLM(PartialUntiedAssociativeLM):
    """Partial-untied LM with HRM-Text's parameterless RMSNorm style only."""

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
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            memory_dim=memory_dim,
            dropout=dropout,
            max_length=max_length,
            untied_token_ids=untied_token_ids,
        )
        self.embedding_scale = math.sqrt(embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = _rms_norm(self.embedding(input_ids) * self.embedding_scale)
        states, _ = self.encoder(embeddings)
        states = _rms_norm(self.dropout(states))
        head_features = F.relu(self.head_fc(states)).square()
        base_features = _rms_norm(self.head_proj(head_features))
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        partial_logits = self.partial_head(base_features)
        full_partial = torch.zeros_like(base_logits)
        index = self.untied_token_ids.view(1, 1, -1).expand(input_ids.size(0), input_ids.size(1), -1)
        full_partial.scatter_add_(2, index, partial_logits.to(full_partial.dtype))
        base_logits = base_logits + full_partial

        query_keys = _rms_norm(self.query_proj(states))
        memory_keys = _rms_norm(self.key_proj(states))
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


class HRMTextComponentAssociativeLM(nn.Module):
    """Partial-untied recurrent LM with cheap HRM-Text-inspired probes.

    This keeps the incumbent exact-token replay and partial untied head, then
    adds parameterless RMS normalization, token-wise low/high latent refinement,
    and gated attention-context mixing into the tied head features.
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
        latent_refine_steps: int,
        high_update_interval: int,
    ) -> None:
        super().__init__()
        if latent_refine_steps < 1:
            raise ValueError("latent_refine_steps must be >= 1.")
        if high_update_interval < 1:
            raise ValueError("high_update_interval must be >= 1.")
        self.latent_refine_steps = latent_refine_steps
        self.high_update_interval = high_update_interval
        self.embedding_scale = math.sqrt(embedding_dim)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.low_update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.high_update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.high_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.token_replay_gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.memory_to_embedding = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.feature_gate = nn.Linear(hidden_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.partial_head = nn.Linear(embedding_dim, untied_token_ids.numel(), bias=True)
        self.memory_scale = nn.Parameter(torch.tensor(6.0))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("untied_token_ids", untied_token_ids.long(), persistent=False)
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=-1).unsqueeze(0),
            persistent=False,
        )

    def _refine_states(self, states: torch.Tensor) -> torch.Tensor:
        low = states
        high = states
        for step in range(self.latent_refine_steps):
            low_update = torch.tanh(self.low_update(torch.cat((states, high), dim=-1)))
            low = _rms_norm(low + low_update)
            if (step + 1) % self.high_update_interval == 0 or step == self.latent_refine_steps - 1:
                high_input = torch.cat((states, low), dim=-1)
                high_update = torch.tanh(self.high_update(high_input))
                high_gate = torch.sigmoid(self.high_gate(torch.cat((high, low), dim=-1)))
                high = _rms_norm(high + high_gate * high_update)
        return high

    def _causal_attention(self, input_ids: torch.Tensor, states: torch.Tensor, refined: torch.Tensor) -> torch.Tensor:
        queries = _rms_norm(self.query_proj(refined))
        keys = _rms_norm(self.key_proj(states))
        scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(queries.size(-1))
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        return attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = _rms_norm(self.embedding(input_ids) * self.embedding_scale)
        states, _ = self.encoder(embeddings)
        states = _rms_norm(self.dropout(states))
        refined = self._refine_states(states)

        attention = self._causal_attention(input_ids, states, refined)
        memory_context = torch.matmul(attention, states)
        head_features = F.relu(self.head_fc(refined)).square()
        base_features = self.head_proj(head_features)
        memory_features = self.memory_to_embedding(_rms_norm(memory_context))
        feature_gate = torch.sigmoid(self.feature_gate(refined))
        base_features = _rms_norm(base_features + feature_gate * memory_features)

        base_logits = F.linear(base_features * self.logit_scale, self.embedding.weight, self.output_bias)
        partial_logits = self.partial_head(base_features)
        partial_index = self.untied_token_ids.view(1, 1, -1).expand(input_ids.size(0), input_ids.size(1), -1)
        base_logits.scatter_add_(2, partial_index, partial_logits.to(base_logits.dtype))

        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        token_gate = torch.sigmoid(self.token_replay_gate(refined))
        gated_attention = (attention * (token_gate * self.memory_scale)).to(base_logits.dtype)
        base_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits


def _build_models(
    config: HRMTextComponentProbeConfig,
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
        "untied_token_ids": partial_token_ids,
    }
    return {
        "partial_untied": PartialUntiedAssociativeLM(**common),
        "magic_norm_partial_untied": MagicNormPartialUntiedAssociativeLM(**common),
        "hrm_text_component_probe": HRMTextComponentAssociativeLM(
            **common,
            latent_refine_steps=config.latent_refine_steps,
            high_update_interval=config.high_update_interval,
        ),
    }


def run_hrm_text_component_probe(config: HRMTextComponentProbeConfig) -> dict[str, Any]:
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

    probe_params = reports["hrm_text_component_probe"]["parameter_count"]
    partial_params = reports["partial_untied"]["parameter_count"]
    return {
        "benchmark": "language_hrm_text_component_probe",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
        },
        "compare_target": "partial_untied",
        "architecture_summary": {
            "candidate": "hrm_text_component_probe",
            "status": "probe_only",
            "components": [
                "magic_norm_partial_untied_is_tested_as_a_separate_component",
                "parameterless_rms_norm_on_embedding_state_and_head_features",
                "tokenwise_low_high_latent_refinement",
                "gated_attention_context_mixed_into_tied_head_features",
                "incumbent_exact_token_replay_and_partial_untied_delta_head",
            ],
            "not_included": [
                "response_only_prefixlm_loss",
                "flash_attention_runtime_stack",
                "fsdp_training_stack",
            ],
        },
        "fairness": {
            "same_dataset": True,
            "same_tokenizer": True,
            "paired_batch_schedule": True,
            "reseed_per_model": bool(config.reseed_per_model),
            "probe_claim": "screens components only; does not promote architecture without longer locked eval",
        },
        "results": reports,
        "parameter_delta_vs_partial_untied": {
            "absolute": int(probe_params - partial_params),
            "ratio": probe_params / max(partial_params, 1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe HRM-Text-inspired recurrent components against partial_untied."
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
    parser.add_argument("--partial-untied-tokens", type=int, default=512)
    parser.add_argument("--latent-refine-steps", type=int, default=2)
    parser.add_argument("--high-update-interval", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = HRMTextComponentProbeConfig(
        cache_path=args.cache_path,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        sequence_length=args.sequence_length,
        seed=args.seed,
        device=args.device,
        cache_dataset_on_device=args.cache_dataset_on_device,
        partial_untied_tokens=args.partial_untied_tokens,
        latent_refine_steps=args.latent_refine_steps,
        high_update_interval=args.high_update_interval,
    )
    payload = run_hrm_text_component_probe(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
