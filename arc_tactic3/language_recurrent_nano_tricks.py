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
from arc_tactic3.language_nanochat_actual_compare import (
    NanochatMiniLM,
    _load_cached_datasets,
    _shared_realtext_config,
    _train_candidate,
)
from arc_tactic3.language_realtext_microbench import AssociativeRecurrentLM, _build_train_batch_schedule
from arc_tactic3.language_throughput_candidates import GRUOnlyLM


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


@dataclass(frozen=True, slots=True)
class RecurrentNanoTricksConfig:
    cache_path: Path
    tokenizer_name: str = "gpt2"
    train_blocks: int = 8192
    val_blocks: int = 512
    sequence_length: int = 127
    batch_size: int = 16
    eval_batch_size: int = 32
    train_steps: int = 384
    eval_interval: int = 96
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 13
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    tensor_batching: bool = False
    cache_dataset_on_device: bool = False
    compute_val_bpb: bool = True
    recurrent_embedding_dim: int = 144
    recurrent_hidden_dim: int = 288
    recurrent_memory_dim: int = 144
    dropout: float = 0.1
    window_size: int = 32
    paired_train_batches: bool = True
    reseed_per_model: bool = True
    train_schedule_seed: int | None = None
    optimizer_recipe: str = "default"
    warmup_steps: int = 0
    lr_schedule: str = "none"
    min_lr_scale: float = 1.0
    untied_rank: int = 48
    partial_untied_tokens: int = 512
    nano_n_layer: int = 4
    nano_n_head: int = 4
    nano_n_kv_head: int = 4
    nano_n_embd: int = 40
    nano_window_pattern: str = "SSSL"
    nano_softcap: float = 15.0
    nano_use_value_embeddings: bool = True
    nano_use_smear: bool = True
    nano_use_backout: bool = True


class RMSNormAssociativeLM(nn.Module):
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
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
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
        embeddings = _rms_norm(self.embedding(input_ids))
        states, _ = self.encoder(embeddings)
        states = _rms_norm(self.dropout(states))
        base_features = _rms_norm(self.hidden_to_embedding(states))
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        gate = torch.sigmoid(self.gate(states))
        gated_attention = (attention * (mem_gate * self.memory_scale)).to(base_logits.dtype)
        base_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits


class ReLU2HeadAssociativeLM(nn.Module):
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
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
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
        head_features = self.head_fc(states)
        head_features = F.relu(head_features).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
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


class NormalizedReLU2HeadAssociativeLM(nn.Module):
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
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
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
        normalized_states = self.head_norm(states)
        head_features = self.head_fc(normalized_states)
        head_features = F.relu(head_features).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
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


class GatedResidualHeadAssociativeLM(nn.Module):
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
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.residual_proj = nn.Linear(hidden_dim, embedding_dim)
        self.mix_gate = nn.Linear(hidden_dim, embedding_dim)
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
        relu2_features = self.head_proj(F.relu(self.head_fc(states)).square())
        residual_features = self.residual_proj(states)
        mix = torch.sigmoid(self.mix_gate(states))
        base_features = mix * relu2_features + (1.0 - mix) * residual_features
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
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


class RefinedHeadAssociativeLM(nn.Module):
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
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.refine_fc = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.refine_proj = nn.Linear(hidden_dim, hidden_dim)
        self.refine_gate = nn.Linear(hidden_dim, hidden_dim)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
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
        refined_update = self.refine_proj(F.glu(self.refine_fc(states), dim=-1))
        refine_mix = torch.sigmoid(self.refine_gate(states))
        refined_states = states + refine_mix * refined_update
        head_features = self.head_proj(F.relu(self.head_fc(refined_states)).square())
        base_logits = F.linear(head_features, self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(refined_states)
        memory_keys = self.key_proj(refined_states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        gate = torch.sigmoid(self.gate(refined_states))
        gated_attention = (attention * (gate * self.memory_scale)).to(base_logits.dtype)
        base_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits


class RefinedHeadAssociativeLM(nn.Module):
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
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.refine_fc = nn.Linear(embedding_dim, embedding_dim)
        self.refine_gate = nn.Linear(embedding_dim, 1)
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
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        refine_features = F.relu(self.refine_fc(base_features))
        refine_gate = torch.sigmoid(self.refine_gate(refine_features))
        merged_features = base_features * (1 - refine_gate) + refine_features * refine_gate
        base_logits = F.linear(merged_features, self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        mem_gate = torch.sigmoid(self.gate(states))
        gated_attention = (attention * (mem_gate * self.memory_scale)).to(base_logits.dtype)
        base_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits


class ReLU2WindowedAssociativeLM(nn.Module):
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
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
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
        head_features = self.head_fc(states)
        head_features = F.relu(head_features).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
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


class UntiedHeadAssociativeLM(nn.Module):
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
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.hidden_to_embedding = nn.Linear(hidden_dim, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=True)
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
        base_logits = self.lm_head(self.hidden_to_embedding(states))
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        gate = torch.sigmoid(self.gate(states))
        memory_logits = torch.zeros_like(base_logits)
        gated_attention = (attention * (gate * self.memory_scale)).to(base_logits.dtype)
        memory_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits + memory_logits


class FactorizedUntiedHeadAssociativeLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        memory_dim: int,
        dropout: float,
        max_length: int,
        untied_rank: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.factor_down = nn.Linear(embedding_dim, untied_rank, bias=False)
        self.factor_up = nn.Linear(untied_rank, vocab_size, bias=True)
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
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = self.factor_up(self.factor_down(base_features))
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
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


class LowRankUntiedDeltaAssociativeLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        memory_dim: int,
        dropout: float,
        max_length: int,
        untied_rank: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.delta_down = nn.Linear(embedding_dim, untied_rank, bias=False)
        self.delta_up = nn.Linear(untied_rank, vocab_size, bias=False)
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
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        base_logits = base_logits + self.delta_up(self.delta_down(base_features))
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
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


class PartialUntiedAssociativeLM(nn.Module):
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
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.partial_head = nn.Linear(embedding_dim, untied_token_ids.numel(), bias=True)
        self.memory_scale = nn.Parameter(torch.tensor(6.0))
        self.register_buffer("untied_token_ids", untied_token_ids.long(), persistent=False)
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=-1).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        partial_logits = self.partial_head(base_features)
        full_partial = torch.zeros_like(base_logits)
        index = self.untied_token_ids.view(1, 1, -1).expand(input_ids.size(0), input_ids.size(1), -1)
        full_partial.scatter_add_(2, index, partial_logits)
        base_logits = base_logits + full_partial
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
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


class SmearAssociativeLM(nn.Module):
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
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.hidden_to_embedding = nn.Linear(hidden_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.memory_scale = nn.Parameter(torch.tensor(6.0))
        self.smear_gate_channels = min(24, embedding_dim)
        self.smear_gate = nn.Linear(self.smear_gate_channels, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=-1).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        if embeddings.size(1) > 1:
            gate = self.smear_lambda.to(embeddings.dtype) * torch.sigmoid(
                self.smear_gate(embeddings[:, 1:, : self.smear_gate_channels])
            )
            embeddings = torch.cat((embeddings[:, :1], embeddings[:, 1:] + gate * embeddings[:, :-1]), dim=1)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        base_logits = F.linear(self.hidden_to_embedding(states), self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
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


class ReLU2SmearAssociativeLM(nn.Module):
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
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.memory_scale = nn.Parameter(torch.tensor(6.0))
        self.smear_gate_channels = min(24, embedding_dim)
        self.smear_gate = nn.Linear(self.smear_gate_channels, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=-1).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        if embeddings.size(1) > 1:
            gate = self.smear_lambda.to(embeddings.dtype) * torch.sigmoid(
                self.smear_gate(embeddings[:, 1:, : self.smear_gate_channels])
            )
            embeddings = torch.cat((embeddings[:, :1], embeddings[:, 1:] + gate * embeddings[:, :-1]), dim=1)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = self.head_fc(states)
        head_features = F.relu(head_features).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
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


class ResidualGatedAssociativeLM(nn.Module):
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
        self.input_residual = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.hidden_to_embedding = nn.Linear(hidden_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.memory_scale = nn.Parameter(torch.tensor(6.0))
        self.resid_lambda = nn.Parameter(torch.ones(1))
        self.input_lambda = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=-1).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        mixed_states = self.resid_lambda.to(states.dtype) * states + self.input_lambda.to(states.dtype) * self.input_residual(
            embeddings
        )
        mixed_states = _rms_norm(mixed_states)
        base_logits = F.linear(self.hidden_to_embedding(mixed_states), self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(mixed_states)
        memory_keys = self.key_proj(mixed_states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        gate = torch.sigmoid(self.gate(mixed_states))
        gated_attention = (attention * (gate * self.memory_scale)).to(base_logits.dtype)
        base_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits


class GroupedFusionHeadAssociativeLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        memory_dim: int,
        dropout: float,
        max_length: int,
        groups: int = 4,
    ) -> None:
        super().__init__()
        if embedding_dim % groups != 0:
            raise ValueError("embedding_dim must divide evenly by groups.")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.head_scrub = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim // groups, embedding_dim // groups),
                nn.ReLU(),
            )
            for _ in range(groups)
        )
        self.head_proj = nn.Linear(embedding_dim, embedding_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.memory_scale = nn.Parameter(torch.tensor(6.0))
        self.groups = groups
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=-1).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        chunked = states.chunk(self.groups, dim=-1)
        scrubbed = []
        for chunk, scrub in zip(chunked, self.head_scrub):
            scrubbed.append(scrub(chunk))
        base_features = torch.cat(scrubbed, dim=-1)
        base_features = self.head_proj(base_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
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


class GatedResidualHeadAssociativeLM(nn.Module):
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
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.residual_proj = nn.Linear(embedding_dim, embedding_dim)
        self.memory_scale = nn.Parameter(torch.tensor(6.0))
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=-1).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = self.head_fc(states)
        head_features = F.relu(head_features).square()
        base_features = self.head_proj(head_features)
        residual = F.relu(self.residual_proj(base_features))
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        residual_logits = F.linear(residual, self.embedding.weight, torch.zeros_like(self.output_bias))
        gate = torch.sigmoid(self.gate(states))
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        gated_attention = (attention * (gate * self.memory_scale)).to(base_logits.dtype)
        base_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits + gate * residual_logits


class ReLU2UntiedHeadAssociativeLM(nn.Module):
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
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=True)
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
        head_features = self.head_fc(states)
        head_features = F.relu(head_features).square()
        base_features = self.head_proj(head_features)
        base_logits = self.lm_head(base_features)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        gate = torch.sigmoid(self.gate(states))
        memory_logits = torch.zeros_like(base_logits)
        gated_attention = (attention * (gate * self.memory_scale)).to(base_logits.dtype)
        memory_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits + memory_logits


def _top_token_ids(dataset, *, count: int, vocab_size: int) -> torch.Tensor:
    flat_tokens = dataset.targets.reshape(-1)
    histogram = torch.bincount(flat_tokens, minlength=vocab_size)
    top_k = min(max(count, 1), vocab_size)
    return torch.topk(histogram, k=top_k, largest=True, sorted=False).indices


def _build_models(
    config: RecurrentNanoTricksConfig,
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
        "recurrent_baseline": AssociativeRecurrentLM(**common),
        "recurrent_champion": ReLU2HeadAssociativeLM(**common),
        "normalized_relu2": NormalizedReLU2HeadAssociativeLM(**common),
        "gated_residual_head": GatedResidualHeadAssociativeLM(**common),
        "refined_head": RefinedHeadAssociativeLM(**common),
        "gru_only": GRUOnlyLM(
            vocab_size=vocab_size,
            embedding_dim=config.recurrent_embedding_dim,
            hidden_dim=config.recurrent_hidden_dim,
            dropout=config.dropout,
        ),
        "factorized_untied": FactorizedUntiedHeadAssociativeLM(
            untied_rank=config.untied_rank,
            **common,
        ),
        "low_rank_untied": LowRankUntiedDeltaAssociativeLM(
            untied_rank=config.untied_rank,
            **common,
        ),
        "partial_untied": PartialUntiedAssociativeLM(
            untied_token_ids=partial_token_ids,
            **common,
        ),
        "full_untied": UntiedHeadAssociativeLM(**common),
        "nanochat_small": NanochatMiniLM(
            vocab_size=vocab_size,
            sequence_length=config.sequence_length,
            n_layer=config.nano_n_layer,
            n_head=config.nano_n_head,
            n_kv_head=config.nano_n_kv_head,
            n_embd=config.nano_n_embd,
            window_pattern=config.nano_window_pattern,
            softcap=config.nano_softcap,
            use_value_embeddings=config.nano_use_value_embeddings,
            use_smear=config.nano_use_smear,
            use_backout=config.nano_use_backout,
        ),
    }


def run_recurrent_nano_tricks(config: RecurrentNanoTricksConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(config)
    partial_token_ids = _top_token_ids(
        train_dataset,
        count=config.partial_untied_tokens,
        vocab_size=vocab_size,
    )
    tokenizer = None
    if config.compute_val_bpb:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True, local_files_only=True)
    shared_config = _shared_realtext_config(config)
    reports: dict[str, dict[str, Any]] = {}
    schedule_seed = config.seed if config.train_schedule_seed is None else config.train_schedule_seed
    batch_schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=shared_config.batch_size,
        steps=shared_config.train_steps,
        seed=schedule_seed,
        drop_last=True,
    )
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
        "benchmark": "language_recurrent_nano_tricks",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
        },
        "results": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Nanochat-inspired recurrent tricks one by one.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--train-blocks", type=int, default=8192)
    parser.add_argument("--val-blocks", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=384)
    parser.add_argument("--eval-interval", type=int, default=96)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = RecurrentNanoTricksConfig(
        cache_path=args.cache_path,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        device=args.device,
    )
    payload = run_recurrent_nano_tricks(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
