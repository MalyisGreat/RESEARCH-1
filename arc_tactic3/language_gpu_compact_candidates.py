from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from arc_tactic3.language_realtext_microbench import (
    RealTextConfig,
    TokenBlockDataset,
    _build_train_batch_schedule,
    set_global_seed,
)
from arc_tactic3.language_nanochat_actual_compare import _train_candidate
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM


@dataclass(frozen=True, slots=True)
class GPUCompactCandidatesConfig:
    cache_path: Path
    train_blocks: int = 512
    val_blocks: int = 64
    train_steps: int = 16
    eval_interval: int = 8
    seed: int = 13
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    sequence_length: int = 127
    dropout: float = 0.1
    embedding_dim: int = 224
    hidden_dim: int = 448
    memory_dim: int = 224
    partial_token_count: int = 1024
    local_window: int = 32
    older_chunk_size: int = 8
    use_amp: bool = torch.cuda.is_available()
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    cache_dataset_on_device: bool = True
    tensor_batching: bool = True
    train_schedule_seed: int | None = None


def _load_cached_datasets(config: GPUCompactCandidatesConfig) -> tuple[TokenBlockDataset, TokenBlockDataset, int]:
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


def _shared_realtext_config(config: GPUCompactCandidatesConfig) -> RealTextConfig:
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
        paired_train_batches=True,
        reseed_per_model=True,
        train_schedule_seed=config.train_schedule_seed,
        dropout=config.dropout,
    )


def _build_partial_maps(vocab_size: int, untied_token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    partial_index = untied_token_ids.long()
    token_to_partial = torch.full((vocab_size,), -1, dtype=torch.long)
    token_to_partial[partial_index.cpu()] = torch.arange(partial_index.numel(), dtype=torch.long)
    return partial_index, token_to_partial


def _scatter_attention_to_partial(
    *,
    attention: torch.Tensor,
    token_ids: torch.Tensor,
    token_to_partial: torch.Tensor,
    partial_size: int,
) -> torch.Tensor:
    slot_index = token_to_partial[token_ids]
    expanded_slots = slot_index.unsqueeze(1).expand(attention.size(0), attention.size(1), slot_index.size(1))
    valid = expanded_slots.ge(0)
    safe_slots = expanded_slots.clamp_min(0)
    memory_partial = attention.new_zeros(attention.size(0), attention.size(1), partial_size)
    memory_partial.scatter_add_(2, safe_slots, attention * valid.to(attention.dtype))
    return memory_partial


def _add_partial_to_base_logits_(
    *,
    base_logits: torch.Tensor,
    partial_logits: torch.Tensor,
    untied_token_ids: torch.Tensor,
) -> torch.Tensor:
    index = untied_token_ids.to(base_logits.device).view(1, 1, -1).expand(base_logits.size(0), base_logits.size(1), -1)
    base_logits.scatter_add_(2, index, partial_logits.to(base_logits.dtype))
    return base_logits


def _masked_causal_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attention = torch.softmax(scores, dim=-1)
    attention = attention * mask
    return attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class FusedCompactPartialUntiedLM(nn.Module):
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
        partial_index, token_to_partial = _build_partial_maps(vocab_size, untied_token_ids)
        self.register_buffer("untied_token_ids", partial_index, persistent=False)
        self.register_buffer("token_to_partial", token_to_partial, persistent=False)
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
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / math.sqrt(query_keys.size(-1))
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        attention = _masked_causal_softmax(scores, causal_mask)
        gate = torch.sigmoid(self.gate(states)).to(attention.dtype)
        memory_partial = _scatter_attention_to_partial(
            attention=attention * (gate * self.memory_scale),
            token_ids=input_ids,
            token_to_partial=self.token_to_partial,
            partial_size=self.untied_token_ids.numel(),
        )
        total_partial = self.partial_head(base_features) + memory_partial.to(base_features.dtype)
        return _add_partial_to_base_logits_(base_logits=base_logits, partial_logits=total_partial, untied_token_ids=self.untied_token_ids)


class FusedWindowedCompactPartialUntiedLM(FusedCompactPartialUntiedLM):
    def __init__(self, *, window_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        max_length = self._causal_mask.size(-1)
        positions = torch.arange(max_length)
        query_positions = positions.view(max_length, 1)
        key_positions = positions.view(1, max_length)
        window_mask = (key_positions < query_positions) & (key_positions >= (query_positions - window_size))
        self.register_buffer("_window_mask", window_mask.unsqueeze(0), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / math.sqrt(query_keys.size(-1))
        causal_mask = self._window_mask[:, : input_ids.size(1), : input_ids.size(1)]
        attention = _masked_causal_softmax(scores, causal_mask)
        gate = torch.sigmoid(self.gate(states)).to(attention.dtype)
        memory_partial = _scatter_attention_to_partial(
            attention=attention * (gate * self.memory_scale),
            token_ids=input_ids,
            token_to_partial=self.token_to_partial,
            partial_size=self.untied_token_ids.numel(),
        )
        total_partial = self.partial_head(base_features) + memory_partial.to(base_features.dtype)
        return _add_partial_to_base_logits_(base_logits=base_logits, partial_logits=total_partial, untied_token_ids=self.untied_token_ids)


class DenseValueWindowedPartialLM(nn.Module):
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
        window_size: int,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.value_proj = nn.Linear(hidden_dim, embedding_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.partial_head = nn.Linear(embedding_dim, untied_token_ids.numel(), bias=True)
        self.memory_scale = nn.Parameter(torch.tensor(6.0))
        partial_index, _ = _build_partial_maps(vocab_size, untied_token_ids)
        self.register_buffer("untied_token_ids", partial_index, persistent=False)
        positions = torch.arange(max_length)
        query_positions = positions.view(max_length, 1)
        key_positions = positions.view(1, max_length)
        window_mask = (key_positions < query_positions) & (key_positions >= (query_positions - window_size))
        self.register_buffer("_window_mask", window_mask.unsqueeze(0), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        base_partial = self.partial_head(base_features)

        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / math.sqrt(query_keys.size(-1))
        causal_mask = self._window_mask[:, : input_ids.size(1), : input_ids.size(1)]
        attention = _masked_causal_softmax(scores, causal_mask)
        memory_values = self.value_proj(states)
        memory_context = torch.matmul(attention.to(memory_values.dtype), memory_values)
        gate = torch.sigmoid(self.gate(states)).to(memory_context.dtype)
        total_partial = base_partial + gate * self.memory_scale.to(base_partial.dtype) * self.partial_head(memory_context)
        return _add_partial_to_base_logits_(base_logits=base_logits, partial_logits=total_partial, untied_token_ids=self.untied_token_ids)


class DensePartialWindowedPartialLM(nn.Module):
    """Dense local memory over partial-token logits.

    This keeps the compact partial-token path but avoids the extra value projection
    and second partial-head application used by DenseValueWindowedPartialLM.
    The goal is to keep the forward-friendly dense local read while making the
    training path cheaper.
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
        window_size: int,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
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
        partial_index, _ = _build_partial_maps(vocab_size, untied_token_ids)
        self.register_buffer("untied_token_ids", partial_index, persistent=False)
        positions = torch.arange(max_length)
        query_positions = positions.view(max_length, 1)
        key_positions = positions.view(1, max_length)
        window_mask = (key_positions < query_positions) & (key_positions >= (query_positions - window_size))
        self.register_buffer("_window_mask", window_mask.unsqueeze(0), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        base_partial = self.partial_head(base_features)

        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / math.sqrt(query_keys.size(-1))
        causal_mask = self._window_mask[:, : input_ids.size(1), : input_ids.size(1)]
        attention = _masked_causal_softmax(scores, causal_mask)
        gate = torch.sigmoid(self.gate(states)).to(base_partial.dtype)
        memory_partial = torch.matmul(attention.to(base_partial.dtype), base_partial)
        total_partial = base_partial + gate * self.memory_scale.to(base_partial.dtype) * memory_partial
        return _add_partial_to_base_logits_(base_logits=base_logits, partial_logits=total_partial, untied_token_ids=self.untied_token_ids)


class FusedLearnedChunkWriterPartialLM(nn.Module):
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
        partial_index, _ = _build_partial_maps(vocab_size, untied_token_ids)
        self.register_buffer("untied_token_ids", partial_index, persistent=False)
        num_chunks = (max_length + chunk_size - 1) // chunk_size
        chunk_ends = torch.arange(num_chunks, dtype=torch.long) * chunk_size + (chunk_size - 1)
        self.register_buffer("_chunk_end_positions", chunk_ends.clamp_max(max_length - 1), persistent=False)

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        return self.dropout(states)

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

    def _compress_chunks(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
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
        return chunk_states, chunk_features, seq_len, num_chunks

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        states = self._encode(input_ids)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        base_partial = self.partial_head(base_features)

        chunk_states, chunk_features, seq_len, num_chunks = self._compress_chunks(states)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(chunk_states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / math.sqrt(query_keys.size(-1))
        target_positions = torch.arange(seq_len, device=input_ids.device).view(1, -1, 1)
        causal_mask = self._chunk_end_positions[:num_chunks].view(1, 1, -1).to(input_ids.device) < target_positions
        attention = _masked_causal_softmax(scores, causal_mask)
        chunk_partial = self.partial_head(chunk_features)
        memory_partial = torch.matmul(attention.to(chunk_partial.dtype), chunk_partial)
        read_gate = torch.sigmoid(self.read_gate(states)).to(memory_partial.dtype)
        total_partial = base_partial + self.memory_scale.to(base_partial.dtype) * read_gate * memory_partial
        return _add_partial_to_base_logits_(base_logits=base_logits, partial_logits=total_partial, untied_token_ids=self.untied_token_ids)


class DenseValueLocalGlobalPartialLM(nn.Module):
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
        if local_window < 1 or older_chunk_size < 1:
            raise ValueError("local_window and older_chunk_size must be positive")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.partial_head = nn.Linear(embedding_dim, untied_token_ids.numel(), bias=True)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.value_proj = nn.Linear(hidden_dim, embedding_dim)
        self.memory_mix = nn.Linear(hidden_dim, 2)
        self.local_memory_scale = nn.Parameter(torch.tensor(6.0))
        self.global_memory_scale = nn.Parameter(torch.tensor(4.0))
        self.local_window = local_window
        self.older_chunk_size = older_chunk_size
        partial_index, _ = _build_partial_maps(vocab_size, untied_token_ids)
        self.register_buffer("untied_token_ids", partial_index, persistent=False)
        positions = torch.arange(max_length)
        query_positions = positions.view(max_length, 1)
        key_positions = positions.view(1, max_length)
        local_mask = (key_positions < query_positions) & (key_positions >= (query_positions - local_window))
        self.register_buffer("_local_mask", local_mask.unsqueeze(0), persistent=False)

    def _base_features(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        return base_features, base_logits

    def _global_context(self, states: torch.Tensor) -> torch.Tensor:
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
        global_queries = self.query_proj(states)
        global_keys = self.key_proj(chunk_summary)
        global_values = self.value_proj(chunk_summary)
        scores = torch.matmul(global_queries, global_keys.transpose(1, 2)) / math.sqrt(global_queries.size(-1))
        chunk_end_positions = ((torch.arange(chunk_count, device=states.device) + 1) * chunk_size - 1).clamp(max=sequence_length - 1)
        query_positions = torch.arange(sequence_length, device=states.device)
        older_cutoff = query_positions - self.local_window
        global_mask = chunk_end_positions.view(1, 1, -1) < older_cutoff.view(1, sequence_length, 1)
        attention = _masked_causal_softmax(scores, global_mask)
        return torch.matmul(attention.to(global_values.dtype), global_values)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        base_features, base_logits = self._base_features(states)
        base_partial = self.partial_head(base_features)

        queries = self.query_proj(states)
        local_keys = self.key_proj(states)
        local_values = self.value_proj(states)
        local_scores = torch.matmul(queries, local_keys.transpose(1, 2)) / math.sqrt(queries.size(-1))
        local_mask = self._local_mask[:, : input_ids.size(1), : input_ids.size(1)]
        local_attention = _masked_causal_softmax(local_scores, local_mask)
        local_context = torch.matmul(local_attention.to(local_values.dtype), local_values)
        global_context = self._global_context(states)

        mix = torch.softmax(self.memory_mix(states), dim=-1).to(base_features.dtype)
        total_partial = (
            base_partial
            + mix[..., :1] * self.local_memory_scale.to(base_features.dtype) * self.partial_head(local_context)
            + mix[..., 1:] * self.global_memory_scale.to(base_features.dtype) * self.partial_head(global_context)
        )
        return _add_partial_to_base_logits_(base_logits=base_logits, partial_logits=total_partial, untied_token_ids=self.untied_token_ids)


class FusedSharedLocalGlobalPartialLM(nn.Module):
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
        if local_window < 1 or older_chunk_size < 1:
            raise ValueError("local_window and older_chunk_size must be positive")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.partial_head = nn.Linear(embedding_dim, untied_token_ids.numel(), bias=True)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.value_proj = nn.Linear(hidden_dim, embedding_dim)
        self.memory_mix = nn.Linear(hidden_dim, 2)
        self.local_memory_scale = nn.Parameter(torch.tensor(6.0))
        self.global_memory_scale = nn.Parameter(torch.tensor(4.0))
        self.local_window = local_window
        self.older_chunk_size = older_chunk_size
        partial_index, token_to_partial = _build_partial_maps(vocab_size, untied_token_ids)
        self.register_buffer("untied_token_ids", partial_index, persistent=False)
        self.register_buffer("token_to_partial", token_to_partial, persistent=False)
        positions = torch.arange(max_length)
        query_positions = positions.view(max_length, 1)
        key_positions = positions.view(1, max_length)
        local_mask = (key_positions < query_positions) & (key_positions >= (query_positions - local_window))
        self.register_buffer("_local_mask", local_mask.unsqueeze(0), persistent=False)

    def _base_features(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        return base_features, base_logits

    def _global_chunk_partial(self, states: torch.Tensor) -> torch.Tensor:
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
        global_queries = self.query_proj(states)
        global_keys = self.key_proj(chunk_summary)
        global_values = self.value_proj(chunk_summary)
        scores = torch.matmul(global_queries, global_keys.transpose(1, 2)) / math.sqrt(global_queries.size(-1))
        chunk_end_positions = ((torch.arange(chunk_count, device=states.device) + 1) * chunk_size - 1).clamp(max=sequence_length - 1)
        query_positions = torch.arange(sequence_length, device=states.device)
        older_cutoff = query_positions - self.local_window
        global_mask = chunk_end_positions.view(1, 1, -1) < older_cutoff.view(1, sequence_length, 1)
        attention = _masked_causal_softmax(scores, global_mask)
        global_context = torch.matmul(attention.to(global_values.dtype), global_values)
        return self.partial_head(global_context)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        base_features, base_logits = self._base_features(states)
        base_partial = self.partial_head(base_features)

        queries = self.query_proj(states)
        local_keys = self.key_proj(states)
        local_scores = torch.matmul(queries, local_keys.transpose(1, 2)) / math.sqrt(queries.size(-1))
        local_mask = self._local_mask[:, : input_ids.size(1), : input_ids.size(1)]
        local_attention = _masked_causal_softmax(local_scores, local_mask)
        local_partial = _scatter_attention_to_partial(
            attention=local_attention,
            token_ids=input_ids,
            token_to_partial=self.token_to_partial,
            partial_size=self.untied_token_ids.numel(),
        )
        global_partial = self._global_chunk_partial(states)

        mix = torch.softmax(self.memory_mix(states), dim=-1).to(base_features.dtype)
        total_partial = (
            base_partial
            + mix[..., :1] * self.local_memory_scale.to(base_features.dtype) * local_partial.to(base_features.dtype)
            + mix[..., 1:] * self.global_memory_scale.to(base_features.dtype) * global_partial.to(base_features.dtype)
        )
        return _add_partial_to_base_logits_(base_logits=base_logits, partial_logits=total_partial, untied_token_ids=self.untied_token_ids)


def _build_models(config: GPUCompactCandidatesConfig, *, vocab_size: int) -> dict[str, Callable[[], nn.Module]]:
    untied_token_ids = torch.arange(config.partial_token_count)
    common = {
        "vocab_size": vocab_size,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "memory_dim": config.memory_dim,
        "dropout": config.dropout,
        "max_length": config.sequence_length,
        "untied_token_ids": untied_token_ids,
    }
    return {
        "partial_untied": lambda: PartialUntiedAssociativeLM(**common),
        "compact_partial": lambda: FusedCompactPartialUntiedLM(**common),
        "compact_window32": lambda: FusedWindowedCompactPartialUntiedLM(window_size=config.local_window, **common),
        "dense_value_window32": lambda: DenseValueWindowedPartialLM(window_size=config.local_window, **common),
        "dense_partial_window32": lambda: DensePartialWindowedPartialLM(window_size=config.local_window, **common),
        "compact_chunk_writer": lambda: FusedLearnedChunkWriterPartialLM(chunk_size=config.older_chunk_size, **common),
        "compact_local_global": lambda: FusedSharedLocalGlobalPartialLM(
            local_window=config.local_window,
            older_chunk_size=config.older_chunk_size,
            **common,
        ),
        "dense_value_local_global": lambda: DenseValueLocalGlobalPartialLM(
            local_window=config.local_window,
            older_chunk_size=config.older_chunk_size,
            **common,
        ),
    }


def _mean_dicts(reports: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("initial_val_loss", "final_val_loss", "train_tok_per_sec", "pure_train_tok_per_sec", "peak_vram_mb")
    out: dict[str, float] = {}
    for key in keys:
        values = [float(report[key]) for report in reports if report.get(key) is not None]
        if values:
            out[key] = sum(values) / len(values)
    return out


def run_gpu_compact_candidates(
    config: GPUCompactCandidatesConfig,
    *,
    seeds: list[int],
    selected_models: list[str] | None = None,
) -> dict[str, Any]:
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(config)
    builders = _build_models(config, vocab_size=vocab_size)
    if selected_models:
        selected = set(selected_models)
        builders = {name: builder for name, builder in builders.items() if name in selected}
    reports: dict[str, list[dict[str, Any]]] = {name: [] for name in builders}
    for seed in seeds:
        shared_config = _shared_realtext_config(config)
        shared_config = RealTextConfig(**{**asdict(shared_config), "seed": seed})
        schedule_seed = seed if config.train_schedule_seed is None else config.train_schedule_seed
        batch_schedule = _build_train_batch_schedule(
            len(train_dataset),
            batch_size=shared_config.batch_size,
            steps=shared_config.train_steps,
            seed=schedule_seed,
            drop_last=True,
        )
        for model_name, builder in builders.items():
            set_global_seed(seed)
            model = builder()
            reports[model_name].append(
                _train_candidate(
                    model,
                    train_dataset,
                    val_dataset,
                    model_name=model_name,
                    tokenizer=None,
                    config=shared_config,
                    compute_val_bpb=False,
                    batch_schedule=batch_schedule,
                )
            )
            del model
            if shared_config.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
    means = {name: _mean_dicts(result_list) for name, result_list in reports.items()}
    baseline = means["partial_untied"]
    deltas: dict[str, dict[str, float]] = {}
    for name, result in means.items():
        if name == "partial_untied":
            continue
        delta_report = {
            "delta_loss_vs_baseline": result["final_val_loss"] - baseline["final_val_loss"],
            "delta_loss_pct_vs_baseline": 100.0 * (result["final_val_loss"] - baseline["final_val_loss"]) / baseline["final_val_loss"],
        }
        if "pure_train_tok_per_sec" in result and "pure_train_tok_per_sec" in baseline:
            delta_report["delta_pure_tok_per_sec_pct_vs_baseline"] = (
                100.0 * (result["pure_train_tok_per_sec"] - baseline["pure_train_tok_per_sec"]) / baseline["pure_train_tok_per_sec"]
            )
        deltas[name] = delta_report
    return {
        "benchmark": "language_gpu_compact_candidates",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
            "seeds": seeds,
        },
        "results_by_seed": reports,
        "mean_results": means,
        "deltas_vs_baseline": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GPU-compact recurrent memory candidates on cached real text.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--train-blocks", type=int, default=512)
    parser.add_argument("--val-blocks", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--extra-seed", type=int, action="append", default=[])
    parser.add_argument("--model", type=str, action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = GPUCompactCandidatesConfig(
        cache_path=args.cache_path,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        device=args.device,
    )
    seeds = [args.seed, *args.extra_seed]
    payload = run_gpu_compact_candidates(config, seeds=seeds, selected_models=args.model or None)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
