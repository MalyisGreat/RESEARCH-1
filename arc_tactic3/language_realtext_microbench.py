from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from array import array
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DownloadConfig, load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from arc_tactic3.language_fastlearn_benchmark import GPT2Block, count_parameters, set_global_seed


_DEFAULT_DATASET_CACHE_DIR = Path(__file__).resolve().parent / "benchmark_runs" / "dataset_cache"


@dataclass(frozen=True, slots=True)
class RealTextConfig:
    seed: int = 13
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    train_split: str = "train"
    validation_split: str = "validation"
    text_column: str = "text"
    tokenizer_name: str = "gpt2"
    streaming: bool = False
    train_token_cap: int | None = None
    validation_token_cap: int | None = None
    sequence_length: int = 64
    max_train_sequences: int = 2048
    max_eval_sequences: int = 256
    train_steps: int = 64
    eval_interval: int = 16
    initial_eval: bool = True
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()
    recurrent_embedding_dim: int = 64
    recurrent_hidden_dim: int = 128
    recurrent_memory_dim: int = 64
    gpt_d_model: int = 64
    gpt_heads: int = 4
    gpt_layers: int = 2
    gpt_ff_dim: int = 256
    dropout: float = 0.1
    tensor_batching: bool = False
    cache_dataset_on_device: bool = False
    auto_cache_dataset_on_device: bool = False
    pin_memory: bool = torch.cuda.is_available()
    dataset_cache_path: Path | None = None
    use_fused_adamw: bool = torch.cuda.is_available()
    local_files_only: bool = True
    tokenization_batch_size: int = 256
    enable_tf32: bool = torch.cuda.is_available()
    paired_train_batches: bool = True
    reseed_per_model: bool = True
    train_schedule_seed: int | None = None
    optimizer_recipe: str = "default"
    warmup_steps: int = 0
    lr_schedule: str = "none"
    min_lr_scale: float = 1.0


class TokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, input_ids: torch.Tensor, targets: torch.Tensor) -> None:
        self.input_ids = input_ids.long()
        self.targets = targets.long()

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "targets": self.targets[index],
        }


def _effective_tokenization_batch_size(*, requested_batch_size: int, max_sequences: int) -> int:
    if max_sequences <= 256:
        return min(requested_batch_size, 32)
    if max_sequences <= 1024:
        return min(requested_batch_size, 64)
    return requested_batch_size


class AssociativeRecurrentLM(nn.Module):
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


_BATCH_SCHEDULE_CACHE: dict[tuple[int, int, int, int | None, bool], list[torch.Tensor]] = {}


class GPT2CausalLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList(
            GPT2Block(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.register_buffer(
            "_position_ids",
            torch.arange(max_length, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = self._position_ids[:, : input_ids.size(1)].expand_as(input_ids)
        states = self.embedding(input_ids) + self.position_embedding(positions)
        states = self.dropout(states)
        causal_mask = self._causal_mask[: input_ids.size(1), : input_ids.size(1)]
        for block in self.blocks:
            states = block(states, causal_mask=causal_mask, key_padding_mask=None)
        return self.output(self.final_norm(states))


def _texts_to_blocks(
    texts: Iterable[str],
    *,
    tokenizer,
    sequence_length: int,
    max_sequences: int,
) -> TokenBlockDataset:
    eos_id = tokenizer.eos_token_id
    token_stream: list[int] = []
    token_budget = max_sequences * (sequence_length + 1)
    batch_size = _effective_tokenization_batch_size(
        requested_batch_size=max(1, getattr(tokenizer, "_codex_batch_size", 256)),
        max_sequences=max_sequences,
    )
    saw_non_empty = False
    pending_texts: list[str] = []

    def _consume_batch(batch_texts: Sequence[str]) -> None:
        nonlocal token_stream
        batch_tokens: Sequence[Sequence[int]] | None = None
        try:
            encoded = tokenizer(batch_texts, add_special_tokens=False)
            batch_tokens = encoded["input_ids"]
        except TypeError:
            batch_tokens = None
        if batch_tokens is None:
            for text in batch_texts:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if not token_ids:
                    continue
                remaining = token_budget - len(token_stream)
                if remaining <= 0:
                    break
                token_ids = token_ids + [eos_id]
                token_stream.extend(token_ids[:remaining])
                if len(token_stream) >= token_budget:
                    break
            return
        for token_ids in batch_tokens:
            if not token_ids:
                continue
            remaining = token_budget - len(token_stream)
            if remaining <= 0:
                break
            token_stream.extend((list(token_ids) + [eos_id])[:remaining])
            if len(token_stream) >= token_budget:
                break
    for text in texts:
        if not text or not text.strip():
            continue
        saw_non_empty = True
        pending_texts.append(text)
        if len(pending_texts) < batch_size:
            continue
        _consume_batch(pending_texts)
        pending_texts = []
        if len(token_stream) >= token_budget:
            break
    if pending_texts and len(token_stream) < token_budget:
        _consume_batch(pending_texts)
    if not saw_non_empty:
        raise ValueError("No non-empty texts available for tokenization.")
    block_size = sequence_length + 1
    usable_tokens = (len(token_stream) // block_size) * block_size
    token_stream = token_stream[:usable_tokens]
    if not token_stream:
        raise ValueError("No tokens available after tokenizing dataset texts.")
    blocks = torch.tensor(token_stream, dtype=torch.long).view(-1, block_size)
    blocks = blocks[:max_sequences]
    input_ids = blocks[:, :-1].contiguous()
    targets = blocks[:, 1:].contiguous()
    return TokenBlockDataset(input_ids, targets)


def _token_ids_to_blocks(token_ids: np.ndarray, *, sequence_length: int) -> TokenBlockDataset:
    block_size = sequence_length + 1
    usable_tokens = (int(token_ids.size) // block_size) * block_size
    if usable_tokens <= 0:
        raise ValueError("Not enough streamed tokens to form a single training block.")
    trimmed = token_ids[:usable_tokens]
    blocks = torch.from_numpy(trimmed.astype(np.int64, copy=False)).view(-1, block_size)
    input_ids = blocks[:, :-1].contiguous()
    targets = blocks[:, 1:].contiguous()
    return TokenBlockDataset(input_ids, targets)


def _serialize_dataset(dataset: TokenBlockDataset) -> dict[str, torch.Tensor]:
    return {
        "input_ids": dataset.input_ids.cpu(),
        "targets": dataset.targets.cpu(),
    }


def _deserialize_dataset(payload: dict[str, torch.Tensor]) -> TokenBlockDataset:
    return TokenBlockDataset(payload["input_ids"], payload["targets"])


def _resolved_dataset_cache_path(config: RealTextConfig) -> Path | None:
    if config.dataset_cache_path is not None:
        return config.dataset_cache_path
    if config.streaming:
        return None
    payload = {
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config,
        "train_split": config.train_split,
        "validation_split": config.validation_split,
        "text_column": config.text_column,
        "tokenizer_name": config.tokenizer_name,
        "sequence_length": config.sequence_length,
        "max_train_sequences": config.max_train_sequences,
        "max_eval_sequences": config.max_eval_sequences,
        "tokenization_batch_size": config.tokenization_batch_size,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    safe_stem = f"{config.dataset_name}_{config.dataset_config}".replace("/", "_")
    return _DEFAULT_DATASET_CACHE_DIR / f"{safe_stem}_{digest}.pt"


def _stream_token_ids(
    stream_rows,
    *,
    tokenizer,
    text_column: str,
    token_cap: int,
) -> np.ndarray:
    if token_cap <= 0:
        raise ValueError("token_cap must be positive.")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must expose an eos_token_id for streamed packing.")
    buffer = array("I")
    tokens_seen = 0
    for row in stream_rows:
        text = row.get(text_column, "")
        if not text or not text.strip():
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        token_ids.append(eos_id)
        remaining = token_cap - tokens_seen
        if remaining <= 0:
            break
        if len(token_ids) > remaining:
            token_ids = token_ids[:remaining]
        buffer.extend(token_ids)
        tokens_seen += len(token_ids)
        if tokens_seen >= token_cap:
            break
    if tokens_seen != token_cap:
        raise ValueError(f"Requested {token_cap} streamed tokens, only collected {tokens_seen}.")
    return np.frombuffer(buffer, dtype=np.uint32)


def load_realtext_datasets(config: RealTextConfig) -> tuple[TokenBlockDataset, TokenBlockDataset, int]:
    dataset_cache_path = _resolved_dataset_cache_path(config)
    if dataset_cache_path is not None and dataset_cache_path.exists():
        payload = torch.load(dataset_cache_path, map_location="cpu", weights_only=False)
        return (
            _deserialize_dataset(payload["train_dataset"]),
            _deserialize_dataset(payload["val_dataset"]),
            int(payload["vocab_size"]),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        use_fast=True,
        local_files_only=config.local_files_only,
    )
    setattr(tokenizer, "_codex_batch_size", config.tokenization_batch_size)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {config.tokenizer_name} does not expose an eos_token_id.")
    download_config = DownloadConfig(local_files_only=config.local_files_only)
    if config.streaming:
        if config.train_token_cap is None or config.validation_token_cap is None:
            raise ValueError("Streaming mode requires both train_token_cap and validation_token_cap.")
        token_cap = config.train_token_cap + config.validation_token_cap
        stream_rows = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.train_split,
            streaming=True,
            download_config=download_config,
        )
        token_ids = _stream_token_ids(
            stream_rows,
            tokenizer=tokenizer,
            text_column=config.text_column,
            token_cap=token_cap,
        )
        train_ids = token_ids[: config.train_token_cap]
        val_ids = token_ids[config.train_token_cap : config.train_token_cap + config.validation_token_cap]
        train_dataset = _token_ids_to_blocks(train_ids, sequence_length=config.sequence_length)
        val_dataset = _token_ids_to_blocks(val_ids, sequence_length=config.sequence_length)
    else:
        dataset_train = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.train_split,
            download_config=download_config,
        )
        dataset_val = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.validation_split,
            download_config=download_config,
        )
        train_dataset = _texts_to_blocks(
            (row[config.text_column] for row in dataset_train),
            tokenizer=tokenizer,
            sequence_length=config.sequence_length,
            max_sequences=config.max_train_sequences,
        )
        val_dataset = _texts_to_blocks(
            (row[config.text_column] for row in dataset_val),
            tokenizer=tokenizer,
            sequence_length=config.sequence_length,
            max_sequences=config.max_eval_sequences,
        )
    if dataset_cache_path is not None:
        dataset_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vocab_size": tokenizer.vocab_size,
                "train_dataset": _serialize_dataset(train_dataset),
                "val_dataset": _serialize_dataset(val_dataset),
            },
            dataset_cache_path,
        )
    return train_dataset, val_dataset, tokenizer.vocab_size


def build_model_builders(config: RealTextConfig, *, vocab_size: int) -> dict[str, Callable[[], nn.Module]]:
    return {
        "associative_recurrent": lambda: AssociativeRecurrentLM(
            vocab_size=vocab_size,
            embedding_dim=config.recurrent_embedding_dim,
            hidden_dim=config.recurrent_hidden_dim,
            memory_dim=config.recurrent_memory_dim,
            dropout=config.dropout,
            max_length=config.sequence_length,
        ),
        "gpt2_like": lambda: GPT2CausalLM(
            vocab_size=vocab_size,
            d_model=config.gpt_d_model,
            n_heads=config.gpt_heads,
            layers=config.gpt_layers,
            ff_dim=config.gpt_ff_dim,
            dropout=config.dropout,
            max_length=config.sequence_length,
        ),
    }


def build_models(config: RealTextConfig, *, vocab_size: int) -> dict[str, nn.Module]:
    return {name: builder() for name, builder in build_model_builders(config, vocab_size=vocab_size).items()}


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda") if value.device != device else value
        for key, value in batch.items()
    }


def _dataset_tensors(
    dataset: TokenBlockDataset,
    *,
    device: torch.device,
    cache_on_device: bool,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = dataset.input_ids
    targets = dataset.targets
    if pin_memory and input_ids.device.type == "cpu":
        input_ids = input_ids.pin_memory()
        targets = targets.pin_memory()
    if cache_on_device:
        input_ids = input_ids.to(device, non_blocking=pin_memory)
        targets = targets.to(device, non_blocking=pin_memory)
    return input_ids, targets


def _iter_tensor_batches(
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    device: torch.device,
    non_blocking: bool,
):
    total = input_ids.size(0)
    if shuffle:
        indices = torch.randperm(total, device=input_ids.device)
    else:
        indices = torch.arange(total, device=input_ids.device)
    stop = total if not drop_last else total - (total % batch_size)
    for start in range(0, stop, batch_size):
        end = min(start + batch_size, total)
        if drop_last and end - start < batch_size:
            break
        batch_indices = indices[start:end]
        batch_input_ids = input_ids.index_select(0, batch_indices)
        batch_targets = targets.index_select(0, batch_indices)
        if batch_input_ids.device != device:
            batch_input_ids = batch_input_ids.to(device, non_blocking=non_blocking)
            batch_targets = batch_targets.to(device, non_blocking=non_blocking)
        yield {
            "input_ids": batch_input_ids,
            "targets": batch_targets,
        }


def _build_train_batch_schedule(
    total_examples: int,
    *,
    batch_size: int,
    steps: int,
    seed: int,
    drop_last: bool = True,
) -> list[torch.Tensor]:
    if total_examples <= 0:
        raise ValueError("total_examples must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if steps <= 0:
        return []
    usable_total = total_examples if not drop_last else total_examples - (total_examples % batch_size)
    if usable_total <= 0:
        raise ValueError("Not enough examples to form a single batch.")
    cache_key = (total_examples, batch_size, steps, seed if seed is not None else 0, drop_last)
    cached = _BATCH_SCHEDULE_CACHE.get(cache_key)
    if cached is not None:
        return [batch.clone() for batch in cached]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    schedule: list[torch.Tensor] = []
    epoch_order = torch.randperm(total_examples, generator=generator)
    epoch_cursor = 0
    while len(schedule) < steps:
        if drop_last and epoch_cursor + batch_size > usable_total:
            epoch_order = torch.randperm(total_examples, generator=generator)
            epoch_cursor = 0
        elif not drop_last and epoch_cursor >= total_examples:
            epoch_order = torch.randperm(total_examples, generator=generator)
            epoch_cursor = 0
        batch_indices = epoch_order[epoch_cursor : epoch_cursor + batch_size]
        if drop_last and batch_indices.numel() < batch_size:
            epoch_order = torch.randperm(total_examples, generator=generator)
            epoch_cursor = 0
            continue
        schedule.append(batch_indices.clone())
        epoch_cursor += batch_size
    _BATCH_SCHEDULE_CACHE[cache_key] = [batch.clone() for batch in schedule]
    return schedule


def _scheduled_batch_from_tensors(
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    device: torch.device,
    non_blocking: bool,
) -> dict[str, torch.Tensor]:
    if batch_indices.device != input_ids.device:
        batch_indices = batch_indices.to(input_ids.device)
    batch_input_ids = input_ids.index_select(0, batch_indices)
    batch_targets = targets.index_select(0, batch_indices)
    if batch_input_ids.device != device:
        batch_input_ids = batch_input_ids.to(device, non_blocking=non_blocking)
        batch_targets = batch_targets.to(device, non_blocking=non_blocking)
    return {
        "input_ids": batch_input_ids,
        "targets": batch_targets,
    }


def _tensor_train_iterator(
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    non_blocking: bool,
):
    while True:
        yield from _iter_tensor_batches(
            input_ids,
            targets,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            device=device,
            non_blocking=non_blocking,
        )


def _loss_and_tokens(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_targets)
    return loss, int(flat_targets.numel())


def _build_optimizer(
    model: nn.Module,
    config: RealTextConfig,
    *,
    model_name: str | None = None,
) -> torch.optim.Optimizer:
    def _transformer_like() -> bool:
        if model_name is None:
            return False
        lowered = model_name.lower()
        return "gpt" in lowered or "nano" in lowered or "transformer" in lowered

    optimizer_kwargs = {
        "lr": config.learning_rate,
    }
    if config.optimizer_recipe == "transformer_fair" and _transformer_like():
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        no_decay_tokens = ("bias", "norm", "embedding", "wte", "lm_head", "value_embeds", "resid_lambdas", "x0_lambdas")
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(token in name.lower() for token in no_decay_tokens):
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)
        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": config.weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    else:
        param_groups = model.parameters()
        optimizer_kwargs["weight_decay"] = config.weight_decay
    if config.use_fused_adamw and config.device == "cuda":
        try:
            return torch.optim.AdamW(param_groups, fused=True, **optimizer_kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(param_groups, **optimizer_kwargs)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: RealTextConfig,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if config.warmup_steps <= 0 and config.lr_schedule == "none":
        return None

    def _lr_lambda(step: int) -> float:
        warmup_steps = max(config.warmup_steps, 0)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if config.lr_schedule == "cosine":
            denom = max(config.train_steps - warmup_steps, 1)
            progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return config.min_lr_scale + (1.0 - config.min_lr_scale) * cosine
        if config.lr_schedule == "linear":
            denom = max(config.train_steps - warmup_steps, 1)
            progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
            return config.min_lr_scale + (1.0 - config.min_lr_scale) * (1.0 - progress)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def evaluate_loss(
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


def train_microbenchmark(
    model: nn.Module,
    train_dataset: TokenBlockDataset,
    val_dataset: TokenBlockDataset,
    *,
    config: RealTextConfig,
    model_name: str | None = None,
    batch_schedule: list[torch.Tensor] | None = None,
) -> dict[str, object]:
    device = torch.device(config.device)
    model.to(device)
    optimizer = _build_optimizer(model, config, model_name=model_name)
    scheduler = _build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=config.use_amp and device.type == "cuda")
    use_amp = config.use_amp and device.type == "cuda"
    val_source: object
    auto_cache_on_device = (
        config.auto_cache_dataset_on_device
        and not config.cache_dataset_on_device
        and config.tensor_batching
        and device.type == "cuda"
    )
    if auto_cache_on_device:
        total_tensor_elements = (
            train_dataset.input_ids.numel()
            + train_dataset.targets.numel()
            + val_dataset.input_ids.numel()
            + val_dataset.targets.numel()
        )
        auto_cache_on_device = total_tensor_elements <= 8_000_000
    cache_on_device = config.cache_dataset_on_device or auto_cache_on_device
    train_source = _dataset_tensors(
        train_dataset,
        device=device,
        cache_on_device=cache_on_device,
        pin_memory=config.pin_memory,
    )
    if config.tensor_batching:
        val_source = _dataset_tensors(
            val_dataset,
            device=device,
            cache_on_device=cache_on_device,
            pin_memory=config.pin_memory,
        )
    else:
        pin_memory = config.pin_memory and device.type == "cuda"
        val_source = DataLoader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=0,
            persistent_workers=False,
        )
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if batch_schedule is None:
        batch_schedule = _build_train_batch_schedule(
            len(train_dataset),
            batch_size=config.batch_size,
            steps=config.train_steps,
            seed=config.seed if config.train_schedule_seed is None else config.train_schedule_seed,
            drop_last=True,
        )

    history: list[dict[str, float]] = []
    tokens_seen = 0
    sequences_seen = 0

    initial_val_loss = float("nan")
    if config.initial_eval:
        initial_val_loss = evaluate_loss(model, val_source, device=device, use_amp=use_amp, config=config)
        history.append(
            {
                "step": 0.0,
                "sequences_seen": 0.0,
                "tokens_seen": 0.0,
                "train_loss": float("nan"),
                "val_loss": initial_val_loss,
            }
        )

    start = time.perf_counter()
    for step, batch_indices in enumerate(batch_schedule, start=1):
        batch = _scheduled_batch_from_tensors(
            train_source[0],
            train_source[1],
            batch_indices,
            device=device,
            non_blocking=config.pin_memory and device.type == "cuda",
        )
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
        if scheduler is not None:
            scheduler.step()

        tokens_seen += token_count
        sequences_seen += batch["input_ids"].size(0)

        if step % config.eval_interval == 0 or step == config.train_steps:
            val_loss = evaluate_loss(model, val_source, device=device, use_amp=use_amp, config=config)
            history.append(
                {
                    "step": float(step),
                    "sequences_seen": float(sequences_seen),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": float(loss.item()),
                    "val_loss": val_loss,
                }
            )

    return {
        "parameter_count": count_parameters(model),
        "history": history,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": history[-1]["val_loss"],
        "training_runtime_seconds": time.perf_counter() - start,
        "paired_train_batches": bool(config.paired_train_batches),
        "reseed_per_model": bool(config.reseed_per_model),
    }


def fairness_summary(model_reports: dict[str, dict[str, object]]) -> dict[str, object]:
    recurrent_params = float(model_reports["associative_recurrent"]["parameter_count"])
    gpt_params = float(model_reports["gpt2_like"]["parameter_count"])
    gap = abs(recurrent_params - gpt_params) / max(recurrent_params, gpt_params, 1.0)
    return {
        "associative_recurrent_parameter_count": int(round(recurrent_params)),
        "gpt2_like_parameter_count": int(round(gpt_params)),
        "relative_parameter_gap": gap,
        "parameter_gap_ok": gap <= 0.15,
        "same_dataset": True,
        "same_tokenizer": True,
        "same_optimizer": "AdamW",
        "same_training_tokens": True,
        "same_training_examples_order": True,
        "paired_batch_schedule": True,
        "epoch_reshuffle": True,
    }


def run_realtext_microbenchmark(config: RealTextConfig) -> dict[str, object]:
    set_global_seed(config.seed)
    if config.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    start = time.perf_counter()
    data_start = time.perf_counter()
    train_dataset, val_dataset, vocab_size = load_realtext_datasets(config)
    dataset_loading_runtime_seconds = time.perf_counter() - data_start
    reports: dict[str, dict[str, object]] = {}
    schedule_seed = config.seed if config.train_schedule_seed is None else config.train_schedule_seed
    batch_schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=config.batch_size,
        steps=config.train_steps,
        seed=schedule_seed,
        drop_last=True,
    )
    for name, builder in build_model_builders(config, vocab_size=vocab_size).items():
        if config.reseed_per_model:
            set_global_seed(config.seed)
        model = builder()
        reports[name] = train_microbenchmark(
            model,
            train_dataset,
            val_dataset,
            config=config,
            model_name=name,
            batch_schedule=batch_schedule,
        )
    runtime_seconds = time.perf_counter() - start
    winner = min(reports.items(), key=lambda item: item[1]["final_val_loss"])[0]
    return {
        "benchmark": "language_realtext_microbench",
        "config": asdict(config),
        "dataset": {
            "name": config.dataset_name,
            "config": config.dataset_config,
            "tokenizer_name": config.tokenizer_name,
            "train_sequences": len(train_dataset),
            "validation_sequences": len(val_dataset),
            "sequence_length": config.sequence_length,
        },
        "fairness": fairness_summary(reports),
        "models": reports,
        "winner_by_final_val_loss": winner,
        "dataset_loading_runtime_seconds": dataset_loading_runtime_seconds,
        "runtime_seconds": runtime_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-text micro-pretraining benchmark for recurrent vs GPT-style LMs.")
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--validation-split", type=str, default="validation")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--train-token-cap", type=int, default=None)
    parser.add_argument("--validation-token-cap", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--max-train-sequences", type=int, default=2048)
    parser.add_argument("--max-eval-sequences", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=16)
    parser.add_argument("--initial-eval", dest="initial_eval", action="store_true")
    parser.add_argument("--no-initial-eval", dest="initial_eval", action="store_false")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--recurrent-embedding-dim", type=int, default=64)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=128)
    parser.add_argument("--recurrent-memory-dim", type=int, default=64)
    parser.add_argument("--gpt-d-model", type=int, default=64)
    parser.add_argument("--gpt-heads", type=int, default=4)
    parser.add_argument("--gpt-layers", type=int, default=2)
    parser.add_argument("--gpt-ff-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tensor-batching", dest="tensor_batching", action="store_true")
    parser.add_argument("--no-tensor-batching", dest="tensor_batching", action="store_false")
    parser.add_argument("--cache-dataset-on-device", dest="cache_dataset_on_device", action="store_true")
    parser.add_argument("--no-cache-dataset-on-device", dest="cache_dataset_on_device", action="store_false")
    parser.add_argument("--auto-cache-dataset-on-device", dest="auto_cache_dataset_on_device", action="store_true")
    parser.add_argument("--no-auto-cache-dataset-on-device", dest="auto_cache_dataset_on_device", action="store_false")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--dataset-cache-path", type=Path, default=None)
    parser.add_argument("--use-fused-adamw", dest="use_fused_adamw", action="store_true")
    parser.add_argument("--no-use-fused-adamw", dest="use_fused_adamw", action="store_false")
    parser.add_argument("--enable-tf32", dest="enable_tf32", action="store_true")
    parser.add_argument("--no-enable-tf32", dest="enable_tf32", action="store_false")
    parser.add_argument("--local-files-only", dest="local_files_only", action="store_true")
    parser.add_argument("--no-local-files-only", dest="local_files_only", action="store_false")
    parser.add_argument("--tokenization-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(
        tensor_batching=RealTextConfig.tensor_batching,
        cache_dataset_on_device=RealTextConfig.cache_dataset_on_device,
        auto_cache_dataset_on_device=RealTextConfig.auto_cache_dataset_on_device,
        pin_memory=RealTextConfig.pin_memory,
        use_fused_adamw=RealTextConfig.use_fused_adamw,
        enable_tf32=RealTextConfig.enable_tf32,
        local_files_only=RealTextConfig.local_files_only,
        initial_eval=RealTextConfig.initial_eval,
    )
    args = parser.parse_args()

    config = RealTextConfig(
        seed=args.seed,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        validation_split=args.validation_split,
        tokenizer_name=args.tokenizer_name,
        streaming=args.streaming,
        train_token_cap=args.train_token_cap,
        validation_token_cap=args.validation_token_cap,
        sequence_length=args.sequence_length,
        max_train_sequences=args.max_train_sequences,
        max_eval_sequences=args.max_eval_sequences,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        initial_eval=args.initial_eval,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        recurrent_embedding_dim=args.recurrent_embedding_dim,
        recurrent_hidden_dim=args.recurrent_hidden_dim,
        recurrent_memory_dim=args.recurrent_memory_dim,
        gpt_d_model=args.gpt_d_model,
        gpt_heads=args.gpt_heads,
        gpt_layers=args.gpt_layers,
        gpt_ff_dim=args.gpt_ff_dim,
        device=args.device,
        tensor_batching=args.tensor_batching,
        cache_dataset_on_device=args.cache_dataset_on_device,
        auto_cache_dataset_on_device=args.auto_cache_dataset_on_device,
        pin_memory=args.pin_memory,
        dataset_cache_path=args.dataset_cache_path,
        use_fused_adamw=args.use_fused_adamw,
        enable_tf32=args.enable_tf32,
        local_files_only=args.local_files_only,
        tokenization_batch_size=args.tokenization_batch_size,
    )
    payload = run_realtext_microbenchmark(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
