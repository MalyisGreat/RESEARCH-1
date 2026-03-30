from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_tactic3.language_fastlearn_benchmark import count_parameters
from arc_tactic3.language_nanochat_actual_compare import _peak_vram_mb
from arc_tactic3.language_realtext_microbench import (
    RealTextConfig,
    TokenBlockDataset,
    _build_optimizer,
    _build_scheduler,
    _build_train_batch_schedule,
    _dataset_tensors,
    _loss_and_tokens,
    _scheduled_batch_from_tensors,
    set_global_seed,
)
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM, _top_token_ids


@dataclass(frozen=True, slots=True)
class StreamingCompareConfig:
    cache_path: Path
    output_path: Path
    target_tokens: int = 5_000_000
    cheap_target_tokens: int = 1_000_000
    seed: int = 13
    sequence_length: int = 127
    batch_size: int = 16
    eval_batch_size: int = 16
    eval_steps: int = 64
    eval_interval: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    cache_dataset_on_device: bool = True
    embedding_dim: int = 144
    hidden_dim: int = 288
    memory_dim: int = 144
    partial_token_count: int = 512
    dropout: float = 0.1
    memory_segments: int = 4
    run_hold: bool = True


@dataclass(frozen=True, slots=True)
class VariantSpec:
    name: str
    streamed: bool
    carry_hidden: bool
    carry_memory: bool


@dataclass
class StreamState:
    hidden: torch.Tensor | None = None
    past_keys: torch.Tensor | None = None
    past_tokens: torch.Tensor | None = None


class StreamingPartialUntiedAssociativeLM(nn.Module):
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
        memory_segments: int,
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
        self.memory_capacity = max_length * memory_segments
        self.register_buffer("untied_token_ids", untied_token_ids.long(), persistent=False)
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=-1).unsqueeze(0),
            persistent=False,
        )

    def _append_memory(
        self,
        *,
        past_keys: torch.Tensor | None,
        past_tokens: torch.Tensor | None,
        current_keys: torch.Tensor,
        current_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_keys = current_keys.detach()
        current_tokens = current_tokens.detach()
        if past_keys is None or past_tokens is None:
            next_keys = current_keys
            next_tokens = current_tokens
        else:
            next_keys = torch.cat([past_keys, current_keys], dim=1)
            next_tokens = torch.cat([past_tokens, current_tokens], dim=1)
        if next_keys.size(1) > self.memory_capacity:
            next_keys = next_keys[:, -self.memory_capacity :].contiguous()
            next_tokens = next_tokens[:, -self.memory_capacity :].contiguous()
        return next_keys, next_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        prev_hidden: torch.Tensor | None = None,
        past_keys: torch.Tensor | None = None,
        past_tokens: torch.Tensor | None = None,
        carry_hidden: bool = False,
        carry_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        embeddings = self.embedding(input_ids)
        states, next_hidden = self.encoder(embeddings, prev_hidden if carry_hidden else None)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        logits = F.linear(base_features, self.embedding.weight, self.output_bias)

        partial_logits = self.partial_head(base_features)
        partial_index = self.untied_token_ids.view(1, 1, -1).expand(input_ids.size(0), input_ids.size(1), -1)
        logits.scatter_add_(2, partial_index, partial_logits)

        query_keys = self.query_proj(states)
        current_keys = self.key_proj(states)
        current_tokens = input_ids

        current_scores = torch.matmul(query_keys, current_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        current_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        masked_current_scores = current_scores.masked_fill(~current_mask, torch.finfo(current_scores.dtype).min)

        attention_parts = []
        value_parts = []
        mask_parts = []

        if carry_memory and past_keys is not None and past_tokens is not None and past_keys.size(1) > 0:
            past_scores = torch.matmul(query_keys, past_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
            attention_parts.append(past_scores)
            value_parts.append(past_tokens.unsqueeze(1).expand(-1, input_ids.size(1), -1))
            mask_parts.append(torch.ones_like(past_scores, dtype=torch.bool))

        attention_parts.append(masked_current_scores)
        value_parts.append(current_tokens.unsqueeze(1).expand(-1, input_ids.size(1), -1))
        mask_parts.append(current_mask.expand(input_ids.size(0), -1, -1))

        all_scores = torch.cat(attention_parts, dim=-1)
        all_mask = torch.cat(mask_parts, dim=-1)
        all_attention = torch.softmax(all_scores, dim=-1)
        all_attention = all_attention * all_mask
        all_attention = all_attention / all_attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        all_value_index = torch.cat(value_parts, dim=-1)
        gate = torch.sigmoid(self.gate(states))
        gated_attention = (all_attention * (gate * self.memory_scale)).to(logits.dtype)
        logits.scatter_add_(2, all_value_index, gated_attention)

        next_memory_keys, next_memory_tokens = past_keys, past_tokens
        if carry_memory:
            next_memory_keys, next_memory_tokens = self._append_memory(
                past_keys=past_keys,
                past_tokens=past_tokens,
                current_keys=current_keys,
                current_tokens=current_tokens,
            )
        return (
            logits,
            next_hidden.detach() if carry_hidden else None,
            next_memory_keys,
            next_memory_tokens,
        )


class ContiguousTokenBatcher:
    def __init__(
        self,
        *,
        tokens: torch.Tensor,
        sequence_length: int,
        batch_size: int,
        steps: int,
        seed: int,
    ) -> None:
        self.tokens = tokens
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.steps = steps
        self.seed = seed
        self._initial_positions = self._build_initial_positions()
        self.reset()

    def _build_initial_positions(self) -> torch.Tensor:
        needed = self.steps * self.sequence_length + self.sequence_length + 1
        max_start = max(self.tokens.numel() - needed - 1, 0)
        if self.batch_size == 1:
            base = torch.zeros(1, dtype=torch.long)
        else:
            base = torch.linspace(0, max_start, steps=self.batch_size, dtype=torch.long)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed)
        jitter = torch.randint(0, max(max_start // max(self.batch_size, 1), 1), (self.batch_size,), generator=rng)
        starts = torch.clamp(base + jitter, max=max_start)
        return starts.long()

    def reset(self) -> None:
        self.positions = self._initial_positions.clone()
        self.step = 0

    def next_batch(self, *, device: torch.device) -> dict[str, torch.Tensor]:
        if self.step >= self.steps:
            raise StopIteration
        slices = []
        targets = []
        for lane in range(self.batch_size):
            pos = int(self.positions[lane].item())
            segment = self.tokens[pos : pos + self.sequence_length + 1]
            if segment.numel() != self.sequence_length + 1:
                raise RuntimeError("ContiguousTokenBatcher ran past the available token stream.")
            slices.append(segment[:-1])
            targets.append(segment[1:])
            self.positions[lane] += self.sequence_length
        self.step += 1
        input_ids = torch.stack(slices).to(device=device, non_blocking=True)
        target_ids = torch.stack(targets).to(device=device, non_blocking=True)
        return {"input_ids": input_ids, "targets": target_ids}


def _load_token_streams(config: StreamingCompareConfig) -> tuple[torch.Tensor, torch.Tensor, int]:
    payload = torch.load(config.cache_path, map_location="cpu", weights_only=False)
    return payload["train_tokens"].long(), payload["val_tokens"].long(), int(payload["vocab_size"])


def _block_train_dataset(config: StreamingCompareConfig, train_tokens: torch.Tensor) -> TokenBlockDataset:
    block_size = config.sequence_length + 1
    train_blocks = train_tokens.view(-1, block_size)
    return TokenBlockDataset(train_blocks[:, :-1].contiguous(), train_blocks[:, 1:].contiguous())


def _make_realtext_config(config: StreamingCompareConfig, *, train_steps: int) -> RealTextConfig:
    return RealTextConfig(
        seed=config.seed,
        sequence_length=config.sequence_length,
        train_steps=train_steps,
        eval_interval=config.eval_interval,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        use_amp=config.use_amp,
        pin_memory=config.pin_memory,
        use_fused_adamw=config.use_fused_adamw,
        tensor_batching=True,
        cache_dataset_on_device=config.cache_dataset_on_device,
        paired_train_batches=True,
        reseed_per_model=True,
        train_schedule_seed=config.seed,
        dropout=config.dropout,
        initial_eval=False,
    )


def _evaluate_stream_loss(
    model: StreamingPartialUntiedAssociativeLM,
    *,
    batcher: ContiguousTokenBatcher,
    device: torch.device,
    use_amp: bool,
    carry_hidden: bool,
    carry_memory: bool,
) -> float:
    model.eval()
    batcher.reset()
    state = StreamState()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(batcher.steps):
            batch = batcher.next_batch(device=device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, next_hidden, next_keys, next_tokens = model(
                    batch["input_ids"],
                    prev_hidden=state.hidden,
                    past_keys=state.past_keys,
                    past_tokens=state.past_tokens,
                    carry_hidden=carry_hidden,
                    carry_memory=carry_memory,
                )
                loss, token_count = _loss_and_tokens(logits, batch["targets"])
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count
            state.hidden = next_hidden
            state.past_keys = next_keys
            state.past_tokens = next_tokens
    return total_loss / max(total_tokens, 1)


def _train_streaming_variant(
    model: StreamingPartialUntiedAssociativeLM,
    *,
    spec: VariantSpec,
    config: StreamingCompareConfig,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
) -> dict[str, Any]:
    device = torch.device(config.device)
    real_config = _make_realtext_config(
        config,
        train_steps=math.ceil(config.target_tokens / (config.sequence_length * config.batch_size)),
    )
    model.to(device)
    optimizer = _build_optimizer(model, real_config, model_name="partial_untied")
    scheduler = _build_scheduler(optimizer, real_config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=real_config.use_amp and device.type == "cuda")
    use_amp = real_config.use_amp and device.type == "cuda"
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]

    steps = real_config.train_steps
    tokens_per_step = config.sequence_length * config.batch_size
    actual_target_tokens = steps * tokens_per_step
    variant_seed_offsets = {
        "stream_reset": 101,
        "stream_carry_hidden": 211,
        "stream_carry_hidden_memory": 307,
    }
    stream_seed = config.seed + variant_seed_offsets.get(spec.name, 401)
    train_stream = ContiguousTokenBatcher(
        tokens=train_tokens,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        steps=steps,
        seed=stream_seed,
    )
    val_stream = ContiguousTokenBatcher(
        tokens=val_tokens,
        sequence_length=config.sequence_length,
        batch_size=config.eval_batch_size,
        steps=config.eval_steps,
        seed=stream_seed + 1,
    )

    history: list[dict[str, float]] = []
    step_times: list[float] = []
    tokens_seen = 0
    state = StreamState()
    latest_val_loss = _evaluate_stream_loss(
        model,
        batcher=val_stream,
        device=device,
        use_amp=use_amp,
        carry_hidden=spec.carry_hidden,
        carry_memory=spec.carry_memory,
    )
    history.append(
        {
            "step": 0.0,
            "tokens_seen": 0.0,
            "train_loss": float("nan"),
            "val_loss": float(latest_val_loss),
        }
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for step in range(1, steps + 1):
        batch = train_stream.next_batch(device=device)
        step_start = time.perf_counter()
        model.train()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, next_hidden, next_keys, next_tokens = model(
                batch["input_ids"],
                prev_hidden=state.hidden,
                past_keys=state.past_keys,
                past_tokens=state.past_tokens,
                carry_hidden=spec.carry_hidden,
                carry_memory=spec.carry_memory,
            )
            loss, token_count = _loss_and_tokens(logits, batch["targets"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameter_list, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_start)
        tokens_seen += token_count
        state.hidden = next_hidden
        state.past_keys = next_keys
        state.past_tokens = next_tokens

        if step % config.eval_interval == 0 or step == steps:
            latest_val_loss = _evaluate_stream_loss(
                model,
                batcher=val_stream,
                device=device,
                use_amp=use_amp,
                carry_hidden=spec.carry_hidden,
                carry_memory=spec.carry_memory,
            )
            history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": float(loss.item()),
                    "val_loss": float(latest_val_loss),
                }
            )
            elapsed = time.perf_counter() - start
            train_tok_per_sec = tokens_seen / max(elapsed, 1e-9)
            eta_seconds = max(actual_target_tokens - tokens_seen, 0) / max(train_tok_per_sec, 1e-9)
            print(
                f"{spec.name} step={step:,}/{steps:,} tok={tokens_seen:,}/{actual_target_tokens:,} "
                f"train={loss.item():.4f} val={latest_val_loss:.4f} tok/s={train_tok_per_sec:,.0f} "
                f"eta={eta_seconds/60:.1f}m",
                flush=True,
            )

    total_time = time.perf_counter() - start
    pure_train_time = sum(step_times)
    return {
        "parameter_count": count_parameters(model),
        "history": history,
        "final_val_loss": latest_val_loss,
        "train_tokens_seen": tokens_seen,
        "train_tok_per_sec": tokens_seen / max(total_time, 1e-9),
        "pure_train_tok_per_sec": tokens_seen / max(pure_train_time, 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "peak_vram_mb": _peak_vram_mb(config.device),
        "total_training_time_seconds": total_time,
        "carry_hidden": spec.carry_hidden,
        "carry_memory": spec.carry_memory,
        "streamed": spec.streamed,
    }


def _train_block_reset_variant(
    *,
    config: StreamingCompareConfig,
    train_dataset: TokenBlockDataset,
    val_tokens: torch.Tensor,
    vocab_size: int,
    partial_token_ids: torch.Tensor,
) -> dict[str, Any]:
    device = torch.device(config.device)
    steps = math.ceil(config.target_tokens / (config.sequence_length * config.batch_size))
    real_config = _make_realtext_config(config, train_steps=steps)
    model = PartialUntiedAssociativeLM(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        memory_dim=config.memory_dim,
        dropout=config.dropout,
        max_length=config.sequence_length,
        untied_token_ids=partial_token_ids,
    )
    model.to(device)
    optimizer = _build_optimizer(model, real_config, model_name="partial_untied")
    scheduler = _build_scheduler(optimizer, real_config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=real_config.use_amp and device.type == "cuda")
    use_amp = real_config.use_amp and device.type == "cuda"
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]

    train_source = _dataset_tensors(
        train_dataset,
        device=device,
        cache_on_device=real_config.cache_dataset_on_device,
        pin_memory=real_config.pin_memory,
    )
    batch_schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=real_config.batch_size,
        steps=real_config.train_steps,
        seed=config.seed,
        drop_last=True,
    )
    val_stream = ContiguousTokenBatcher(
        tokens=val_tokens,
        sequence_length=config.sequence_length,
        batch_size=config.eval_batch_size,
        steps=config.eval_steps,
        seed=config.seed + 11,
    )
    history: list[dict[str, float]] = []
    step_times: list[float] = []
    tokens_seen = 0
    eval_model = StreamingPartialUntiedAssociativeLM(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        memory_dim=config.memory_dim,
        dropout=config.dropout,
        max_length=config.sequence_length,
        untied_token_ids=partial_token_ids,
        memory_segments=config.memory_segments,
    ).to(device)
    eval_model.load_state_dict(model.state_dict(), strict=False)
    latest_val_loss = _evaluate_stream_loss(
        eval_model,
        batcher=val_stream,
        device=device,
        use_amp=use_amp,
        carry_hidden=False,
        carry_memory=False,
    )
    history.append({"step": 0.0, "tokens_seen": 0.0, "train_loss": float("nan"), "val_loss": float(latest_val_loss)})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    actual_target_tokens = steps * config.sequence_length * config.batch_size

    for step, batch_indices in enumerate(batch_schedule, start=1):
        batch = _scheduled_batch_from_tensors(
            train_source[0],
            train_source[1],
            batch_indices,
            device=device,
            non_blocking=real_config.pin_memory and device.type == "cuda",
        )
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
        if scheduler is not None:
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_start)
        tokens_seen += token_count

        if step % config.eval_interval == 0 or step == steps:
            eval_model.load_state_dict(model.state_dict(), strict=False)
            latest_val_loss = _evaluate_stream_loss(
                eval_model,
                batcher=val_stream,
                device=device,
                use_amp=use_amp,
                carry_hidden=False,
                carry_memory=False,
            )
            history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": float(loss.item()),
                    "val_loss": float(latest_val_loss),
                }
            )
            elapsed = time.perf_counter() - start
            train_tok_per_sec = tokens_seen / max(elapsed, 1e-9)
            eta_seconds = max(actual_target_tokens - tokens_seen, 0) / max(train_tok_per_sec, 1e-9)
            print(
                f"block_reset step={step:,}/{steps:,} tok={tokens_seen:,}/{actual_target_tokens:,} "
                f"train={loss.item():.4f} val={latest_val_loss:.4f} tok/s={train_tok_per_sec:,.0f} "
                f"eta={eta_seconds/60:.1f}m",
                flush=True,
            )

    total_time = time.perf_counter() - start
    pure_train_time = sum(step_times)
    return {
        "parameter_count": count_parameters(model),
        "history": history,
        "final_val_loss": latest_val_loss,
        "train_tokens_seen": tokens_seen,
        "train_tok_per_sec": tokens_seen / max(total_time, 1e-9),
        "pure_train_tok_per_sec": tokens_seen / max(pure_train_time, 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "peak_vram_mb": _peak_vram_mb(config.device),
        "total_training_time_seconds": total_time,
        "carry_hidden": False,
        "carry_memory": False,
        "streamed": False,
    }


def _run_suite(config: StreamingCompareConfig, *, target_tokens: int) -> dict[str, Any]:
    run_config = StreamingCompareConfig(**{**asdict(config), "target_tokens": target_tokens})
    set_global_seed(run_config.seed)
    train_tokens, val_tokens, vocab_size = _load_token_streams(run_config)
    train_dataset = _block_train_dataset(run_config, train_tokens)
    partial_token_ids = _top_token_ids(train_dataset, count=run_config.partial_token_count, vocab_size=vocab_size)

    results: dict[str, Any] = {}
    results["block_reset"] = _train_block_reset_variant(
        config=run_config,
        train_dataset=train_dataset,
        val_tokens=val_tokens,
        vocab_size=vocab_size,
        partial_token_ids=partial_token_ids,
    )

    variants = [
        VariantSpec(name="stream_reset", streamed=True, carry_hidden=False, carry_memory=False),
        VariantSpec(name="stream_carry_hidden", streamed=True, carry_hidden=True, carry_memory=False),
        VariantSpec(name="stream_carry_hidden_memory", streamed=True, carry_hidden=True, carry_memory=True),
    ]
    for spec in variants:
        set_global_seed(run_config.seed)
        model = StreamingPartialUntiedAssociativeLM(
            vocab_size=vocab_size,
            embedding_dim=run_config.embedding_dim,
            hidden_dim=run_config.hidden_dim,
            memory_dim=run_config.memory_dim,
            dropout=run_config.dropout,
            max_length=run_config.sequence_length,
            untied_token_ids=partial_token_ids,
            memory_segments=run_config.memory_segments,
        )
        results[spec.name] = _train_streaming_variant(
            model,
            spec=spec,
            config=run_config,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
        )
    return {
        "benchmark": "language_partial_untied_streaming_compare",
        "config": {
            **asdict(run_config),
            "cache_path": str(run_config.cache_path),
            "output_path": str(run_config.output_path),
        },
        "results": results,
    }


def run_streaming_compare(config: StreamingCompareConfig) -> dict[str, Any]:
    payload = {"cheap": _run_suite(config, target_tokens=config.cheap_target_tokens)}
    if config.run_hold:
        payload["hold"] = _run_suite(config, target_tokens=config.target_tokens)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare block-reset partial_untied against streamed carried-state variants.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=5_000_000)
    parser.add_argument("--cheap-target-tokens", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-hold", action="store_true")
    args = parser.parse_args()

    config = StreamingCompareConfig(
        cache_path=args.cache_path,
        output_path=args.output_path,
        target_tokens=args.target_tokens,
        cheap_target_tokens=args.cheap_target_tokens,
        seed=args.seed,
        device=args.device,
        run_hold=not args.no_hold,
    )
    payload = run_streaming_compare(config)
    text = json.dumps(payload, indent=2)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
