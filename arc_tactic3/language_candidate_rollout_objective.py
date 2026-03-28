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
import torch.nn.functional as F
from torch import nn

from arc_tactic3.language_fastlearn_benchmark import count_parameters, set_global_seed
from arc_tactic3.language_nanochat_actual_compare import ReLU2HeadAssociativeLM, _load_cached_datasets
from arc_tactic3.language_realtext_microbench import (
    RealTextConfig,
    TokenBlockDataset,
    _build_optimizer,
    _build_scheduler,
    _build_train_batch_schedule,
    _dataset_tensors,
    _loss_and_tokens,
    _scheduled_batch_from_tensors,
)
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM, _top_token_ids


OBJECTIVE_MODES = (
    "ce_only",
    "ce_plus_sequence",
    "ce_plus_rollout",
    "ce_plus_both",
)


@dataclass(frozen=True, slots=True)
class RolloutObjectiveConfig:
    cache_path: Path
    model_variant: str = "partial_untied"
    tokenizer_name: str = "gpt2"
    train_blocks: int = 2048
    val_blocks: int = 128
    sequence_length: int = 127
    batch_size: int = 16
    eval_batch_size: int = 32
    train_steps: int = 64
    eval_interval: int = 16
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 13
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    cache_dataset_on_device: bool = False
    recurrent_embedding_dim: int = 144
    recurrent_hidden_dim: int = 288
    recurrent_memory_dim: int = 144
    dropout: float = 0.1
    partial_untied_tokens: int = 512
    rollout_prefix_length: int = 8
    rollout_loss_weight: float = 0.5
    sequence_focus_weight: float = 0.25
    sequence_focus_temperature: float = 1.5
    rollout_teacher_force_prob: float = 0.0
    paired_train_batches: bool = True
    reseed_per_objective: bool = True
    train_schedule_seed: int | None = None
    optimizer_recipe: str = "default"
    warmup_steps: int = 0
    lr_schedule: str = "none"
    min_lr_scale: float = 1.0


def _shared_realtext_config(config: RolloutObjectiveConfig) -> RealTextConfig:
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
        cache_dataset_on_device=config.cache_dataset_on_device,
        paired_train_batches=config.paired_train_batches,
        reseed_per_model=config.reseed_per_objective,
        train_schedule_seed=config.train_schedule_seed,
        optimizer_recipe=config.optimizer_recipe,
        warmup_steps=config.warmup_steps,
        lr_schedule=config.lr_schedule,
        min_lr_scale=config.min_lr_scale,
    )


def _build_model(
    config: RolloutObjectiveConfig,
    *,
    vocab_size: int,
    partial_token_ids: torch.Tensor,
) -> nn.Module:
    common = {
        "vocab_size": vocab_size,
        "embedding_dim": config.recurrent_embedding_dim,
        "hidden_dim": config.recurrent_hidden_dim,
        "memory_dim": config.recurrent_memory_dim,
        "dropout": config.dropout,
        "max_length": config.sequence_length,
    }
    if config.model_variant == "recurrent_champion":
        return ReLU2HeadAssociativeLM(**common)
    if config.model_variant == "partial_untied":
        return PartialUntiedAssociativeLM(untied_token_ids=partial_token_ids, **common)
    raise ValueError(f"Unknown model_variant: {config.model_variant}")


def _iter_eval_batches(
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    non_blocking: bool,
):
    total = input_ids.size(0)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_indices = torch.arange(start, end, dtype=torch.long, device=input_ids.device)
        yield _scheduled_batch_from_tensors(
            input_ids,
            targets,
            batch_indices,
            device=device,
            non_blocking=non_blocking,
        )


def _sequence_focus_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    token_nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    scaled = token_nll / max(temperature, 1e-4)
    scaled = scaled.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
    weights = torch.softmax(scaled, dim=-1)
    weights = weights * mask
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    focused = (weights * token_nll).sum(dim=-1)
    active = (mask.sum(dim=-1) > 0).float()
    return (focused * active).sum() / active.sum().clamp_min(1.0)


def _sequence_exact_hits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_hits = (logits.argmax(dim=-1) == targets).float() * mask
    return (token_hits.sum(dim=-1) == mask.sum(dim=-1)).float()


def _rollout_continuation(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    prefix_length: int,
    teacher_force_prob: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2 or targets.ndim != 2:
        raise ValueError("input_ids and targets must both be rank-2 tensors.")
    if input_ids.shape != targets.shape:
        raise ValueError("input_ids and targets must have the same shape.")
    prefix_length = max(1, min(prefix_length, input_ids.size(1)))
    target_start = prefix_length - 1
    rollout_ids = input_ids[:, :prefix_length].clone()
    step_logits: list[torch.Tensor] = []
    for target_index in range(target_start, targets.size(1)):
        logits = model(rollout_ids)
        next_logits = logits[:, -1, :]
        step_logits.append(next_logits)
        if target_index + 1 >= input_ids.size(1):
            continue
        next_tokens = next_logits.argmax(dim=-1)
        if teacher_force_prob > 0.0:
            teacher_tokens = targets[:, target_index]
            teacher_mask = torch.rand(
                next_tokens.shape,
                device=next_tokens.device,
            ) < teacher_force_prob
            next_tokens = torch.where(teacher_mask, teacher_tokens, next_tokens)
        rollout_ids = torch.cat((rollout_ids, next_tokens.unsqueeze(1)), dim=1)
    rollout_logits = torch.stack(step_logits, dim=1)
    rollout_targets = targets[:, target_start:]
    rollout_mask = torch.ones_like(rollout_targets, dtype=rollout_logits.dtype)
    return rollout_logits, rollout_targets, rollout_mask


def _objective_loss(
    mode: str,
    *,
    ce_loss: torch.Tensor,
    seq_focus_loss: torch.Tensor,
    rollout_loss: torch.Tensor,
    rollout_seq_focus_loss: torch.Tensor,
    config: RolloutObjectiveConfig,
) -> torch.Tensor:
    if mode == "ce_only":
        return ce_loss
    if mode == "ce_plus_sequence":
        return ce_loss + config.sequence_focus_weight * seq_focus_loss
    if mode == "ce_plus_rollout":
        return ce_loss + config.rollout_loss_weight * rollout_loss
    if mode == "ce_plus_both":
        return (
            ce_loss
            + config.sequence_focus_weight * seq_focus_loss
            + config.rollout_loss_weight * (rollout_loss + config.sequence_focus_weight * rollout_seq_focus_loss)
        )
    raise ValueError(f"Unknown objective mode: {mode}")


def _evaluate_objective(
    model: nn.Module,
    val_source: tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    config: RolloutObjectiveConfig,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    token_total = 0
    sequence_hits = 0.0
    sequence_total = 0.0
    rollout_loss_sum = 0.0
    rollout_token_total = 0
    rollout_sequence_hits = 0.0
    rollout_sequence_total = 0.0
    continuation_start = max(0, min(config.rollout_prefix_length - 1, config.sequence_length - 1))
    with torch.no_grad():
        for batch in _iter_eval_batches(
            val_source[0],
            val_source[1],
            batch_size=config.eval_batch_size,
            device=device,
            non_blocking=config.pin_memory and device.type == "cuda",
        ):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch["input_ids"])
                loss, tokens = _loss_and_tokens(logits, batch["targets"])
                sequence_mask = torch.ones_like(batch["targets"], dtype=logits.dtype)
                teacher_sequence_hits = _sequence_exact_hits(logits, batch["targets"], sequence_mask)
                rollout_logits, rollout_targets, rollout_mask = _rollout_continuation(
                    model,
                    input_ids=batch["input_ids"],
                    targets=batch["targets"],
                    prefix_length=config.rollout_prefix_length,
                )
                rollout_loss, rollout_tokens = _loss_and_tokens(rollout_logits, rollout_targets)
                rollout_hits = _sequence_exact_hits(rollout_logits, rollout_targets, rollout_mask)
            loss_sum += float(loss.item()) * tokens
            token_total += tokens
            sequence_hits += float(teacher_sequence_hits.sum().item())
            sequence_total += float(teacher_sequence_hits.numel())
            rollout_loss_sum += float(rollout_loss.item()) * rollout_tokens
            rollout_token_total += rollout_tokens
            rollout_sequence_hits += float(rollout_hits.sum().item())
            rollout_sequence_total += float(rollout_hits.numel())
    return {
        "val_loss": loss_sum / max(token_total, 1),
        "val_sequence_accuracy": sequence_hits / max(sequence_total, 1.0),
        "val_rollout_loss": rollout_loss_sum / max(rollout_token_total, 1),
        "val_rollout_sequence_accuracy": rollout_sequence_hits / max(rollout_sequence_total, 1.0),
        "rollout_continuation_start": float(continuation_start),
    }


def _train_for_objective(
    model: nn.Module,
    train_dataset: TokenBlockDataset,
    val_dataset: TokenBlockDataset,
    *,
    objective_mode: str,
    config: RolloutObjectiveConfig,
    batch_schedule: list[torch.Tensor],
) -> dict[str, Any]:
    device = torch.device(config.device)
    use_amp = config.use_amp and device.type == "cuda"
    shared_config = _shared_realtext_config(config)
    model.to(device)
    optimizer = _build_optimizer(model, shared_config, model_name=config.model_variant)
    scheduler = _build_scheduler(optimizer, shared_config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]
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

    initial_metrics = _evaluate_objective(model, val_source, device=device, config=config, use_amp=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history: list[dict[str, float]] = [
        {
            "step": 0.0,
            "tokens_seen": 0.0,
            **initial_metrics,
        }
    ]
    step_times: list[float] = []
    tokens_seen = 0
    start = time.perf_counter()
    for step, batch_indices in enumerate(batch_schedule, start=1):
        batch = _scheduled_batch_from_tensors(
            train_source[0],
            train_source[1],
            batch_indices,
            device=device,
            non_blocking=config.pin_memory and device.type == "cuda",
        )
        step_start = time.perf_counter()
        model.train()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch["input_ids"])
            ce_loss, token_count = _loss_and_tokens(logits, batch["targets"])
            full_mask = torch.ones_like(batch["targets"], dtype=logits.dtype)
            seq_focus_loss = _sequence_focus_loss(
                logits,
                batch["targets"],
                full_mask,
                temperature=config.sequence_focus_temperature,
            )
            rollout_logits, rollout_targets, rollout_mask = _rollout_continuation(
                model,
                input_ids=batch["input_ids"],
                targets=batch["targets"],
                prefix_length=config.rollout_prefix_length,
                teacher_force_prob=config.rollout_teacher_force_prob,
            )
            rollout_loss, _ = _loss_and_tokens(rollout_logits, rollout_targets)
            rollout_seq_focus_loss = _sequence_focus_loss(
                rollout_logits,
                rollout_targets,
                rollout_mask,
                temperature=config.sequence_focus_temperature,
            )
            loss = _objective_loss(
                objective_mode,
                ce_loss=ce_loss,
                seq_focus_loss=seq_focus_loss,
                rollout_loss=rollout_loss,
                rollout_seq_focus_loss=rollout_seq_focus_loss,
                config=config,
            )
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
        if step % config.eval_interval == 0 or step == config.train_steps:
            metrics = _evaluate_objective(model, val_source, device=device, config=config, use_amp=use_amp)
            history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "train_ce_loss": float(ce_loss.item()),
                    "train_sequence_focus_loss": float(seq_focus_loss.item()),
                    "train_rollout_loss": float(rollout_loss.item()),
                    "train_rollout_sequence_focus_loss": float(rollout_seq_focus_loss.item()),
                    **metrics,
                }
            )
            print(
                f"[{objective_mode}] step {step}/{config.train_steps} "
                f"val_loss={metrics['val_loss']:.4f} rollout_loss={metrics['val_rollout_loss']:.4f}",
                flush=True,
            )
    total_time = time.perf_counter() - start
    pure_train_time = sum(step_times)
    final_metrics = history[-1]
    return {
        "parameter_count": count_parameters(model),
        "initial_val_loss": float(initial_metrics["val_loss"]),
        "initial_val_sequence_accuracy": float(initial_metrics["val_sequence_accuracy"]),
        "initial_val_rollout_loss": float(initial_metrics["val_rollout_loss"]),
        "initial_val_rollout_sequence_accuracy": float(initial_metrics["val_rollout_sequence_accuracy"]),
        "final_val_loss": float(final_metrics["val_loss"]),
        "final_val_sequence_accuracy": float(final_metrics["val_sequence_accuracy"]),
        "final_val_rollout_loss": float(final_metrics["val_rollout_loss"]),
        "final_val_rollout_sequence_accuracy": float(final_metrics["val_rollout_sequence_accuracy"]),
        "train_tokens_seen": tokens_seen,
        "train_tok_per_sec": tokens_seen / max(total_time, 1e-9),
        "pure_train_tok_per_sec": tokens_seen / max(pure_train_time, 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "pure_train_time_seconds": pure_train_time,
        "eval_overhead_seconds": max(total_time - pure_train_time, 0.0),
        "peak_vram_mb": (
            (torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else 0.0
        ),
        "history": history,
    }


def run_rollout_objective_compare(
    config: RolloutObjectiveConfig,
    *,
    objective_modes: tuple[str, ...] = OBJECTIVE_MODES,
) -> dict[str, Any]:
    invalid_modes = sorted(set(objective_modes) - set(OBJECTIVE_MODES))
    if invalid_modes:
        raise ValueError(f"Unsupported objective modes: {invalid_modes}")
    set_global_seed(config.seed)
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(config)
    partial_token_ids = _top_token_ids(
        train_dataset,
        count=config.partial_untied_tokens,
        vocab_size=vocab_size,
    )
    schedule_seed = config.seed if config.train_schedule_seed is None else config.train_schedule_seed
    batch_schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=config.batch_size,
        steps=config.train_steps,
        seed=schedule_seed,
        drop_last=True,
    )
    reports: dict[str, dict[str, Any]] = {}
    for objective_mode in objective_modes:
        if config.reseed_per_objective:
            set_global_seed(config.seed)
        model = _build_model(config, vocab_size=vocab_size, partial_token_ids=partial_token_ids)
        reports[objective_mode] = _train_for_objective(
            model,
            train_dataset,
            val_dataset,
            objective_mode=objective_mode,
            config=config,
            batch_schedule=batch_schedule,
        )
    return {
        "benchmark": "language_candidate_rollout_objective",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
        },
        "objective_modes": list(objective_modes),
        "results": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CE-only and rollout/sequence-focused objectives on cached real-text recurrent microbenchmarks."
    )
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--model-variant", type=str, default="partial_untied", choices=("partial_untied", "recurrent_champion"))
    parser.add_argument("--train-blocks", type=int, default=2048)
    parser.add_argument("--val-blocks", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-amp", action="store_true", default=torch.cuda.is_available())
    parser.add_argument("--no-amp", action="store_false", dest="use_amp")
    parser.add_argument("--pin-memory", action="store_true", default=torch.cuda.is_available())
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--use-fused-adamw", action="store_true", default=torch.cuda.is_available())
    parser.add_argument("--no-fused-adamw", action="store_false", dest="use_fused_adamw")
    parser.add_argument("--cache-dataset-on-device", action="store_true")
    parser.add_argument("--recurrent-embedding-dim", type=int, default=144)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=288)
    parser.add_argument("--recurrent-memory-dim", type=int, default=144)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--partial-untied-tokens", type=int, default=512)
    parser.add_argument("--rollout-prefix-length", type=int, default=8)
    parser.add_argument("--rollout-loss-weight", type=float, default=0.5)
    parser.add_argument("--sequence-focus-weight", type=float, default=0.25)
    parser.add_argument("--sequence-focus-temperature", type=float, default=1.5)
    parser.add_argument("--rollout-teacher-force-prob", type=float, default=0.0)
    parser.add_argument("--optimizer-recipe", type=str, default="default", choices=("default", "transformer_fair"))
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-schedule", type=str, default="none", choices=("none", "linear", "cosine"))
    parser.add_argument("--min-lr-scale", type=float, default=1.0)
    parser.add_argument("--objectives", nargs="+", default=list(OBJECTIVE_MODES), choices=OBJECTIVE_MODES)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = RolloutObjectiveConfig(
        cache_path=args.cache_path,
        model_variant=args.model_variant,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        device=args.device,
        use_amp=args.use_amp,
        pin_memory=args.pin_memory,
        use_fused_adamw=args.use_fused_adamw,
        cache_dataset_on_device=args.cache_dataset_on_device,
        recurrent_embedding_dim=args.recurrent_embedding_dim,
        recurrent_hidden_dim=args.recurrent_hidden_dim,
        recurrent_memory_dim=args.recurrent_memory_dim,
        dropout=args.dropout,
        partial_untied_tokens=args.partial_untied_tokens,
        rollout_prefix_length=args.rollout_prefix_length,
        rollout_loss_weight=args.rollout_loss_weight,
        sequence_focus_weight=args.sequence_focus_weight,
        sequence_focus_temperature=args.sequence_focus_temperature,
        rollout_teacher_force_prob=args.rollout_teacher_force_prob,
        optimizer_recipe=args.optimizer_recipe,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        min_lr_scale=args.min_lr_scale,
    )
    payload = run_rollout_objective_compare(config, objective_modes=tuple(args.objectives))
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
