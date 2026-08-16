from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
LANGUAGE_DIR = THIS_DIR.parent
sys.path.insert(0, str(LANGUAGE_DIR))

import standalone_longseq_anchor_train as base


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def hardware_report(device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_count": torch.cuda.device_count(),
                "gpu_total_memory_mb": props.total_memory / (1024 * 1024),
                "gpu_major": props.major,
                "gpu_minor": props.minor,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            }
        )
    return report


def save_checkpoint(
    path: Path,
    *,
    config: base.TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    tokens_seen: int,
    history: list[dict[str, float]],
    step_times: list[float],
    h100_args: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "benchmark": "h100_wave10_fullvocab_train",
            "config": {
                **asdict(config),
                "cache_path": str(config.cache_path),
                "output_dir": str(config.output_dir),
                "resume_checkpoint": str(config.resume_checkpoint) if config.resume_checkpoint else None,
            },
            "h100_args": h100_args,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "history": history,
            "step_times": step_times,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        tmp,
    )
    tmp.replace(path)


def train(config: base.TrainConfig, *, log_interval: int, compile_model: bool, compile_mode: str) -> None:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    autocast_kwargs = {"device_type": "cuda", "dtype": dtype, "enabled": device.type == "cuda"}

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    result_path = output_dir / "result.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics_jsonl = output_dir / "metrics.jsonl"
    metrics_csv = output_dir / "metrics.csv"
    metric_fields = [
        "event",
        "step",
        "train_steps",
        "tokens_seen",
        "train_loss",
        "val_loss_full_vocab",
        "learning_rate",
        "grad_norm",
        "candidate_size",
        "step_time_ms",
        "eval_seconds",
        "pure_train_tok_per_sec",
        "rolling_100_tok_per_sec",
        "peak_allocated_mb",
        "peak_reserved_mb",
    ]

    run_meta = {
        "run_name": config.run_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": "training uses sampled+anchor objective; validation is full-vocabulary cross entropy over all validation tokens",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
            "output_dir": str(config.output_dir),
            "resume_checkpoint": str(config.resume_checkpoint) if config.resume_checkpoint else None,
        },
        "hardware": hardware_report(device),
        "compile_model": compile_model,
        "compile_mode": compile_mode,
    }
    base.write_json_atomic(output_dir / "run_meta.json", run_meta)
    base.write_json_atomic(
        state_path,
        {
            "status": "loading_cache",
            "run_name": config.run_name,
            "cache_path": str(config.cache_path),
            "train_steps": config.train_steps,
        },
    )

    train_inputs, train_targets, val_inputs, val_targets, vocab_size = base.load_cache(config)
    if vocab_size != config.vocab_size:
        raise RuntimeError(f"cache vocab size {vocab_size} != configured vocab size {config.vocab_size}")
    fixed_candidate_ids = base.top_token_ids(train_targets, count=config.sampled_vocab_size, vocab_size=config.vocab_size)
    schedule = base.build_batch_schedule(
        len(train_inputs),
        batch_size=config.batch_size,
        steps=config.train_steps,
        seed=config.seed,
    )

    model: torch.nn.Module = base.CausalConvFactorizedLM(config).to(device)
    parameter_count = base.count_parameters(model)
    if compile_model:
        model = torch.compile(model, mode=compile_mode)

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=device.type == "cuda",
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda" and config.amp_dtype == "fp16")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    history: list[dict[str, float]] = []
    step_times: list[float] = []
    tokens_seen = 0
    start_step = 1

    if config.resume_checkpoint is not None and config.resume_checkpoint.exists():
        checkpoint = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
        if cuda_rng_state_all is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])
        history = list(checkpoint.get("history", []))
        step_times = list(checkpoint.get("step_times", []))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        start_step = int(checkpoint.get("step", 0)) + 1
        print(f"RESUMED step={start_step - 1} tokens={tokens_seen}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"START run={config.run_name} device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} "
        f"params={parameter_count:,} seq={config.sequence_length} batch={config.batch_size} steps={config.train_steps} "
        f"amp={config.amp_dtype} compile={compile_model}",
        flush=True,
    )
    base.write_json_atomic(
        state_path,
        {
            "status": "running",
            "run_name": config.run_name,
            "step": start_step - 1,
            "train_steps": config.train_steps,
            "tokens_seen": tokens_seen,
            "parameter_count": parameter_count,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "full_vocab_validation": True,
        },
    )

    latest_val_loss = float("nan")
    latest_train_loss = float("nan")
    latest_grad_norm = float("nan")
    candidate_sizes: list[int] = []
    wall_start = time.perf_counter()

    for step in range(start_step, config.train_steps + 1):
        batch_indices = schedule[step - 1]
        batch_inputs = train_inputs.index_select(0, batch_indices).to(device, non_blocking=True)
        batch_targets = train_targets.index_select(0, batch_indices).to(device, non_blocking=True)
        if batch_inputs.dtype != torch.long:
            batch_inputs = batch_inputs.long()
        if batch_targets.dtype != torch.long:
            batch_targets = batch_targets.long()

        current_lr = base.scheduled_learning_rate(config, step)
        base.set_optimizer_lr(optimizer, current_lr)
        step_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(**autocast_kwargs):
            loss, token_count, candidate_size = base.anchor_loss(
                model,
                batch_inputs,
                batch_targets,
                fixed_candidate_ids=fixed_candidate_ids,
                config=config,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        step_duration = time.perf_counter() - step_start
        step_times.append(step_duration)
        tokens_seen += int(token_count)
        candidate_sizes.append(int(candidate_size))
        latest_train_loss = float(loss.detach().item())
        latest_grad_norm = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
        pure_time = sum(step_times)
        pure_tps = tokens_seen / max(pure_time, 1e-9)
        recent_times = step_times[-100:]
        recent_tokens = config.batch_size * config.sequence_length * len(recent_times)
        rolling_tps = recent_tokens / max(sum(recent_times), 1e-9)
        peak_allocated = (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None
        )
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024) if device.type == "cuda" else None

        if step == 1 or step % log_interval == 0:
            row = {
                "event": "train",
                "step": step,
                "train_steps": config.train_steps,
                "tokens_seen": tokens_seen,
                "train_loss": latest_train_loss,
                "val_loss_full_vocab": latest_val_loss,
                "learning_rate": current_lr,
                "grad_norm": latest_grad_norm,
                "candidate_size": candidate_size,
                "step_time_ms": step_duration * 1000.0,
                "eval_seconds": "",
                "pure_train_tok_per_sec": pure_tps,
                "rolling_100_tok_per_sec": rolling_tps,
                "peak_allocated_mb": peak_allocated,
                "peak_reserved_mb": peak_reserved,
            }
            append_csv(metrics_csv, metric_fields, row)
            write_jsonl(metrics_jsonl, row)
            base.write_json_atomic(
                state_path,
                {
                    "status": "running",
                    "run_name": config.run_name,
                    "step": step,
                    "train_steps": config.train_steps,
                    "tokens_seen": tokens_seen,
                    "latest_train_loss": latest_train_loss,
                    "latest_val_loss_full_vocab": latest_val_loss,
                    "latest_learning_rate": current_lr,
                    "latest_grad_norm": latest_grad_norm,
                    "pure_train_tok_per_sec": pure_tps,
                    "rolling_100_tok_per_sec": rolling_tps,
                    "peak_allocated_mb": peak_allocated,
                    "peak_reserved_mb": peak_reserved,
                    "checkpoint_path": str(checkpoint_path),
                },
            )
            print(
                f"TRAIN step={step}/{config.train_steps} tokens={tokens_seen} loss={latest_train_loss:.4f} "
                f"lr={current_lr:.6g} grad={latest_grad_norm:.3f} pure_tok_s={pure_tps:.0f} "
                f"roll100_tok_s={rolling_tps:.0f}",
                flush=True,
            )

        should_eval = step % config.eval_interval == 0 or step == config.train_steps
        if should_eval:
            eval_start = time.perf_counter()
            latest_val_loss = base.evaluate_full_loss(
                model,
                val_inputs,
                val_targets,
                config=config,
                device=device,
                autocast_kwargs=autocast_kwargs,
            )
            eval_seconds = time.perf_counter() - eval_start
            history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": latest_train_loss,
                    "val_loss_full_vocab": latest_val_loss,
                    "learning_rate": current_lr,
                    "grad_norm": latest_grad_norm,
                    "pure_train_tok_per_sec": pure_tps,
                    "rolling_100_tok_per_sec": rolling_tps,
                    "peak_allocated_mb": float(peak_allocated) if peak_allocated is not None else float("nan"),
                    "peak_reserved_mb": float(peak_reserved) if peak_reserved is not None else float("nan"),
                    "eval_seconds": eval_seconds,
                }
            )
            row = {
                "event": "eval_full_vocab",
                "step": step,
                "train_steps": config.train_steps,
                "tokens_seen": tokens_seen,
                "train_loss": latest_train_loss,
                "val_loss_full_vocab": latest_val_loss,
                "learning_rate": current_lr,
                "grad_norm": latest_grad_norm,
                "candidate_size": candidate_size,
                "step_time_ms": "",
                "eval_seconds": eval_seconds,
                "pure_train_tok_per_sec": pure_tps,
                "rolling_100_tok_per_sec": rolling_tps,
                "peak_allocated_mb": peak_allocated,
                "peak_reserved_mb": peak_reserved,
            }
            append_csv(metrics_csv, metric_fields, row)
            write_jsonl(metrics_jsonl, row)
            print(
                f"EVAL_FULL_VOCAB step={step}/{config.train_steps} tokens={tokens_seen} "
                f"train={latest_train_loss:.4f} val_full={latest_val_loss:.4f} eval_s={eval_seconds:.1f}",
                flush=True,
            )
            base.write_json_atomic(
                state_path,
                {
                    "status": "running",
                    "run_name": config.run_name,
                    "step": step,
                    "train_steps": config.train_steps,
                    "tokens_seen": tokens_seen,
                    "latest_train_loss": latest_train_loss,
                    "latest_val_loss_full_vocab": latest_val_loss,
                    "latest_learning_rate": current_lr,
                    "latest_grad_norm": latest_grad_norm,
                    "pure_train_tok_per_sec": pure_tps,
                    "rolling_100_tok_per_sec": rolling_tps,
                    "peak_allocated_mb": peak_allocated,
                    "peak_reserved_mb": peak_reserved,
                    "history": history,
                    "checkpoint_path": str(checkpoint_path),
                },
            )

        should_checkpoint = (
            (config.checkpoint_interval > 0 and step % config.checkpoint_interval == 0)
            or should_eval
            or step == config.train_steps
        )
        if should_checkpoint:
            h100_args = {"compile_model": compile_model, "compile_mode": compile_mode}
            save_checkpoint(
                checkpoint_path,
                config=config,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                tokens_seen=tokens_seen,
                history=history,
                step_times=step_times,
                h100_args=h100_args,
            )
            if config.milestone_checkpoint_interval > 0 and step % config.milestone_checkpoint_interval == 0:
                milestone = output_dir / f"checkpoint.step{step}_tokens{tokens_seen}.pt"
                save_checkpoint(
                    milestone,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    tokens_seen=tokens_seen,
                    history=history,
                    step_times=step_times,
                    h100_args=h100_args,
                )
                print(f"MILESTONE {milestone}", flush=True)

    pure_time = sum(step_times)
    wall_seconds = time.perf_counter() - wall_start
    result = {
        "benchmark": "h100_wave10_fullvocab_train",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
            "output_dir": str(config.output_dir),
            "resume_checkpoint": str(config.resume_checkpoint) if config.resume_checkpoint else None,
        },
        "hardware": hardware_report(device),
        "report": {
            "parameter_count": parameter_count,
            "train_tokens_seen": tokens_seen,
            "final_train_loss": latest_train_loss,
            "final_val_loss_full_vocab": latest_val_loss,
            "pure_train_tok_per_sec": tokens_seen / max(pure_time, 1e-9),
            "wall_tok_per_sec": tokens_seen / max(wall_seconds, 1e-9),
            "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
            "step_time_median_ms": statistics.median(step_times) * 1000.0,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda"
            else None,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024)
            if device.type == "cuda"
            else None,
            "candidate_size_mean": statistics.fmean(candidate_sizes) if candidate_sizes else None,
            "history": history,
            "checkpoint_path": str(checkpoint_path),
            "metrics_csv": str(metrics_csv),
            "metrics_jsonl": str(metrics_jsonl),
        },
    }
    base.write_json_atomic(result_path, result)
    base.write_json_atomic(
        state_path,
        {
            "status": "completed",
            "run_name": config.run_name,
            "step": config.train_steps,
            "train_steps": config.train_steps,
            "tokens_seen": tokens_seen,
            "final_train_loss": latest_train_loss,
            "final_val_loss_full_vocab": latest_val_loss,
            "pure_train_tok_per_sec": result["report"]["pure_train_tok_per_sec"],
            "wall_tok_per_sec": result["report"]["wall_tok_per_sec"],
            "peak_allocated_mb": result["report"]["peak_allocated_mb"],
            "peak_reserved_mb": result["report"]["peak_reserved_mb"],
            "checkpoint_path": str(checkpoint_path),
            "result_path": str(result_path),
        },
    )
    print("RESULT " + json.dumps(result["report"], sort_keys=True), flush=True)


def parse_args() -> tuple[base.TrainConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="H100/H200 Wave10 350M full-vocab-validation trainer.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="wave10_350m_h100_fullvocab")
    parser.add_argument("--target-tokens", type=int, default=5_000_000_000)
    parser.add_argument("--train-steps", type=int, default=0, help="Overrides --target-tokens when > 0.")
    parser.add_argument("--eval-interval", type=int, default=2_500)
    parser.add_argument("--checkpoint-interval", type=int, default=2_500)
    parser.add_argument("--milestone-checkpoint-interval", type=int, default=25_000)
    parser.add_argument("--val-blocks", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=10_160)
    parser.add_argument("--embedding-dim", type=int, default=2_304)
    parser.add_argument("--conv-layers", type=int, default=4)
    parser.add_argument("--conv-kernel-size", type=int, default=7)
    parser.add_argument("--conv-rank", type=int, default=1_536)
    parser.add_argument("--memory-rank", type=int, default=256)
    parser.add_argument("--memory-kernel-size", type=int, default=128)
    parser.add_argument("--sampled-vocab-size", type=int, default=32_768)
    parser.add_argument("--token-stride", type=int, default=4)
    parser.add_argument("--token-chunk-size", type=int, default=2_048)
    parser.add_argument("--full-eval-token-chunk-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile; keep enabled consistently for resume.")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    args = parser.parse_args()

    train_steps = args.train_steps
    if train_steps <= 0:
        train_steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))

    config = base.TrainConfig(
        cache_path=args.cache_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        train_steps=train_steps,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        milestone_checkpoint_interval=args.milestone_checkpoint_interval,
        val_blocks=args.val_blocks,
        seed=args.seed,
        embedding_dim=args.embedding_dim,
        block_type="multi_scale_lowrank_conv_memory",
        conv_layers=args.conv_layers,
        conv_kernel_size=args.conv_kernel_size,
        conv_rank=args.conv_rank,
        memory_rank=args.memory_rank,
        landmark_stride=args.memory_kernel_size,
        sampled_vocab_size=args.sampled_vocab_size,
        token_stride=args.token_stride,
        token_chunk_size=args.token_chunk_size,
        full_eval_token_chunk_size=args.full_eval_token_chunk_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        amp_dtype=args.amp_dtype,
        resume_checkpoint=args.resume_checkpoint,
    )
    return config, args


def main() -> None:
    config, args = parse_args()
    train(config, log_interval=args.log_interval, compile_model=args.compile, compile_mode=args.compile_mode)


if __name__ == "__main__":
    main()
