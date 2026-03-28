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

from arc_tactic3.language_realtext_microbench import (
    RealTextConfig,
    TokenBlockDataset,
    _build_optimizer,
    _build_scheduler,
    _build_train_batch_schedule,
    _dataset_tensors,
    _loss_and_tokens,
    _scheduled_batch_from_tensors,
    evaluate_loss,
    set_global_seed,
)
from arc_tactic3.language_nanochat_actual_compare import _peak_vram_mb
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM, _top_token_ids
from arc_tactic3.language_fastlearn_benchmark import count_parameters


@dataclass(frozen=True, slots=True)
class PartialUntiedWatchConfig:
    cache_path: Path
    output_dir: Path
    target_tokens: int = 50_000_000
    seed: int = 13
    sequence_length: int = 127
    batch_size: int = 16
    eval_batch_size: int = 32
    eval_interval: int = 2048
    log_interval: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    cache_dataset_on_device: bool = True
    train_schedule_seed: int | None = None
    val_blocks: int = 512
    embedding_dim: int = 144
    hidden_dim: int = 288
    memory_dim: int = 144
    partial_token_count: int = 512
    dropout: float = 0.1
    initial_eval: bool = True


def _load_cached_datasets(config: PartialUntiedWatchConfig) -> tuple[TokenBlockDataset, TokenBlockDataset, int]:
    payload = torch.load(config.cache_path, map_location="cpu", weights_only=False)
    block_size = config.sequence_length + 1
    train_blocks = payload["train_tokens"].long().view(-1, block_size)
    val_blocks = payload["val_tokens"].long().view(-1, block_size)
    train_dataset = TokenBlockDataset(train_blocks[:, :-1].contiguous(), train_blocks[:, 1:].contiguous())
    val_dataset = TokenBlockDataset(
        val_blocks[: config.val_blocks, :-1].contiguous(),
        val_blocks[: config.val_blocks, 1:].contiguous(),
    )
    return train_dataset, val_dataset, int(payload["vocab_size"])


def _make_realtext_config(config: PartialUntiedWatchConfig, *, train_steps: int) -> RealTextConfig:
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
        train_schedule_seed=config.train_schedule_seed,
        dropout=config.dropout,
        initial_eval=config.initial_eval,
    )


def _bar(progress: float, width: int = 28) -> str:
    progress = min(max(progress, 0.0), 1.0)
    filled = int(progress * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    last_error: PermissionError | None = None
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _format_eta(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0.0:
        return "?"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def watch_partial_untied_run(
    output_dir: Path,
    *,
    refresh_seconds: float = 5.0,
    once: bool = False,
) -> None:
    state_path = output_dir / "state.json"
    final_path = output_dir / "final.json"
    metrics_path = output_dir / "metrics.jsonl"

    last_seen_step = -1
    last_seen_eval_step = -1
    while True:
        state = _load_json_if_exists(state_path)
        final_payload = _load_json_if_exists(final_path)
        if state is None:
            print(f"waiting for state file: {state_path}", flush=True)
            if once:
                return
            time.sleep(refresh_seconds)
            continue

        progress = float(state.get("progress", 0.0))
        step = int(state.get("step", 0))
        train_steps = int(state.get("train_steps", 0))
        tokens_seen = int(state.get("tokens_seen", 0))
        target_tokens = int(state.get("target_tokens", 0))
        train_tok_per_sec = float(state.get("train_tok_per_sec", float("nan")))
        pure_train_tok_per_sec = float(state.get("pure_train_tok_per_sec", float("nan")))
        train_loss = state.get("latest_train_loss")
        val_loss = state.get("latest_val_loss")
        eta_seconds = state.get("eta_seconds")
        peak_vram_mb = state.get("peak_vram_mb")
        status = str(state.get("status", "running"))

        if step != last_seen_step or once or status == "completed":
            train_loss_text = "----" if train_loss is None or not math.isfinite(float(train_loss)) else f"{float(train_loss):.4f}"
            val_loss_text = "----" if val_loss is None or not math.isfinite(float(val_loss)) else f"{float(val_loss):.4f}"
            print(
                f"partial_untied_50m {_bar(progress)} {progress * 100:5.1f}% "
                f"step={step:,}/{train_steps:,} tok={tokens_seen:,}/{target_tokens:,} "
                f"train={train_loss_text} val={val_loss_text} "
                f"tok/s={train_tok_per_sec:,.0f} pure_tok/s={pure_train_tok_per_sec:,.0f} "
                f"eta={_format_eta(float(eta_seconds) if eta_seconds is not None else None)} "
                f"vram={float(peak_vram_mb):.0f}MB status={status}",
                flush=True,
            )
            last_seen_step = step

        if metrics_path.exists():
            with metrics_path.open("rb") as handle:
                try:
                    handle.seek(-4096, 2)
                except OSError:
                    handle.seek(0)
                tail_text = handle.read().decode("utf-8", errors="replace")
            for line in reversed(tail_text.splitlines()):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("kind") == "eval":
                    eval_step = int(payload.get("step", 0))
                    if eval_step != last_seen_eval_step or once or status == "completed":
                        print(
                            f"latest eval: step={eval_step:,} tokens={int(payload.get('tokens_seen', 0)):,} "
                            f"val={float(payload.get('val_loss', float('nan'))):.4f} "
                            f"train={float(payload.get('train_loss', float('nan'))):.4f}",
                            flush=True,
                        )
                        last_seen_eval_step = eval_step
                    break

        if final_payload is not None and status == "completed":
            report = final_payload.get("report", {})
            print(
                f"completed final_val={float(report.get('final_val_loss', float('nan'))):.4f} "
                f"tokens={int(report.get('train_tokens_seen', 0)):,} "
                f"train_tok/s={float(report.get('train_tok_per_sec', float('nan'))):,.0f} "
                f"pure_tok/s={float(report.get('pure_train_tok_per_sec', float('nan'))):,.0f}",
                flush=True,
            )
            return

        if once:
            return
        time.sleep(refresh_seconds)


def run_partial_untied_watch(config: PartialUntiedWatchConfig, *, print_progress: bool = True) -> dict[str, Any]:
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(config)
    tokens_per_step = config.sequence_length * config.batch_size
    train_steps = math.ceil(config.target_tokens / tokens_per_step)
    actual_target_tokens = train_steps * tokens_per_step
    real_config = _make_realtext_config(config, train_steps=train_steps)

    set_global_seed(config.seed)
    partial_token_ids = _top_token_ids(train_dataset, count=config.partial_token_count, vocab_size=vocab_size)
    model = PartialUntiedAssociativeLM(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        memory_dim=config.memory_dim,
        dropout=config.dropout,
        max_length=config.sequence_length,
        untied_token_ids=partial_token_ids,
    )
    device = torch.device(real_config.device)
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
    val_source = _dataset_tensors(
        val_dataset,
        device=device,
        cache_on_device=real_config.cache_dataset_on_device,
        pin_memory=real_config.pin_memory,
    )

    schedule_seed = config.seed if config.train_schedule_seed is None else config.train_schedule_seed
    batch_schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=real_config.batch_size,
        steps=real_config.train_steps,
        seed=schedule_seed,
        drop_last=True,
    )

    metrics_path = config.output_dir / "metrics.jsonl"
    state_path = config.output_dir / "state.json"
    final_path = config.output_dir / "final.json"

    history: list[dict[str, float]] = []
    step_times: list[float] = []
    tokens_seen = 0
    sequences_seen = 0
    latest_val_loss = float("nan")
    initial_val_loss = float("nan")
    if real_config.initial_eval:
        initial_val_loss = evaluate_loss(model, val_source, device=device, use_amp=use_amp, config=real_config)
        latest_val_loss = initial_val_loss
        history.append(
            {
                "step": 0.0,
                "sequences_seen": 0.0,
                "tokens_seen": 0.0,
                "train_loss": float("nan"),
                "val_loss": float(initial_val_loss),
            }
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    last_report = start
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
        step_duration = time.perf_counter() - step_start
        step_times.append(step_duration)
        tokens_seen += token_count
        sequences_seen += batch["input_ids"].size(0)

        elapsed = time.perf_counter() - start
        progress = step / real_config.train_steps
        train_tok_per_sec = tokens_seen / max(elapsed, 1e-9)
        pure_train_tok_per_sec = tokens_seen / max(sum(step_times), 1e-9)
        remaining_tokens = max(actual_target_tokens - tokens_seen, 0)
        eta_seconds = remaining_tokens / max(train_tok_per_sec, 1e-9)
        state_payload = {
            "step": step,
            "train_steps": real_config.train_steps,
            "progress": progress,
            "tokens_seen": tokens_seen,
            "target_tokens": actual_target_tokens,
            "train_tok_per_sec": train_tok_per_sec,
            "pure_train_tok_per_sec": pure_train_tok_per_sec,
            "eta_seconds": eta_seconds,
            "latest_train_loss": float(loss.item()),
            "latest_val_loss": latest_val_loss,
            "peak_vram_mb": _peak_vram_mb(real_config.device),
        }

        if step % config.log_interval == 0 or step == real_config.train_steps:
            _write_json(state_path, state_payload)
            _append_jsonl(
                metrics_path,
                {
                    "kind": "train",
                    **state_payload,
                    "timestamp": time.time(),
                },
            )
            if print_progress:
                print(
                    f"partial_untied_50m {_bar(progress)} {step}/{real_config.train_steps} "
                    f"train={loss.item():.4f} val={latest_val_loss:.4f} tok={tokens_seen:,}/{actual_target_tokens:,} "
                    f"tok/s={train_tok_per_sec:,.0f} eta={eta_seconds/60:.1f}m",
                    flush=True,
                )
            last_report = time.perf_counter()

        if step % real_config.eval_interval == 0 or step == real_config.train_steps:
            latest_val_loss = evaluate_loss(model, val_source, device=device, use_amp=use_amp, config=real_config)
            history.append(
                {
                    "step": float(step),
                    "sequences_seen": float(sequences_seen),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": float(loss.item()),
                    "val_loss": float(latest_val_loss),
                }
            )
            eval_payload = {
                "kind": "eval",
                "step": step,
                "tokens_seen": tokens_seen,
                "val_loss": latest_val_loss,
                "train_loss": float(loss.item()),
                "train_tok_per_sec": train_tok_per_sec,
                "pure_train_tok_per_sec": pure_train_tok_per_sec,
                "peak_vram_mb": _peak_vram_mb(real_config.device),
                "timestamp": time.time(),
            }
            _write_json(state_path, {**state_payload, "latest_val_loss": latest_val_loss})
            _append_jsonl(metrics_path, eval_payload)
            if print_progress:
                print(
                    f"eval step={step:,} tokens={tokens_seen:,} val={latest_val_loss:.4f} "
                    f"train_tok/s={train_tok_per_sec:,.0f} pure_tok/s={pure_train_tok_per_sec:,.0f}",
                    flush=True,
                )

    total_time = time.perf_counter() - start
    pure_train_time = sum(step_times)
    report = {
        "parameter_count": count_parameters(model),
        "history": history,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": latest_val_loss,
        "train_tokens_seen": tokens_seen,
        "train_tok_per_sec": tokens_seen / max(total_time, 1e-9),
        "pure_train_tok_per_sec": tokens_seen / max(pure_train_time, 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "pure_train_time_seconds": pure_train_time,
        "eval_overhead_seconds": max(total_time - pure_train_time, 0.0),
        "peak_vram_mb": _peak_vram_mb(real_config.device),
        "total_training_time_seconds": total_time,
    }
    final_payload = {
        "benchmark": "language_partial_untied_watch",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
            "output_dir": str(config.output_dir),
            "resolved_train_steps": train_steps,
            "tokens_per_step": tokens_per_step,
            "actual_target_tokens": actual_target_tokens,
        },
        "report": report,
    }
    _write_json(final_path, final_payload)
    _write_json(
        state_path,
        {
            "status": "completed",
            "step": train_steps,
            "train_steps": train_steps,
            "progress": 1.0,
            "tokens_seen": tokens_seen,
            "target_tokens": actual_target_tokens,
            "latest_val_loss": latest_val_loss,
            "train_tok_per_sec": report["train_tok_per_sec"],
            "pure_train_tok_per_sec": report["pure_train_tok_per_sec"],
            "peak_vram_mb": report["peak_vram_mb"],
            "final_path": str(final_path),
        },
    )
    if print_progress:
        print(
            f"completed partial_untied_50m final_val={latest_val_loss:.4f} tokens={tokens_seen:,} "
            f"train_tok/s={report['train_tok_per_sec']:,.0f} pure_tok/s={report['pure_train_tok_per_sec']:,.0f}",
            flush=True,
        )
    return final_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run monitored long training for partial_untied and write live progress files.")
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-tokens", type=int, default=50_000_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-interval", type=int, default=2048)
    parser.add_argument("--log-interval", type=int, default=256)
    parser.add_argument("--val-blocks", type=int, default=512)
    parser.add_argument("--watch-dir", type=Path)
    parser.add_argument("--watch-refresh", type=float, default=5.0)
    parser.add_argument("--watch-once", action="store_true")
    args = parser.parse_args()

    if args.watch_dir is not None:
        watch_partial_untied_run(args.watch_dir, refresh_seconds=args.watch_refresh, once=args.watch_once)
        return

    if args.cache_path is None or args.output_dir is None:
        parser.error("--cache-path and --output-dir are required unless --watch-dir is used")

    config = PartialUntiedWatchConfig(
        cache_path=args.cache_path,
        output_dir=args.output_dir,
        target_tokens=args.target_tokens,
        seed=args.seed,
        device=args.device,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        val_blocks=args.val_blocks,
    )
    payload = run_partial_untied_watch(config, print_progress=True)
    print(json.dumps({"final_val_loss": payload["report"]["final_val_loss"], "final_path": str(config.output_dir / "final.json")}, indent=2))


if __name__ == "__main__":
    main()
