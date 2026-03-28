from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from arc_tactic3.language_nanochat_watch import NanochatWatchConfig, run_nanochat_watch
from arc_tactic3.language_partial_untied_watch import (
    PartialUntiedWatchConfig,
    _bar,
    _format_eta,
    _load_json_if_exists,
    _write_json,
    run_partial_untied_watch,
)


@dataclass(frozen=True, slots=True)
class ReplicationBenchmarkConfig:
    cache_path: Path
    output_root: Path
    seeds: tuple[int, ...] = (13, 29, 41)
    target_tokens: int = 50_000_000
    device: str = "cuda"
    sequence_length: int = 127
    batch_size: int = 16
    eval_batch_size: int = 32
    eval_interval: int = 2048
    log_interval: int = 256
    val_blocks: int = 512


def _summary_path(output_root: Path) -> Path:
    return output_root / "summary.json"


def _state_path(output_root: Path) -> Path:
    return output_root / "state.json"


def _run_dir(output_root: Path, model_name: str, seed: int) -> Path:
    return output_root / f"{model_name}_seed{seed}"


def _aggregate_results(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"completed_runs": 0, "models": {}}
    for model_name, payloads in results.items():
        if not payloads:
            continue
        reports = [payload["report"] for payload in payloads]
        summary["models"][model_name] = {
            "seeds": [payload["config"]["seed"] for payload in payloads],
            "mean_final_val_loss": statistics.fmean(report["final_val_loss"] for report in reports),
            "mean_train_tok_per_sec": statistics.fmean(report["train_tok_per_sec"] for report in reports),
            "mean_pure_train_tok_per_sec": statistics.fmean(report["pure_train_tok_per_sec"] for report in reports),
            "mean_peak_vram_mb": statistics.fmean(report["peak_vram_mb"] for report in reports),
            "parameter_count": reports[0]["parameter_count"],
            "completed_runs": len(payloads),
        }
        summary["completed_runs"] += len(payloads)
    return summary


def run_replication_benchmark(config: ReplicationBenchmarkConfig, *, print_progress: bool = True) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    total_runs = len(config.seeds) * 2
    results: dict[str, list[dict[str, Any]]] = {"partial_untied": [], "nanochat_small": []}
    manifest = {
        "benchmark": "replication_50m_head_to_head",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
            "output_root": str(config.output_root),
        },
        "total_runs": total_runs,
    }
    _write_json(config.output_root / "manifest.json", manifest)

    run_index = 0
    for seed in config.seeds:
        for model_name in ("partial_untied", "nanochat_small"):
            run_index += 1
            run_dir = _run_dir(config.output_root, model_name, seed)
            final_path = run_dir / "final.json"
            top_state = {
                "status": "running",
                "run_index": run_index,
                "total_runs": total_runs,
                "current_model": model_name,
                "current_seed": seed,
                "current_run_dir": str(run_dir),
                "completed_runs": sum(len(items) for items in results.values()),
            }
            _write_json(_state_path(config.output_root), top_state)
            if print_progress:
                print(f"=== [{run_index}/{total_runs}] {model_name} seed={seed} ===", flush=True)

            if final_path.exists():
                payload = json.loads(final_path.read_text(encoding="utf-8"))
            else:
                if model_name == "partial_untied":
                    payload = run_partial_untied_watch(
                        PartialUntiedWatchConfig(
                            cache_path=config.cache_path,
                            output_dir=run_dir,
                            target_tokens=config.target_tokens,
                            seed=seed,
                            sequence_length=config.sequence_length,
                            batch_size=config.batch_size,
                            eval_batch_size=config.eval_batch_size,
                            eval_interval=config.eval_interval,
                            log_interval=config.log_interval,
                            device=config.device,
                            val_blocks=config.val_blocks,
                        ),
                        print_progress=print_progress,
                    )
                else:
                    payload = run_nanochat_watch(
                        NanochatWatchConfig(
                            cache_path=config.cache_path,
                            output_dir=run_dir,
                            target_tokens=config.target_tokens,
                            seed=seed,
                            sequence_length=config.sequence_length,
                            batch_size=config.batch_size,
                            eval_batch_size=config.eval_batch_size,
                            eval_interval=config.eval_interval,
                            log_interval=config.log_interval,
                            device=config.device,
                            val_blocks=config.val_blocks,
                        ),
                        print_progress=print_progress,
                    )
            results[model_name].append(payload)
            summary = _aggregate_results(results)
            _write_json(_summary_path(config.output_root), summary)
            _write_json(
                _state_path(config.output_root),
                {
                    "status": "running",
                    "run_index": run_index,
                    "total_runs": total_runs,
                    "current_model": model_name,
                    "current_seed": seed,
                    "current_run_dir": str(run_dir),
                    "completed_runs": sum(len(items) for items in results.values()),
                    "summary_path": str(_summary_path(config.output_root)),
                },
            )

    final_summary = _aggregate_results(results)
    _write_json(_summary_path(config.output_root), final_summary)
    _write_json(
        _state_path(config.output_root),
        {
            "status": "completed",
            "run_index": total_runs,
            "total_runs": total_runs,
            "completed_runs": total_runs,
            "summary_path": str(_summary_path(config.output_root)),
        },
    )
    if print_progress:
        print(json.dumps(final_summary, indent=2), flush=True)
    return final_summary


def watch_replication_benchmark(output_root: Path, *, refresh_seconds: float = 5.0, once: bool = False) -> None:
    top_state_path = _state_path(output_root)
    last_signature: tuple[Any, ...] | None = None
    while True:
        top_state = _load_json_if_exists(top_state_path)
        summary = _load_json_if_exists(_summary_path(output_root))
        if top_state is None:
            print(f"waiting for state file: {top_state_path}", flush=True)
            if once:
                return
            time.sleep(refresh_seconds)
            continue
        status = str(top_state.get("status", "running"))
        run_index = int(top_state.get("run_index", 0))
        total_runs = int(top_state.get("total_runs", 0))
        current_model = top_state.get("current_model")
        current_seed = top_state.get("current_seed")
        current_run_dir = top_state.get("current_run_dir")
        signature = (status, run_index, current_model, current_seed, current_run_dir)
        sub_state = None
        if current_run_dir:
            sub_state = _load_json_if_exists(Path(current_run_dir) / "state.json")

        if signature != last_signature or once or status == "completed":
            if status == "completed":
                print(f"replication_50m {_bar(1.0)} {total_runs}/{total_runs} completed", flush=True)
            elif sub_state is None:
                print(f"replication_50m {_bar((run_index - 1) / max(total_runs, 1))} [{run_index}/{total_runs}] {current_model} seed={current_seed}", flush=True)
            else:
                sub_progress = float(sub_state.get("progress", 0.0))
                overall_progress = ((run_index - 1) + sub_progress) / max(total_runs, 1)
                tokens_seen = int(sub_state.get("tokens_seen", 0))
                target_tokens = int(sub_state.get("target_tokens", 0))
                train_loss = sub_state.get("latest_train_loss")
                val_loss = sub_state.get("latest_val_loss")
                eta_seconds = sub_state.get("eta_seconds")
                train_tok_per_sec = float(sub_state.get("train_tok_per_sec", float("nan")))
                train_loss_text = "----" if train_loss is None or not math.isfinite(float(train_loss)) else f"{float(train_loss):.4f}"
                val_loss_text = "----" if val_loss is None or not math.isfinite(float(val_loss)) else f"{float(val_loss):.4f}"
                print(
                    f"replication_50m {_bar(overall_progress)} [{run_index}/{total_runs}] {current_model} seed={current_seed} "
                    f"train={train_loss_text} val={val_loss_text} tok={tokens_seen:,}/{target_tokens:,} "
                    f"tok/s={train_tok_per_sec:,.0f} eta={_format_eta(float(eta_seconds) if eta_seconds is not None else None)}",
                    flush=True,
                )
            if summary is not None and summary.get("models"):
                print(json.dumps(summary, indent=2), flush=True)
            last_signature = signature

        if status == "completed":
            return
        if once:
            return
        time.sleep(refresh_seconds)


def _parse_seeds(seed_text: str) -> tuple[int, ...]:
    return tuple(int(piece.strip()) for piece in seed_text.split(",") if piece.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or watch the exact 3-seed 50M partial_untied vs nanochat replication benchmark.")
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--target-tokens", type=int, default=50_000_000)
    parser.add_argument("--seeds", type=str, default="13,29,41")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--watch-dir", type=Path)
    parser.add_argument("--watch-refresh", type=float, default=5.0)
    parser.add_argument("--watch-once", action="store_true")
    args = parser.parse_args()

    if args.watch_dir is not None:
        watch_replication_benchmark(args.watch_dir, refresh_seconds=args.watch_refresh, once=args.watch_once)
        return
    if args.cache_path is None or args.output_root is None:
        parser.error("--cache-path and --output-root are required unless --watch-dir is used")

    config = ReplicationBenchmarkConfig(
        cache_path=args.cache_path,
        output_root=args.output_root,
        seeds=_parse_seeds(args.seeds),
        target_tokens=args.target_tokens,
        device=args.device,
    )
    run_replication_benchmark(config, print_progress=True)


if __name__ == "__main__":
    main()
