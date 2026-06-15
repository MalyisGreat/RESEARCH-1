from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_DIR = REPO_ROOT / "artifacts" / "benchmark_runs" / "language"

VARIANT = "causal_conv_mixer_sampled_vocab_anchor16"
SEQUENCE_LENGTH = 10_160
TRAIN_STEPS_2B = 196_851
EVAL_INTERVAL = 9_843
MILESTONE_STEP_600M = 59_058
MILESTONE_TOKENS_600M = 600_029_280
VAL_BLOCKS = 32
BASELINE_40M_600M_VAL = 5.094722971320152

RUN_80M = LANGUAGE_DIR / "longseq_anchor16_80m_2b_20260603"
RUN_80M_RETRY = LANGUAGE_DIR / "longseq_anchor16_80m_2b_lr1e3_20260603"
RUN_160M = LANGUAGE_DIR / "longseq_anchor16_160m_2b_20260603"
OUTPUT_80M_RETRY = LANGUAGE_DIR / "language_longseq_anchor16_80m_2b_lr1e3_seq10160_seed13_20260603.json"
OUTPUT_160M = LANGUAGE_DIR / "language_longseq_anchor16_160m_2b_seq10160_seed13_20260603.json"
CACHE_2B = RUN_80M / "cache" / "finewebedu_train2000203011_val325152_seq10160_gpt2.pt"

WATCHER_LOG = LANGUAGE_DIR / "longseq_scaling_goal_watcher_20260603.log"
WATCHER_STATE = LANGUAGE_DIR / "longseq_scaling_goal_watcher_20260603.state.json"
PROGRESS_CSV = LANGUAGE_DIR / "longseq_scaling_goal_progress_20260603.csv"


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def log(message: str) -> None:
    WATCHER_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{utcnow()} {message}"
    print(line, flush=True)
    with WATCHER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(char in text for char in [",", "\"", "\n", "\r"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def append_progress(run_name: str, run_dir: Path) -> None:
    payload = state_payload(run_dir)
    if payload is None:
        return
    pid = pid_from_file(run_dir)
    row = {
        "timestamp": utcnow(),
        "run": run_name,
        "pid": pid if pid is not None else "",
        "pid_alive": process_exists(pid),
        "status": payload.get("status"),
        "step": payload.get("step"),
        "train_steps": payload.get("train_steps"),
        "tokens_seen": payload.get("tokens_seen"),
        "latest_train_loss": payload.get("latest_train_loss"),
        "latest_val_loss": payload.get("latest_val_loss"),
        "pure_train_tok_per_sec": payload.get("pure_train_tok_per_sec"),
        "peak_vram_mb": payload.get("peak_vram_mb"),
    }
    header = list(row.keys())
    PROGRESS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not PROGRESS_CSV.exists()
    with PROGRESS_CSV.open("a", encoding="utf-8", newline="") as handle:
        if write_header:
            handle.write(",".join(header) + "\n")
        handle.write(",".join(csv_value(row[key]) for key in header) + "\n")


def load_watcher_state() -> dict[str, Any]:
    state = load_json(WATCHER_STATE)
    if state is None:
        return {
            "scaling_decided": False,
            "scaling_holds": False,
            "primary_80m_unstable": False,
            "retry_80m_pending": False,
            "retry_80m_launched": False,
            "launch_160m_pending": False,
            "launched_160m": False,
            "160m_unstable": False,
            "restart_counts": {},
        }
    state.setdefault("restart_counts", {})
    state.setdefault("160m_unstable", False)
    return state


def process_exists(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return str(pid) in result.stdout


def pid_from_file(run_dir: Path) -> int | None:
    pid_path = run_dir / "logs" / "train.pid"
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def state_file(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.rglob(f"{VARIANT}.state.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def state_payload(run_dir: Path) -> dict[str, Any] | None:
    path = state_file(run_dir)
    return load_json(path) if path is not None else None


def result_exists(run_dir: Path) -> bool:
    result_path = next(run_dir.rglob(f"{VARIANT}.json"), None)
    if result_path is None:
        return False
    payload = load_json(result_path)
    return bool(payload and payload.get("report"))


def stdout_path(run_dir: Path) -> Path:
    return run_dir / "logs" / "train_stdout.log"


def stderr_path(run_dir: Path) -> Path:
    return run_dir / "logs" / "train_stderr.log"


def parse_eval_from_stdout(run_dir: Path, step: int) -> float | None:
    path = stdout_path(run_dir)
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    pattern = re.compile(rf"{re.escape(VARIANT)} eval step={step}\s+val=([0-9.]+)")
    found: float | None = None
    for line in lines:
        match = pattern.search(line)
        if match:
            found = float(match.group(1))
    return found


def milestone_checkpoint(run_dir: Path, step: int, tokens: int) -> Path | None:
    pattern = f"{VARIANT}.checkpoint.step{step}_tokens{tokens}.pt"
    matches = list(run_dir.rglob(pattern))
    return matches[0] if matches else None


def gpu_used_mb() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                return float(line.split(",")[0].strip())
            except ValueError:
                return None
    return None


def common_args(run_dir: Path, output: Path, conv_dim: int, conv_rank: int, *, learning_rate: float) -> list[str]:
    args = [
        sys.executable,
        "-u",
        "-m",
        "arc_tactic3.language_longseq_replay_probe",
        "--output-dir",
        str(run_dir),
        "--train-blocks",
        str(TRAIN_STEPS_2B),
        "--val-blocks",
        str(VAL_BLOCKS),
        "--sequence-length",
        str(SEQUENCE_LENGTH),
        "--batch-size",
        "1",
        "--eval-batch-size",
        "1",
        "--train-steps",
        str(TRAIN_STEPS_2B),
        "--eval-interval",
        str(EVAL_INTERVAL),
        "--eval-loss-mode",
        "full",
        "--max-step-seconds",
        "600",
        "--max-eval-seconds",
        "1200",
        "--max-gpu-used-mb",
        "2000",
        "--learning-rate",
        str(learning_rate),
        "--variant-checkpoint-interval",
        "1000",
        "--milestone-checkpoint-interval",
        str(MILESTONE_STEP_600M),
        "--resume-variant-checkpoints",
        "--train-log-interval",
        "100",
        "--sampled-vocab-size",
        "16384",
        "--full-loss-interval",
        "4",
        "--full-eval-token-chunk-size",
        "1024",
        "--train-loss-token-chunk-size",
        "1024",
        "--conv-embedding-dim",
        str(conv_dim),
        "--conv-rank",
        str(conv_rank),
        "--variants",
        VARIANT,
        "--no-cache-dataset-on-device",
        "--no-pin-memory",
        "--output",
        str(output),
    ]
    if CACHE_2B.exists():
        args.extend(["--cache-path", str(CACHE_2B)])
    return args


def launch_training(run_dir: Path, output: Path, conv_dim: int, conv_rank: int, *, learning_rate: float) -> int:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "train_stdout.log").open("a", encoding="utf-8")
    stderr = (log_dir / "train_stderr.log").open("a", encoding="utf-8")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        common_args(run_dir, output, conv_dim, conv_rank, learning_rate=learning_rate),
        cwd=REPO_ROOT,
        stdout=stdout,
        stderr=stderr,
        creationflags=creationflags,
    )
    (log_dir / "train.pid").write_text(str(process.pid), encoding="utf-8")
    log(
        f"launched run={run_dir.name} pid={process.pid} conv_dim={conv_dim} "
        f"conv_rank={conv_rank} lr={learning_rate}"
    )
    return process.pid


def restart_if_needed(
    run_name: str,
    run_dir: Path,
    output: Path,
    conv_dim: int,
    conv_rank: int,
    *,
    learning_rate: float,
    state: dict[str, Any],
) -> None:
    if result_exists(run_dir):
        return
    if run_name == "160m" and state.get("160m_unstable"):
        log("restart_blocked_unstable run=160m")
        return
    pid = pid_from_file(run_dir)
    if process_exists(pid):
        return
    counts = state.setdefault("restart_counts", {})
    count = int(counts.get(run_name, 0))
    if count >= 3:
        log(f"restart_limit_reached run={run_name}")
        return
    used = gpu_used_mb()
    if used is not None and used > 2000:
        log(f"restart_wait_gpu run={run_name} gpu_used_mb={used:.0f}")
        return
    counts[run_name] = count + 1
    write_json(WATCHER_STATE, state)
    log(f"restarting run={run_name} attempt={counts[run_name]}")
    launch_training(run_dir, output, conv_dim, conv_rank, learning_rate=learning_rate)


def is_nonfinite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number != number or number in (float("inf"), float("-inf"))


def detect_primary_instability(state: dict[str, Any]) -> None:
    if state.get("primary_80m_unstable"):
        return
    payload = state_payload(RUN_80M)
    if payload is None:
        return
    step = int(payload.get("step", 0) or 0)
    latest_train = payload.get("latest_train_loss")
    latest_val = payload.get("latest_val_loss")
    if step < EVAL_INTERVAL and not is_nonfinite(latest_train):
        return
    if is_nonfinite(latest_train) or is_nonfinite(latest_val):
        state["primary_80m_unstable"] = True
        state["retry_80m_pending"] = True
        state["primary_80m_unstable_step"] = step
        state["primary_80m_unstable_train_loss"] = str(latest_train)
        state["primary_80m_unstable_val_loss"] = str(latest_val)
        write_json(WATCHER_STATE, state)
        log(
            "primary_80m_unstable "
            f"step={step} train={latest_train} val={latest_val} "
            "queued_retry=80m_lr1e3"
        )


def detect_160m_instability(state: dict[str, Any]) -> None:
    if state.get("160m_unstable") or not state.get("launched_160m"):
        return
    payload = state_payload(RUN_160M)
    if payload is None:
        return
    step = int(payload.get("step", 0) or 0)
    latest_train = payload.get("latest_train_loss")
    latest_val = payload.get("latest_val_loss")
    if is_nonfinite(latest_train) or is_nonfinite(latest_val):
        state["160m_unstable"] = True
        state["160m_unstable_step"] = step
        state["160m_unstable_train_loss"] = str(latest_train)
        state["160m_unstable_val_loss"] = str(latest_val)
        write_json(WATCHER_STATE, state)
        log(f"160m_unstable step={step} train={latest_train} val={latest_val} restart_blocked")


def scaling_run_dir(state: dict[str, Any]) -> Path:
    return RUN_80M_RETRY if state.get("primary_80m_unstable") else RUN_80M


def update_scaling_decision(state: dict[str, Any]) -> None:
    if state.get("scaling_decided"):
        return
    run_dir = scaling_run_dir(state)
    if run_dir == RUN_80M_RETRY and not state.get("retry_80m_launched"):
        return
    val = parse_eval_from_stdout(run_dir, MILESTONE_STEP_600M)
    payload = state_payload(run_dir)
    if val is None and payload is not None:
        try:
            step = int(payload.get("step", 0))
            latest_val = float(payload.get("latest_val_loss"))
        except (TypeError, ValueError):
            step = 0
            latest_val = float("nan")
        if step >= MILESTONE_STEP_600M:
            val = latest_val
    if val is None:
        return
    checkpoint = milestone_checkpoint(run_dir, MILESTONE_STEP_600M, MILESTONE_TOKENS_600M)
    if checkpoint is None:
        log(f"scaling_val_seen_waiting_checkpoint run={run_dir.name} val={val}")
        return
    scaling_holds = val == val and val < BASELINE_40M_600M_VAL
    state["scaling_decided"] = True
    state["scaling_holds"] = scaling_holds
    state["loss_80m_600m"] = val
    state["scaling_run"] = str(run_dir)
    state["baseline_40m_600m"] = BASELINE_40M_600M_VAL
    state["checkpoint_80m_600m"] = str(checkpoint)
    state["launch_160m_pending"] = scaling_holds
    write_json(WATCHER_STATE, state)
    log(
        "scaling_decision "
        f"run={run_dir.name} val_80m_600m={val} baseline_40m_600m={BASELINE_40M_600M_VAL:.6f} "
        f"holds={scaling_holds}"
    )


def maybe_launch_80m_retry(state: dict[str, Any]) -> None:
    if not state.get("retry_80m_pending") or state.get("retry_80m_launched"):
        return
    if result_exists(RUN_80M_RETRY) or process_exists(pid_from_file(RUN_80M_RETRY)):
        state["retry_80m_launched"] = True
        state["retry_80m_pending"] = False
        write_json(WATCHER_STATE, state)
        return
    if not CACHE_2B.exists():
        log("launch_80m_retry_wait_cache")
        return
    used = gpu_used_mb()
    if used is not None and used > 2000:
        log(f"launch_80m_retry_wait_gpu gpu_used_mb={used:.0f}")
        return
    launch_training(RUN_80M_RETRY, OUTPUT_80M_RETRY, conv_dim=1034, conv_rank=295, learning_rate=0.001)
    state["retry_80m_launched"] = True
    state["retry_80m_pending"] = False
    write_json(WATCHER_STATE, state)


def maybe_launch_160m(state: dict[str, Any]) -> None:
    if not state.get("launch_160m_pending") or state.get("launched_160m"):
        return
    if result_exists(RUN_160M) or process_exists(pid_from_file(RUN_160M)):
        state["launched_160m"] = True
        state["launch_160m_pending"] = False
        write_json(WATCHER_STATE, state)
        return
    if not CACHE_2B.exists():
        log("launch_160m_wait_cache")
        return
    used = gpu_used_mb()
    if used is not None and used > 2000:
        log(f"launch_160m_wait_gpu gpu_used_mb={used:.0f}")
        return
    launch_training(RUN_160M, OUTPUT_160M, conv_dim=1831, conv_rank=531, learning_rate=0.0008)
    state["launched_160m"] = True
    state["launch_160m_pending"] = False
    write_json(WATCHER_STATE, state)


def summarize(run_name: str, run_dir: Path) -> str:
    payload = state_payload(run_dir)
    pid = pid_from_file(run_dir)
    alive = process_exists(pid)
    if payload is None:
        tail = ""
        try:
            tail = stdout_path(run_dir).read_text(encoding="utf-8", errors="ignore").splitlines()[-1]
        except (OSError, IndexError):
            pass
        return f"{run_name}: pid_alive={alive} no_state tail={tail}"
    return (
        f"{run_name}: pid_alive={alive} status={payload.get('status')} step={payload.get('step')}/"
        f"{payload.get('train_steps')} tokens={payload.get('tokens_seen')} "
        f"train={payload.get('latest_train_loss')} val={payload.get('latest_val_loss')}"
    )


def main() -> None:
    log("watcher_start")
    while True:
        state = load_watcher_state()
        if not state.get("primary_80m_unstable"):
            restart_if_needed(
                "80m",
                RUN_80M,
                LANGUAGE_DIR / "language_longseq_anchor16_80m_2b_seq10160_seed13_20260603.json",
                conv_dim=1034,
                conv_rank=295,
                learning_rate=0.002,
                state=state,
            )
        detect_primary_instability(state)
        maybe_launch_80m_retry(state)
        if state.get("retry_80m_launched"):
            restart_if_needed(
                "80m_retry_lr1e3",
                RUN_80M_RETRY,
                OUTPUT_80M_RETRY,
                conv_dim=1034,
                conv_rank=295,
                learning_rate=0.001,
                state=state,
            )
        update_scaling_decision(state)
        maybe_launch_160m(state)
        detect_160m_instability(state)
        if state.get("launched_160m"):
            restart_if_needed(
                "160m",
                RUN_160M,
                OUTPUT_160M,
                conv_dim=1831,
                conv_rank=531,
                learning_rate=0.0008,
                state=state,
            )
        log(
            f"heartbeat {summarize('80m', RUN_80M)} | "
            f"{summarize('80m_retry', RUN_80M_RETRY)} | "
            f"{summarize('160m', RUN_160M)}"
        )
        append_progress("80m", RUN_80M)
        append_progress("80m_retry", RUN_80M_RETRY)
        append_progress("160m", RUN_160M)
        scaling_source_done = result_exists(scaling_run_dir(state))
        if result_exists(RUN_80M) and scaling_source_done and (not state.get("scaling_holds") or result_exists(RUN_160M)):
            log("watcher_complete")
            return
        time.sleep(60)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
