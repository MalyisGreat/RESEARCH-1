from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
ROOT = Path(__file__).resolve().parent
CACHE = REPO / "artifacts/benchmark_runs/language/neuron_search_20260605/manual_self/real_cache_finewebedu_sample_seq10160_train192_val8_gpt2.pt"
SEARCH = REPO / "artifacts/benchmark_runs/language/training_system_search_20260712"
SEEDS = (13, 17, 23, 29, 31)
CONDITIONS = {
    "phrase23": (SEARCH / "phrase_induction_train.py", "2,3"),
    "phrase234": (SEARCH / "phrase_induction_train.py", "2,3,4"),
    "phrase23_semantic": (SEARCH / "phrase_semantic_induction_train.py", "2,3"),
}


def run(condition: str, trainer: Path, orders: str, seed: int) -> None:
    name = f"{condition}_seed{seed}"
    output = ROOT / name
    result = output / "result.json"
    if result.exists():
        print(f"SKIP {name}: result exists", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PHRASE_ORDERS"] = orders
    command = [
        sys.executable,
        "-u",
        str(trainer),
        "--cache-path", str(CACHE),
        "--output-dir", str(output),
        "--run-name", name,
        "--embedding-dim", "512",
        "--block-type", "multi_scale_lowrank_conv_memory",
        "--conv-layers", "2",
        "--conv-rank", "192",
        "--memory-rank", "64",
        "--batch-size", "1",
        "--train-steps", "1000",
        "--sampled-vocab-size", "4096",
        "--token-stride", "24",
        "--token-chunk-size", "20000",
        "--full-eval-token-chunk-size", "512",
        "--val-blocks", "8",
        "--seed", str(seed),
        "--eval-interval", "250",
        "--checkpoint-interval", "0",
        "--learning-rate", "0.0006",
        "--min-learning-rate", "0.00001",
        "--warmup-steps", "64",
    ]
    print(f"RUN {name}", flush=True)
    with (ROOT / f"{name}.console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        raise SystemExit(f"{name} failed with exit code {code}")


def main() -> None:
    for condition, (trainer, orders) in CONDITIONS.items():
        for seed in SEEDS:
            run(condition, trainer, orders, seed)


if __name__ == "__main__":
    main()
