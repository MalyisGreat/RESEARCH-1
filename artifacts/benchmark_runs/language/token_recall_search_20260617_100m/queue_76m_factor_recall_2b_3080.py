from __future__ import annotations

import json
from pathlib import Path
import urllib.request


ROOT = Path(r"E:\CODEXRESEARCH")
HUB_ROOT = ROOT / "house_compute_hub"
OUT_ROOT = (
    ROOT
    / "RESEARCH-1"
    / "artifacts"
    / "benchmark_runs"
    / "language"
    / "token_recall_search_20260617_100m"
)
WORKER_ID = "mwstroud-mwstr-6aea1cf3"
RUN_NAME = "factor_recall_3080_76m_2b_seed13_common3b_20260617_0310"


def main() -> int:
    token = (HUB_ROOT / "data" / "hub-token.txt").read_text(encoding="utf-8").strip()
    remote_script = rf"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUNBUFFERED = '1'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

Write-Host "FACTOR_RECALL_76M_2B_START run={RUN_NAME} at $(Get-Date -Format o)"
$gpu = (nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader | Select-Object -First 1).Trim()
Write-Host "GPU=$gpu"
if ($gpu -notmatch 'RTX 3080') {{ throw "refusing non-3080 GPU: $gpu" }}
if ($gpu -match '4060') {{ throw "refusing 4060-class GPU: $gpu" }}

$root = 'D:\CodexLLM\token_recall_76m_2b_20260617'
New-Item -ItemType Directory -Force -Path $root | Out-Null
$trainer = 'D:\CodexLLM\token_recall_100m_20260617\token_recall_train.py'
if (!(Test-Path -LiteralPath $trainer)) {{ throw "missing transferred trainer: $trainer" }}
python -m py_compile $trainer

$cache = 'D:\CodexLLM\research1_longseq\cache\finewebedu_fresh_after2b_train3000289275_val325152_seq10160_gpt2.pt'
if (!(Test-Path -LiteralPath $cache)) {{ throw "missing 3B-token cache: $cache" }}

$out = Join-Path $root '{RUN_NAME}'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out 'run.log'

$trainerArgs = @(
  $trainer,
  '--cache-path', $cache,
  '--output-dir', $out,
  '--run-name', '{RUN_NAME}',
  '--vocab-size', '50257',
  '--sequence-length', '10160',
  '--batch-size', '1',
  '--seed', '13',
  '--train-steps', '196851',
  '--eval-interval', '9843',
  '--checkpoint-interval', '9843',
  '--milestone-checkpoint-interval', '49213',
  '--val-blocks', '32',
  '--embedding-dim', '896',
  '--block-type', 'multi_scale_lowrank_conv_memory',
  '--conv-layers', '2',
  '--conv-kernel-size', '7',
  '--conv-rank', '320',
  '--memory-rank', '64',
  '--landmark-stride', '128',
  '--sampled-vocab-size', '16384',
  '--token-stride', '4',
  '--token-chunk-size', '8192',
  '--full-eval-token-chunk-size', '256',
  '--learning-rate', '0.0006',
  '--min-learning-rate', '0.00006',
  '--warmup-steps', '2000',
  '--recall-mode', 'factor_recall_gated_multiscale',
  '--recall-initial-scale', '256'
)

& python -u @trainerArgs 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) {{ throw "76M factor-recall trainer failed with exit code $LASTEXITCODE" }}
Write-Host "FACTOR_RECALL_76M_2B_DONE run={RUN_NAME} out=$out at $(Get-Date -Format o)"
"""
    body = {
        "command": remote_script,
        "cwd": r"C:\Users\mwstr\CodexHouseWorker",
        "target_workers": [WORKER_ID],
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8787/api/job",
        data=json.dumps(body).encode("utf-8"),
        headers={"X-Hub-Token": token, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    job = payload.get("job") or {}
    assignment = (job.get("assignments") or {}).get(WORKER_ID) or {}
    summary = {
        "ok": payload.get("ok"),
        "job_id": job.get("id"),
        "job_status": job.get("status"),
        "assignment_status": assignment.get("status"),
        "worker": WORKER_ID,
        "run_name": RUN_NAME,
        "remote_output_dir": rf"D:\CodexLLM\token_recall_76m_2b_20260617\{RUN_NAME}",
        "tokens_target": 2_000_006_160,
        "train_steps": 196_851,
        "param_config": {
            "embedding_dim": 896,
            "conv_rank": 320,
            "memory_rank": 64,
            "conv_layers": 2,
            "expected_params": 76_219_801,
        },
    }
    out = OUT_ROOT / "queued_76m_factor_recall_2b_3080.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "queue_file": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
