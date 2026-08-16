from __future__ import annotations

import base64
import json
from pathlib import Path
import urllib.request


ROOT = Path(r"E:\CODEXRESEARCH")
HUB_ROOT = ROOT / "house_compute_hub"
TRAINER = (
    ROOT
    / "RESEARCH-1"
    / "artifacts"
    / "benchmark_runs"
    / "language"
    / "token_recall_search_20260616"
    / "token_recall_train.py"
)
WORKER_ID = "mwstroud-mwstr-6aea1cf3"
RUN_NAME = "candidate_3080_40m_100m_seed13_factor_recall_scale256_common600m_20260617_0240"


def queue_job() -> dict:
    token = (HUB_ROOT / "data" / "hub-token.txt").read_text(encoding="utf-8").strip()
    trainer_b64 = base64.b64encode(TRAINER.read_bytes()).decode("ascii")

    remote_script = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUNBUFFERED = '1'
$env:CUDA_VISIBLE_DEVICES = '0'

Write-Host "CANDIDATE_100M_START run=__RUN_NAME__ at $(Get-Date -Format o)"
$gpu = (nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader | Select-Object -First 1).Trim()
Write-Host "GPU=$gpu"
if ($gpu -notmatch 'RTX 3080') {
  throw "refusing non-3080 GPU: $gpu"
}
if ($gpu -match '4060') {
  throw "refusing 4060-class GPU: $gpu"
}

$root = 'D:\CodexLLM\token_recall_100m_20260617'
New-Item -ItemType Directory -Force -Path $root | Out-Null
$trainer = Join-Path $root 'token_recall_train.py'
$b64 = @'
__TRAINER_B64__
'@
[IO.File]::WriteAllBytes($trainer, [Convert]::FromBase64String($b64))
python -m py_compile $trainer

$cache = 'D:\CodexLLM\research1_longseq\cache\finewebedu_train600088338_val325152_seq10160_gpt2.pt'
if (!(Test-Path -LiteralPath $cache)) {
  throw "missing cache: $cache"
}

$out = Join-Path $root '__RUN_NAME__'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out 'run.log'

$trainerArgs = @(
  $trainer,
  '--cache-path', $cache,
  '--output-dir', $out,
  '--run-name', '__RUN_NAME__',
  '--vocab-size', '50257',
  '--sequence-length', '10160',
  '--batch-size', '1',
  '--seed', '13',
  '--train-steps', '9843',
  '--eval-interval', '984',
  '--checkpoint-interval', '984',
  '--milestone-checkpoint-interval', '9843',
  '--val-blocks', '32',
  '--embedding-dim', '512',
  '--block-type', 'multi_scale_lowrank_conv_memory',
  '--conv-layers', '2',
  '--conv-kernel-size', '7',
  '--conv-rank', '192',
  '--memory-rank', '64',
  '--landmark-stride', '128',
  '--sampled-vocab-size', '16384',
  '--token-stride', '4',
  '--token-chunk-size', '20000',
  '--full-eval-token-chunk-size', '512',
  '--learning-rate', '0.0006',
  '--min-learning-rate', '0.00001',
  '--warmup-steps', '512',
  '--recall-mode', 'factor_recall_gated_multiscale',
  '--recall-initial-scale', '256'
)

& python -u @trainerArgs 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) {
  throw "candidate trainer failed with exit code $LASTEXITCODE"
}
Write-Host "CANDIDATE_100M_DONE run=__RUN_NAME__ out=$out at $(Get-Date -Format o)"
""".replace("__RUN_NAME__", RUN_NAME).replace("__TRAINER_B64__", trainer_b64)

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
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    payload = queue_job()
    job = payload.get("job") or {}
    assignment = (job.get("assignments") or {}).get(WORKER_ID) or {}
    print(
        json.dumps(
            {
                "ok": payload.get("ok"),
                "job_id": job.get("id"),
                "job_status": job.get("status"),
                "worker": WORKER_ID,
                "assignment_status": assignment.get("status"),
                "run_name": RUN_NAME,
            },
            indent=2,
            sort_keys=True,
        )
    )
