from __future__ import annotations

import base64
import json
from pathlib import Path
import textwrap
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
RUN_NAME = "candidate_3080_40m_100m_seed13_factor_recall_scale256_common600m_20260617_0248"
REMOTE_ROOT = r"D:\CodexLLM\token_recall_100m_20260617"
REMOTE_CACHE = r"D:\CodexLLM\research1_longseq\cache\finewebedu_train600088338_val325152_seq10160_gpt2.pt"
CHUNK_SIZE = 12_000


def post_job(token: str, command: str) -> dict:
    body = {
        "command": command,
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


def chunk_command(chunk: str, index: int, count: int) -> str:
    reset = "Remove-Item -LiteralPath $b64Path -Force -ErrorAction SilentlyContinue" if index == 0 else ""
    return textwrap.dedent(
        f"""\
        $ErrorActionPreference = 'Stop'
        $ProgressPreference = 'SilentlyContinue'
        $root = '{REMOTE_ROOT}'
        New-Item -ItemType Directory -Force -Path $root | Out-Null
        $b64Path = Join-Path $root 'token_recall_train.py.b64'
        {reset}
        $chunk = @'
        {chunk}
        '@
        [IO.File]::AppendAllText($b64Path, $chunk, [Text.Encoding]::ASCII)
        Write-Host "TRAINER_CHUNK {index + 1}/{count} bytes=$($chunk.Length) file=$b64Path"
        """
    )


def run_command() -> str:
    return textwrap.dedent(
        f"""\
        $ErrorActionPreference = 'Stop'
        $ProgressPreference = 'SilentlyContinue'
        $env:PYTHONUNBUFFERED = '1'
        $env:CUDA_VISIBLE_DEVICES = '0'

        Write-Host "CANDIDATE_100M_START run={RUN_NAME} at $(Get-Date -Format o)"
        $gpu = (nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader | Select-Object -First 1).Trim()
        Write-Host "GPU=$gpu"
        if ($gpu -notmatch 'RTX 3080') {{ throw "refusing non-3080 GPU: $gpu" }}
        if ($gpu -match '4060') {{ throw "refusing 4060-class GPU: $gpu" }}

        $root = '{REMOTE_ROOT}'
        $b64Path = Join-Path $root 'token_recall_train.py.b64'
        $trainer = Join-Path $root 'token_recall_train.py'
        if (!(Test-Path -LiteralPath $b64Path)) {{ throw "missing trainer b64: $b64Path" }}
        $b64 = [IO.File]::ReadAllText($b64Path, [Text.Encoding]::ASCII)
        [IO.File]::WriteAllBytes($trainer, [Convert]::FromBase64String($b64))
        python -m py_compile $trainer

        $cache = '{REMOTE_CACHE}'
        if (!(Test-Path -LiteralPath $cache)) {{ throw "missing cache: $cache" }}

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
        if ($LASTEXITCODE -ne 0) {{ throw "candidate trainer failed with exit code $LASTEXITCODE" }}
        Write-Host "CANDIDATE_100M_DONE run={RUN_NAME} out=$out at $(Get-Date -Format o)"
        """
    )


def main() -> int:
    token = (HUB_ROOT / "data" / "hub-token.txt").read_text(encoding="utf-8").strip()
    trainer_b64 = base64.b64encode(TRAINER.read_bytes()).decode("ascii")
    chunks = [trainer_b64[i : i + CHUNK_SIZE] for i in range(0, len(trainer_b64), CHUNK_SIZE)]
    queued = []
    for index, chunk in enumerate(chunks):
        payload = post_job(token, chunk_command(chunk, index, len(chunks)))
        job = payload.get("job") or {}
        queued.append({"kind": "chunk", "index": index, "job_id": job.get("id"), "status": job.get("status")})
    payload = post_job(token, run_command())
    job = payload.get("job") or {}
    queued.append({"kind": "run", "job_id": job.get("id"), "status": job.get("status"), "run_name": RUN_NAME})
    out = OUT_ROOT / "queued_candidate_3080_chunked.json"
    out.write_text(json.dumps({"run_name": RUN_NAME, "jobs": queued}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_name": RUN_NAME, "jobs": queued, "queue_file": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
