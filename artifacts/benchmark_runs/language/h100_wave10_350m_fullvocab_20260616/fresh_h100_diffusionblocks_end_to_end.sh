#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/dev/RESEARCH-1}"
PYTHON="${PYTHON:-/home/dev/research1-venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/research1-cache}"
RUN_ROOT="${RUN_ROOT:-/home/dev/research1-runs/h100-diffusionblocks-probe}"
CACHE_PATH="${CACHE_ROOT}/finewebedu_train4196352_val131136_seq2048_gpt2.pt"
CANDIDATE_IDS_PATH="${CACHE_PATH}.top8192.pt"
PROBE="${REPO}/artifacts/benchmark_runs/language/h100_wave10_350m_fullvocab_20260616/diffusion_blocks_h100_probe.py"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

export PYTHONPATH="${REPO}"
export HF_HOME="${HF_HOME:-/scratch/hf}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${CACHE_ROOT}" "${RUN_ROOT}"

"${PYTHON}" - <<'PY'
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

print({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "fla": str(chunk_gated_delta_rule)}, flush=True)
PY

if [[ ! -f "${CACHE_PATH}" ]]; then
  "${PYTHON}" -u "${REPO}/scripts/build_fineweb_cache.py" \
    --cache-path "${CACHE_PATH}" \
    --train-tokens 4196352 \
    --val-tokens 131136 \
    --sequence-length 2048 \
    --tokenization-batch-size 256
fi

if [[ ! -f "${CANDIDATE_IDS_PATH}" ]]; then
  CACHE_PATH="${CACHE_PATH}" CANDIDATE_IDS_PATH="${CANDIDATE_IDS_PATH}" "${PYTHON}" - <<'PY'
import os
import torch

payload = torch.load(os.environ["CACHE_PATH"], map_location="cpu", weights_only=False, mmap=True)
counts = torch.bincount(payload["train_tokens"].long(), minlength=50_257)
candidate_ids = counts.topk(8192).indices.long()
torch.save({"candidate_ids": candidate_ids}, os.environ["CANDIDATE_IDS_PATH"])
print({"candidate_ids": int(candidate_ids.numel())}, flush=True)
PY
fi

run_arm() {
  local mode="$1"
  "${PYTHON}" -u "${PROBE}" \
    --mode "${mode}" \
    --cache "${CACHE_PATH}" \
    --candidate-ids "${CANDIDATE_IDS_PATH}" \
    --output-root "${RUN_ROOT}" \
    --run-name "delta_router_compute_matched_${STAMP}" \
    --architecture delta_router \
    --seed 13 \
    --sequence-length 2048 \
    --batch-size 16 \
    --steps 64 \
    --num-blocks 4 \
    --dim 640 \
    --rank 512 \
    --memory-rank 128 \
    --sampled-vocab 8192 \
    --token-stride 32 \
    --token-chunk 4096 \
    --learning-rate 0.0003 \
    --min-learning-rate 0.00003 \
    --warmup-steps 8 \
    --overlap 0.1 \
    --val-blocks 2 \
    --sample-tokens 8 \
    --timing-warmup 4 \
    --log-interval 8
}

run_arm baseline
run_arm dblock

"${PYTHON}" - "${RUN_ROOT}" "${STAMP}" <<'PY'
import json
import pathlib
import sys

root, stamp = pathlib.Path(sys.argv[1]), sys.argv[2]
results = {}
for mode in ("baseline", "dblock"):
    path = root / f"{mode}_delta_router_compute_matched_{stamp}" / "result.json"
    results[mode] = json.loads(path.read_text())
base, dblock = results["baseline"], results["dblock"]
payload = {
    "protocol": "same seed, data schedule, architecture, and full-network-equivalent block-token compute",
    "results": results,
    "comparison": {
        "parameter_delta": dblock["parameter_count"] - base["parameter_count"],
        "peak_allocated_ratio": dblock["peak_allocated_mb"] / base["peak_allocated_mb"],
        "effective_speed_ratio": dblock["effective_full_network_tok_per_sec"] / base["effective_full_network_tok_per_sec"],
        "approx_validation_loss_delta": dblock["val_full_vocab_loss"] - base["val_full_vocab_loss"],
        "loss_warning": "Baseline is exact AR cross-entropy; DiffusionBlocks is approximate four-step decoded cross-entropy.",
    },
}
(root / f"comparison_{stamp}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload["comparison"], indent=2, sort_keys=True), flush=True)
PY
