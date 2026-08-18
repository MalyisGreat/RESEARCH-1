#!/usr/bin/env bash
set -euo pipefail

: "${CACHE_PATH:?Set CACHE_PATH to the node-local GPT-2 FineWeb-Edu cache}"

RUN_ROOT="${RUN_ROOT:-$HOME/research1-runs/h100-diffusionblocks-probe}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
CANDIDATE_IDS_PATH="${CANDIDATE_IDS_PATH:-${CACHE_PATH}.top32768.pt}"
SUMMARY="${RUN_ROOT}/comparison_${STAMP}.json"

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${RUN_ROOT}"

run_arm() {
  local mode="$1"
  local name="delta_router_100m_compute_matched_${STAMP}"
  local out="${RUN_ROOT}/${mode}_${name}"
  mkdir -p "${out}"
  python -u "${SCRIPT_DIR}/diffusion_blocks_h100_probe.py" \
    --mode "${mode}" \
    --cache "${CACHE_PATH}" \
    --candidate-ids "${CANDIDATE_IDS_PATH}" \
    --output-root "${RUN_ROOT}" \
    --run-name "${name}" \
    --architecture delta_router \
    --seed 13 \
    --sequence-length 2048 \
    --batch-size 16 \
    --steps 128 \
    --num-blocks 4 \
    --dim 640 \
    --rank 512 \
    --memory-rank 128 \
    --sampled-vocab 8192 \
    --token-stride 32 \
    --token-chunk 4096 \
    --learning-rate 0.0003 \
    --min-learning-rate 0.00003 \
    --warmup-steps 16 \
    --overlap 0.1 \
    --val-blocks 4 \
    --sample-tokens 16 \
    --timing-warmup 8 \
    --log-interval 16 2>&1 | tee "${out}/launcher.log"
}

run_arm baseline
run_arm dblock

python - "${RUN_ROOT}" "${STAMP}" "${SUMMARY}" <<'PY'
import json
import pathlib
import sys

root, stamp, destination = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
results = {}
for mode in ("baseline", "dblock"):
    path = root / f"{mode}_delta_router_100m_compute_matched_{stamp}" / "result.json"
    results[mode] = json.loads(path.read_text())

base = results["baseline"]
dblock = results["dblock"]
payload = {
    "protocol": "same seed, data schedule, architecture, and full-network-equivalent block-token compute",
    "results": results,
    "comparison": {
        "parameter_delta": dblock["parameter_count"] - base["parameter_count"],
        "peak_allocated_ratio": dblock["peak_allocated_mb"] / base["peak_allocated_mb"],
        "effective_speed_ratio": dblock["effective_full_network_tok_per_sec"] / base["effective_full_network_tok_per_sec"],
        "approx_validation_loss_delta": dblock["val_full_vocab_loss"] - base["val_full_vocab_loss"],
        "loss_warning": "The baseline is exact AR cross-entropy; DiffusionBlocks is approximate four-step decoded cross-entropy and is not a likelihood-equivalent metric.",
    },
}
destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload["comparison"], indent=2, sort_keys=True))
PY
