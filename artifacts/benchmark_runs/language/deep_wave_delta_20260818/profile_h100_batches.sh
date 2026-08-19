#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/RESEARCH-1}"
CACHE_PATH="${CACHE_PATH:?set CACHE_PATH to the token cache on the GPU node}"
RUN_ROOT="${RUN_ROOT:-${REPO}/artifacts/benchmark_runs/language/deep_wave_delta_20260818/profiles}"
CANDIDATES="${RUN_ROOT}/candidate_ids_32768.pt"
mkdir -p "${RUN_ROOT}"

for batch in 4 8 12 16; do
  python "${REPO}/artifacts/benchmark_runs/language/deep_wave_delta_20260818/deep_wave_delta_h100_train.py" \
    --cache-path "${CACHE_PATH}" \
    --output-dir "${RUN_ROOT}/b${batch}" \
    --run-name "deep_wave_delta_profile_b${batch}" \
    --preset 350m --train-steps 30 --sequence-length 2048 --batch-size "${batch}" \
    --eval-interval 30 --val-blocks 2 --sampled-vocab-size 32768 --token-stride 4 \
    --token-chunk-size 4096 --full-eval-token-chunk-size 2048 \
    --learning-rate 3e-4 --min-learning-rate 3e-5 --warmup-steps 5 \
    --candidate-ids-path "${CANDIDATES}" --timing-warmup-steps 10 \
    --amp-dtype bf16 --cache-mmap --skip-checkpoints
done

