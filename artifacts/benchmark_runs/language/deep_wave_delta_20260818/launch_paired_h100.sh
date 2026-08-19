#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/RESEARCH-1}"
CACHE_PATH="${CACHE_PATH:?set CACHE_PATH to the token cache on the GPU node}"
RUN_ROOT="${RUN_ROOT:-${REPO}/artifacts/benchmark_runs/language/deep_wave_delta_20260818/runs}"
TOKENS="${TOKENS:-100000000}"
SEED="${SEED:-13}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
CANDIDATES="${RUN_ROOT}/candidate_ids_32768.pt"
COMMON=(
  --cache-path "${CACHE_PATH}"
  --target-tokens "${TOKENS}"
  --sequence-length 2048
  --batch-size 16
  --eval-interval 250
  --val-blocks 32
  --sampled-vocab-size 32768
  --token-stride 4
  --token-chunk-size 4096
  --full-eval-token-chunk-size 2048
  --learning-rate 3e-4
  --min-learning-rate 3e-5
  --warmup-steps 200
  --weight-decay 0.01
  --amp-dtype bf16
  --seed "${SEED}"
  --candidate-ids-path "${CANDIDATES}"
  --timing-warmup-steps 10
  --cache-mmap
  --save-final-checkpoint-only
  --final-weights-only
)

mkdir -p "${RUN_ROOT}"

python "${REPO}/artifacts/benchmark_runs/language/h100_wave10_350m_fullvocab_20260616/h100_wave10_fullvocab_train.py" \
  "${COMMON[@]}" \
  --output-dir "${RUN_ROOT}/control_delta_gain_350m_${STAMP}" \
  --run-name "control_delta_gain_350m_${STAMP}" \
  --embedding-dim 2304 --conv-layers 4 --conv-kernel-size 7 --conv-rank 1536 \
  --memory-rank 256 --memory-kernel-size 128 \
  --architecture delta_gain --collapsed-conv

python "${REPO}/artifacts/benchmark_runs/language/deep_wave_delta_20260818/deep_wave_delta_h100_train.py" \
  "${COMMON[@]}" \
  --output-dir "${RUN_ROOT}/deep_wave_delta_350m_${STAMP}" \
  --run-name "deep_wave_delta_350m_${STAMP}" \
  --preset 350m

