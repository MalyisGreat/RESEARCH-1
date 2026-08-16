#!/usr/bin/env bash
set -euo pipefail

: "${CACHE_PATH:?Set CACHE_PATH to the token cache .pt file}"

RUN_ROOT="${RUN_ROOT:-/workspace/research1/runs}"
RUN_NAME="${RUN_NAME:-wave10_350m_h100_fullvocab_smoke_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/${RUN_NAME}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_DIR}"

python -u "${SCRIPT_DIR}/h100_wave10_fullvocab_train.py" \
  --cache-path "${CACHE_PATH}" \
  --output-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --train-steps 20 \
  --eval-interval 10 \
  --checkpoint-interval 10 \
  --milestone-checkpoint-interval 0 \
  --val-blocks 2 \
  --batch-size "${BATCH_SIZE:-1}" \
  --sequence-length 10160 \
  --embedding-dim 2304 \
  --conv-layers 4 \
  --conv-kernel-size 7 \
  --conv-rank 1536 \
  --memory-rank 256 \
  --memory-kernel-size 128 \
  --sampled-vocab-size 32768 \
  --token-stride 4 \
  --token-chunk-size 1024 \
  --full-eval-token-chunk-size 1024 \
  --learning-rate 0.0002 \
  --min-learning-rate 0.00002 \
  --warmup-steps 20 \
  --weight-decay 0.0001 \
  --amp-dtype bf16 \
  --log-interval 1 \
  ${EXTRA_ARGS:-}
