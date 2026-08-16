#!/usr/bin/env bash
set -euo pipefail

# Required: point this at a GPT-2-tokenized cache containing train_tokens and val_tokens.
# The cache must use vocab_size=50257 and sequence_length+1 contiguous blocks.
: "${CACHE_PATH:?Set CACHE_PATH to the token cache .pt file}"

RUN_ROOT="${RUN_ROOT:-/workspace/research1/runs}"
RUN_NAME="${RUN_NAME:-wave10_350m_h100_fullvocab_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/${RUN_NAME}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${OUT_DIR}/torchinductor_cache}"

mkdir -p "${OUT_DIR}"

python -u "${SCRIPT_DIR}/h100_wave10_fullvocab_train.py" \
  --cache-path "${CACHE_PATH}" \
  --output-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --target-tokens 5000000000 \
  --eval-interval 2500 \
  --checkpoint-interval 2500 \
  --milestone-checkpoint-interval 25000 \
  --val-blocks 64 \
  --batch-size "${BATCH_SIZE:-4}" \
  --sequence-length 10160 \
  --embedding-dim 2304 \
  --conv-layers 4 \
  --conv-kernel-size 7 \
  --conv-rank 1536 \
  --memory-rank 256 \
  --memory-kernel-size 128 \
  --sampled-vocab-size 32768 \
  --token-stride 4 \
  --token-chunk-size 2048 \
  --full-eval-token-chunk-size 2048 \
  --learning-rate 0.0002 \
  --min-learning-rate 0.00002 \
  --warmup-steps 2000 \
  --weight-decay 0.0001 \
  --amp-dtype bf16 \
  --log-interval 100 \
  ${EXTRA_ARGS:-}
