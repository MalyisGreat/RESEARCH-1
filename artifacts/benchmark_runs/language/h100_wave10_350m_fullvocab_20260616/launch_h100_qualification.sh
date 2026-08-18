#!/usr/bin/env bash
set -uo pipefail

: "${CACHE_PATH:?Set CACHE_PATH to the uploaded 1B token cache}"

RUN_ROOT="${RUN_ROOT:-$HOME/research1-runs/h100-qualification}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-20}"
TOKEN_CHUNK_SIZE="${TOKEN_CHUNK_SIZE:-8192}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-4096}"
CANDIDATE_IDS_PATH="${CANDIDATE_IDS_PATH:-${CACHE_PATH}.top32768.pt}"

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_ROOT}"

run_arm() {
  local arm="$1"
  shift
  local name="wave10_350m_${arm}_${STAMP}"
  local out="${RUN_ROOT}/${name}"
  mkdir -p "${out}"
  printf 'START_ARM %s\n' "${arm}" | tee "${out}/launcher.log"
  python -u "${SCRIPT_DIR}/h100_wave10_fullvocab_train.py" \
    --cache-path "${CACHE_PATH}" \
    --output-dir "${out}" \
    --run-name "${name}" \
    --train-steps "${TRAIN_STEPS}" \
    --eval-interval "${TRAIN_STEPS}" \
    --checkpoint-interval 0 \
    --milestone-checkpoint-interval 0 \
    --val-blocks 2 \
    --batch-size "${BATCH_SIZE}" \
    --sequence-length 10160 \
    --embedding-dim 2304 \
    --conv-layers 4 \
    --conv-kernel-size 7 \
    --conv-rank 1536 \
    --memory-rank 256 \
    --memory-kernel-size 128 \
    --sampled-vocab-size 32768 \
    --token-stride 4 \
    --token-chunk-size "${TOKEN_CHUNK_SIZE}" \
    --full-eval-token-chunk-size "${EVAL_CHUNK_SIZE}" \
    --candidate-ids-path "${CANDIDATE_IDS_PATH}" \
    --learning-rate 0.0002 \
    --min-learning-rate 0.00002 \
    --warmup-steps 20 \
    --weight-decay 0.0001 \
    --amp-dtype bf16 \
    --log-interval 1 \
    --timing-warmup-steps 3 \
    --skip-checkpoints \
    "$@" 2>&1 | tee -a "${out}/launcher.log"
  local status="${PIPESTATUS[0]}"
  printf 'END_ARM %s exit=%s\n' "${arm}" "${status}" | tee -a "${out}/launcher.log"
  return "${status}"
}

baseline_status=0
collapsed_status=0
run_arm baseline || baseline_status=$?
run_arm collapsed --collapsed-conv || collapsed_status=$?

printf '{"baseline_exit":%s,"collapsed_exit":%s,"batch_size":%s,"train_steps":%s}\n' \
  "${baseline_status}" "${collapsed_status}" "${BATCH_SIZE}" "${TRAIN_STEPS}" \
  | tee "${RUN_ROOT}/qualification_${STAMP}.json"

if (( baseline_status != 0 || collapsed_status != 0 )); then
  exit 1
fi
