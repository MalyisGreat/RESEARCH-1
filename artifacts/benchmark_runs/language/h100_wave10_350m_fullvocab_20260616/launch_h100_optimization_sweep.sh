#!/usr/bin/env bash
set -uo pipefail

: "${CACHE_PATH:?Set CACHE_PATH to a GPT-2 FineWeb-Edu token cache on the node}"

RUN_ROOT="${RUN_ROOT:-$HOME/research1-runs/h100-optimization}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
TRAIN_STEPS="${TRAIN_STEPS:-30}"
TIMING_WARMUP_STEPS="${TIMING_WARMUP_STEPS:-10}"
TOKEN_CHUNK_SIZE="${TOKEN_CHUNK_SIZE:-8192}"
CANDIDATE_IDS_PATH="${CANDIDATE_IDS_PATH:-${CACHE_PATH}.top32768.pt}"
SUMMARY="${RUN_ROOT}/sweep_${STAMP}.jsonl"

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_ROOT}"

run_arm() {
  local batch="$1"
  local compile_mode="$2"
  local loss_kernel="$3"
  local name="wave10_350m_collapsed_b${batch}_${compile_mode}_${loss_kernel}_${STAMP}"
  local out="${RUN_ROOT}/${name}"
  local compile_args=()
  if [[ "${compile_mode}" != "eager" ]]; then
    compile_args=(--compile --compile-mode "${compile_mode}")
  fi
  mkdir -p "${out}"
  printf 'START_ARM batch=%s mode=%s loss=%s\n' "${batch}" "${compile_mode}" "${loss_kernel}" | tee "${out}/launcher.log"
  python -u "${SCRIPT_DIR}/h100_wave10_fullvocab_train.py" \
    --cache-path "${CACHE_PATH}" \
    --output-dir "${out}" \
    --run-name "${name}" \
    --train-steps "${TRAIN_STEPS}" \
    --eval-interval "${TRAIN_STEPS}" \
    --checkpoint-interval 0 \
    --milestone-checkpoint-interval 0 \
    --val-blocks 2 \
    --batch-size "${batch}" \
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
    --full-eval-token-chunk-size 4096 \
    --candidate-ids-path "${CANDIDATE_IDS_PATH}" \
    --learning-rate 0.0002 \
    --min-learning-rate 0.00002 \
    --warmup-steps 2000 \
    --weight-decay 0.0001 \
    --amp-dtype bf16 \
    --log-interval 5 \
    --timing-warmup-steps "${TIMING_WARMUP_STEPS}" \
    --loss-kernel "${loss_kernel}" \
    --collapsed-conv \
    --skip-checkpoints \
    "${compile_args[@]}" 2>&1 | tee -a "${out}/launcher.log"
  local status="${PIPESTATUS[0]}"
  python - "${out}/result.json" "${name}" "${batch}" "${compile_mode}" "${loss_kernel}" "${status}" >> "${SUMMARY}" <<'PY'
import json
import pathlib
import sys

result_path, name, batch, mode, loss_kernel, status = sys.argv[1:]
row = {"name": name, "batch_size": int(batch), "mode": mode, "loss_kernel": loss_kernel, "exit": int(status)}
path = pathlib.Path(result_path)
if path.exists():
    row.update(json.loads(path.read_text())["report"])
print(json.dumps(row, sort_keys=True))
PY
  printf 'END_ARM batch=%s mode=%s loss=%s exit=%s\n' "${batch}" "${compile_mode}" "${loss_kernel}" "${status}" | tee -a "${out}/launcher.log"
  return 0
}

run_arm 1 eager torch
run_arm 1 reduce-overhead torch
run_arm 4 reduce-overhead torch
for batch in 1 2 4 8; do
  run_arm "${batch}" eager liger
done
run_arm 1 reduce-overhead liger

python - "${SUMMARY}" "${RUN_ROOT}/sweep_${STAMP}.json" <<'PY'
import json
import pathlib
import sys

source, destination = map(pathlib.Path, sys.argv[1:])
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
successful = [row for row in rows if row["exit"] == 0]
successful.sort(key=lambda row: row.get("pure_train_tok_per_sec", 0), reverse=True)
destination.write_text(json.dumps({"arms": rows, "ranking": successful}, indent=2) + "\n")
if successful:
    winner = successful[0]
    print(
        "WINNER "
        f"batch={winner['batch_size']} mode={winner['mode']} loss={winner['loss_kernel']} "
        f"tok_s={winner['pure_train_tok_per_sec']:.0f} "
        f"peak_mb={winner['peak_allocated_mb']:.0f}"
    )
PY
