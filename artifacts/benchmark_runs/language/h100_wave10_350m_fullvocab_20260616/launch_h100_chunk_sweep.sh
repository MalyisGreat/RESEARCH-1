#!/usr/bin/env bash
set -uo pipefail

: "${CACHE_PATH:?Set CACHE_PATH to a GPT-2 FineWeb-Edu token cache on the node}"

RUN_ROOT="${RUN_ROOT:-$HOME/research1-runs/h100-chunk-sweep}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
CANDIDATE_IDS_PATH="${CANDIDATE_IDS_PATH:-${CACHE_PATH}.top32768.pt}"
SUMMARY="${RUN_ROOT}/chunk_sweep_${STAMP}.jsonl"

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${RUN_ROOT}"

run_arm() {
  local batch="$1"
  local chunk="$2"
  local name="wave10_350m_collapsed_compile_b${batch}_chunk${chunk}_${STAMP}"
  local out="${RUN_ROOT}/${name}"
  mkdir -p "${out}"
  python -u "${SCRIPT_DIR}/h100_wave10_fullvocab_train.py" \
    --cache-path "${CACHE_PATH}" --output-dir "${out}" --run-name "${name}" \
    --train-steps 24 --eval-interval 24 --checkpoint-interval 0 --milestone-checkpoint-interval 0 \
    --val-blocks 2 --batch-size "${batch}" --sequence-length 10160 \
    --embedding-dim 2304 --conv-layers 4 --conv-kernel-size 7 --conv-rank 1536 \
    --memory-rank 256 --memory-kernel-size 128 --sampled-vocab-size 32768 --token-stride 4 \
    --token-chunk-size "${chunk}" --full-eval-token-chunk-size 4096 \
    --candidate-ids-path "${CANDIDATE_IDS_PATH}" --learning-rate 0.0002 \
    --min-learning-rate 0.00002 --warmup-steps 2000 --weight-decay 0.0001 \
    --amp-dtype bf16 --log-interval 8 --timing-warmup-steps 8 \
    --collapsed-conv --compile --compile-mode reduce-overhead --loss-kernel torch --skip-checkpoints \
    2>&1 | tee "${out}/launcher.log"
  local status="${PIPESTATUS[0]}"
  python - "${out}/result.json" "${name}" "${batch}" "${chunk}" "${status}" >> "${SUMMARY}" <<'PY'
import json
import pathlib
import sys

result_path, name, batch, chunk, status = sys.argv[1:]
row = {"name": name, "batch_size": int(batch), "token_chunk_size": int(chunk), "exit": int(status)}
path = pathlib.Path(result_path)
if path.exists():
    row.update(json.loads(path.read_text())["report"])
print(json.dumps(row, sort_keys=True))
PY
  return 0
}

for chunk in 4096 8192 12288 16384; do
  run_arm 1 "${chunk}"
done
for chunk in 8192 16384; do
  run_arm 4 "${chunk}"
done
for chunk in 8192 16384; do
  run_arm 8 "${chunk}"
done

python - "${SUMMARY}" "${RUN_ROOT}/chunk_sweep_${STAMP}.json" <<'PY'
import json
import pathlib
import sys

source, destination = map(pathlib.Path, sys.argv[1:])
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
successful = sorted(
    (row for row in rows if row["exit"] == 0),
    key=lambda row: row.get("pure_train_tok_per_sec", 0),
    reverse=True,
)
destination.write_text(json.dumps({"arms": rows, "ranking": successful}, indent=2) + "\n")
if successful:
    winner = successful[0]
    print(
        f"WINNER batch={winner['batch_size']} chunk={winner['token_chunk_size']} "
        f"tok_s={winner['pure_train_tok_per_sec']:.0f} peak_mb={winner['peak_allocated_mb']:.0f}"
    )
PY
