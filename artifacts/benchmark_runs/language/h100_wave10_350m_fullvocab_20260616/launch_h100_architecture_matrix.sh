#!/usr/bin/env bash
set -uo pipefail

: "${CACHE_PATH:?Set CACHE_PATH to the node-local GPT-2 FineWeb-Edu cache}"

RUN_ROOT="${RUN_ROOT:-$HOME/research1-runs/h100-architecture-matrix}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
CANDIDATE_IDS_PATH="${CANDIDATE_IDS_PATH:-${CACHE_PATH}.top32768.pt}"
SUMMARY="${RUN_ROOT}/matrix_${STAMP}.jsonl"

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${RUN_ROOT}"

run_arm() {
  local architecture="$1"
  local seed="$2"
  local name="${architecture}_350m_100m_seed${seed}_${STAMP}"
  local out="${RUN_ROOT}/${name}"
  local checkpoint_args=(--skip-checkpoints)
  if [[ "${seed}" == "13" ]]; then
    checkpoint_args=(--save-final-checkpoint-only --final-weights-only)
  fi
  mkdir -p "${out}"
  printf 'START_ARM architecture=%s seed=%s\n' "${architecture}" "${seed}" | tee "${out}/launcher.log"
  python -u "${SCRIPT_DIR}/h100_wave10_fullvocab_train.py" \
    --cache-path "${CACHE_PATH}" --output-dir "${out}" --run-name "${name}" \
    --train-steps 1230 --eval-interval 246 --checkpoint-interval 0 --milestone-checkpoint-interval 0 \
    --val-blocks 32 --batch-size 8 --sequence-length 10160 \
    --embedding-dim 2304 --conv-layers 4 --conv-kernel-size 7 --conv-rank 1536 \
    --memory-rank 256 --memory-kernel-size 128 --sampled-vocab-size 32768 --token-stride 4 \
    --token-chunk-size 16384 --full-eval-token-chunk-size 4096 \
    --candidate-ids-path "${CANDIDATE_IDS_PATH}" --learning-rate 0.0002 \
    --min-learning-rate 0.00002 --warmup-steps 100 --weight-decay 0.0001 \
    --amp-dtype bf16 --seed "${seed}" --log-interval 50 --timing-warmup-steps 15 \
    --collapsed-conv --architecture "${architecture}" --cache-mmap \
    --compile --compile-mode reduce-overhead --loss-kernel torch \
    "${checkpoint_args[@]}" 2>&1 | tee -a "${out}/launcher.log"
  local status="${PIPESTATUS[0]}"
  python - "${out}/result.json" "${name}" "${architecture}" "${seed}" "${status}" >> "${SUMMARY}" <<'PY'
import json
import pathlib
import sys

result_path, name, architecture, seed, status = sys.argv[1:]
row = {"name": name, "architecture": architecture, "seed": int(seed), "exit": int(status)}
path = pathlib.Path(result_path)
if path.exists():
    report = json.loads(path.read_text())["report"]
    row.update({key: report.get(key) for key in (
        "parameter_count", "train_tokens_seen", "final_train_loss",
        "final_val_loss_full_vocab", "pure_train_tok_per_sec", "wall_tok_per_sec",
        "peak_allocated_mb", "peak_reserved_mb", "cache_load_seconds", "history",
    )})
print(json.dumps(row, sort_keys=True))
PY
  printf 'END_ARM architecture=%s seed=%s exit=%s\n' "${architecture}" "${seed}" "${status}" | tee -a "${out}/launcher.log"
  return 0
}

for seed in 13 17 23; do
  for architecture in wave delta_gain delta_router; do
    run_arm "${architecture}" "${seed}"
  done
done

python - "${SUMMARY}" "${RUN_ROOT}/matrix_${STAMP}.json" <<'PY'
import json
import pathlib
import statistics
import sys

source, destination = map(pathlib.Path, sys.argv[1:])
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
successful = [row for row in rows if row["exit"] == 0]
groups = {}
for architecture in ("wave", "delta_gain", "delta_router"):
    arms = [row for row in successful if row["architecture"] == architecture]
    if not arms:
        continue
    groups[architecture] = {
        "runs": len(arms),
        "val_loss_mean": statistics.mean(row["final_val_loss_full_vocab"] for row in arms),
        "val_loss_stdev": statistics.stdev(row["final_val_loss_full_vocab"] for row in arms) if len(arms) > 1 else 0.0,
        "tok_per_sec_mean": statistics.mean(row["pure_train_tok_per_sec"] for row in arms),
        "peak_allocated_mb_mean": statistics.mean(row["peak_allocated_mb"] for row in arms),
        "parameter_count": arms[0]["parameter_count"],
    }
ranking = sorted(groups, key=lambda name: groups[name]["val_loss_mean"])
payload = {"arms": rows, "architectures": groups, "quality_ranking": ranking}
destination.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"quality_ranking": ranking, "architectures": groups}, indent=2))
PY
