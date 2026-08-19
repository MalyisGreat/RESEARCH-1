# Deep Wave-Delta 350M-Class, 100M-Token H100 Run

Date: 2026-08-18/19

## Result

- Architecture: Deep Wave-Delta
- Parameters: 351,677,889
- Training tokens: 100,024,320
- Exact full-vocabulary validation loss: 4.233846
- Final sampled training loss: 4.420129
- Pure training throughput: 43,927.7 tokens/second
- Peak allocated VRAM: 70,456 MiB
- Peak reserved VRAM: 76,574 MiB
- GPU: NVIDIA H100 80GB HBM3
- Seed: 13
- Total mission cost, including setup/tests/data/training/export: $5.51226

The full-vocabulary validation curve improved at every evaluation anchor, from
6.312491 at 6.144M tokens to 4.233846 at 100.024M tokens. Gradients remained
finite throughout the run.

## Training Configuration

- Sequence length: 2,048
- Batch size: 12
- Optimizer steps: 4,070
- Precision: BF16
- Sampled training vocabulary: 32,768
- Exact evaluation vocabulary: 50,257
- Learning rate: 200-step linear warmup to 3e-4, cosine decay to 3e-5
- Weight decay: 0.01
- Evaluation interval: 250 steps
- Validation blocks per evaluation: 32

The cache was built directly on the node from streamed FineWeb-Edu data. It
contains 102,450,000 training tokens and 1,049,088 validation tokens at sequence
length 2,048. The training run consumed the first 100,024,320 training tokens.

## Qualification

- Local tests: 7 passed, 1 CUDA-only test skipped.
- H100 tests: 8 passed, including fused FLA delta recurrence versus the
  reference recurrence and a 257-token BF16 autocast regression.
- 351.7M-parameter CUDA smoke test: finite forward/backward gradients.
- Batch 4: 36,930 tok/s, 27,399 MiB peak allocated.
- Batch 8: 42,352 tok/s, 49,309 MiB peak allocated.
- Batch 12: 44,483 tok/s, 70,427 MiB peak allocated.
- `torch.compile` with max autotuning was rejected after more than eight minutes
  without completing its first step. The production run used the proven eager
  path.

## Prior Matrix Comparison

This run finished substantially below the earlier 350M-class, 100M-token matrix:

| Run | Seed(s) | Final exact loss | Throughput | Peak VRAM |
| --- | --- | ---: | ---: | ---: |
| Deep Wave-Delta | 13 | 4.233846 | 43,928 tok/s | 70,456 MiB |
| Delta + gain | 13 | 4.862141 | 143,128 tok/s | 59,702 MiB |
| Delta + gain | 3-seed mean | 4.851118 | 142,497 tok/s | 59,677 MiB |
| Delta router | 3-seed mean | 4.843453 | 142,336 tok/s | 59,676 MiB |
| Wave control | 3-seed mean | 5.016933 | 145,647 tok/s | 56,906 MiB |

This is descriptive, not an architecture-isolated comparison. Deep Wave-Delta
used sequence length 2,048 and 4,070 optimizer updates. The earlier matrix used
sequence length 10,160 and 1,230 updates, a different cache split, and a lower
peak learning rate. A matched control at sequence length 2,048 is required to
separate the architecture gain from the update-count and schedule gain.

## Artifacts

- Local checkpoint: `deep_wave_delta_350m_100m_checkpoint.pt`
- Checkpoint size: 1,406,863,447 bytes
- Checkpoint SHA-256: `4d818d437af8a7b4a978856c688107e3a8331bc3a4271a03e8f3d763f3b4f8a8`
- GiveMeANode checkpoint artifact: `art-zvt77`
- GiveMeANode evidence artifact: `art-3nekc`
- Mission: <https://givemeanode.com/missions/deep-wave-delta-100m>
- Plot: `loss_comparison_100m.png`

The H100 node was stopped after both artifacts were exported. The checkpoint was
downloaded locally and its SHA-256 matched the export receipt.

## Production Command

```bash
CUDA_VISIBLE_DEVICES=0 /home/dev/research1-venv/bin/python -u \
  artifacts/benchmark_runs/language/deep_wave_delta_20260818/deep_wave_delta_h100_train.py \
  --cache-path /scratch/research1-cache/finewebedu_train102450000_val1049088_seq2048_gpt2.pt \
  --output-dir /scratch/deep_wave_delta_350m_100m_seed13 \
  --run-name deep_wave_delta_350m_100m_seed13 \
  --preset 350m --target-tokens 100000000 \
  --sequence-length 2048 --batch-size 12 \
  --eval-interval 250 --val-blocks 32 \
  --sampled-vocab-size 32768 --token-stride 4 \
  --learning-rate 0.0003 --min-learning-rate 0.00003 \
  --warmup-steps 200 --weight-decay 0.01 --seed 13 \
  --amp-dtype bf16 \
  --candidate-ids-path /scratch/research1-cache/candidate_ids_top32768.pt \
  --timing-warmup-steps 10 --cache-on-device \
  --save-final-checkpoint-only --final-weights-only
```
