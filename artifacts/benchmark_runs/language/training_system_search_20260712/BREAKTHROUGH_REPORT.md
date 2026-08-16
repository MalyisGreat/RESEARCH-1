# Verified Breakthrough: Sorted Causal Induction Wave10

## Result

The winning system satisfies the declared breakthrough criterion on an RTX 2080
SUPER:

- controlled raw training speedup: **1.5292x**
- seed-13 full-vocabulary validation: **6.790970** candidate vs **6.797395** reference
- seed-17 full-vocabulary validation: **6.794833** candidate vs **6.818093** reference
- peak VRAM: **1.62 GB** candidate vs **3.05 GB** reference
- finite training on every completed screen
- exact-induction top-1: **47.38%** candidate vs **30.17%** reference

This is not an activation swap. It combines an exact convolution
reparameterization with a new causal token-addressable primitive and uses the
resulting quality margin to reduce expensive full-vocabulary anchors.

## Algorithm

For token sequence `x[0:n]`, construct keys `(batch_id, token_id)` and perform a
stable GPU sort. Stability keeps equal-token positions in causal order. For each
position `t`, the previous item in its equal-key run is the most recent prior
occurrence `p(t)` of the same token. The retrieved value is:

`r[t] = x[p(t) + 1]`

when a prior occurrence exists. This is exactly the token that followed the
previous matching key, the classic induction/copy operation. A learned scalar
and rank-conditioned gate add one sparse update to logit `r[t]`:

`logits[t, r[t]] += softplus(scale) * sigmoid(W_g h[t] + b_g)`

The index is causal, collision-free, and derived only from input tokens at or
before `t`. Construction is `O(n log n)` through one stable sort; retrieval and
logit updates are `O(n)`. It applies to both sampled-vocabulary and
full-vocabulary loss branches.

## Winning system configuration

- base family: Wave10 / `multi_scale_lowrank_conv_memory`
- exact collapsed multiscale depthwise convolution
- embedding dimension: 512
- layers: 2
- convolution/factor rank: 192
- memory rank: 64
- global head: normalized identity
- factor recall: gated multiscale
- sampled vocabulary: 4,096
- rotating full-vocabulary anchor stride: 24
- learning rate: 6e-4 cosine to 1e-5, 64-step warmup
- batch/sequence: 1 x 10,160

## Controlled timing

The alternating-order paired benchmark includes forward, backward, gradient
unscaling/clipping, and fused AdamW:

- reference aggregate step: 102.7447 ms
- candidate aggregate step: 67.1866 ms
- speedup: 1.529244x

Artifact: `paired_sorted_induction_stride24_result.json`.

## Longer equal-wall confirmation

The controlled speed ratio maps 654 reference steps to approximately 1,000
candidate steps.

| System | Steps | Tokens | Train seconds | Full-vocab val | Tok/s |
|---|---:|---:|---:|---:|---:|
| Reference | 654 | 6,644,640 | 66.37 | 6.503624 | 100,114 |
| Candidate | 1,000 | 10,160,000 | 62.89 | 6.477781 | 161,560 |

The candidate processes 52.9% more tokens in 5.2% less measured training time
and finishes with better validation loss.

## Behavioral audit

Across all eight validation blocks:

| Metric | Reference | Candidate |
|---|---:|---:|
| Full-vocab loss | 6.797395 | 6.790970 |
| Overall top-1 | 13.15% | 13.52% |
| Exact induction opportunities | 8,769 | 8,769 |
| Induction-opportunity top-1 | 30.17% | 47.38% |

Generation from these 3.5M-token models is highly repetitive for both systems.
The candidate did not materially regress repetition or diversity, but neither
checkpoint is large enough to support useful prose-generation claims.

## Primary command

```powershell
python -u E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\training_system_search_20260712\sorted_induction_train.py `
  --cache-path E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\real_cache_finewebedu_sample_seq10160_train192_val8_gpt2.pt `
  --output-dir <unique-output-directory> `
  --run-name <unique-run-name> `
  --embedding-dim 512 --block-type multi_scale_lowrank_conv_memory `
  --conv-layers 2 --conv-rank 192 --memory-rank 64 `
  --recall-mode factor_recall_gated_multiscale --recall-initial-scale 256 `
  --train-steps 1000 --sampled-vocab-size 4096 --token-stride 24 `
  --token-chunk-size 20000 --full-eval-token-chunk-size 512 `
  --val-blocks 8 --seed 13 --eval-interval 1000 --checkpoint-interval 0 `
  --learning-rate 0.0006 --min-learning-rate 0.00001 --warmup-steps 64 `
  --weight-decay 0.0001 --amp-dtype fp16
```

## Artifacts

- implementation: `sorted_induction_train.py`
- exact convolution implementation: `fused_wave10_probe.py`
- controlled benchmark: `paired_system_benchmark.py`
- behavioral evaluator: `evaluate_breakthrough_behavior.py`
- seed-13 screen: `sorted_induction_identity_rank192_stride24_vocab4096_350_seed13/`
- seed-17 screen: `sorted_induction_identity_rank192_stride24_vocab4096_350_seed17/`
- longer candidate: `longconfirm_candidate_1000_seed13/`
- longer reference: `longconfirm_reference_654_seed13/`
- eight-block behavior: `breakthrough_behavior_full8_seed13.json`

## Limits and next scale test

This proves a local training-system and induction improvement, not broad model
superiority at billions of tokens. The next justified experiment is a matched
40M-80M parameter run long enough for generation quality to emerge, retaining a
plain Wave10 control and full-vocabulary validation. The sparse induction scale
and gate should be logged separately to detect over-reliance or repetition at
scale.
