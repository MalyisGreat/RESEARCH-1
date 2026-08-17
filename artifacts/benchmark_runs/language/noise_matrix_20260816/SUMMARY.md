# Five-Seed Paired Retrieval Matrix

Date: 2026-08-16  
GPU: NVIDIA GeForce RTX 2080 SUPER  
Protocol: seeds 13, 17, 23, 29, 31; same cache and schedule; 1,000 steps / 10.16M token exposures; full-vocabulary validation at 2.54M, 5.08M, 7.62M, and 10.16M tokens

## Endpoint Results

| Condition | Mean full-vocab loss | Across-seed SD | Mean tok/s | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| Phrase 2+3 | 6.500939 | 0.005116 | 216,066 | 1,558.7 MB |
| Phrase 2+3+4 | 6.495826 | 0.005119 | 204,609 | 1,558.8 MB |
| Phrase 2+3 + semantic | 6.494800 | 0.005246 | 199,409 | 1,572.6 MB |

Baseline endpoints range from 6.493363 to 6.506854. An isolated unpaired improvement near 0.005 is therefore not credible on this protocol. Paired comparisons are much more precise because each condition uses the identical seed and batch schedule.

| Paired change vs Phrase 2+3 | Mean loss change | 95% CI | Per-seed range |
| --- | ---: | ---: | ---: |
| Add order 4 | -0.005112 | [-0.005125, -0.005100] | -0.005127 to -0.005099 |
| Add semantic retrieval | -0.006138 | [-0.006362, -0.005915] | -0.006353 to -0.005866 |

Both validation effects persist at every measured checkpoint. Order 4 changes from -0.00538 at 2.54M tokens to -0.00511 at 10.16M. Semantic retrieval remains near -0.006 throughout.

## Critical Training Finding

These results do **not** show that phrase or semantic recall improves learning. The shared model tensors are bit-identical across all three conditions for a matched seed, all logged training losses are identical, and the recall parameters remain at initialization.

The active trainer sets `recall_mode="none"`. `model_cross_entropy_sum_chunked` therefore bypasses `candidate_logits` during training, while `evaluate_full_loss` and generation always call `candidate_logits`. Phrase order and semantic retrieval in this matrix are fixed inference-time logit corrections attached to the same trained base checkpoint.

This explains the extremely small paired variance. It is a deterministic evaluation ablation, not an independently trained architecture comparison. The results are still useful for measuring inference behavior, but must not be described as faster learning or better scaling.

## Repetition

Six prompts per checkpoint, 64 generated tokens, temperature 0.8, top-k 40.

| Condition | Repeat-1 | Repeat-2 | Repeat-3 | Repeat-4 |
| --- | ---: | ---: | ---: | ---: |
| Phrase 2+3 | 38.33% | 14.29% | 4.68% | 1.58% |
| Phrase 2+3+4 | 38.44% | 14.44% | 4.78% | 1.69% |
| Phrase 2+3 + semantic | 40.94% | 17.41% | 6.08% | 2.73% |

Order 4 is essentially neutral on repetition in this small sample. Semantic recall increases mean repetition at every order, but the five-seed paired confidence intervals remain wide because six prompts are insufficient: for example, Repeat-1 changes by +2.60 percentage points with a 95% CI of roughly -1.16 to +6.37 points. The direction is concerning and agrees with the earlier seed-13 screen, but a larger prompt suite is needed for a precise estimate.

## Throughput Caveat

The matrix ran conditions in blocks rather than randomized interleaving. Sustained-session throughput drifted downward, so the raw condition means overstate architectural overhead. They should not be used as clean speed estimates. Earlier adjacent seed-13 measurements placed order-4 overhead near 1.6% and always-on semantic overhead near 2.7%; a randomized short timing harness is needed for publication-quality throughput.

## Decision

1. Keep order 4 as an inexpensive inference ablation, not as proven training progress.
2. Do not scale always-on semantic logit retrieval; its small inference-loss gain comes with concerning repetition.
3. Fix the experimental contract before more comparisons: explicitly choose whether recall participates in training, record that choice in every result, and assert that intended new parameters receive finite nonzero gradients.
4. Prioritize dense hidden-state retrieval for the next fresh-data comparison. Unlike the direct-logit variants, it changes `features`, receives training gradients under the current trainer, and previously produced the strongest validation result.

Artifacts:

- `validation_curves.png`: mean validation curves with across-seed SD bands
- `matrix_metrics.csv`: per-seed endpoint metrics
- `matrix_summary.json`: paired differences and confidence intervals
- `repetition_matrix.json`: prompts, samples, and repetition metrics
- `run_matrix.py`: resumable exact training runner
