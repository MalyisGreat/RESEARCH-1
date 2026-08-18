# H100 Architecture and Systems Study

This directory preserves the no-checkpoint results from the 2026-08-18 H100 study. The study compared three approximately 350M-parameter long-sequence architectures under a paired, three-seed protocol after optimizing the training path for an H100 80GB.

## Result

The delta-rule memory is the clear early-training winner. The learned write router adds a smaller, consistent improvement.

| Architecture | Parameters | Full-vocab val loss | Perplexity | Paired vs Wave | Wins | Pure tok/s | Peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Delta + learned write router | 349,401,942 | 4.843453 +/- 0.006759 | 126.91 | -0.173480 | 3/3 | 142,336 | 59,676 MB |
| Delta + read gain | 349,397,332 | 4.851118 +/- 0.011852 | 127.88 | -0.165815 | 3/3 | 142,497 | 59,677 MB |
| Collapsed Wave10 control | 350,307,921 | 5.016933 +/- 0.001793 | 150.95 | control | - | 145,647 | 56,906 MB |

At equal training tokens, delta-router reduced perplexity by 15.93% relative to Wave. It was 2.27% slower in pure training throughput and used 2.77GB more peak allocated VRAM. The router beat fixed-write delta in all three seeds, but its mean loss advantage was only 0.007665.

## Protocol

- GPU: NVIDIA H100 80GB HBM3
- Sequence length: 10,160
- Batch size: 8 sequences, 81,280 tokens per optimizer step
- Run length: 1,230 steps, 99,974,400 tokens per run
- Total paired architecture training: 899,769,600 tokens
- Seeds: 13, 17, 23
- Optimizer schedule: warmup to 2e-4 over 100 steps, cosine decay to 2e-5
- Validation: full GPT-2 vocabulary on all 32 held-out validation blocks every 246 steps
- Sampled training vocabulary and data order: fixed within each paired seed

The Wave endpoint was unusually repeatable: 5.0189995, 5.0157894, and 5.0160092. Delta-router endpoints were 4.8490197, 4.8454074, and 4.8359316. This makes the main delta-memory effect much larger than measured run-to-run noise in this screen.

## Architecture Definitions

`wave` is the optimized Wave10 control with the exactly collapsed multi-branch depthwise convolution.

`delta_gain` replaces one Wave memory block with a two-head low-rank gated delta-rule state. It uses learned queries, keys, values, retention, and a positive learned token-wise read gain with a fixed write strength.

`delta_router` adds a learned token-wise write gate to `delta_gain`. This is the best loss result, although the extra gain over fixed-write delta is modest.

## Systems Findings

- Exact convolution collapse improved the original eager batch-1 path from about 92.1K to 110.1K tok/s without changing validation loss.
- Compiling the actual feature projector, rather than the previously ineffective compile path, raised batch-1 throughput to about 129.2K tok/s.
- Batch 8 with a 16,384-token loss chunk sustained 145.6K tok/s for Wave and about 142.3K tok/s for delta-router.
- Memory-mapped host cache startup took about 5ms. A directly built 100M-token cloud cache was produced in 63.1 seconds.
- Caching the full dataset on the GPU was rejected: it approached the memory limit and triggered an autotuning allocation failure for only a small speed gain.
- Liger fused linear cross entropy was rejected because it reduced batch-1 throughput to about 82K tok/s.
- `max-autotune` was rejected after excessive compile/autotune cost; vendor kernels won the important shapes.
- Profiling showed the sustained run was GPU-compute-bound, not CPU or data-loader bound: matrix multiplies were about 45% of CUDA time, vocabulary loss at least 23%, compiled feature projection about 16%, and depthwise convolution about 13%.

## Behavioral Check

The three seed-13 checkpoints were reconstructed strictly and sampled with the same 12 prompts, RNG, temperature, top-k, and 48-token budget.

| Architecture | Unique token ratio | Repeated 4-gram fraction | Max token run | Mean entropy | Password recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wave | 0.7951 | 0.0000 | 1.0000 | 5.0966 | 0 |
| Delta + read gain | 0.7361 | 0.0000 | 1.0833 | 4.8968 | 0 |
| Delta + router | 0.7309 | 0.0037 | 1.0000 | 4.8218 | 0 |

None of the checkpoints entered a catastrophic loop, but both delta variants were less diverse and more confident. All three models remained substantially undertrained: completions were grammatical fragments but generally off-topic, failed simple factual and code prompts, and failed exact password recall. Therefore this study establishes a strong early optimization advantage, not assistant quality or billion-token scaling.

## Decision

`delta_router` is the best measured architecture of the three. For the next scaling run, keep `delta_gain` as the lower-complexity control because the router's extra loss gain is small and the behavioral screen shows slightly more repetition and lower entropy.

The next decisive experiment is a checkpointed 1B-token paired run of delta-router and delta-gain, with full-vocabulary validation and the same behavior suite at intermediate checkpoints. Wave no longer needs another equal-budget run unless used as a sparse scaling reference.

## Artifacts

The extracted directory `h100-architecture-matrix/` contains:

- `ARCHITECTURE_MATRIX_SUMMARY.md` and `.json`
- `architecture_validation_curves.png` and `architecture_validation_curves.csv`
- `BEHAVIOR_EVALUATION.json`, including every prompt and completion
- per-run `result.json`, `metrics.csv`, `metrics.jsonl`, `run_meta.json`, and `state.json`
- the matrix launcher log and raw matrix summary

No model checkpoints are included.
