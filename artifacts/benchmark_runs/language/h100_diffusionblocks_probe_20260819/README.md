# H100 DiffusionBlocks Probe

Date: 2026-08-19

This is a bounded feasibility test of the DiffusionBlocks training method on the
project's current `delta_router` low-rank conv-memory language architecture. It
is not a scale result and does not establish language-model likelihood parity.

## Protocol

- GPU: one NVIDIA H100 80 GB HBM3
- model: four blocks, width 640, rank 512, memory rank 128
- sequence length: 2,048
- batch size: 16
- seed: 13
- training objective: identical sampled-vocabulary plus sparse full-vocabulary anchors
- data: identical GPT-2-tokenized FineWeb-Edu cache and batch schedule
- normal arm: 64 end-to-end updates, 2,097,152 tokens
- DiffusionBlocks arm: 64 updates per block, 256 one-block updates, 8,388,608
  presented tokens
- compute matching: both arms execute 2,097,152 full-network-equivalent tokens

The diffusion arm follows the paper's EDM-style embedding denoising recipe:
L2-normalized target embeddings, log-normal noise with `P_mean=-1.2` and
`P_std=1.2`, sigma range `[0.002, 80]`, four equi-probability noise intervals
with 10% overlap, one trained block per update, and four-step Euler decoding.

## Results

| Metric | Normal AR | DiffusionBlocks | Difference |
|---|---:|---:|---:|
| Parameters | 70,437,078 | 72,244,438 | +1,807,360 (+2.57%) |
| Full-network-equivalent tokens | 2,097,152 | 2,097,152 | matched |
| Final training loss | 7.2765 | 7.4260 | +0.1495 |
| Validation cross-entropy | 7.3907 exact AR | 9.7104 approximate decode | +2.3196 |
| Effective full-network tok/s | 530,684 | 231,530 | 0.436x |
| Raw one-block tok/s | 530,684 | 926,119 | 1.745x |
| Peak allocated VRAM | 7,809.8 MB | 4,568.7 MB | -41.5% |
| Finite gradients | yes | yes | both stable |

The validation numbers are useful as a behavioral screen but are not identical
probabilistic quantities. The normal arm reports exact autoregressive
cross-entropy. The diffusion arm reports cross-entropy after a deterministic,
four-step, all-position embedding decode; DiffusionBlocks does not directly
define ordinary autoregressive perplexity.

## Conclusion

The mechanism works mechanically on this architecture: all four block-local
objectives train with finite gradients, four-step decoding runs, and peak
allocated VRAM falls substantially. The quality and compute-efficiency result is
negative. At matched block-token compute, the adapted model is 2.29x slower and
its decoded validation loss is much worse. Its samples are also qualitatively
worse, although neither 70M model is meaningfully trained after only 2.1M
full-network-equivalent tokens.

Do not scale this version. Its largest systems penalty is repeated execution of
the shared embedding and vocabulary head on every one-block update. Its likely
quality problem is a deeper mismatch: independent noisy target-embedding
denoisers do not preserve the progressive causal state transformation that the
delta-router stack learns end to end.

The paper's larger parameter-memory claim is not reproduced here because all
blocks remain resident on one H100. This probe measures real activation,
gradient, optimizer, and parameter memory for the implemented single-GPU
version.

## Artifacts

- `h100-diffusionblocks-probe/comparison_20260819_004551.json`: complete comparison
- `h100-diffusionblocks-probe/*/result.json`: arm-level configuration, samples, and results
- `h100-diffusionblocks-probe/*/metrics.jsonl`: per-interval training traces
- `h100-diffusionblocks-probe-results-20260819.tar.gz`: retrieved remote archive

Implementation and launch scripts live in
`artifacts/benchmark_runs/language/h100_wave10_350m_fullvocab_20260616/`.
