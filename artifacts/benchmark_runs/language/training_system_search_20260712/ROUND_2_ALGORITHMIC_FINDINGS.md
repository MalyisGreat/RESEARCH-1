# Training System Search: Algorithmic Round 2

## Direction change

This round stopped treating small trainer changes as the research objective. The
search moved to structured transforms and hierarchical state computation drawn
from signal processing and multigrid methods.

Relevant external starting points:

- Shi, Wang, and Fox, *Sequence Modeling with Multiresolution Convolutional
  Memory* (ICML 2023).
- Huynh, Maire, and Walter, *Multigrid Neural Memory* (ICML 2020).
- Shaj et al., *Kalman Linear Attention* (2026), using information-form
  Bayesian filtering and associative scans.

These are inspiration and prior art, not claims that the local variants
replicate those papers.

## Stronger engineering baseline

The exact collapsed Wave10 convolution, rotating sparse full-vocabulary anchor,
4,096-token sampled head, identity global head, and rank-192 blocks produced:

- seed 13, 400 steps: full-vocab val 6.785382, 154,812 tok/s, 1.79 GB peak
- reference seed 13, 256 steps: full-vocab val 6.797395, 97,007 tok/s

An alternating-order, warmed, same-process benchmark is more conservative and
authoritative for raw kernel speed:

- reference aggregate step: 90.8911 ms
- candidate aggregate step: 60.9682 ms
- speedup: 1.49079x

Therefore this remains just below the declared 1.5x breakthrough gate. It is a
substantially better engineering baseline, not the desired novel algorithm.

## Spectral channel FFN

Design: replace dense block FFNs with two learned circulant channel operators,
implemented as FFT, learned complex diagonal, inverse FFT, squared ReLU, then a
second learned spectral operator.

Probe result:

- dense block step: 18.3691 ms
- spectral block step: 19.0282 ms
- speedup: 0.965x
- gradients and outputs finite

Decision: kill on this Windows/RTX 2080 stack. Float32 complex transforms and
unfused cuFFT launches erase the asymptotic advantage. Preserve the prototype
for a future fused CUDA/Triton environment.

## Causal multigrid memory

Design: replace the single 128-tap low-rank memory filter with completed causal
block summaries at resolutions 8, 32, 128, and 512. A token-conditioned softmax
gate selects levels, while learned rank-wise scales transform each level.

Block probe:

- collapsed Wave10: 15.8406 ms
- multigrid memory: 15.1170 ms
- speedup: 1.0479x
- finite forward/backward

Matched language screen, rank 224, 350 steps:

- Wave10 control: val 6.793358, 132,290 tok/s
- multigrid: val 6.795477, 132,856 tok/s

Decision: viable but not a breakthrough. Preserve it for explicit long-range
and state-tracking probes; do not spend another round tuning pooling widths.

## Next large lever

The next candidate must add a high-bandwidth identity-preserving memory rather
than another smooth convolution. The planned primitive combines error-correcting
token sketches with uncertainty-weighted state updates: multiple signed hashes
store causal token evidence, collision disagreement estimates uncertainty, and
the model gates reliable retrieved evidence into factor logits. This draws from
streaming algorithms, coding theory, and Bayesian filtering.
