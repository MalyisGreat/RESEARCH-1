# agent04 rank competition writeup

## Design

Implemented Local Rank Rivalry inside the copied trainer at `rank_competition_train.py`.

For each token and each local group of four rank or FFN channels:

`centered = u_g - mean(u_g)`

`score = centered / sqrt(mean(centered^2) + eps)`

`weight = group_size * softmax(score / temperature)`

`out = activation(u) * (1 + sigmoid(strength) * (weight - 1))`

The primitive is used twice inside the existing multi-scale low-rank conv memory block: once on the memory-rank channels after the causal rankwise depthwise conv, and once on the expanded FFN channels before `ffn_out`. It does not sort over sequence and does not use seq x seq attention.

Candidate v2 adds per-group mean-absolute-energy preservation:

`out_v2 = out * mean(abs(activation(u))) / (mean(abs(out)) + eps)`

## Why It Might Help

`CausalMultiScaleLowRankConvMemoryBlock` has independent low-rank memory channels and a wide ReLU-square FFN. Those channels can redundantly co-fire. Local rank rivalry forces nearby channels to compete after cheap group normalization, so above-group channels get amplified and below-group channels are inhibited. In principle this could make memory-rank capacity more specialized and reduce diffuse FFN activation.

## What Is Novel

This is not an activation swap. It is a block-local channel allocation primitive applied to both memory-rank and expanded FFN features, using groupwise normalized softmax competition with no sequence sorting and no attention matrix.

## Tests

CPU-only smoke and finite-gradient test passed for baseline, candidate v1, and candidate v2 with `CUDA_VISIBLE_DEVICES=-1`.

The copied trainer was also given a local `--sequence-length` CLI flag because the shared cache requires `sequence_length=255` and the original CLI did not expose that field.

## Screen Protocol

Reduced to 64 train steps because CPU throughput was slow. Baseline and candidates used the same seed, cache, sampled-vocab settings, optimizer settings, and mini model config:

`--sequence-length 255 --train-steps 64 --eval-interval 64 --checkpoint-interval 0 --milestone-checkpoint-interval 0 --val-blocks 8 --embedding-dim 192 --conv-layers 2 --conv-kernel-size 7 --conv-rank 96 --memory-rank 32 --landmark-stride 64 --sampled-vocab-size 4096 --token-stride 4 --token-chunk-size 512 --full-eval-token-chunk-size 512 --learning-rate 0.0006 --min-learning-rate 0.00001`

## Metrics

Baseline `multi_scale_lowrank_conv_memory`: train 9.59754753112793, val 10.832715392112732, params 15250481, VRAM null, tok/sec 65.15254413672825.

Candidate v1 `rank_competition_lowrank_conv_memory`: train 9.597831726074219, val 10.83288323879242, params 15250485, VRAM null, tok/sec 80.47606488003925.

Candidate v2 `energy_preserving_rank_competition_lowrank_conv_memory`: train 9.597482681274414, val 10.832782506942749, params 15250485, VRAM null, tok/sec 165.88483359029297.

## Decision

Kill for now. v2 recovered most of the v1 loss regression and had slightly lower train loss, but both variants were still worse than baseline validation loss. The apparent CPU throughput gains are noisy and not a credible benefit for a softmax-based extra primitive.

Next ablation only if revived: memory-only energy-preserving rivalry, with the FFN path left exactly baseline, to isolate whether low-rank memory competition helps without perturbing FFN optimization.
