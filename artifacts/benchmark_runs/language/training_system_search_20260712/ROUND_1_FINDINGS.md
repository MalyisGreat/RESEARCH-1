# Training System Search: Round 1

## Gate

The reference is the original 40M Wave10 plus factor-recall screen at 256 steps:

- full-vocabulary validation loss: 6.797395
- throughput: 97,007 tokens/second
- mean step time: 104.735 ms

A breakthrough requires at least 145,511 tokens/second at equivalent validation
quality, or at least 2x lower wall time to the same validation loss.

## Verified findings

| Candidate | Steps | Full-vocab val | Tok/s | Decision |
|---|---:|---:|---:|---|
| Collapsed Wave10, rank 192, rotating stride 8, vocab 4096 | 350 | 6.762773 | 133,500 | Keep |
| Collapsed Wave10, rank 128, rotating stride 8, vocab 4096 | 350 | 6.838624 | 140,726 | Kill |
| Collapsed Wave10, rank 224, rotating stride 8, vocab 4096 | 350 | 6.748324 | 128,032 | Keep |
| Collapsed Wave10, rank 224, rotating stride 12, vocab 4096 | 350 | 6.793358 | 132,290 | Keep as equivalent-quality fast point |
| Same, vocab 3072 | 350 | 6.805424 | 134,111 | Kill |
| Same, vocab 2048 | 350 | 6.817110 | 137,128 | Kill |
| Same, no clipping, LR 2.5e-4 | 350 | 7.109079 | 135,533 | Kill |
| Same, no clipping, LR 6e-4 | 350 | 6.844534 | 139,149 | Kill |
| Same, accumulation 2, LR 6e-4 | 350 | 6.934515 | 140,838 | Kill |
| Same, accumulation 2, LR 1.2e-3 | 350 | 6.820210 | 143,422 | Kill |
| Same, accumulation 2, LR 1.5e-3 | 350 | 6.805926 | 140,070 | Kill |

The collapsed convolution is an exact reparameterization of the average of the
three linear depthwise convolution branches. Forward error was 4.768e-7 and
input-gradient error was 1.746e-10. It is the safest optimization found in this
round.

Gradient clipping is active: 108 of 127 finite measured steps exceeded a norm
of 1.0, with maximum 10.86. Removing clipping is not behaviorally exact and the
language screens reject it.

## Invalid runs

The directories below inherited `block_type=relu_square` because the command
omitted `--block-type multi_scale_lowrank_conv_memory`. They are not Wave10
comparisons and must not be used to rank convolution widths or sampled-vocab
sizes:

- `rank128_rotating_stride8_vocab4096_350_seed13`
- `wallclock380_collapsed_rotating_stride8_vocab2048_matched_seed13`

Their checkpoint configs preserve the evidence of the configuration error.

## Current conclusion

The best measured system improves equal-wall validation and achieves 1.38x raw
throughput, but no candidate yet satisfies the declared breakthrough gate.
Rank 224 buys useful quality at a favorable kernel shape. The next round should
target the dense block projections and FFN, which dominate step time after the
exact convolution collapse; further anchor or sampled-head sparsification has
shown diminishing returns.
