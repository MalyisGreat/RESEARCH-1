# agent02_conv_conditioned writeup

Artifact directory: `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned`

## Design

Implemented `CausalBranchDisagreementConditionedLowRankConvMemoryBlock` in the isolated trainer copy at `standalone_longseq_anchor_train_conv_conditioned.py:426`.

The baseline `CausalMultiScaleLowRankConvMemoryBlock` averages three causal depthwise branch outputs and applies a fixed ReLU-square neuron. The candidate keeps the same causal multi-scale conv residual and low-rank causal memory residual, but conditions the FFN neuron on the per-token disagreement among the multi-scale conv branches.

For branch outputs `b_i(t,c)`:

```text
m = mean_i b_i
d = mean_i (b_i - m)^2
z = causal_dwconv(context_down(LN(d)))
[shift, gain] = context_up(silu(z))
h = ffn_in(LN(x_after_conv_memory))
y = relu(h + 0.10 * tanh(shift))^2 * (1 + 0.25 * tanh(gain))
out = x_after_conv_memory + ffn_out(y)
```

This keeps sequence cost linear: depthwise causal convs plus token-local low-rank projections. It uses no attention or seq x seq operation.

## Ablation

Candidate v2 `CausalBranchDisagreementGainOnlyLowRankConvMemoryBlock` at `standalone_longseq_anchor_train_conv_conditioned.py:498` removes the shift term:

```text
gain = context_up(silu(causal_dwconv(context_down(LN(d)))))
y = relu(ffn_in(LN(x_after_conv_memory)))^2 * (1 + 0.25 * tanh(gain))
```

## Why it targets the baseline

The baseline already computes multi-scale causal branch responses but discards branch-level disagreement when it averages them. Disagreement is a local signal for scale ambiguity, boundary changes, and branch conflict. Conditioning the neuron on that signal lets the FFN change sharpness or threshold only where the causal conv branches disagree, while leaving the successful low-rank memory branch intact.

## Smoke and gradient checks

`smoke_grad_params.py` passed compile/import and finite forward/backward gradient checks on CPU with `CUDA_VISIBLE_DEVICES=-1`.

Mini screen parameter counts:

| block | params |
| --- | ---: |
| baseline `multi_scale_lowrank_conv_memory` | 15,250,481 |
| v1 `branch_disagreement_conditioned_lowrank_conv_memory` | 15,314,417 |
| v2 `branch_disagreement_gain_only_lowrank_conv_memory` | 15,289,073 |

## 64-step CPU screen

Shared cache: `screen_cache_synth_seq255_train768_val64_gpt2.pt`

Common config: seed 13, seq 255, 64 train steps, eval interval 64, val blocks 8, embedding dim 192, 2 conv layers, kernel 7, conv rank 96, memory rank 32, landmark stride 64, sampled vocab 4096, token stride 4, token chunks 512, LR 0.0006 to min LR 0.00001.

| run | final train | final val | tok/sec | peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| baseline | 9.5975475311 | 10.8327153921 | 56.6298374715 | null |
| v1 shift+gain | 9.6072959900 | 10.8301025629 | 75.5278412087 | null |
| v2 gain-only | 9.5992565155 | 10.8300732374 | 169.0786280320 | null |

## Recommendation

Keep only v2 for the next screen. V1 improved val loss by about 0.0026 but had worse train loss and extra params. V2 matched/slightly beat v1 val loss, had train loss much closer to baseline, used fewer added params, and was much faster in this CPU run. The val edge is tiny, so do not promote as a win yet; scale only to a 128-step same-cache repeat and at least one alternate seed before any larger run.

## Failure notes

Two setup mistakes were corrected before successful screens: the first baseline launch failed because the Tee log directory did not exist, and the second failed because the copied trainer did not expose `--sequence-length`. The isolated copy now has the CLI arg; no shared trainer file was edited.
