# Adaptive Local Basis Neuron

## Design

`AdaptiveLocalBasisNeuron` is a cheap token/channel-local mixture over four fixed nonlinear basis responses. For an input activation `z[b,t,c]`:

```text
b(z) = [relu(z)^2, -relu(-z)^2, z * tanh(z), z * sigmoid(z)]
pi[b,t,c,k] = softmax_k(z[b,t,c] * a[c,k] + beta[c,k])
phi(z)[b,t,c] = sum_k pi[b,t,c,k] * b_k(z[b,t,c])
```

The learned parameters are only per-channel slopes and biases for the basis router. The initialization biases the first basis toward the existing `relu^2` behavior, while nonzero initial slopes allow token/channel-local routing by activation sign and magnitude.

v1, `CausalMultiScaleLowRankConvMemoryAdaptiveBasisBlock`, applies `phi` at all three nonlinear sites in `CausalMultiScaleLowRankConvMemoryBlock`: multi-scale conv output before `mix`, low-rank causal memory output before `memory_up`, and the FFN hidden before `ffn_out`.

v2, `CausalMultiScaleLowRankConvMemoryFFNAdaptiveBasisBlock`, keeps the baseline conv and memory paths unchanged and applies `phi` only to the FFN hidden. This was added after v1 was slower and worse.

## Why It Might Help This Block

The target block mixes local multi-scale depthwise features and a low-rank causal memory stream, then expands through a small FFN. The fixed baseline nonlinearities force all token/channel activations through one response shape. A local basis router can choose positive quadratic, negative quadratic, saturating signed, or smooth gated behavior per token and channel without sequence-by-sequence attention or a large gating projection.

## Results

All runs used the shared seq255 cache, seed 13, CPU only, 64 train steps, sampled vocab 4096, token stride 4, val blocks 8, embedding dim 192, 2 layers, conv rank 96, memory rank 32, and the same optimizer settings.

| run | block type | params | train loss | val loss | tok/sec | peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | `multi_scale_lowrank_conv_memory` | 15,250,481 | 9.5975475311 | 10.8327153921 | 91.0328 | null |
| candidate_v1 | `multi_scale_lowrank_conv_memory_adaptive_basis` | 15,260,209 | 9.5989379883 | 10.8331118822 | 61.9050 | null |
| candidate_v2 | `multi_scale_lowrank_conv_memory_ffn_adaptive_basis` | 15,256,625 | 9.5977964401 | 10.8326779604 | 117.8833 | null |

Finite-gradient checks passed for baseline, v1, and v2 blocks and tiny full models. `torch.cuda.is_available()` was false under `CUDA_VISIBLE_DEVICES=-1`.

## Recommendation

Kill v1. It is worse on train and val loss and slower.

Keep v2 only as a small follow-up, not as a scale candidate yet. Its validation loss is lower than baseline by only 0.000037, while train loss is slightly worse; this is effectively a tie at 64 CPU steps. The throughput advantage should be treated as noisy until repeated.

Next ablation if kept: rerun baseline vs v2 for 128 steps with two seeds, then test an FFN-only router with the negative quadratic basis removed to see whether inhibitory basis capacity is helping or just adding variance.
