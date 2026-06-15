# agent01_memory_coupling writeup

Artifact directory: `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent01_memory_coupling`

Copied trainer: `standalone_longseq_anchor_train_memory_coupling.py`

Shared trainer was not edited.

## Design

The first design, `CausalMemorySelectedThresholdBasisLowRankConvMemoryBlock`, keeps the existing multiscale depthwise convolution and low-rank causal memory residual, but also routes the low-rank causal memory state over learned FFN threshold/gain bases per token.

Core equations:

`c_t = mean_k DWConv_k(LN_c(x))_t`

`x'_t = x_t + dropout(W_mix relu(c_t)^2)`

`m_t = DWConv_mem(W_down LN_m(x'))_t`

`x''_t = x'_t + dropout(W_up silu(m_t))`

`a_t = softmax(W_router LN_r(m_t))`

`tau_t = 0.25 tanh(a_t T)`

`gamma_t = 1 + 0.5 tanh(a_t G)`

`h_t = gamma_t * relu(W_ff LN_f(x'')_t - tau_t)^2`

`y_t = x''_t + dropout(W_out h_t)`

This is memory-neuron coupling because the memory path changes the activation threshold of each FFN neuron before the ReLU-square nonlinearity. It is not just a residual memory add or output gate.

The second design, `CausalMemoryTiledThresholdLowRankConvMemoryBlock`, was a quick cheaper iteration after v1 proved slow. It tiles the rank-space memory channels across FFN hidden groups and uses them as per-token threshold/gain coordinates:

`tau_rank_t = 0.20 tanh(m_t) * s_tau`

`gamma_rank_t = 1 + 0.20 tanh(m_t) * s_gamma`

`tau_t = tile(tau_rank_t)[:hidden_dim]`

`gamma_t = tile(gamma_rank_t)[:hidden_dim]`

`h_t = gamma_t * relu(W_ff LN_f(x'')_t - tau_t)^2`

## Why It Could Help

`CausalMultiScaleLowRankConvMemoryBlock` currently adds the causal low-rank memory back into the residual stream, then uses a static FFN activation threshold. If the memory vector summarizes older context, it may be more useful as a token-conditioned decision boundary for FFN neurons: old-context features can decide which hidden features become active at the current token without using seq-by-seq attention.

## Novelty

The tested change uses the existing causal memory signal to alter FFN neuron thresholds and gain before the activation, rather than using a token gate over outputs, changing the activation family, or adding transformer-style attention. v1 does this through learned threshold/gain basis selection; v2 does it through tiled rank-space threshold groups.

## Protocol

All tests were CPU-only with `CUDA_VISIBLE_DEVICES=-1`. No CUDA device was used; `torch.cuda.is_available()` was false in the smoke/gradient test.

Screen cache: `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\screen_cache_synth_seq255_train768_val64_gpt2.pt`

Reduced screen: 64 train steps, eval at step 64, sequence length 255, val blocks 8, embedding dim 192, conv layers 2, conv rank 96, memory rank 32, landmark stride 64, sampled vocab 4096, seed 13, default warmup 2000.

## Metrics

| Run | Block type | Params | Train loss | Val loss | Tok/sec | Mean step ms | Median step ms | Peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | `multi_scale_lowrank_conv_memory` | 15,250,481 | 9.5975475 | 10.8327154 | 99.1843 | 2570.97 | 1483.97 | null |
| candidate v1 | `memory_selected_threshold_basis_lowrank_conv_memory` | 15,257,017 | 9.6050529 | 10.8323523 | 57.1076 | 4465.26 | 3419.03 | null |
| candidate v2 | `memory_tiled_threshold_lowrank_conv_memory` | 15,250,609 | 9.5979557 | 10.8327206 | 132.6463 | 1922.41 | 1610.33 | null |

Smoke/gradient checks passed for baseline, v1, and v2. All checked input and parameter gradients were finite.

## Decision

Kill for now. v1 is too slow and has worse train loss despite a tiny val-loss edge that is likely noise. v2 is stable and cheap, but it does not improve validation loss over baseline; its mean CPU throughput looked better, but median step time was slightly worse than baseline, so this is not a robust speed win.

If this direction is revisited despite the kill recommendation, the next ablation should test v2 threshold strength values `0.05`, `0.10`, and `0.20`, plus a no-memory-residual variant where the low-rank causal memory only conditions thresholds and is not also added to `x`.
