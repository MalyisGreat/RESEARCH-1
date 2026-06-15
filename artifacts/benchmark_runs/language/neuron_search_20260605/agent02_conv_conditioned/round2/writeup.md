# agent02_conv_conditioned round2

## Design

Block type: `conv_context_conditioned_neuron`

The candidate starts from `CausalMultiScaleLowRankConvMemoryBlock` and changes the FFN neuron into a conv-context-conditioned rectified square:

`branches = causal_depthwise_convs(LN(x))`

`mean = mean(branches)`

`disagreement = var(branches)`

`peak = max(branches) - min(branches)`

`hidden = relu(W_ffn LN(x))^2 * (0.5 + sigmoid(W_ctx [disagreement, peak]))`

The primitive is local to the block, uses causal convolution branch statistics only, has linear sequence cost, and does not use attention.

## Evidence

- Compile: `python -B -m py_compile agent02_conv_conditioned_trainer.py` passed.
- Tiny finite-gradient check passed: finite forward/loss/gradients.
- Mini parameter check: baseline `609824`, candidate `613984`.
- Screen protocol: shared seq255 cache, CPU-only, 32 train steps, same optimizer/eval/cache/sampled-vocab settings for baseline and candidate.

## Screen Metrics

Baseline `multi_scale_lowrank_conv_memory`:

- `final_train_loss`: 8.986903190612793
- `final_val_loss`: 10.833873510360718
- `pure_train_tok_per_sec`: 260.32797898982875
- `parameter_count`: 7419489
- `peak_vram_mb`: null

Candidate `conv_context_conditioned_neuron`:

- `final_train_loss`: 8.994543075561523
- `final_val_loss`: 10.827611923217773
- `pure_train_tok_per_sec`: 222.246594297002
- `parameter_count`: 7456545
- `peak_vram_mb`: null

## Recommendation

Kill for now. The candidate improved short-screen validation by about 0.0063 but regressed train loss, added parameters, and slowed CPU throughput by about 14.6%. This is not strong enough to scale without an ablation showing the disagreement gate is responsible and stable.

Next ablation: keep the context gate but reduce overhead with a rank-16 context projection, or replace `[disagreement, peak] -> expansion*dim` with per-channel scalar gates before the FFN expansion.
