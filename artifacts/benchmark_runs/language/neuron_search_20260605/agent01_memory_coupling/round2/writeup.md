# agent01_memory_coupling round2

## Design

Block type: `memory_threshold_basis_lowrank`

The block keeps the current best multi-scale causal depthwise convolution and low-rank causal memory path, but reuses the low-rank memory features inside the FFN neuron. For each token, the memory path produces a hidden-width threshold offset and a hidden-width basis shift:

`m = silu(depthwise_causal_conv(memory_down(memory_norm(x))))`

`x = x + memory_up(m)`

`hidden = relu(ffn_in(ffn_norm(x)) - 0.25 * tanh(memory_threshold(m)) + 0.1 * memory_basis(m))^2`

This is intended to move the ReLU-square activation boundary and hidden basis per token from causal memory context, not to gate the block output.

## Evidence

Compile passed with `python -B -m py_compile`.

Tiny forward/backward smoke:

- baseline `multi_scale_lowrank_conv_memory`: 28,401 params, loss 5.556443214416504, finite loss/grads true, 30/30 nonzero grad tensors
- candidate `memory_threshold_basis_lowrank`: 29,489 params, loss 5.609872341156006, finite loss/grads true, 33/33 nonzero grad tensors

Matched 32-step CPU screen, shared seq255 cache:

- baseline: final_train_loss 8.986903190612793, final_val_loss 10.833873510360718, pure_train_tok_per_sec 286.0195735532777, params 7,419,489, peak_vram_mb null
- candidate: final_train_loss 8.996628761291504, final_val_loss 10.837329149246216, pure_train_tok_per_sec 202.42224059392578, params 7,425,825, peak_vram_mb null

## Recommendation

Kill. The candidate is structurally valid, but it loses on train loss, validation loss, parameter count, and CPU throughput in the matched short screen.

Next ablation, if revisited: reduce the memory-to-hidden projection cost with grouped or rank-factored threshold/basis offsets and test threshold-only vs basis-only separately.
