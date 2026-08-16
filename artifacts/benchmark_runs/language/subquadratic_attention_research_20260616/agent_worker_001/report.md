# Subquadratic Attention Prototype for Low-Rank Conv-Memory LM

## Trainer and block interface

Source inspected: `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\standalone_longseq_anchor_train.py`.

Relevant interface:

- `TrainConfig.sequence_length` defaults to `10160`; `batch_size` defaults to `1`; `embedding_dim` defaults to `1831`.
- `CausalConvFactorizedLM.features(input_ids)` embeds integer tokens to `states` shaped `[batch, tokens, dim]`.
- Every mixer block receives and returns `[B, T, D]`.
- The current best family, `CausalMultiScaleLowRankConvMemoryBlock`, applies:
  - `LayerNorm` then causal depthwise conv branches on `[B, D, T]`, using left padding only.
  - residual `mix(relu(conv_output)^2)`.
  - `LayerNorm`, `memory_down: D -> memory_rank`, causal depthwise conv in low-rank space, `memory_up: memory_rank -> D`.
  - relu-squared FFN with expansion `2`.
- The loss path consumes `model.factor_down(model.features(input_ids))`, so a compatible new primitive only needs to preserve `[B, T, D]`.

Prior context from the trainer tree:

- Existing `landmark_attention` attends each token to prefix landmarks, materializing `[B, heads, T, landmarks]`.
- The research board says the landmark-attention screen was slower and lost to the dense/conv alternatives, while `multi_scale_lowrank_conv_memory` became the promoted family.

## Proposed primitive

Prototype: `CausalChunkedLinearAttentionMemory`, inserted before the FFN in `CausalMultiScaleLowRankConvMemoryLinearAttentionBlock`.

For normalized input `u_t in R^D`:

```text
q_t = (elu(W_q u_t) + 1) / sqrt(R)
k_t =  elu(W_k u_t) + 1
v_t = W_v u_t
S_t = S_{t-1} + k_t outer v_t
z_t = z_{t-1} + k_t
a_t = (q_t S_t) / (q_t dot z_t + eps)
y_t = x_t + dropout(W_o a_t * sigmoid(W_g u_t))
```

Shapes:

- `q_t`, `k_t`: `[B, T, R]`, where `R = feature_rank`.
- `v_t`, `a_t`: `[B, T, M]`, where `M = value_rank`.
- recurrent KV state: `[B, R, M]`.
- recurrent normalizer state: `[B, R]`.
- output remains `[B, T, D]`.

The implementation scans by chunks. Inside each chunk it uses `cumsum` over `k outer v`; across chunks it carries only the final `[B, R, M]` and `[B, R]` states. This keeps the causal prefix exact for the selected feature map while avoiding any `[T, T]` attention matrix.

## Why this may help this architecture

The current block is strong locally: causal multi-scale depthwise convolution handles short and mid-range patterns, and the low-rank conv-memory branch adds a learned causal summary with finite kernel width. The missing capability is content-addressed global recall over the whole prefix.

This branch adds global causal content routing while staying aligned with the current design:

- Low-rank internal state, matching the architecture's preference for cheap bottlenecks.
- No requirement that `D=1831` divide attention heads.
- Positive-feature linear attention gives each token a content-conditioned prefix mixture instead of a prefix mean or fixed convolution.
- A scalar residual gate is initialized with bias `-2.0`, so the new branch starts conservative.
- The branch can be placed after conv-memory and before FFN, letting local features feed the global recurrent state.

## Subquadratic cost

For sequence length `T`, feature rank `R`, value rank `M`, model dim `D`, and training chunk size `C`:

- Time: `O(T * D * (2R + 2M) + T * R * M)`.
- Chunked training state memory: `O(C * R * M + T * M)` plus projections and activations.
- Streaming inference state memory: `O(R * M + R)`.
- No `[T, T]` or `[T, landmarks]` score matrix is required.

Default compatibility estimate for `D=1831`, `T=10160`, `R=32`, `M=64`, `C=1024`:

- Primitive parameters: `357,046`.
- Full proposed block parameters with `memory_rank=64`: `17,430,158`.
- Approx projection multiply-adds: `3,571,768,320`.
- Approx state update multiply-adds: `20,807,680`.
- Max chunk KV elements: `2,097,152`.
- Streaming state elements: `2,080`.

The projection cost is non-trivial, but it is linear in sequence length and small compared with a full hidden-size quadratic attention path at `T=10160`. It should be tested as an additive branch, not as a replacement for the current local mixer.

## Implementation files

- `linear_attention_memory.py`: primitive, full compatible block, parameter count helper, cost helper.
- `test_smoke.py`: CPU-only validation runner.
- `result.json`: generated validation result.

All files are under:

`E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\subquadratic_attention_research_20260616\agent_worker_001`

## Tests run

Command:

```powershell
python test_smoke.py
```

Result: passed on CPU.

Checks:

- `py_compile` syntax check for both Python files.
- Import smoke by importing `linear_attention_memory`.
- Primitive forward shape check on `[2, 9, 17]`.
- Full block forward shape check on `[2, 11, 17]`.
- Backward pass with all finite input and parameter gradients.
- Causality check: perturbing future tokens produced zero max absolute change in earlier outputs for both primitive and full block.
- Double-precision directional finite-difference check for the primitive:
  - analytical directional grad: `0.0005948430437086221`
  - finite-difference directional grad: `0.0005948430636948387`
  - relative error: `3.3599142184767453e-08`

No training was launched. No GPU was used intentionally.

## Risks and what failed

No validation failure occurred in the smoke tests.

Risks:

- Under trainer fp16 autocast, long prefix sums can overflow if accumulated in fp16. The prototype explicitly computes recurrent state math in fp32 for fp16/bf16 inputs, which is safer but increases bandwidth and activation memory.
- The branch adds about `0.36M` parameters at `D=1831,R=32,M=64`; this is modest, but its linear projections add real per-token cost.
- ELU+1 linear attention is not softmax attention. It may blur retrieval when many unrelated prefix keys share positive features.
- Chunked vectorization avoids quadratic memory but still stores chunk-level `R*M` prefix activations for backprop. Very large `R`, `M`, or `C` can become memory-bound.
- The branch was only validated mechanically; there is no loss or throughput evidence yet.

## Recommended next ablation

Run a tiny fair CPU or permitted non-3080/non-4060 smoke only if capacity allows:

1. Add this block type to an isolated experimental copy of the trainer, not the live trainer.
2. Compare current `multi_scale_lowrank_conv_memory` against `multi_scale_lowrank_conv_memory_linear_attention` for only a tiny diagnostic, e.g. `seq=512`, `steps=8-16`, `batch=1`, `R=16`, `M=32`, `chunk=128`.
3. Track tokens/sec, peak memory, first loss movement, and whether gradients stay finite under the trainer's autocast settings.
4. If stable, test `R/M` grid: `(16,32)`, `(32,32)`, `(32,64)` before any long run.

Recommendation: keep as a candidate global-content branch for a small ablation. Do not promote until it shows loss benefit per wall-clock over the promoted conv-memory block.
