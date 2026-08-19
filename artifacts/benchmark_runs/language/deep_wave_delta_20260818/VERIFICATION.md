# Verification status

## Completed locally

- Import and bytecode compilation passed for every Python file.
- Seven model tests passed; the CUDA-only FLA comparison was skipped locally.
- Forward/backward loss and every populated gradient were finite.
- Causality passed by changing suffix tokens and checking that prefix logits were unchanged.
- Cached token-by-token inference matched the parallel sequence path within `4e-5` tolerance.
- Parameter and persistent-state accounting matched the tensors in the model.
- The shared production trainer completed a one-step CPU integration smoke using the real
  sampled-plus-anchor loss and exact 50,257-token full-vocabulary validation.

Final post-fix trainer smoke:

```text
parameters:              10,157,225
training tokens:                 31
train loss:                 7.893611
full-vocab validation:      10.907697
gradient norm:               2.965704
finite failure:                  none
```

These losses are only plumbing checks on random synthetic tokens. They are not evidence that the
architecture is better.

## Required on the H100 before training

1. Run the CUDA fused-versus-reference test. It is a hard gate.
2. Run the batch/VRAM probe and choose the largest stable batch.
3. Run the paired 100M-token candidate and 349.4M delta-gain control.
4. Compare exact full-vocabulary validation, throughput, VRAM, and gradient stability.
5. Run the existing behavior suite on both checkpoints.

No scaling claim has been made yet. The implementation is complete; the empirical promotion test is
still outstanding.
