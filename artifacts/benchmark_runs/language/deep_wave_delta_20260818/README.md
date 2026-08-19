# Deep Wave-Delta

This is the fully isolated implementation of the next architecture candidate. It preserves the
project's attention-free, linear-sequence design while addressing the strongest scaling failure in
Wave10: only one 512-value global state inside an extremely shallow four-block model.

## Architecture

Each block computes:

```text
l_t = adaptive_multiscale_causal_conv(RMSNorm(x_t))
x_t = x_t + l_t
f_t = low_rank_local_FIR(RMSNorm(x_t))
x_t = x_t + f_t

on recurrent layers:
  S'_t = a_t S_(t-1)
  S_t = S'_t + b_t k_t (v_t - k_t^T S'_t)^T
  m_t = Output(HeadRMSNorm(q_t^T S_t) * SiLU(output_gate_t))
  x_t = x_t + 2 sigmoid(competition_t) * m_t

x_t = x_t + gated_stable_ReLU_squared_FFN(RMSNorm(x_t))
```

The important distinction from the previous delta experiment is additive dual memory. The local
128-token low-rank FIR is not deleted when persistent memory is added. Four recurrent layers each
have `8 x 64 x 64 = 32,768` state values, for 131,072 global state values total. That is 256 times
the previous 512-value state. Local convolution scales are also no longer a linearly averaged bank:
each branch is nonlinear before a token-conditioned softmax chooses among it.

Training remains O(sequence length). Cached generation holds fixed convolution/FIR buffers and the
four recurrent matrices, making work and storage per generated token independent of prefix length.
There is no sequence-by-sequence attention matrix.

## Honest presets

| Preset | Parameters | Depth | Width | Persistent state values |
|---|---:|---:|---:|---:|
| 10m | 10,157,225 | 8 | 128 | 8,192 |
| 100m | 101,454,901 | 12 | 704 | 131,072 |
| 350m | 351,677,889 | 16 | 1,408 | 131,072 |

The 350M control is the existing 349.4M delta-gain model. The paired launcher uses identical cache,
seed, token budget, sampled vocabulary, anchor stride, validation cadence, and exact full-vocabulary
validation. Only model architecture differs.

## Verification

```bash
python -m pytest artifacts/benchmark_runs/language/deep_wave_delta_20260818/test_deep_wave_delta.py -q
python artifacts/benchmark_runs/language/deep_wave_delta_20260818/smoke_and_profile.py \
  --preset 10m --sequence-length 128 --steps 3
```

The tests enforce finite forward/backward gradients, strict causality, parameter/state accounting,
and equality between parallel sequence evaluation and token-by-token cached inference.

## H100 workflow

First find the largest batch that fits:

```bash
CACHE_PATH=/workspace/data/cache.pt \
bash artifacts/benchmark_runs/language/deep_wave_delta_20260818/profile_h100_batches.sh
```

Then run the paired 100M-token screen:

```bash
CACHE_PATH=/workspace/data/cache.pt TOKENS=100000000 \
bash artifacts/benchmark_runs/language/deep_wave_delta_20260818/launch_paired_h100.sh
```

Do not begin a billion-token run from this code alone. Promotion requires lower exact full-vocabulary
validation loss than delta-gain, finite gradients, acceptable throughput/VRAM, and no regression on
the existing repetition, exact-recall, stale-fact, and language-sample suites.

## Cached sampling

```bash
python artifacts/benchmark_runs/language/deep_wave_delta_20260818/sample_checkpoint.py \
  --checkpoint RUN/checkpoint.pt \
  --architecture-config RUN/deep_wave_delta_config.json \
  --prompt "The capital of France is"
```
