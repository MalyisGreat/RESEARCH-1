# Token-Addressable Recall Search 2026-06-16

## Objective

Test whether the current best low-rank conv-memory language model is missing a high-bandwidth token-addressable recall/copy path into the logits.

All runs here used the local RTX 2080 SUPER. No 4060 Ti was used. The main trainer was not modified; experiments are isolated in:

`E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\token_recall_search_20260616\token_recall_train.py`

## Implemented Variants

### 1. Direct Token-Recall Logit Bias

Modes:

- `token_recall_multiscale`
- `token_recall_gated_multiscale`

For candidate token `c` at position `t`, build exact causal token-presence features:

```text
r_w(t,c) = 1[c appears in x[max(0,t-w+1):t]]
r_prefix(t,c) = 1[c appears in x[0:t]]
b(t,c) = sum_i softplus(a_i) * gate_i(h_t) * r_i(t,c)
logits'(t,c) = logits(t,c) + b(t,c)
```

This directly tests the missing copy path. It improved short screens, but the full-vocab path is too expensive for long training unless heavily optimized.

### 2. Factor-Space Token Recall

Mode:

- `factor_recall_gated_multiscale`

This is the scalable candidate. It retrieves the output-factor vectors for exact prior token IDs, averages them over causal windows and prefix, gates them from the current hidden state, and adds the result before the normal output projection.

```text
v_i = stopgrad(W_out_factor[x_i])
m_w(t) = mean(v_i for i in [max(0,t-w+1), t])
m_prefix(t) = mean(v_i for i in [0,t])
g_t = sigmoid(W_g h_t)
h'_t = h_t + sum_i softplus(a_i) * g_t,i * m_i(t)
logits_t = W_out_factor h'_t + b
```

Why it matters: it gives the architecture an exact token-addressed retrieval path without quadratic attention and without materializing `sequence x vocab` history tensors. It adds only `+776` parameters in the 40M local config.

## Smoke / Stability

Passed:

- `python -m py_compile token_recall_train.py`
- finite forward/backward gradient test for:
  - `none`
  - `token_recall_gated_multiscale`
  - `factor_recall_multiscale`
  - `factor_recall_gated_multiscale`
- dense full-vocab candidate-logit finite check

## Short Seq255 Evidence

Config:

```text
seq=255, train_steps=512, val_blocks=16, dim=192, layers=2, conv_rank=96,
memory_rank=32, sampled_vocab=4096, token_stride=4
```

Baseline full-vocab validation:

| Seed | Baseline val |
| ---: | ---: |
| 13 | 7.3674437605 |
| 17 | 7.3706933158 |
| 29 | 7.3388844914 |

Direct token-logit recall, scale `0.025`:

| Seed | Val | Delta |
| ---: | ---: | ---: |
| 13 | 7.3517266480 | -0.0157171124 |
| 17 | 7.3548387142 | -0.0158546016 |

Factor-space recall scale sweep:

| Seed | Scale | Val | Delta |
| ---: | ---: | ---: | ---: |
| 13 | 32 | 7.3353494172 | -0.0320943433 |
| 13 | 64 | 7.3326988283 | -0.0347449321 |
| 13 | 128 | 7.3204984721 | -0.0469452884 |
| 17 | 32 | 7.3251808315 | -0.0455124843 |
| 17 | 64 | 7.3241321591 | -0.0465611568 |
| 17 | 128 | 7.3177825495 | -0.0529107663 |
| 29 | 32 | 7.2976195698 | -0.0412649217 |
| 29 | 64 | 7.2944074030 | -0.0444770884 |

Scale `512` was too high on long-seq and is not recommended.

## Long Seq10160 Evidence

Config:

```text
seq=10160, dim=512, layers=2, conv_rank=192, memory_rank=64,
sampled_vocab=8192, token_stride=4, full-vocab validation, val_blocks=4
```

### 256 Steps

| Seed | Mode | Scale | Val | Delta | Tok/s | Peak VRAM MB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 13 | baseline | - | 6.7391919408 | 0.0000000000 | 122285.63 | 2975.12 |
| 13 | factor recall | 128 | 6.6765535071 | -0.0626384337 | 108804.24 | 3054.70 |
| 13 | factor recall | 256 | 6.6703455476 | -0.0688463932 | 108295.74 | 3054.70 |
| 17 | baseline | - | 6.7365382056 | 0.0000000000 | 121140.72 | 2973.66 |
| 17 | factor recall | 128 | 6.6874214182 | -0.0491167874 | 107897.24 | 3054.02 |
| 17 | factor recall | 256 | 6.6829192736 | -0.0536189320 | 108916.60 | 3054.02 |
| 13 | factor recall | 512 | 6.7150364865 | -0.0241554544 | 108196.40 | 3054.70 |

### 1024 Steps

| Seed | Mode | Scale | Step 512 val | Final val | Final delta | Tok/s | Speed ratio | Peak VRAM MB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | baseline | - | 6.4663407801 | 6.4147516630 | 0.0000000000 | 123801.29 | 1.000 | 2975.12 |
| 13 | factor recall | 256 | 6.4035149619 | 6.3735646826 | -0.0411869804 | 109328.48 | 0.883 | 3054.70 |
| 17 | baseline | - | 6.4580827189 | 6.4080369805 | 0.0000000000 | 123501.56 | 1.000 | 2973.66 |
| 17 | factor recall | 256 | 6.4331444076 | 6.3889420676 | -0.0190949128 | 109333.09 | 0.885 | 3054.02 |

Mean 1024-step delta across seeds: `-0.0301409466`.

Mean speed ratio across seeds: `0.8841867285`.

Parameter delta: `+776`.

VRAM delta: about `+80 MB`.

## Interpretation

This is the best local evidence so far for the missing critical component diagnosis. The direct logit-bias copy path proves that exact token-addressed recall helps. The factor-space version keeps most of the useful behavior while staying linear and cheap enough for long-sequence runs.

The effect weakens as training progresses, but it persists through 1024 long-seq steps and two seeds. That makes it stronger evidence than the earlier activation/neuron swaps.

## Kill / Keep / Scale

Kill for now:

- Dense direct full-vocab token-logit recall as implemented here. It wins short screens but is too expensive in the anchor path.
- Scale `512` for factor recall. It still beats baseline but loses much of the gain and has a bad first-step loss.

Keep:

- `factor_recall_gated_multiscale`
- Preferred scale range for next screen: `128-256`
- Current best local setting: `factor_recall_gated_multiscale`, `recall_initial_scale=256`

Next ablations:

- Add a scheduled recall scale: start around `64-128`, ramp or cap near `256`, avoid `512`.
- Test no-detach output-factor retrieval versus current stop-grad retrieval.
- Test removing prefix or reducing windows to see whether the gain is local copy or prefix memory.
- Run a 40M-100M token local/3080 screen with full-vocab validation before any 5B run.

## Representative Commands

Compile:

```powershell
python -m py_compile E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\token_recall_search_20260616\token_recall_train.py
```

1024-step long-seq baseline shape:

```powershell
python E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\token_recall_search_20260616\token_recall_train.py `
  --cache-path E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\real_cache_finewebedu_sample_seq10160_train192_val8_gpt2.pt `
  --sequence-length 10160 --train-steps 1024 --eval-interval 512 --val-blocks 4 `
  --embedding-dim 512 --block-type multi_scale_lowrank_conv_memory --conv-layers 2 `
  --conv-rank 192 --memory-rank 64 --landmark-stride 128 --sampled-vocab-size 8192 `
  --token-stride 4 --token-chunk-size 20000 --full-eval-token-chunk-size 512 `
  --recall-mode factor_recall_gated_multiscale --recall-initial-scale 256
```
