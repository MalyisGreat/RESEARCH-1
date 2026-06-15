# Manual Neuron Search Synthesis - 2026-06-05

All code and artifacts are under `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self`.

## Current Lead

`phase_residual_blend_half_neuron` is the current local lead. It preserves the baseline squared neuron and adds a learned per-channel signed phase residual:

`h = relu(z)^2 + sigmoid(a_c) * softplus(z) * tanh(z) * abs(tanh(z))`, with `a_c = 0` init.

Cost: +128 params in the 4.93M mini model, +2048 params in the 40.4M long-sequence model. It keeps tokenizer/objective/eval unchanged and is block-local/linear in sequence length.

1024-step seq255 real FineWebEdu deltas across seeds: -0.005130, -0.001666, -0.002024; mean -0.002940.
128-step seq10160 40M-ish real FineWebEdu deltas across seeds: -0.009420, -0.010180, -0.006860; mean -0.008820.

## Important Real-Text Results

| group | variant | val delta | train delta | speed ratio | param delta |
|---|---:|---:|---:|---:|---:|
| screen_128_seed13_real_seq10160_40m | phase_residual_blend_half_neuron | -0.009420 | -0.008512 | 0.945 | 2048 |
| screen_128_seed17_real_seq10160_40m | phase_residual_blend_half_neuron | -0.010180 | -0.009571 | 0.106 | 2048 |
| screen_128_seed29_real_seq10160_40m | phase_residual_blend_half_neuron | -0.006860 | -0.006656 | 0.956 | 2048 |
| screen_1024_seed13_real_half1024 | phase_residual_blend_half_neuron | -0.005130 | 0.000217 | 0.957 | 128 |
| screen_1024_seed17_real_half1024 | phase_residual_blend_half_neuron | -0.001666 | 0.001400 | 0.976 | 128 |
| screen_1024_seed29_real_half1024 | phase_residual_blend_half_neuron | -0.002024 | 0.001210 | 1.082 | 128 |
| screen_256_seed13_real_blend_strength | phase_residual_blend_half_neuron | -0.002844 | -0.004618 | 1.094 | 128 |
| screen_256_seed13_real_blend_strength | phase_residual_blend_large_neuron | -0.001653 | -0.003331 | 1.098 | 128 |
| screen_256_seed13_real_blend_strength | phase_residual_blend_neuron | -0.000486 | -0.001223 | 1.074 | 128 |
| screen_256_seed13_real_blend_strength | phase_residual_blend_tiny_neuron | -0.000178 | -0.000546 | 1.137 | 128 |
| screen_256_seed13_real_phase | phase_amplitude_neuron | -0.009628 | -0.137911 | 1.170 | 16640 |
| screen_256_seed13_real_phase | phase_amplitude_replace_neuron | -0.009614 | -0.137927 | 1.148 | 8320 |
| screen_256_seed13_real_phase | stable_competition_neuron | 0.010327 | -0.011565 | 1.139 | 2 |
| screen_256_seed17_real_blend_strength | phase_residual_blend_half_neuron | -0.002545 | -0.003885 | 1.107 | 128 |
| screen_256_seed17_real_blend_strength | phase_residual_blend_large_neuron | -0.001489 | -0.003493 | 1.176 | 128 |
| screen_256_seed17_real_blend_strength | phase_residual_blend_neuron | -0.000673 | -0.002297 | 1.132 | 128 |
| screen_256_seed17_real_blend_strength | phase_residual_blend_tiny_neuron | -0.000059 | -0.000189 | 1.157 | 128 |
| screen_256_seed17_real_phase | stable_competition_neuron | 0.016978 | 0.015782 | 0.976 | 2 |
| screen_256_seed17_real_phase | phase_amplitude_replace_neuron | 0.019403 | 0.100903 | 1.062 | 8320 |
| screen_256_seed17_real_phase | phase_amplitude_neuron | 0.019433 | 0.101158 | 0.941 | 16640 |
| screen_256_seed29_real_blend_strength | phase_residual_blend_half_neuron | -0.001391 | -0.002423 | 1.086 | 128 |
| screen_256_seed29_real_blend_strength | phase_residual_blend_large_neuron | -0.000681 | -0.000756 | 1.032 | 128 |
| screen_256_seed29_real_blend_strength | phase_residual_blend_neuron | -0.000356 | -0.000553 | 1.058 | 128 |
| screen_256_seed29_real_blend_strength | phase_residual_blend_tiny_neuron | 0.000096 | -0.000367 | 1.084 | 128 |
| screen_256_seed29_real_phase | stable_competition_neuron | 0.003610 | 0.003725 | 0.977 | 2 |
| screen_256_seed29_real_phase | phase_amplitude_replace_neuron | 0.058672 | 0.043880 | 1.087 | 8320 |
| screen_256_seed29_real_phase | phase_amplitude_neuron | 0.058709 | 0.043815 | 1.005 | 16640 |

## Decisions

- Kill full `phase_amplitude_neuron` and `phase_amplitude_replace_neuron` for now despite synthetic wins: real-text seeds 17 and 29 showed instability for full replacement.
- Kill `stable_competition_neuron` for real-text scale-up: synthetic wins did not transfer reliably to real FineWebEdu.
- Keep `phase_residual_blend_half_neuron`: it won every 256-step real strength seed, every 1024-step seq255 real seed, and every 128-step seq10160 40M-ish real seed tested.
- Still do not call this a breakthrough until it survives a longer 40M seq10160 screen, but it is now the best candidate for local promotion.

## Commands

Commands are tracked in `commands.txt` plus shell history in this thread; this synthesis was generated from per-run `result.json` files.
