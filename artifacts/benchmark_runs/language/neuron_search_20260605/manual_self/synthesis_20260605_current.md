# Manual Neuron Search Synthesis - Current 2026-06-05

Artifact root: `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self`

## Status

No 5B-token or 3080 scale-up run has been started. No candidate is cleared for scale-up yet; the current research leader is `rank_competition_memory_suppressed_centered_gate_neuron`.

`rank_competition_memory_suppressed_centered_gate_neuron` is the current long-gate leader on real FineWebEdu seq10160 val8: n=5, wins=5, mean val delta=-0.010564, best=-0.019104, worst=-0.000137, mean speed ratio=0.838, mean param delta=135170.0. This is a real measured signal, but it is not scale-cleared because longer confirmation has exposed stability risk.

Longer weak-seed confirmation for `rank_competition_memory_suppressed_centered_gate_neuron` has at least one non-finite candidate loss; that blocks 3080 or 5B-token scale-up until a stable second iteration beats the same controls.

## Ranked Aggregate Buckets

| variant | cache | seq | val blocks | steps | n | wins | mean val delta | best | worst | mean speed | mean param delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hidden_drop_p30_square_neuron | real | 10160 | 8 | 2048 | 1 | 1 | -0.110305 | -0.110305 | -0.110305 | 0.991 | 0.0 |
| hidden_drop_p25_square_neuron | real | 10160 | 8 | 2048 | 1 | 1 | -0.102049 | -0.102049 | -0.102049 | 0.986 | 0.0 |
| hidden_drop_extreme_square_neuron | real | 10160 | 8 | 2048 | 1 | 1 | -0.091567 | -0.091567 | -0.091567 | 0.991 | 0.0 |
| hidden_drop_ultra_square_neuron | real | 10160 | 8 | 2048 | 1 | 1 | -0.077874 | -0.077874 | -0.077874 | 0.990 | 0.0 |
| hidden_drop_high_square_neuron | real | 10160 | 8 | 2048 | 1 | 1 | -0.058923 | -0.058923 | -0.058923 | 0.988 | 0.0 |
| hidden_drop_mid_square_neuron | real | 10160 | 8 | 2048 | 1 | 1 | -0.048029 | -0.048029 | -0.048029 | 0.990 | 0.0 |
| rank_competition_fixed_feather_hidden_drop_neuron | real | 10160 | 8 | 2048 | 1 | 1 | -0.035606 | -0.035606 | -0.035606 | 0.975 | 0.0 |
| hidden_drop_square_neuron | real | 10160 | 8 | 2048 | 3 | 3 | -0.034577 | -0.036506 | -0.031223 | 0.988 | 0.0 |
| rank_competition_fixed_feather_neuron | real | 10160 | 8 | 2048 | 1 | 0 | 0.000111 | 0.000111 | 0.000111 | 0.982 | 0.0 |
| rank_competition_feather_neuron | real | 10160 | 8 | 2048 | 1 | 0 | 0.000121 | 0.000121 | 0.000121 | 0.982 | 2.0 |
| rank_competition_ultrasoft_neuron | real | 10160 | 8 | 2048 | 1 | 0 | 0.000931 | 0.000931 | 0.000931 | 0.983 | 2.0 |
| rank_competition_centered_residual_neuron | real | 10160 | 8 | 2048 | 1 | 0 | 0.001093 | 0.001093 | 0.001093 | 0.891 | 2050.0 |
| rank_competition_soft_neuron | real | 10160 | 8 | 2048 | 1 | 0 | 0.003544 | 0.003544 | 0.003544 | 0.983 | 2.0 |
| rank_competition_mild_neuron | real | 10160 | 8 | 2048 | 1 | 0 | 0.007532 | 0.007532 | 0.007532 | 0.983 | 2.0 |
| rank_competition_memory_suppressed_stopgrad_mask_centered_gate_neuron | real | 10160 | 8 | 2048 | 1 | 0 | 0.008068 | 0.008068 | 0.008068 | 0.849 | 135170.0 |
| rank_competition_neuron | real | 10160 | 8 | 2048 | 1 | 0 | 0.012115 | 0.012115 | 0.012115 | 0.982 | 2.0 |
| rank_competition_memory_suppressed_bounded_centered_gate_neuron | real | 10160 | 8 | 2048 | 1 | 0 | 269.588360 | 269.588360 | 269.588360 | 0.830 | 135170.0 |
| rank_competition_memory_suppressed_centered_gate_neuron | real | 10160 | 8 | 1024 | 5 | 5 | -0.010564 | -0.019104 | -0.000137 | 0.838 | 135170.0 |
| rank_competition_memory_centered_gate_neuron | real | 10160 | 8 | 1024 | 5 | 4 | -0.005945 | -0.014662 | 0.003996 | 0.869 | 135170.0 |
| hidden_drop_ultra_square_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.004926 | -0.004926 | -0.004926 | 0.982 | 0.0 |
| hidden_drop_extreme_square_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.004329 | -0.004329 | -0.004329 | 0.982 | 0.0 |
| hidden_drop_p25_square_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.004293 | -0.004293 | -0.004293 | 0.982 | 0.0 |
| phase_residual_memory_gate_neuron | real | 10160 | 8 | 1024 | 5 | 3 | -0.004064 | -0.014127 | 0.002403 | 0.893 | 135168.0 |
| rank_competition_centered_residual_neuron | real | 10160 | 8 | 1024 | 3 | 3 | -0.003985 | -0.005062 | -0.002814 | 0.892 | 2050.0 |
| rank_competition_neuron | real | 10160 | 8 | 1024 | 3 | 3 | -0.003895 | -0.004362 | -0.003204 | 0.990 | 2.0 |
| hidden_drop_high_square_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.003874 | -0.003874 | -0.003874 | 0.980 | 0.0 |
| phase_residual_memory_centered_gate_neuron | real | 10160 | 8 | 1024 | 5 | 3 | -0.003573 | -0.012090 | 0.004837 | 0.881 | 135168.0 |
| hidden_drop_p30_square_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.003039 | -0.003039 | -0.003039 | 0.981 | 0.0 |
| hidden_drop_mid_square_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.002804 | -0.002804 | -0.002804 | 0.980 | 0.0 |
| rank_competition_mild_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.002389 | -0.002389 | -0.002389 | 0.974 | 2.0 |
| hidden_drop_p35_square_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.002247 | -0.002247 | -0.002247 | 0.984 | 0.0 |
| rank_competition_fixed_feather_hidden_drop_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.002130 | -0.002130 | -0.002130 | 0.964 | 0.0 |
| hidden_drop_square_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.001979 | -0.001979 | -0.001979 | 0.978 | 0.0 |
| rank_competition_soft_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.001207 | -0.001207 | -0.001207 | 0.973 | 2.0 |
| rank_competition_memory_scalar_centered_gate_neuron | real | 10160 | 8 | 1024 | 3 | 2 | -0.000859 | -0.003014 | 0.002049 | 0.877 | 2180.0 |
| rank_competition_mild_centered_residual_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.000693 | -0.000693 | -0.000693 | 0.882 | 2050.0 |
| memory_energy_drop_square_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.000680 | -0.000680 | -0.000680 | 0.939 | 0.0 |
| rank_competition_memory_suppressed_stopgrad_mask_centered_gate_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.000440 | -0.000440 | -0.000440 | 0.839 | 135170.0 |
| rank_competition_ultrasoft_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.000435 | -0.000435 | -0.000435 | 0.977 | 2.0 |
| hidden_drop_p40_square_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.000303 | -0.000303 | -0.000303 | 0.985 | 0.0 |
| phase_residual_blend_centered_neuron | real | 10160 | 8 | 1024 | 3 | 1 | -0.000280 | -0.001613 | 0.000392 | 0.909 | 2048.0 |
| phase_residual_blend_normed_neuron | real | 10160 | 8 | 1024 | 3 | 2 | -0.000177 | -0.001067 | 0.000749 | 0.882 | 2048.0 |
| rank_competition_fixed_feather_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.000159 | -0.000159 | -0.000159 | 0.972 | 0.0 |
| rank_competition_feather_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.000155 | -0.000155 | -0.000155 | 0.977 | 2.0 |
| phase_residual_blend_half_neuron | real | 10160 | 8 | 1024 | 3 | 2 | -0.000097 | -0.001021 | 0.000953 | 0.923 | 2048.0 |
| rank_competition_memory_suppressed_bounded_centered_gate_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.000095 | -0.000095 | -0.000095 | 0.817 | 135170.0 |
| rank_competition_fixed_trace_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.000070 | -0.000070 | -0.000070 | 0.970 | 0.0 |
| phase_residual_conv_energy_gate_neuron | real | 10160 | 8 | 1024 | 3 | 2 | -0.000065 | -0.000959 | 0.000956 | 0.895 | 2050.0 |
| hidden_drop_low_square_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.000032 | -0.000032 | -0.000032 | 0.978 | 0.0 |
| rank_competition_fixed_dust_neuron | real | 10160 | 8 | 1024 | 1 | 1 | -0.000012 | -0.000012 | -0.000012 | 0.971 | 0.0 |
| rank_competition_fixed_feather_channel_drop_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.000076 | 0.000076 | 0.000076 | 0.964 | 0.0 |
| rank_competition_memory_suppressed_energy_matched_centered_gate_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.000158 | 0.000158 | 0.000158 | 0.791 | 135170.0 |
| channel_drop_square_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.000240 | 0.000240 | 0.000240 | 0.979 | 0.0 |
| rank_competition_memory_suppressed_small_centered_gate_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.000587 | 0.000587 | 0.000587 | 0.824 | 135170.0 |
| rank_competition_memory_suppressed_bounded_small_centered_gate_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.000624 | 0.000624 | 0.000624 | 0.817 | 135170.0 |
| rank_competition_memory_suppressed_normed_centered_gate_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.000638 | 0.000638 | 0.000638 | 0.790 | 135170.0 |
| phase_residual_memory_bounded_gate_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.001059 | 0.001059 | 0.001059 | 0.875 | 135168.0 |
| phase_residual_memory_scalar_gate_neuron | real | 10160 | 8 | 1024 | 3 | 1 | 0.001098 | -0.001846 | 0.004472 | 0.911 | 2178.0 |
| rank_competition_memory_uncertainty_centered_gate_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.001116 | 0.001116 | 0.001116 | 0.816 | 135170.0 |
| phase_residual_memory_scalar_centered_gate_neuron | real | 10160 | 8 | 1024 | 3 | 1 | 0.001157 | -0.001427 | 0.004062 | 0.896 | 2178.0 |
| phase_residual_memory_tiny_gate_neuron | real | 10160 | 8 | 1024 | 2 | 1 | 0.001168 | -0.000383 | 0.002719 | 0.895 | 135168.0 |
| phase_residual_memory_rms_gate_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.001195 | 0.001195 | 0.001195 | 0.845 | 135168.0 |
| rank_competition_soft_centered_residual_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.001213 | 0.001213 | 0.001213 | 0.883 | 2050.0 |
| phase_residual_memory_small_gate_neuron | real | 10160 | 8 | 1024 | 2 | 0 | 0.001669 | 0.000549 | 0.002790 | 0.899 | 135168.0 |
| phase_residual_memory_zero_mean_gate_neuron | real | 10160 | 8 | 1024 | 2 | 0 | 0.001821 | 0.001218 | 0.002423 | 0.888 | 135168.0 |
| memory_square_scalar_gain_neuron | real | 10160 | 8 | 1024 | 2 | 1 | 0.001940 | -0.000185 | 0.004065 | 1.096 | 130.0 |
| rank_competition_memory_small_centered_gate_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.002827 | 0.002827 | 0.002827 | 0.858 | 135170.0 |
| rank_competition_memory_normed_centered_gate_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.003501 | 0.003501 | 0.003501 | 0.780 | 135170.0 |
| rank_competition_memory_within_group_centered_gate_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 0.004106 | 0.004106 | 0.004106 | 0.858 | 135170.0 |
| rank_competition_memory_suppressed_aux_head_centered_gate_neuron | real | 10160 | 8 | 1024 | 1 | 0 | 24.916751 | 24.916751 | 24.916751 | 0.803 | 1183748.0 |
| rank_competition_memory_scalar_centered_gate_neuron | real | 10160 | 8 | 128 | 3 | 3 | -0.034445 | -0.047632 | -0.020808 | 0.893 | 2180.0 |
| rank_competition_centered_residual_neuron | real | 10160 | 8 | 128 | 3 | 3 | -0.026540 | -0.032039 | -0.020528 | 0.912 | 2050.0 |
| rank_competition_memory_factor_centered_gate_neuron | real | 10160 | 8 | 128 | 3 | 3 | -0.024971 | -0.034521 | -0.014710 | 0.882 | 36898.0 |
| rank_competition_memory_group_centered_gate_neuron | real | 10160 | 8 | 128 | 3 | 3 | -0.024090 | -0.048681 | -0.004134 | 0.889 | 4130.0 |
| rank_competition_memory_centered_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.034071 | -0.048253 | -0.026713 | 0.906 | 135170.0 |
| phase_residual_memory_scalar_centered_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.030160 | -0.041701 | -0.015058 | 0.923 | 2178.0 |
| phase_residual_memory_centered_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.025115 | -0.043316 | -0.015662 | 0.911 | 135168.0 |
| phase_residual_memory_novelty_centered_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.025107 | -0.043323 | -0.015650 | 0.858 | 135168.0 |
| phase_residual_memory_sparse_centered_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.025089 | -0.043314 | -0.015635 | 0.874 | 135168.0 |
| phase_residual_memory_bounded_centered_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.025085 | -0.043299 | -0.015638 | 0.908 | 135168.0 |
| phase_residual_blend_centered_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.021207 | -0.023526 | -0.018843 | 0.938 | 2048.0 |
| phase_residual_memory_conv_centered_agreement_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.021203 | -0.023516 | -0.018845 | 0.879 | 2050.0 |
| rank_competition_memory_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.017391 | -0.036077 | -0.007495 | 0.902 | 135170.0 |
| phase_residual_memory_scalar_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.017348 | -0.033925 | -0.000734 | 0.918 | 2178.0 |
| phase_residual_memory_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.014298 | -0.032641 | -0.002748 | 0.921 | 135168.0 |
| phase_residual_memory_rms_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.014296 | -0.032634 | -0.002745 | 0.886 | 135168.0 |
| phase_residual_memory_delta_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.014287 | -0.032641 | -0.002722 | 0.924 | 135168.0 |
| phase_residual_memory_novelty_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.014286 | -0.032648 | -0.002724 | 0.855 | 135168.0 |
| phase_residual_memory_sparse_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.014285 | -0.032639 | -0.002728 | 0.888 | 135168.0 |
| phase_residual_memory_bounded_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.014282 | -0.032635 | -0.002719 | 0.904 | 135168.0 |
| phase_residual_memory_boundary_centered_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.013912 | -0.032014 | -0.004382 | 0.874 | 135168.0 |
| phase_residual_memory_normed_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.012891 | -0.031883 | -0.001213 | 0.883 | 135168.0 |
| phase_residual_memory_group_gate_neuron | real | 10160 | 4 | 128 | 3 | 2 | -0.011229 | -0.029214 | 0.011428 | 0.936 | 4128.0 |
| phase_residual_memory_novelty_tiny_gate_neuron | real | 10160 | 4 | 128 | 3 | 2 | -0.010733 | -0.029869 | 0.000931 | 0.870 | 135168.0 |
| phase_residual_boundary_centered_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.009425 | -0.011020 | -0.007595 | 0.905 | 2048.0 |
| phase_residual_memory_boundary_scalar_gate_neuron | real | 10160 | 4 | 128 | 3 | 2 | -0.008910 | -0.017476 | 0.004625 | 0.898 | 2178.0 |
| phase_residual_memory_conv_disagreement_damp_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.008828 | -0.010192 | -0.006864 | 0.893 | 2050.0 |
| phase_residual_memory_conv_agreement_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.008823 | -0.010174 | -0.006861 | 0.875 | 2050.0 |
| phase_residual_blend_half_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.008820 | -0.010180 | -0.006860 | 0.669 | 2048.0 |
| rank_competition_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.008587 | -0.012573 | -0.003840 | 1.021 | 2.0 |
| phase_residual_memory_boundary_gate_neuron | real | 10160 | 4 | 128 | 3 | 2 | -0.008524 | -0.027659 | 0.002882 | 0.890 | 135168.0 |
| phase_residual_conv_energy_gate_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.008327 | -0.009760 | -0.006278 | 0.920 | 2050.0 |
| phase_residual_blend_normed_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.006816 | -0.008598 | -0.004558 | 0.911 | 2048.0 |
| memory_square_scalar_gain_neuron | real | 10160 | 4 | 128 | 3 | 2 | -0.006738 | -0.015577 | 0.007900 | 1.150 | 130.0 |
| memory_square_conv_agreement_gain_neuron | real | 10160 | 4 | 128 | 3 | 2 | -0.006737 | -0.015666 | 0.008009 | 0.963 | 132.0 |
| memory_square_centered_gain_neuron | real | 10160 | 4 | 128 | 3 | 1 | -0.006051 | -0.024559 | 0.003631 | 1.125 | 133120.0 |
| memory_square_gain_neuron | real | 10160 | 4 | 128 | 3 | 1 | -0.006040 | -0.024550 | 0.003645 | 1.109 | 133120.0 |
| memory_square_small_gain_neuron | real | 10160 | 4 | 128 | 3 | 1 | -0.005988 | -0.024507 | 0.003658 | 1.134 | 133120.0 |
| memory_square_novelty_gain_neuron | real | 10160 | 4 | 128 | 3 | 1 | -0.005986 | -0.024534 | 0.003638 | 0.933 | 133120.0 |
| mem_threshold_neuron | real | 10160 | 4 | 128 | 3 | 1 | -0.005444 | -0.029148 | 0.008071 | 0.970 | 266240.0 |
| phase_residual_blend_large_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.004326 | -0.006116 | -0.001909 | 0.955 | 2048.0 |
| phase_residual_blend_quarter_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.003931 | -0.005562 | -0.001586 | 0.951 | 2048.0 |
| bottleneck_aware_neuron | real | 10160 | 4 | 128 | 3 | 2 | -0.003487 | -0.009336 | 0.007920 | 0.982 | 1247232.0 |
| stateful_threshold_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.003213 | -0.005849 | -0.000854 | 0.905 | 2.0 |
| phase_residual_boundary_blend_neuron | real | 10160 | 4 | 128 | 3 | 3 | -0.002235 | -0.004076 | -0.000075 | 0.898 | 2048.0 |
| phase_residual_blend_neuron | real | 10160 | 4 | 128 | 3 | 2 | -0.001346 | -0.002383 | 0.000276 | 0.938 | 2048.0 |
| stable_competition_neuron | real | 10160 | 4 | 128 | 3 | 2 | -0.001233 | -0.009904 | 0.007415 | 0.949 | 4.0 |
| phase_residual_blend_tiny_neuron | real | 10160 | 4 | 128 | 3 | 2 | -0.000157 | -0.000426 | 0.000188 | 0.957 | 2048.0 |
| conv_disagreement_neuron | real | 10160 | 4 | 128 | 3 | 0 | 0.001366 | 0.000082 | 0.003130 | 0.998 | 0.0 |
| stable_square_neuron | real | 10160 | 4 | 128 | 3 | 0 | 0.004759 | 0.001013 | 0.012243 | 0.965 | 2.0 |
| adaptive_basis_neuron | real | 10160 | 4 | 128 | 3 | 0 | 0.018713 | 0.000306 | 0.030824 | 0.933 | 3078.0 |
| phase_residual_blend_half_neuron | real | 255 | 16 | 1024 | 3 | 3 | -0.002940 | -0.005130 | -0.001666 | 1.005 | 128.0 |
| phase_residual_blend_neuron | real | 255 | 16 | 1024 | 3 | 3 | -0.000713 | -0.001314 | -0.000335 | 0.939 | 128.0 |
| phase_residual_blend_half_neuron | real | 255 | 8 | 256 | 3 | 3 | -0.002260 | -0.002844 | -0.001391 | 1.096 | 128.0 |
| phase_residual_blend_large_neuron | real | 255 | 8 | 256 | 3 | 3 | -0.001274 | -0.001653 | -0.000681 | 1.102 | 128.0 |
| phase_residual_blend_neuron | real | 255 | 8 | 256 | 3 | 3 | -0.000505 | -0.000673 | -0.000356 | 1.084 | 128.0 |
| phase_residual_blend_tiny_neuron | real | 255 | 8 | 256 | 3 | 2 | -0.000047 | -0.000178 | 0.000096 | 1.126 | 128.0 |
| stable_competition_neuron | real | 255 | 8 | 256 | 3 | 0 | 0.010305 | 0.003610 | 0.016978 | 1.031 | 2.0 |
| phase_amplitude_replace_neuron | real | 255 | 8 | 256 | 3 | 1 | 0.022820 | -0.009614 | 0.058672 | 1.099 | 8320.0 |
| phase_amplitude_neuron | real | 255 | 8 | 256 | 3 | 1 | 0.022838 | -0.009628 | 0.058709 | 1.038 | 16640.0 |
| phase_amplitude_neuron | synthetic | 255 | 2 | 256 | 3 | 3 | -0.021119 | -0.046844 | -0.001471 | 1.095 | 16640.0 |
| phase_amplitude_replace_neuron | synthetic | 255 | 2 | 256 | 3 | 3 | -0.021073 | -0.046737 | -0.001455 | 1.074 | 8320.0 |
| phase_amplitude_neutral_neuron | synthetic | 255 | 2 | 256 | 3 | 2 | -0.004366 | -0.010325 | 0.001026 | 1.127 | 0.0 |
| stable_competition_neuron | synthetic | 255 | 2 | 256 | 3 | 2 | -0.004289 | -0.010999 | 0.001746 | 1.031 | 2.0 |
| phase_amplitude_one_extra_neuron | synthetic | 255 | 2 | 256 | 3 | 1 | -0.003059 | -0.011229 | 0.001899 | 1.087 | 8320.0 |
| phase_amplitude_neuron | synthetic | 255 | 2 | 64 | 1 | 1 | -0.002849 | -0.002849 | -0.002849 | 0.878 | 16640.0 |
| stable_competition_neuron | synthetic | 255 | 2 | 64 | 1 | 1 | -0.001258 | -0.001258 | -0.001258 | 0.639 | 2.0 |
| phase_amplitude_neutral_neuron | synthetic | 255 | 2 | 64 | 1 | 1 | -0.001093 | -0.001093 | -0.001093 | 0.767 | 0.0 |
| stable_square_neuron | synthetic | 255 | 2 | 64 | 1 | 1 | -0.000865 | -0.000865 | -0.000865 | 0.482 | 1.0 |
| rank_competition_neuron | synthetic | 255 | 2 | 64 | 1 | 1 | -0.000806 | -0.000806 | -0.000806 | 0.703 | 1.0 |
| conv_disagreement_neuron | synthetic | 255 | 2 | 64 | 1 | 1 | -0.000273 | -0.000273 | -0.000273 | 0.685 | 0.0 |
| phase_amplitude_neuron | synthetic | 255 | 2 | 12 | 1 | 1 | -0.008631 | -0.008631 | -0.008631 | 2.410 | 16640.0 |
| stable_square_neuron | synthetic | 255 | 2 | 12 | 1 | 1 | -0.000890 | -0.000890 | -0.000890 | 1.960 | 1.0 |
| rank_competition_neuron | synthetic | 255 | 2 | 12 | 1 | 1 | -0.000838 | -0.000838 | -0.000838 | 1.549 | 1.0 |
| conv_disagreement_neuron | synthetic | 255 | 2 | 12 | 1 | 1 | -0.000286 | -0.000286 | -0.000286 | 1.247 | 0.0 |
| stateful_threshold_neuron | synthetic | 255 | 2 | 12 | 1 | 1 | -0.000102 | -0.000102 | -0.000102 | 0.810 | 1.0 |
| mem_threshold_neuron | synthetic | 255 | 2 | 12 | 1 | 0 | 0.005462 | 0.005462 | 0.005462 | 1.047 | 3328.0 |
| bottleneck_aware_neuron | synthetic | 255 | 2 | 12 | 1 | 0 | 0.006266 | 0.006266 | 0.006266 | 2.954 | 10624.0 |
| adaptive_basis_neuron | synthetic | 255 | 2 | 12 | 1 | 0 | 0.006277 | 0.006277 | 0.006277 | 1.734 | 195.0 |
| rank_competition_memory_suppressed_aux_head_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.021078 | -0.021078 | -0.021078 | 17.184 | 9986.0 |
| rank_competition_memory_group_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.001180 | -0.001180 | -0.001180 | 0.940 | 337.0 |
| phase_residual_memory_group_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.001091 | -0.001091 | -0.001091 | 0.798 | 336.0 |
| rank_competition_mild_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.000643 | -0.000643 | -0.000643 | 10.181 | 1.0 |
| rank_competition_soft_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.000398 | -0.000398 | -0.000398 | 19.544 | 1.0 |
| rank_competition_centered_residual_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.000170 | -0.000170 | -0.000170 | 1.087 | 129.0 |
| rank_competition_ultrasoft_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.000153 | -0.000153 | -0.000153 | 11.827 | 1.0 |
| rank_competition_mild_centered_residual_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.000122 | -0.000122 | -0.000122 | 15.052 | 129.0 |
| hidden_drop_p40_square_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.000077 | -0.000077 | -0.000077 | 26.975 | 0.0 |
| rank_competition_fixed_feather_channel_drop_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.000076 | -0.000076 | -0.000076 | 22.039 | 0.0 |
| memory_energy_drop_square_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.000031 | -0.000031 | -0.000031 | 4.327 | 0.0 |
| rank_competition_fixed_feather_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.000031 | -0.000031 | -0.000031 | 13.418 | 0.0 |
| rank_competition_feather_neuron | synthetic | 255 | 2 | 1 | 1 | 1 | -0.000015 | -0.000015 | -0.000015 | 18.932 | 1.0 |
| hidden_drop_high_square_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000000 | 0.000000 | 0.000000 | 23.811 | 0.0 |
| hidden_drop_ultra_square_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000000 | 0.000000 | 0.000000 | 17.798 | 0.0 |
| hidden_drop_p35_square_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000015 | 0.000015 | 0.000015 | 23.366 | 0.0 |
| hidden_drop_extreme_square_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000031 | 0.000031 | 0.000031 | 19.444 | 0.0 |
| hidden_drop_p30_square_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000031 | 0.000031 | 0.000031 | 18.850 | 0.0 |
| rank_competition_fixed_trace_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000031 | 0.000031 | 0.000031 | 18.501 | 0.0 |
| rank_competition_fixed_feather_hidden_drop_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000031 | 0.000031 | 0.000031 | 11.622 | 0.0 |
| hidden_drop_square_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000031 | 0.000031 | 0.000031 | 16.535 | 0.0 |
| rank_competition_fixed_dust_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000031 | 0.000031 | 0.000031 | 18.592 | 0.0 |
| hidden_drop_low_square_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000031 | 0.000031 | 0.000031 | 17.220 | 0.0 |
| hidden_drop_p25_square_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000061 | 0.000061 | 0.000061 | 19.958 | 0.0 |
| channel_drop_square_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000061 | 0.000061 | 0.000061 | 11.900 | 0.0 |
| hidden_drop_mid_square_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000076 | 0.000076 | 0.000076 | 21.786 | 0.0 |
| rank_competition_soft_centered_residual_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000214 | 0.000214 | 0.000214 | 19.452 | 129.0 |
| phase_residual_boundary_centered_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000301 | 0.000301 | 0.000301 | 1.117 | 128.0 |
| phase_residual_boundary_blend_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000397 | 0.000397 | 0.000397 | 1.154 | 128.0 |
| phase_residual_blend_quarter_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000422 | 0.000422 | 0.000422 | 1.321 | 128.0 |
| phase_residual_blend_normed_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000631 | 0.000631 | 0.000631 | 1.414 | 128.0 |
| phase_residual_memory_conv_centered_agreement_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000672 | 0.000672 | 0.000672 | 1.179 | 129.0 |
| phase_residual_blend_centered_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000673 | 0.000673 | 0.000673 | 1.525 | 128.0 |
| phase_residual_conv_energy_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000808 | 0.000808 | 0.000808 | 1.322 | 129.0 |
| phase_residual_memory_conv_disagreement_damp_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000859 | 0.000859 | 0.000859 | 1.022 | 129.0 |
| phase_residual_memory_conv_agreement_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.000860 | 0.000860 | 0.000860 | 1.138 | 129.0 |
| phase_residual_memory_novelty_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003877 | 0.003877 | 0.003877 | 1.078 | 1792.0 |
| phase_residual_memory_bounded_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003877 | 0.003877 | 0.003877 | 0.828 | 1792.0 |
| phase_residual_memory_sparse_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003877 | 0.003877 | 0.003877 | 1.087 | 1792.0 |
| phase_residual_memory_delta_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003877 | 0.003877 | 0.003877 | 1.002 | 1792.0 |
| phase_residual_memory_zero_mean_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003877 | 0.003877 | 0.003877 | 1.030 | 1792.0 |
| phase_residual_memory_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003877 | 0.003877 | 0.003877 | 1.309 | 1792.0 |
| phase_residual_memory_rms_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003877 | 0.003877 | 0.003877 | 1.234 | 1792.0 |
| phase_residual_memory_normed_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003920 | 0.003920 | 0.003920 | 0.996 | 1792.0 |
| phase_residual_memory_small_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003978 | 0.003978 | 0.003978 | 1.213 | 1792.0 |
| phase_residual_memory_novelty_tiny_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003978 | 0.003978 | 0.003978 | 1.175 | 1792.0 |
| phase_residual_memory_boundary_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003997 | 0.003997 | 0.003997 | 1.159 | 1792.0 |
| phase_residual_memory_sparse_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003998 | 0.003998 | 0.003998 | 0.833 | 1792.0 |
| phase_residual_memory_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003998 | 0.003998 | 0.003998 | 1.082 | 1792.0 |
| phase_residual_memory_novelty_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003998 | 0.003998 | 0.003998 | 1.194 | 1792.0 |
| phase_residual_memory_bounded_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.003998 | 0.003998 | 0.003998 | 1.053 | 1792.0 |
| rank_competition_memory_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004032 | 0.004032 | 0.004032 | 1.628 | 1793.0 |
| phase_residual_memory_tiny_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004045 | 0.004045 | 0.004045 | 0.952 | 1792.0 |
| phase_residual_memory_boundary_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004056 | 0.004056 | 0.004056 | 1.039 | 1792.0 |
| memory_square_small_gain_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004101 | 0.004101 | 0.004101 | 1.189 | 1664.0 |
| memory_square_gain_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004101 | 0.004101 | 0.004101 | 1.119 | 1664.0 |
| memory_square_centered_gain_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004101 | 0.004101 | 0.004101 | 1.125 | 1664.0 |
| memory_square_novelty_gain_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004101 | 0.004101 | 0.004101 | 1.185 | 1664.0 |
| rank_competition_memory_within_group_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004126 | 0.004126 | 0.004126 | 0.965 | 1793.0 |
| rank_competition_memory_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004158 | 0.004158 | 0.004158 | 1.606 | 1793.0 |
| rank_competition_memory_normed_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004170 | 0.004170 | 0.004170 | 1.074 | 1793.0 |
| rank_competition_memory_small_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004204 | 0.004204 | 0.004204 | 1.170 | 1793.0 |
| rank_competition_memory_suppressed_small_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004243 | 0.004243 | 0.004243 | 6.543 | 1793.0 |
| rank_competition_memory_suppressed_bounded_small_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004305 | 0.004305 | 0.004305 | 18.560 | 1793.0 |
| rank_competition_memory_suppressed_normed_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004335 | 0.004335 | 0.004335 | 10.616 | 1793.0 |
| rank_competition_memory_suppressed_energy_matched_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004335 | 0.004335 | 0.004335 | 7.007 | 1793.0 |
| rank_competition_memory_uncertainty_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004362 | 0.004362 | 0.004362 | 1.174 | 1793.0 |
| rank_competition_memory_suppressed_stopgrad_mask_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004381 | 0.004381 | 0.004381 | 16.730 | 1793.0 |
| rank_competition_memory_suppressed_bounded_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004381 | 0.004381 | 0.004381 | 17.659 | 1793.0 |
| rank_competition_memory_suppressed_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.004387 | 0.004387 | 0.004387 | 1.097 | 1793.0 |
| rank_competition_memory_factor_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.007366 | 0.007366 | 0.007366 | 0.939 | 693.0 |
| rank_competition_memory_scalar_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.011615 | 0.011615 | 0.011615 | 0.957 | 142.0 |
| phase_residual_memory_scalar_centered_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.011738 | 0.011738 | 0.011738 | 1.199 | 141.0 |
| phase_residual_memory_scalar_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.011809 | 0.011809 | 0.011809 | 1.176 | 141.0 |
| memory_square_scalar_gain_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.011828 | 0.011828 | 0.011828 | 1.140 | 13.0 |
| memory_square_conv_agreement_gain_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.011828 | 0.011828 | 0.011828 | 0.939 | 14.0 |
| phase_residual_memory_boundary_scalar_gate_neuron | synthetic | 255 | 2 | 1 | 1 | 0 | 0.011841 | 0.011841 | 0.011841 | 1.052 | 141.0 |

## 40M Seq10160 1024-Step Gate

| seed | val blocks | variant | baseline val | candidate val | val delta | train delta | speed ratio | peak VRAM MB | param delta |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 31 | 8 | channel_drop_square_neuron | 6.490791 | 6.491031 | 0.000240 | 0.046618 | 0.979 | 4282.3 | 0 |
| 31 | 8 | hidden_drop_extreme_square_neuron | 6.490791 | 6.486462 | -0.004329 | 0.096739 | 0.982 | 4300.4 | 0 |
| 31 | 8 | hidden_drop_high_square_neuron | 6.490791 | 6.486917 | -0.003874 | 0.050913 | 0.980 | 4300.4 | 0 |
| 31 | 8 | hidden_drop_low_square_neuron | 6.490791 | 6.490759 | -0.000032 | 0.011531 | 0.978 | 4300.4 | 0 |
| 31 | 8 | hidden_drop_mid_square_neuron | 6.490791 | 6.487987 | -0.002804 | 0.039851 | 0.980 | 4150.9 | 0 |
| 31 | 8 | hidden_drop_p25_square_neuron | 6.490791 | 6.486498 | -0.004293 | 0.111406 | 0.982 | 4150.9 | 0 |
| 31 | 8 | hidden_drop_p30_square_neuron | 6.490791 | 6.487752 | -0.003039 | 0.129694 | 0.981 | 4300.4 | 0 |
| 31 | 8 | hidden_drop_p35_square_neuron | 6.490791 | 6.488544 | -0.002247 | 0.144044 | 0.984 | 4150.9 | 0 |
| 31 | 8 | hidden_drop_p40_square_neuron | 6.490791 | 6.490488 | -0.000303 | 0.159148 | 0.985 | 4300.4 | 0 |
| 31 | 8 | hidden_drop_square_neuron | 6.490791 | 6.488812 | -0.001979 | 0.029357 | 0.978 | 4150.9 | 0 |
| 31 | 8 | hidden_drop_ultra_square_neuron | 6.490791 | 6.485865 | -0.004926 | 0.075429 | 0.982 | 4150.9 | 0 |
| 31 | 8 | memory_energy_drop_square_neuron | 6.490791 | 6.490111 | -0.000680 | 0.031156 | 0.939 | 4525.3 | 0 |
| 13 | 8 | memory_square_scalar_gain_neuron | 6.490744 | 6.494809 | 0.004065 | -0.022828 | 1.120 | 3751.2 | 130 |
| 29 | 8 | memory_square_scalar_gain_neuron | 6.498116 | 6.497931 | -0.000185 | -0.005935 | 1.073 | 3751.1 | 130 |
| 13 | 8 | phase_residual_blend_centered_neuron | 6.490744 | 6.491136 | 0.000392 | -0.009129 | 0.918 | 4069.3 | 2048 |
| 17 | 8 | phase_residual_blend_centered_neuron | 6.493108 | 6.491495 | -0.001613 | 0.003826 | 0.904 | 4069.3 | 2048 |
| 29 | 8 | phase_residual_blend_centered_neuron | 6.498116 | 6.498497 | 0.000381 | -0.009044 | 0.906 | 4069.3 | 2048 |
| 13 | 8 | phase_residual_blend_half_neuron | 6.490744 | 6.491697 | 0.000953 | -0.005627 | 0.926 | 4207.1 | 2048 |
| 17 | 8 | phase_residual_blend_half_neuron | 6.493108 | 6.492087 | -0.001021 | 0.001006 | 0.919 | 4212.1 | 2048 |
| 29 | 8 | phase_residual_blend_half_neuron | 6.498116 | 6.497893 | -0.000223 | -0.002445 | 0.924 | 4209.3 | 2048 |
| 13 | 8 | phase_residual_blend_normed_neuron | 6.490744 | 6.491493 | 0.000749 | -0.004712 | 0.882 | 4753.5 | 2048 |
| 17 | 8 | phase_residual_blend_normed_neuron | 6.493108 | 6.492041 | -0.001067 | 0.000309 | 0.879 | 4759.4 | 2048 |
| 29 | 8 | phase_residual_blend_normed_neuron | 6.498116 | 6.497903 | -0.000213 | -0.002431 | 0.884 | 4756.6 | 2048 |
| 13 | 8 | phase_residual_conv_energy_gate_neuron | 6.490744 | 6.491700 | 0.000956 | -0.005527 | 0.897 | 4794.3 | 2050 |
| 17 | 8 | phase_residual_conv_energy_gate_neuron | 6.493108 | 6.492149 | -0.000959 | 0.000687 | 0.892 | 4800.3 | 2050 |
| 29 | 8 | phase_residual_conv_energy_gate_neuron | 6.498116 | 6.497923 | -0.000193 | -0.002611 | 0.895 | 4797.5 | 2050 |
| 13 | 8 | phase_residual_memory_bounded_gate_neuron | 6.490744 | 6.491803 | 0.001059 | -0.033816 | 0.875 | 4873.1 | 135168 |
| 13 | 8 | phase_residual_memory_centered_gate_neuron | 6.490744 | 6.491577 | 0.000833 | -0.034964 | 0.886 | 4654.7 | 135168 |
| 17 | 8 | phase_residual_memory_centered_gate_neuron | 6.493108 | 6.481018 | -0.012090 | -0.004365 | 0.881 | 4654.7 | 135168 |
| 29 | 8 | phase_residual_memory_centered_gate_neuron | 6.498116 | 6.489002 | -0.009114 | -0.021511 | 0.884 | 4654.7 | 135168 |
| 31 | 8 | phase_residual_memory_centered_gate_neuron | 6.490791 | 6.495628 | 0.004837 | -0.027872 | 0.872 | 4799.0 | 135168 |
| 43 | 8 | phase_residual_memory_centered_gate_neuron | 6.487358 | 6.485026 | -0.002332 | 0.008096 | 0.881 | 4795.7 | 135168 |
| 13 | 8 | phase_residual_memory_gate_neuron | 6.490744 | 6.491959 | 0.001215 | -0.034353 | 0.898 | 4656.5 | 135168 |
| 17 | 8 | phase_residual_memory_gate_neuron | 6.493108 | 6.478981 | -0.014127 | -0.008809 | 0.893 | 4656.5 | 135168 |
| 29 | 8 | phase_residual_memory_gate_neuron | 6.498116 | 6.489211 | -0.008905 | -0.014373 | 0.894 | 4656.5 | 135168 |
| 31 | 8 | phase_residual_memory_gate_neuron | 6.490791 | 6.493194 | 0.002403 | -0.023071 | 0.886 | 4652.4 | 135168 |
| 43 | 8 | phase_residual_memory_gate_neuron | 6.487358 | 6.486451 | -0.000907 | 0.011914 | 0.894 | 4652.5 | 135168 |
| 13 | 8 | phase_residual_memory_rms_gate_neuron | 6.490744 | 6.491939 | 0.001195 | -0.034204 | 0.845 | 4913.9 | 135168 |
| 13 | 8 | phase_residual_memory_scalar_centered_gate_neuron | 6.490744 | 6.494806 | 0.004062 | -0.030021 | 0.902 | 4149.4 | 2178 |
| 17 | 8 | phase_residual_memory_scalar_centered_gate_neuron | 6.493108 | 6.493943 | 0.000835 | -0.047475 | 0.892 | 4149.4 | 2178 |
| 29 | 8 | phase_residual_memory_scalar_centered_gate_neuron | 6.498116 | 6.496689 | -0.001427 | -0.002527 | 0.894 | 4149.4 | 2178 |
| 13 | 8 | phase_residual_memory_scalar_gate_neuron | 6.490744 | 6.495216 | 0.004472 | -0.020620 | 0.913 | 4755.0 | 2178 |
| 17 | 8 | phase_residual_memory_scalar_gate_neuron | 6.493108 | 6.493776 | 0.000668 | -0.047395 | 0.908 | 4761.9 | 2178 |
| 29 | 8 | phase_residual_memory_scalar_gate_neuron | 6.498116 | 6.496270 | -0.001846 | -0.004212 | 0.911 | 4758.1 | 2178 |
| 13 | 8 | phase_residual_memory_small_gate_neuron | 6.490744 | 6.491292 | 0.000549 | -0.033736 | 0.910 | 4193.0 | 135168 |
| 31 | 8 | phase_residual_memory_small_gate_neuron | 6.490791 | 6.493581 | 0.002790 | -0.019648 | 0.887 | 4193.0 | 135168 |
| 13 | 8 | phase_residual_memory_tiny_gate_neuron | 6.490744 | 6.490361 | -0.000383 | -0.032923 | 0.901 | 4655.6 | 135168 |
| 31 | 8 | phase_residual_memory_tiny_gate_neuron | 6.490791 | 6.493511 | 0.002719 | -0.018039 | 0.889 | 4655.6 | 135168 |
| 13 | 8 | phase_residual_memory_zero_mean_gate_neuron | 6.490744 | 6.491962 | 0.001218 | -0.034364 | 0.894 | 4799.5 | 135168 |
| 31 | 8 | phase_residual_memory_zero_mean_gate_neuron | 6.490791 | 6.493214 | 0.002423 | -0.023129 | 0.881 | 4806.3 | 135168 |
| 13 | 8 | rank_competition_centered_residual_neuron | 6.490744 | 6.486664 | -0.004080 | -0.018986 | 0.886 | 4713.1 | 2050 |
| 17 | 8 | rank_competition_centered_residual_neuron | 6.493108 | 6.488046 | -0.005062 | -0.005932 | 0.893 | 4717.9 | 2050 |
| 29 | 8 | rank_competition_centered_residual_neuron | 6.498116 | 6.495302 | -0.002814 | -0.021266 | 0.895 | 4715.2 | 2050 |
| 31 | 8 | rank_competition_feather_neuron | 6.490791 | 6.490636 | -0.000155 | -0.000175 | 0.977 | 4321.4 | 2 |
| 31 | 8 | rank_competition_fixed_dust_neuron | 6.490791 | 6.490779 | -0.000012 | 0.000067 | 0.971 | 4318.1 | 0 |
| 31 | 8 | rank_competition_fixed_feather_channel_drop_neuron | 6.490791 | 6.490867 | 0.000076 | 0.046030 | 0.964 | 4318.4 | 0 |
| 31 | 8 | rank_competition_fixed_feather_hidden_drop_neuron | 6.490791 | 6.488661 | -0.002130 | 0.029151 | 0.964 | 4338.0 | 0 |
| 31 | 8 | rank_competition_fixed_feather_neuron | 6.490791 | 6.490632 | -0.000159 | -0.000176 | 0.972 | 4170.9 | 0 |
| 31 | 8 | rank_competition_fixed_trace_neuron | 6.490791 | 6.490722 | -0.000070 | 0.000009 | 0.970 | 4321.1 | 0 |
| 13 | 8 | rank_competition_memory_centered_gate_neuron | 6.490744 | 6.488335 | -0.002409 | -0.045860 | 0.877 | 4231.8 | 135170 |
| 17 | 8 | rank_competition_memory_centered_gate_neuron | 6.493108 | 6.478446 | -0.014662 | -0.014524 | 0.868 | 4231.8 | 135170 |
| 29 | 8 | rank_competition_memory_centered_gate_neuron | 6.498116 | 6.486530 | -0.011585 | -0.022380 | 0.870 | 4231.8 | 135170 |
| 31 | 8 | rank_competition_memory_centered_gate_neuron | 6.490791 | 6.494787 | 0.003996 | -0.031864 | 0.859 | 4691.6 | 135170 |
| 43 | 8 | rank_competition_memory_centered_gate_neuron | 6.487358 | 6.482294 | -0.005064 | -0.001104 | 0.870 | 4691.6 | 135170 |
| 31 | 8 | rank_competition_memory_normed_centered_gate_neuron | 6.490791 | 6.494293 | 0.003501 | -0.029362 | 0.780 | 5040.0 | 135170 |
| 13 | 8 | rank_competition_memory_scalar_centered_gate_neuron | 6.490744 | 6.492793 | 0.002049 | -0.035562 | 0.872 | 4651.2 | 2180 |
| 17 | 8 | rank_competition_memory_scalar_centered_gate_neuron | 6.493108 | 6.490094 | -0.003014 | -0.058647 | 0.879 | 4651.0 | 2180 |
| 29 | 8 | rank_competition_memory_scalar_centered_gate_neuron | 6.498116 | 6.496504 | -0.001612 | -0.014090 | 0.881 | 4651.1 | 2180 |
| 31 | 8 | rank_competition_memory_small_centered_gate_neuron | 6.490791 | 6.493618 | 0.002827 | -0.027966 | 0.858 | 4691.6 | 135170 |
| 31 | 8 | rank_competition_memory_suppressed_aux_head_centered_gate_neuron | 6.490791 | 31.407543 | 24.916751 | 16.731038 | 0.803 | 5065.8 | 1183748 |
| 31 | 8 | rank_competition_memory_suppressed_bounded_centered_gate_neuron | 6.490791 | 6.490696 | -0.000095 | -0.018425 | 0.817 | 4999.2 | 135170 |
| 31 | 8 | rank_competition_memory_suppressed_bounded_small_centered_gate_neuron | 6.490791 | 6.491415 | 0.000624 | -0.020679 | 0.817 | 4998.4 | 135170 |
| 13 | 8 | rank_competition_memory_suppressed_centered_gate_neuron | 6.490744 | 6.482527 | -0.008217 | -0.031756 | 0.847 | 4392.1 | 135170 |
| 17 | 8 | rank_competition_memory_suppressed_centered_gate_neuron | 6.493108 | 6.474004 | -0.019104 | -0.026080 | 0.836 | 4392.1 | 135170 |
| 29 | 8 | rank_competition_memory_suppressed_centered_gate_neuron | 6.498116 | 6.481968 | -0.016147 | -0.001595 | 0.839 | 4392.1 | 135170 |
| 31 | 8 | rank_competition_memory_suppressed_centered_gate_neuron | 6.490791 | 6.490654 | -0.000137 | -0.018128 | 0.828 | 4999.2 | 135170 |
| 43 | 8 | rank_competition_memory_suppressed_centered_gate_neuron | 6.487358 | 6.478144 | -0.009214 | 0.003342 | 0.839 | 4392.1 | 135170 |
| 31 | 8 | rank_competition_memory_suppressed_energy_matched_centered_gate_neuron | 6.490791 | 6.490949 | 0.000158 | -0.019517 | 0.791 | 4931.2 | 135170 |
| 31 | 8 | rank_competition_memory_suppressed_normed_centered_gate_neuron | 6.490791 | 6.491429 | 0.000638 | -0.019341 | 0.790 | 5076.8 | 135170 |
| 31 | 8 | rank_competition_memory_suppressed_small_centered_gate_neuron | 6.490791 | 6.491378 | 0.000587 | -0.020516 | 0.824 | 4851.8 | 135170 |
| 31 | 8 | rank_competition_memory_suppressed_stopgrad_mask_centered_gate_neuron | 6.490791 | 6.490351 | -0.000440 | -0.020765 | 0.839 | 4959.2 | 135170 |
| 31 | 8 | rank_competition_memory_uncertainty_centered_gate_neuron | 6.490791 | 6.491907 | 0.001116 | -0.022945 | 0.816 | 4930.9 | 135170 |
| 31 | 8 | rank_competition_memory_within_group_centered_gate_neuron | 6.490791 | 6.494897 | 0.004106 | -0.032350 | 0.858 | 4840.5 | 135170 |
| 31 | 8 | rank_competition_mild_centered_residual_neuron | 6.490791 | 6.490098 | -0.000693 | -0.009747 | 0.882 | 4716.9 | 2050 |
| 31 | 8 | rank_competition_mild_neuron | 6.490791 | 6.488403 | -0.002389 | -0.002588 | 0.974 | 4171.2 | 2 |
| 13 | 8 | rank_competition_neuron | 6.490744 | 6.486382 | -0.004362 | -0.009203 | 0.997 | 3711.4 | 2 |
| 17 | 8 | rank_competition_neuron | 6.493108 | 6.489904 | -0.003204 | -0.012473 | 0.984 | 3711.4 | 2 |
| 29 | 8 | rank_competition_neuron | 6.498116 | 6.493997 | -0.004118 | -0.001002 | 0.988 | 3711.4 | 2 |
| 31 | 8 | rank_competition_soft_centered_residual_neuron | 6.490791 | 6.492004 | 0.001213 | -0.009362 | 0.883 | 4716.6 | 2050 |
| 31 | 8 | rank_competition_soft_neuron | 6.490791 | 6.489584 | -0.001207 | -0.001250 | 0.973 | 4321.4 | 2 |
| 31 | 8 | rank_competition_ultrasoft_neuron | 6.490791 | 6.490356 | -0.000435 | 0.000016 | 0.977 | 4171.2 | 2 |

## Failed Or Non-Finite Runs

| group | seed | seq | val blocks | steps | variant | baseline val | candidate val | speed ratio | peak VRAM MB | result |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| screen_2048_seed31_real_seq10160_val8_seed31_suppressed_2048_v1 | 31 | 10160 | 8 | 2048 | rank_competition_memory_suppressed_centered_gate_neuron | 6.788070 |  | 0.843 | 4851.8 | `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\screen_2048_seed31_real_seq10160_val8_seed31_suppressed_2048_v1_rank_competition_memory_suppressed_centered_gate_neuron\result.json` |

## Decisions

- Reject full phase-amplitude replacement for scale-up: real-text seeds were unstable.
- Reject stable competition for scale-up: synthetic gains did not transfer to real FineWebEdu.
- Reject cheap scalar memory gates for scale-up: strong 128-step wins did not survive the 1024-step gate.
- Reject memory-square gain variants for scale-up: they are fast, but long 1024-step quality regressed on seed 13.
- Reject memory-novelty variants for now: 128-step quality matched equivalent raw-memory gates but with lower speed.
- Reject memory-conv agreement variants for scale-up: the centered agreement path duplicated the simpler centered residual result and was slower.
- Reject boundary-local residual variants for scale-up: 128-step wins were weaker than centered residual and memory-centered gates.
- Reject constrained memory-gate variants for scale-up: the apparent val4 win was a validation-block mismatch artifact; val8 verification on seed 13 lost to baseline for bounded and RMS gates.
- Keep `rank_competition_neuron` as a robust cheap survivor: val8 1024-step seeds 13/17/29 were 3/3 wins, mean val delta -0.003895, mean speed ratio about 0.990, and only +2 params.
- Demote `rank_competition_memory_centered_gate_neuron`: val8 1024-step seeds 13/17/29/31/43 were 4/5 wins, mean val delta -0.005945, best -0.014662, worst +0.003996, mean speed ratio about 0.869, and +135170 params.
- Keep `rank_competition_memory_suppressed_centered_gate_neuron` as the 1024-step research leader, not a scale-up candidate: val8 1024-step seeds 13/17/29/31/43 were 5/5 wins, mean val delta -0.010564, best -0.019104, worst -0.000137, mean speed ratio about 0.838, and +135170 params, but the 2048-step weak-seed confirmation diverged to NaN.
- Reject `rank_competition_memory_scalar_centered_gate_neuron` for scale-up: it was the best 128-step val8 cost ablation, but the 1024-step gate fell to 2/3 wins, mean val delta -0.000859, and a seed 13 regression.
- Keep `rank_competition_centered_residual_neuron` as a cheap modest survivor: val8 1024-step seeds 13/17/29 were 3/3 wins, mean val delta -0.003985, but it is only slightly better than `rank_competition_neuron` and much slower.
- Reject group and factorized memory-centered rank gates for now: 128-step screens were 3/3 wins but worse than the scalar/no-memory cost ablations at higher parameter or speed cost.
- Reject `rank_competition_memory_small_centered_gate_neuron` and `rank_competition_memory_normed_centered_gate_neuron` for now: both targeted the seed 31 failure, but 1024-step seed 31 still regressed by +0.002827 and +0.003501 respectively.
- Reject `rank_competition_memory_uncertainty_centered_gate_neuron` and `rank_competition_memory_within_group_centered_gate_neuron` for now: both failed the focused 1024-step seed 31 gate.
- Reject the first stable-suppressed second iterations for scale-up: small-gain, bounded-small, and RMS-limited routes lost the 1024-step seed 31 gate; the bounded route barely won at 1024 but catastrophically regressed at 2048 with val delta +269.588360.
- Reject `rank_competition_memory_suppressed_stopgrad_mask_centered_gate_neuron` for scale-up: detaching the suppression mask fixed the 2048 divergence mode and improved the seed 31 1024 margin to -0.000440, but 2048 validation still regressed by +0.008068.
- Reject `rank_competition_memory_suppressed_aux_head_centered_gate_neuron`: the separate residual head diverged by the 1024-step gate, with val delta +24.916751 and +1183748 params.
- Keep `rank_competition_mild_neuron` and `rank_competition_soft_neuron` as cheap 1024-only pressure-curve survivors, not scale-up candidates: seed 31 1024 deltas were -0.002389 and -0.001207 at +2 params and about 0.97 speed ratio, but 2048 deltas were +0.007532 and +0.003544.
- Reject the softer pressure endpoints for scale-up: `rank_competition_ultrasoft_neuron` and `rank_competition_feather_neuron` stayed stable and near baseline at 2048, but still lost by +0.000931 and +0.000121 respectively.
- Reject fixed-pressure rank competition for scale-up: freezing the feather pressure removed the learned inhibition drift and added 0 params, but the 2048 weak-seed confirmation still lost by +0.000111; fixed trace and dust were weaker already at 1024.
- Do not start the 5B-token run or queue this for 3080 scale-up yet. This round found real 1024-step improvements and one stability fix for the suppressed-memory route, but no candidate has beaten the matched 2048 weak-seed baseline.
- Next ablation: move away from adding signed memory-route capacity and test block-local regularizers or data-scale confirmations that can reduce the repeated lower-train/worse-val pattern without changing tokenizer, objective, or metric.

## Commands Used

Latest 3-seed 1024-step hybrid gate:

```powershell
$cache='E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\real_cache_finewebedu_sample_seq10160_train192_val8_gpt2.pt'
$env:CUDA_VISIBLE_DEVICES='0'
$env:MANUAL_SEARCH_CACHE=$cache
$env:MANUAL_SEARCH_SEQUENCE_LENGTH='10160'
$env:MANUAL_SEARCH_VAL_BLOCKS='8'
$env:MANUAL_SEARCH_STEPS='1024'
$env:MANUAL_SEARCH_TAG='real_seq10160_40m_1024'
$env:MANUAL_SEARCH_BLOCKS='rank_competition_memory_centered_gate_neuron'
$env:MANUAL_SEARCH_REUSE_EXISTING='1'
$env:MANUAL_SEARCH_EMBEDDING_DIM='512'
$env:MANUAL_SEARCH_CONV_LAYERS='2'
$env:MANUAL_SEARCH_CONV_KERNEL_SIZE='7'
$env:MANUAL_SEARCH_CONV_RANK='192'
$env:MANUAL_SEARCH_MEMORY_RANK='64'
$env:MANUAL_SEARCH_LANDMARK_STRIDE='128'
$env:MANUAL_SEARCH_SAMPLED_VOCAB_SIZE='24576'
$env:MANUAL_SEARCH_TOKEN_STRIDE='4'
$env:MANUAL_SEARCH_TOKEN_CHUNK_SIZE='512'
$env:MANUAL_SEARCH_FULL_EVAL_TOKEN_CHUNK_SIZE='1024'
foreach($seed in 13,17,29){
  $env:MANUAL_SEARCH_SEED=[string]$seed
  python -u 'E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\manual_neuron_search.py'
}
```

Aggregation and verification:

```powershell
python -u 'E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\aggregate_manual_results.py'
python -B -m py_compile 'E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\manual_neuron_search.py' 'E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\aggregate_manual_results.py'
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

## Generated Files

- `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\aggregate_results_20260605_current.json`
- `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\aggregate_results_20260605_current.csv`
- `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\loss_delta_1024_key_variants.png`
