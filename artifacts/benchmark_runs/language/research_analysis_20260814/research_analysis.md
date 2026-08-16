# Complete Research Analysis

Generated 2026-08-14 from the repository's result JSON, JSONL metrics, CSV curves, and recognized logs.

## Coverage

- 1723 JSON files, 3 JSONL files, and 33 CSV files were inspected.
- 3467 deduplicated validation-loss points form 976 unique curves.
- 107 unique multi-checkpoint experiments are shown together in the fully opaque overlay.
- 107 distinct experiments with at least three checkpoints are rendered across 7 atlas pages.
- Regime counts: long_context_lm=424, short_lm=123, neuron_search=423, synthetic_fastlearn=0, other=6.
- Raw caches and checkpoints were not loaded; their result metadata remains represented through result files and manifests.

## Quality Rules

- All curves are preserved in `all_points.csv`; no curve was silently dropped because it used a different tokenizer or objective.
- Regime colors are descriptive, not evidence that losses are directly comparable across regimes.
- The causal scaling frontier is only a descriptive best-so-far envelope, not a fitted scaling law.
- Some rows are derived checkpoint-history CSVs or previously extracted curve tables; they are retained for provenance but should not be counted as independent training runs.

## Findings

- The strongest long-context result is Wave10 76M low-rank conv-memory at about 4.0899 validation loss around 4.013B tokens. Its direct run reaches 4.1517 at 2.2B, followed by three separately recorded continuation segments through 4.013B.
- A prior comparison CSV merged those Wave10 continuation segments with a different 160M dense-anchor run. The analysis now groups CSV rows by original run source; the apparent 4.09-to-4.58 zig-zag was a data-joining error, not model deterioration.
- The 160M dense anchor reaches 4.5841 at 5B tokens and 4.6422 at 2B, so the larger dense line did not beat the best low-rank conv-memory checkpoint.
- The earlier 8M local short-context comparison remains a separate regime: partial-untied is about 5.33 versus the NanoChat-inspired mini-port about 5.40 at 50M cached-token exposures.
- Synthetic fast-learning screens favor fast_gru over the GPT2-like control at every tested small/medium/large scale, but those are adaptation metrics, not language-model validation loss.
- Neuron-search deltas include strong one-seed rows and a repeated three-seed hidden-drop-square signal; they remain short screens and are not scale-cleared architecture wins.

## Long-Run Ranking

| Curve | Regime | Tokens | Final loss | Best loss | Params | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `wave10_76m_to5b_loss_so_far_20260615 [20260615-141307-266267]` | long_context_lm | 4013200000 | 4.0899 | 4.0899 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260615-141307-266267\mwstroud-mwstr-6aea1cf3.log` |
| `wave10_76m_actual_loss_so_far_20260615 [20260615-133335-509e45]` | long_context_lm | 3048000000 | 4.1240 | 4.1240 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260615-133335-509e45\mwstroud-mwstr-6aea1cf3.log` |
| `wave10_76m_to5b_loss_so_far_20260615 [20260605-024617-ebeaa2]` | long_context_lm | 2200107360 | 4.1517 | 4.1517 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260605-024617-ebeaa2\mwstroud-mwstr-6aea1cf3.log` |
| `wave10_76m_to5b_loss_so_far_20260615 [20260614-235439-ba54d5]` | long_context_lm | 2540000000 | 4.1525 | 4.1523 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260614-235439-ba54d5\mwstroud-mwstr-6aea1cf3.log` |
| `wave10_76m_to5b_clean_loss_so_far_20260615 [20260604-022628-5c855b]` | long_context_lm | 5000000160 | 4.5841 | 4.5841 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260604-022628-5c855b\mwstroud-mwstr-6aea1cf3.log` |
| `longseq_anchor16_160m_2b_cosine_20260603_restart2 [20260603-181136-f3f589]` | long_context_lm | 2000006160 | 4.6422 | 4.6422 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260603-181136-f3f589\mwstroud-mwstr-6aea1cf3.log` |
| `wave10_host2080_lowrank_conv_memory_stride4_continue300m_shifted_to_3080_retry1 [20260605-011144-50d666]` | long_context_lm | 300014640 | 4.6809 | 4.6809 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260605-011144-50d666\mwstroud-mwstr-6aea1cf3.log` |
| `wave6_3080_multiscale_stride4_continue500m_total49215_resume_after_cuda [20260604-205451-0ffce6]` | long_context_lm | 500024400 | 4.7431 | 4.7431 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260604-205451-0ffce6\mwstroud-mwstr-6aea1cf3.log` |
| `wave4_3080_dense_stride4_fresh_continue500m [20260604-180347-2a8730]` | long_context_lm | 500024400 | 4.7682 | 4.7682 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260604-180347-2a8730\mwstroud-mwstr-6aea1cf3.log` |
| `wave6_3080_multiscale_stride4_continue500m_total49215 [20260604-202043-ffd781]` | long_context_lm | 400019520 | 4.7744 | 4.7744 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260604-202043-ffd781\mwstroud-mwstr-6aea1cf3.log` |
| `wave6_4060_multiscale_stride4_continue300m [20260604-191304-3fe8cd]` | long_context_lm | 300014640 | 4.7892 | 4.7892 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260604-191304-3fe8cd\t-mwstr-07260bd4.log` |
| `wave2_3080_dense_stride8_90m_continue500m [20260604-163118-2e4251]` | long_context_lm | 500024400 | 4.7943 | 4.7943 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260604-163118-2e4251\mwstroud-mwstr-6aea1cf3.log` |
| `language_longseq_anchor16_80m_fresh1b_after2b_seq10160_seed13_20260603/causal_conv_mixer_sampled_vocab_anchor16` | long_context_lm | 3204728160 | 4.8059 | 4.7715 | 79999460.0 | `artifacts\benchmark_runs\language\longseq_anchor16_80m_fresh1b_after2b_20260603\language_longseq_anchor16_80m_fresh1b_after2b_seq10160_seed13_20260603.json` |
| `wave10_host2080_lowrank_conv_memory_stride4_continue300m` | long_context_lm | 200009760 | 4.8113 | 4.8113 |  | `artifacts\benchmark_runs\language\wave10_lowrank_conv_memory_20260604\host_2080_lowrank_conv_memory_stride4_continue300m.stdout.log` |
| `wave6_3080_multiscale_stride4_continue300m [20260604-193841-a8232f]` | long_context_lm | 300014640 | 4.8140 | 4.8140 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260604-193841-a8232f\mwstroud-mwstr-6aea1cf3.log` |
| `language_longseq_anchor16_80m_2b_lr1e3_seq10160_seed13_20260603/causal_conv_mixer_sampled_vocab_anchor16` | long_context_lm | 2000006160 | 4.8262 | 4.8169 | 79999460.0 | `artifacts\benchmark_runs\language\language_longseq_anchor16_80m_2b_lr1e3_seq10160_seed13_20260603.json` |
| `wave6_host2080_multiscale_stride4_continue300m` | long_context_lm | 300014640 | 4.8792 | 4.8792 |  | `artifacts\benchmark_runs\language\wave6_multiscale_20260604\host_2080_multiscale_continue300m.stdout.log` |
| `wave6_host2080_multiscale_stride4_continue300m/standalone_longseq_anchor_train` | long_context_lm | 300014640 | 4.8792 | 4.8792 | 51037745.0 | `artifacts\benchmark_runs\language\wave6_multiscale_20260604\wave6_host2080_multiscale_stride4_continue300m\result.json` |
| `wave4_4060_dense_stride4_continue500m [20260604-191304-307977]` | long_context_lm | 399999200 | 4.8873 | 4.8873 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260604-191304-307977\t-mwstr-07260bd4.log` |
| `longseq_anchor16_80m_600m_2060_20260603_1845 [20260603-184409-9ceb40]` | long_context_lm | 600029280 | 4.9125 | 4.9125 |  | `E:\CODEXRESEARCH\house_compute_hub\runs\20260603-184409-9ceb40\Chichen-JoshS-10a81805.log` |

## Outputs

- `clean_long_context_scaling.png`: primary, publication-style comparison of the three mature comparable LM families.
- `all_experiments_opaque_overlay.png`: every unique multi-checkpoint experiment on one large, fully opaque graph.
- `experiment_curve_atlas.pdf`: every unique experiment with at least three observed validation checkpoints, one opaque curve per panel.
- `experiment_curve_atlas_pages/` and `experiment_curve_atlas_index.csv`: page images and source index for the atlas.
- `all_curves_validation_loss_vs_tokens.png`: every extracted multi-point curve, colored by regime.
- `all_curves_validation_loss_vs_tokens_clipped.png`: readable loss view clipped at 12; raw points remain in the CSV.
- `major_curves_validation_loss_vs_tokens.png`: long-running curves with readable annotations.
- `all_final_loss_vs_tokens.png`: final-loss comparison across every curve.
- `final_loss_vs_parameters.png`: parameter-scale comparison where parameter metadata exists.
- `throughput_vs_final_loss.png`: speed/quality tradeoff where throughput metadata exists.
- `empirical_quality_frontier.png`: descriptive best-so-far frontier.
- `fastlearn_scaling.png` and `fastlearn_scaling.csv`: synthetic adaptation scaling curves.
- `neuron_search_tradeoff.png` and `neuron_search_summary.csv`: neuron-search quality/speed curves.
- `artifact_inventory.csv`: every research artifact file with size and heavy-artifact classification; caches/checkpoints are inventoried but not loaded.
- `all_points.csv`, `curve_summary.csv`, and `quality_audit.json`: inspectable source tables and QA metadata.
