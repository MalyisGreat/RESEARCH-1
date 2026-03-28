# Negative Results

This file exists to preserve the parts of the search that did **not** hold up.

## Candidate Families That Did Not Promote

### Long-memory compression branches

- `local_global_memory`
- `learned_compressor`
- chunked token-memory rewrites

Why they matter:

- several of these branches showed real cheap-probe gains
- most of them regressed or flattened under medium or long holds
- this is part of the evidence that exact-token memory remained important in the current regime

Representative artifacts:

- [`language_candidate_local_global_memory_cheap.json`](../artifacts/benchmark_runs/language/language_candidate_local_global_memory_cheap.json)
- [`language_candidate_learned_compressor_smoke.json`](../artifacts/benchmark_runs/language/language_candidate_learned_compressor_smoke.json)
- [`language_recurrent_memory_rewrites_tokenmemory_cheap_20260328.json`](../artifacts/benchmark_runs/language/language_recurrent_memory_rewrites_tokenmemory_cheap_20260328.json)
- [`language_recurrent_hold_compare_long_20260328.json`](../artifacts/benchmark_runs/language/language_recurrent_hold_compare_long_20260328.json)

### Memory-slot and token-basis branches

- `slot_memory`
- `dynamic_token_basis`

These did not become headline directions because they were weak or inconclusive relative to `partial_untied`.

Artifacts:

- [`language_candidate_slot_memory_cheap.json`](../artifacts/benchmark_runs/language/language_candidate_slot_memory_cheap.json)
- [`language_candidate_dynamic_token_basis_seed13.json`](../artifacts/benchmark_runs/language/language_candidate_dynamic_token_basis_seed13.json)

### GPU-compact / dense-value memory rewrites

These runs were useful because they clarified the bottleneck, but they did not replace the champion:

- `dense_value_window32`
- `dense_partial_window32`
- compact shortlist-memory variants

Artifacts:

- [`language_gpu_targeted_small_20260328.json`](../artifacts/benchmark_runs/language/language_gpu_targeted_small_20260328.json)
- [`language_gpu_targeted_medium_20260328.json`](../artifacts/benchmark_runs/language/language_gpu_targeted_medium_20260328.json)
- [`language_gpu_targeted_hold_20260328.json`](../artifacts/benchmark_runs/language/language_gpu_targeted_hold_20260328.json)
- [`language_gpu_compact_candidates_hold2seed_20260328.json`](../artifacts/benchmark_runs/language/language_gpu_compact_candidates_hold2seed_20260328.json)

## Why Keep Negative Results?

Because they are part of the actual result:

- short-budget improvements were often not reliable
- speed-only changes often damaged longer-horizon quality
- better forward-time GPU behavior did not automatically improve training efficiency

The repo is stronger if these failures stay visible.
