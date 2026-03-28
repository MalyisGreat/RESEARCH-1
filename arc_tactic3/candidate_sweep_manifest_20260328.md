# Candidate Sweep Manifest

This manifest lists the five self-contained candidate branches added for later one-at-a-time benchmarking.

## Status

- Candidate files present in `E:\DEVNEW\arc_tactic3`
- Candidate smoke tests present in `E:\DEVNEW\arc_tactic3\tests`
- Combined smoke verification passed:
  - `python -m pytest E:\DEVNEW\arc_tactic3\tests\test_language_candidate_local_global_memory.py E:\DEVNEW\arc_tactic3\tests\test_language_candidate_learned_compressor.py E:\DEVNEW\arc_tactic3\tests\test_language_candidate_slot_memory.py E:\DEVNEW\arc_tactic3\tests\test_language_candidate_rollout_objective.py E:\DEVNEW\arc_tactic3\tests\test_language_candidate_dynamic_token_basis.py -q --rootdir E:\DEVNEW\arc_tactic3`

## Recommended Sweep Order

1. `language_candidate_local_global_memory.py`
   - First architecture test for exact recent memory plus compressed older memory.
   - Main knobs:
     - `--local-window`
     - `--older-chunk-size`
     - `--partial-untied-tokens`

2. `language_candidate_learned_compressor.py`
   - Tests whether long-term compression should be learned instead of fixed chunk mean pooling.
   - Main knobs:
     - `--chunk-size`
     - `--partial-untied-tokens`

3. `language_candidate_slot_memory.py`
   - Structural alternative: persistent learned memory slots instead of token/chunk traces.
   - Main knobs:
     - `--slot-count`
     - `--slot-dim`

4. `language_candidate_dynamic_token_basis.py`
   - Tests adaptive token-memory bandwidth instead of a fixed top-token subset.
   - Main knobs:
     - `--token-basis-rank`
     - `--routing-experts`
     - `--routing-top-k`

5. `language_candidate_rollout_objective.py`
   - Objective-only test: CE vs sequence-focused vs rollout-enhanced training.
   - Main knobs:
     - `--objectives`
     - `--rollout-prefix-length`

## Cheap First Pass Commands

### 1. Local/Global Memory

```powershell
python -m arc_tactic3.language_candidate_local_global_memory `
  --cache-path E:\DEVNEW\arc_tactic3\benchmark_runs\fineweb_edu_first20m_gpt2tokens_cache.pt `
  --train-blocks 1024 `
  --val-blocks 64 `
  --train-steps 16 `
  --eval-interval 8 `
  --sequence-length 127 `
  --seed 13 `
  --device cuda `
  --local-window 32 `
  --older-chunk-size 8 `
  --partial-untied-tokens 2048 `
  --output E:\DEVNEW\arc_tactic3\benchmark_runs\language_candidate_local_global_memory_cheap.json
```

### 2. Learned Compressor

```powershell
python E:\DEVNEW\arc_tactic3\language_candidate_learned_compressor.py `
  --cache-path E:\DEVNEW\arc_tactic3\benchmark_runs\fineweb_edu_first20m_gpt2tokens_cache.pt `
  --train-blocks 1024 `
  --val-blocks 64 `
  --train-steps 16 `
  --eval-interval 8 `
  --chunk-size 8 `
  --partial-untied-tokens 1024 `
  --output E:\DEVNEW\arc_tactic3\benchmark_runs\language_candidate_learned_compressor_smoke.json
```

### 3. Slot Memory

```powershell
python -m arc_tactic3.language_candidate_slot_memory `
  --cache-path E:\DEVNEW\arc_tactic3\benchmark_runs\fineweb_edu_first20m_gpt2tokens_cache.pt `
  --train-blocks 1024 `
  --val-blocks 64 `
  --sequence-length 127 `
  --train-steps 16 `
  --eval-interval 8 `
  --seed 13 `
  --device cuda `
  --recurrent-embedding-dim 144 `
  --recurrent-hidden-dim 288 `
  --recurrent-memory-dim 144 `
  --partial-untied-tokens 1024 `
  --slot-count 8 `
  --slot-dim 144 `
  --output E:\DEVNEW\arc_tactic3\benchmark_runs\language_candidate_slot_memory_cheap.json
```

### 4. Dynamic Token Basis

```powershell
python E:\DEVNEW\arc_tactic3\language_candidate_dynamic_token_basis.py `
  --cache-path E:\DEVNEW\arc_tactic3\benchmark_runs\fineweb_edu_first20m_gpt2tokens_cache.pt `
  --train-blocks 1024 `
  --val-blocks 64 `
  --train-steps 16 `
  --eval-interval 8 `
  --seed 13 `
  --device cuda `
  --partial-untied-tokens 512 `
  --token-basis-rank 48 `
  --routing-experts 4 `
  --routing-top-k 2 `
  --output E:\DEVNEW\arc_tactic3\benchmark_runs\language_candidate_dynamic_token_basis_seed13.json
```

### 5. Rollout / Objective Sweep

```powershell
python E:\DEVNEW\arc_tactic3\language_candidate_rollout_objective.py --cache-path E:\DEVNEW\arc_tactic3\benchmark_runs\fineweb_edu_first20m_gpt2tokens_cache.pt --model-variant partial_untied --train-blocks 1024 --val-blocks 64 --train-steps 16 --eval-interval 8 --sequence-length 127 --batch-size 16 --eval-batch-size 32 --rollout-prefix-length 8 --objectives ce_only ce_plus_sequence ce_plus_rollout ce_plus_both --output E:\DEVNEW\arc_tactic3\benchmark_runs\language_candidate_rollout_partial_cheap_20260328.json
```

## Suggested Later Hold Order

1. Local/Global Memory
2. Learned Compressor
3. Dynamic Token Basis
4. Slot Memory
5. Objective sweep on `partial_untied` and the best surviving architecture candidate
