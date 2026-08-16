# 2026-06-16 2B Resume LR Probe

All four schedules resumed the same 76.2M parameter `multi_scale_lowrank_conv_memory` checkpoint on the RTX 3080 only.

- Source checkpoint: `D:\CodexLLM\research1_longseq\runs\wave10_3080_lowrank_conv_memory_76m_3b_scratch_existingcache_20260605\checkpoint.step196860_tokens2000097600.pt`
- Cache: `D:\CodexLLM\research1_longseq\cache\finewebedu_fresh_after2b_train3000289275_val325152_seq10160_gpt2.pt`
- Trainer wrapper: `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\lr_schedule_probe_trainer_20260616.py`
- Target worker: `mwstroud-mwstr-6aea1cf3`
- GPU guard observed for every run: `GPU_NAME=NVIDIA GeForce RTX 3080`
- Resume: step 196860, 2,000,097,600 tokens
- End: step 206703, 2,100,102,480 tokens
- Tokens added: 100,004,880
- Shared config: seq 10160, sampled vocab 32768, token stride 4, val blocks 32, same cache and seed/resume state

## Ranked Results

| rank | schedule | job | val @ 200000 | val @ 205000 | final val | final train | final LR | tok/s | peak VRAM MB |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | aggressive_4e4_floor_1e4 | `20260616-002458-f85b5d` | 4.2246595992582 | 4.17313960372932 | 4.16321746832508 | 4.04121685028076 | 0.0001 | 92619.8446062377 | 5425.7177734375 |
| 2 | control_smart_decay | `20260616-002449-c29f66` | 4.18412526076234 | 4.17872215209045 | 4.17599690357062 | 4.05567932128906 | 0.000192094221891567 | 92619.5147585609 | 5425.7177734375 |
| 3 | flat_2e4_slow_decay | `20260616-002455-23981e` | 4.18417013743966 | 4.17923646030698 | 4.17634209405015 | 4.0559344291687 | 0.000193205228233665 | 92620.441940305 | 5425.7177734375 |
| 4 | high_floor_1e4 | `20260616-002452-e7d7ff` | 4.19165617344886 | 4.18826157833178 | 4.18449506799536 | 4.06350612640381 | 0.000225582221994184 | 92620.6404079662 | 5425.7177734375 |

## LR Shapes Actually Used

| schedule | resume LR | first-step LR | midpoint LR | final LR |
|---|---:|---:|---:|---:|
| control_smart_decay | 0.00020085837648600258 | 0.0002008574947750793 | 0.00019649649960513802 | 0.00019209422189156653 |
| high_floor_1e4 | 0.00023162646654207079 | 0.00023162585846557196 | 0.0002286182755897504 | 0.00022558222199418383 |
| flat_2e4_slow_decay | 0.0001 | 0.0001002 | 0.00019842040356936517 | 0.0001932052282336645 |
| aggressive_4e4_floor_1e4 | 0.0001 | 0.0001006 | 0.0002625694352099171 | 0.0001 |

## Interpretation

The aggressive schedule is the only clear final-loss winner in this 100M-token screen. It was not monotonically better: its first eval at step 200000 was much worse than the others, but after LR cooled it reached the best mid-run and final validation loss. That makes it promising, but it needs an ablation before a full 5B continuation because the win could be from the late cool-to-1e-4 phase rather than the 4e-4 burst.

The control schedule barely improved over the 2.0B checkpoint in this short window. Flat-ish was essentially tied with control and does not justify replacing it. High-floor was consistently worse and should be killed unless there is a separate reason to value it.

## Recommended Next Probe

Run a 200M-300M confirmation from the same 2.0B checkpoint with:

1. `aggressive_4e4_floor_1e4` repeated exactly.
2. A no-burst ablation: warm/hold near 2e-4 briefly, then cool to 1e-4 over the same 100M window.
3. A gentler burst: warm to 3e-4, cool to 1e-4.

Do not launch the full 5B run from this result alone. The aggressive schedule is the lead candidate, but the early validation damage and late recovery need isolation.

## Launcher Pattern

Each job downloaded the base trainer and `lr_schedule_probe_trainer_20260616.py`, set `LR_PROBE_MODE`, then ran:

```powershell
python -u $wrapper `
  --cache-path $cache `
  --output-dir $out `
  --run-name $runName `
  --train-steps 206703 `
  --eval-interval 5000 `
  --checkpoint-interval 0 `
  --milestone-checkpoint-interval 0 `
  --val-blocks 32 `
  --embedding-dim 896 `
  --block-type multi_scale_lowrank_conv_memory `
  --conv-layers 2 `
  --conv-kernel-size 7 `
  --conv-rank 320 `
  --memory-rank 64 `
  --landmark-stride 128 `
  --sampled-vocab-size 32768 `
  --token-stride 4 `
  --warmup-steps 2000 `
  --learning-rate 0.0003 `
  --min-learning-rate 0.00001 `
  --resume-checkpoint $ckpt
```
