# Commands

All commands were run from `E:\CODEXRESEARCH` with `CUDA_VISIBLE_DEVICES=-1`.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; python -m py_compile E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\adaptive_basis_train.py E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\smoke_grad_params.py
```

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; python E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\smoke_grad_params.py
```

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; python -m py_compile E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\adaptive_basis_train.py; if($LASTEXITCODE -ne 0){ exit $LASTEXITCODE }; $out='E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\baseline'; New-Item -ItemType Directory -Force -Path $out | Out-Null; python E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\adaptive_basis_train.py --cache-path E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\screen_cache_synth_seq255_train768_val64_gpt2.pt --output-dir $out --run-name agent03_baseline_multiscale_lowrank_cpu64 --sequence-length 255 --seed 13 --train-steps 64 --eval-interval 64 --checkpoint-interval 0 --milestone-checkpoint-interval 0 --val-blocks 8 --embedding-dim 192 --block-type multi_scale_lowrank_conv_memory --conv-layers 2 --conv-kernel-size 7 --conv-rank 96 --memory-rank 32 --landmark-stride 64 --sampled-vocab-size 4096 --token-stride 4 --token-chunk-size 512 --full-eval-token-chunk-size 512 --learning-rate 0.0006 --min-learning-rate 0.00001 2>&1 | Tee-Object -FilePath (Join-Path $out 'run.log'); exit $LASTEXITCODE
```

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; $out='E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\candidate_v1'; New-Item -ItemType Directory -Force -Path $out | Out-Null; python E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\adaptive_basis_train.py --cache-path E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\screen_cache_synth_seq255_train768_val64_gpt2.pt --output-dir $out --run-name agent03_adaptive_basis_cpu64 --sequence-length 255 --seed 13 --train-steps 64 --eval-interval 64 --checkpoint-interval 0 --milestone-checkpoint-interval 0 --val-blocks 8 --embedding-dim 192 --block-type multi_scale_lowrank_conv_memory_adaptive_basis --conv-layers 2 --conv-kernel-size 7 --conv-rank 96 --memory-rank 32 --landmark-stride 64 --sampled-vocab-size 4096 --token-stride 4 --token-chunk-size 512 --full-eval-token-chunk-size 512 --learning-rate 0.0006 --min-learning-rate 0.00001 2>&1 | Tee-Object -FilePath (Join-Path $out 'run.log'); exit $LASTEXITCODE
```

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; python -m py_compile E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\adaptive_basis_train.py E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\smoke_grad_params.py
```

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; python E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\smoke_grad_params.py
```

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; $out='E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\candidate_v2'; New-Item -ItemType Directory -Force -Path $out | Out-Null; python E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent03_adaptive_basis\adaptive_basis_train.py --cache-path E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\screen_cache_synth_seq255_train768_val64_gpt2.pt --output-dir $out --run-name agent03_ffn_adaptive_basis_cpu64 --sequence-length 255 --seed 13 --train-steps 64 --eval-interval 64 --checkpoint-interval 0 --milestone-checkpoint-interval 0 --val-blocks 8 --embedding-dim 192 --block-type multi_scale_lowrank_conv_memory_ffn_adaptive_basis --conv-layers 2 --conv-kernel-size 7 --conv-rank 96 --memory-rank 32 --landmark-stride 64 --sampled-vocab-size 4096 --token-stride 4 --token-chunk-size 512 --full-eval-token-chunk-size 512 --learning-rate 0.0006 --min-learning-rate 0.00001 2>&1 | Tee-Object -FilePath (Join-Path $out 'run.log'); exit $LASTEXITCODE
```
