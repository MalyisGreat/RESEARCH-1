# Commands Used

```powershell
New-Item -ItemType Directory -Force -Path E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned
Copy-Item -LiteralPath E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\standalone_longseq_anchor_train.py -Destination E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\standalone_longseq_anchor_train_conv_conditioned.py -Force
```

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; python -m py_compile E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\standalone_longseq_anchor_train_conv_conditioned.py 2>&1 | Tee-Object -FilePath E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\py_compile.log
```

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; python E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\smoke_grad_params.py 2>&1 | Tee-Object -FilePath E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\smoke_grad_params.log
```

```powershell
New-Item -ItemType Directory -Force -Path E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\baseline,E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\candidate_v1,E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\candidate_v2
```

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; python E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\standalone_longseq_anchor_train_conv_conditioned.py --cache-path E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\screen_cache_synth_seq255_train768_val64_gpt2.pt --output-dir E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\baseline --run-name agent02_baseline_64 --block-type multi_scale_lowrank_conv_memory --sequence-length 255 --train-steps 64 --eval-interval 64 --checkpoint-interval 0 --milestone-checkpoint-interval 0 --val-blocks 8 --embedding-dim 192 --conv-layers 2 --conv-kernel-size 7 --conv-rank 96 --memory-rank 32 --landmark-stride 64 --sampled-vocab-size 4096 --token-stride 4 --token-chunk-size 512 --full-eval-token-chunk-size 512 --learning-rate 0.0006 --min-learning-rate 0.00001 2>&1 | Tee-Object -FilePath E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\baseline\screen.log
```

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; python E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\standalone_longseq_anchor_train_conv_conditioned.py --cache-path E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\screen_cache_synth_seq255_train768_val64_gpt2.pt --output-dir E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\candidate_v1 --run-name agent02_candidate_v1_64 --block-type branch_disagreement_conditioned_lowrank_conv_memory --sequence-length 255 --train-steps 64 --eval-interval 64 --checkpoint-interval 0 --milestone-checkpoint-interval 0 --val-blocks 8 --embedding-dim 192 --conv-layers 2 --conv-kernel-size 7 --conv-rank 96 --memory-rank 32 --landmark-stride 64 --sampled-vocab-size 4096 --token-stride 4 --token-chunk-size 512 --full-eval-token-chunk-size 512 --learning-rate 0.0006 --min-learning-rate 0.00001 2>&1 | Tee-Object -FilePath E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\candidate_v1\screen.log
```

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; python E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\standalone_longseq_anchor_train_conv_conditioned.py --cache-path E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\screen_cache_synth_seq255_train768_val64_gpt2.pt --output-dir E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\candidate_v2 --run-name agent02_candidate_v2_gain_only_64 --block-type branch_disagreement_gain_only_lowrank_conv_memory --sequence-length 255 --train-steps 64 --eval-interval 64 --checkpoint-interval 0 --milestone-checkpoint-interval 0 --val-blocks 8 --embedding-dim 192 --conv-layers 2 --conv-kernel-size 7 --conv-rank 96 --memory-rank 32 --landmark-stride 64 --sampled-vocab-size 4096 --token-stride 4 --token-chunk-size 512 --full-eval-token-chunk-size 512 --learning-rate 0.0006 --min-learning-rate 0.00001 2>&1 | Tee-Object -FilePath E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\agent02_conv_conditioned\candidate_v2\screen.log
```

Failed setup attempts:

```text
First baseline screen launch failed before training because baseline\screen.log parent directory did not exist.
Second baseline screen launch failed before training because the isolated copy did not expose --sequence-length yet.
```
