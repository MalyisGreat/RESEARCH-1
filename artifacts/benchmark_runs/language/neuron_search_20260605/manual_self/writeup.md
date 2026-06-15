# Manual Neuron Search

Trainer: `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\standalone_longseq_anchor_train.py`
Cache: `E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\real_cache_finewebedu_sample_seq10160_train192_val8_gpt2.pt`
Steps per screen: `2048`
Seed: `31`

## Baseline

```json
{
  "block_type": "multi_scale_lowrank_conv_memory",
  "command": "trainer.train({'cache_path': WindowsPath('E:/CODEXRESEARCH/RESEARCH-1/artifacts/benchmark_runs/language/neuron_search_20260605/manual_self/real_cache_finewebedu_sample_seq10160_train192_val8_gpt2.pt'), 'output_dir': WindowsPath('E:/CODEXRESEARCH/RESEARCH-1/artifacts/benchmark_runs/language/neuron_search_20260605/manual_self/screen_2048_seed31_real_seq10160_val8_seed31_hidden_drop_p35_2048_v1_baseline'), 'run_name': 'screen_2048_seed31_real_seq10160_val8_seed31_hidden_drop_p35_2048_v1_baseline', 'vocab_size': 50257, 'sequence_length': 10160, 'batch_size': 1, 'train_steps': 2048, 'eval_interval': 2048, 'val_blocks': 8, 'checkpoint_interval': 0, 'milestone_checkpoint_interval': 0, 'seed': 31, 'embedding_dim': 512, 'block_type': 'multi_scale_lowrank_conv_memory', 'conv_layers': 2, 'conv_rank': 192, 'conv_kernel_size': 7, 'memory_rank': 64, 'attention_heads': 4, 'landmark_stride': 128, 'sampled_vocab_size': 24576, 'token_stride': 4, 'token_chunk_size': 512, 'full_eval_token_chunk_size': 1024, 'learning_rate': 0.0006, 'min_learning_rate': 1e-05, 'warmup_steps': 512, 'weight_decay': 0.0001, 'amp_dtype': 'fp16', 'resume_checkpoint': None})",
  "elapsed_s": 253.63746410000022,
  "final_train_loss": 4.497857570648193,
  "final_val_loss": 6.788070307097097,
  "ok": true,
  "output_dir": "E:\\CODEXRESEARCH\\RESEARCH-1\\artifacts\\benchmark_runs\\language\\neuron_search_20260605\\manual_self\\screen_2048_seed31_real_seq10160_val8_seed31_hidden_drop_p35_2048_v1_baseline",
  "parameter_count": 40437009,
  "peak_vram_mb": 3672.59912109375,
  "pure_train_tok_per_sec": 83436.35336503015,
  "result_path": "E:\\CODEXRESEARCH\\RESEARCH-1\\artifacts\\benchmark_runs\\language\\neuron_search_20260605\\manual_self\\screen_2048_seed31_real_seq10160_val8_seed31_hidden_drop_p35_2048_v1_baseline\\result.json"
}
```

## Ranked Variants

### hidden_drop_p35_square_neuron

- Design: p=0.35 hidden-dropout squared neuron
- Equations: `h=dropout(relu(Wf norm(x))^2, p=0.35 during train); y=ffn_out(h)`
- Why: The 2048 curve improved through p=0.30 even as 1024 weakened, so this tests whether later generalization still benefits from heavier hidden thinning.
- Novelty: Very high train-time activation thinning endpoint for the squared-neuron conv-memory FFN.
- Metrics: `{"block_type": "hidden_drop_p35_square_neuron", "final_train_loss": 4.783977508544922, "final_val_loss": 6.6740689311440535, "grad_ok": true, "ok": true, "param_delta": 0, "parameter_count": 40437009, "speed_ratio_vs_baseline": 0.9712497055956684, "tok_per_sec": 81037.53364176169, "val_delta_vs_baseline": -0.11400137595304383}`
- Decision: keep for longer local screen.
