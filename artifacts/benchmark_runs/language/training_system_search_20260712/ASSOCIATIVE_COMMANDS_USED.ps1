$root = 'E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\training_system_search_20260712'
$cache = 'E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\real_cache_finewebedu_sample_seq10160_train192_val8_gpt2.pt'

$common = @(
  '--cache-path', $cache,
  '--embedding-dim', '512',
  '--block-type', 'multi_scale_lowrank_conv_memory',
  '--conv-layers', '2',
  '--conv-rank', '192',
  '--memory-rank', '64',
  '--recall-mode', 'factor_recall_gated_multiscale',
  '--recall-initial-scale', '256',
  '--sampled-vocab-size', '4096',
  '--token-stride', '24',
  '--token-chunk-size', '20000',
  '--full-eval-token-chunk-size', '512',
  '--val-blocks', '8',
  '--checkpoint-interval', '0',
  '--learning-rate', '0.0006',
  '--min-learning-rate', '0.00001',
  '--warmup-steps', '64',
  '--weight-decay', '0.0001',
  '--amp-dtype', 'fp16'
)

# Final combined 350-step screen. Change seed/output together for seed 17.
$env:PHRASE_ORDERS = '2,3,4'
$env:PHRASE_HISTORY = '1'
$env:SEMANTIC_TABLES = '2'
$env:SEMANTIC_CANDIDATES = '3'
$name = 'phrase234_semantic_t2k3_stride24_350_seed13'
python -u "$root\phrase_semantic_induction_train.py" @common `
  --output-dir "$root\$name" --run-name $name `
  --train-steps 350 --eval-interval 350 --seed 13

# Phrase-only ablation.
$name = 'phrase_induction_orders234_stride24_350_seed13'
python -u "$root\phrase_induction_train.py" @common `
  --output-dir "$root\$name" --run-name $name `
  --train-steps 350 --eval-interval 350 --seed 13

# Longer combined order-2+3 confirmation used in the report.
$env:PHRASE_ORDERS = '2,3'
$name = 'phrase23_semantic_t2k3_stride24_1000_seed13'
python -u "$root\phrase_semantic_induction_train.py" @common `
  --output-dir "$root\$name" --run-name $name `
  --train-steps 1000 --eval-interval 250 --seed 13

# Trained-checkpoint interventions.
python -u "$root\evaluate_associative_breakthrough.py" `
  --checkpoint "$root\phrase23_semantic_t2k3_stride24_1000_seed13\checkpoint.pt" `
  --cache $cache `
  --output "$root\associative_breakthrough_interventions_1000_seed13.json" `
  --val-blocks 8
