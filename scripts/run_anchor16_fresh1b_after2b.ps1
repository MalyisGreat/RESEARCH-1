$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUNBUFFERED = '1'
$env:TOKENIZERS_PARALLELISM = 'true'

$repo = 'E:\CODEXRESEARCH\RESEARCH-1'
$runtimeCacheRoot = Join-Path $repo 'artifacts\runtime_cache'
$runtimeTempRoot = Join-Path $repo 'artifacts\tmp'
New-Item -ItemType Directory -Force -Path `
  $runtimeCacheRoot, `
  (Join-Path $runtimeCacheRoot 'hf_home'), `
  (Join-Path $runtimeCacheRoot 'hf_datasets'), `
  (Join-Path $runtimeCacheRoot 'hf_hub'), `
  (Join-Path $runtimeCacheRoot 'hf_xet'), `
  (Join-Path $runtimeCacheRoot 'torch'), `
  (Join-Path $runtimeCacheRoot 'transformers'), `
  $runtimeTempRoot | Out-Null
$env:HF_HOME = Join-Path $runtimeCacheRoot 'hf_home'
$env:HF_DATASETS_CACHE = Join-Path $runtimeCacheRoot 'hf_datasets'
$env:HF_HUB_CACHE = Join-Path $runtimeCacheRoot 'hf_hub'
$env:HF_XET_CACHE = Join-Path $runtimeCacheRoot 'hf_xet'
$env:TORCH_HOME = Join-Path $runtimeCacheRoot 'torch'
$env:TRANSFORMERS_CACHE = Join-Path $runtimeCacheRoot 'transformers'
$env:XDG_CACHE_HOME = Join-Path $runtimeCacheRoot 'xdg'
$env:TEMP = $runtimeTempRoot
$env:TMP = $runtimeTempRoot
$python = 'C:\Users\joshj\miniconda3\python.exe'
$oldCache = Join-Path $repo 'artifacts\benchmark_runs\language\longseq_anchor16_80m_2b_20260603\cache\finewebedu_train2000203011_val325152_seq10160_gpt2.pt'
$resumeCheckpoint = Join-Path $repo 'artifacts\benchmark_runs\language\longseq_anchor16_80m_3b_lr1e3_continue_20260603\variant_results\seq10160_steps295276_val32_batch1_eval9843_init1_seed13\anchor16_2p205b_resume.pt'

$outDir = Join-Path $repo 'artifacts\benchmark_runs\language\longseq_anchor16_80m_fresh1b_after2b_20260603'
$cachePath = Join-Path $outDir 'cache\fresh_after2b_plus_valskip_train1000106586_val325152_seq10160_gpt2.pt'
$variantDir = Join-Path $outDir 'variant_results\seq10160_steps315426_val32_batch1_eval9843_init1_seed13'
$outputJson = Join-Path $outDir 'language_longseq_anchor16_80m_fresh1b_after2b_seq10160_seed13_20260603.json'

$sourceTrainTokenOffset = 2000528163
$freshTrainBlocks = 98426
$globalTrainSteps = 315426

New-Item -ItemType Directory -Force -Path $variantDir | Out-Null
Copy-Item -LiteralPath $resumeCheckpoint -Destination (Join-Path $variantDir 'causal_conv_mixer_sampled_vocab_anchor16.checkpoint.pt') -Force

Set-Location $repo
Write-Host "FRESH1B_AFTER2B_START $(Get-Date -Format o)"
Write-Host "resume_checkpoint=$resumeCheckpoint"
Write-Host "output_dir=$outDir"
Write-Host "cache_path=$cachePath"
Write-Host "source_train_token_offset=$sourceTrainTokenOffset"
Write-Host "fresh_train_blocks=$freshTrainBlocks global_train_steps=$globalTrainSteps"
Write-Host "runtime_cache_root=$runtimeCacheRoot"
Write-Host "runtime_temp_root=$runtimeTempRoot"

$progressPath = "$($cachePath).skip_progress.json"
$maxCacheAttempts = 80
$code = 1
for ($attempt = 1; $attempt -le $maxCacheAttempts; $attempt++) {
  Write-Host "FRESH1B_AFTER2B_ATTEMPT=$attempt $(Get-Date -Format o)"
  if (Test-Path $progressPath) {
    Write-Host "FRESH1B_AFTER2B_SKIP_PROGRESS $(Get-Content $progressPath -Raw)"
  }

  & $python -u -m arc_tactic3.language_longseq_replay_probe `
    --output-dir $outDir `
    --cache-path $cachePath `
    --validation-cache-path $oldCache `
    --train-token-offset $sourceTrainTokenOffset `
    --train-blocks $freshTrainBlocks `
    --val-blocks 32 `
    --sequence-length 10160 `
    --batch-size 1 `
    --eval-batch-size 1 `
    --train-steps $globalTrainSteps `
    --eval-interval 9843 `
    --eval-loss-mode full `
    --max-gpu-used-mb 2000 `
    --max-step-seconds 600 `
    --max-eval-seconds 1200 `
    --resume-variant-checkpoints `
    --resume-fresh-cache `
    --variant-checkpoint-interval 1000 `
    --milestone-checkpoint-interval 98426 `
    --train-log-interval 100 `
    --learning-rate 0.001 `
    --sampled-vocab-size 16384 `
    --full-loss-token-stride 8 `
    --full-eval-token-chunk-size 1024 `
    --train-loss-token-chunk-size 1024 `
    --conv-embedding-dim 1034 `
    --conv-layers 1 `
    --conv-rank 295 `
    --tokenization-batch-size 2048 `
    --no-pin-memory `
    --no-cache-dataset-on-device `
    --variants causal_conv_mixer_sampled_vocab_anchor16 `
    --output $outputJson

  $code = $LASTEXITCODE
  Write-Host "FRESH1B_AFTER2B_ATTEMPT_EXIT=$code $(Get-Date -Format o)"
  if ($code -eq 0) {
    Write-Host "FRESH1B_AFTER2B_EXIT=0 $(Get-Date -Format o)"
    exit 0
  }
  if (Test-Path $cachePath) {
    Write-Host "FRESH1B_AFTER2B_CACHE_EXISTS_AFTER_FAILURE_NOT_RETRYING"
    break
  }
  if ($attempt -lt $maxCacheAttempts) {
    Start-Sleep -Seconds 10
  }
}

Write-Host "FRESH1B_AFTER2B_EXIT=$code $(Get-Date -Format o)"
exit $code
