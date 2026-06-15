$ErrorActionPreference = 'Stop'
$Root = 'E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language'
$RunRoot = Join-Path $Root 'wave9_multiscale_memory_20260604'
$Script = Join-Path $Root 'standalone_longseq_anchor_train.py'
$Cache = Join-Path $Root 'longseq_anchor16_40m_600m_20260602\cache\finewebedu_train600088338_val325152_seq10160_gpt2.pt'
$Out = Join-Path $RunRoot 'wave9_host2080_multiscale_memory_stride4_60m'
$RunName = 'wave9_host2080_multiscale_memory_stride4_60m'

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
Write-Host "WAVE9_MS_MEMORY_LOCAL_START run=$RunName host=$env:COMPUTERNAME cache=$Cache"
python -u $Script `
  --cache-path $Cache `
  --output-dir $Out `
  --run-name $RunName `
  --train-steps 5906 `
  --eval-interval 5906 `
  --checkpoint-interval 5906 `
  --milestone-checkpoint-interval 0 `
  --val-blocks 24 `
  --embedding-dim 640 `
  --block-type multi_scale_memory `
  --conv-layers 2 `
  --conv-kernel-size 7 `
  --conv-rank 224 `
  --memory-rank 64 `
  --sampled-vocab-size 32768 `
  --token-stride 4 `
  --warmup-steps 500 `
  --learning-rate 0.0006 `
  --min-learning-rate 0.00001
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "WAVE9_MS_MEMORY_LOCAL_RESULT_PATH $Out\result.json"
if (Test-Path -LiteralPath (Join-Path $Out 'result.json')) {
  Get-Content -LiteralPath (Join-Path $Out 'result.json') -Raw
}
Write-Host "WAVE9_MS_MEMORY_LOCAL_DONE run=$RunName"
