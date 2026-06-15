$ErrorActionPreference = 'Stop'
$Root = 'E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language'
$RunRoot = Join-Path $Root 'wave6_multiscale_20260604'
$Script = Join-Path $Root 'standalone_longseq_anchor_train.py'
$Cache = Join-Path $Root 'longseq_anchor16_40m_600m_20260602\cache\finewebedu_train600088338_val325152_seq10160_gpt2.pt'
$Resume = Join-Path $RunRoot 'wave6_host2080_multiscale_stride4_continue100m\checkpoint.pt'
$Out = Join-Path $RunRoot 'wave6_host2080_multiscale_stride4_continue300m'
$RunName = 'wave6_host2080_multiscale_stride4_continue300m'

Write-Host "WAVE6_LOCAL_CONT300_START run=$RunName host=$env:COMPUTERNAME cache=$Cache resume=$Resume"
python -u $Script `
  --cache-path $Cache `
  --output-dir $Out `
  --run-name $RunName `
  --train-steps 29529 `
  --eval-interval 9843 `
  --checkpoint-interval 9843 `
  --milestone-checkpoint-interval 0 `
  --val-blocks 24 `
  --embedding-dim 640 `
  --block-type multi_scale `
  --conv-layers 2 `
  --conv-kernel-size 7 `
  --conv-rank 224 `
  --sampled-vocab-size 32768 `
  --token-stride 4 `
  --warmup-steps 500 `
  --learning-rate 0.0006 `
  --min-learning-rate 0.00001 `
  --resume-checkpoint $Resume
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "WAVE6_LOCAL_CONT300_RESULT_PATH $Out\result.json"
if (Test-Path -LiteralPath (Join-Path $Out 'result.json')) {
  Get-Content -LiteralPath (Join-Path $Out 'result.json') -Raw
}
Write-Host "WAVE6_LOCAL_CONT300_DONE run=$RunName"
