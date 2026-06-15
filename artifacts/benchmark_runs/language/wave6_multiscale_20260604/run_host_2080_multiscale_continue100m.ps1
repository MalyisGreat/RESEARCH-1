$ErrorActionPreference = 'Stop'
$repo = 'E:\CODEXRESEARCH'
$root = Join-Path $repo 'RESEARCH-1\artifacts\benchmark_runs\language'
$cache = Join-Path $root 'longseq_anchor16_40m_600m_20260602\cache\finewebedu_train600088338_val325152_seq10160_gpt2.pt'
$script = Join-Path $root 'standalone_longseq_anchor_train.py'
$outRoot = Join-Path $root 'wave6_multiscale_20260604'
$resume = Join-Path $outRoot 'wave6_host2080_multiscale_stride4_60m\checkpoint.pt'
$runName = 'wave6_host2080_multiscale_stride4_continue100m'
$out = Join-Path $outRoot $runName
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
if (!(Test-Path -LiteralPath $resume)) { throw "resume checkpoint missing: $resume" }
Write-Host "WAVE6_LOCAL_CONT_START run=$runName host=$env:COMPUTERNAME cache=$cache resume=$resume"
python -u $script `
  --cache-path $cache `
  --output-dir $out `
  --run-name $runName `
  --train-steps 9843 `
  --eval-interval 1969 `
  --checkpoint-interval 1969 `
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
  --resume-checkpoint $resume
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "WAVE6_LOCAL_CONT_RESULT_PATH $out\result.json"
if (Test-Path -LiteralPath (Join-Path $out 'result.json')) {
  Get-Content -LiteralPath (Join-Path $out 'result.json') -Raw
}
Write-Host "WAVE6_LOCAL_CONT_DONE run=$runName"
