$ErrorActionPreference = 'Stop'
$repo = 'E:\CODEXRESEARCH'
$root = Join-Path $repo 'RESEARCH-1\artifacts\benchmark_runs\language'
$cache = Join-Path $root 'longseq_anchor16_40m_600m_20260602\cache\finewebedu_train600088338_val325152_seq10160_gpt2.pt'
$script = Join-Path $root 'standalone_longseq_anchor_train.py'
$outRoot = Join-Path $root 'wave5_landmark_attention_20260604'
$runName = 'wave5_host2080_landmark_stride4_60m'
$out = Join-Path $outRoot $runName
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
Write-Host "WAVE5_LOCAL_START run=$runName host=$env:COMPUTERNAME cache=$cache"
python -u $script `
  --cache-path $cache `
  --output-dir $out `
  --run-name $runName `
  --train-steps 5906 `
  --eval-interval 5906 `
  --checkpoint-interval 5906 `
  --milestone-checkpoint-interval 0 `
  --val-blocks 24 `
  --embedding-dim 640 `
  --block-type landmark_attention `
  --conv-layers 2 `
  --conv-kernel-size 7 `
  --conv-rank 224 `
  --attention-heads 4 `
  --landmark-stride 128 `
  --sampled-vocab-size 32768 `
  --token-stride 4 `
  --warmup-steps 500 `
  --learning-rate 0.0006 `
  --min-learning-rate 0.00001
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "WAVE5_LOCAL_RESULT_PATH $out\result.json"
if (Test-Path -LiteralPath (Join-Path $out 'result.json')) {
  Get-Content -LiteralPath (Join-Path $out 'result.json') -Raw
}
Write-Host "WAVE5_LOCAL_DONE run=$runName"
