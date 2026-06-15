$ErrorActionPreference = "Stop"

$root = "E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language"
$script = Join-Path $root "standalone_longseq_anchor_train.py"
$cache = Join-Path $root "longseq_anchor16_40m_600m_20260602\cache\finewebedu_train600088338_val325152_seq10160_gpt2.pt"
$runName = "wave3_host2080_dense_stride4_60m"
$out = Join-Path $root (Join-Path "wave3_gated_block_20260604" $runName)

if (!(Test-Path -LiteralPath $script)) {
    throw "trainer missing: $script"
}
if (!(Test-Path -LiteralPath $cache)) {
    throw "cache missing: $cache"
}

Write-Host "WAVE3_LOCAL_START run=$runName host=$env:COMPUTERNAME cache=$cache"
python -u $script `
    --cache-path $cache `
    --output-dir $out `
    --run-name $runName `
    --train-steps 5906 `
    --eval-interval 5906 `
    --checkpoint-interval 5906 `
    --milestone-checkpoint-interval 0 `
    --val-blocks 24 `
    --embedding-dim 704 `
    --block-type relu_square `
    --conv-layers 2 `
    --conv-kernel-size 7 `
    --conv-rank 224 `
    --sampled-vocab-size 32768 `
    --token-stride 4 `
    --warmup-steps 500 `
    --learning-rate 0.0006 `
    --min-learning-rate 0.00001
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Write-Host "WAVE3_LOCAL_RESULT_PATH $out\result.json"
if (Test-Path -LiteralPath (Join-Path $out "result.json")) {
    Get-Content -LiteralPath (Join-Path $out "result.json") -Raw
}
Write-Host "WAVE3_LOCAL_DONE run=$runName"
