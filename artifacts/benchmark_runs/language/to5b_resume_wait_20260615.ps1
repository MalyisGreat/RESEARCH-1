$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUNBUFFERED = '1'
$env:CUDA_VISIBLE_DEVICES = '0'

$root = 'D:\CodexLLM\research1_longseq'
$sourceRunName = 'wave10_3080_lowrank_conv_memory_76m_3b_scratch_existingcache_20260605'
$runName = 'wave10_3080_lowrank_conv_memory_76m_to5b_fresh_after2b_wait_resume_20260615'
$oldInlineRunName = 'wave10_3080_lowrank_conv_memory_76m_to5b_fresh_after2b_resume_20260615'
$activeProofName = 'hidden_drop_p35_square_neuron_3080_300m_fresh_after2b_20260614'
$cache = Join-Path $root 'cache\finewebedu_fresh_after2b_train3000289275_val325152_seq10160_gpt2.pt'
$sourceRun = Join-Path (Join-Path $root 'runs') $sourceRunName
$out = Join-Path (Join-Path $root 'runs') $runName
$script = Join-Path $root 'standalone_longseq_anchor_train.py'
$scriptUrl = 'http://192.168.68.76:8790/standalone_longseq_anchor_train.py'
$targetSteps = 492126
$targetTokens = 5000000160

function Get-ProcessMatches([string]$Pattern) {
  @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.CommandLine -and
      $_.CommandLine -match $Pattern -and
      $_.ProcessId -ne $PID
    })
}

function Stop-StaleInlineResume {
  $stale = Get-ProcessMatches ([regex]::Escape($oldInlineRunName))
  foreach ($proc in $stale) {
    Write-Host "STOP_STALE_INLINE_RESUME pid=$($proc.ProcessId)"
    & taskkill.exe /PID $proc.ProcessId /T /F | Out-String | Write-Host
  }
}

function Wait-ForActiveProof {
  while ($true) {
    $active = Get-ProcessMatches ([regex]::Escape($activeProofName))
    if ($active.Count -eq 0) {
      Write-Host "WAIT_CURRENT_3080_JOB_DONE $(Get-Date -Format o)"
      return
    }
    $pids = ($active | ForEach-Object { $_.ProcessId }) -join ','
    Write-Host "WAIT_CURRENT_3080_JOB active_pids=$pids count=$($active.Count) $(Get-Date -Format o)"
    Start-Sleep -Seconds 60
  }
}

New-Item -ItemType Directory -Force -Path $root, (Join-Path $root 'runs'), $out | Out-Null
Write-Host "LOWRANK76M_TO5B_WAIT_WRAPPER_START $(Get-Date -Format o) host=$env:COMPUTERNAME run=$runName"
Stop-StaleInlineResume
Wait-ForActiveProof
Stop-StaleInlineResume

$gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1).Trim()
Write-Host "GPU_NAME=$gpuName"
if ($gpuName -notmatch 'RTX 3080') {
  throw "refusing to run on non-RTX-3080 GPU: $gpuName"
}
Write-Host "GPU_BEFORE"
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits

if (-not (Test-Path -LiteralPath $cache)) {
  throw "missing fresh-after-2B cache: $cache"
}
if (-not (Test-Path -LiteralPath $sourceRun)) {
  throw "missing source run dir: $sourceRun"
}

$checkpointCandidates = @()
$primaryCheckpoint = Join-Path $sourceRun 'checkpoint.pt'
if (Test-Path -LiteralPath $primaryCheckpoint) {
  $checkpointCandidates += $primaryCheckpoint
}
$checkpointCandidates += @(Get-ChildItem -LiteralPath $sourceRun -Filter 'checkpoint*.pt' -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending |
  ForEach-Object { $_.FullName })
$checkpoint = $checkpointCandidates | Select-Object -First 1
if (-not $checkpoint) {
  Write-Host 'AVAILABLE_CHECKPOINTS:'
  Get-ChildItem -LiteralPath $sourceRun -Filter 'checkpoint*.pt' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime |
    Select-Object FullName,Length,LastWriteTime |
    Format-Table -AutoSize |
    Out-String |
    Write-Host
  throw 'no checkpoint found for resume'
}

Write-Host "USING_RESUME_CHECKPOINT=$checkpoint"
Write-Host "FRESH_AFTER2B_CACHE=$cache"
Write-Host "TARGET_STEPS=$targetSteps TARGET_TOKENS=$targetTokens"

try {
  & curl.exe -L --fail --retry 5 --connect-timeout 10 -o $script $scriptUrl
  if ($LASTEXITCODE -ne 0) {
    throw "curl exit=$LASTEXITCODE"
  }
  Write-Host "TRAINER_REFRESHED_FROM=$scriptUrl"
} catch {
  if (-not (Test-Path -LiteralPath $script)) {
    throw "trainer download failed and no local trainer exists: $_"
  }
  Write-Host "TRAINER_REFRESH_FAILED_USING_EXISTING=$script error=$_"
}

python -B -c "p=r'$script'; compile(open(p, encoding='utf-8').read(), p, 'exec'); print('TRAINER_SYNTAX_OK')"
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

$inspect = "import json, torch; p=r'$checkpoint'; ck=torch.load(p, map_location='cpu', weights_only=False); cfg=ck.get('config', {}) if isinstance(ck, dict) else {}; print('RESUME_META '+json.dumps({'checkpoint': p, 'step': int(ck.get('step', -1)), 'tokens_seen': int(ck.get('tokens_seen', -1)), 'config_run_name': cfg.get('run_name') if isinstance(cfg, dict) else None, 'config_block_type': cfg.get('block_type') if isinstance(cfg, dict) else None}, sort_keys=True))"
python -c $inspect
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

python -u $script `
  --cache-path $cache `
  --output-dir $out `
  --run-name $runName `
  --train-steps $targetSteps `
  --eval-interval 9843 `
  --checkpoint-interval 9843 `
  --milestone-checkpoint-interval 98430 `
  --val-blocks 32 `
  --embedding-dim 896 `
  --block-type multi_scale_lowrank_conv_memory `
  --conv-layers 2 `
  --conv-kernel-size 7 `
  --conv-rank 320 `
  --memory-rank 64 `
  --landmark-stride 128 `
  --sampled-vocab-size 32768 `
  --token-stride 4 `
  --warmup-steps 2000 `
  --learning-rate 0.0006 `
  --min-learning-rate 0.00001 `
  --resume-checkpoint $checkpoint
$code = $LASTEXITCODE

Write-Host "LOWRANK76M_TO5B_RESUME_EXIT=$code $(Get-Date -Format o)"
Write-Host "GPU_AFTER"
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
if ($code -ne 0) {
  exit $code
}
Write-Host "LOWRANK76M_TO5B_RESULT_PATH $out\result.json"
if (Test-Path -LiteralPath (Join-Path $out 'result.json')) {
  Get-Content -LiteralPath (Join-Path $out 'result.json') -Raw
}
Write-Host "LOWRANK76M_TO5B_RESUME_DONE $(Get-Date -Format o)"
