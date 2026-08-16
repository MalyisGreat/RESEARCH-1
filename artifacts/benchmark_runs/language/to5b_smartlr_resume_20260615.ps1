$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUNBUFFERED = '1'
$env:CUDA_VISIBLE_DEVICES = '0'

$root = 'D:\CodexLLM\research1_longseq'
$sourceRunName = 'wave10_3080_lowrank_conv_memory_76m_3b_scratch_existingcache_20260605'
$runName = 'wave10_3080_lowrank_conv_memory_76m_to5b_fresh_after2b_smartlr_20260615'
$cache = Join-Path $root 'cache\finewebedu_fresh_after2b_train3000289275_val325152_seq10160_gpt2.pt'
$sourceRun = Join-Path (Join-Path $root 'runs') $sourceRunName
$out = Join-Path (Join-Path $root 'runs') $runName
$script = Join-Path $root 'standalone_longseq_anchor_train.py'
$scriptUrl = 'http://192.168.68.76:8790/standalone_longseq_anchor_train.py'

$targetSteps = 492126
$targetTokens = 5000000160
$maxLr = 0.0003
$minLr = 0.00001
$warmupSteps = 2000
$evalInterval = 5000
$checkpointInterval = 1000
$milestoneCheckpointInterval = 98430

function Get-ProcessMatches([string]$Pattern) {
  @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.CommandLine -and
      $_.CommandLine -match $Pattern -and
      $_.ProcessId -ne $PID
    })
}

function Stop-StaleSameRun {
  $stale = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.CommandLine -and
      $_.Name -match '^python(\.exe)?$' -and
      $_.CommandLine -match [regex]::Escape($runName) -and
      $_.CommandLine -match 'standalone_longseq_anchor_train\.py' -and
      $_.ProcessId -ne $PID
    })
  foreach ($proc in $stale) {
    Write-Host "STOP_STALE_SAME_RUN_TRAINER pid=$($proc.ProcessId)"
    & taskkill.exe /PID $proc.ProcessId /T /F | Out-String | Write-Host
  }
}

New-Item -ItemType Directory -Force -Path $root, (Join-Path $root 'runs'), $out | Out-Null
Write-Host "LOWRANK76M_TO5B_SMARTLR_START $(Get-Date -Format o) host=$env:COMPUTERNAME run=$runName"
Stop-StaleSameRun

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
$existingOutCheckpoint = Join-Path $out 'checkpoint.pt'
if (Test-Path -LiteralPath $existingOutCheckpoint) {
  $checkpointCandidates += $existingOutCheckpoint
}
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
Write-Host "LR_POLICY max_lr=$maxLr min_lr=$minLr warmup_steps=$warmupSteps eval_interval=$evalInterval checkpoint_interval=$checkpointInterval milestone_checkpoint_interval=$milestoneCheckpointInterval"

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

$inspect = "import json, math, torch; p=r'$checkpoint'; target=$targetSteps; warm=$warmupSteps; max_lr=$maxLr; min_lr=$minLr; ck=torch.load(p, map_location='cpu', weights_only=False); cfg=ck.get('config', {}) if isinstance(ck, dict) else {}; step=int(ck.get('step', -1)); progress=min(max((step-warm)/max(target-warm,1),0.0),1.0); decay=0.5*(1.0+math.cos(math.pi*progress)); lr=min_lr+(max_lr-min_lr)*decay; print('RESUME_META '+json.dumps({'checkpoint': p, 'step': step, 'tokens_seen': int(ck.get('tokens_seen', -1)), 'config_run_name': cfg.get('run_name') if isinstance(cfg, dict) else None, 'config_block_type': cfg.get('block_type') if isinstance(cfg, dict) else None, 'scheduled_resume_lr': lr}, sort_keys=True))"
python -c $inspect
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

python -u $script `
  --cache-path $cache `
  --output-dir $out `
  --run-name $runName `
  --train-steps $targetSteps `
  --eval-interval $evalInterval `
  --checkpoint-interval $checkpointInterval `
  --milestone-checkpoint-interval $milestoneCheckpointInterval `
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
  --warmup-steps $warmupSteps `
  --learning-rate $maxLr `
  --min-learning-rate $minLr `
  --resume-checkpoint $checkpoint
$code = $LASTEXITCODE

Write-Host "LOWRANK76M_TO5B_SMARTLR_EXIT=$code $(Get-Date -Format o)"
Write-Host "GPU_AFTER"
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
if ($code -ne 0) {
  exit $code
}
Write-Host "LOWRANK76M_TO5B_SMARTLR_RESULT_PATH $out\result.json"
if (Test-Path -LiteralPath (Join-Path $out 'result.json')) {
  Get-Content -LiteralPath (Join-Path $out 'result.json') -Raw
}
Write-Host "LOWRANK76M_TO5B_SMARTLR_DONE $(Get-Date -Format o)"
