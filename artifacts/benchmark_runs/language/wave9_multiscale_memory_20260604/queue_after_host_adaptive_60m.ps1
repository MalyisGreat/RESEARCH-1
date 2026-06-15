param(
  [int]$WaitPid = 38792
)

$ErrorActionPreference = 'Stop'
$Root = 'E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language'
$RunRoot = Join-Path $Root 'wave9_multiscale_memory_20260604'
$Script = Join-Path $RunRoot 'run_host_2080_multiscale_memory_stride4_60m.ps1'
$Stdout = Join-Path $RunRoot 'host_2080_multiscale_memory_stride4_60m.stdout.log'
$Stderr = Join-Path $RunRoot 'host_2080_multiscale_memory_stride4_60m.stderr.log'

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
Write-Host "WAVE9_MS_MEMORY_QUEUE_WAIT wait_pid=$WaitPid script=$Script"
$proc = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
if ($proc) {
  Wait-Process -Id $WaitPid
}
Write-Host "WAVE9_MS_MEMORY_QUEUE_LAUNCH $(Get-Date -Format o)"
$child = Start-Process powershell -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -ArgumentList @(
  '-NoProfile',
  '-ExecutionPolicy',
  'Bypass',
  '-File',
  $Script
)
Write-Host "WAVE9_MS_MEMORY_QUEUE_STARTED pid=$($child.Id) stdout=$Stdout stderr=$Stderr"
