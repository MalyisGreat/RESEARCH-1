$ErrorActionPreference = 'Stop'
$WaitPid = 34004
$RunRoot = 'E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\wave7_dilated_multiscale_20260604'
$Script = Join-Path $RunRoot 'run_host_2080_dilated_multiscale_stride4_60m.ps1'
$Stdout = Join-Path $RunRoot 'host_2080_dilated_multiscale_stride4_60m.stdout.log'
$Stderr = Join-Path $RunRoot 'host_2080_dilated_multiscale_stride4_60m.stderr.log'

Write-Host "WAVE7_QUEUE_WAIT wait_pid=$WaitPid script=$Script"
if (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
  Wait-Process -Id $WaitPid
}
Write-Host "WAVE7_QUEUE_LAUNCH $(Get-Date -Format o)"
Remove-Item $Stdout, $Stderr -ErrorAction SilentlyContinue
$proc = Start-Process powershell -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Script)
Write-Host "WAVE7_QUEUE_STARTED pid=$($proc.Id) stdout=$Stdout stderr=$Stderr"
