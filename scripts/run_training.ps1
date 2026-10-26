<#
One complete run for one config, end to end:
  1. train all epochs of the config (main.py), resuming from <work_dir>\checkpoint.pt after any interruption
     (reboot, power cut, crash, stop_training.ps1);
  2. export the results (tools\export_run.py): results\<config name>\report.md + embeddings + weights + CSVs.
Safe to start over and over - the scheduled task from register_training_task.ps1 does exactly that: a finished
step is skipped, and it exits at once if the run is already going.

    powershell -ExecutionPolicy Bypass -File scripts\run_training.ps1 -Config config\ntu\ntu60_xsub_joint.yaml

While it runs, this window shows the training log as it grows (a progress line with % done and time left every
10 minutes). The same is in <work_dir>\log.txt; <work_dir>\progress.txt always holds the current status; and
<work_dir>\console_<step>_<time>.out / .err hold everything the Python scripts print (.err: progress bars, errors).
#>
param(
    [Parameter(Mandatory = $true)][string]$Config,
    [string]$Python = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe')
)

$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$workDir = (Select-String -Path $Config -Pattern '^work_dir:\s*(\S+)').Matches[0].Groups[1].Value
$results = Join-Path 'results' ([IO.Path]::GetFileNameWithoutExtension($Config))
New-Item -ItemType Directory -Force $workDir | Out-Null
$pidFile = Join-Path $workDir 'train.pid'
$trained = Test-Path (Join-Path $workDir 'DONE')
$exported = Test-Path (Join-Path $results 'report.md')

if ($trained -and $exported) {
    Write-Output "Already finished and exported ($results\report.md). Delete $workDir and $results to run again."
    exit 0
}
if (Test-Path $pidFile) {
    $old = Get-Process -Id ([int](Get-Content $pidFile)) -ErrorAction SilentlyContinue
    if ($old -and $old.ProcessName -like 'python*') {
        Write-Output "Already running (pid $($old.Id))."
        exit 0
    }
}
foreach ($m in (Select-String -Path $Config -Pattern '^\s*(data|label)_path:\s*(\S+)').Matches) {
    if (-not (Test-Path $m.Groups[2].Value)) {
        Write-Output "Missing $($m.Groups[2].Value): prepare the dataset first (WORKSTATION.md step 4)."
        exit 2
    }
}

$env:PYTHONUNBUFFERED = '1'
$env:TQDM_MININTERVAL = '60'  # progress bar once a minute instead of ten times a second

function Show-NewLines([string]$File, [int]$Shown) {
    # print the lines added to $File since the last call; returns the new line count
    if (-not (Test-Path $File)) { return $Shown }
    $lines = @(Get-Content $File)
    if ($lines.Count -gt $Shown) { $lines[$Shown..($lines.Count - 1)] | ForEach-Object { Write-Host $_ } }
    return $lines.Count
}

function Invoke-Step([string]$Name, [string[]]$Arguments, [string]$Follow) {
    # runs one Python step with its output in files, and meanwhile shows the new lines of $Follow in this window
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $out = Join-Path $workDir "console_${Name}_$stamp.out"
    $err = Join-Path $workDir "console_${Name}_$stamp.err"
    if (-not $Follow) { $Follow = $out }
    $shown = if (Test-Path $Follow) { @(Get-Content $Follow).Count } else { 0 }
    $p = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $repo -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $p.Handle  # without this, Windows PowerShell loses the exit code
    Set-Content -Path $pidFile -Value $p.Id
    Write-Host "$(Get-Date -Format s) $Name started (pid $($p.Id)); logs in $workDir"
    while (-not $p.WaitForExit(15000)) { $shown = Show-NewLines $Follow $shown }
    $shown = Show-NewLines $Follow $shown
    Write-Host "$(Get-Date -Format s) $Name exited with code $($p.ExitCode)"
    if ($p.ExitCode -ne 0) {
        Write-Host "---- last lines of $err :"
        Get-Content $err -Tail 25 | ForEach-Object { Write-Host $_ }
    }
    return $p.ExitCode
}

if (-not $trained) {
    $code = Invoke-Step 'train' @('main.py', '--config', "`"$Config`"", '--resume', 'True') (Join-Path $workDir 'log.txt')
    if ($code -ne 0 -or -not (Test-Path (Join-Path $workDir 'DONE'))) { exit 1 }
}
$code = Invoke-Step 'export' @('tools\export_run.py', '--config', "`"$Config`"")
if ($code -eq 0) { Write-Output "Run complete. Results: $repo\$results\report.md" }
exit $code
