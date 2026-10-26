<#
Stops a training run started by run_training.ps1 / register_training_task.ps1 and disables its scheduled task
so it does not start again. Nothing is lost beyond the current epoch: run_training.ps1 resumes from the last
finished one.

    powershell -ExecutionPolicy Bypass -File scripts\stop_training.ps1 -Config config\ntu\ntu60_xsub_joint.yaml

Continue later:  Enable-ScheduledTask -TaskName mstgcn-ntu60_xsub_joint
Remove the task: Unregister-ScheduledTask -TaskName mstgcn-ntu60_xsub_joint -Confirm:$false
#>
param(
    [Parameter(Mandatory = $true)][string]$Config,
    [string]$TaskName = ('mstgcn-' + [IO.Path]::GetFileNameWithoutExtension($Config))
)

$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Disable-ScheduledTask -TaskName $TaskName | Out-Null
    Write-Output "Disabled task '$TaskName'."
}
$workDir = (Select-String -Path $Config -Pattern '^work_dir:\s*(\S+)').Matches[0].Groups[1].Value
$pidFile = Join-Path $workDir 'train.pid'
if (Test-Path $pidFile) {
    $p = Get-Process -Id ([int](Get-Content $pidFile)) -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -like 'python*') {
        # /T: the whole tree (a venv python.exe is a launcher that starts the real interpreter + DataLoader workers)
        taskkill /PID $p.Id /T /F | Out-Null
        Write-Output "Stopped main.py (pid $($p.Id))."
    } else {
        Write-Output 'Training was not running.'
    }
}
