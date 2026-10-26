<#
Registers a scheduled task that runs scripts\run_training.ps1 for one config as YOUR Windows account, whether
or not you are signed in, so training keeps going after you sign out and while other people use the machine.
The task fires again every 10 minutes; run_training.ps1 exits at once if the run is already going or finished,
so this only matters after a reboot or crash, when it resumes from the last finished epoch and, after the
last epoch, writes the results (results\<config name>\report.md).
eport.md).

    powershell -ExecutionPolicy Bypass -File scripts\register_training_task.ps1 -Config config\ntu\ntu60_xsub_joint.yaml

Asks for your Windows password once (Task Scheduler stores it to sign the task in; a PIN does not work).
#>
param(
    [Parameter(Mandatory = $true)][string]$Config,
    [string]$TaskName = ('mstgcn-' + [IO.Path]::GetFileNameWithoutExtension($Config)),
    [int]$RetryMinutes = 10
)

$repo = Split-Path $PSScriptRoot -Parent
$runner = Join-Path $PSScriptRoot 'run_training.ps1'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runner`" -Config `"$Config`"" `
    -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $RetryMinutes)
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun -Priority 7
$cred = Get-Credential -UserName "$env:USERDOMAIN\$env:USERNAME" `
    -Message 'Your Windows password (Task Scheduler needs it to run training while you are signed out)'
if (-not $cred) { exit 1 }

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -User $cred.UserName -Password $cred.GetNetworkCredential().Password -RunLevel Limited -Force -ErrorAction Stop |
    Out-Null

Write-Output "Registered '$TaskName': first start in ~1 minute, then a check every $RetryMinutes minutes."
Write-Output "Status : Get-ScheduledTaskInfo -TaskName '$TaskName'"
Write-Output "Stop   : powershell -ExecutionPolicy Bypass -File scripts\stop_training.ps1 -Config $Config"
