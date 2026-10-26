<#
One screen showing what training uses: the GPU as a whole (nvidia-smi), then per python process (training, its
DataLoader workers, the export) CPU, RAM, GPU memory and GPU compute, and the machine totals. No admin needed.

    powershell -ExecutionPolicy Bypass -File scripts\show_usage.ps1             # once
    powershell -ExecutionPolicy Bypass -File scripts\show_usage.ps1 -Watch 10   # every 10 s, Ctrl+C to stop

On Windows nvidia-smi prints N/A for per-process GPU memory; the Windows GPU counters used here do not.
#>
param([int]$Watch = 0)

$ncpu = [Environment]::ProcessorCount
$pidOf = { param($instance) [int]($instance -replace '^pid_(\d+)_.*', '$1') }
do {
    if ($Watch) { Clear-Host }
    Write-Output (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv
    } else {
        Write-Output 'nvidia-smi not found: the NVIDIA driver is not installed (WORKSTATION.md, "What needs admin rights").'
    }

    $gpuMem = @{}; $gpuUse = @{}; $cpu = @{}
    foreach ($s in (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples) {
        $gpuMem[(& $pidOf $s.InstanceName)] += $s.CookedValue
    }
    foreach ($s in (Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples) {
        if ($s.InstanceName -match 'engtype_(cuda|compute)') { $gpuUse[(& $pidOf $s.InstanceName)] += $s.CookedValue }
    }
    $samples = (Get-Counter '\Process(python*)\% Processor Time', '\Process(python*)\ID Process' -ErrorAction SilentlyContinue).CounterSamples
    foreach ($g in ($samples | Group-Object InstanceName)) {
        $id = [int]($g.Group | Where-Object { $_.Path -like '*\id process' }).CookedValue
        $cpu[$id] = ($g.Group | Where-Object { $_.Path -like '*\% processor time' }).CookedValue / $ncpu
    }

    $rows = @(Get-Process python* -ErrorAction SilentlyContinue | ForEach-Object {
        [pscustomobject]@{
            PID                = $_.Id
            Process            = $_.ProcessName
            'CPU % of machine' = [math]::Round([double]$cpu[$_.Id], 1)
            'RAM GB'           = [math]::Round($_.PrivateMemorySize64 / 1GB, 2)
            'GPU memory GB'    = [math]::Round([double]$gpuMem[$_.Id] / 1GB, 2)
            'GPU compute %'    = [math]::Round([double]$gpuUse[$_.Id], 1)
        }
    })
    if ($rows.Count) {
        $rows | Format-Table -AutoSize | Out-String -Width 200
        Write-Output ('python total: CPU {0:N1}% of the machine, RAM {1:N1} GB, GPU memory {2:N1} GB' -f
            ($rows | Measure-Object 'CPU % of machine' -Sum).Sum, ($rows | Measure-Object 'RAM GB' -Sum).Sum,
            ($rows | Measure-Object 'GPU memory GB' -Sum).Sum)
    } else {
        Write-Output 'No python process is running.'
    }
    $os = Get-CimInstance Win32_OperatingSystem
    $busy = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples[0].CookedValue
    Write-Output ('machine: CPU {0:N0}% busy, RAM {1:N1} of {2:N1} GB in use (everyone, incl. file cache in use)' -f
        $busy, (($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB), ($os.TotalVisibleMemorySize / 1MB))
    if ($Watch) { Start-Sleep -Seconds $Watch }
} while ($Watch)
