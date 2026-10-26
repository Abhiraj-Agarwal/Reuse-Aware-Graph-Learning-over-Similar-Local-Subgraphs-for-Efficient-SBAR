<#
Installs everything the code needs and checks that the GPU works. No admin needed. Run once after cloning, and
again after a `git pull` that changed requirements.txt (it skips what is already there).

    powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

1. checks the NVIDIA driver (nvidia-smi);
2. installs Python 3.12 for your account if it is missing (winget), and creates .venv in the repo;
3. installs PyTorch with CUDA from PyTorch's own package index (the regular `pip install torch` on Windows gets a
   CPU-only build); the CUDA build is chosen from the "CUDA Version" the driver reports, and it brings its own CUDA
   runtime and cuDNN, so no CUDA toolkit is needed;
4. installs requirements.txt (numpy, PyYAML, tqdm, tensorboardX, networkx, tensorboard, thop, matplotlib);
5. runs a small computation on the GPU.
#>
param(
    [string]$TorchVersion = '2.10.0',  # the version this code was tested with
    [string]$Cuda = 'auto'             # or cu126 / cu128 / cu130
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

# 1. the GPU driver
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw 'nvidia-smi not found: the NVIDIA driver is not installed. Ask the admin (WORKSTATION.md, admin item 1).'
}
$smi = (nvidia-smi) -join "`n"
if ($Cuda -eq 'auto') {
    if ($smi -notmatch 'CUDA Version:\s*(\d+)\.(\d+)') { throw 'Could not read the CUDA Version from nvidia-smi.' }
    $driverCuda = [version]"$($Matches[1]).$($Matches[2])"
    if ($driverCuda -ge [version]'12.8') { $Cuda = 'cu128' }
    elseif ($driverCuda -ge [version]'12.0') { $Cuda = 'cu126' }  # runs on any CUDA 12 driver
    else { throw "The driver only supports CUDA $driverCuda; PyTorch needs 12.x. Ask the admin to update the driver." }
    Write-Output "Driver supports CUDA $driverCuda -> PyTorch build $Cuda"
}

# 2. Python 3.12 and the virtual environment
function Find-Python312 {
    # returns the command for a working Python 3.12, or $null. The "python" in WindowsApps that only opens the
    # Microsoft Store is skipped because it does not answer the version query.
    $ErrorActionPreference = 'Continue'
    $candidates = @(@('py', '-3.12'), @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"), @('python'))
    foreach ($c in $candidates) {
        if (-not (Get-Command $c[0] -ErrorAction SilentlyContinue)) { continue }
        $v = & $c[0] @($c | Select-Object -Skip 1) -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($v -eq '3.12') { return , $c }
    }
    return $null
}
$venvPython = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    $py = Find-Python312
    if (-not $py) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            throw 'Python 3.12 is not installed and winget is unavailable: install it from python.org (untick "for all users").'
        }
        Write-Output 'Installing Python 3.12 for this account (winget) ...'
        winget install --id Python.Python.3.12 -e --scope user --silent --accept-package-agreements --accept-source-agreements
        $py = Find-Python312
        if (-not $py) { throw 'Python 3.12 still not found after installing; open a new terminal and run this again.' }
    }
    Write-Output "Creating .venv with $($py -join ' ')"
    & $py[0] @($py | Select-Object -Skip 1) -m venv .venv
    if (-not (Test-Path $venvPython)) { throw 'Creating .venv failed.' }
}
& $venvPython -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print('Python', sys.version.split()[0])"
if ($LASTEXITCODE) { throw '.venv is not Python 3.12: delete the .venv folder and install Python 3.12.' }

# 3. packages
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install "torch==$TorchVersion" --index-url "https://download.pytorch.org/whl/$Cuda"
if ($LASTEXITCODE) { throw "Installing torch $TorchVersion ($Cuda) failed." }
& $venvPython -c "import torch"
if ($LASTEXITCODE) {
    throw 'import torch failed. If the error mentions WinError 126 / c10.dll, the Microsoft Visual C++ Redistributable is missing (WORKSTATION.md, admin item 2).'
}
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE) { throw 'Installing requirements.txt failed.' }

# 4. check that PyTorch sees and can use the GPU
& $venvPython -c @'
import torch
assert torch.cuda.is_available(), 'PyTorch cannot use the GPU (driver too old for this build? try -Cuda cu126)'
x = torch.randn(1024, 1024, device='cuda')
torch.cuda.synchronize(); (x @ x).sum().item()
p = torch.cuda.get_device_properties(0)
print('OK: torch {}, CUDA {}, cuDNN {}, GPU {} with {:.1f} GB'.format(
    torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), p.name, p.total_memory / 1e9))
'@
if ($LASTEXITCODE) { throw 'GPU check failed (see the message above).' }
Write-Output 'Environment ready: .venv'
