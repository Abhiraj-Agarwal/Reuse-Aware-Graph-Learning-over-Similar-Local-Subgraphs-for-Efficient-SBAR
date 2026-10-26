<#
Turns the downloaded NTU RGB+D 60 skeleton zip into the files the configs read:

    data\ntu\xsub\  train_data_joint.npy  train_label.pkl  val_data_joint.npy  val_label.pkl
    data\ntu\xview\ (the same)

    powershell -ExecutionPolicy Bypass -File scripts\prepare_ntu.ps1 -Zip D:\datasets\ntu\nturgbd_skeletons_s001_to_s017.zip

Next to the zip it extracts nturgb+d_skeletons\ (56880 .skeleton files, ~15 GB) and downloads the official list of
302 broken clips, then converts (~15 min per benchmark, ~10 GB each in data\ntu). A finished benchmark is marked
with data\ntu\<benchmark>\COMPLETE and skipped next time; an interrupted one is redone. -Benchmark xsub does only
cross-subject; -DeleteExtracted removes the extracted folder at the end (the zip is kept).
#>
param(
    [Parameter(Mandatory = $true)][string]$Zip,
    [string[]]$Benchmark = @('xsub', 'xview'),
    [switch]$DeleteExtracted,
    [string]$Python = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe')
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$Zip = (Resolve-Path $Zip).Path
$dir = Split-Path $Zip -Parent
$raw = Join-Path $dir 'nturgb+d_skeletons'
$missing = Join-Path $dir 'samples_with_missing_skeletons.txt'
$out = Join-Path $repo 'data\ntu'
$expected = @{ xsub = @(40091, 16487); xview = @(37646, 18932) }

# 1. extract (tar is built into Windows 10/11 and is much faster than Expand-Archive)
$count = if (Test-Path $raw) { @(Get-ChildItem $raw -Filter *.skeleton).Count } else { 0 }
if ($count -ne 56880) {
    Write-Output "Extracting $Zip to $dir (a few minutes) ..."
    tar -xf $Zip -C $dir
    if ($LASTEXITCODE) { throw 'Extracting failed (is the zip complete? It should be about 5.8 GB).' }
    $count = @(Get-ChildItem $raw -Filter *.skeleton).Count
}
if ($count -ne 56880) { throw "Found $count .skeleton files in $raw, expected 56880 (incomplete download?)." }
Write-Output "Raw skeletons: $count files in $raw"

# 2. the official list of clips with missing / broken skeletons
if (-not (Test-Path $missing)) {
    Invoke-WebRequest 'https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt' `
        -OutFile $missing -UseBasicParsing
}
Write-Output "Broken-clip list: $missing"

# 3. convert, one benchmark at a time
foreach ($b in $Benchmark) {
    $done = Join-Path $out "$b\COMPLETE"
    if (Test-Path $done) { Write-Output "$b already prepared, skipping."; continue }
    Write-Output "Converting $b ..."
    Push-Location (Join-Path $repo 'data_gen')
    try {
        & $Python ntu_gendata.py --data_path $raw --ignored_sample_path $missing --out_folder $out --benchmark $b
        if ($LASTEXITCODE) { throw "Converting $b failed." }
    } finally { Pop-Location }
    # 4. check the clip counts against the official split sizes
    $check = & $Python -c "import numpy as np, sys; d = sys.argv[1]; print(*(np.load(d + '/' + p + '_data_joint.npy', mmap_mode='r').shape[0] for p in ('train', 'val')))" (Join-Path $out $b)
    $n = $check -split ' ' | ForEach-Object { [int]$_ }
    if ($n[0] -ne $expected[$b][0] -or $n[1] -ne $expected[$b][1]) {
        throw "$b has $($n[0]) train / $($n[1]) val clips, expected $($expected[$b][0]) / $($expected[$b][1])."
    }
    Set-Content -Path $done -Value (Get-Date -Format s)
    Write-Output "$b ready: $($n[0]) train / $($n[1]) val clips in $out\$b"
}

if ($DeleteExtracted) {
    Remove-Item -Recurse -Force $raw
    Write-Output "Deleted $raw (the zip is kept)."
}
