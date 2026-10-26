# Training on the Windows GPU workstation

Machine: i9-12900K (16 cores / 24 threads), 32 GB RAM, **NVIDIA RTX 4500 Ada, 24 GB VRAM**, 256 GB SSD + 2 TB HDD,
Windows, your own (non-admin) account. Rule: use at most **50% of the memory** (12 GB VRAM, 16 GB RAM); GPU and CPU
time are not limited. All commands are PowerShell. VS Code is not needed.

| script | what it does | how often |
|---|---|---|
| `scripts\setup_env.ps1` | installs Python 3.12 (if missing), PyTorch with CUDA and all packages into `.venv`; checks the GPU | once |
| `scripts\prepare_ntu.ps1` | downloaded NTU zip -> the training data in `data\ntu\` | once |
| `scripts\run_training.ps1` | **the complete run**: all epochs, resuming after any interruption, then the results export | per run |
| `scripts\show_usage.ps1` | what the run uses: GPU, VRAM, CPU, RAM | any time |
| `scripts\register_training_task.ps1`, `scripts\stop_training.ps1` | optional: run in the background through Task Scheduler | if needed |

## 0. Before you start

- [ ] NTU RGB+D access approved by ROSE Lab (https://rose1.ntu.edu.sg/dataset/actionRecognition/). Approval is
      manual and can take days, so request it first. Only the skeleton zip is needed, not the videos.
- [ ] ~55 GB free disk (see "Time and resources" at the end).
- [ ] `nvidia-smi` works on the machine (otherwise see "What needs admin rights" at the end).

Where to put things: the code in your profile on C: (the SSD, private to you), the raw download on D: (the HDD).
Check free space with `Get-PSDrive C, D`.

## 1. Get the code

```powershell
git --version        # if "not recognized": winget install --id Git.Git -e --scope user   (then open a new terminal)
cd $HOME
git clone https://github.com/Abhiraj-Agarwal/Reuse-Aware-Graph-Learning-over-Similar-Local-Subgraphs-for-Efficient-SBAR sbar
cd sbar
```

Later updates: `cd $HOME\sbar; git pull`.

## 2. Install everything (~10 min, mostly the 2.5 GB PyTorch download)

```powershell
cd $HOME\sbar
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

This is the only install step. It installs Python 3.12 for your account if it is missing, creates `.venv`, installs
PyTorch 2.10 with CUDA (matched to the driver; it brings its own CUDA runtime and cuDNN, so there's no CUDA toolkit to
install), then `requirements.txt`. It ends with `OK: torch 2.10.0+cu128 ... GPU NVIDIA RTX 4500 Ada Generation with 25.8 GB`.

You can't use a plain `pip install -r requirements.txt` instead: on Windows that installs a CPU-only PyTorch.
`setup_env.ps1` installs the CUDA build first.

## 3. Smoke test (~3 min, random data)

```powershell
.venv\Scripts\python.exe tools\make_dummy_data.py
powershell -ExecutionPolicy Bypass -File scripts\run_training.ps1 -Config config\smoke_test.yaml
```

The whole pipeline in miniature: 2 epochs, then the export. It should show `Device: NVIDIA RTX 4500 Ada Generation`
and end with `Run complete. Results: ...\results\smoke_test\report.md`. To run it again, delete `work_dir\smoke_test`,
`runs\smoke_test` and `results\smoke_test`.

## 4. Dataset

Download `nturgbd_skeletons_s001_to_s017.zip` (~5.8 GB) from the ROSE Lab page in the browser on the workstation
(it lands in `Downloads`), then:

```powershell
New-Item -ItemType Directory -Force D:\datasets\ntu | Out-Null
Move-Item $HOME\Downloads\nturgbd_skeletons_s001_to_s017.zip D:\datasets\ntu\
cd $HOME\sbar
powershell -ExecutionPolicy Bypass -File scripts\prepare_ntu.ps1 -Zip D:\datasets\ntu\nturgbd_skeletons_s001_to_s017.zip -DeleteExtracted
```

`prepare_ntu.ps1` extracts the zip next to itself, checks all 56880 clips are there, downloads the official list of
302 broken clips, converts both benchmarks, checks the clip counts and deletes the extracted folder. It takes
~30-40 min. Run it again if it is interrupted; finished benchmarks are skipped. `-Benchmark xsub` does only the
cross-subject split if you want to start training sooner.

The raw files can live anywhere (you pass `-Zip`). The training data must end up in the repo's `data\ntu\`, because
the configs read it from there:

```
D:\datasets\ntu\                              raw download (anywhere you like)
    nturgbd_skeletons_s001_to_s017.zip
    samples_with_missing_skeletons.txt        downloaded by prepare_ntu.ps1
    nturgb+d_skeletons\  (56880 *.skeleton)   extracted; deleted with -DeleteExtracted

C:\Users\<you>\sbar\                          the repo
    data\ntu\xsub\   train_data_joint.npy (7.2 GB, 40091 clips)  train_label.pkl
                     val_data_joint.npy   (3.0 GB, 16487 clips)  val_label.pkl   COMPLETE
    data\ntu\xview\  the same for cross-view (37646 / 18932 clips)
```

To keep the 10 GB per benchmark somewhere else, run `data_gen\ntu_gendata.py` yourself with `--out_folder <dir>`
and change `data_path` / `label_path` in the config to the absolute paths. Never commit or share the data (NTU
licence); `data/` is git-ignored.

## 5. Optional: check speed and memory (1 min)

```powershell
.venv\Scripts\python.exe tools\bench_train.py --config config\ntu\ntu60_xsub_joint.yaml --frames 300 --batch 8 12 --gpu-mem-fraction 0.48
```

It prints seconds per clip, peak VRAM and the projected time for all 110 epochs. The configs use 8 clips per step x 4
accumulation steps = 32, the original batch, in fp32.

## 6. Start the run

```powershell
cd $HOME\sbar
powershell -ExecutionPolicy Bypass -File scripts\run_training.ps1 -Config config\ntu\ntu60_xsub_joint.yaml
```

That's all. It checks the data, trains all 110 epochs, then writes `results\ntu60_xsub_joint\`. The window shows
the training log as it grows: after 2 minutes, then every 10 minutes, a line like

```
Progress: epoch 37/110, batch 2500/5011 | run 33.2% done | 0.31 s/batch | left ~7h 40m (finishes ~Tue 27 Oct 06:10) | running 3h 50m this session
```

**Leaving it running: press Win+L (lock).** Locking keeps your session, the window and the run going, and when you
unlock, the window is still there with everything printed since. Other people can use "Switch user" to sign in to
their own accounts without affecting it. Three things stop it: signing out, a restart (including Windows Update),
or a power cut. Then just run the same command again and it continues from the last finished epoch.
In the old console window (not Windows Terminal), don't click inside it: selecting text pauses the program until
you press Esc.

Optional: so that it also restarts by itself after a reboot while you are away, run
`powershell -ExecutionPolicy Bypass -File scripts\register_training_task.ps1 -Config config\ntu\ntu60_xsub_joint.yaml`
(asks for your Windows password). It checks every 10 minutes and starts the run only if it isn't already running,
so it works alongside the open window. It runs without a window, so check progress with `progress.txt` (section 7).
Remove it after the run with `Unregister-ScheduledTask mstgcn-ntu60_xsub_joint -Confirm:$false`.

## 7. Check on it later

```powershell
cd $HOME\sbar
Get-Content work_dir\ntu60\xsub\mstgcn_joint\progress.txt             # status: epoch, % done, time left, best accuracy
Get-Content work_dir\ntu60\xsub\mstgcn_joint\log.txt -Tail 30         # the log (add -Wait to follow it)
powershell -ExecutionPolicy Bypass -File scripts\show_usage.ps1 -Watch 10   # GPU, VRAM, CPU, RAM of the run
.venv\Scripts\tensorboard.exe --logdir runs                            # curves at http://localhost:6006
```

`progress.txt` is rewritten every 10 minutes. If its "updated" time is old, the run isn't going: run the start command
again. Errors are in `work_dir\...\console_train_<time>.err`. Validation accuracy is only measured after epoch 50
(the original schedule).

GPU commands (`nvidia-smi` comes with the NVIDIA driver; nothing to install):

```powershell
nvidia-smi                       # snapshot: utilisation, memory, temperature, power
nvidia-smi -l 5                  # refresh every 5 s (Ctrl+C stops)
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 60 -f gpu_log.csv
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion   # works even without the NVIDIA driver
```

On Windows `nvidia-smi` shows `N/A` for the memory of each process; `show_usage.ps1` shows it. In Task Manager >
Performance > GPU, switch a graph to "Cuda" or "Compute_0" to see training load (the default "3D" graph stays at 0).

## 8. After the run

- Results: `results\ntu60_xsub_joint\report.md` (see below). Copy the folder anywhere; it is self-contained.
- Memory is freed automatically. When the run ends (or is stopped), its processes exit, and Windows releases all
  their RAM and GPU memory at once; the DataLoader worker processes are tied to the main one and exit with it.
  Nothing has to be freed by hand. `nvidia-smi` should show no python process afterwards; if one is still listed,
  it's still running (or hung): stop it with `scripts\stop_training.ps1 -Config ...` or close its window.
- Disk is not freed automatically. The run leaves ~1.6 GB (checkpoints, per-epoch weights, TensorBoard logs,
  results). The per-epoch weights `runs\...\mstgcn_joint-<epoch>-<step>.pt` (0.7 GB) can be deleted once you have the
  results; keep `data\ntu\` for further runs.
- To train again from scratch with the same config, delete its `work_dir\...`, `runs\...` and `results\...` folders.

Stop a run early: close the window or press Ctrl+C. Run the start command again later to continue. Export a
different checkpoint: `.venv\Scripts\python.exe tools\export_run.py --config config\ntu\ntu60_xsub_joint.yaml --weights best --out results\xsub_best`.

## Checkpointing

After every epoch, `work_dir\...\checkpoint.pt` is rewritten with the model, the optimizer state, the epoch, the
best accuracy and the history. It goes to a temporary file first, is forced to disk and only then swapped in, and the
previous checkpoint is kept as `checkpoint.pt.prev`. So a power cut at any moment leaves a readable checkpoint; a
damaged one is detected and `.prev` is used. An interruption loses at most the epoch in progress (~5-8 min). The
model has no dropout and each epoch's data order is fixed by the seed, so a resumed run computes the same thing as
an uninterrupted one. The report counts the interruptions.

## The results: `results\<config name>\`

Start with `report.md`. It explains everything with the numbers of that run:

| file | contents |
|---|---|
| `report.md` | 1 result (val / train top-1, top-5, best epoch) · 2 what was trained (data, input, blocks, channels, parameters, GMACs, epochs, optimiser, lr schedule, batch, precision, seed, limits) · 3 block by block: output shape, cost, receptive field, and how well the embedding after that block separates the actions · 4 per-epoch history · 5 per-action accuracy, most confused pairs · 6 every file and array explained · 7 how the run differs from the original MST-GCN training · 8 machine and software versions |
| `embeddings_train.npz`, `embeddings_val.npz` | per clip: `sample_name`, `label`, `pred`, `logits` (60), `embedding` (448, what the classifier sees), `block_01`...`block_10` (embedding after each block) |
| `model_weights.pt` | the trained weights (final epoch) |
| `summary.json` | all numbers of the report, machine-readable |
| `training_history.csv`, `per_class_accuracy.csv`, `confusion_matrix.csv` | tables for spreadsheets / plots |
| `config.yaml`, `run_arguments.yaml` | the config as given, and every argument with defaults filled in |

The embeddings come from the trained model computed like the original code: fp32, no augmentation and no reuse
or caching. They are the raw baseline for the reuse-aware experiments.

## Memory limits

From `resource_limits:` in the config (`resource_limits.py`); the log shows them at the start:

| | setting | effect |
|---|---|---|
| VRAM | `gpu_mem_fraction: 0.48` | PyTorch may use at most 11.5 GiB; with the ~0.5 GB CUDA context that is ~50% of the 24 GB. Batch 8 in fp32 needs about 7-8 GB. |
| RAM | `ram_gb: 16` | hard cap of 16 GB (50%) for training + its worker processes; it needs ~5-8 GB (each epoch logs "peak committed RAM"). |
| GPU / CPU time | `gpu_duty: null`, `cpu_percent: null` | not limited: the GPU runs flat out. The run gets below-normal CPU priority, so other people's programs come first. |

## Time and resources (NTU-60 xsub, 300 frames, 110 epochs, RTX 4500 Ada)

These are estimates; step 5 and the progress lines give the real numbers. Each training clip costs ~104 GFLOP
(17.3 GMACs forward, ~3x that with the backward pass), so one epoch of 40,091 clips is ~4.2 PFLOP. The RTX 4500 Ada
peaks at ~40 TFLOPS fp32; a model of small graph and temporal convolutions typically reaches 25-45% of that.

| stage | time |
|---|---|
| setup (step 2) | ~10 min |
| dataset preparation (step 4, CPU) | ~30-40 min, plus the download |
| training, per epoch | ~4-8 min |
| training, 110 epochs | **~8-14 h** |
| validation after each of epochs 51-110 | ~0.5-1 min each, ~0.5-1 h in total |
| export | ~5-10 min |
| **complete run (step 6)** | **~9-15 h** (most likely ~10-12 h) |

| resource | use |
|---|---|
| GPU | ~7-8 GB VRAM, utilisation near 100% (no time limit) |
| CPU | ~1-2 of 24 threads (main loop + 2 light DataLoader workers) |
| RAM | ~5-8 GB, cap 16 GB; Windows also caches the 7 GB data file and gives that back when others need memory |
| disk, once per benchmark | 5.8 GB zip + ~15 GB extracted during step 4 (deleted with `-DeleteExtracted`) + 10.2 GB data |
| disk, once | ~5-6 GB `.venv` |
| disk, per run | ~1.6 GB: checkpoints 50 MB, per-epoch weights 0.7 GB, TensorBoard ~0.1 GB, logs < 0.1 GB, results ~0.7 GB |

## What needs admin rights

Normally nothing: Git, Python, `.venv`, PyTorch with its CUDA runtime, all packages, the dataset, the limits and the
optional scheduled task all work from a standard account. Each item below starts with a check you can run yourself.
The admin runs the fix in PowerShell opened with **"Run as administrator"**. `<you>` is your account name (`whoami`).

**1. NVIDIA driver**: needed if `nvidia-smi` is not found, or `setup_env.ps1` says the driver is too old.

```powershell
Get-Command nvidia-smi; nvidia-smi --query-gpu=name,driver_version --format=csv            # check (you)
# fix (admin): download the RTX 4500 Ada driver from https://www.nvidia.com/Download/index.aspx
# (NVIDIA RTX / Quadro > NVIDIA RTX Series > RTX 4500 Ada Generation > Windows 11, Production Branch), then:
& "$HOME\Downloads\<downloaded driver>.exe" -s -noreboot
Restart-Computer
```

**2. Microsoft Visual C++ Redistributable 2015-2022 (x64)**: needed if `setup_env.ps1` says `import torch` failed
with `WinError 126` / `c10.dll`. It is usually already installed.

```powershell
Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64' | Select-Object Version, Installed   # check (you)
winget install --id Microsoft.VCRedist.2015+.x64 -e                                                                 # fix (admin)
```

**3. "Log on as a batch job"**: only for the optional scheduled task, if it fails with 0x80070569.

```powershell
Get-ScheduledTaskInfo mstgcn-ntu60_xsub_joint | Select-Object LastRunTime, LastTaskResult   # check (you)
# fix (admin): secpol.msc > Local Policies > User Rights Assignment > "Log on as a batch job" > add <you>; or:
$sid = (New-Object System.Security.Principal.NTAccount('<you>')).Translate([System.Security.Principal.SecurityIdentifier]).Value
secedit /export /cfg $env:TEMP\rights.inf /areas USER_RIGHTS
$inf = Get-Content $env:TEMP\rights.inf
if ($inf -match '^SeBatchLogonRight') { $inf = $inf -replace '^(SeBatchLogonRight = .*)$', ('$1,*' + $sid) }
else { $inf = $inf -replace '^\[Privilege Rights\]$', ("[Privilege Rights]`r`nSeBatchLogonRight = *" + $sid) }
$inf | Set-Content $env:TEMP\rights.inf -Encoding Unicode
secedit /configure /db $env:TEMP\rights.sdb /cfg $env:TEMP\rights.inf /areas USER_RIGHTS
```

**4. Git / Python for all users**: only if per-user installs or `winget` are blocked.

```powershell
winget install --id Git.Git -e --scope machine          # fix (admin)
winget install --id Python.Python.3.12 -e --scope machine
```

**5. Remote Desktop (optional, Windows Pro / Enterprise)**: to check on the run from your laptop.

```powershell
Set-ItemProperty 'HKLM:\System\CurrentControlSet\Control\Terminal Server' -Name fDenyTSConnections -Value 0   # fix (admin)
Enable-NetFirewallRule -DisplayGroup 'Remote Desktop'
Add-LocalGroupMember -Group 'Remote Desktop Users' -Member '<you>'
hostname                                   # you, on the workstation: the name to connect to
mstsc /v:<workstation name>                # you, on your laptop
```

Closing the Remote Desktop window only disconnects you, so the run keeps going (like Win+L). Don't use Start > Sign out.

**6. Fewer surprise restarts (optional)**: an update restart only costs the current epoch, but you can avoid it.

```powershell
Set-ItemProperty 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings' -Name ActiveHoursStart -Value 8   # fix (admin): no
Set-ItemProperty 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings' -Name ActiveHoursEnd -Value 2     # restarts 08:00-02:00
powercfg /change standby-timeout-ac 0      # never sleep on mains power (training already keeps the PC awake while it runs)
powercfg /change hibernate-timeout-ac 0
```
