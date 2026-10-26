# Running this baseline

## What this code actually is

Despite the repo name, the model code here is **MST-GCN** (Chen et al., *Multi-Scale Spatial Temporal Graph
Convolutional Network for Skeleton-Based Action Recognition*, AAAI 2021), not the full STTGC-Net from the
ICVGIP'23 paper. The paper's STTA block (spatial + temporal multi-head self-attention, Sec. 3.3) is **not in
this repository**, and the hyper-parameters (112/224/448 channels, 300 frames, lr 0.15, 110 epochs) are MST-GCN's,
not the paper's (64/128/256 channels, 64 frames, lr 0.001, 60 epochs).
So results from this code should be reported as an MST-GCN / ST-GCN-family baseline.

Pipeline: `data_gen/` (raw NTU `.skeleton` -> normalised `.npy`) -> `feeders/feeder.py` (Dataset) ->
`model/ST_GCN.py` (10 x [multi-scale GCN -> multi-scale TCN] blocks, GAP, FC) -> `main.py` (train / eval loop).

## 1. Install

```bash
# install torch for your CUDA version first (see requirements.txt), then
pip install -r requirements.txt
```

## 2. Smoke test (no dataset needed, CPU or GPU, ~2-3 min on CPU)

```bash
python tools/make_dummy_data.py
python main.py --config config/smoke_test.yaml
```

This should print 2 training epochs, write a checkpoint to `runs/smoke_test/`, and print an eval accuracy
(meaningless on random data).

## 3. Prepare NTU RGB+D 60

1. Request/download `nturgbd_skeletons_s001_to_s017.zip` from the ROSE Lab
   (https://rose1.ntu.edu.sg/dataset/actionRecognition/) and unzip it to `data/nturgbd_raw/nturgb+d_skeletons/`.
2. Download the list of samples with missing skeletons from the official NTU repo
   (`NTU_RGBD_samples_with_missing_skeletons.txt` in https://github.com/shahroudy/NTURGB-D/tree/master/Matlab)
   and save it as `data/nturgbd_raw/samples_with_missing_skeletons.txt`.
3. Generate the joint data (both xsub and xview, ~30 min; `--benchmark xsub` for one):

   ```bash
   cd data_gen
   python ntu_gendata.py
   cd ..
   ```

   Output: `data/ntu/{xsub,xview}/{train,val}_data_joint.npy` with shape `(N, 3, 300, 25, 2)` and
   `{train,val}_label.pkl`. It needs ~20 GB of disk; the arrays are written in place on disk, not held in RAM.

Optional bone stream: `python data_gen/gen_bone_data.py` from the repo root (edit the `datasets` set at the top
first; it only lists `ntu/xview`). `gen_motion_data.py` / `merge_*.py` have hard-coded `/data/ChenZhan/...`
paths that must be edited before use.

## 4. Train / evaluate

Single GPU (or CPU) — plain python, no launcher needed:

```bash
python main.py --config config/ntu/ntu60_xsub_joint.yaml
```

Multi-GPU (original setup: 4 GPUs x batch 8):

```bash
torchrun --nproc_per_node=4 main.py --config config/ntu/ntu60_xsub_joint.yaml --batch-size 8
```

Evaluate a checkpoint:

```bash
python main.py --config config/ntu/ntu60_xsub_joint.yaml --phase test --weights runs/ntu60/xsub/mstgcn_joint-109-<step>.pt
```

Notes on the original training loop (unchanged):
- Evaluation only runs after epoch `step[0]` (50); checkpoints are saved every epoch after epoch 50.
- If `model_saved_name` already exists, `main.py` asks `delete it? y/n` on stdin (only in an interactive
  terminal and without `--resume True`).
- Logs: `work_dir/.../log.txt`; TensorBoard: `tensorboard --logdir runs`.

Added for long runs on a shared machine (see [WORKSTATION.md](WORKSTATION.md) for the Windows workstation steps):
- `--resume True` continues from `<work_dir>/checkpoint.pt` (falling back to `checkpoint.pt.prev`), which is
  rewritten after every epoch together with the per-epoch history; a finished run writes `<work_dir>/DONE`.
- `tools/export_run.py`: after training, writes `results/<config name>/` with `report.md` (config, results,
  block-by-block analysis, history, what every file means), per-clip embeddings after every block, weights, CSVs.
- `amp` (mixed precision, off by default) and `grad_accum` (optimizer step every N batches) in the config; the
  NTU-60 configs use batch 8 x 4 = the original effective batch 32, in fp32.
- `resource_limits` in the config caps CPU, GPU memory, GPU busy time and optionally RAM (`resource_limits.py`).
- `scripts/*.ps1` (Windows): `setup_env` installs Python / PyTorch with CUDA / requirements, `prepare_ntu` builds
  the NTU data from the zip, `run_training` does a complete run (train + export, resumable) with live progress,
  `show_usage` shows what it uses; `register_training_task` / `stop_training` for running it via Task Scheduler.
- Progress lines (run % done, time left, finish time) every 10 minutes in the log and `<work_dir>/progress.txt`.

## Changes made to get it running (no model / training logic changed)

- `model/activations.py`: restored from git history (it was deleted in the last commit but is still imported).
- `model/__init__.py`, `config/ntu/train_joint_amstgcn_ntu.yaml`: `AEMST_GCN` -> `ST_GCN` (the module never existed).
- `main.py`:
  - `yaml.load(f, Loader=yaml.FullLoader)` (required by PyYAML >= 6);
  - accepts `--local-rank` as well as `--local_rank` (torch >= 2 launchers) and reads `LOCAL_RANK` from the env;
  - runs as a single process without `torch.distributed` when not started by a launcher, and falls back to CPU
    when CUDA is unavailable; under `torchrun` the original DDP + SyncBatchNorm + NCCL path is unchanged;
  - **bug fix**: `eval()` did `output = self.model(data)`, but the model returns `(features, logits)`;
    it now unpacks `_, output = ...` (eval crashed before this fix);
  - removed `verbose=True` from the (unused) `ReduceLROnPlateau`, which recent torch no longer accepts;
  - `test_feeder_args['debug']` -> `.get('debug', False)` so test mode works with the provided configs.
- New: `requirements.txt`, `config/ntu/ntu60_{xsub,xview}_joint.yaml`, `config/smoke_test.yaml`,
  `tools/make_dummy_data.py`, this file.
- `feeders/feeder.py`: DataLoader workers on Windows re-open the memory-mapped data instead of receiving a pickled
  copy of the whole array.
- `data_gen/preprocess.py`: the rotation steps are vectorised (same output bit for bit, ~15x faster overall);
  `data_gen/ntu_gendata.py` writes through a memory map and takes `--benchmark`.
- `main.py`: the learning rate is a plain float (a numpy float in the optimizer state blocked loading checkpoints).
