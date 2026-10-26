"""Export everything a finished training run produced into one folder that explains itself.

    python tools/export_run.py --config config/ntu/ntu60_xsub_joint.yaml [--weights final|best|<file.pt>]

Runs the trained model once over the train and val clips (fp32, no augmentation) and writes results/<config name>/:

    report.md                 read this first: config, result, block-by-block table, history, what every file holds
    summary.json              the same numbers, machine-readable
    embeddings_<split>.npz    per clip: label, prediction, logits, final embedding, embedding after every block
    training_history.csv      one row per epoch
    per_class_accuracy.csv    val accuracy per action
    confusion_matrix.csv      val clips: rows = true action, columns = predicted action
    model_weights.pt          the exported weights (state_dict)
    config.yaml               the exact config of the run

scripts/run_training.ps1 runs this automatically after the last epoch. report.md is written last, so its presence
means the export finished.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import resource_limits
from model.utils import import_class
from inspect_data import NTU60_ACTIONS, receptive_fields  # tools/inspect_data.py


def cpu_name():
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
            return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
    if os.path.exists('/proc/cpuinfo'):
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor()


def ram_gb():
    if sys.platform == 'win32':
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ('total', 'avail', 'pagefile_total', 'pagefile_avail',
                                                          'virtual_total', 'virtual_avail', 'ext')]
        s = MemoryStatus(dwLength=ctypes.sizeof(MemoryStatus))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return round(s.total / 2 ** 30, 1)
    return round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30, 1)


def git_info():
    try:
        run = lambda *a: subprocess.run(['git'] + list(a), cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return {'commit': run('rev-parse', '--short', 'HEAD'),
                'uncommitted_changes': bool(run('status', '--porcelain', '--untracked-files=no'))}
    except OSError:
        return {'commit': 'unknown', 'uncommitted_changes': None}


def machine_info(device):
    info = {'os': platform.platform(), 'cpu': cpu_name(), 'logical_cpus': os.cpu_count(), 'ram_gb': ram_gb(),
            'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'numpy': np.__version__,
            'cudnn_tf32_convolutions': torch.backends.cudnn.allow_tf32}
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 1e9, 1))
    return info


def load_weights(arg, cfg, ckpt):
    if arg.weights == 'final':
        return ckpt['model'], 'final model (after epoch {})'.format(ckpt['epoch'] + 1)
    if arg.weights == 'best':
        best_epoch = ckpt['best_acc'][0]
        files = glob.glob('{}-{}-*.pt'.format(cfg['model_saved_name'], best_epoch - 1))
        if not files:
            sys.exit('no weight file for the best epoch {} next to {}'.format(best_epoch, cfg['model_saved_name']))
        return torch.load(files[0], map_location='cpu'), 'best validation epoch {} ({})'.format(best_epoch, files[0])
    return torch.load(arg.weights, map_location='cpu'), arg.weights


def run_model(model, cfg, split, device, batch, throttle):
    args = dict(cfg['{}_feeder_args'.format('train' if split == 'train' else 'test')])
    args.update(random_choose=False, random_shift=False, random_move=False)  # the clips as stored, no augmentation
    dataset = import_class(cfg['feeder'])(**args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False,
                                         num_workers=cfg.get('num_worker', 0), pin_memory=device.type == 'cuda')
    n, people = len(dataset), cfg['model_args']['num_person']
    blocks = [np.zeros((n, args[1]), np.float32) for args in cfg['model_args']['block_args']]
    logits = np.zeros((n, cfg['model_args']['num_class']), np.float32)
    pooled = {}
    hooks = [layer.register_forward_hook(lambda mod, inp, out, k=k: pooled.__setitem__(k, out.mean(dim=(2, 3))))
             for k, layer in enumerate(model.layers)]
    i, start = 0, time.time()
    with torch.no_grad():
        for b, (data, label, index) in enumerate(loader):
            throttle.start()
            _, out = model(data.float().to(device))
            throttle.stop()
            m = data.shape[0]
            for k in range(len(blocks)):
                # average over time and joints (the hook), then over the people, exactly as the model's own pooling
                blocks[k][i:i + m] = pooled[k].view(m, people, -1).mean(1).cpu().numpy()
            logits[i:i + m] = out.cpu().numpy()
            i += m
            if b % 50 == 0:
                print('  {} {}/{} clips'.format(split, i, n), flush=True)
    for h in hooks:
        h.remove()
    print('  {} done in {:.1f} min'.format(split, (time.time() - start) / 60))
    return dict(sample_name=np.array(dataset.sample_name), label=np.array(dataset.label, dtype=np.int64),
                logits=logits, blocks=blocks)


def block_separability(train, val, num_class):
    """Per block: nearest-class-mean accuracy on val (class means from train) and cosine similarities on val."""
    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    rows = []
    for k in range(len(val['blocks'])):
        va, vy = unit(val['blocks'][k]), val['label']
        classes = [c for c in range(num_class) if (vy == c).sum() > 1]
        sums = np.stack([va[vy == c].sum(0) for c in classes])
        counts = np.array([(vy == c).sum() for c in classes], dtype=np.float64)
        same = ((sums ** 2).sum(1) - counts).sum() / (counts * (counts - 1)).sum()
        total = sums.sum(0)
        diff = ((total ** 2).sum() - (sums ** 2).sum()) / (counts.sum() ** 2 - (counts ** 2).sum())
        row = {'cos_same_action': float(same), 'cos_different_action': float(diff)}
        tr, ty = unit(train['blocks'][k]), train['label']
        present = [c for c in range(num_class) if (ty == c).any()]
        means = unit(np.stack([tr[ty == c].mean(0) for c in present]))
        pred = np.array(present)[(va @ means.T).argmax(1)]
        row['nearest_class_mean_acc'] = float((pred == vy).mean())
        rows.append(row)
    return rows


def model_structure(model, cfg, frames):
    """Per block: channels, stride, output shape, parameters, GMACs per clip, receptive field."""
    from torch.utils.flop_counter import FlopCounterMode
    m = cfg['model_args']
    x = torch.randn(1, 3, frames, m['num_point'], m['num_person'])
    counter = FlopCounterMode(display=False)
    shapes = {}
    hooks = [layer.register_forward_hook(lambda mod, i, o, k=k: shapes.__setitem__(k, list(o.shape[1:])))
             for k, layer in enumerate(model.layers)]
    with torch.no_grad(), counter:
        model(x)
    for h in hooks:
        h.remove()
    flops = counter.get_flop_counts()
    reach = receptive_fields(model, frames, m['num_point'], m['num_person'])
    rows = []
    for k, (layer, args) in enumerate(zip(model.layers, m['block_args'])):
        rows.append({'block': k + 1, 'in_channels': args[0], 'out_channels': args[1], 'residual': bool(args[2]),
                     'temporal_stride': args[3], 'output_C_T_V': shapes[k],
                     'params': sum(p.numel() for p in layer.parameters()),
                     'gmacs_per_clip': round(sum(flops.get('Model.layers.{}'.format(k), {}).values()) / 2e9, 3),
                     'frames_seen': reach[k][1], 'joints_seen': reach[k][2]})
    total = {'params': sum(p.numel() for p in model.parameters()),
             'gmacs_per_clip': round(counter.get_total_flops() / 2e9, 3)}
    return rows, total


def classification(res, num_class):
    pred = res['logits'].argmax(1)
    top5 = np.argsort(-res['logits'], 1)[:, :5]
    y = res['label']
    conf = np.zeros((num_class, num_class), np.int64)
    np.add.at(conf, (y, pred), 1)
    return {'top1': float((pred == y).mean()), 'top5': float((top5 == y[:, None]).any(1).mean()),
            'pred': pred, 'confusion': conf}


def action(c):
    return NTU60_ACTIONS[c] if c < len(NTU60_ACTIONS) else 'class {}'.format(c)


def pct(x):
    return '-' if x is None else '{:.2f}%'.format(100 * x)


def write_report(path, s):
    cfg, ma, tr = s['config'], s['config']['model_args'], s['training']
    blocks, hist = s['blocks'], s['training']['history']
    L = []
    w = L.append
    w('# Run report: {}'.format(s['run_name']))
    w('')
    w('Generated {} from `{}`. Exported weights: {}. Code: git `{}`{}.'.format(
        s['generated'], s['config_path'], s['weights'], s['git']['commit'],
        ' with uncommitted changes' if s['git']['uncommitted_changes'] else ''))
    w('')
    w('## 1. Result')
    w('')
    w('| | top-1 | top-5 |')
    w('|---|---|---|')
    for split in ('val', 'train'):
        if split in s['results']:
            r = s['results'][split]
            w('| {} ({} clips), exported weights | {} | {} |'.format(split, r['clips'], pct(r['top1']), pct(r['top5'])))
    w('| best val top-1 seen during training (epoch {}) | {} | |'.format(tr['best_epoch'], pct(tr['best_val_acc'])))
    w('')
    w('"Best during training" picks the epoch by its val accuracy (as the original code reports it), so it is '
      'slightly optimistic; the final-epoch val number is the unbiased one.')
    w('')
    w('## 2. What was trained')
    w('')
    w('| | |')
    w('|---|---|')
    w('| data | `{}` (train) / `{}` (val) |'.format(cfg['train_feeder_args']['data_path'], cfg['test_feeder_args']['data_path']))
    w('| input per clip | 3 coordinates (x, y, z) x {} frames x {} joints x {} people; no augmentation |'.format(
        s['input_frames'], ma['num_point'], ma['num_person']))
    w('| model | `{}`: MST-GCN, {} blocks, channels {}, graph `{}` ({}), block type `{}`, s_scale {}, t_scale {}, '
      'temporal kernel {}, activation {}, attention {} |'.format(
          cfg['model'], len(ma['block_args']), ' / '.join(str(b[1]) for b in ma['block_args']), ma['graph'],
          ma['graph_args'].get('labeling_mode'), ma['block_type'], ma['s_scale'], ma['t_scale'], ma['kernel_size'],
          ma['act'], ma['atten']))
    w('| size | {:.3f} M parameters, {} GMACs per clip ({} frames, both people) |'.format(
        s['model_total']['params'] / 1e6, s['model_total']['gmacs_per_clip'], s['input_frames']))
    w('| classes | {} actions |'.format(ma['num_class']))
    w('| epochs | {} (lr warm-up over {}, x0.1 at epochs {}) |'.format(cfg['num_epoch'], cfg['warm_up_epoch'], cfg['step']))
    w('| optimiser | {} lr {}, momentum 0.9, nesterov {}, weight decay {} |'.format(
        cfg['optimizer'], cfg['base_lr'], cfg['nesterov'], cfg['weight_decay']))
    w('| batch | {} clips x {} accumulation steps = {} clips per optimizer step |'.format(
        cfg['batch_size'], cfg.get('grad_accum', 1), cfg['batch_size'] * cfg.get('grad_accum', 1)))
    w('| precision | {} |'.format('mixed (fp16 autocast)' if cfg.get('amp') else 'fp32 (as the original code)'))
    w('| seed | {} |'.format(cfg.get('seed', 1)))
    w('| resource limits | {} |'.format(cfg.get('resource_limits') or 'none'))
    w('')
    w('Full config: `config.yaml` (as given) and `run_arguments.yaml` (with every default filled in) in this folder.')
    w('')
    w('## 3. Block by block: what each part of the model produces')
    w('')
    w('| block | channels in -> out | temporal stride | output (C x T x V) per person | params | GMACs per clip '
      '| one output sees (frames / joints) | nearest-class-mean val acc | cos same action | cos different action |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for b in blocks:
        w('| {} | {} -> {} | {} | {} | {:,} | {} | {} / {} | {} | {:.3f} | {:.3f} |'.format(
            b['block'], b['in_channels'], b['out_channels'], b['temporal_stride'], ' x '.join(map(str, b['output_C_T_V'])),
            b['params'], b['gmacs_per_clip'], b['frames_seen'], b['joints_seen'], pct(b.get('nearest_class_mean_acc')),
            b['cos_same_action'], b['cos_different_action']))
    w('')
    w('Every block is a multi-scale graph convolution over the 25 joints followed by a multi-scale temporal convolution. '
      'Reading the columns:')
    w('')
    w('- **output**: feature map of one person after the block (C channels, T time steps, V joints); a stride of 2 '
      'halves the time steps.')
    w('- **GMACs per clip**: multiply-accumulates the block costs for one clip with both people; summed over the blocks '
      'this is (almost all of) the model cost in section 2.')
    w('- **one output sees**: the receptive field. How many input frames and joints a single output position of the block '
      'depends on (middle time step, right hand). Early blocks are local; later ones see nearly the whole clip.')
    w('- **embedding after the block** (`block_XX` in the .npz files): the block output averaged over time steps, joints '
      'and the people, one vector of C numbers per clip.')
    w('- **nearest-class-mean val acc**: classify each val clip by the closest (cosine) average train embedding of an '
      'action, using that block\'s embedding. Shows how much action information each block has built up, with no '
      'extra training.')
    w('- **cos same / different action**: mean cosine similarity between val clips of the same action and between clips '
      'of different actions. The features are non-negative (ReLU), so both are high; the gap between them is what matters, '
      'and how similar clips are at each depth is the redundancy that reuse-aware computation would exploit.')
    w('')
    w('## 4. Training history')
    w('')
    w('{} epochs; {:.1f} h of training (sum over epochs, with the GPU throttled to the resource limits); '
      'first epoch started {}, last finished {}; resumed {} time(s) after an interruption.'.format(
          len(hist), tr['train_hours'], tr['first_epoch_started'], tr['last_epoch_finished'], tr['resumes']))
    w('')
    w('| epoch | lr | train loss | train acc | val acc | minutes | peak RAM GB | peak GPU GB |')
    w('|---|---|---|---|---|---|---|---|')
    for h in hist:
        w('| {} | {:.5g} | {:.4f} | {} | {} | {} | {} | {} |'.format(
            h['epoch'], h['lr'], h['train_loss'], pct(h['train_acc']), pct(h['val_acc']), h['train_minutes'],
            h.get('ram_gb', '-'), h.get('gpu_gb', '-')))
    w('')
    w('Val accuracy is only measured after epoch {} (original schedule). Train accuracy is the running average over the '
      'epoch while the weights change.'.format(cfg['step'][0]))
    w('')
    if 'val' in s['results']:
        per = s['results']['val']['per_class']
        ranked = sorted(per, key=lambda r: r['acc'])
        w('## 5. Per action (val)')
        w('')
        w('Hardest: ' + ', '.join('A{:03d} {} {}'.format(r['id'], r['action'], pct(r['acc'])) for r in ranked[:5]) + '.')
        w('')
        w('Easiest: ' + ', '.join('A{:03d} {} {}'.format(r['id'], r['action'], pct(r['acc'])) for r in ranked[-5:]) + '.')
        w('')
        w('Most confused (true -> predicted, share of the true action\'s clips):')
        w('')
        for c in s['results']['val']['most_confused']:
            w('- A{:03d} {} -> A{:03d} {}: {} clips ({})'.format(c['true'], c['true_action'], c['pred'],
                                                               c['pred_action'], c['clips'], pct(c['share'])))
        w('')
        w('All actions: `per_class_accuracy.csv`; full matrix: `confusion_matrix.csv`.')
        w('')
    w('## 6. Files in this folder')
    w('')
    w('| file | contents |')
    w('|---|---|')
    for f, d in s['files'].items():
        w('| `{}` | {} |'.format(f, d))
    w('')
    w('Arrays in `embeddings_<split>.npz` (N = clips in the split, row i of every array is the same clip):')
    w('')
    w('| key | shape | meaning |')
    w('|---|---|---|')
    w('| `sample_name` | N | NTU file name, e.g. S001C002P003R002A013 = setup 1, camera 2, performer 3, '
      'repetition 2, action 13 |')
    w('| `label` | N | true action, 0-based (A013 -> 12) |')
    w('| `pred` | N | predicted action, 0-based |')
    w('| `logits` | N x {} | classifier scores before softmax |'.format(ma['num_class']))
    w('| `embedding` | N x {} | the clip embedding the classifier sees (= `block_{:02d}`) |'.format(
        ma['block_args'][-1][1], len(ma['block_args'])))
    for b in blocks:
        w('| `block_{:02d}` | N x {} | embedding after block {} (see section 3) |'.format(b['block'], b['out_channels'], b['block']))
    w('')
    w('```python')
    w('import numpy as np')
    w("d = np.load('embeddings_val.npz')")
    w("X, y = d['embedding'], d['label']          # one row per clip")
    w("early = d['block_01']                       # after the first block")
    w('```')
    w('')
    w('## 7. How this run differs from the original MST-GCN training')
    w('')
    w('The model, data preprocessing, optimiser, learning-rate schedule, epochs, input length and seed are unchanged. '
      'Differences:')
    w('')
    w('- 1 GPU instead of 4. The original used 4 x 8 clips with synchronised BatchNorm; here {} clips x {} accumulation '
      'steps give the same averaged gradient over {} clips and the same number of optimizer steps per epoch, but '
      'BatchNorm statistics come from {} clips at a time instead of 32.'.format(
          cfg['batch_size'], cfg.get('grad_accum', 1), cfg['batch_size'] * cfg.get('grad_accum', 1), cfg['batch_size']))
    w('- Precision: {}. cuDNN TF32 convolutions: {} (PyTorch\'s default on Ampere and newer GPUs such as the RTX 4500 Ada; the '
      'original code did not change it).'.format('mixed fp16' if cfg.get('amp') else 'fp32, as the original',
                                                 s['machine']['cudnn_tf32_convolutions']))
    w('- The CPU / GPU limits only add pauses; they do not change any number. Interrupted runs continue from the last '
      'finished epoch with the same weights, optimizer state and data order (no dropout or augmentation), so they '
      'reproduce an uninterrupted run up to GPU floating-point non-determinism.')
    w('- The step counter in TensorBoard and in the weight file names counts batches of {} clips.'.format(cfg['batch_size']))
    w('')
    w('## 8. Machine and software')
    w('')
    for k, v in s['machine'].items():
        w('- {}: {}'.format(k, v))
    w('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', default='final', help="'final' (checkpoint.pt), 'best' or a .pt file")
    parser.add_argument('--out', help='default: results/<config name>')
    arg = parser.parse_args()

    os.chdir(ROOT)  # config paths are relative to the repo
    with open(arg.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # main.py saves every argument it ran with (config + defaults such as optimizer and seed) in the work dir
    run_args_path = os.path.join(cfg['work_dir'], 'config.yaml')
    if os.path.exists(run_args_path):
        with open(run_args_path) as f:
            cfg = dict(yaml.load(f, Loader=yaml.FullLoader), **cfg)
    run_name = os.path.splitext(os.path.basename(arg.config))[0]
    out = arg.out or os.path.join('results', run_name)
    os.makedirs(out, exist_ok=True)
    report = os.path.join(out, 'report.md')
    if os.path.exists(report):
        os.remove(report)  # written again at the very end

    ckpt = torch.load(os.path.join(cfg['work_dir'], 'checkpoint.pt'), map_location='cpu')
    weights, weights_desc = load_weights(arg, cfg, ckpt)
    weights = {k.split('module.')[-1]: v for k, v in weights.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for line in resource_limits.apply(cfg.get('resource_limits'), device):
        print('Resource limits: ' + line)
    resource_limits.keep_awake()
    throttle = resource_limits.GpuThrottle((cfg.get('resource_limits') or {}).get('gpu_duty'), device)

    Model = import_class(cfg['model'])
    model = Model(**cfg['model_args'])
    model.load_state_dict(weights)
    model.eval()
    frames = np.load(cfg['test_feeder_args']['data_path'], mmap_mode='r').shape[2]

    print('Model structure, cost and receptive fields ...')
    structure, total = model_structure(model, cfg, frames)
    model.to(device)
    num_class = cfg['model_args']['num_class']
    res = {}
    for split in ('train', 'val'):  # train for the per-block class means, val for everything reported
        print('Running the model over {} ...'.format(split))
        res[split] = run_model(model, cfg, split, device, cfg.get('test_batch_size', 32), throttle)
    print('Analysing ...')
    sep = block_separability(res['train'], res['val'], num_class)
    blocks = [dict(a, **b) for a, b in zip(structure, sep)]

    files = {'report.md': 'this report', 'summary.json': 'all numbers in this report, machine-readable',
             'config.yaml': 'the config file the run was started with',
             'model_weights.pt': 'exported weights: ' + weights_desc,
             'training_history.csv': 'one row per epoch (section 4)'}
    results = {}
    for split, r in res.items():
        c = classification(r, num_class)
        results[split] = {'clips': int(len(r['label'])), 'top1': c['top1'], 'top5': c['top5']}
        arrays = {'sample_name': r['sample_name'], 'label': r['label'], 'pred': c['pred'], 'logits': r['logits'],
                  'embedding': r['blocks'][-1]}
        arrays.update({'block_{:02d}'.format(k + 1): b for k, b in enumerate(r['blocks'])})
        np.savez(os.path.join(out, 'embeddings_{}.npz'.format(split)), **arrays)
        files['embeddings_{}.npz'.format(split)] = '{} {} clips: labels, predictions, logits, embeddings (below)'.format(
            len(r['label']), split)
        if split == 'val':
            conf = c['confusion']
            n = conf.sum(1)
            per = [{'id': k + 1, 'action': action(k), 'clips': int(n[k]),
                    'acc': float(conf[k, k] / n[k]) if n[k] else None} for k in range(num_class)]
            results[split]['per_class'] = [p for p in per if p['acc'] is not None]
            off = conf.copy()
            np.fill_diagonal(off, 0)
            pairs = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:10], off.shape))[0]
            results[split]['most_confused'] = [
                {'true': int(t) + 1, 'true_action': action(t), 'pred': int(p) + 1, 'pred_action': action(p),
                 'clips': int(off[t, p]), 'share': float(off[t, p] / n[t])} for t, p in pairs if off[t, p] > 0]
            with open(os.path.join(out, 'per_class_accuracy.csv'), 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(['action_id', 'action', 'val_clips', 'val_accuracy'])
                for p in per:
                    wr.writerow([p['id'], p['action'], p['clips'], '' if p['acc'] is None else round(p['acc'], 4)])
            with open(os.path.join(out, 'confusion_matrix.csv'), 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(['true \\ predicted'] + ['A{:03d}'.format(k + 1) for k in range(num_class)])
                for k in range(num_class):
                    wr.writerow(['A{:03d}'.format(k + 1)] + conf[k].tolist())
            files['per_class_accuracy.csv'] = 'val accuracy per action (section 5)'
            files['confusion_matrix.csv'] = 'val clips: rows = true action, columns = predicted action'

    hist = ckpt.get('history', [])
    with open(os.path.join(out, 'training_history.csv'), 'w', newline='') as f:
        keys = ['epoch', 'lr', 'train_loss', 'train_acc', 'val_acc', 'train_minutes', 'finished', 'ram_gb', 'gpu_gb']
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(hist)
    torch.save(weights, os.path.join(out, 'model_weights.pt'))
    shutil.copy2(arg.config, os.path.join(out, 'config.yaml'))
    if os.path.exists(run_args_path):
        shutil.copy2(run_args_path, os.path.join(out, 'run_arguments.yaml'))
        files['run_arguments.yaml'] = 'every argument main.py ran with: the config plus its defaults'

    fmt = '%Y-%m-%d %H:%M:%S'
    first = (datetime.datetime.strptime(hist[0]['finished'], fmt) -
             datetime.timedelta(minutes=hist[0]['train_minutes'])).strftime(fmt) if hist else '-'
    summary = {
        'run_name': run_name, 'config_path': arg.config, 'generated': time.strftime(fmt), 'weights': weights_desc,
        'git': git_info(), 'machine': machine_info(device), 'config': cfg, 'input_frames': int(frames),
        'model_total': total, 'blocks': blocks, 'results': results, 'files': files,
        'training': {'epochs_done': len(hist), 'best_epoch': ckpt['best_acc'][0], 'best_val_acc': ckpt['best_acc'][1],
                     'train_hours': round(sum(h['train_minutes'] for h in hist) / 60, 2),
                     'first_epoch_started': first, 'last_epoch_finished': hist[-1]['finished'] if hist else '-',
                     'resumes': ckpt.get('resumes', 0), 'history': hist}}
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=str)
    write_report(report, summary)
    print('Wrote {}'.format(os.path.abspath(out)))


if __name__ == '__main__':
    main()
