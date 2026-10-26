"""Measure training speed and peak GPU memory of the model in a config, on random data.

Prints seconds per training sample, peak memory, and the projected epoch / full-run time
for NTU RGB+D splits, so you can check a machine before preparing the real dataset.

Example (from the repo root):
    python tools/bench_train.py --config config/ntu/ntu60_xsub_joint.yaml --frames 300 64 --batch 16 32 --amp

Times are with the GPU running flat out; with `gpu_duty` in the config's resource_limits, divide the
throughput by gpu_duty (0.5 -> runs take 2x longer). --gpu-mem-fraction applies the same memory cap
as resource_limits.gpu_mem_fraction, so batches that would not fit under it show as "out of memory".
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.utils import import_class

# number of training samples per split (NTU-60 with the 302 missing skeletons removed)
SPLITS = {'ntu60-xsub': 40091, 'ntu60-xview': 37646, 'ntu120-xsub': 63026, 'ntu120-xset': 54468}


def bench(model_args, model_name, frames, batch, amp, iters, warmup, device):
    model = import_class(model_name)(**model_args).to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    scaler = torch.amp.GradScaler(enabled=amp and device.type == 'cuda')
    x = torch.randn(batch, 3, frames, model_args['num_point'], model_args['num_person'], device=device)
    y = torch.randint(0, model_args['num_class'], (batch,), device=device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + iters):
        if i == warmup:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
        with torch.autocast(device.type, enabled=amp and device.type == 'cuda'):
            _, out = model(x)
            loss = F.cross_entropy(out, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    per_sample = (time.time() - start) / iters / batch
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else float('nan')
    return per_sample, peak


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/ntu/ntu60_xsub_joint.yaml')
    parser.add_argument('--frames', type=int, nargs='+', default=[300, 64])
    parser.add_argument('--batch', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--amp', action='store_true', help='also measure mixed precision (fp16 autocast)')
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=None, help='defaults to num_epoch in the config')
    parser.add_argument('--gpu-mem-fraction', type=float, default=None,
                        help='cap PyTorch GPU memory at this fraction of total VRAM')
    arg = parser.parse_args()

    with open(arg.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    epochs = arg.epochs or cfg.get('num_epoch', 110)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print('device: {}'.format(torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'))
    if arg.gpu_mem_fraction and device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(arg.gpu_mem_fraction)
        print('GPU memory capped at {:.0%} of {:.1f} GB'.format(
            arg.gpu_mem_fraction, torch.cuda.get_device_properties(0).total_memory / 1e9))

    for frames in arg.frames:
        for batch in arg.batch:
            for amp in ([False, True] if arg.amp else [False]):
                try:
                    sec, peak = bench(cfg['model_args'], cfg['model'], frames, batch, amp,
                                      arg.iters, arg.warmup, device)
                except torch.OutOfMemoryError:
                    print('T={:3d} batch={:3d} amp={!s:5}: out of memory'.format(frames, batch, amp))
                    torch.cuda.empty_cache()
                    continue
                proj = ', '.join('{} {:.1f} min/epoch ({:.1f} h for {} epochs)'.format(
                    k, sec * n / 60, sec * n * epochs / 3600, epochs) for k, n in SPLITS.items())
                print('T={:3d} batch={:3d} amp={!s:5}: {:.4f} s/sample, peak mem {:.2f} GB | {}'.format(
                    frames, batch, amp, sec, peak, proj))
