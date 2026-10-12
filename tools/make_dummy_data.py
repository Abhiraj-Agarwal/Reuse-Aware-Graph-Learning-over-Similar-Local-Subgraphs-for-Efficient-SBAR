"""Write a tiny random dataset in the exact format produced by data_gen/ntu_gendata.py.

Used only to check that the training / eval pipeline runs end to end (config/smoke_test.yaml).
Data: float32 array (N, C=3, T, V=25, M=2); labels: pickle of (sample_names, labels).
"""
import argparse
import os
import pickle

import numpy as np


def make_split(out_dir, part, num_samples, num_frames, num_class, rng):
    data = rng.standard_normal((num_samples, 3, num_frames, 25, 2)).astype(np.float32)
    # Like NTU, most samples have a single actor: zero out the second body.
    single = rng.random(num_samples) < 0.8
    data[single, :, :, :, 1] = 0
    labels = rng.integers(0, num_class, num_samples).tolist()
    names = ['S001C001P001R001A{:03d}.skeleton'.format(l + 1) for l in labels]

    np.save(os.path.join(out_dir, '{}_data_joint.npy'.format(part)), data)
    with open(os.path.join(out_dir, '{}_label.pkl'.format(part)), 'wb') as f:
        pickle.dump((names, labels), f)
    print('{}: data {} -> {}'.format(part, data.shape, out_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_folder', default='data/dummy')
    parser.add_argument('--num_train', type=int, default=32)
    parser.add_argument('--num_val', type=int, default=16)
    parser.add_argument('--num_frames', type=int, default=300)
    parser.add_argument('--num_class', type=int, default=60)
    arg = parser.parse_args()

    os.makedirs(arg.out_folder, exist_ok=True)
    rng = np.random.default_rng(0)
    make_split(arg.out_folder, 'train', arg.num_train, arg.num_frames, arg.num_class, rng)
    make_split(arg.out_folder, 'val', arg.num_val, arg.num_frames, arg.num_class, rng)
