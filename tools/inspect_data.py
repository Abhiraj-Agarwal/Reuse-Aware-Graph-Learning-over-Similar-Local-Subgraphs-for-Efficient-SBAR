"""Look inside the data and the model from the terminal. Run from the repo root.

    python tools/inspect_data.py summary --config config/ntu/ntu60_xsub_joint.yaml
    python tools/inspect_data.py lengths --raw data/nturgbd_raw/nturgb+d_skeletons [--action 11]
    python tools/inspect_data.py padding --data data/ntu/xsub/val_data_joint.npy --label data/ntu/xsub/val_label.pkl --action 11
    python tools/inspect_data.py shapes --config config/ntu/ntu60_xsub_joint.yaml
    python tools/inspect_data.py reach --config config/ntu/ntu60_xsub_joint.yaml
    python tools/inspect_data.py embed --config config/ntu/ntu60_xsub_joint.yaml --weights runs/.../mstgcn_joint-109-<step>.pt --actions 11 12
"""
import argparse
import collections
import os
import pickle
import statistics
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.utils import import_class

NTU60_ACTIONS = [
    'drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop', 'pick up', 'throw', 'sit down',
    'stand up', 'clapping', 'reading', 'writing', 'tear up paper', 'wear jacket', 'take off jacket',
    'wear a shoe', 'take off a shoe', 'wear on glasses', 'take off glasses', 'put on a hat/cap',
    'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something', 'reach into pocket', 'hopping',
    'jump up', 'make a phone call', 'playing with phone/tablet', 'typing on a keyboard',
    'pointing to something', 'taking a selfie', 'check time (watch)', 'rub two hands', 'nod head/bow',
    'shake head', 'wipe face', 'salute', 'put palms together', 'cross hands in front', 'sneeze/cough',
    'staggering', 'falling down', 'touch head (headache)', 'touch chest', 'touch back (backache)',
    'touch neck (neckache)', 'nausea/vomiting', 'fan self', 'punch/slap other person',
    'kick other person', 'push other person', 'pat on back of other', 'point finger at other',
    'hugging', 'giving object', 'touch other person\'s pocket', 'handshaking', 'walking towards',
    'walking apart',
]


def name(action):  # action = 1-based NTU id (A011 -> 11)
    return NTU60_ACTIONS[action - 1] if 1 <= action <= len(NTU60_ACTIONS) else '?'


def load_cfg(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_labels(path):
    with open(path, 'rb') as f:
        names, labels = pickle.load(f, encoding='latin1')
    return names, np.array(labels)


def cmd_summary(arg):
    cfg = load_cfg(arg.config)
    m = cfg['model_args']
    graph = import_class(m['graph'])(**m['graph_args'])
    print('Actions (classes) the model predicts : {}'.format(m['num_class']))
    print('Joints per skeleton                  : {}  ({} bones)'.format(m['num_point'], len(graph.inward)))
    print('People per clip                      : {}'.format(m['num_person']))
    counts = {}
    for part in ('train', 'test'):
        args = cfg['{}_feeder_args'.format(part)]
        data = np.load(args['data_path'], mmap_mode='r')
        _, labels = load_labels(args['label_path'])
        counts[part] = collections.Counter(labels.tolist())
        print('{:5s} data shape (clips, xyz, frames, joints, people): {}  -> {} clips, {} actions present'.format(
            part, data.shape, len(labels), len(counts[part])))
    print('\n id  action                         train  test')
    for c in range(m['num_class']):
        print('A{:03d} {:30s} {:5d} {:5d}'.format(c + 1, name(c + 1), counts['train'][c], counts['test'][c]))


def cmd_lengths(arg):
    lens = collections.defaultdict(list)
    for f in os.listdir(arg.raw):
        if not f.endswith('.skeleton'):
            continue
        with open(os.path.join(arg.raw, f)) as fh:
            lens[int(f[17:20])].append(int(fh.readline()))  # first line of the file = number of frames
    actions = [arg.action] if arg.action else sorted(lens)
    print(' id  action                         clips   min  median  mean   max  >300')
    for a in actions:
        L = lens[a]
        print('A{:03d} {:30s} {:5d} {:5d} {:7d} {:5.0f} {:5d} {:5d}'.format(
            a, name(a), len(L), min(L), int(statistics.median(L)), statistics.mean(L), max(L), sum(l > 300 for l in L)))
    every = [l for L in lens.values() for l in L]
    print('\nAll clips: {}  shortest {}  median {}  longest {}  longer than 300: {}'.format(
        len(every), min(every), int(statistics.median(every)), max(every), sum(l > 300 for l in every)))


def real_length(clip):
    # clip: (3, T, V, M). Padding repeats the clip from frame 0, so the first frame that equals frame 0 again
    # is where the copy starts.
    first = clip[:, 0]
    for t in range(1, clip.shape[1]):
        if np.array_equal(clip[:, t], first):
            return t
    return clip.shape[1]


def cmd_padding(arg):
    data = np.load(arg.data, mmap_mode='r')
    names, labels = load_labels(arg.label)
    idx = np.where(labels == arg.action - 1)[0] if arg.action else np.arange(len(labels))
    idx = idx[:arg.n]
    T = data.shape[2]
    print('clip                          action                   real frames  copies to fill {}'.format(T))
    for i in idx:
        L = real_length(np.asarray(data[i]))
        print('{:29s} A{:03d} {:20s} {:11d} {:10.1f}'.format(names[i], labels[i] + 1, name(labels[i] + 1), L, T / L))


def cmd_shapes(arg):
    cfg = load_cfg(arg.config)
    model = import_class(cfg['model'])(**cfg['model_args']).eval()
    m = cfg['model_args']
    x = torch.randn(1, 3, arg.frames, m['num_point'], m['num_person'])
    print('input clip: {}  (clips, xyz, frames, joints, people)'.format(list(x.shape)))
    hooks = [l.register_forward_hook(lambda mod, i, o, k=k: print(
        'after block {:2d}: {}  (people, features, time steps, joints)'.format(k + 1, list(o.shape))))
        for k, l in enumerate(model.layers)]
    with torch.no_grad():
        features, logits = model(x)
    for h in hooks:
        h.remove()
    emb = features.mean(dim=(2, 3)).view(1, m['num_person'], -1).mean(1)
    print('clip embedding after averaging time, joints and people: {}'.format(list(emb.shape)))
    print('classifier weights (one row per action): {}'.format(list(model.fc.weight.shape)))
    print('output scores: {}  (one per action)'.format(list(logits.shape)))


def receptive_fields(model, frames, num_point, num_person, joint=11):
    # how many input frames and joints one output position of each block depends on (its "receptive field"),
    # found by back-propagating from a single output position (middle time step, `joint`) to the input.
    # Returns one (time steps, frames seen, joints seen) tuple per block.
    T, V, M = frames, num_point, num_person
    x = torch.randn(1, 3, T, V, M, requires_grad=True)
    h = model.data_bn(x.permute(0, 4, 3, 1, 2).reshape(1, M * V * 3, T))
    h = h.view(1, M, V, 3, T).permute(0, 1, 3, 4, 2).reshape(M, 3, T, V)
    out = []
    for layer in model.layers:
        h = layer(h)
        g, = torch.autograd.grad(h[0, :, h.shape[2] // 2, joint].abs().sum(), x, retain_graph=True)
        g = g[0, :, :, :, 0].abs()
        seen = (g.sum(dim=(0, 2)) > 0).nonzero().flatten()
        out.append((h.shape[2], int(seen.max() - seen.min() + 1), int((g.sum(dim=(0, 1)) > 0).sum())))
    return out


def cmd_reach(arg):
    cfg = load_cfg(arg.config)
    model = import_class(cfg['model'])(**cfg['model_args']).eval()
    m = cfg['model_args']
    T, V = arg.frames, m['num_point']
    print('block  time steps  frames per step  one output sees: frames (seconds at 30 fps)  joints')
    for k, (steps, span, joints) in enumerate(receptive_fields(model, T, V, m['num_person'], arg.joint)):
        print('{:5d} {:11d} {:16d} {:16d} ({:4.1f} s) {:19d}/{}'.format(
            k + 1, steps, T // steps, span, span / 30, joints, V))


def cmd_embed(arg):
    cfg = load_cfg(arg.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = import_class(cfg['model'])(**cfg['model_args'])
    weights = torch.load(arg.weights, map_location='cpu')
    model.load_state_dict({k.split('module.')[-1]: v for k, v in weights.items()})
    model.to(device).eval()
    args = cfg['test_feeder_args']
    data = np.load(arg.data or args['data_path'], mmap_mode='r')
    _, labels = load_labels(arg.label or args['label_path'])
    embs, tags = [], []
    for a in arg.actions:
        idx = np.where(labels == a - 1)[0][:arg.k]
        if len(idx) == 0:
            print('A{:03d} {:20s}: no clips in this data file, skipped'.format(a, name(a)))
            continue
        x = torch.from_numpy(np.asarray(data[idx])).float().to(device)
        with torch.no_grad():
            features, logits = model(x)
        N, M = x.shape[0], x.shape[4]
        e = features.mean(dim=(2, 3)).view(N, M, -1).mean(1)
        embs.append(e)
        tags += [a] * N
        pred = logits.argmax(1).cpu().numpy() + 1
        print('A{:03d} {:20s}: {} clips -> predicted {}'.format(a, name(a), N, ['A{:03d}'.format(p) for p in pred]))
    E = torch.nn.functional.normalize(torch.cat(embs), dim=1)
    sim = (E @ E.T).cpu().numpy()
    tags = np.array(tags)
    print('\nCosine similarity between clip embeddings (1 = identical direction, 0 = unrelated):')
    present = [a for a in arg.actions if (tags == a).any()]
    for a in present:
        for b in present:
            block = sim[np.ix_(tags == a, tags == b)]
            if a == b:
                block = block[~np.eye(len(block), dtype=bool)]
            if block.size == 0:
                continue
            print('  A{:03d} vs A{:03d}: mean {:.3f}'.format(a, b, block.mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('summary', help='classes, joints, data shapes, clips per action')
    p.add_argument('--config', default='config/ntu/ntu60_xsub_joint.yaml')
    p = sub.add_parser('lengths', help='real frame count per clip, from the raw .skeleton files')
    p.add_argument('--raw', default='data/nturgbd_raw/nturgb+d_skeletons')
    p.add_argument('--action', type=int, help='NTU action id, e.g. 11 = reading')
    p = sub.add_parser('padding', help='real length vs the 300-frame padded length, from the generated .npy')
    p.add_argument('--data', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--action', type=int)
    p.add_argument('--n', type=int, default=10)
    p = sub.add_parser('shapes', help='tensor shape after every block of the model')
    p.add_argument('--config', default='config/ntu/ntu60_xsub_joint.yaml')
    p.add_argument('--frames', type=int, default=300)
    p = sub.add_parser('reach', help='how many frames and joints each block can see')
    p.add_argument('--config', default='config/ntu/ntu60_xsub_joint.yaml')
    p.add_argument('--frames', type=int, default=300)
    p.add_argument('--joint', type=int, default=11, help='0-based joint index (11 = right hand)')
    p = sub.add_parser('embed', help='clip embeddings of a trained model and how similar they are')
    p.add_argument('--config', default='config/ntu/ntu60_xsub_joint.yaml')
    p.add_argument('--weights', required=True)
    p.add_argument('--data')
    p.add_argument('--label')
    p.add_argument('--actions', type=int, nargs='+', default=[11, 12])
    p.add_argument('--k', type=int, default=5)
    arg = parser.parse_args()
    {'summary': cmd_summary, 'lengths': cmd_lengths, 'padding': cmd_padding,
     'shapes': cmd_shapes, 'reach': cmd_reach, 'embed': cmd_embed}[arg.cmd](arg)
