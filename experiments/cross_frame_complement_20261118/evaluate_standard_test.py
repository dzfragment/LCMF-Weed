"""Evaluate reference substitutions on all of exp4's official test triplets."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader, Dataset

from evaluate import (ARMS, HERE, SIZE, architecture_flags, args_for, assert_checkpoint,
                      digest, keep_list, load_labels, model_paths, pack, read_image, trainer)
from evaluate_followup import appearance
from recover_standard_test import TEST


MODES = {'full': (0, 1, 2), 'ref1_twice': (0, 1, 1), 'ref2_twice': (0, 2, 2),
         'key_tripled': (0, 0, 0)}


def choose_donors(rows, descriptors):
    groups = np.asarray([row['group'] for row in rows])
    strata = defaultdict(list)
    for idx, row in enumerate(rows):
        strata[(row['kind'], *row['blur'][1:])].append(idx)
    features = (descriptors - descriptors.mean(axis=0)) / np.maximum(descriptors.std(axis=0), 1e-5)
    distances = np.linalg.norm(features[:, None] - features[None], axis=2)
    rng = np.random.default_rng(20261118)
    donors = np.full((3, len(rows)), -1, dtype=int)
    kinds = np.asarray([row['kind'] for row in rows])
    for indices in strata.values():
        count = Counter(groups[indices])
        pool = np.asarray(indices)
        if 2 * max(count.values()) + 3 > len(pool):
            pool = np.asarray([idx for idx, row in enumerate(rows)
                               if row['blur'][1:] == rows[indices[0]]['blur'][1:]])
        costs = distances[np.ix_(indices, pool)] + 100 * (kinds[indices, None] != kinds[pool])
        banned = groups[indices, None] == groups[pool]
        for repeat in range(len(donors)):
            ranking = costs + rng.uniform(0, .05, size=costs.shape)
            ranking[banned] = np.inf
            sources, targets = linear_sum_assignment(ranking)
            assert len(sources) == len(indices) and np.isfinite(ranking[sources, targets]).all()
            donors[repeat, np.asarray(indices)[sources]] = pool[targets]
            banned[sources, targets] = True
            assert len(set(donors[repeat, indices])) == len(indices)
    for idx, row in enumerate(rows):
        assert len(set(donors[:, idx])) == 3
        for donor in donors[:, idx]:
            assert rows[donor]['group'] != row['group'] and rows[donor]['blur'][1:] == row['blur'][1:]
            assert not set(rows[donor]['hashes']) & set(row['hashes'])
    return donors


class OfficialTest(Dataset):
    def __init__(self, manifest, limit):
        self.folders = sorted(TEST.glob('seq_*'))
        self.limit = limit
        records = json.loads(manifest.read_text())['sequences']
        assert set(records) == {folder.name for folder in self.folders}
        self.rows = [dict(records[folder.name], name=folder.name) for folder in self.folders]
        self.frames, features = [], []
        for folder, row in zip(self.folders, self.rows, strict=True):
            paths = [folder / f'frame_{idx}.jpg' for idx in range(3)]
            assert row['hashes'] == [sha256(path.read_bytes()).hexdigest() for path in paths]
            images = [read_image(path) for path in paths]
            self.frames.append(pack(images))
            features.append([value for frame in images[1:] for value in appearance(frame)])
        self.donors = choose_donors(self.rows, np.asarray(features))

    def __len__(self):
        return min(self.limit, len(self.folders)) if self.limit is not None else len(self.folders)

    def __getitem__(self, index):
        donor_indices = self.donors[:, index]
        folder = self.folders[index]
        images = torch.cat([self.frames[index], *[self.frames[donor][1:] for donor in donor_indices]])
        return dict(self.rows[index], gt=load_labels(folder / 'labels_frame0.txt'), images=images,
                    donors=[self.rows[donor]['name'] for donor in donor_indices])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=HERE / 'standard_test_manifest.json')
    parser.add_argument('--output', type=Path, default=HERE / 'standard_test_predictions.pkl')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    dataset = OfficialTest(args.manifest, args.limit)
    loader = iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                             collate_fn=keep_list))
    training = trainer()
    from pad_core import PADSingleFrame, decode_predictions

    paths, sf_path = model_paths(args.seeds, 'concat')
    arm = ARMS[0]
    model_args = args_for(architecture_flags(arm))
    models = {}
    for seed, path in paths.items():
        assert_checkpoint(path, arm)
        models[seed] = training.load_ckpt(str(path), model_args, 'cuda', prefer_ema=False)[0].eval()
    sf_ck = torch.load(sf_path, map_location='cpu', weights_only=False)
    sf = PADSingleFrame(sf_ck['base_model'], sf_ck['head_cfg']).cuda().eval()
    sf_state = sf_ck['model']
    sf.load_state_dict(sf_state.state_dict() if isinstance(sf_state, torch.nn.Module) else sf_state,
                       strict=True)
    plan = MODES | {f'wrong_both_{idx}': (0, first, first + 1)
                    for idx, first in enumerate((3, 5, 7), 1)}
    rows = []
    for batch_index, batch in enumerate(loader):
        frames = torch.stack([row.pop('images') for row in batch]).cuda().float() / 255
        for row, pred in zip(batch, decode_predictions(sf(frames[:, 0].contiguous()), SIZE), strict=True):
            row['predictions'] = {'sf_key': pred}
        for seed, model in models.items():
            for name, slots in plan.items():
                for row, pred in zip(batch, decode_predictions(model(frames[:, slots]), SIZE), strict=True):
                    row['predictions'][f'mf{seed}_{name}'] = pred
        rows.extend(batch)
        if batch_index % 8 == 0 or len(rows) == len(dataset):
            print(f'standard test: {len(rows)}/{len(dataset)}', flush=True)
    output = dict(rows=rows, model_hashes={seed: digest(path) for seed, path in paths.items()} |
                  {'sf': digest(sf_path)}, manifest_sha256=digest(args.manifest), test_root=str(TEST))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    with temp.open('wb') as handle:
        pickle.dump(output, handle)
    temp.replace(args.output)


if __name__ == '__main__':
    main()
