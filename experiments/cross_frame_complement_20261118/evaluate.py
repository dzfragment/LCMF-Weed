"""Frozen-checkpoint cross-frame diagnostics; does not change existing runs or data."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import sys
from zlib import crc32

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
ABLATION = HERE.parent / 'concat_ablation12_20261106'
DATA_ROOT = Path(os.environ.get('LCMF_DATA_ROOT', HERE.parents[1] / 'dataset')).resolve()
REAL = Path(os.environ.get('LCMF_COMPLEMENT_REAL_ROOT', DATA_ROOT / 'cross_frame_real')).resolve()
CLEAN = DATA_ROOT / 'robustness/clean'
SIZE = 1024
HALF = SIZE / 2
sys.path.insert(0, str(ABLATION))
from common import ARMS, architecture_flags, args_for, assert_checkpoint, digest, trainer


def load_labels(path):
    rows = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if rows.shape[1] != 7 or not np.isfinite(rows).all() or np.any(rows[:, 0] != 0):
        raise ValueError(f'Unexpected labels: {path}')
    center, radius, points = rows[:, 1:3], rows[:, 3:5] / 2, rows[:, 5:7]
    return np.concatenate([center - radius, center + radius, points], axis=1) * SIZE


def read_image(path):
    image = cv2.imread(str(path))
    if image is None or image.shape != (SIZE, SIZE, 3):
        raise ValueError(f'Expected 1024x1024 RGB image: {path}')
    return image


def estimates(key, refs):
    sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=.014)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    key_features, key_desc = sift.detectAndCompute(cv2.cvtColor(key, cv2.COLOR_BGR2GRAY), None)
    result = []
    for ref in refs:
        if np.array_equal(key, ref):
            result.append(np.eye(3, dtype=np.float32))
            continue
        features, desc = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
        if key_desc is None or desc is None or min(len(desc), len(key_desc)) < 2:
            result.append(None)
            continue
        matches = [near for near, far in matcher.knnMatch(desc, key_desc, k=2)
                   if near.distance < .75 * far.distance]
        if len(matches) < 12:
            result.append(None)
            continue
        ref_xy = np.float32([features[match.queryIdx].pt for match in matches])
        key_xy = np.float32([key_features[match.trainIdx].pt for match in matches])
        transform, inliers = cv2.findHomography(ref_xy, key_xy, cv2.RANSAC, 3.)
        if (transform is None or inliers.sum() < 12 or inliers.mean() < .35 or
                np.ptp(ref_xy[inliers.ravel().astype(bool)], axis=0).min() < 100):
            result.append(None)
            continue
        corners = np.float32([[[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]]])
        warped = cv2.perspectiveTransform(corners, transform)[0]
        area = cv2.contourArea(warped.astype(np.float32), oriented=True)
        result.append(transform.astype(np.float32) if .4 < area / SIZE**2 < 2.5 else None)
    return result


def crop_shift(key, ref):
    key = key[np.all((key > 12) & (key < SIZE - 12), axis=1)]
    ref = ref[np.all((ref > 12) & (ref < SIZE - 12), axis=1)]
    differences = (key[:, None] - ref).reshape(-1, 2)
    votes = Counter(map(tuple, np.rint(differences).astype(int)))
    if not votes or votes.most_common(1)[0][1] < 3:
        return None
    (delta, count), *rest = votes.most_common(2)
    if rest and rest[0][1] == count:
        return None
    return np.asarray(delta, dtype=np.float32)


def ref_versions(image, delta, axis):
    weak = cv2.GaussianBlur(image, (0, 0), 1.25)
    strong = cv2.GaussianBlur(image, (0, 0), 8.)
    weight = np.clip((HALF + 32 - (np.arange(SIZE, dtype=np.float32) + delta)) / 64, 0, 1)
    if axis == 1:
        weight = weight[:, None, None]
    else:
        weight = weight[None, :, None]
    return tuple(np.rint(quality * weak + (1 - quality) * strong).astype(np.uint8)
                 for quality in (weight, 1 - weight))


class Sequences(Dataset):
    def __init__(self, case, limit):
        self.case = case
        if case == 'controlled':
            self.folders = sorted((CLEAN / 'test').glob('seq_*'))
            self.provenance = json.loads((CLEAN / 'provenance.json').read_text())
        else:
            self.folders = sorted((REAL / case).glob('group_*'))
            self.provenance = {}
        if limit:
            self.folders = self.folders[:limit]
        if not self.folders:
            raise ValueError(case)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        folder = self.folders[index]
        if self.case != 'controlled':
            originals = [REAL / 'real_testset' / folder.name / f'{idx:03d}.png' for idx in (1, 2, 3)]
            images = [read_image(folder / path.name) for path in originals]
            gt = load_labels(folder / '001.txt')
            hashes = [sha256(path.read_bytes()).hexdigest() for path in originals]
            transforms = estimates(images[0], images[1:])
            return dict(name=folder.name, gt=gt, images=pack(images), hashes=hashes,
                        transforms=transforms)

        gt = load_labels(folder / 'labels_frame0.txt')
        shifts = [crop_shift(gt[:, 4:], load_labels(folder / f'labels_frame{idx}.txt')[:, 4:])
                  for idx in (1, 2)]
        if any(shift is None for shift in shifts):
            return dict(name=folder.name, skip='ambiguous crop translation')
        key, first, second = (read_image(folder / f'frame_{idx}.jpg') for idx in (0, 1, 2))
        axis = crc32(folder.name.encode()) % 2
        key = cv2.GaussianBlur(key, (0, 0), 5.5)
        first_a, first_b = ref_versions(first, shifts[0][axis], axis)
        second_a, second_b = ref_versions(second, shifts[1][axis], axis)
        points = gt[:, 4:]
        inside = np.all((points > 12) & (points < SIZE - 12), axis=1)
        for shift in shifts:
            native = points - shift
            inside &= np.all((native > 40) & (native < SIZE - 40), axis=1)
        region = np.full(len(gt), -1, dtype=np.int8)
        region[inside & (points[:, axis] < HALF - 48)] = 0
        region[inside & (points[:, axis] > HALF + 48)] = 1
        transforms = [np.float32([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
                      for shift in shifts]
        prov = self.provenance[folder.name]
        return dict(name=folder.name, gt=gt, axis=axis, region=region, transforms=transforms,
                    cluster=prov.get('group') or prov['source'], kind=prov['kind'],
                    images=pack([key, first_a, first_b, second_a, second_b]))


def pack(images):
    return torch.from_numpy(np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images])
                            .transpose(0, 3, 1, 2).copy())


def keep_list(batch):
    return batch


def warp(pred, transform):
    if transform is None or not len(pred):
        return np.empty((0, 7), dtype=np.float32)
    result = pred.copy()
    result[:, 5:] = cv2.perspectiveTransform(pred[:, 5:].copy()[None], transform)[0]
    inside = ((result[:, 5:] >= 0) & (result[:, 5:] < SIZE)).all(axis=1)
    return result[inside]


def late_fuse(preds):
    candidates = np.concatenate(preds)
    order = np.argsort(-candidates[:, 4], kind='stable')
    kept = []
    while len(order) and len(kept) < 300:
        winner, order = order[0], order[1:]
        kept.append(winner)
        delta = candidates[order, 5:] - candidates[winner, 5:]
        order = order[(delta ** 2).sum(axis=1) > 8**2]
    return candidates[kept]


def model_paths(seeds, fusion):
    suite = HERE.parent / 'concat_core_multiseed_20261111'
    if fusion == 'avg':
        suite = HERE.parent / 'fusion_avg_multiseed_20261115'
        mf = {str(seed): suite / f'seed{seed}/results/13_fusion_avg/pad_stage3_last.pt'
              for seed in seeds}
    else:
        mf = {str(seed): ((ABLATION if seed == 42 else suite / f'seed{seed}') /
                          'results/01_main/pad_stage3_last.pt') for seed in seeds}
    sf = Path(os.environ.get('LCMF_NATOCC_SF_CHECKPOINT',
                             HERE.parent / 'single_frame_baseline/results/pad_sf_natocc/p2l/last.pt'))
    return mf, sf


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['real_testset_blurred', 'real_testset', 'controlled'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--fusion', choices=['concat', 'avg'], default='concat')
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    ds = Sequences(args.case, args.limit)
    loader = iter(DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                             collate_fn=keep_list))
    arm = ARMS[0] | {'fusion': args.fusion}
    training = trainer()
    from pad_core import PADSingleFrame, decode_predictions

    mf_paths, sf_path = model_paths(args.seeds, args.fusion)
    model_args = args_for(architecture_flags(arm))
    models = {}
    for seed, path in mf_paths.items():
        assert_checkpoint(path, arm)
        model, _ = training.load_ckpt(str(path), model_args, 'cuda', prefer_ema=False)
        models[seed] = model.eval()
    sf_ck = torch.load(sf_path, map_location='cpu', weights_only=False)
    sf = PADSingleFrame(sf_ck['base_model'], sf_ck['head_cfg']).cuda().eval()
    sf_state = sf_ck['model']
    sf.load_state_dict(sf_state.state_dict() if isinstance(sf_state, torch.nn.Module) else sf_state,
                       strict=True)
    hashes = {seed: digest(path) for seed, path in mf_paths.items()} | {'sf': digest(sf_path)}
    controlled = args.case == 'controlled'
    pairs = ({'AA': (0, 1, 3), 'AB': (0, 1, 4), 'BA': (0, 2, 3), 'BB': (0, 2, 4),
              'KK': (0, 0, 0)} if controlled else
             {'R1R2': (0, 1, 2), 'R1R1': (0, 1, 1), 'R2R2': (0, 2, 2), 'KK': (0, 0, 0)})
    rows, skipped = [], []
    for batch_index, batch in enumerate(loader):
        skipped.extend(row for row in batch if 'skip' in row)
        batch = [row for row in batch if 'skip' not in row]
        if not batch:
            continue
        frames = torch.stack([row.pop('images') for row in batch]).cuda().float() / 255
        for row in batch:
            row['predictions'] = {}
        for name, model in models.items():
            for condition, slots in pairs.items():
                preds = decode_predictions(model(frames[:, slots]), SIZE)
                for row, pred in zip(batch, preds):
                    row['predictions'][f'mf{name}_{condition}'] = pred
        sf_preds = [decode_predictions(sf(frames[:, idx].contiguous()), SIZE)
                    for idx in range(frames.shape[1])]
        for idx, row in enumerate(batch):
            key_pred = sf_preds[0][idx]
            row['predictions']['sf_K'] = key_pred
            warped = {slot: warp(sf_preds[slot][idx], row['transforms'][ref_idx])
                      for ref_idx, slots in enumerate(((1, 2), (3, 4)) if controlled else ((1,), (2,)))
                      for slot in slots}
            for condition, (_, first, second) in pairs.items():
                if condition != 'KK' and (controlled or condition == 'R1R2'):
                    row['predictions'][f'sf_fuse_{condition}'] = late_fuse(
                        [key_pred, warped[first], warped[second]])
            if not controlled:
                row['predictions']['sf_R1'] = warped[1]
                row['predictions']['sf_R2'] = warped[2]
        rows.extend(batch)
        if batch_index % 8 == 0 or len(rows) + len(skipped) == len(ds):
            print(f'{args.case}: {len(rows)}/{len(ds)} evaluated, {len(skipped)} skipped', flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    with temp.open('wb') as handle:
        pickle.dump(dict(case=args.case, fusion=args.fusion, rows=rows, skipped=skipped, hashes=hashes), handle)
    temp.replace(args.output)


if __name__ == '__main__':
    main()
