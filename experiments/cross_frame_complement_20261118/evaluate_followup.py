"""Same-instance directional blur and appearance-balanced reference placebo."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import pickle
from zlib import crc32

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
import torch
from torch.utils.data import DataLoader, Dataset

from evaluate import (ARMS, CLEAN, HALF, HERE, REAL, SIZE,
                      architecture_flags, args_for, assert_checkpoint,
                      crop_shift, digest, keep_list, late_fuse, load_labels, model_paths, pack,
                      read_image, ref_versions, trainer, warp)
from recover_standard_test import TEST
from summarize import controlled_audit, public_session


DIRECTIONAL = {'XX': (0, 1, 3), 'XY': (0, 1, 4), 'YX': (0, 2, 3), 'YY': (0, 2, 4),
               'XK': (0, 1, 0), 'YK': (0, 2, 0), 'KX': (0, 0, 3), 'KY': (0, 0, 4),
               'KK': (0, 0, 0)}


def blur_axis(image, orientation):
    sigma_x, sigma_y = (9., 1.25) if orientation == 'X' else (1.25, 9.)
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)


def stripe_profile(coordinates, phase):
    return np.clip(.5 + np.sin(2 * np.pi * (coordinates - phase) / 48), 0, 1)


def stripe_versions(image, axis, shift, phase):
    key_coordinates = np.arange(SIZE, dtype=np.float32) + shift[axis]
    mask = stripe_profile(key_coordinates, phase)
    weight = mask[:, None, None] if axis else mask[None, :, None]
    clear = cv2.GaussianBlur(image, (0, 0), 1.25).astype(np.float32)
    blurred = cv2.GaussianBlur(image, (0, 0), 9.).astype(np.float32)
    return [np.rint(blurred + mixed * (clear - blurred)).astype(np.uint8)
            for mixed in (weight, 1 - weight)]


def stripes_eligibility(gt, axis, phase):
    coordinates = gt[:, 4 + axis]
    width = gt[:, 2 + axis] - gt[:, axis]
    sample = gt[:, axis, None] + width[:, None] * np.linspace(0, 1, 33)[None]
    fractions = stripe_profile(sample, phase).mean(axis=1)
    near_boundary = np.abs((coordinates - phase + 12) % 24 - 12) <= 5.
    return near_boundary & (width >= 24) & (fractions >= .25) & (fractions <= .75)


class ControlledViews(Dataset):
    def __init__(self, limit, case, official_clean=None, manifest=None):
        self.official_clean = official_clean
        if official_clean is None:
            _, allowed = controlled_audit()
            self.provenance = json.loads((CLEAN / 'provenance.json').read_text())
            self.folders = [CLEAN / 'test' / name for name in sorted(allowed)][:limit]
        else:
            self.provenance = json.loads(manifest.read_text())['sequences']
            self.folders = sorted(TEST.glob('seq_*'))[:limit]
            assert len(self.provenance) == 575
        self.case = case

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        folder = self.folders[index]
        gt = load_labels(folder / 'labels_frame0.txt')
        refs = [load_labels(folder / f'labels_frame{idx}.txt')[:, 4:] for idx in (1, 2)]
        provenance = self.provenance[folder.name]
        if self.official_clean is None:
            shifts = [crop_shift(gt[:, 4:], points) for points in refs]
            paths = [folder / f'frame_{idx}.jpg' for idx in range(3)]
            cluster = public_session(provenance['source'])
        else:
            origins = np.asarray(provenance['origins'], dtype=np.float32)
            shifts = origins[1:] - origins[0]
            paths = [self.official_clean / folder.name / f'frame_{idx}.png' for idx in range(3)]
            assert [digest(path) for path in paths] == provenance['clean_hashes']
            cluster = provenance['group']
        if any(shift is None for shift in shifts):
            return dict(name=folder.name, skip='ambiguous crop translation')
        images = [read_image(path) for path in paths]
        key = cv2.GaussianBlur(images[0], (0, 0), 5.5 if self.case == 'regional' else 7.)
        eligible = np.all((gt[:, 4:] > 40) & (gt[:, 4:] < SIZE - 40), axis=1)
        for shift, points in zip(shifts, refs, strict=True):
            native = gt[:, 4:] - shift
            eligible &= np.all((native > 40) & (native < SIZE - 40), axis=1)
            eligible &= np.any(np.linalg.norm(native[:, None] - points[None], axis=2) < .1, axis=1)
        if self.case == 'regional':
            axis = crc32(folder.name.encode()) % 2
            versions = [version for image, shift in zip(images[1:], shifts, strict=True)
                        for version in ref_versions(image, shift[axis], axis)]
            eligible &= np.abs(gt[:, 4 + axis] - HALF) > 48
        elif self.case == 'interleaved':
            geometry = crc32(folder.name.encode())
            axis, phase = geometry % 2, (geometry // 2) % 48
            versions = [version for image, shift in zip(images[1:], shifts, strict=True)
                        for version in stripe_versions(image, axis, shift, phase)]
            eligible &= stripes_eligibility(gt, axis, phase)
        else:
            versions = [blur_axis(image, axis) for image in images[1:] for axis in ('X', 'Y')]
        transforms = [np.float32([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
                      for shift in shifts]
        return dict(name=folder.name, gt=gt, eligible=eligible,
                    cluster=cluster, kind=provenance['kind'],
                    transforms=transforms, images=pack([key, *versions]))


def appearance(image):
    sample = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stat = sample.astype(float).reshape(-1, 3)
    return [np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()), *stat.mean(axis=0), *stat.std(axis=0)]


def assign_donors(patterns, descriptors, related, repeats=3):
    by_pattern = defaultdict(list)
    for idx, pattern in enumerate(patterns):
        by_pattern[pattern].append(idx)
    rng = np.random.default_rng(20261118)
    donors = np.full((repeats, len(patterns)), -1, dtype=int)
    for indices in by_pattern.values():
        feats = descriptors[indices]
        scaled = (feats - feats.mean(axis=0)) / np.maximum(feats.std(axis=0), 1e-5)
        distance = np.linalg.norm(scaled[:, None] - scaled[None], axis=2)
        prohibited = related[np.ix_(indices, indices)].copy()
        for repeat in range(repeats):
            cost = distance + rng.uniform(0, .05, size=distance.shape)
            cost[prohibited] = np.inf
            recipients, selected = linear_sum_assignment(cost)
            donors[repeat, np.array(indices)[recipients]] = np.array(indices)[selected]
            prohibited[recipients, selected] = True
    assert all(sorted(row) == list(range(len(patterns))) for row in donors)
    assert np.all(donors != np.arange(len(patterns)))
    return donors


def sift_features(detector, image):
    points, desc = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    return np.float32([point.pt for point in points]), desc


def geometrically_related(key, ref):
    key_xy, key_desc = key
    ref_xy, ref_desc = ref
    if key_desc is None or ref_desc is None or len(key_desc) < 2:
        return False
    matches = [near for near, far in cv2.BFMatcher(cv2.NORM_L2).knnMatch(ref_desc, key_desc, k=2)
               if near.distance < .75 * far.distance]
    if len(matches) < 12:
        return False
    source = ref_xy[[match.queryIdx for match in matches]]
    target = key_xy[[match.trainIdx for match in matches]]
    transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.)
    if transform is None or inliers.sum() < 12 or inliers.mean() < .35:
        return False
    if np.ptp(source[inliers.ravel().astype(bool)], axis=0).min() < 100:
        return False
    corners = np.float32([[[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]]])
    warped = cv2.perspectiveTransform(corners, transform)[0]
    return .4 < cv2.contourArea(warped, oriented=True) / SIZE**2 < 2.5


def screened_donors(names, patterns, descriptors, hashes, geometry):
    related = np.array([[bool(set(first) & set(second)) for second in hashes] for first in hashes])
    checked = {}
    while True:
        cluster = connected_components(related, directed=False, return_labels=True)[1]
        donors = assign_donors(patterns, descriptors, cluster[:, None] == cluster[None, :])
        failed = False
        for choices in donors:
            for recipient, donor in enumerate(choices):
                if (recipient, donor) not in checked:
                    checked[recipient, donor] = tuple(geometrically_related(geometry[recipient][0], ref)
                                                       for ref in geometry[donor][1:])
                if any(checked[recipient, donor]):
                    related[recipient, donor] = related[donor, recipient] = True
                    failed = True
        if not failed:
            break
    return donors, cluster, dict(
        candidate_pairs_checked=len(checked), final_pairs_screened=int(donors.size),
        plausible_matches=[dict(recipient=names[key], donor=names[donor], registrations=flags)
                           for (key, donor), flags in checked.items() if any(flags)],
        clusters={name: int(label) for name, label in zip(names, cluster, strict=True)})


class Placebo(Dataset):
    def __init__(self, limit):
        self.folders = sorted((REAL / 'real_testset_blurred').glob('group_*'))
        self.limit = limit
        self.frames = []
        self.labels = []
        self.hashes, self.patterns, features, geometry = [], [], [], []
        detector = cv2.SIFT_create(nfeatures=1200, contrastThreshold=.014)
        for folder in self.folders:
            paths = [folder / f'{idx:03d}.png' for idx in (1, 2, 3)]
            raw = [REAL / 'real_testset' / folder.name / path.name for path in paths]
            hashes = [sha256(path.read_bytes()).hexdigest() for path in raw]
            self.hashes.append(hashes)
            self.patterns.append(tuple(sha256(path.read_bytes()).hexdigest() != hash_value
                                       for path, hash_value in zip(paths[1:], hashes[1:], strict=True)))
            frames = [read_image(path) for path in paths]
            features.append([value for frame in frames[1:] for value in appearance(frame)])
            geometry.append([sift_features(detector, read_image(path)) for path in raw])
            self.frames.append(pack(frames))
            self.labels.append(load_labels(folder / '001.txt'))
        self.donors, self.clusters, self.geometry_audit = screened_donors(
            [folder.name for folder in self.folders], self.patterns, np.asarray(features), self.hashes, geometry)
        print(f'placebo: screened {self.geometry_audit["candidate_pairs_checked"]} donor pairs; '
              f'{len(self.geometry_audit["plausible_matches"])} overlapping candidates blocked', flush=True)

    def __len__(self):
        return min(self.limit, len(self.folders)) if self.limit else len(self.folders)

    def __getitem__(self, index):
        donor_indices = self.donors[:, index]
        images = torch.cat([self.frames[index], *[self.frames[donor][1:] for donor in donor_indices]])
        return dict(name=self.folders[index].name, images=images, gt=self.labels[index],
                    donor_names=[self.folders[donor].name for donor in donor_indices],
                    pattern=self.patterns[index], hashes=self.hashes[index], cluster=int(self.clusters[index]))


def placebo_modes():
    slots = {'MATCH': (0, 1, 2), 'KK': (0, 0, 0)}
    for repeat, first in enumerate((3, 5, 7), 1):
        slots[f'WRONG1_{repeat}'] = (0, first, 2)
        slots[f'WRONG2_{repeat}'] = (0, 1, first + 1)
        slots[f'WRONGBOTH_{repeat}'] = (0, first, first + 1)
    return slots


def infer_sf(sf, frames, batch, decode, plan):
    preds = [decode(sf(frames[:, slot].contiguous()), SIZE) for slot in range(5)]
    for idx, row in enumerate(batch):
        key = preds[0][idx]
        row['predictions']['sf_KK'] = key
        refs = [warp(preds[slot][idx], row['transforms'][(slot - 1) // 2]) for slot in range(1, 5)]
        for mode, (_, first, second) in plan.items():
            if mode != 'KK':
                row['predictions'][f'sf_{mode}'] = late_fuse([key, *[refs[slot-1] for slot in (first, second) if slot]])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=['regional', 'directional', 'interleaved', 'placebo'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--fusion', choices=['concat', 'avg'], default='concat')
    parser.add_argument('--official-clean', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    controlled = args.case != 'placebo'
    if bool(args.official_clean) != bool(args.manifest) or (args.official_clean and not controlled):
        parser.error('official-clean and manifest are required together, for controlled cases only')
    dataset = ControlledViews(args.limit, args.case, args.official_clean, args.manifest) if controlled else Placebo(args.limit)
    plan = ({name: slots for name, slots in DIRECTIONAL.items() if name in ('XX', 'XY', 'YX', 'YY', 'KK')}
            if args.case == 'regional' else DIRECTIONAL if controlled else placebo_modes())
    loader = iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                             collate_fn=keep_list))
    training = trainer()
    from pad_core import PADSingleFrame, decode_predictions

    mf_paths, sf_path = model_paths(args.seeds, args.fusion)
    arm = ARMS[0] | {'fusion': args.fusion}
    model_args = args_for(architecture_flags(arm))
    models = {}
    for seed, path in mf_paths.items():
        assert_checkpoint(path, arm)
        models[seed] = training.load_ckpt(str(path), model_args, 'cuda', prefer_ema=False)[0].eval()
    sf = None
    if controlled:
        sf_ck = torch.load(sf_path, map_location='cpu', weights_only=False)
        sf = PADSingleFrame(sf_ck['base_model'], sf_ck['head_cfg']).cuda().eval()
        state = sf_ck['model']
        sf.load_state_dict(state.state_dict() if isinstance(state, torch.nn.Module) else state, strict=True)
    hashes = {seed: digest(path) for seed, path in mf_paths.items()}
    if controlled:
        hashes['sf'] = digest(sf_path)
    rows, skipped = [], []
    for batch_index, batch in enumerate(loader):
        skipped.extend(row for row in batch if 'skip' in row)
        batch = [row for row in batch if 'skip' not in row]
        if not batch:
            continue
        frames = torch.stack([row.pop('images') for row in batch]).cuda().float() / 255
        for row in batch:
            row['predictions'] = {}
        for seed, model in models.items():
            for name, slots in plan.items():
                for row, pred in zip(batch, decode_predictions(model(frames[:, slots]), SIZE), strict=True):
                    row['predictions'][f'mf{seed}_{name}'] = pred
        if controlled:
            infer_sf(sf, frames, batch, decode_predictions, plan)
        rows.extend(batch)
        if batch_index % 8 == 0 or len(rows) + len(skipped) == len(dataset):
            print(f'{args.case}: {len(rows)}/{len(dataset)} evaluated, {len(skipped)} skipped', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    with temp.open('wb') as handle:
        audit = {} if controlled else {'geometry_audit': dataset.geometry_audit}
        pickle.dump(dict(case=args.case, rows=rows, skipped=skipped, hashes=hashes, fusion=args.fusion,
                         official_manifest_sha256=digest(args.manifest) if args.manifest else None, **audit), handle)
    temp.replace(args.output)


if __name__ == '__main__':
    main()
