#!/usr/bin/env python3
"""Build natural-texture occluder patch bank from exp4 training images only."""
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

DATASET = Path(os.environ.get('LCMF_DATA_ROOT', Path(__file__).resolve().parents[1] / 'dataset'))
DATASET_ENV = str(DATASET)
os.environ.setdefault('WEED_SRC_ROOT', DATASET_ENV)
os.environ.setdefault('WEED_EXP_ROOT', DATASET_ENV)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_sliding as s2  # noqa: E402

OUT = DATASET / 'patch_bank_train.npy'

IMG_DIRS = [DATASET / 'weedataset' / 'train' / 'images',
            DATASET / 'public_weed_le256_split_1021' / 'train' / 'images']
LBL_DIRS = [DATASET / 'weedataset' / 'train' / 'labels',
            DATASET / 'public_weed_le256_split_1021' / 'train' / 'labels']
PATCH = 128
N_BG = 2000
N_LEAF = 1000


def private_train_names():
    all_images = s2.collect_images_from_dirs([
        (s2.SRC_TRAIN_IMAGES, s2.SRC_TRAIN_LABELS),
        (s2.SRC_VAL_IMAGES, s2.SRC_VAL_LABELS),
        (s2.SRC_TEST_IMAGES, s2.SRC_TEST_LABELS),
    ])
    groups = {}
    for b in sorted(all_images):
        groups.setdefault(s2.get_group_key(b), []).append(b)
    gk = sorted(groups)
    import numpy as np
    np.random.seed(42)
    video = [g for g in gk if g.startswith('video_')]
    sess = sorted([g for g in gk if g.startswith('session_')],
                  key=lambda x: len(groups[x]), reverse=True)
    single = [g for g in gk if g.startswith('single_')]
    train = set(video) | {sess[0], sess[1], sess[5]}
    perm = np.random.permutation(len(single))
    train |= {single[i] for i in perm[:5]}
    return {b for g in train for b in groups[g]}


def green_frac(p):
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return float(((h > 30) & (h < 90) & (s > 40) & (v > 40)).mean())


def main():
    private_train = private_train_names()
    rng = random.Random(20261101)
    np_rng = np.random.RandomState(20261101)
    bg, leaf = [], []
    for di, (img_dir, lbl_dir) in enumerate(zip(IMG_DIRS, LBL_DIRS)):
        for f in sorted(os.listdir(img_dir)):
            if not f.endswith(('.jpg', '.jpeg', '.png')):
                continue
            if di == 0 and os.path.splitext(f)[0] not in private_train:
                continue
            img = cv2.imread(str(img_dir / f))
            if img is None:
                continue
            h, w = img.shape[:2]
            if h < PATCH or w < PATCH:
                continue
            boxes = []
            lp = lbl_dir / (os.path.splitext(f)[0] + '.txt')
            if lp.exists():
                for line in open(lp):
                    q = line.split()
                    if len(q) >= 7:
                        cx, cy, bw, bh = map(float, q[1:5])
                        kx, ky = float(q[5]), float(q[6])
                        boxes.append((cx * w, cy * h, bw * w, bh * h, kx * w, ky * h))
            # background texture
            for _ in range(8):
                if len(bg) >= N_BG:
                    break
                x = rng.randint(0, w - PATCH)
                y = rng.randint(0, h - PATCH)
                cxp, cyp = x + PATCH / 2, y + PATCH / 2
                if any(abs(cxp - bx) < bw / 2 + 6 and abs(cyp - by) < bh / 2 + 6
                       for bx, by, bw, bh, _, _ in boxes):
                    continue
                crop = img[y:y + PATCH, x:x + PATCH]
                if crop.std() < 18:
                    continue
                bg.append(crop)
            # leaf texture from inside target boxes
            for bx, by, bw, bh, kx, ky in boxes:
                if len(leaf) >= N_LEAF:
                    break
                for _ in range(2):
                    size = rng.choice([48, 64, 80])
                    px = rng.uniform(bx - bw / 2, bx + bw / 2)
                    py = rng.uniform(by - bh / 2, by + bh / 2)
                    if np.hypot(px - kx, py - ky) < 12:
                        continue
                    x, y = int(px - size / 2), int(py - size / 2)
                    if x < 0 or y < 0 or x + size > w or y + size > h:
                        continue
                    crop = img[y:y + size, x:x + size]
                    if crop.shape[:2] != (size, size):
                        continue
                    if green_frac(crop) < 0.12:
                        continue
                    leaf.append(cv2.resize(crop, (PATCH, PATCH)))
            if len(bg) >= N_BG and len(leaf) >= N_LEAF:
                break
        if len(bg) >= N_BG and len(leaf) >= N_LEAF:
            break
    print('bg', len(bg), 'leaf', len(leaf))
    if len(bg) < 100 or len(leaf) < 100:
        raise RuntimeError('not enough patches')
    bank = np.concatenate([np.stack(bg, 0), np.stack(leaf, 0)], 0)
    np.save(OUT, bank)
    print('saved', OUT, bank.shape)


if __name__ == '__main__':
    main()
