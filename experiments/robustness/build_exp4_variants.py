#!/usr/bin/env python3
"""在 clean test 池上生成受控遮挡/模糊变体.

设计:
  - clean/test 的 575 条序列与标准 test 的几何/标签逐一对应 (build_clean_test.py 已校验).
  - 每个变体只改目标帧; 未改帧和所有标签用 symlink 指向 clean, 省空间.
  - 遮挡使用真实植物/土壤纹理 patch (主实验) + 纯绿方块对照.
  - 遮挡选择用同一个 per-seq 置换, p 增大时是嵌套的 (nested), 曲线可比.
  - 记录每个 GT 的关键点是否被遮挡, 以及每帧 Laplacian 方差.

生成:
  exp1_occ_key: p = 0, .05, .10, .15, .20, .30, .40, .50, .65, .80, .90
  exp1_blur_key: sigma = 0, .5, 1, 1.5, 2, 3, 4, 5, 6, 8
  exp2: occ_refs_p050, occ_all_p050, blur_refs_s4, blur_all_s4
  control: occ_green_key_p050
"""
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

DATASET = Path(os.environ.get('LCMF_DATA_ROOT', Path(__file__).resolve().parents[2] / 'dataset')).resolve()
CLEAN = DATASET / 'robustness' / 'clean' / 'test'
OUT_ROOT = DATASET / 'robustness' / 'variants'
BANK_PATH = DATASET / 'patch_bank_train.npy'

OCC_P = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.90]
BLUR_SIGMA = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
AREA_RATIO = 0.36
BASE_SEED = 20261101

SPLITS_TRAIN_IMGS = [
    DATASET / 'public_weed_le256_split_1021' / 'train' / 'images',
    DATASET / 'weedataset' / 'train' / 'images',
]
SPLITS_TRAIN_LBLS = [
    DATASET / 'public_weed_le256_split_1021' / 'train' / 'labels',
    DATASET / 'weedataset' / 'train' / 'labels',
]


# ───────────────────────────── patch bank ─────────────────────────────

def build_patch_bank(n_patches=3000, patch=128, seed=20261101):
    if BANK_PATH.exists():
        bank = np.load(BANK_PATH)
        print(f'[patch_bank] reuse {bank.shape}')
        return bank
    rng = np.random.RandomState(seed)
    patches = []
    sources = []
    for img_dir, lbl_dir in zip(SPLITS_TRAIN_IMGS, SPLITS_TRAIN_LBLS):
        files = sorted(f for f in os.listdir(img_dir)
                       if f.endswith(('.jpg', '.jpeg', '.png')))
        for f in files:
            img = cv2.imread(str(img_dir / f))
            if img is None:
                continue
            h, w = img.shape[:2]
            if h < patch or w < patch:
                continue
            boxes = []
            lp = lbl_dir / (os.path.splitext(f)[0] + '.txt')
            if lp.exists():
                for line in open(lp):
                    p = line.split()
                    if len(p) >= 5:
                        cx, cy, bw, bh = map(float, p[1:5])
                        boxes.append((cx * w, cy * h, bw * w, bh * h))
            for _ in range(12):
                x = rng.randint(0, w - patch)
                y = rng.randint(0, h - patch)
                cx, cy = x + patch / 2, y + patch / 2
                if any(abs(cx - bx) < (bw / 2 + 8) and abs(cy - by) < (bh / 2 + 8)
                       for bx, by, bw, bh in boxes):
                    continue
                crop = img[y:y + patch, x:x + patch]
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if g.var() < 60:
                    continue
                patches.append(crop)
                sources.append(f)
                if len(patches) >= n_patches:
                    break
            if len(patches) >= n_patches:
                break
        if len(patches) >= n_patches:
            break
    if len(patches) < n_patches:
        print(f'[patch_bank] 只采到 {len(patches)} 块')
    bank = np.stack(patches, 0)
    np.save(BANK_PATH, bank)
    print(f'[patch_bank] built {bank.shape} -> {BANK_PATH}')
    return bank


# ───────────────────────────── degradation ─────────────────────────────

def gauss_blur(img, sigma):
    if sigma <= 0:
        return img
    k = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def _patch_paste(img, patch, cx, cy, size, rng):
    h, w = img.shape[:2]
    size = int(max(8, min(size, 160)))
    if size >= w or size >= h:
        return img, None
    p = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), (size // 2, size // 2),
                0, 0, 360, 255, -1)
    feather = max(1, size // 12)
    mask = cv2.GaussianBlur(mask, (2 * feather + 1,) * 2, feather / 2)
    x = int(round(cx - size / 2))
    y = int(round(cy - size / 2))
    x = max(0, min(w - size, x))
    y = max(0, min(h - size, y))
    roi = img[y:y + size, x:x + size].astype(np.float32)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    img[y:y + size, x:x + size] = np.clip(
        roi * (1 - alpha) + p.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return img, (x, y, size, cx, cy)


def apply_occlusion_frame(img, boxes, kpts, p_box, rng, bank, green=False,
                          area_ratio=AREA_RATIO):
    """在 frame 上遮挡 p_box 比例的 GT 框.

    Returns: img, covered_flags (len=len(boxes)), occ_info
    """
    n = len(boxes)
    covered = [False] * n
    if n == 0 or p_box <= 0:
        return img, covered, []
    order = list(range(n))
    rng.shuffle(order)
    n_sel = int(round(p_box * n))
    info = []
    h, w = img.shape[:2]
    for rank, bi in enumerate(order):
        x1, y1, x2, y2 = boxes[bi]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        size = int(round(np.sqrt(area_ratio * bw * bh)))
        size = max(8, min(size, int(0.9 * min(bw, bh)), 120))
        # 保证参数与 p 无关: 对每个 box 都抽样, 只有 rank < n_sel 才应用
        padx = max(0.0, bw / 2 - size / 2)
        pady = max(0.0, bh / 2 - size / 2)
        cx = (x1 + x2) / 2 + rng.uniform(-padx, padx) if padx > 0 else (x1 + x2) / 2
        cy = (y1 + y2) / 2 + rng.uniform(-pady, pady) if pady > 0 else (y1 + y2) / 2
        if green:
            patch = np.zeros((size, size, 3), np.uint8)
            patch[:, :] = (0, rng.randint(60, 180), 0)
        else:
            patch = bank[rng.randrange(len(bank))]
        if rank < n_sel:
            img, placed = _patch_paste(img, patch, cx, cy, size, rng)
            if placed is not None:
                kx, ky = kpts[bi][:2]
                d = np.hypot(kx - cx, ky - cy)
                covered[bi] = bool(d <= size * 0.45)
                info.append({'box': bi, 'size': size, 'cx': cx, 'cy': cy,
                             'covered': covered[bi]})
    return img, covered, info


def load_boxes_kpts(path):
    boxes, kpts = [], []
    if not os.path.exists(path):
        return boxes, kpts
    for line in open(path):
        p = line.split()
        if len(p) < 7:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        kx, ky = float(p[5]), float(p[6])
        # 以 1024 像素坐标返回
        boxes.append(((cx - bw / 2) * 1024, (cy - bh / 2) * 1024,
                      (cx + bw / 2) * 1024, (cy + bh / 2) * 1024))
        kpts.append((kx * 1024, ky * 1024))
    return boxes, kpts


def sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


# ───────────────────────────── variant builder ─────────────────────────────

def make_variant(name, factor, mode, value, bank, green=False, seed=BASE_SEED):
    out = OUT_ROOT / name / 'test'
    if (OUT_ROOT / name / 'meta.json').exists() and out.exists():
        print(f'[skip] {name}')
        return
    if out.exists():
        shutil.rmtree(out)
    os.makedirs(out)
    meta = {'name': name, 'factor': factor, 'mode': mode,
            'value': value, 'area_ratio': AREA_RATIO,
            'occluder': 'green' if green else ('natural' if factor == 'occlusion' else 'gaussian'),
            'seed': seed, 'coverage': {}, 'sharpness': {}}

    seqs = sorted(d for d in os.listdir(CLEAN)
                  if d.startswith('seq_') and os.path.isdir(CLEAN / d))
    for seq in seqs:
        src = CLEAN / seq
        dst = out / seq
        os.makedirs(dst)
        # 标签全部 symlink
        for i in range(3):
            os.symlink(src / f'labels_frame{i}.txt', dst / f'labels_frame{i}.txt')
        cov_seq = {}
        sharp_seq = {}
        for fi in range(3):
            do = (mode == 'all') or (mode == 'key' and fi == 0) or \
                 (mode == 'refs' and fi > 0)
            # level-0 必须是恒等: 不重编码, 直接 symlink
            if float(value) == 0.0:
                do = False
            src_img = src / f'frame_{fi}.jpg'
            dst_img = dst / f'frame_{fi}.jpg'
            img = cv2.imread(str(src_img))
            if img is None:
                raise RuntimeError(f'cannot read {src_img}')
            sharp_before = sharpness(img)
            if not do:
                os.symlink(src_img, dst_img)
                sharp_seq[f'frame{fi}'] = sharp_before
                continue
            if factor == 'blur':
                img2 = gauss_blur(img, float(value))
                sharp_seq[f'frame{fi}'] = sharpness(img2)
                cv2.imwrite(str(dst_img), img2, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                boxes, kpts = load_boxes_kpts(str(src / f'labels_frame{fi}.txt'))
                # per-frame 独立 rng; seq + frame 固定
                rng = random.Random(f'{seq}|occ|{seed}|{fi}|{mode}')
                img2, covered, info = apply_occlusion_frame(
                    img, boxes, kpts, float(value), rng, bank, green=green)
                cov_seq[f'frame{fi}'] = covered
                sharp_seq[f'frame{fi}'] = sharpness(img2)
                cv2.imwrite(str(dst_img), img2, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if factor == 'occlusion':
            meta['coverage'][seq] = cov_seq
        meta['sharpness'][seq] = sharp_seq
    with open(OUT_ROOT / name / 'meta.json', 'w') as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f'[done] {name}: {len(seqs)} seq')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None,
                    help='逗号分隔的 variant 名; 默认全部')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    bank = build_patch_bank()
    os.makedirs(OUT_ROOT, exist_ok=True)

    jobs = []
    # Experiment 1: keyframe-only
    for p in OCC_P:
        tag = f'p{int(round(p * 100)):03d}'
        jobs.append((f'occ_key_{tag}', 'occlusion', 'key', p, False))
    for s in BLUR_SIGMA:
        tag = f's{str(s).replace(".", "p")}'
        jobs.append((f'blur_key_{tag}', 'blur', 'key', s, False))

    # Experiment 2: key / refs / all at fixed severity
    jobs += [
        ('occ_refs_p050', 'occlusion', 'refs', 0.50, False),
        ('occ_all_p050', 'occlusion', 'all', 0.50, False),
        ('blur_refs_s4p0', 'blur', 'refs', 4.0, False),
        ('blur_all_s4p0', 'blur', 'all', 4.0, False),
        ('occ_green_key_p050', 'occlusion', 'key', 0.50, True),
    ]
    if args.only:
        keep = {x.strip() for x in args.only.split(',') if x.strip()}
        jobs = [j for j in jobs if j[0] in keep]
    for name, factor, mode, value, green in jobs:
        if args.force:
            for p in (OUT_ROOT / name,):
                if p.exists():
                    shutil.rmtree(p)
        make_variant(name, factor, mode, value, bank, green=green)

    print('\nALL VARIANTS DONE ->', OUT_ROOT)


if __name__ == '__main__':
    main()
