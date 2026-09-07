#!/usr/bin/env python3
"""
CropAndWeed 子集(旋转后)划分:按图随机划分 7 : 1.5 : 1.5

输入: $WEED_EXP_ROOT/public_weed_le256_rotated/{images,labels}
输出: $WEED_EXP_ROOT/public_weed_le256_split/{train,val,test}/{images,labels}
      + manifest.json (文件 → split 映射 + 统计)
"""

import os
import sys
import json
import shutil
import numpy as np

_DATA_ROOT = os.environ.get('WEED_EXP_ROOT')
if not _DATA_ROOT:
    raise SystemExit('[dataset_tools] 请设置环境变量 WEED_EXP_ROOT：实验数据集根目录（旧 exp2：'
                     'weed1024 / public_weed_le256_* / weed_public_* / weed_sliding_merged 等所在）')

SRC_DIR = os.path.join(_DATA_ROOT, 'public_weed_le256_rotated')
OUT_DIR = os.path.join(_DATA_ROOT, 'public_weed_le256_split')

RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SEED = 20260810


def main():
    print("=" * 60)
    print("CropAndWeed 子集划分 (按图, 7:1.5:1.5)")
    print(f"  输入: {SRC_DIR}")
    print(f"  输出: {OUT_DIR}")
    print(f"  种子: {SEED}")
    print("=" * 60)

    img_dir = os.path.join(SRC_DIR, 'images')
    lbl_dir = os.path.join(SRC_DIR, 'labels')

    files = sorted(f for f in os.listdir(img_dir)
                   if f.endswith(('.jpg', '.png', '.JPG', '.PNG')))
    n_total = len(files)
    print(f"共 {n_total} 张图片")

    # 目标数量: 7/1.5/1.5 精确分配
    n_train = int(round(n_total * RATIOS['train']))
    n_val = int(round(n_total * RATIOS['val']))
    n_test = n_total - n_train - n_val
    print(f"目标: train {n_train} / val {n_val} / test {n_test}")

    # 按图随机打乱
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_total)
    split_of = {}
    for i, idx in enumerate(perm):
        if i < n_train:
            split_of[files[idx]] = 'train'
        elif i < n_train + n_val:
            split_of[files[idx]] = 'val'
        else:
            split_of[files[idx]] = 'test'

    # 复制
    stats = {s: {'images': 0, 'labels': 0, 'ave': 0, 'vwg': 0, 'other': 0}
             for s in RATIOS}
    for f, split in split_of.items():
        name = os.path.splitext(f)[0]
        dst_img = os.path.join(OUT_DIR, split, 'images')
        dst_lbl = os.path.join(OUT_DIR, split, 'labels')
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)
        shutil.copy2(os.path.join(img_dir, f), os.path.join(dst_img, f))
        shutil.copy2(os.path.join(lbl_dir, name + '.txt'),
                     os.path.join(dst_lbl, name + '.txt'))
        stats[split]['images'] += 1
        with open(os.path.join(lbl_dir, name + '.txt')) as lf:
            stats[split]['labels'] += sum(1 for _ in lf)
        if f.startswith('ave-'):
            stats[split]['ave'] += 1
        elif f.startswith('vwg-'):
            stats[split]['vwg'] += 1
        else:
            stats[split]['other'] += 1

    # 汇总
    print(f"\n{'split':<8}{'图片':>6}{'标注':>8}{'ave':>6}{'vwg':>6}  占比")
    print("-" * 46)
    for s in ['train', 'val', 'test']:
        st = stats[s]
        print(f"{s:<8}{st['images']:>6}{st['labels']:>8}{st['ave']:>6}"
              f"{st['vwg']:>6}  {st['images']/n_total*100:.1f}%")

    # manifest
    manifest = {
        'seed': SEED,
        'n_total': n_total,
        'ratio': RATIOS,
        'split_of': {f: s for f, s in sorted(split_of.items())},
        'stats': stats,
        'note': '按图随机划分(未按 session 分组);相邻帧可能存在内容重叠',
    }
    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nmanifest: {os.path.join(OUT_DIR, 'manifest.json')}")
    print("完成!")


if __name__ == "__main__":
    main()
