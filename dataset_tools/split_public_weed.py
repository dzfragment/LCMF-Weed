#!/usr/bin/env python3
"""
CropAndWeed 子集(旋转后)划分:按 session 分组划分 7 : 1.5 : 1.5

v2 修复: 旧版按图随机划分导致同一 recording session 的相邻帧散布在不同 split
(ave/vwg 前缀后第一段数字为 session, 同 session 为同一录像的连续帧, 内容近乎重复)。
现改为整 session 分配: 随机打乱 session 顺序后, 贪心放入当前图片缺口最大的 split,
在保持组内纯净的同时近似目标比例。

输入: $WEED_EXP_ROOT/public_weed_le256_rotated/{images,labels}
输出: $WEED_EXP_ROOT/public_weed_le256_split/{train,val,test}/{images,labels}
      + manifest.json (文件 → split 映射 + 统计)
"""

import os
import re
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


def session_of(name):
    """ave-0007-0007 → '0007'（recording session）; 无法解析时独立成组"""
    m = re.match(r'(ave|vwg)-(\d{4})-\d{4}$', name)
    return m.group(2) if m else f'self_{name}'


def main():
    print("=" * 60)
    print("CropAndWeed 子集划分 (按 session 分组, 7:1.5:1.5)")
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

    n_train = int(round(n_total * RATIOS['train']))
    n_val = int(round(n_total * RATIOS['val']))
    n_test = n_total - n_train - n_val
    print(f"目标: train {n_train} / val {n_val} / test {n_test}")

    # 按 session 分组
    sessions = {}
    for f in files:
        sessions.setdefault(session_of(os.path.splitext(f)[0]), []).append(f)
    print(f"共 {len(sessions)} 个 session")

    # 随机打乱 session 顺序, 贪心放入图片缺口比例最大的 split
    rng = np.random.RandomState(SEED)
    order = list(sessions)
    rng.shuffle(order)
    target = {'train': n_train, 'val': n_val, 'test': n_test}
    counts = {k: 0 for k in target}
    split_of = {}
    for key in order:
        n = len(sessions[key])
        best = max(target, key=lambda s: (target[s] - counts[s]) / target[s])
        for f in sessions[key]:
            split_of[f] = best
        counts[best] += n
    print(f"实际: " + " / ".join(f"{s} {counts[s]}" for s in ['train', 'val', 'test']))

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
        'n_sessions': len(sessions),
        'split_of': {f: s for f, s in sorted(split_of.items())},
        'stats': stats,
        'note': '按 session 分组划分(同 session 录像帧整体进同一 split); '
                'v2 修复旧版按图随机划分的相邻帧泄露',
    }
    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nmanifest: {os.path.join(OUT_DIR, 'manifest.json')}")
    print("完成!")


if __name__ == "__main__":
    main()
