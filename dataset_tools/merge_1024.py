#!/usr/bin/env python3
"""
合并两个 1024×1024 单帧数据集(模型基座训练用):
  旧: $WEED_EXP_ROOT/weed1024                       (train 644 / val 136 / test 140)
  新: $WEED_EXP_ROOT/public_weed_le256_tiles        (train 1632 / val 350 / test 348)
  → $WEED_EXP_ROOT/weed_merged_1024                 (train 2276 / val 486 / test 488)

标注格式已确认一致: YOLO-pose 7列 (cls cx cy w h kx ky), 类别0
"""

import os
import shutil
import cv2
import numpy as np

_DATA_ROOT = os.environ.get('WEED_EXP_ROOT')
if not _DATA_ROOT:
    raise SystemExit('[dataset_tools] 请设置环境变量 WEED_EXP_ROOT：实验数据集根目录（旧 exp2：'
                     'weed1024 / public_weed_le256_* / weed_public_* / weed_sliding_merged 等所在）')

OLD = os.path.join(_DATA_ROOT, 'weed1024')
NEW = os.path.join(_DATA_ROOT, 'public_weed_le256_tiles')
OUT = os.path.join(_DATA_ROOT, 'weed_merged_1024')

SPLITS = ['train', 'val', 'test']


def copy_split(src, split):
    """复制一个 split 的 images/labels, 返回 (图数, 标注数)"""
    n_img = n_lb = 0
    for sub in ['images', 'labels']:
        sdir = os.path.join(src, split, sub)
        ddir = os.path.join(OUT, split, sub)
        os.makedirs(ddir, exist_ok=True)
        for f in sorted(os.listdir(sdir)):
            if sub == 'images' and not f.endswith(('.jpg', '.png', '.JPG', '.PNG')):
                continue
            if sub == 'labels' and not f.endswith('.txt'):
                continue
            shutil.copy2(os.path.join(sdir, f), os.path.join(ddir, f))
            if sub == 'images':
                n_img += 1
            else:
                with open(os.path.join(sdir, f)) as lf:
                    n_lb += sum(1 for _ in lf)
    return n_img, n_lb


def main():
    print("=" * 60)
    print("合并 1024×1024 单帧数据集 (模型基座)")
    print(f"  旧: {OLD}")
    print(f"  新: {NEW}")
    print(f"  输出: {OUT}")
    print("=" * 60)

    # 命名冲突检查
    old_names = set()
    for split in SPLITS:
        idir = os.path.join(OLD, split, 'images')
        if os.path.exists(idir):
            old_names |= {f for f in os.listdir(idir)
                          if f.endswith(('.jpg', '.png', '.JPG', '.PNG'))}
    new_names = set()
    for split in SPLITS:
        idir = os.path.join(NEW, split, 'images')
        if os.path.exists(idir):
            new_names |= {f for f in os.listdir(idir)
                          if f.endswith(('.jpg', '.png', '.JPG', '.PNG'))}
    dup = old_names & new_names
    print(f"  旧数据 {len(old_names)} 张, 新数据 {len(new_names)} 张, 重名 {len(dup)} 张")
    if dup:
        print(f"  [FATAL] 存在重名: {sorted(dup)[:5]}")
        return
    print("  ✓ 无命名冲突")

    # 复制
    totals = {}
    grand = {'images': 0, 'labels': 0}
    for split in SPLITS:
        oi, ol = copy_split(OLD, split)
        ni, nl = copy_split(NEW, split)
        totals[split] = (oi + ni, ol + nl)
        grand['images'] += oi + ni
        grand['labels'] += ol + nl
        print(f"  [{split}] 旧 {oi}图/{ol}标注 + 新 {ni}图/{nl}标注 "
              f"= {oi+ni}图/{ol+nl}标注")

    # YAML
    yaml_path = os.path.join(OUT, "weed_merged.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"""# 合并 1024×1024 单帧数据集 (weed1024 + public_weed_le256_tiles)
train: ./train/images
val: ./val/images
test: ./test/images

nc: 1
kpt_shape: [1, 2]
flip_idx: [0]
names: ['weed']
""")
    print(f"\n  YAML: {yaml_path}")

    # 尺寸抽查
    bad = 0
    rng = np.random.RandomState(0)
    for split in SPLITS:
        idir = os.path.join(OUT, split, 'images')
        files = sorted(os.listdir(idir))
        picked = rng.choice(files, min(20, len(files)), replace=False)
        for f in picked:
            im = cv2.imread(os.path.join(idir, f))
            if im is None or im.shape[0] != 1024 or im.shape[1] != 1024:
                bad += 1
                print(f"  [WARN] {split}/{f} 尺寸异常")
    print(f"  尺寸抽查: {bad} 张异常 (每 split 抽 20 张)")

    print(f"\n  合计: {grand['images']} 图 / {grand['labels']} 标注")
    print("完成!")


if __name__ == "__main__":
    main()
