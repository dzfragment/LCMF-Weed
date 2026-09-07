#!/usr/bin/env python3
"""
把 weed_sliding_merged 序列数据集转成"仅关键帧"标准 YOLO 单帧数据集。

多帧模型输入 3 帧 (frame_0/1/2)，此脚本只提取每个序列的 frame_0（关键帧）及其标签
labels_frame0.txt，输出为标准 YOLO 结构：
    weed_sliding_merged_keyframe/
        train/images/<seq>.jpg        （frame_0.jpg 拷贝）
        train/labels/<seq>.txt        （labels_frame0.txt 拷贝）
        val/...   test/... 同构

用于训练单帧关键帧基线（SF-P2L-P4 在相同数据分布上的单帧对照，与多帧模型对比）。

用法:
    python convert_keyframe_dataset.py            # 全量转换
    python convert_keyframe_dataset.py --smoke    # 每 split 只转前 3 个序列（验证用）
"""
import argparse
import os
import shutil
import sys

# ── 路径 ────────────────────────────────────────────────────────
# 源数据 weed_sliding_merged（含 train/val/test 序列目录）位于仓库外，
# 通过环境变量提供；关键帧输出默认写到源目录旁 <src>_keyframe。
try:
    SRC_ROOT = os.environ['DATASET_ROOT']
except KeyError:
    raise SystemExit('请设置环境变量 DATASET_ROOT（weed_sliding_merged 根目录）')
DST_ROOT = os.environ.get('KEYFRAME_ROOT') or (SRC_ROOT + '_keyframe')
SPLITS = ('train', 'val', 'test')

FRAME = 'frame_0.jpg'
LABEL = 'labels_frame0.txt'


def convert_split(split, limit=None):
    src_split = os.path.join(SRC_ROOT, split)
    dst_images = os.path.join(DST_ROOT, split, 'images')
    dst_labels = os.path.join(DST_ROOT, split, 'labels')
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    seq_dirs = sorted([
        d for d in os.listdir(src_split)
        if os.path.isdir(os.path.join(src_split, d))
    ])
    if limit:
        seq_dirs = seq_dirs[:limit]

    n_ok = n_skip = 0
    for seq in seq_dirs:
        seq_dir = os.path.join(src_split, seq)
        img_src = os.path.join(seq_dir, FRAME)
        lbl_src = os.path.join(seq_dir, LABEL)
        if not (os.path.isfile(img_src) and os.path.isfile(lbl_src)):
            print(f'  [skip] {seq}: 缺 {FRAME} 或 {LABEL}')
            n_skip += 1
            continue
        # 图片拷贝（避免软链被标准 Dataset 的 JPEG 修复逻辑原地改写污染源数据）
        shutil.copyfile(img_src, os.path.join(dst_images, f'{seq}.jpg'))
        shutil.copyfile(lbl_src, os.path.join(dst_labels, f'{seq}.txt'))
        n_ok += 1
    return n_ok, n_skip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='每 split 只转 3 个序列用于验证')
    args = parser.parse_args()
    limit = 3 if args.smoke else None

    print(f'源: {SRC_ROOT}')
    print(f'目标: {DST_ROOT}')
    print(f'模式: {"SMOKE(每split 3序列)" if limit else "全量"}')
    print()

    for split in SPLITS:
        n_ok, n_skip = convert_split(split, limit)
        print(f'[{split}] 转换 {n_ok} 序列, 跳过 {n_skip}')

    print(f'\n完成。输出目录: {DST_ROOT}')


if __name__ == '__main__':
    main()
