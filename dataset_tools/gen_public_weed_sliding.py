#!/usr/bin/env python3
"""
CropAndWeed 子集滑动窗口多帧生成(单独规则)

规则(与老数据集 weedataset_sliding_v6 的区别):
  1. 输入: 旋转后整图 1088×1920(不拆分、不用镜像 padding)
  2. 窗口: 1024×1024,三帧位置全部限制在原图内
     - 垂直(长边1920,可滑0~896): 帧间 |dy|∈[110,150],
       一个序列内三帧方向严格一致(全向下或全向上),序列之间方向随机
     - 水平(短边1088,可滑0~64): 帧间 |dx|∈[3,16],方向独立随机
  3. 标注: 每帧独立映射,相交面积≥30%保留,关键点夹边
  4. 噪声: 机制与老流程一致(振动失焦/过曝/随机遮挡/标注遮挡/背景遮挡),
     概率参数复用 config(可单独调)
  5. 序列编号: 从老数据集数量+1 开始,保证与 weedataset_sliding_v6 合并零冲突
     (老: train 1027 / val 68 / test 83 → 新: 1028 / 69 / 84 起)

输入: $WEED_EXP_ROOT/public_weed_le256_split/{train,val,test}/{images,labels}
输出: $WEED_EXP_ROOT/weed_public_sliding/{train,val,test}/seq_XXXXX/
"""

import os
import sys
import cv2
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RESIZE_TARGET, TILE_SIZE,
    MIN_AREA_RATIO, MIN_LABELS_FOR_KEYFRAME,
    BLUR_PROB_KEYFRAME, BLUR_PROB_REFERENCE,
    MAX_ATTEMPTS_PER_IMAGE,
)
from common import (load_labels, save_labels, apply_noises,
                    generate_overexposure_params)

import common
# 新数据集单独规则: Frame 0 随机遮挡概率 (覆盖 common 模块级常量,
# apply_noises 内部读取时生效; 不影响老流程)
#   train/val: 45% (老数据集为 30%)
#   test:      30% (回到默认; 但 frame0 标注区域遮挡覆盖率 75%, 严苛遮挡)
RANDOM_OCC_PROB_KEYFRAME_BY_SPLIT = {
    'train': 0.45,
    'val': 0.45,
    'test': 0.30,
}
# test 的 frame0 标注区域遮挡块覆盖率(占标注框面积); None=模块默认 36%
F0_LABEL_OCC_AREA_RATIO_BY_SPLIT = {
    'train': None,
    'val': None,
    'test': 0.60,
}
# test frame0 遮挡块上限(放开全局 40px → 60px)
F0_LABEL_OCC_MAX_SIZE_BY_SPLIT = {
    'train': None,
    'val': None,
    'test': 60,
}
# test frame0 块边长占框短边系数 (1.0=可占满短边, 配合 75% 覆盖率)
F0_LABEL_OCC_SHORT_FACTOR_BY_SPLIT = {
    'train': None,
    'val': None,
    'test': 1.0,
}

_DATA_ROOT = os.environ.get('WEED_EXP_ROOT')
if not _DATA_ROOT:
    raise SystemExit('[dataset_tools] 请设置环境变量 WEED_EXP_ROOT：实验数据集根目录（旧 exp2：'
                     'weed1024 / public_weed_le256_* / weed_public_* / weed_sliding_merged 等所在）')

SRC_ROOT = os.path.join(_DATA_ROOT, 'public_weed_le256_split')
OUT_ROOT = os.path.join(_DATA_ROOT, 'weed_public_sliding')

# ═══ 新数据集滑动规则 ═══
FRAME_SIZE = 1024
SRC_W, SRC_H = 1088, 1920          # 旋转后尺寸
MAX_ORIGIN_X = SRC_W - FRAME_SIZE  # 64 (短边可滑空间)
MAX_ORIGIN_Y = SRC_H - FRAME_SIZE  # 896 (长边可滑空间)

DY_MIN, DY_MAX = 110, 150          # 垂直帧间位移幅度(双向,同序列内同向)
DX_MIN, DX_MAX = 3, 16             # 水平帧间位移幅度(短边受限)

SAMPLES_PER_IMAGE = {'train': 4, 'val': 2, 'test': 4}   # 每图最多序列数
SEQ_START = {'train': 1028, 'val': 84, 'test': 84}      # 老数据集数量+1
SEED_BASE = 20260810

VIZ_COUNT = 4  # 可视化抽样序列数


# ══════════════════════════════════════════════════════════════════════
# 位置采样:三帧窗口坐标,保证全部在原图内
# ══════════════════════════════════════════════════════════════════════

def sample_sequence_positions(rng, dir_y):
    """采样一个序列的三帧窗口左上角坐标。

    Args:
        rng: 随机源
        dir_y: +1=从上往下滑, -1=从下往上滑

    Returns:
        [(x0,y0),(x1,y1),(x2,y2)] 或 None(无法满足)
    """
    # ── 垂直: 同方向, 幅度独立 ──
    dy1 = dir_y * rng.randint(DY_MIN, DY_MAX + 1)
    dy2 = dir_y * rng.randint(DY_MIN, DY_MAX + 1)
    # Frame 0 的 y 可行区间: 保证 y1=y0+dy1, y2=y0+dy1+dy2 都在 [0, MAX_ORIGIN_Y]
    y0_min = max(0, -dy1, -(dy1 + dy2))
    y0_max = min(MAX_ORIGIN_Y, MAX_ORIGIN_Y - dy1,
                 MAX_ORIGIN_Y - (dy1 + dy2))
    if y0_max < y0_min:
        return None
    y0 = rng.randint(y0_min, y0_max)
    y1, y2 = y0 + dy1, y0 + dy1 + dy2

    # ── 水平: 方向独立, 幅度受限 ──
    dx1 = _sample_dx(rng)
    dx2 = _sample_dx(rng)
    # 同样约束三帧都在 [0, MAX_ORIGIN_X]
    x0_min = max(0, -dx1, -(dx1 + dx2))
    x0_max = min(MAX_ORIGIN_X, MAX_ORIGIN_X - dx1,
                 MAX_ORIGIN_X - (dx1 + dx2))
    if x0_max < x0_min:
        return None
    x0 = rng.randint(x0_min, x0_max)
    x1, x2 = x0 + dx1, x0 + dx1 + dx2

    return [(x0, y0), (x1, y1), (x2, y2)]


def _sample_dx(rng):
    while True:
        dx = rng.randint(-DX_MAX, DX_MAX + 1)
        if abs(dx) >= DX_MIN:
            return dx


# ══════════════════════════════════════════════════════════════════════
# 标注映射(支持非方形源图)
# ══════════════════════════════════════════════════════════════════════

def count_visible_labels(labels, fx, fy, fsize, min_area_ratio=MIN_AREA_RATIO):
    """统计窗口内可见标注数"""
    count = 0
    for lb in labels:
        px = lb['cx'] * SRC_W
        py = lb['cy'] * SRC_H
        pw = lb['w'] * SRC_W
        ph = lb['h'] * SRC_H

        x1, y1 = px - pw / 2, py - ph / 2
        x2, y2 = px + pw / 2, py + ph / 2
        fx1, fy1 = fx, fy
        fx2, fy2 = fx + fsize, fy + fsize

        x1c, y1c = max(x1, fx1), max(y1, fy1)
        x2c, y2c = min(x2, fx2), min(y2, fy2)
        pw_c, ph_c = x2c - x1c, y2c - y1c
        if pw_c <= 0 or ph_c <= 0:
            continue
        if pw * ph > 0 and (pw_c * ph_c) / (pw * ph) >= min_area_ratio:
            count += 1
    return count


def map_labels_to_frame(labels, fx, fy, fsize, min_area_ratio=MIN_AREA_RATIO):
    """将源图归一化标注映射到窗口局部坐标(源图为 SRC_W×SRC_H)"""
    result = []
    for lb in labels:
        px = lb['cx'] * SRC_W
        py = lb['cy'] * SRC_H
        pw = lb['w'] * SRC_W
        ph = lb['h'] * SRC_H

        x1, y1 = px - pw / 2, py - ph / 2
        x2, y2 = px + pw / 2, py + ph / 2
        fx1, fy1 = fx, fy
        fx2, fy2 = fx + fsize, fy + fsize

        x1c, y1c = max(x1, fx1), max(y1, fy1)
        x2c, y2c = min(x2, fx2), min(y2, fy2)
        pw_c, ph_c = x2c - x1c, y2c - y1c
        if pw_c <= 0 or ph_c <= 0:
            continue

        orig_area = pw * ph
        if orig_area <= 0:
            continue
        if (pw_c * ph_c) / orig_area < min_area_ratio:
            continue

        kx_in = ky_in = None
        if lb['kx'] is not None and lb['ky'] is not None:
            kx_in = np.clip(lb['kx'] * SRC_W - fx, 0, fsize)
            ky_in = np.clip(lb['ky'] * SRC_H - fy, 0, fsize)

        result.append({
            'cls': lb['cls'],
            'cx': ((x1c + x2c) / 2 - fx) / fsize,
            'cy': ((y1c + y2c) / 2 - fy) / fsize,
            'w': pw_c / fsize,
            'h': ph_c / fsize,
            'kx': kx_in / fsize if kx_in is not None else None,
            'ky': ky_in / fsize if ky_in is not None else None,
        })
    return result


# ══════════════════════════════════════════════════════════════════════
# 单张源图处理
# ══════════════════════════════════════════════════════════════════════

def process_single_image(img, labels, samples_target, seq_idx, dst_dir, rng,
                         f0_label_occ_area_ratio=None,
                         f0_label_occ_max_size=None,
                         f0_label_occ_short_factor=None):
    """生成 samples_target 个合格序列, 返回 (生成数, 下一个seq编号)

    f0_label_occ_area_ratio: frame0 标注区域遮挡覆盖率(仅影响 frame0),
                             None 用模块默认。
    """
    n_gen = 0
    attempts = 0
    while n_gen < samples_target and attempts < MAX_ATTEMPTS_PER_IMAGE:
        attempts += 1

        # 每序列独立掷垂直方向 (上/下)
        dir_y = 1 if rng.random() < 0.5 else -1
        pos = sample_sequence_positions(rng, dir_y)
        if pos is None:
            continue

        (x0, y0), (x1, y1), (x2, y2) = pos

        # Frame 0 可见标注过滤
        if count_visible_labels(labels, x0, y0, FRAME_SIZE) < MIN_LABELS_FOR_KEYFRAME:
            continue

        # 提取三帧(直接裁切,不越界)
        frame0 = img[y0:y0 + FRAME_SIZE, x0:x0 + FRAME_SIZE].copy()
        frame1 = img[y1:y1 + FRAME_SIZE, x1:x1 + FRAME_SIZE].copy()
        frame2 = img[y2:y2 + FRAME_SIZE, x2:x2 + FRAME_SIZE].copy()

        # 每帧独立映射标注
        labels_f0 = map_labels_to_frame(labels, x0, y0, FRAME_SIZE)
        labels_f1 = map_labels_to_frame(labels, x1, y1, FRAME_SIZE)
        labels_f2 = map_labels_to_frame(labels, x2, y2, FRAME_SIZE)

        # 过曝参数序列级共享
        oe_params = generate_overexposure_params(frame0)

        # 模糊预掷骰: 至少一帧清晰
        blur_f0 = rng.random() < BLUR_PROB_KEYFRAME
        blur_f1 = rng.random() < BLUR_PROB_REFERENCE
        blur_f2 = rng.random() < BLUR_PROB_REFERENCE
        if blur_f0 and blur_f1 and blur_f2:
            unblur = rng.randint(0, 2)
            if unblur == 0:
                blur_f0 = False
            elif unblur == 1:
                blur_f1 = False
            else:
                blur_f2 = False

        frame0_n = apply_noises(frame0, 0, labels_f0, oe_params, blur_flag=blur_f0,
                                label_occ_area_ratio=f0_label_occ_area_ratio,
                                label_occ_max_size=f0_label_occ_max_size,
                                label_occ_short_factor=f0_label_occ_short_factor)
        frame1_n = apply_noises(frame1, 1, labels_f1, oe_params, blur_flag=blur_f1)
        frame2_n = apply_noises(frame2, 2, labels_f2, oe_params, blur_flag=blur_f2)

        # 保存
        seq_name = f"seq_{seq_idx:05d}"
        seq_dir = os.path.join(dst_dir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)
        cv2.imwrite(os.path.join(seq_dir, "frame_0.jpg"), frame0_n,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(seq_dir, "frame_1.jpg"), frame1_n,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(seq_dir, "frame_2.jpg"), frame2_n,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        save_labels(os.path.join(seq_dir, "labels_frame0.txt"), labels_f0)
        save_labels(os.path.join(seq_dir, "labels_frame1.txt"), labels_f1)
        save_labels(os.path.join(seq_dir, "labels_frame2.txt"), labels_f2)

        n_gen += 1
        seq_idx += 1

    return n_gen, seq_idx


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

def main(split_only=None):
    print("=" * 60)
    print("CropAndWeed 子集滑动窗口多帧生成(单独规则)")
    print(f"  源图 {SRC_W}×{SRC_H} | 窗口 {FRAME_SIZE} | "
          f"垂直 ±[{DY_MIN},{DY_MAX}] 同向 | 水平 ±{DX_MAX}")
    print(f"  输入: {SRC_ROOT}")
    print(f"  输出: {OUT_ROOT}")
    print("=" * 60)

    import shutil
    if split_only is None:
        if os.path.exists(OUT_ROOT):
            shutil.rmtree(OUT_ROOT)
        os.makedirs(OUT_ROOT, exist_ok=True)
    else:
        # 单 split 模式: 清空该 split 旧输出, 防止残留混入
        sp_dir = os.path.join(OUT_ROOT, split_only)
        if os.path.exists(sp_dir):
            shutil.rmtree(sp_dir)
        os.makedirs(sp_dir, exist_ok=True)

    total = {'train': 0, 'val': 0, 'test': 0}
    splits = [split_only] if split_only else ['train', 'val', 'test']
    for split in splits:
        src_img = os.path.join(SRC_ROOT, split, 'images')
        src_lbl = os.path.join(SRC_ROOT, split, 'labels')
        dst_dir = os.path.join(OUT_ROOT, split)
        os.makedirs(dst_dir, exist_ok=True)

        files = sorted(f for f in os.listdir(src_img)
                       if f.endswith(('.jpg', '.png', '.JPG', '.PNG')))
        rng = random.Random(SEED_BASE + sum(map(ord, split)))
        seq_idx = SEQ_START[split]
        n_img = 0
        n_seq = 0
        n_skip = 0

        print(f"\n[{split}] {len(files)} 张源图, 每图 ≤{SAMPLES_PER_IMAGE[split]} 序列, "
              f"seq 从 {seq_idx:05d} 起")

        # 按 split 设置 Frame 0 随机遮挡概率
        common.RANDOM_OCC_PROB_KEYFRAME = RANDOM_OCC_PROB_KEYFRAME_BY_SPLIT[split]
        print(f"          Frame0 随机遮挡概率: "
              f"{common.RANDOM_OCC_PROB_KEYFRAME:.0%}")

        for f in files:
            base = os.path.splitext(f)[0]
            img = cv2.imread(os.path.join(src_img, f))
            if img is None:
                continue
            labels = load_labels(os.path.join(src_lbl, base + '.txt'))
            if len(labels) < MIN_LABELS_FOR_KEYFRAME:
                n_skip += 1
                continue

            n_gen, seq_idx = process_single_image(
                img, labels, SAMPLES_PER_IMAGE[split], seq_idx, dst_dir, rng,
                f0_label_occ_area_ratio=F0_LABEL_OCC_AREA_RATIO_BY_SPLIT[split],
                f0_label_occ_max_size=F0_LABEL_OCC_MAX_SIZE_BY_SPLIT[split],
                f0_label_occ_short_factor=F0_LABEL_OCC_SHORT_FACTOR_BY_SPLIT[split])
            n_img += 1
            n_seq += n_gen
            if n_gen == 0:
                n_skip += 1
            if n_img % 100 == 0:
                print(f"  已处理 {n_img} 张, 序列 {n_seq}")

        total[split] = n_seq
        print(f"  [{split}] 完成: {n_seq} 序列 (跳过 {n_skip} 张无合格窗口)")

    print(f"\n{'='*60}")
    print(f"  合计: train {total['train']} / val {total['val']} / "
          f"test {total['test']} 序列")
    print(f"{'='*60}")

    # 可视化
    if split_only is None or split_only == 'train':
        create_visualization()
    print("\n完成!")


def create_visualization():
    """抽样序列, 每序列三帧并排(带标注)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seq_root = os.path.join(OUT_ROOT, 'train')
    seq_names = sorted(d for d in os.listdir(seq_root)
                       if os.path.isdir(os.path.join(seq_root, d)))
    rng = np.random.RandomState(11)
    if len(seq_names) > VIZ_COUNT:
        picked = rng.choice(seq_names, VIZ_COUNT, replace=False)
    else:
        picked = seq_names

    fig, axes = plt.subplots(len(picked), 3, figsize=(14, 4.4 * len(picked)))
    if len(picked) == 1:
        axes = np.array([axes])

    for idx, seq_name in enumerate(picked):
        seq_dir = os.path.join(seq_root, seq_name)
        for fi in range(3):
            ax = axes[idx, fi]
            img = cv2.imread(os.path.join(seq_dir, f"frame_{fi}.jpg"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lbs = load_labels(os.path.join(seq_dir, f"labels_frame{fi}.txt"))
            for lb in lbs:
                px, py = lb['cx'] * FRAME_SIZE, lb['cy'] * FRAME_SIZE
                pw, ph = lb['w'] * FRAME_SIZE, lb['h'] * FRAME_SIZE
                cv2.rectangle(img, (int(px - pw/2), int(py - ph/2)),
                              (int(px + pw/2), int(py + ph/2)), (0, 0, 255), 2)
                if lb['kx'] is not None:
                    cv2.circle(img, (int(lb['kx'] * FRAME_SIZE),
                                     int(lb['ky'] * FRAME_SIZE)),
                               4, (0, 255, 0), -1)
            ax.imshow(img)
            ax.set_title(f"{seq_name} | frame_{fi} ({len(lbs)} lbs)", fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    viz_path = os.path.join(OUT_ROOT, "sliding_check.png")
    plt.savefig(viz_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  可视化: {viz_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=['train', 'val', 'test'], default=None,
                    help="只处理指定集合(默认全部)")
    args = ap.parse_args()
    main(args.split)
