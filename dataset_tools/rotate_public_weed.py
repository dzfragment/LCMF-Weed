#!/usr/bin/env python3
"""
CropAndWeed 子集 (public_weed_le256) 旋转 90°(顺时针),标注同步换算。

为什么旋转:新数据集 1920×1088 是横屏,直接做 1024 滑动窗口时垂直方向
(1088-1024=64px)几乎没有滑动空间。旋转成 1088×1920 竖屏后,垂直方向有
896px 空间,可以延续老数据集"从上到下"的滑动方向;短边 1088 变成水平方向,
后续滑动逻辑需限制该方向的移动不超出原图。

旋转换算(顺时针 90°,原图 W=1920, H=1088 → 新图 W'=1088, H'=1920):
  像素: (x, y) → (x', y') = (H-1-y, x)
  归一化标注:
    cx' = 1 - cy,  cy' = cx
    w'  = h,       h'  = w       (宽高互换)
    kx' = 1 - ky,  ky' = kx

输入: $WEED_EXP_ROOT/public_weed_le256/{images,labels}
输出: $WEED_EXP_ROOT/public_weed_le256_rotated/{images,labels} + rotate_check.png
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_labels, save_labels

_DATA_ROOT = os.environ.get('WEED_EXP_ROOT')
if not _DATA_ROOT:
    raise SystemExit('[dataset_tools] 请设置环境变量 WEED_EXP_ROOT：实验数据集根目录（旧 exp2：'
                     'weed1024 / public_weed_le256_* / weed_public_* / weed_sliding_merged 等所在）')

SRC_IMG = os.path.join(_DATA_ROOT, 'public_weed_le256/images')
SRC_LBL = os.path.join(_DATA_ROOT, 'public_weed_le256/labels')
OUT_DIR = os.path.join(_DATA_ROOT, 'public_weed_le256_rotated')

VIZ_COUNT = 6   # 可视化抽样张数


def rotate_label_90cw(lb):
    """顺时针旋转 90° 的标注换算(归一化坐标)"""
    kx = lb['kx'] if lb['kx'] is not None else None
    ky = lb['ky'] if lb['ky'] is not None else None
    return {
        'cls': lb['cls'],
        'cx':  1.0 - lb['cy'],
        'cy':  lb['cx'],
        'w':   lb['h'],
        'h':   lb['w'],
        'kx':  1.0 - ky if kx is not None else None,
        'ky':  kx if kx is not None else None,
    }


def check_label_valid(lb, name):
    """校验标注是否在合法范围内"""
    for key, v in [('cx', lb['cx']), ('cy', lb['cy']),
                   ('w', lb['w']), ('h', lb['h'])]:
        if not (0.0 <= v <= 1.0):
            print(f"  [WARN] {name} {key}={v:.4f} 越界")
            return False
    if lb['kx'] is not None:
        for key, v in [('kx', lb['kx']), ('ky', lb['ky'])]:
            if not (0.0 <= v <= 1.0):
                print(f"  [WARN] {name} {key}={v:.4f} 越界")
                return False
    return True


def draw_labels(img, labels, scale=1.0):
    """在图上画标注框 + 关键点(直接修改 img)"""
    h, w = img.shape[:2]
    for lb in labels:
        px = lb['cx'] * w
        py = lb['cy'] * h
        pw = lb['w'] * w
        ph = lb['h'] * h
        x1, y1 = int(px - pw / 2), int(py - ph / 2)
        x2, y2 = int(px + pw / 2), int(py + ph / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if lb['kx'] is not None:
            cv2.circle(img, (int(lb['kx'] * w), int(lb['ky'] * h)),
                       4, (0, 255, 0), -1)
    return img


def create_visualization(sample_names):
    """原图(带标注) | 旋转后(带标注) 对比拼图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(sample_names)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n))

    for idx, name in enumerate(sample_names):
        # 原图
        orig = cv2.imread(os.path.join(SRC_IMG, name + '.jpg'))
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig_labels = load_labels(os.path.join(SRC_LBL, name + '.txt'))
        draw_labels(orig, orig_labels)

        # 旋转后
        rot = cv2.imread(os.path.join(OUT_DIR, 'images', name + '.jpg'))
        rot = cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)
        rot_labels = load_labels(os.path.join(OUT_DIR, 'labels', name + '.txt'))
        draw_labels(rot, rot_labels)

        ax0, ax1 = axes[idx]
        ax0.imshow(orig)
        ax0.set_title(f"{name}\norig {orig.shape[1]}x{orig.shape[0]} ({len(orig_labels)} lbs)",
                      fontsize=9)
        ax0.axis('off')
        ax1.imshow(rot)
        ax1.set_title(f"rot90cw {rot.shape[1]}x{rot.shape[0]} ({len(rot_labels)} lbs)",
                      fontsize=9)
        ax1.axis('off')

    plt.tight_layout()
    viz_path = os.path.join(OUT_DIR, "rotate_check.png")
    plt.savefig(viz_path, dpi=110, bbox_inches='tight')
    plt.close()
    return viz_path


def main():
    print("=" * 60)
    print("CropAndWeed 子集旋转 90°(顺时针) + 标注换算")
    print(f"  输入: {SRC_IMG}")
    print(f"  输出: {OUT_DIR}")
    print("=" * 60)

    os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'labels'), exist_ok=True)

    files = sorted(f for f in os.listdir(SRC_IMG)
                   if f.endswith(('.jpg', '.png', '.JPG', '.PNG')))
    print(f"共 {len(files)} 张图片")

    n_img, n_lb = 0, 0
    n_warn = 0
    for f in files:
        name = os.path.splitext(f)[0]
        img = cv2.imread(os.path.join(SRC_IMG, f))
        if img is None:
            print(f"  [WARN] 无法读取 {f}, 跳过")
            continue

        labels = load_labels(os.path.join(SRC_LBL, name + '.txt'))

        # 旋转
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rot_labels = [rotate_label_90cw(lb) for lb in labels]

        # 校验
        for lb in rot_labels:
            if not check_label_valid(lb, name):
                n_warn += 1

        cv2.imwrite(os.path.join(OUT_DIR, 'images', f), rot,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        save_labels(os.path.join(OUT_DIR, 'labels', name + '.txt'), rot_labels)

        n_img += 1
        n_lb += len(rot_labels)
        if n_img % 200 == 0:
            print(f"  已处理 {n_img} 张...")

    print(f"\n完成: {n_img} 张图片, {n_lb} 个标注 (越界警告 {n_warn} 条)")
    print(f"旋转后尺寸: 1088×1920 (已验证 {n_img > 0} 张)")

    # 校验输出尺寸
    import glob
    bad_size = 0
    for p in glob.glob(os.path.join(OUT_DIR, 'images', '*.jpg'))[:]:
        im = cv2.imread(p)
        if im is None or im.shape[0] != 1920 or im.shape[1] != 1088:
            bad_size += 1
    print(f"尺寸校验: {bad_size} 张异常")

    # 可视化
    print("\n[生成可视化]")
    rng = np.random.RandomState(42)
    candidates = [os.path.splitext(f)[0] for f in files]
    # 优先选标注多的,覆盖 ave / vwg
    def n_lbs(name):
        return len(load_labels(os.path.join(SRC_LBL, name + '.txt')))
    cand_sorted = sorted(candidates, key=n_lbs, reverse=True)
    ave_pool = [c for c in cand_sorted if c.startswith('ave-')][:10]
    vwg_pool = [c for c in cand_sorted if c.startswith('vwg-')][:10]
    picked = (rng.choice(ave_pool, 3, replace=False).tolist() +
              rng.choice(vwg_pool, 3, replace=False).tolist())
    viz_path = create_visualization(picked)
    print(f"  可视化已保存: {viz_path}")


if __name__ == "__main__":
    main()
