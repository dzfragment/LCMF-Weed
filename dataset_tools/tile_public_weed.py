#!/usr/bin/env python3
"""
CropAndWeed 子集(旋转后,已划分)拆块:每张 1088×1920 → 2 张 1024×1024

布局(重叠 128px 覆盖全高):
  上块: x∈[32,1056], y∈[0,1024]
  下块: x∈[32,1056], y∈[896,1920]
  (宽 1088 放不下 1024 窗口,水平居中取舍,两边各丢 32px)

标注映射:bbox 与块相交部分裁剪,保留面积 ≥ MIN_AREA_RATIO×原面积;
关键点夹到块边界内。与 stage1_split 逻辑一致。

输入: $WEED_EXP_ROOT/public_weed_le256_split/{train,val,test}/{images,labels}
输出: $WEED_EXP_ROOT/public_weed_le256_tiles/{train,val,test}/{images,labels}
      + tiles_manifest.json
"""

import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_labels, save_labels
from config import MIN_AREA_RATIO

_DATA_ROOT = os.environ.get('WEED_EXP_ROOT')
if not _DATA_ROOT:
    raise SystemExit('[dataset_tools] 请设置环境变量 WEED_EXP_ROOT：实验数据集根目录（旧 exp2：'
                     'weed1024 / public_weed_le256_* / weed_public_* / weed_sliding_merged 等所在）')

SRC_ROOT = os.path.join(_DATA_ROOT, 'public_weed_le256_split')
OUT_ROOT = os.path.join(_DATA_ROOT, 'public_weed_le256_tiles')

TILE = 1024
OVERLAP_PX = 128          # 上下重叠量
H_CROP = (1088 - TILE) // 2  # 水平居中: 每边丢 32px

# 两块位置: (名称, x0, y0)
TILES = [
    ("tile_0", H_CROP, 0),                        # 上块
    ("tile_1", H_CROP, 1920 - TILE),              # 下块 (y=896, 重叠128)
]

VIZ_COUNT = 4  # 可视化抽样组数


def clip_bbox_to_tile(px, py, pw, ph, tile_origin_x, tile_origin_y, tile_size):
    """将 bbox 截断到 tile 范围内, 返回 (px_in, py_in, pw_clipped, ph_clipped) 或 None"""
    x1, y1 = px - pw / 2, py - ph / 2
    x2, y2 = px + pw / 2, py + ph / 2

    tx1, ty1 = tile_origin_x, tile_origin_y
    tx2, ty2 = tile_origin_x + tile_size, tile_origin_y + tile_size

    x1c, y1c = max(x1, tx1), max(y1, ty1)
    x2c, y2c = min(x2, tx2), min(y2, ty2)

    pw_c, ph_c = x2c - x1c, y2c - y1c
    if pw_c <= 0 or ph_c <= 0:
        return None

    return (x1c + x2c) / 2 - tx1, (y1c + y2c) / 2 - ty1, pw_c, ph_c


def split_image(img, labels):
    """拆块, 返回 dict: tile_name -> (tile_img, tile_labels)"""
    h, w = img.shape[:2]
    result = {}
    for tname, x0, y0 in TILES:
        tile = img[y0:y0 + TILE, x0:x0 + TILE].copy()
        tile_labels = []
        for lb in labels:
            px, py = lb['cx'] * w, lb['cy'] * h
            pw, ph = lb['w'] * w, lb['h'] * h

            clipped = clip_bbox_to_tile(px, py, pw, ph, x0, y0, TILE)
            if clipped is None:
                continue
            px_in, py_in, pw_c, ph_c = clipped

            orig_area = pw * ph
            if orig_area <= 0:
                continue
            if (pw_c * ph_c) / orig_area < MIN_AREA_RATIO:
                continue

            # 关键点夹到块内
            kx_in = ky_in = None
            if lb['kx'] is not None and lb['ky'] is not None:
                kx_in = max(0.0, min(TILE, lb['kx'] * w - x0))
                ky_in = max(0.0, min(TILE, lb['ky'] * h - y0))

            tile_labels.append({
                'cls': lb['cls'],
                'cx': px_in / TILE,
                'cy': py_in / TILE,
                'w': pw_c / TILE,
                'h': ph_c / TILE,
                'kx': kx_in / TILE if kx_in is not None else None,
                'ky': ky_in / TILE if ky_in is not None else None,
            })
        result[tname] = (tile, tile_labels)
    return result


def main():
    print("=" * 60)
    print("CropAndWeed 拆块: 1088×1920 → 2×1024×1024 (重叠128px)")
    print(f"  输入: {SRC_ROOT}")
    print(f"  输出: {OUT_ROOT}")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        src_img = os.path.join(SRC_ROOT, split, 'images')
        src_lbl = os.path.join(SRC_ROOT, split, 'labels')
        dst_img = os.path.join(OUT_ROOT, split, 'images')
        dst_lbl = os.path.join(OUT_ROOT, split, 'labels')
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)

        files = sorted(f for f in os.listdir(src_img)
                       if f.endswith(('.jpg', '.png', '.JPG', '.PNG')))
        n_src = len(files)
        n_tile = 0
        n_orig_lb = 0
        n_map_lb = 0
        n_empty = 0
        manifest = {}

        for f in files:
            base = os.path.splitext(f)[0]
            img = cv2.imread(os.path.join(src_img, f))
            if img is None:
                print(f"  [WARN] 无法读取 {f}")
                continue
            labels = load_labels(os.path.join(src_lbl, base + '.txt'))
            n_orig_lb += len(labels)

            tiles = split_image(img, labels)
            for tname, (tile_img, tile_labels) in tiles.items():
                out_name = f"{base}_{tname}"
                cv2.imwrite(os.path.join(dst_img, out_name + '.jpg'), tile_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                save_labels(os.path.join(dst_lbl, out_name + '.txt'), tile_labels)
                manifest[out_name] = {
                    'split': split, 'src': base, 'tile': tname,
                    'bbox': {'x0': TILES[0 if tname.endswith('0') else 1][1],
                             'y0': TILES[0 if tname.endswith('0') else 1][2]},
                }
                n_map_lb += len(tile_labels)
                n_tile += 1
                if len(tile_labels) == 0:
                    n_empty += 1

        print(f"\n[{split}]")
        print(f"  源图 {n_src} → 块 {n_tile} (期望 {n_src * 2})")
        print(f"  标注: 原 {n_orig_lb} → 映射 {n_map_lb} "
              f"({n_map_lb / max(n_orig_lb, 1) * 100:.1f}%)")
        print(f"  空标注块: {n_empty} ({n_empty / max(n_tile, 1) * 100:.1f}%)")

        with open(os.path.join(OUT_ROOT, f"manifest_{split}.json"), 'w') as mf:
            json.dump(manifest, mf, indent=2, ensure_ascii=False)

    # 汇总 manifest
    all_m = {}
    for split in ['train', 'val', 'test']:
        with open(os.path.join(OUT_ROOT, f"manifest_{split}.json")) as mf:
            all_m.update(json.load(mf))
    with open(os.path.join(OUT_ROOT, "tiles_manifest.json"), 'w') as mf:
        json.dump(all_m, mf, indent=2, ensure_ascii=False)

    # 可视化
    create_visualization()
    print("\n完成!")


def create_visualization():
    """抽样: 原图(带标注) + 上下两块(带标注)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(7)
    src_img = os.path.join(SRC_ROOT, 'train', 'images')
    src_lbl = os.path.join(SRC_ROOT, 'train', 'labels')
    files = sorted(f for f in os.listdir(src_img)
                   if f.endswith(('.jpg', '.png')))
    picked = rng.choice(files, VIZ_COUNT, replace=False)

    fig, axes = plt.subplots(VIZ_COUNT, 3, figsize=(14, 4.6 * VIZ_COUNT))

    def draw(img, labels):
        h, w = img.shape[:2]
        for lb in labels:
            px, py = lb['cx'] * w, lb['cy'] * h
            pw, ph = lb['w'] * w, lb['h'] * h
            cv2.rectangle(img, (int(px - pw/2), int(py - ph/2)),
                          (int(px + pw/2), int(py + ph/2)), (0, 0, 255), 2)
            if lb['kx'] is not None:
                cv2.circle(img, (int(lb['kx'] * w), int(lb['ky'] * h)),
                           4, (0, 255, 0), -1)

    for idx, f in enumerate(picked):
        base = os.path.splitext(f)[0]
        orig = cv2.imread(os.path.join(src_img, f))
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        labels = load_labels(os.path.join(src_lbl, base + '.txt'))
        draw(orig, labels)

        t0_img = cv2.imread(os.path.join(OUT_ROOT, 'train', 'images',
                                         f"{base}_tile_0.jpg"))
        t0_img = cv2.cvtColor(t0_img, cv2.COLOR_BGR2RGB)
        t0_lb = load_labels(os.path.join(OUT_ROOT, 'train', 'labels',
                                         f"{base}_tile_0.txt"))
        t1_img = cv2.imread(os.path.join(OUT_ROOT, 'train', 'images',
                                         f"{base}_tile_1.jpg"))
        t1_img = cv2.cvtColor(t1_img, cv2.COLOR_BGR2RGB)
        t1_lb = load_labels(os.path.join(OUT_ROOT, 'train', 'labels',
                                         f"{base}_tile_1.txt"))
        draw(t0_img, t0_lb)
        draw(t1_img, t1_lb)

        axes[idx, 0].imshow(orig)
        axes[idx, 0].set_title(f"{base}\norig 1088×1920 ({len(labels)} lbs)", fontsize=9)
        axes[idx, 0].axis('off')
        axes[idx, 1].imshow(t0_img)
        axes[idx, 1].set_title(f"tile_0 (top) {len(t0_lb)} lbs", fontsize=9)
        axes[idx, 1].axis('off')
        axes[idx, 2].imshow(t1_img)
        axes[idx, 2].set_title(f"tile_1 (bottom) {len(t1_lb)} lbs", fontsize=9)
        axes[idx, 2].axis('off')

    plt.tight_layout()
    viz_path = os.path.join(OUT_ROOT, "tile_check.png")
    plt.savefig(viz_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  可视化: {viz_path}")


if __name__ == "__main__":
    main()
