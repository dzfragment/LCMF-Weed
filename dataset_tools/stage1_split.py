"""
第一阶段：大图拆分
1. 所有图片统一 resize 到 2048×2048
2. 2×2 切分成 4 张 1024×1024 子图
3. 标注映射 + 跨边界处理
"""

import os
import sys
import cv2
import numpy as np

# 确保能找到 config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS,
    STAGE1_OUT, STAGE1_YAML,
    RESIZE_TARGET, TILE_SIZE, GRID_COLS, GRID_ROWS,
    MIN_AREA_RATIO, VIZ_SAMPLE_COUNT,
)
from common import load_labels, save_labels


def bbox_to_pixels(label, img_size):
    """将归一化标注转为像素坐标。
    返回 (px, py, pw, ph, pkx, pky)"""
    w_img, h_img = img_size
    px = label['cx'] * w_img
    py = label['cy'] * h_img
    pw = label['w'] * w_img
    ph = label['h'] * h_img
    pkx = label['kx'] * w_img if label['kx'] is not None else None
    pky = label['ky'] * h_img if label['ky'] is not None else None
    return px, py, pw, ph, pkx, pky


def clip_bbox_to_tile(px, py, pw, ph, tile_origin_x, tile_origin_y, tile_size):
    """将 bbox 截断到 tile 范围内，返回截断后的 (px_in, py_in, pw_clipped, ph_clipped)"""
    x1 = px - pw / 2
    y1 = py - ph / 2
    x2 = px + pw / 2
    y2 = py + ph / 2

    tile_x1 = tile_origin_x
    tile_y1 = tile_origin_y
    tile_x2 = tile_origin_x + tile_size
    tile_y2 = tile_origin_y + tile_size

    x1_clip = max(x1, tile_x1)
    y1_clip = max(y1, tile_y1)
    x2_clip = min(x2, tile_x2)
    y2_clip = min(y2, tile_y2)

    pw_clipped = x2_clip - x1_clip
    ph_clipped = y2_clip - y1_clip

    if pw_clipped <= 0 or ph_clipped <= 0:
        return None  # 完全不在 tile 内

    px_in = (x1_clip + x2_clip) / 2 - tile_origin_x
    py_in = (y1_clip + y2_clip) / 2 - tile_origin_y

    return px_in, py_in, pw_clipped, ph_clipped


def map_labels_to_tiles(labels, tile_size, grid_rows, grid_cols):
    """将标注映射到各 tile。
    返回 dict: {(row, col): [label_dict_normalized_to_tile]}"""
    tile_labels = {(r, c): [] for r in range(grid_rows) for c in range(grid_cols)}

    img_size = (tile_size * grid_cols, tile_size * grid_rows)  # 2048×2048

    for lb in labels:
        px, py, pw, ph, pkx, pky = bbox_to_pixels(lb, img_size)

        # 判断 bbox 中心点落在哪个 tile
        col = int(px // tile_size)
        row = int(py // tile_size)

        # 边界情况（中心点刚好在 edge 上）
        col = min(col, grid_cols - 1)
        row = min(row, grid_rows - 1)

        tile_origin_x = col * tile_size
        tile_origin_y = row * tile_size

        # 截断 bbox
        clipped = clip_bbox_to_tile(px, py, pw, ph, tile_origin_x, tile_origin_y, tile_size)
        if clipped is None:
            continue

        px_in, py_in, pw_clipped, ph_clipped = clipped

        # 面积占比检查
        original_area = pw * ph
        if original_area <= 0:
            continue
        clipped_area = pw_clipped * ph_clipped
        area_ratio = clipped_area / original_area

        if area_ratio < MIN_AREA_RATIO:
            continue

        # 处理关键点
        kx_in, ky_in = None, None
        if pkx is not None and pky is not None:
            # 将关键点也截断到 tile 内
            kx_in = max(0, min(tile_size, pkx - tile_origin_x))
            ky_in = max(0, min(tile_size, pky - tile_origin_y))

        # 重新归一化到 tile_size × tile_size
        cx_norm = px_in / tile_size
        cy_norm = py_in / tile_size
        w_norm = pw_clipped / tile_size
        h_norm = ph_clipped / tile_size

        kx_norm = kx_in / tile_size if kx_in is not None else None
        ky_norm = ky_in / tile_size if ky_in is not None else None

        tile_labels[(row, col)].append({
            'cls': lb['cls'],
            'cx': cx_norm,
            'cy': cy_norm,
            'w': w_norm,
            'h': h_norm,
            'kx': kx_norm,
            'ky': ky_norm,
        })

    return tile_labels


def create_yaml(output_dir, yaml_name):
    """创建数据集 YAML 配置文件"""
    yaml_path = os.path.join(output_dir, yaml_name)
    content = f"""# 拆分后数据集配置 (1024×1024 tiles)
train: ./train/images
val: ./val/images
test: ./test/images

nc: 1
kpt_shape: [1, 2]
flip_idx: [0]
names: ['weed']
"""
    with open(yaml_path, 'w') as f:
        f.write(content)
    return yaml_path


def create_visualization(output_dir, split_name, images_dir, labels_dir):
    """随机抽样创建可视化大图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle

    # 找所有原始图片名（去重 tile 后缀）
    all_tiles = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    # 提取基本名
    base_names = sorted(set(
        f.rsplit('_tile_', 1)[0] for f in all_tiles
    ))

    # 随机抽样
    np.random.seed(42)
    sample_count = min(VIZ_SAMPLE_COUNT, len(base_names))
    sampled = np.random.choice(base_names, sample_count, replace=False)

    fig, axes = plt.subplots(sample_count, 5, figsize=(25, 5 * sample_count))
    if sample_count == 1:
        axes = np.array([axes])

    for idx, base in enumerate(sampled):
        # 读取原图
        # 需要反查原图路径
        orig_img = None
        orig_labels = None
        for img_dir, lbl_dir in [(TRAIN_IMAGES, TRAIN_LABELS), (VAL_IMAGES, VAL_LABELS)]:
            for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                orig_path = os.path.join(img_dir, base + ext)
                if os.path.exists(orig_path):
                    orig_img = cv2.imread(orig_path)
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                    # resize for display
                    orig_img = cv2.resize(orig_img, (RESIZE_TARGET, RESIZE_TARGET))
                    orig_labels = load_labels(os.path.join(lbl_dir, base + '.txt'))
                    break
            if orig_img is not None:
                break

        if orig_img is None:
            continue

        h_orig, w_orig = orig_img.shape[:2]

        # 画原图
        ax = axes[idx, 0]
        ax.imshow(orig_img)
        ax.set_title(f"{base}\n(orig, {len(orig_labels)} labels)", fontsize=9)
        ax.axis('off')

        # 标注
        for lb in orig_labels:
            px = lb['cx'] * w_orig
            py = lb['cy'] * h_orig
            pw = lb['w'] * w_orig
            ph = lb['h'] * h_orig
            rect = Rectangle((px - pw/2, py - ph/2), pw, ph,
                             linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            if lb['kx'] is not None:
                kx = lb['kx'] * w_orig
                ky = lb['ky'] * h_orig
                ax.plot(kx, ky, 'go', markersize=2)

        # 画 4 块 tile
        for ti, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            ax = axes[idx, ti + 1]
            tile_name = f"{base}_tile_{row}_{col}"
            tile_path = os.path.join(images_dir, tile_name + '.jpg')
            if not os.path.exists(tile_path):
                tile_path = os.path.join(images_dir, tile_name + '.png')
            tile_img = cv2.imread(tile_path)
            if tile_img is not None:
                tile_img = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
                ax.imshow(tile_img)

            label_path = os.path.join(labels_dir, tile_name + '.txt')
            tile_lbs = load_labels(label_path)
            ax.set_title(f"tile_{row}_{col} ({len(tile_lbs)} labels)", fontsize=9)
            ax.axis('off')

            for lb in tile_lbs:
                px_t = lb['cx'] * TILE_SIZE
                py_t = lb['cy'] * TILE_SIZE
                pw_t = lb['w'] * TILE_SIZE
                ph_t = lb['h'] * TILE_SIZE
                rect = Rectangle((px_t - pw_t/2, py_t - ph_t/2), pw_t, ph_t,
                                 linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                if lb['kx'] is not None:
                    kx_t = lb['kx'] * TILE_SIZE
                    ky_t = lb['ky'] * TILE_SIZE
                    ax.plot(kx_t, ky_t, 'go', markersize=2)

    plt.tight_layout()
    viz_path = os.path.join(output_dir, "split_viz.png")
    plt.savefig(viz_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  可视化已保存: {viz_path}")
    return viz_path


def main():
    print("=" * 60)
    print("第一阶段：大图拆分 (Resize 2048 → 2×2 Split)")
    print("  去重合并 train+val 后重新 80/20 划分")
    print("=" * 60)

    # ---- 收集全部图片，去重 ----
    all_images = {}  # basename → (src_images_dir, src_labels_dir)
    for img_dir, lbl_dir in [(TRAIN_IMAGES, TRAIN_LABELS), (VAL_IMAGES, VAL_LABELS)]:
        for f in os.listdir(img_dir):
            if not f.endswith(('.jpg', '.png', '.JPG', '.PNG')):
                continue
            base = os.path.splitext(f)[0]
            if base not in all_images:
                all_images[base] = (img_dir, lbl_dir)
    print(f"去重后共 {len(all_images)} 张原图 (train+val 合并)")

    # ---- 固定种子随机划分 80/20 ----
    bases = sorted(all_images.keys())
    np.random.seed(42)
    perm = np.random.permutation(len(bases))
    split_n = int(len(bases) * 0.8)
    train_bases = set(bases[i] for i in perm[:split_n])
    val_bases   = set(bases[i] for i in perm[split_n:])
    print(f"划分: train {len(train_bases)} / val {len(val_bases)}")

    # ---- 处理 train ----
    print("\n[Train 拆分]")
    train_img_out = os.path.join(STAGE1_OUT, "train", "images")
    train_lbl_out = os.path.join(STAGE1_OUT, "train", "labels")

    train_files = [(base, *all_images[base]) for base in train_bases]
    train_stats = _process_split_from_list(train_files, train_img_out, train_lbl_out)

    print(f"\n  Train 统计:")
    print(f"    输入图片: {train_stats['total_images']}")
    print(f"    输出 tiles: {train_stats['total_tiles']}")
    print(f"    原始标注: {train_stats['total_original_labels']}")
    print(f"    映射标注: {train_stats['total_mapped_labels']}")
    print(f"    丢弃标注总计: {train_stats['discarded_total']}")
    print(f"    无标注图片: {train_stats['skipped_no_label']}")

    # ---- 处理 val ----
    print("\n[Val 拆分]")
    val_img_out = os.path.join(STAGE1_OUT, "val", "images")
    val_lbl_out = os.path.join(STAGE1_OUT, "val", "labels")

    val_files = [(base, *all_images[base]) for base in val_bases]
    val_stats = _process_split_from_list(val_files, val_img_out, val_lbl_out)

    print(f"\n  Val 统计:")
    print(f"    输入图片: {val_stats['total_images']}")
    print(f"    输出 tiles: {val_stats['total_tiles']}")
    print(f"    原始标注: {val_stats['total_original_labels']}")
    print(f"    映射标注: {val_stats['total_mapped_labels']}")
    print(f"    丢弃标注总计: {val_stats['discarded_total']}")
    print(f"    无标注图片: {val_stats['skipped_no_label']}")

    # 汇总
    total_orig = train_stats['total_original_labels'] + val_stats['total_original_labels']
    total_mapped = train_stats['total_mapped_labels'] + val_stats['total_mapped_labels']
    total_discarded = train_stats['discarded_total'] + val_stats['discarded_total']
    print(f"\n  汇总: {total_orig} 个标注 → {total_mapped} 个映射标注 "
          f"(丢弃 {total_discarded}, {total_discarded/max(total_orig,1)*100:.1f}%)")

    yaml_path = create_yaml(STAGE1_OUT, STAGE1_YAML)
    print(f"\n  YAML 已保存: {yaml_path}")

    print("\n[生成可视化 split_viz.png]")
    create_visualization(STAGE1_OUT, "train", train_img_out, train_lbl_out)

    print("\n第一阶段完成！")


def _process_split_from_list(file_list, dst_images_dir, dst_labels_dir):
    """file_list: [(base_name, src_images_dir, src_labels_dir), ...]"""
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    stats = {
        'total_images': 0,
        'total_tiles': 0,
        'total_original_labels': 0,
        'total_mapped_labels': 0,
        'discarded_total': 0,
        'skipped_no_label': 0,
    }

    for base, src_img_dir, src_lbl_dir in file_list:
        # 找图片文件
        img_path = None
        for ext in ['.jpg', '.png', '.JPG', '.PNG']:
            candidate = os.path.join(src_img_dir, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        label_path = os.path.join(src_lbl_dir, base + '.txt')
        labels = load_labels(label_path)
        n_orig = len(labels)

        if img.shape[0] != RESIZE_TARGET or img.shape[1] != RESIZE_TARGET:
            img = cv2.resize(img, (RESIZE_TARGET, RESIZE_TARGET), interpolation=cv2.INTER_LINEAR)

        tile_labels = map_labels_to_tiles(labels, TILE_SIZE, GRID_ROWS, GRID_COLS)

        n_mapped = 0
        for (row, col), tl in tile_labels.items():
            x1 = col * TILE_SIZE
            y1 = row * TILE_SIZE
            tile_img = img[y1:y1 + TILE_SIZE, x1:x1 + TILE_SIZE].copy()
            tile_name = f"{base}_tile_{row}_{col}.jpg"
            cv2.imwrite(os.path.join(dst_images_dir, tile_name), tile_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            save_labels(os.path.join(dst_labels_dir, tile_name.replace('.jpg', '.txt')), tl)
            n_mapped += len(tl)
            stats['total_tiles'] += 1

        stats['total_images'] += 1
        stats['total_original_labels'] += n_orig
        stats['total_mapped_labels'] += n_mapped
        if n_orig == 0:
            stats['skipped_no_label'] += 1

        if stats['total_images'] % 20 == 0:
            print(f"  已处理 {stats['total_images']} 张... (标注原 {n_orig} → 映射 {n_mapped})")

    stats['discarded_total'] = stats['total_original_labels'] - stats['total_mapped_labels']
    return stats


if __name__ == "__main__":
    main()
