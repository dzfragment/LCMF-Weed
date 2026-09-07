"""
第二阶段（滑动窗口版）：从 2048×2048 源图滑动窗口提取多帧序列

与旧版 stage2_synthetic.py 的核心区别：
1. 从 resize 后的 2048 源图滑动窗口提取 3 帧（旧版从固定 1024 tile 裁切）
2. 每帧独立计算可见标注 —— 窗口偏移会导致不同杂草进入/离开视野
3. 标注区域遮挡基于每帧实际的可见标注坐标（旧版三帧共用 Frame 0 标注）

工作流程：
  weedataset (train/val/test, 各种分辨率)
    → resize 到 2048×2048
    → mirror padding 384px
    → 随机/网格采样 Frame 0 位置 (需 ≥4 个标注)
    → Frame 1/2 按偏移量提取
    → 每帧独立映射标注 + 施加噪声/曝光/遮挡
    → 输出序列目录
"""

import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    # 源数据
    SRC_TRAIN_IMAGES, SRC_TRAIN_LABELS,
    SRC_VAL_IMAGES, SRC_VAL_LABELS,
    SRC_TEST_IMAGES, SRC_TEST_LABELS,
    # 输出
    STAGE2_SLIDING_OUT, STAGE2_SLIDING_YAML,
    # 参数
    RESIZE_TARGET, TILE_SIZE,
    FRAME1_DX_MIN, FRAME1_DX_MAX, FRAME1_DY_MIN, FRAME1_DY_MAX,
    FRAME2_DX_MIN, FRAME2_DX_MAX, FRAME2_DY_MIN, FRAME2_DY_MAX,
    DX_MIN_MAGNITUDE, DY_MIN_MAGNITUDE,
    MIRROR_PADDING,
    MIN_LABELS_FOR_KEYFRAME,
    OCCLUDE_LABEL_RATIO_KEYFRAME,
    BLUR_PROB_KEYFRAME, BLUR_PROB_REFERENCE,
    MIN_AREA_RATIO,
    BASE_SEED,
    MAX_ATTEMPTS_PER_IMAGE,
    SLIDING_SAMPLES_TRAIN, SLIDING_SAMPLES_VAL, SLIDING_SAMPLES_TEST,
    SLIDING_STRIDE,
    SYN_VIZ_SAMPLE_COUNT,
)
from common import (load_labels, save_labels,
                    apply_noises, apply_vibration_blur,
                    apply_overexposure, apply_random_occlusion,
                    apply_label_occlusion,
                    gaussian_blur_only, generate_overexposure_params,
                    get_group_key)

FRAME_SIZE = TILE_SIZE  # 1024


# ══════════════════════════════════════════════════════════════════════
# 标注映射 —— 核心改动：将源图标注映射到滑动窗口内的局部坐标
# ══════════════════════════════════════════════════════════════════════

def clip_bbox_to_frame(px, py, pw, ph, fx, fy, fsize):
    """将像素坐标 bbox 截断到 frame 范围内。

    Args:
        px, py: bbox 中心在源图中的像素坐标
        pw, ph: bbox 宽高（像素）
        fx, fy: frame 左上角在源图中的像素坐标
        fsize:  frame 尺寸 (1024)

    Returns:
        (px_in, py_in, pw_clipped, ph_clipped) 或 None (完全不可见)
        px_in, py_in 是相对于 frame 左上角的坐标
    """
    x1 = px - pw / 2
    y1 = py - ph / 2
    x2 = px + pw / 2
    y2 = py + ph / 2

    frame_x1, frame_y1 = fx, fy
    frame_x2, frame_y2 = fx + fsize, fy + fsize

    # 截断
    x1_clip = max(x1, frame_x1)
    y1_clip = max(y1, frame_y1)
    x2_clip = min(x2, frame_x2)
    y2_clip = min(y2, frame_y2)

    pw_clipped = x2_clip - x1_clip
    ph_clipped = y2_clip - y1_clip

    if pw_clipped <= 0 or ph_clipped <= 0:
        return None

    # 转到 frame 局部坐标
    px_in = (x1_clip + x2_clip) / 2 - frame_x1
    py_in = (y1_clip + y2_clip) / 2 - frame_y1

    return px_in, py_in, pw_clipped, ph_clipped


def map_labels_to_frame(labels, fx, fy, fsize, src_size, min_area_ratio=MIN_AREA_RATIO):
    """将源图归一化标注映射到指定 frame 的局部归一化坐标。

    Args:
        labels:   源图归一化标注列表 (cx,cy,w,h 都是 0~1)
        fx, fy:   frame 左上角在源图中的像素坐标
        fsize:    frame 边长 (1024)
        src_size: 源图边长 (2048)
        min_area_ratio: bbox 截断后面积低于此比例则丢弃

    Returns:
        list[dict]: frame 局部归一化标注
    """
    result = []
    for lb in labels:
        # 源图像素坐标
        px = lb['cx'] * src_size
        py = lb['cy'] * src_size
        pw = lb['w'] * src_size
        ph = lb['h'] * src_size

        clipped = clip_bbox_to_frame(px, py, pw, ph, fx, fy, fsize)
        if clipped is None:
            continue

        px_in, py_in, pw_clipped, ph_clipped = clipped

        # 面积占比过滤
        orig_area = pw * ph
        if orig_area <= 0:
            continue
        if (pw_clipped * ph_clipped) / orig_area < min_area_ratio:
            continue

        # 关键点映射
        kx_in, ky_in = None, None
        if lb['kx'] is not None and lb['ky'] is not None:
            kx_src = lb['kx'] * src_size
            ky_src = lb['ky'] * src_size
            kx_in = np.clip(kx_src - fx, 0, fsize)
            ky_in = np.clip(ky_src - fy, 0, fsize)

        # 归一化到 frame 尺寸
        result.append({
            'cls': lb['cls'],
            'cx':  px_in / fsize,
            'cy':  py_in / fsize,
            'w':   pw_clipped / fsize,
            'h':   ph_clipped / fsize,
            'kx':  kx_in / fsize if kx_in is not None else None,
            'ky':  ky_in / fsize if ky_in is not None else None,
        })
    return result


def count_visible_labels(labels, fx, fy, fsize, src_size):
    """统计 frame 内可见标注数量（用于判断是否满足最小标注数）"""
    count = 0
    for lb in labels:
        px = lb['cx'] * src_size
        py = lb['cy'] * src_size
        pw = lb['w'] * src_size
        ph = lb['h'] * src_size

        clipped = clip_bbox_to_frame(px, py, pw, ph, fx, fy, fsize)
        if clipped is None:
            continue
        _, _, pw_clipped, ph_clipped = clipped
        if pw * ph > 0 and (pw_clipped * ph_clipped) / (pw * ph) >= MIN_AREA_RATIO:
            count += 1
    return count


# ══════════════════════════════════════════════════════════════════════
# 帧提取
# ══════════════════════════════════════════════════════════════════════

def extract_frame(padded_img, fx, fy, fsize, pad):
    """从 padded 图上提取一帧。

    Args:
        padded_img: 已 padding 的源图
        fx, fy:     frame 左上角在源图（未 padding）中的像素坐标
        fsize:      frame 尺寸
        pad:        padding 宽度

    返回 1024×1024 BGR 图像
    """
    x_start = pad + fx
    y_start = pad + fy
    return padded_img[y_start:y_start + fsize,
                      x_start:x_start + fsize].copy()




# ══════════════════════════════════════════════════════════════════════
# 候选位置采样
# ══════════════════════════════════════════════════════════════════════

def sample_candidate_positions(labels, src_size, fsize, pad):
    """混合采样候选 Frame 0 位置：网格 + 随机。

    约束 Frame 0 位置以确保 Frame 1/2 的滑动偏移不超出源图边界，
    避免进入镜像 padding 区域产生镜像伪影。
    同时考虑水平双向偏移（dx 可正可负）。

    Returns:
        list of (fx, fy): Frame 0 左上角在源图中的像素坐标
    """
    max_origin = src_size - fsize  # 1024

    # 帧间累积最大偏移量（双向：正方向和负方向都要考虑）
    max_cumul_neg_dx = abs(FRAME1_DX_MIN) + FRAME2_DX_MAX
    max_cumul_pos_dx = FRAME1_DX_MAX + FRAME2_DX_MAX
    # dy 单向（始终 ≥ 0），无需预留负方向空间
    max_cumul_dy = FRAME1_DY_MAX + FRAME2_DY_MAX

    # Frame 0 安全范围（防止任何方向的累积偏移越界）
    fx_min, fx_max = max_cumul_neg_dx, max_origin - max_cumul_pos_dx
    fy_min, fy_max = 0, max_origin - max_cumul_dy

    candidates = set()

    # 1. 网格采样（在安全范围内）
    for fy in range(fy_min, fy_max + 1, SLIDING_STRIDE):
        for fx in range(fx_min, fx_max + 1, SLIDING_STRIDE):
            candidates.add((fx, fy))

    # 角落也覆盖（安全范围内的角落）
    candidates.add((fx_min, fy_min))
    candidates.add((fx_max, fy_min))
    candidates.add((fx_min, fy_max))
    candidates.add((fx_max, fy_max))

    # 2. 随机补充
    n_random = MAX_ATTEMPTS_PER_IMAGE - len(candidates)
    for _ in range(max(0, n_random)):
        fx = random.randint(fx_min, fx_max)
        fy = random.randint(fy_min, fy_max)
        candidates.add((fx, fy))

    return sorted(candidates)


# ══════════════════════════════════════════════════════════════════════
# 单张源图处理
# ══════════════════════════════════════════════════════════════════════

def process_single_image(img, labels, samples_target, base_seq_idx, dst_dir):
    """处理一张源图，生成多组合格序列。

    Args:
        img:            resize 后的 2048×2048 BGR 图像
        labels:         源图归一化标注列表
        samples_target: 期望生成多少组序列
        base_seq_idx:   起始序列编号
        dst_dir:        输出目录

    Returns:
        (sequences_generated, next_seq_idx)
    """
    src_size = RESIZE_TARGET  # 2048
    fsize = FRAME_SIZE         # 1024
    pad = MIRROR_PADDING       # 384

    # 固定随机种子以保证复现（派生自序列编号）
    SEED = BASE_SEED + base_seq_idx * 100
    random.seed(SEED)
    np.random.seed(SEED)

    # Mirror padding
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                cv2.BORDER_REFLECT_101)

    # 候选位置
    candidates = sample_candidate_positions(labels, src_size, fsize, pad)

    # 筛选合格位置（Frame 0 可见标注 ≥ MIN_LABELS_FOR_KEYFRAME）
    valid_positions = []
    for (fx, fy) in candidates:
        n_visible = count_visible_labels(labels, fx, fy, fsize, src_size)
        if n_visible >= MIN_LABELS_FOR_KEYFRAME:
            valid_positions.append((fx, fy))

    # 随机打乱，取前 samples_target 个
    random.shuffle(valid_positions)
    selected = valid_positions[:samples_target]

    seq_idx = base_seq_idx
    for (fx, fy) in selected:
        # Frame 1 偏移（保证最小幅度，避免帧间几乎无位移）
        while True:
            dx1 = random.randint(FRAME1_DX_MIN, FRAME1_DX_MAX)
            if abs(dx1) >= DX_MIN_MAGNITUDE:
                break
        while True:
            dy1 = random.randint(FRAME1_DY_MIN, FRAME1_DY_MAX)
            if abs(dy1) >= DY_MIN_MAGNITUDE:
                break

        # Frame 2 偏移 — 必须与 Frame 1 同方向且位移更大
        dx_sign = 1 if dx1 >= 0 else -1
        dy_sign = 1 if dy1 >= 0 else -1
        dx2_add = random.randint(FRAME2_DX_MIN, FRAME2_DX_MAX)
        dy2_add = random.randint(FRAME2_DY_MIN, FRAME2_DY_MAX)
        dx2 = dx1 + dx_sign * dx2_add
        dy2 = dy1 + dy_sign * dy2_add

        # 提取三帧（每个帧在源图中位置不同）
        frame0_img = extract_frame(padded, fx,      fy,      fsize, pad)
        frame1_img = extract_frame(padded, fx + dx1, fy + dy1, fsize, pad)
        frame2_img = extract_frame(padded, fx + dx2, fy + dy2, fsize, pad)

        # 每帧独立映射可见标注
        labels_f0 = map_labels_to_frame(labels, fx,      fy,      fsize, src_size)
        labels_f1 = map_labels_to_frame(labels, fx + dx1, fy + dy1, fsize, src_size)
        labels_f2 = map_labels_to_frame(labels, fx + dx2, fy + dy2, fsize, src_size)

        # 过曝参数序列级预生成（三帧同一区域、同一亮度）
        oe_params = generate_overexposure_params(frame0_img)

        # ── 模糊预掷骰：保证至少有一帧清晰 ──
        blur_f0 = random.random() < BLUR_PROB_KEYFRAME
        blur_f1 = random.random() < BLUR_PROB_REFERENCE
        blur_f2 = random.random() < BLUR_PROB_REFERENCE
        if blur_f0 and blur_f1 and blur_f2:
            # 随机选一帧保持清晰
            unblur = random.randint(0, 2)
            if unblur == 0:
                blur_f0 = False
            elif unblur == 1:
                blur_f1 = False
            else:
                blur_f2 = False

        # 施加噪声（传入 blur_flag 控制模糊是否施加）
        frame0_noisy = apply_noises(frame0_img, 0, labels_f0, oe_params,
                                    blur_flag=blur_f0)
        frame1_noisy = apply_noises(frame1_img, 1, labels_f1, oe_params,
                                    blur_flag=blur_f1)
        frame2_noisy = apply_noises(frame2_img, 2, labels_f2, oe_params,
                                    blur_flag=blur_f2)

        # 保存
        seq_name = f"seq_{seq_idx:05d}"
        seq_dir = os.path.join(dst_dir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)

        cv2.imwrite(os.path.join(seq_dir, "frame_0.jpg"), frame0_noisy,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(seq_dir, "frame_1.jpg"), frame1_noisy,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(seq_dir, "frame_2.jpg"), frame2_noisy,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 保存三帧各自的标注
        save_labels(os.path.join(seq_dir, "labels_frame0.txt"), labels_f0)
        save_labels(os.path.join(seq_dir, "labels_frame1.txt"), labels_f1)
        save_labels(os.path.join(seq_dir, "labels_frame2.txt"), labels_f2)

        seq_idx += 1

    return seq_idx - base_seq_idx, seq_idx


# ══════════════════════════════════════════════════════════════════════
# 单集合处理
# ══════════════════════════════════════════════════════════════════════

def collect_images_from_dirs(src_dirs):
    """从多个 (images_dir, labels_dir) 中收集所有图片。

    Args:
        src_dirs: list of (images_dir, labels_dir) tuples

    Returns:
        dict: basename → (src_images_dir, src_labels_dir)
    """
    all_images = {}
    for img_dir, lbl_dir in src_dirs:
        for f in os.listdir(img_dir):
            if not f.endswith(('.jpg', '.png', '.JPG', '.PNG')):
                continue
            base = os.path.splitext(f)[0]
            if base not in all_images:
                all_images[base] = (img_dir, lbl_dir)
    return all_images


def process_split_from_pool(image_pool, dst_dir, samples_per_image,
                            split_name, global_seq_idx=1):
    """从图片池中处理指定集合。

    Args:
        image_pool: dict: basename → (src_images_dir, src_labels_dir)
        dst_dir:    输出目录
        samples_per_image: 每张源图生成几组序列
        split_name: 集合名称 (train/val/test)
        global_seq_idx: 起始序列编号

    Returns:
        (stats, next_seq_idx)
    """
    # 清理旧数据，防止多次运行混入陈旧序列
    import shutil
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    stats = {
        'total_source': 0,
        'total_sequences': 0,
        'total_skipped_no_label': 0,
        'total_skipped_few_labels': 0,
        'total_skipped_no_valid_pos': 0,
    }

    for base_name in sorted(image_pool.keys()):
        src_img_dir, src_lbl_dir = image_pool[base_name]

        # 找图片文件
        img_path = None
        for ext in ['.jpg', '.png', '.JPG', '.PNG']:
            candidate = os.path.join(src_img_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] 无法读取: {base_name}")
            continue

        label_path = os.path.join(src_lbl_dir, base_name + '.txt')
        labels = load_labels(label_path)

        stats['total_source'] += 1

        if len(labels) == 0:
            stats['total_skipped_no_label'] += 1
            continue
        if len(labels) < MIN_LABELS_FOR_KEYFRAME:
            stats['total_skipped_few_labels'] += 1
            continue

        # Resize 到 2048×2048
        if img.shape[0] != RESIZE_TARGET or img.shape[1] != RESIZE_TARGET:
            img = cv2.resize(img, (RESIZE_TARGET, RESIZE_TARGET),
                            interpolation=cv2.INTER_LINEAR)

        n_gen, global_seq_idx = process_single_image(
            img, labels, samples_per_image, global_seq_idx, dst_dir
        )

        if n_gen == 0:
            stats['total_skipped_no_valid_pos'] += 1
        else:
            stats['total_sequences'] += n_gen

        if stats['total_source'] % 10 == 0:
            print(f"  [{split_name}] 已处理 {stats['total_source']} 张源图, "
                  f"累计 {stats['total_sequences']} 序列")

    return stats, global_seq_idx


# ══════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════

def create_syn_visualization(output_dir, split_name, seq_dir):
    """随机抽样绘制序列可视化"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    seq_names = sorted([
        d for d in os.listdir(seq_dir)
        if os.path.isdir(os.path.join(seq_dir, d))
    ])
    np.random.seed(99)
    sample_count = min(SYN_VIZ_SAMPLE_COUNT, len(seq_names))
    if sample_count == 0:
        print("  [WARN] 没有序列可可视化")
        return
    sampled = np.random.choice(seq_names, sample_count, replace=False)

    fig, axes = plt.subplots(sample_count, 3, figsize=(15, 5 * sample_count))
    if sample_count == 1:
        axes = np.array([axes])

    for idx, seq_name in enumerate(sampled):
        seq_path = os.path.join(seq_dir, seq_name)
        for fi, frame_name in enumerate(['frame_0.jpg', 'frame_1.jpg',
                                          'frame_2.jpg']):
            ax = axes[idx, fi]
            frame_path = os.path.join(seq_path, frame_name)
            if os.path.exists(frame_path):
                frame_img = cv2.imread(frame_path)
                frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                ax.imshow(frame_img)

            # 加载该帧自己的标注
            label_file = f"labels_frame{fi}.txt"
            label_path = os.path.join(seq_path, label_file)
            lbs = load_labels(label_path)
            ax.set_title(f"{seq_name} | frame_{fi}\n({len(lbs)} labels)",
                         fontsize=9)

            for lb in lbs:
                px_t = lb['cx'] * FRAME_SIZE
                py_t = lb['cy'] * FRAME_SIZE
                pw_t = lb['w'] * FRAME_SIZE
                ph_t = lb['h'] * FRAME_SIZE
                rect = Rectangle((px_t - pw_t / 2, py_t - ph_t / 2),
                                 pw_t, ph_t,
                                 linewidth=1, edgecolor='red',
                                 facecolor='none')
                ax.add_patch(rect)
                if lb['kx'] is not None:
                    kx_t = lb['kx'] * FRAME_SIZE
                    ky_t = lb['ky'] * FRAME_SIZE
                    ax.plot(kx_t, ky_t, 'go', markersize=3)
            ax.axis('off')

    plt.tight_layout()
    viz_path = os.path.join(output_dir, f"syn_viz_{split_name}.png")
    plt.savefig(viz_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  可视化已保存: {viz_path}")


def create_noise_demo(output_dir, seq_dir):
    """噪声效果展示"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seq_names = sorted([
        d for d in os.listdir(seq_dir)
        if os.path.isdir(os.path.join(seq_dir, d))
    ])
    if not seq_names:
        return

    demo_seq = seq_names[0]
    frame_path = os.path.join(seq_dir, demo_seq, "frame_0.jpg")
    orig = cv2.imread(frame_path)
    if orig is None:
        return

    label_path = os.path.join(seq_dir, demo_seq, "labels_frame0.txt")
    demo_labels = load_labels(label_path)

    random.seed(777)
    np.random.seed(777)

    vib = apply_vibration_blur(orig.copy())
    over = apply_overexposure(orig.copy())
    rand_occ = apply_random_occlusion(orig.copy())
    lab_occ = apply_label_occlusion(orig.copy(), demo_labels,
                                     OCCLUDE_LABEL_RATIO_KEYFRAME)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titles = ['Original', 'Vibration Blur', 'Overexposure',
              'Random Occlusion', 'Label Occlusion']
    imgs = [orig, vib, over, rand_occ, lab_occ]

    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    viz_path = os.path.join(output_dir, "noise_demo.png")
    plt.savefig(viz_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  噪声展示已保存: {viz_path}")


# ══════════════════════════════════════════════════════════════════════
# YAML 配置
# ══════════════════════════════════════════════════════════════════════

def create_stage2_yaml(output_dir):
    yaml_path = os.path.join(output_dir, STAGE2_SLIDING_YAML)
    content = f"""# 滑动窗口多帧合成数据集
train: ./train
val: ./val
test: ./test

nc: 1
kpt_shape: [1, 2]
flip_idx: [0]
names: ['weed']

num_frames: 3
frame_size: {FRAME_SIZE}
min_labels_keyframe: {MIN_LABELS_FOR_KEYFRAME}
"""
    with open(yaml_path, 'w') as f:
        f.write(content)
    return yaml_path


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("第二阶段（滑动窗口版）：合成多帧数据集")
    print("  从 2048×2048 源图滑动窗口提取 3 帧序列")
    print("  每帧独立映射标注 | 每帧独立施加噪声")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════
    # 1. 合并全部源图 (train+val+test)，按来源分组后重新划分
    #    防止同一视频帧/连拍照片的内容级数据泄露
    # ═══════════════════════════════════════════════════════════════
    print("\n[数据划分] 合并 train+val+test → 按来源分组 → 组级划分...")
    all_images = collect_images_from_dirs([
        (SRC_TRAIN_IMAGES, SRC_TRAIN_LABELS),
        (SRC_VAL_IMAGES,   SRC_VAL_LABELS),
        (SRC_TEST_IMAGES,  SRC_TEST_LABELS),
    ])
    print(f"  合并 train+val+test 去重后: {len(all_images)} 张源图")

    # 按 group_key 分组
    groups = {}  # group_key → list of basenames
    for basename in sorted(all_images.keys()):
        gk = get_group_key(basename)
        groups.setdefault(gk, []).append(basename)

    n_groups = len(groups)
    n_imgs = sum(len(v) for v in groups.values())
    assert n_imgs == len(all_images), "分组后图片数不一致"
    print(f"  分组结果: {n_groups} 个独立组 (视频/连拍/单张)")

    # 按组类型分层划分，按合格源图数（≥4标注）分配
    # Video (1组72张合格) → train
    # Session: 按合格图片数降序: 45, 27, 22, 19, 10, 3
    #   train: 45+27+3=75 | val: 22+10=32 | test: 19
    # Single (15张均合格): 5/5/5
    group_keys = sorted(groups.keys())
    np.random.seed(42)

    video_groups  = sorted(gk for gk in group_keys if gk.startswith('video_'))
    session_groups = sorted(gk for gk in group_keys if gk.startswith('session_'))
    single_groups  = sorted(gk for gk in group_keys if gk.startswith('single_'))

    print(f"  组类型: video={len(video_groups)}, session={len(session_groups)}, single={len(single_groups)}")

    # Video: 全部放入 train
    train_groups = set(video_groups)

    # Session: 按大小分配 (已按字母序, 实际大小: 45,27,22,19,10,3)
    sess_sorted = sorted(session_groups, key=lambda gk: len(groups[gk]), reverse=True)
    train_groups.update([sess_sorted[0], sess_sorted[1], sess_sorted[5]])  # 45+27+3=75合格
    val_groups   = {sess_sorted[2], sess_sorted[4]}                         # 22+10=32合格
    test_groups  = {sess_sorted[3]}                                         # 19合格

    # Single (15组): 5 train, 5 val, 5 test
    perm_sg = np.random.permutation(len(single_groups))
    train_groups.update(single_groups[i] for i in perm_sg[:5])
    val_groups.update(single_groups[i] for i in perm_sg[5:10])
    test_groups.update(single_groups[i] for i in perm_sg[10:15])

    # 构建各 split 的图片池
    train_pool = {}
    val_pool   = {}
    test_pool  = {}
    for gk in train_groups:
        for b in groups[gk]:
            train_pool[b] = all_images[b]
    for gk in val_groups:
        for b in groups[gk]:
            val_pool[b] = all_images[b]
    for gk in test_groups:
        for b in groups[gk]:
            test_pool[b] = all_images[b]

    print(f"  划分结果: train {len(train_pool)} 图 ({len(train_groups)} 组) / "
          f"val {len(val_pool)} 图 ({len(val_groups)} 组) / "
          f"test {len(test_pool)} 图 ({len(test_groups)} 组)")

    # 防御性检查：图片级 & 组级均无重叠
    assert len(set(train_pool) & set(val_pool)) == 0, \
        "BUG: train/val 图片重叠"
    assert len(set(train_pool) & set(test_pool)) == 0, \
        "BUG: train/test 图片重叠"
    assert len(set(val_pool) & set(test_pool)) == 0, \
        "BUG: val/test 图片重叠"
    assert train_groups.isdisjoint(val_groups), \
        "BUG: train/val 组重叠"
    assert train_groups.isdisjoint(test_groups), \
        "BUG: train/test 组重叠"
    assert val_groups.isdisjoint(test_groups), \
        "BUG: val/test 组重叠"
    print("  ✓ 去重检查通过: train/val/test 图片级 & 组级均无重叠")

    # ═══════════════════════════════════════════════════════════════
    # 4. 处理各集合
    # ═══════════════════════════════════════════════════════════════

    # ---- Train ----
    print("\n[Train 滑动窗口多帧合成]")
    train_dst = os.path.join(STAGE2_SLIDING_OUT, "train")
    train_stats, next_seq = process_split_from_pool(
        train_pool, train_dst,
        samples_per_image=SLIDING_SAMPLES_TRAIN,
        split_name="train",
        global_seq_idx=1,
    )
    print(f"\n  Train 统计:")
    print(f"    遍历源图: {train_stats['total_source']}")
    print(f"    生成序列: {train_stats['total_sequences']}")
    print(f"    跳过(无标注): {train_stats['total_skipped_no_label']}")
    print(f"    跳过(标注<{MIN_LABELS_FOR_KEYFRAME}): {train_stats['total_skipped_few_labels']}")
    print(f"    跳过(无合格窗口): {train_stats['total_skipped_no_valid_pos']}")

    # ---- Val ----
    print("\n[Val 滑动窗口多帧合成]")
    val_dst = os.path.join(STAGE2_SLIDING_OUT, "val")
    val_stats, next_seq = process_split_from_pool(
        val_pool, val_dst,
        samples_per_image=SLIDING_SAMPLES_VAL,
        split_name="val",
        global_seq_idx=next_seq,
    )
    print(f"\n  Val 统计:")
    print(f"    遍历源图: {val_stats['total_source']}")
    print(f"    生成序列: {val_stats['total_sequences']}")
    print(f"    跳过(无标注): {val_stats['total_skipped_no_label']}")
    print(f"    跳过(标注<{MIN_LABELS_FOR_KEYFRAME}): {val_stats['total_skipped_few_labels']}")
    print(f"    跳过(无合格窗口): {val_stats['total_skipped_no_valid_pos']}")

    # ---- Test ----
    print("\n[Test 滑动窗口多帧合成]")
    test_dst = os.path.join(STAGE2_SLIDING_OUT, "test")
    test_stats, _ = process_split_from_pool(
        test_pool, test_dst,
        samples_per_image=SLIDING_SAMPLES_TEST,
        split_name="test",
        global_seq_idx=next_seq,
    )
    print(f"\n  Test 统计:")
    print(f"    遍历源图: {test_stats['total_source']}")
    print(f"    生成序列: {test_stats['total_sequences']}")
    print(f"    跳过(无标注): {test_stats['total_skipped_no_label']}")
    print(f"    跳过(标注<{MIN_LABELS_FOR_KEYFRAME}): {test_stats['total_skipped_few_labels']}")
    print(f"    跳过(无合格窗口): {test_stats['total_skipped_no_valid_pos']}")

    # ---- 汇总 ----
    total_seq = (train_stats['total_sequences'] +
                 val_stats['total_sequences'] +
                 test_stats['total_sequences'])
    total_skipped = (train_stats['total_skipped_no_label'] +
                     train_stats['total_skipped_few_labels'] +
                     train_stats['total_skipped_no_valid_pos'] +
                     val_stats['total_skipped_no_label'] +
                     val_stats['total_skipped_few_labels'] +
                     val_stats['total_skipped_no_valid_pos'] +
                     test_stats['total_skipped_no_label'] +
                     test_stats['total_skipped_few_labels'] +
                     test_stats['total_skipped_no_valid_pos'])
    print(f"\n{'='*60}")
    print(f"  汇总: {total_seq} 个序列 ({total_seq * 3} 帧)")
    print(f"  跳过源图: {total_skipped}")
    print(f"{'='*60}")

    # YAML
    yaml_path = create_stage2_yaml(STAGE2_SLIDING_OUT)
    print(f"\n  YAML 已保存: {yaml_path}")

    # 可视化
    print("\n[生成可视化]")
    create_syn_visualization(STAGE2_SLIDING_OUT, "train", train_dst)
    create_noise_demo(STAGE2_SLIDING_OUT, train_dst)

    print("\n第二阶段（滑动窗口版）完成！")


if __name__ == "__main__":
    main()
