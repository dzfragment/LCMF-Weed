"""
数据集处理共享工具模块
标注加载/保存 + 噪声函数 + 源图分组
"""

import os
import re
import cv2
import numpy as np
import random

from config import (
    # 振动失焦
    BLUR_PROB_KEYFRAME, BLUR_PROB_REFERENCE,
    GAUSS_KERNEL_MIN, GAUSS_KERNEL_MAX,
    GAUSS_SIGMA_MIN, GAUSS_SIGMA_MAX,
    MOTION_LEN_MIN, MOTION_LEN_MAX,
    # 过曝
    OVEREXPOSE_PROB,
    OVEREXPOSE_COUNT_MIN, OVEREXPOSE_COUNT_MAX,
    OVEREXPOSE_AREA_MIN, OVEREXPOSE_AREA_MAX,
    OVEREXPOSE_BRIGHT_MIN, OVEREXPOSE_BRIGHT_MAX,
    OVEREXPOSE_SAT_MIN, OVEREXPOSE_SAT_MAX,
    # 随机遮挡
    RANDOM_OCC_PROB_KEYFRAME, RANDOM_OCC_PROB_REFERENCE,
    OCCLUSION_COUNT_MIN, OCCLUSION_COUNT_MAX,
    OCCLUSION_SIZE_MIN, OCCLUSION_SIZE_MAX,
    OCCLUSION_ALPHA_MIN, OCCLUSION_ALPHA_MAX,
    # 标注区域遮挡
    OCCLUDE_LABEL_RATIO_KEYFRAME,
    OCCLUDE_LABEL_RATIO_REFERENCE,
    OCCLUDE_BBOX_AREA_RATIO,
    OCCLUDE_SIZE_MAX,
    # 背景区域遮挡
    BG_OCCLUDE_COUNT_MIN, BG_OCCLUDE_COUNT_MAX,
    BG_OCCLUDE_SIZE_MIN, BG_OCCLUDE_SIZE_MAX,
)


# ══════════════════════════════════════════════════════════════════════
# 标注加载 / 保存
# ══════════════════════════════════════════════════════════════════════

def load_labels(label_path):
    """加载 YOLO 格式标注文件。
    格式: class_id cx cy w h [kpt_x kpt_y]
    返回 list of dict"""
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            kx, ky = None, None
            if len(parts) >= 7:
                kx, ky = float(parts[5]), float(parts[6])
            labels.append({'cls': cls, 'cx': cx, 'cy': cy, 'w': w, 'h': h,
                           'kx': kx, 'ky': ky})
    return labels


def save_labels(label_path, labels):
    """保存 YOLO 格式标注"""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w') as f:
        for lb in labels:
            parts = [str(lb['cls']),
                     f"{lb['cx']:.6f}", f"{lb['cy']:.6f}",
                     f"{lb['w']:.6f}", f"{lb['h']:.6f}"]
            if lb['kx'] is not None and lb['ky'] is not None:
                parts.append(f"{lb['kx']:.6f}")
                parts.append(f"{lb['ky']:.6f}")
            f.write(' '.join(parts) + '\n')


# ══════════════════════════════════════════════════════════════════════
# 噪声函数
# ══════════════════════════════════════════════════════════════════════

def gaussian_blur_only(img):
    """仅高斯模糊（用于 Frame 0 失焦，不含运动模糊）"""
    ksize = random.choice([k for k in range(GAUSS_KERNEL_MIN,
                           GAUSS_KERNEL_MAX + 1, 2)])
    sigma = random.uniform(GAUSS_SIGMA_MIN, GAUSS_SIGMA_MAX)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def motion_blur(img, length, angle_deg):
    """定向运动模糊"""
    angle_rad = np.deg2rad(angle_deg)
    kernel_size = max(3, length)
    kernel = np.zeros((kernel_size, kernel_size))
    center = (kernel_size - 1) / 2
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    for i in range(length):
        x = center + (i - length / 2) * cos_a
        y = center + (i - length / 2) * sin_a
        xi = int(np.clip(np.round(x), 0, kernel_size - 1))
        yi = int(np.clip(np.round(y), 0, kernel_size - 1))
        kernel[yi, xi] = 1
    kernel = kernel / (kernel.sum() + 1e-8)
    return cv2.filter2D(img, -1, kernel)


def apply_vibration_blur(img):
    """振动失焦：高斯模糊 + 定向运动模糊 (仅 Frame 1/2)"""
    img = gaussian_blur_only(img)
    length = random.randint(MOTION_LEN_MIN, MOTION_LEN_MAX)
    angle = random.uniform(0, 180)
    img = motion_blur(img, length, angle)
    return img


def generate_overexposure_params(img):
    """预生成过曝参数（序列级，三帧共享同一组参数）。

    Returns:
        dict 或 None（不做过曝时返回 None，概率由调用方控制）
    """
    h, w = img.shape[:2]
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)
    target_area_ratio = random.uniform(OVEREXPOSE_AREA_MIN, OVEREXPOSE_AREA_MAX)
    target_area = target_area_ratio * w * h
    ratio = random.uniform(0.5, 2.0)
    b = int(np.sqrt(target_area / (np.pi * ratio)))
    a = int(b * ratio)
    a = max(10, min(a, w // 2))
    b = max(10, min(b, h // 2))
    bright_factor = random.uniform(OVEREXPOSE_BRIGHT_MIN, OVEREXPOSE_BRIGHT_MAX)
    sat_factor = random.uniform(OVEREXPOSE_SAT_MIN, OVEREXPOSE_SAT_MAX)
    return {'cx': cx, 'cy': cy, 'a': a, 'b': b,
            'bright_factor': bright_factor, 'sat_factor': sat_factor}


def apply_overexposure(img, params=None):
    """局部过曝。

    Args:
        img:    输入图像
        params: 预生成的参数 dict（由 generate_overexposure_params 产出）。
                为 None 时随机生成（向后兼容可视化等独立调用）。
    """
    h, w = img.shape[:2]
    if params is None:
        params = generate_overexposure_params(img)

    cx, cy = params['cx'], params['cy']
    a, b = params['a'], params['b']
    bright_factor = params['bright_factor']
    sat_factor = params['sat_factor']

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (a, b), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask,
                            (max(3, min(a, b) // 2 * 2 + 1),) * 2,
                            a / 4)
    mask_f = mask.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + (bright_factor - 1.0) * mask_f)
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + (sat_factor - 1.0) * mask_f)
    hsv = np.clip(hsv, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img


def apply_random_occlusion(img):
    """画面随机位置杂草遮挡"""
    h, w = img.shape[:2]
    n_occlude = random.randint(OCCLUSION_COUNT_MIN, OCCLUSION_COUNT_MAX)
    result = img.copy().astype(np.float32)
    for _ in range(n_occlude):
        size = random.randint(OCCLUSION_SIZE_MIN, OCCLUSION_SIZE_MAX)
        alpha = random.uniform(OCCLUSION_ALPHA_MIN, OCCLUSION_ALPHA_MAX)
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        color = (0, random.randint(60, 180), 0)
        patch = np.ones((size, size, 3), dtype=np.float32)
        for c in range(3):
            patch[:, :, c] = color[c]
        roi = result[y:y + size, x:x + size]
        result[y:y + size, x:x + size] = (1 - alpha) * roi + alpha * patch
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_label_occlusion(img, labels, occlusion_ratio, area_ratio=None,
                          max_size=None, short_side_factor=0.8):
    """基于标注框的杂草遮挡：
    按 occlusion_ratio 比例随机选中若干标注框，
    在每个选中框内放置覆盖约 area_ratio(默认 OCCLUDE_BBOX_AREA_RATIO)面积的遮挡块。

    Args:
        img: 输入图像
        labels: 标注列表
        occlusion_ratio: 选中多少比例的标注框
        area_ratio: 遮挡块占标注框面积比 (默认模块常量)
        max_size: 遮挡块最大边长 (默认 OCCLUDE_SIZE_MAX)
        short_side_factor: 遮挡块边长相对标注框短边的上限系数
                           (0.8=老行为; 1.0=允许占满短边, 配合高覆盖率)
    """
    if area_ratio is None:
        area_ratio = OCCLUDE_BBOX_AREA_RATIO
    if max_size is None:
        max_size = OCCLUDE_SIZE_MAX
    if short_side_factor is None:
        short_side_factor = 0.8

    if len(labels) == 0:
        return img

    h, w = img.shape[:2]
    result = img.copy().astype(np.float32)

    n_select = int(np.ceil(len(labels) * occlusion_ratio))
    if n_select <= 0:
        return img
    selected = random.sample(labels, min(n_select, len(labels)))

    for lb in selected:
        # 标注框像素坐标
        cx_px = lb['cx'] * w
        cy_px = lb['cy'] * h
        bw_px = lb['w'] * w
        bh_px = lb['h'] * h

        # 遮挡块尺寸：占标注框面积的约 area_ratio
        oc_size = int(np.sqrt(bw_px * bh_px * area_ratio))
        oc_size = max(8, min(oc_size, max_size,
                           int(min(bw_px, bh_px) * short_side_factor)))

        # 在标注框内随机放置
        x_min = max(0, int(cx_px - bw_px / 2))
        y_min = max(0, int(cy_px - bh_px / 2))
        x_max = min(w - oc_size, int(cx_px + bw_px / 2) - oc_size)
        y_max = min(h - oc_size, int(cy_px + bh_px / 2) - oc_size)

        if x_max <= x_min or y_max <= y_min:
            continue

        ox = random.randint(x_min, x_max)
        oy = random.randint(y_min, y_max)

        color = (0, random.randint(60, 180), 0)

        result[oy:oy + oc_size, ox:ox + oc_size] = color

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_background_occlusion(img, labels):
    """在非标注区域随机放置不透明绿色遮挡块。

    模拟未被标注的杂草（false negative）——模型不应假设每个绿色块都有标注，
    迫使模型学习真正的杂草视觉特征而非依赖"绿色=杂草"的捷径。
    遮挡块中心不落在任何标注框内，确保不破坏已有的正样本标注。
    """
    h, w = img.shape[:2]
    result = img.copy()

    n_target = random.randint(BG_OCCLUDE_COUNT_MIN, BG_OCCLUDE_COUNT_MAX)

    # 预先计算每个标注框的像素边界，用于碰撞检测
    label_rects = []
    for lb in labels:
        cx = lb['cx'] * w
        cy = lb['cy'] * h
        bw = lb['w'] * w
        bh = lb['h'] * h
        label_rects.append({
            'x1': int(cx - bw / 2), 'y1': int(cy - bh / 2),
            'x2': int(cx + bw / 2), 'y2': int(cy + bh / 2),
        })

    def _is_inside_any_label(x, y, size):
        """遮挡块中心是否落入任一标注框"""
        cx_b = x + size / 2
        cy_b = y + size / 2
        for r in label_rects:
            if r['x1'] <= cx_b <= r['x2'] and r['y1'] <= cy_b <= r['y2']:
                return True
        return False

    placed = 0
    max_attempts = n_target * 10
    for _ in range(max_attempts):
        if placed >= n_target:
            break
        size = random.randint(BG_OCCLUDE_SIZE_MIN, BG_OCCLUDE_SIZE_MAX)
        if size >= w or size >= h:
            continue
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)

        if not _is_inside_any_label(x, y, size):
            # BGR 绿色，留一定色度变化增加多样性
            color = (0, random.randint(60, 180), 0)
            result[y:y + size, x:x + size] = color
            placed += 1

    return result


def apply_vignette(img):
    """暗角效果"""
    h, w = img.shape[:2]
    edge_factor = random.uniform(VIGNETTE_EDGE_MIN, VIGNETTE_EDGE_MAX)
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist
    mask = 1.0 - (1.0 - edge_factor) * np.clip(dist, 0, 1) ** 2
    result = img.astype(np.float32)
    for c in range(3):
        result[:, :, c] *= mask
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_noises(img, frame_idx, labels, oe_params=None, blur_flag=None,
                 label_occ_area_ratio=None, label_occ_max_size=None,
                 label_occ_short_factor=None):
    """各噪声类型独立随机施加，不再从池子抽选。

    Args:
        img:       帧图像
        frame_idx: 0=关键帧(高斯模糊), 1/2=参考帧(振动失焦)
        labels:    该帧的标注列表（用于标注区域遮挡）
        oe_params: 预生成的过曝参数（序列级，三帧共用）
        blur_flag: 是否施加模糊；None 时自行随机决定，
                   True/False 时强制施加/不施加（用于"至少一帧清晰"约束）
        label_occ_area_ratio: 标注区域遮挡块覆盖率(占标注框面积比)；
                   None 用模块默认
        label_occ_max_size: 遮挡块最大边长; None 用模块默认
        label_occ_short_factor: 块边长相对框短边系数; None 用模块默认 0.8
    """
    # ── 1. 模糊（独立概率，无暗角）──
    if frame_idx == 0:
        blur_prob = BLUR_PROB_KEYFRAME
        blur_fn = gaussian_blur_only
    else:
        blur_prob = BLUR_PROB_REFERENCE
        blur_fn = apply_vibration_blur

    if blur_flag is None:
        blur_flag = random.random() < blur_prob

    if blur_flag:
        img = blur_fn(img)

    # ── 2. 过曝（独立概率 50%）──
    if random.random() < OVEREXPOSE_PROB:
        img = apply_overexposure(img, oe_params)

    # ── 3. 随机位置杂草遮挡（独立概率 Frame0:30%, Frame1/2:15%）──
    rand_occ_prob = RANDOM_OCC_PROB_KEYFRAME if frame_idx == 0 else RANDOM_OCC_PROB_REFERENCE
    if random.random() < rand_occ_prob:
        img = apply_random_occlusion(img)

    # ── 4. 标注区域遮挡（始终执行，按配置比例选中标注框）──
    label_occ_ratio = OCCLUDE_LABEL_RATIO_KEYFRAME if frame_idx == 0 else OCCLUDE_LABEL_RATIO_REFERENCE
    img = apply_label_occlusion(img, labels, label_occ_ratio,
                                area_ratio=label_occ_area_ratio,
                                max_size=label_occ_max_size,
                                short_side_factor=label_occ_short_factor)

    # ── 5. 背景区域不透明遮挡（始终执行）──
    img = apply_background_occlusion(img, labels)

    return img


# ══════════════════════════════════════════════════════════════════════
# 源图分组（防止数据泄露）
# ══════════════════════════════════════════════════════════════════════

def get_group_key(basename):
    """返回源图的组标识符。同组图片必须进入同一个 train/val/test split。

    分组规则（v6 修复）：
      - frame_XX-MM-SS-FF  →  同视频为一组 (video_frame_XX)
                              修复：v5 按分钟分组导致同一视频的连续分钟散布在
                              train/val/test 中，边界帧仅差 9 秒造成严重泄露。
      - image_YYYYMMDD_HHMMSS_NNNN  →  同日田间拍摄为一组 (session_YYYYMMDD)
                              修复：v5 按小时+分钟分组导致同一田间拍摄的连拍照片
                              散布在不同集中。
      - 其他  →  各自独立 (single_<basename>)

    Args:
        basename: 文件名（不含扩展名）

    Returns:
        str: 组标识符
    """
    # 视频帧: frame_00-03-24 → video_frame_00 (整段视频一组)
    m = re.match(r'(frame_\d{2})-\d{2}-\d{2}$', basename)
    if m:
        return f'video_{m.group(1)}'

    # 连拍: image_20251110_162504_0055 → session_20251110 (同日田间拍摄一组)
    m = re.match(r'image_(\d{8})_\d{6}_\d{4}$', basename)
    if m:
        return f'session_{m.group(1)}'

    # CropAndWeed (public_weed_le256): ave-XXXX-YYYY / vwg-XXXX-YYYY
    # XXXX = recording session, YYYY = image id → 同 session 一组
    m = re.match(r'(ave|vwg)-\d{4}-\d{4}$', basename)
    if m:
        return f'cropandweed_{m.group(1)}'

    # 单张: 各自独立
    return f'single_{basename}'
