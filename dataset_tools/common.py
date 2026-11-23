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


# 自然遮挡 patch bank / 序列级遮挡模式 (由 data builder 设置)
PATCH_BANK = None
CURRENT_OCC_MODE = 'all'


def set_patch_bank(bank):
    global PATCH_BANK
    PATCH_BANK = bank


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


def _pick_patch():
    """从训练 patch bank 取一块自然纹理；bank 未设置时用随机噪声兜底."""
    if PATCH_BANK is not None and len(PATCH_BANK):
        return PATCH_BANK[random.randrange(len(PATCH_BANK))]
    base = np.random.randint(40, 200, (128, 128, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (5, 5), 1.0)


def _shape_mask(h, w, shape):
    mask = np.zeros((h, w), np.uint8)
    if shape == 'poly':
        n = random.randint(5, 9)
        pts = []
        for i in range(n):
            ang = 2 * np.pi * i / n + random.uniform(-0.15, 0.15)
            r = random.uniform(0.35, 0.5)
            pts.append([int(w / 2 + r * w * np.cos(ang)),
                        int(h / 2 + r * h * np.sin(ang))])
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    else:
        cv2.ellipse(mask, (w // 2, h // 2), (max(1, w // 2), max(1, h // 2)),
                    0, 0, 360, 255, -1)
    return cv2.GaussianBlur(mask, (5, 5), 1.0)


def _paste_patch(img, cx, cy, out_w, out_h, angle=0, opacity=1.0):
    """把自然 patch 以椭圆/多边形 mask 贴到 (cx,cy). 返回 img."""
    h, w = img.shape[:2]
    out_w = int(max(8, min(out_w, w - 2)))
    out_h = int(max(8, min(out_h, h - 2)))
    # 用一个带旋转的外接矩形画 mask
    pad = int(np.ceil(np.hypot(out_w, out_h) / 2)) + 3
    cx_i, cy_i = int(round(cx)), int(round(cy))
    x1, y1 = cx_i - pad, cy_i - pad
    x2, y2 = cx_i + pad, cy_i + pad
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return img
    mh, mw = y2c - y1c, x2c - x1c
    local = np.zeros((mh, mw), np.uint8)
    lcx, lcy = cx_i - x1c, cy_i - y1c
    shape = 'poly' if random.random() < 0.35 else 'ellipse'
    if shape == 'poly':
        n = random.randint(5, 9)
        pts = []
        for i in range(n):
            ang = 2 * np.pi * i / n + random.uniform(-0.2, 0.2)
            rr = random.uniform(0.32, 0.5)
            pts.append([int(lcx + rr * out_w * np.cos(ang)),
                        int(lcy + rr * out_h * np.sin(ang))])
        cv2.fillPoly(local, [np.array(pts, np.int32)], 255)
    else:
        cv2.ellipse(local, (int(lcx), int(lcy)),
                    (max(1, out_w // 2), max(1, out_h // 2)),
                    angle, 0, 360, 255, -1)
    local = cv2.GaussianBlur(local, (5, 5), 1.0)
    patch = _pick_patch()
    p = cv2.resize(patch, (mw, mh), interpolation=cv2.INTER_AREA)
    roi = img[y1c:y2c, x1c:x2c].astype(np.float32)
    a = (local.astype(np.float32) / 255.0)[..., None] * float(opacity)
    img[y1c:y2c, x1c:x2c] = np.clip(
        roi * (1 - a) + p.astype(np.float32) * a, 0, 255).astype(np.uint8)
    return img


def apply_random_occlusion(img):
    """画面随机位置自然纹理遮挡."""
    h, w = img.shape[:2]
    n_occlude = random.randint(OCCLUSION_COUNT_MIN, OCCLUSION_COUNT_MAX)
    for _ in range(n_occlude):
        size_w = random.randint(max(8, OCCLUSION_SIZE_MIN),
                                max(9, OCCLUSION_SIZE_MAX))
        size_h = random.randint(max(8, OCCLUSION_SIZE_MIN),
                                max(9, OCCLUSION_SIZE_MAX))
        if size_w >= w or size_h >= h:
            continue
        cx = random.uniform(size_w / 2, w - size_w / 2)
        cy = random.uniform(size_h / 2, h - size_h / 2)
        img = _paste_patch(img, cx, cy, size_w, size_h,
                           angle=random.uniform(0, 180),
                           opacity=random.uniform(0.85, 1.0))
    return img


def apply_label_occlusion(img, labels, occlusion_ratio, area_ratio=None,
                          max_size=None, short_side_factor=0.8):
    """基于真实纹理的标注区域遮挡.

    混合三种放置:
      - centered patch (目标框中心)
      - offset stripe (中心偏离关键点 20~40px, 细长/椭圆仍覆盖关键点)
      - 框内随机 patch
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
    n_select = int(np.ceil(len(labels) * occlusion_ratio))
    if n_select <= 0:
        return img
    selected = random.sample(labels, min(n_select, len(labels)))
    for lb in selected:
        cx = lb['cx'] * w
        cy = lb['cy'] * h
        bw = max(2.0, lb['w'] * w)
        bh = max(2.0, lb['h'] * h)
        kx = lb.get('kx')
        ky = lb.get('ky')
        kx = kx * w if kx is not None else cx
        ky = ky * h if ky is not None else cy
        base = int(np.sqrt(bw * bh * area_ratio))
        mode = random.random()
        if mode < 0.28 and kx is not None:
            # offset stripe: 中心 20~40px 偏离关键点，长轴穿过关键点
            d = random.uniform(20, 40)
            theta = random.uniform(0, 2 * np.pi)
            ow = random.uniform(2 * d + 20, 2 * d + 60)
            oh = random.uniform(12, 28)
            ocx = kx + d * np.cos(theta)
            ocy = ky + d * np.sin(theta)
            angle = np.degrees(theta)
        elif mode < 0.64:
            # centered-ish patch
            ow = base
            oh = base
            angle = random.uniform(0, 180)
            ocx = cx + random.uniform(-max(1, bw / 2 - base / 2),
                                      max(1, bw / 2 - base / 2))
            ocy = cy + random.uniform(-max(1, bh / 2 - base / 2),
                                      max(1, bh / 2 - base / 2))
        else:
            # random inside box
            ow = random.uniform(0.45, 1.1) * min(bw, max(12, base))
            oh = random.uniform(0.45, 1.1) * min(bh, max(12, base))
            ow = max(8, min(ow, max_size, min(bw, bh) * short_side_factor))
            oh = max(8, min(oh, max_size, min(bw, bh) * short_side_factor))
            angle = random.uniform(0, 180)
            ocx = random.uniform(cx - bw / 2, cx + bw / 2)
            ocy = random.uniform(cy - bh / 2, cy + bh / 2)
        if mode < 0.28:
            ow = min(ow, 160); oh = min(oh, 60)
        else:
            ow = min(ow, max_size); oh = min(oh, max_size)
        img = _paste_patch(img, ocx, ocy, ow, oh, angle=angle,
                           opacity=random.uniform(0.8, 1.0))
    return img


def apply_background_occlusion(img, labels):
    """在非标注区域随机放自然纹理遮挡 (不贴整株植物, 只贴纹理块)."""
    h, w = img.shape[:2]
    n_target = random.randint(BG_OCCLUDE_COUNT_MIN, BG_OCCLUDE_COUNT_MAX)
    label_rects = []
    for lb in labels:
        cx = lb['cx'] * w
        cy = lb['cy'] * h
        bw = lb['w'] * w
        bh = lb['h'] * h
        label_rects.append({'x1': cx - bw / 2, 'y1': cy - bh / 2,
                            'x2': cx + bw / 2, 'y2': cy + bh / 2})

    def inside(x, y, size):
        cxb, cyb = x + size / 2, y + size / 2
        return any(r['x1'] <= cxb <= r['x2'] and r['y1'] <= cyb <= r['y2']
                   for r in label_rects)

    placed = 0
    for _ in range(n_target * 10):
        if placed >= n_target:
            break
        size_w = random.randint(max(8, BG_OCCLUDE_SIZE_MIN),
                                max(9, BG_OCCLUDE_SIZE_MAX))
        size_h = random.randint(max(8, BG_OCCLUDE_SIZE_MIN),
                                max(9, BG_OCCLUDE_SIZE_MAX))
        if size_w >= w or size_h >= h:
            continue
        cx = random.uniform(size_w / 2, w - size_w / 2)
        cy = random.uniform(size_h / 2, h - size_h / 2)
        if inside(cx - size_w / 2, cy - size_h / 2, min(size_w, size_h)):
            continue
        img = _paste_patch(img, cx, cy, size_w, size_h,
                           angle=random.uniform(0, 180),
                           opacity=random.uniform(0.8, 1.0))
        placed += 1
    return img


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

    # ── 3. 遮挡（受序列级 CURRENT_OCC_MODE 控制）──
    mode = CURRENT_OCC_MODE
    occ_on = (mode == 'all' or (mode == 'key_only' and frame_idx == 0)
              or (mode == 'refs_only' and frame_idx > 0))
    if not occ_on:
        return img

    rand_occ_prob = RANDOM_OCC_PROB_KEYFRAME if frame_idx == 0 else RANDOM_OCC_PROB_REFERENCE
    if random.random() < rand_occ_prob:
        img = apply_random_occlusion(img)

    # ── 4. 标注区域遮挡（按配置比例选中标注框）──
    label_occ_ratio = OCCLUDE_LABEL_RATIO_KEYFRAME if frame_idx == 0 else OCCLUDE_LABEL_RATIO_REFERENCE
    img = apply_label_occlusion(img, labels, label_occ_ratio,
                                area_ratio=label_occ_area_ratio,
                                max_size=label_occ_max_size,
                                short_side_factor=label_occ_short_factor)

    # ── 5. 背景区域自然纹理遮挡 ──
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
