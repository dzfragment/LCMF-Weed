"""
FFTAlign 配套工具函数：单帧对齐、链式对齐、特征空间平移。

这些函数只依赖 fft_align.core，不依赖任何模型/训练代码。
"""

import torch
import torch.nn.functional as F

from .core import FFTAlign


def align_frame(ref, mov, downsample=8, do_mse_check=True):
    """
    单帧对齐：mov → ref。

    ref, mov: (H, W, 3) numpy 或 (3, H, W) tensor
    返回: (aligned, dx, dy, snr)
        aligned — (3, H, W) tensor；dx, dy — 全分辨率整数位移；snr — 相关峰 SNR
    """
    import numpy as np

    # 统一转为 tensor
    if isinstance(ref, np.ndarray):
        if ref.ndim == 3 and ref.shape[-1] == 3:  # HWC → CHW
            ref = torch.from_numpy(ref.transpose(2, 0, 1).copy())
        ref = ref.float() / 255.0 if ref.max() > 1.0 else ref.float()
    if isinstance(mov, np.ndarray):
        if mov.ndim == 3 and mov.shape[-1] == 3:
            mov = torch.from_numpy(mov.transpose(2, 0, 1).copy())
        mov = mov.float() / 255.0 if mov.max() > 1.0 else mov.float()

    ref = ref.unsqueeze(0)
    mov = mov.unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aligner = FFTAlign(downsample=downsample, do_mse_check=do_mse_check).to(device)
    aligned, dx, dy, snr = aligner(mov.to(device), ref.to(device), return_displacement=True)

    return (aligned.squeeze(0).cpu(),
            int(dx.item()), int(dy.item()),
            snr[0])


def chain_align_frames(key, ref1, ref2, downsample=8, do_mse_check=True):
    """
    链式对齐三帧（生产管线 chain_align=True 时的图像层对齐方式）。

    ref1 → key（直接）
    ref2 → ref1 → key（链式：先对 ref1 求相对位移，再累加）

    key, ref1, ref2: (H, W, 3) numpy 或 (3, H, W) tensor
    返回: (aligned_ref1, aligned_ref2, (dx1, dy1), (dx2, dy2), (snr1, snr2))
        dx2/dy2 为 ref2 → key 的累加位移，snr2 为 ref2 → ref1 的相对对齐 SNR
    """
    import numpy as np

    def _to_tensor(img):
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[-1] == 3:
                img = torch.from_numpy(img.transpose(2, 0, 1).copy())
            img = img.float() / 255.0 if img.max() > 1.0 else img.float()
        return img.unsqueeze(0)

    key_t = _to_tensor(key)
    ref1_t = _to_tensor(ref1)
    ref2_t = _to_tensor(ref2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aligner = FFTAlign(downsample=downsample, do_mse_check=do_mse_check).to(device)

    # Step 1: ref1 → key (直接)
    aligned1, dx1, dy1, snr1 = aligner(
        ref1_t.to(device), key_t.to(device), return_displacement=True)

    # Step 2: ref2 → ref1 (相对位移)
    _, dx2_rel, dy2_rel, snr2_rel = aligner(
        ref2_t.to(device), ref1_t.to(device), return_displacement=True)

    # Step 3: 链式累加位移 → ref2 → key
    dx2 = dx1 + dx2_rel
    dy2 = dy1 + dy2_rel
    aligned2 = FFTAlign._apply_shift(ref2_t.to(device), dx2, dy2)

    return (aligned1.squeeze(0).cpu(),
            aligned2.squeeze(0).cpu(),
            (int(dx1.item()), int(dy1.item())),
            (int(dx2.item()), int(dy2.item())),
            (snr1[0], snr2_rel[0]))


def apply_feature_shift(feat, dx, dy, image_hw, return_mask=False,
                        mode='bilinear'):
    """按输入图像坐标系中的 FFT 位移平移 P3/P4 特征。

    与 :meth:`FFTAlign._apply_shift` 保持相同方向约定：
        out(y, x) = feat(y + dy_feat, x + dx_feat)

    与整数切片不同，这里保留 ``dx / stride``、``dy / stride`` 的分数部分，
    避免 P3/P4 上的粗暴取整吞掉 FFT 修正量。

    Args:
        feat: (B, C, Hf, Wf) 特征图。
        dx, dy: (B,) 输入图像像素坐标系中的位移。
        image_hw: 输入图像的 (H, W)。
        return_mask: 是否返回有效采样区域 mask，形状 (B, 1, Hf, Wf)。
        mode: grid_sample 插值模式，默认 bilinear。
    Returns:
        shifted；return_mask=True 时返回 (shifted, valid_mask)。
    """
    if feat.dim() != 4:
        raise ValueError(f'feat must be BCHW, got shape={tuple(feat.shape)}')
    B, _, Hf, Wf = feat.shape
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f'invalid image_hw={image_hw}')

    # FFT 在 float32 中产生位移；网格也用 float32 构造，随后由
    # grid_sample 按输入特征 dtype 输出。每个 batch 样本可有独立位移。
    dx = torch.as_tensor(dx, device=feat.device, dtype=torch.float32).reshape(-1)
    dy = torch.as_tensor(dy, device=feat.device, dtype=torch.float32).reshape(-1)
    if dx.numel() != B or dy.numel() != B:
        raise ValueError(f'displacement batch mismatch: feat B={B}, '
                         f'dx={dx.numel()}, dy={dy.numel()}')

    dx_feat = dx * (float(Wf) / float(image_w))
    dy_feat = dy * (float(Hf) / float(image_h))

    # align_corners=False 下像素中心归一化坐标为 (2*x+1)/W-1。
    yy, xx = torch.meshgrid(
        torch.arange(Hf, device=feat.device, dtype=torch.float32),
        torch.arange(Wf, device=feat.device, dtype=torch.float32),
        indexing='ij',
    )
    src_x = xx.unsqueeze(0) + dx_feat.view(B, 1, 1)
    src_y = yy.unsqueeze(0) + dy_feat.view(B, 1, 1)
    grid_x = (2.0 * src_x + 1.0) / Wf - 1.0
    grid_y = (2.0 * src_y + 1.0) / Hf - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    # A half-precision normalized grid can round a small learned displacement
    # back to zero BEFORE autocast promotes grid_sample. Keep geometry in FP32
    # on CPU and CUDA; autograd still reaches dx/dy through the float conversion.
    with torch.autocast(device_type=feat.device.type, enabled=False):
        shifted = F.grid_sample(
            feat.float(), grid, mode=mode, padding_mode='zeros', align_corners=False)
        if not return_mask:
            return shifted.to(dtype=feat.dtype)
        ones = torch.ones((B, 1, Hf, Wf), device=feat.device, dtype=torch.float32)
        valid_mask = F.grid_sample(
            ones, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        valid_mask = valid_mask.clamp(0.0, 1.0)
    return shifted.to(dtype=feat.dtype), valid_mask.to(dtype=feat.dtype)
