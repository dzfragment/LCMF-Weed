"""
models.fft_align — 兼容层（历史路径）。

FFT 对齐算法已拆分为独立包 ``fft_align``（本目录下的顶层包），
本文件保留原导入路径供历史脚本使用：

    from models.fft_align import FFTAlign          # 仍可用
    from models.fft_align import chain_align_frames # 仍可用

新增代码请直接使用独立包：

    from fft_align import FFTAlign, align_frame, chain_align_frames

唯一保留在本文件的符号是 apply_fft_image_align（模型集成包装，
依赖 MultiFrameYOLO 的 forward 结构，不属于纯对齐算法）。
"""

from fft_align import (
    FFTAlign,
    align_frame,
    chain_align_frames,
    apply_feature_shift,
)
from fft_align.consistent import ConsistentAligner

__all__ = [
    'FFTAlign',
    'align_frame',
    'chain_align_frames',
    'apply_feature_shift',
    'apply_fft_image_align',
    'apply_consistent_image_align',
]


# ═══════════════════════════════════════════════════════════
# 模型集成（依赖 MultiFrameYOLO，保留在 models 层）
# ═══════════════════════════════════════════════════════════

def apply_fft_image_align(model, downsample=8, do_mse_check=True,
                          chain_align=True):
    """
    包装 MultiFrameYOLO：在 backbone 之前做 FFT 梯度幅值对齐。

    新管线（chain_align=True，链式对齐）:
        3帧 → [ref1→key 直接, ref2→ref1→key 链式] → backbone → DAV2 精修 → fusion

    旧管线（chain_align=False，直接对齐）:
        3帧 → [ref1→key, ref2→key] → backbone → DAV2 精修 → fusion

    Args:
        model: MultiFrameYOLO 实例
        downsample: FFT 降采样倍数 (4 或 8)
        do_mse_check: 启用 MSE 方向校验
        chain_align: True=链式对齐 (ref2→ref1→key), False=直接对齐 (ref2→key)
    Returns:
        model (原地修改并返回)
    """
    device = next(model.parameters()).device

    # 两个对齐器：r1→key 和 r2→ref1
    model._fft_align_r1 = FFTAlign(
        downsample=downsample, do_mse_check=do_mse_check,
    ).to(device)
    model._fft_align_r2 = FFTAlign(
        downsample=downsample, do_mse_check=do_mse_check,
    ).to(device)

    model._image_align_method = 'fft'
    model._fft_chain_align = chain_align

    orig_forward = model.forward

    def fft_aligned_forward(frames):
        key = frames[:, 0]
        ref1 = frames[:, 1]
        ref2 = frames[:, 2]

        # ref1 → key 始终直接对齐
        ref1_aligned, dx1, dy1, _ = model._fft_align_r1(
            ref1, key, return_displacement=True)

        if chain_align:
            # 链式对齐: ref2 → ref1 → key
            # Step 1: ref2 → ref1 (相对位移)
            _, dx2_rel, dy2_rel, _ = model._fft_align_r2(
                ref2, ref1, return_displacement=True)
            # Step 2: 位移累加
            dx2 = dx1 + dx2_rel
            dy2 = dy1 + dy2_rel
            # Step 3: 用累加位移平移 ref2
            ref2_aligned = FFTAlign._apply_shift(ref2, dx2, dy2)
        else:
            # 直接对齐: ref2 → key
            ref2_aligned = model._fft_align_r2(ref2, key)

        frames_aligned = torch.stack([key, ref1_aligned, ref2_aligned], dim=1)
        return orig_forward(frames_aligned)

    model.forward = fft_aligned_forward
    return model


# ═══════════════════════════════════════════════════════════
# 模型集成 (推荐): 三帧一致性约束管线
# ═══════════════════════════════════════════════════════════

def apply_consistent_image_align(model, downsample=4, k=5):
    """
    包装 MultiFrameYOLO：RGB 层用 ConsistentAligner 对齐三帧 (备选方案)。

    注意: v4 项目的主路径是 feature 模式 (RGB 只估计位移, 实际平移在 P2/P3/P4
    特征层), 对应训练参数 --consistent-align (模型内置一致位移估计)。
    本函数是 image 模式入口 (RGB 层整数平移), 仅当确需 RGB 层对齐时使用。

    一致性管线: 三帧 top-K 候选 + 矢量自洽约束 d(f1->f0)+d(f2->f1)≈d(f2->f0)
    + 期望注入 + 质量评分 + 回退。weed_sliding_merged 三集 40px 口径 100%。

    Args:
        model: MultiFrameYOLO 实例
        downsample: FFT 降采样倍数 (4=推荐, 8=速度)
        k: 每帧对相关面候选数 (默认 5)
    Returns:
        model (原地修改并返回)
    """
    import torch

    aligner = ConsistentAligner(downsample=downsample, k=k)
    model._consistent_aligner = aligner
    model._image_align_method = 'consistent'
    model._fft_chain_align = True

    orig_forward = model.forward

    def consistent_aligned_forward(frames):
        key = frames[:, 0]
        ref1 = frames[:, 1]
        ref2 = frames[:, 2]
        ref1_aligned, ref2_aligned = aligner.align_batch(key, ref1, ref2)
        frames_aligned = torch.stack([key, ref1_aligned, ref2_aligned], dim=1)
        return orig_forward(frames_aligned)

    model.forward = consistent_aligned_forward
    return model
