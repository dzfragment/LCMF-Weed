"""
fft_align: 独立的 FFT 快速帧间对齐模块。

从 models.fft_align 拆分而来，核心算法与模型/训练代码解耦，
可单独 import、测试和复用。

用法:
    from fft_align import FFTAlign, align_frame, chain_align_frames

    aligner = FFTAlign(downsample=4, do_mse_check=True).to(device).eval()
    dx, dy, snr = aligner.find_displacement(mov, ref)   # (B,3,H,W) tensor

    aligned, dx, dy, snr = align_frame(ref_np, mov_np)  # numpy 接口

    a1, a2, (dx1, dy1), (dx2, dy2), (snr1, snr2) = \
        chain_align_frames(key, ref1, ref2)             # 三帧链式对齐
"""

from .core import FFTAlign
from .utils import align_frame, chain_align_frames, apply_feature_shift
from .consistent import ConsistentAligner

__all__ = [
    'FFTAlign',
    'align_frame',
    'chain_align_frames',
    'apply_feature_shift',
    'ConsistentAligner',
]

__version__ = '1.1.0'
