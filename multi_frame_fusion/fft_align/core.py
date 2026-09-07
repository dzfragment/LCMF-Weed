"""
FFTAlign 核心算法：基于 FFT 互相关的快速全局平移对齐。

从 models/fft_align.py 拆分出的独立实现，不依赖任何模型/训练代码，
可单独 import、测试和复用。

方法流程（对应「快速帧间对齐方法」文档）：
  1. 提取 G 通道，降采样
  2. Sobel 梯度幅值（抑制均匀土壤噪声）
  3. FFT 去均值互相关 → 整数位移 + SNR
  4. MSE 自校验消歧方向
  5. 整数切片平移（无插值）
  6. 链式对齐：ref2 → ref1 → key（见 utils.chain_align_frames）

速度: ~3ms/帧对 (1024², 8× 降采样, GPU)
精度: 方向正确率 99.7%, 位移误差 ±4px (8×) / ±2px (4×)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# Sobel 核（用于梯度幅值计算）
# ═══════════════════════════════════════════════════════════

_SOBEL_X = torch.tensor(
    [[-1., 0., 1.],
     [-2., 0., 2.],
     [-1., 0., 1.]],
    dtype=torch.float32,
).view(1, 1, 3, 3)

_SOBEL_Y = torch.tensor(
    [[-1., -2., -1.],
     [ 0.,  0.,  0.],
     [ 1.,  2.,  1.]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


class FFTAlign(nn.Module):
    """
    FFT 互相关全局平移对齐（梯度幅值 + MSE 校验版）。

    流程：
      1. 提取 G 通道，降采样 (2.1)
      2. Sobel 梯度幅值 + 去均值 (2.1)
      3. FFT 互相关 → 峰值位移 + SNR (2.2)
      4. MSE 自校验消歧方向 (2.3)
      5. 整数切片平移 (2.4)

    位移约定:
        find_displacement(mov, ref) 返回的 (dx, dy) 表示「mov 内容相对 ref
        的位移」。对滑动窗口合成的序列，该值 = 窗口偏移的相反数，
        与标签位置差 (pos_mov - pos_ref) 直接可比。

    参数:
        downsample: 降采样倍数。8=速度优先(±4px), 4=精度优先(±2px)
        do_mse_check: 是否启用 MSE 自校验消歧方向（推荐 True）
        normalize: 是否使用相位归一化互相关 (cross-power spectrum)。
            默认 True（项目统一使用归一化，可压平原点自相关峰，
            对梯度稀疏场景更稳）。代价是对噪声更敏感。
        suppress_origin: 原点邻域抑制半径（降采样空间像素）。
            >0 时把相关面原点 ±r 邻域置为 -inf 再找峰，用于
            raw 互相关下原点自相关峰压制真实峰的场景。
            注意: 真实位移小于 r*downsample 时会误杀真峰，
            仅适合已知位移较大的场景（如滑动窗口序列）。
    """
    def __init__(self, downsample=8, do_mse_check=True, normalize=True,
                 suppress_origin=0):
        super().__init__()
        self.downsample = downsample
        self.do_mse_check = do_mse_check
        self.normalize = normalize
        self.suppress_origin = suppress_origin

        # 注册 Sobel 核为 buffer（自动跟随模型 device）
        self.register_buffer('_sobel_x', _SOBEL_X.clone(), persistent=False)
        self.register_buffer('_sobel_y', _SOBEL_Y.clone(), persistent=False)

    # ── 内部方法 ──

    @staticmethod
    def _extract_g(x):
        """提取 G 通道（RGB/BGR 的 index 一致）。"""
        return x[:, 1:2, :, :]

    def _gradient_magnitude(self, x):
        """
        Sobel 梯度幅值 + 去均值。

        均匀土壤区域梯度接近零，去均值后不再贡献噪声。
        grad(I) = sqrt((I * Gx)² + (I * Gy)²)
        返回去均值后的梯度幅值图。

        x: (B, 1, Hs, Ws) 降采样后的单通道图
        返回: (B, 1, Hs, Ws) 去均值的梯度幅值图
        """
        B, C, H, W = x.shape
        sx = self._sobel_x.to(dtype=x.dtype)
        sy = self._sobel_y.to(dtype=x.dtype)
        gx = F.conv2d(x, sx, padding=1, groups=C)
        gy = F.conv2d(x, sy, padding=1, groups=C)
        mag = torch.sqrt(gx ** 2 + gy ** 2)
        return mag - mag.mean(dim=[1, 2, 3], keepdim=True)

    def _compute_correlation(self, ref, mov):
        """
        FFT 互相关: corr = IFFT( FFT(ref_dm) × conj(FFT(mov_dm)) )

        ref, mov: (B, 1, Hs, Ws)，内部会去均值。
        返回: (B, Hs, Ws) 互相关图。
        """
        B, _, Hs, Ws = ref.shape
        ref_dm = ref - ref.mean(dim=[1, 2, 3], keepdim=True)
        mov_dm = mov - mov.mean(dim=[1, 2, 3], keepdim=True)
        F_ref = torch.fft.rfft2(ref_dm.squeeze(1))
        F_mov = torch.fft.rfft2(mov_dm.squeeze(1))
        cross = F_ref * F_mov.conj()
        if self.normalize:
            # 相位归一化: 每个频率只保留相位, 压平原点自相关峰
            cross = cross / cross.abs().clamp_min(1e-6)
        corr = torch.fft.irfft2(cross, s=(Hs, Ws))
        return corr

    def _suppress_origin_peak(self, corr):
        """把原点 ±r 邻域（含边界 wrap）置为 -inf，强制找非零位移峰。

        corr: (B, Hs, Ws)。注意零位移在 (0,0) 及其 wrap 邻域。
        """
        B, H, W = corr.shape
        r = int(self.suppress_origin)
        corr = corr.clone()
        idx_y = torch.arange(-r, r + 1, device=corr.device) % H
        idx_x = torch.arange(-r, r + 1, device=corr.device) % W
        corr[:, idx_y[:, None], idx_x[None, :]] = float('-inf')
        return corr

    def _correlation_peak(self, corr, Hs, Ws):
        """
        从相关面提取峰值位移和 SNR。

        返回: (dx_ds, dy_ds, snr_list) — 降采样空间位移 + 每样本 SNR
        """
        B = corr.shape[0]
        corr_flat = corr.view(B, -1)
        peak_vals, max_idx = torch.max(corr_flat, dim=1)
        py_ds = max_idx // Ws
        px_ds = max_idx % Ws

        dx_list, dy_list, snr_list = [], [], []
        for b in range(B):
            py, px = py_ds[b].item(), px_ds[b].item()
            dy = -(py if py <= Hs // 2 else py - Hs)
            dx = -(px if px <= Ws // 2 else px - Ws)
            peak = peak_vals[b].item()
            vals = corr[b]
            vals = vals[vals != float('-inf')]
            std = vals.std().item() if vals.numel() > 1 else 0.0
            snr = peak / std if std > 0 else 0.0
            dx_list.append(float(dx))
            dy_list.append(float(dy))
            snr_list.append(snr)

        return (torch.tensor(dx_list, device=corr.device),
                torch.tensor(dy_list, device=corr.device),
                snr_list)

    def _mse_validate(self, ref_grad, mov_grad, dx_ds, dy_ds):
        """
        MSE 自校验：在降采样梯度图上计算 +方向 和 -方向 的重叠区 MSE，
        选更低的消除方向歧义。

        ref_grad, mov_grad: (B, 1, Hs, Ws) 降采样梯度幅值图
        返回: (dx_ds, dy_ds) 校验后的位移
        """
        B, _, Hs, Ws = ref_grad.shape
        dx_out = dx_ds.clone()
        dy_out = dy_ds.clone()

        for b in range(B):
            ddx, ddy = dx_ds[b].item(), dy_ds[b].item()

            if abs(ddx) < 0.5 and abs(ddy) < 0.5:
                continue

            best_mse = float('inf')
            best_sign = 1

            for sign in [1, -1]:
                sdx, sdy = sign * ddx, sign * ddy
                y0_r = max(0, int(-sdy)); y1_r = min(Hs, int(Hs - sdy))
                x0_r = max(0, int(-sdx)); x1_r = min(Ws, int(Ws - sdx))
                y0_m = max(0, int(sdy));  y1_m = min(Hs, int(Hs + sdy))
                x0_m = max(0, int(sdx));  x1_m = min(Ws, int(Ws + sdx))

                h_overlap = y1_r - y0_r
                w_overlap = x1_r - x0_r
                if h_overlap <= 1 or w_overlap <= 1:
                    continue

                mse = F.mse_loss(
                    ref_grad[b:b + 1, :, y0_r:y1_r, x0_r:x1_r],
                    mov_grad[b:b + 1, :, y0_m:y1_m, x0_m:x1_m],
                )
                if mse.item() < best_mse:
                    best_mse = mse.item()
                    best_sign = sign

            if best_sign == -1:
                dx_out[b] = -ddx
                dy_out[b] = -ddy

        return dx_out, dy_out

    @staticmethod
    def _apply_shift(mov, dx, dy):
        """
        整数切片平移，边界填 0。

        用 tensor 切片搬像素，不做插值，与 grid_sample 不同：
        整数位移精确搬运，无双线性模糊。

        mov: (B, C, H, W) 待平移帧
        dx, dy: (B,) 整数像素位移
        返回: (B, C, H, W) 平移后的帧
        """
        B, C, H, W = mov.shape
        device = mov.device
        out = torch.zeros_like(mov)

        for b in range(B):
            ddx = int(round(dx[b].item()))
            ddy = int(round(dy[b].item()))

            if ddy >= 0:
                y_dst = slice(None, H - ddy)
                y_src = slice(ddy, None)
            else:
                y_dst = slice(-ddy, None)
                y_src = slice(None, H + ddy)

            if ddx >= 0:
                x_dst = slice(None, W - ddx)
                x_src = slice(ddx, None)
            else:
                x_dst = slice(-ddx, None)
                x_src = slice(None, W + ddx)

            y0d = y_dst.start or 0
            y1d = y_dst.stop if y_dst.stop is not None else H
            x0d = x_dst.start or 0
            x1d = x_dst.stop if x_dst.stop is not None else W
            y0s = y_src.start or 0
            y1s = y_src.stop if y_src.stop is not None else H
            x0s = x_src.start or 0
            x1s = x_src.stop if x_src.stop is not None else W

            if y1d > y0d and x1d > x0d:
                out[b, :, y0d:y1d, x0d:x1d] = mov[b, :, y0s:y1s, x0s:x1s]

        return out

    # ── 公开接口 ──

    def find_displacement(self, mov, ref):
        """
        计算 mov → ref 的位移。

        mov, ref: (B, 3, H, W)，值域 [0,1] float32
        返回: dx, dy — (B,) 全分辨率整数像素位移；snr_list — 每样本相关峰 SNR
        """
        B, _, H, W = mov.shape
        ds = self.downsample
        Hs, Ws = H // ds, W // ds

        # Step 1: 提取 G 通道 + 降采样
        g_ref = F.interpolate(
            self._extract_g(ref), size=(Hs, Ws),
            mode='bilinear', align_corners=False,
        )
        g_mov = F.interpolate(
            self._extract_g(mov), size=(Hs, Ws),
            mode='bilinear', align_corners=False,
        )

        # Step 2: 梯度幅值 + 去均值
        ref_grad = self._gradient_magnitude(g_ref)
        mov_grad = self._gradient_magnitude(g_mov)

        # Step 3: FFT 互相关 → 位移
        corr = self._compute_correlation(ref_grad, mov_grad)
        if self.suppress_origin:
            corr = self._suppress_origin_peak(corr)
        dx_ds, dy_ds, snr_list = self._correlation_peak(corr, Hs, Ws)

        # Step 4: MSE 自校验消歧方向
        if self.do_mse_check:
            dx_ds, dy_ds = self._mse_validate(ref_grad, mov_grad, dx_ds, dy_ds)

        # 还原到全分辨率
        dx = (dx_ds * ds).round()
        dy = (dy_ds * ds).round()

        return dx, dy, snr_list

    @torch.no_grad()
    def forward(self, mov, ref, return_displacement=False):
        """
        对齐 moving 帧到 reference 帧。

        Args:
            mov: (B, 3, H, W) 待对齐帧
            ref: (B, 3, H, W) 参考帧
            return_displacement: 是否同时返回位移值
        Returns:
            aligned: (B, 3, H, W) 对齐后的帧（边界填 0）
            若 return_displacement=True，返回 (aligned, dx, dy, snr_list)
        """
        # AMP 下输入可能是 float16，内部计算统一 upcast 到 float32
        orig_dtype = mov.dtype
        if mov.dtype != torch.float32:
            mov_f = mov.float()
            ref_f = ref.float()
        else:
            mov_f, ref_f = mov, ref

        dx, dy, snr_list = self.find_displacement(mov_f, ref_f)
        aligned = self._apply_shift(mov_f, dx, dy)

        if orig_dtype != torch.float32:
            aligned = aligned.to(dtype=orig_dtype)

        if return_displacement:
            return aligned, dx, dy, snr_list
        return aligned
