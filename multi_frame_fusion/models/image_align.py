"""
ImageAlign: 图像层全局平移对齐

下采样后逐位移计算互相关，高效 GPU 实现。
速度: ~5-15ms/次。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageAlign(nn.Module):
    """
    图像层全局平移对齐。无可训练参数。

    参数:
        search_radius: 全分辨率搜索半径 (像素)，默认 32
        downsample: 搜索前下采样倍数 (默认 4, 1024→256)
    """
    def __init__(self, search_radius=32, downsample=4):
        super().__init__()
        self.search_radius = search_radius
        self.downsample = downsample

    def forward(self, nbr, ref):
        B, C, H, W = nbr.shape
        device = nbr.device
        ds = self.downsample
        Rd = self.search_radius // ds  # 下采样后搜索半径

        # 下采样 + 归一化
        Hs, Ws = H // ds, W // ds
        nbr_ds = F.interpolate(nbr, size=(Hs, Ws), mode='bilinear', align_corners=False)
        ref_ds = F.interpolate(ref, size=(Hs, Ws), mode='bilinear', align_corners=False)
        nbr_ds = (nbr_ds - nbr_ds.mean(dim=[1,2,3], keepdim=True)) / (nbr_ds.std(dim=[1,2,3], keepdim=True) + 1e-8)
        ref_ds = (ref_ds - ref_ds.mean(dim=[1,2,3], keepdim=True)) / (ref_ds.std(dim=[1,2,3], keepdim=True) + 1e-8)

        # 预分配分数矩阵
        Kd = 2 * Rd + 1
        scores = torch.zeros(B, Kd * Kd, device=device)

        # 逐位移计算全图点积
        for i, dy in enumerate(range(-Rd, Rd + 1)):
            for j, dx in enumerate(range(-Rd, Rd + 1)):
                nb = torch.roll(nbr_ds, shifts=(dy, dx), dims=(2, 3))
                idx = i * Kd + j
                scores[:, idx] = (nb * ref_ds).sum(dim=[1,2,3]) / (Hs * Ws * C)

        # argmax → 最佳偏移
        best_idx = scores.argmax(dim=1)
        best_x = (best_idx % Kd).float() - Rd
        best_y = (best_idx // Kd).float() - Rd
        best_x = best_x * ds  # 恢复到全分辨率
        best_y = best_y * ds

        # grid_sample 平移
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
        base = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        flow = base + torch.stack([best_x, best_y], dim=-1).view(B, 1, 1, 2)
        flow[..., 0] = 2.0 * flow[..., 0] / max(W - 1, 1) - 1.0
        flow[..., 1] = 2.0 * flow[..., 1] / max(H - 1, 1) - 1.0
        return F.grid_sample(nbr, flow, mode='bilinear',
                             padding_mode='border', align_corners=True)
