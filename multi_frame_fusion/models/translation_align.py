"""
RigidAlign: 全局刚体运动对齐 (平移 + 旋转)

利用先验知识：固定垂直摄像头，帧间仅存在 2D 刚体运动。
阶段 1：互相关搜索平移 (tx, ty)
阶段 2：平移对齐后的特征预测残差旋转 θ
最终用仿射变换 warp: [x'] = R(θ)·x + t
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RigidAlign(nn.Module):
    """
    全局刚体对齐：互相关搜平移 → 平移 warp → 预测旋转 → 仿射 warp。

    参数:
        in_ch: 输入通道数
        proj_ch: 互相关投影子空间通道数 (默认 32)
        search_radius: 平移搜索半径 (默认 16，P4 分辨率)
    """
    def __init__(self, in_ch, proj_ch=32, search_radius=16):
        super().__init__()
        self.search_radius = search_radius
        self.num_candidates = (2 * search_radius + 1) ** 2

        # 互相关投影
        self.proj_key = nn.Conv2d(in_ch, proj_ch, 1)
        self.proj_ref = nn.Conv2d(in_ch, proj_ch, 1)

        # 平移预测：互相关 → FC → softmax → soft-argmax
        self.trans_fc1 = nn.Linear(self.num_candidates, 64)
        self.trans_fc2 = nn.Linear(64, self.num_candidates)

        # 旋转预测：GAP(warped_ref, key) → FC → θ
        self.rot_fc1 = nn.Linear(in_ch * 2, 32)
        self.rot_fc2 = nn.Linear(32, 1)  # 单值 θ (弧度)

        # ── 初始化 ──
        for proj in [self.proj_key, self.proj_ref]:
            nn.init.kaiming_normal_(proj.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(proj.bias)
        nn.init.kaiming_normal_(self.trans_fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.trans_fc1.bias)
        nn.init.zeros_(self.trans_fc2.weight)
        nn.init.zeros_(self.trans_fc2.bias)
        nn.init.kaiming_normal_(self.rot_fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.rot_fc1.bias)
        nn.init.zeros_(self.rot_fc2.weight)
        nn.init.zeros_(self.rot_fc2.bias)

        # 预计算平移位移网格
        offsets = []
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                offsets.append([dx, dy])
        self.register_buffer('offset_grid', torch.tensor(offsets, dtype=torch.float32))

    def forward(self, nbr, ref):
        """
        nbr: (B, C, H, W) 邻居帧特征
        ref: (B, C, H, W) 关键帧特征
        返回: (B, C, H, W) 刚体对齐后特征
        """
        B, C, H, W = nbr.shape
        device = nbr.device
        R = self.search_radius
        K = 2 * R + 1

        # ━━━ 阶段 1：互相关搜平移 ━━━
        nbr_proj = self.proj_key(nbr)
        ref_proj = self.proj_ref(ref)

        ref_unfold = F.unfold(ref_proj, kernel_size=K, padding=R)
        ref_unfold = ref_unfold.view(B, -1, K * K, H, W)  # (B, P, K^2, H, W)
        nbr_exp = nbr_proj.unsqueeze(2)                    # (B, P, 1, H, W)
        corr = (nbr_exp * ref_unfold).sum(dim=1)            # (B, K^2, H, W)

        corr_gap = corr.mean([2, 3])                        # (B, K^2)
        x = F.relu(self.trans_fc1(corr_gap))
        logits = self.trans_fc2(x)
        weights = F.softmax(logits, dim=1)
        dxdy = weights @ self.offset_grid                   # (B, 2): tx, ty

        # ━━━ 阶段 2：平移 warp + 预测残差旋转 ━━━
        nbr_trans = self._warp_translation(nbr, dxdy, H, W, device)

        # 从平移对齐后的特征预测旋转角
        gap_trans = nbr_trans.mean([2, 3])  # (B, C)
        gap_ref = ref.mean([2, 3])
        x2 = torch.cat([gap_trans, gap_ref], dim=1)        # (B, 2C)
        x2 = F.relu(self.rot_fc1(x2))
        theta = self.rot_fc2(x2) * 0.1                       # tanh-like 缩放，最大 ±0.1 rad ≈ ±5.7°

        # ━━━ 阶段 3：仿射 warp (旋转 + 平移) ━━━
        aligned = self._warp_affine(nbr, dxdy, theta, H, W, device)
        return aligned

    def _warp_translation(self, x, dxdy, H, W, device):
        """纯平移 warp"""
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
        base = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        flow = base + dxdy.view(-1, 1, 1, 2)
        flow[..., 0] = 2.0 * flow[..., 0] / max(W - 1, 1) - 1.0
        flow[..., 1] = 2.0 * flow[..., 1] / max(H - 1, 1) - 1.0
        return F.grid_sample(x, flow, mode='bilinear',
                             padding_mode='border', align_corners=True)

    def _warp_affine(self, x, dxdy, theta, H, W, device):
        """仿射变换 warp：绕图像中心旋转 + 平移"""
        B = x.shape[0]
        # 旋转矩阵
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        R_mat = torch.stack([
            cos_t, -sin_t,
            sin_t,  cos_t,
        ], dim=1).view(B, 2, 2)  # (B, 2, 2)

        # 绕中心旋转：先平移到中心，旋转，再平移回去，最后加平移量
        center = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0], device=device)
        t_neg = -center.unsqueeze(0)                        # (1, 2)
        t_pos = center.unsqueeze(0)

        # 变换：x' = R(x - center) + center + t
        # 等价于 x' = R·x + (center + t - R·center)
        t_affine = t_pos + dxdy - (R_mat @ t_neg.unsqueeze(-1)).squeeze(-1)

        # 构建仿射矩阵 [R | t]
        affine = torch.cat([R_mat, t_affine.unsqueeze(-1)], dim=-1)  # (B, 2, 3)
        grid = F.affine_grid(affine, x.shape, align_corners=True)
        return F.grid_sample(x, grid, mode='bilinear',
                             padding_mode='border', align_corners=True)
