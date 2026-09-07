"""
Pyramid Block-Matching Alignment — 无参数确定性对齐

无任何可训练参数，使用特征向量余弦相似度做金字塔块匹配：
  L3 (1/4 res): 大窗口 search，覆盖大位移
  L2 (1/2 res): 上采样 L3 flow + 小窗口精修
  L1 (full res): 上采样 L2 flow + 小窗口精修
  → warp 输出对齐特征

适用场景: 刚体为主的小位移对齐（杂草帧间平移）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidMatchAlign(nn.Module):
    """
    金字塔块匹配对齐模块。

    参数:
        in_ch: 输入通道数
        search_radius: 每层搜索半径，从粗到细 (默认 [5, 3, 1])
        num_levels: 金字塔层数 (默认 3)
        global_match: True=全局匹配(top-3平均), False=逐像素匹配
    """
    def __init__(self, in_ch, search_radius=(5, 3, 1), num_levels=3, global_match=True):
        super().__init__()
        self.in_ch = in_ch
        self.search_radius = tuple(search_radius)
        self.num_levels = num_levels
        self.global_match = global_match
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, nbr, ref):
        """
        nbr: (B, C, H, W) 邻居帧特征
        ref: (B, C, H, W) 参考帧特征（key frame）
        返回: (B, C, H, W) 对齐后的邻居帧特征
        """
        B, C, H, W = nbr.shape

        # ── 构建金字塔 (原始 → avg_pool → avg_pool) ──
        pyrs_nbr = [nbr]
        pyrs_ref = [ref]
        for _ in range(self.num_levels - 1):
            pyrs_nbr.append(self.down(pyrs_nbr[-1]))
            pyrs_ref.append(self.down(pyrs_ref[-1]))

        # ── 从粗到细匹配 ──
        flow = None

        for lvl in reversed(range(self.num_levels)):
            nbr_lvl = pyrs_nbr[lvl]
            ref_lvl = pyrs_ref[lvl]
            radius = self.search_radius[lvl]

            if flow is not None:
                # 上采样 flow 到当前分辨率
                _, _, H_l, W_l = nbr_lvl.shape
                flow = F.interpolate(flow, size=(H_l, W_l),
                                     mode='bilinear', align_corners=False) * 2.0
                # warp nbr 以减小搜索范围
                nbr_lvl = self._warp(nbr_lvl, flow)

            # 在当前层做匹配
            if self.global_match:
                residual = self._global_match(nbr_lvl, ref_lvl, radius)
            else:
                residual = self._block_match(nbr_lvl, ref_lvl, radius)
            flow = residual if flow is None else flow + residual

        # ── 用最终 flow warp 原始 nbr ──
        aligned = self._warp(nbr, flow)
        return aligned

    def _block_match(self, nbr, ref, radius):
        """
        逐像素块匹配: 对 nbr 每个位置，在 ref 的 (2r+1)×(2r+1) 窗口内
        搜索余弦相似度最高的位置，返回偏移量 flow。
        """
        B, C, H, W = nbr.shape
        K = 2 * radius + 1
        device = nbr.device

        # 归一化做余弦相似度
        nbr_norm = F.normalize(nbr, dim=1)           # (B, C, H, W)
        ref_norm = F.normalize(ref, dim=1)

        # 去通道均值
        nbr_centered = nbr_norm - nbr_norm.mean(dim=[2,3], keepdim=True)
        ref_centered = ref_norm - ref_norm.mean(dim=[2,3], keepdim=True)

        ref_unfold = F.unfold(ref_centered, kernel_size=K, padding=radius)  # (B, C*K^2, H*W)
        ref_unfold = ref_unfold.view(B, C, K * K, H, W)                 # (B, C, K^2, H, W)

        # nbr 向量广播到所有候选位置
        nbr_flat = nbr_norm.unsqueeze(2)  # (B, C, 1, H, W)

        # 逐候选位置的余弦相似度 (点积)
        sim = (nbr_flat * ref_unfold).sum(dim=1)  # (B, K^2, H, W)

        # 取最佳匹配的索引
        best_idx = sim.argmax(dim=1)  # (B, H, W)

        # 索引 → 偏移
        dy = best_idx // K - radius
        dx = best_idx % K - radius

        flow = torch.stack([dx.float(), dy.float()], dim=1)  # (B, 2, H, W)
        return flow

    def _global_match(self, nbr, ref, radius):
        """
        全局匹配: 全图统一的偏移量。
        对每个候选位移计算全图余弦相似度，取 top-3 按相似度加权平均。
        返回: (B, 2, 1, 1) 全局 flow
        """
        B, C, H, W = nbr.shape
        K = 2 * radius + 1
        device = nbr.device

        nbr_norm = F.normalize(nbr, dim=1)
        ref_norm = F.normalize(ref, dim=1)

        # 去通道均值，消除通道偏置对匹配的干扰
        nbr_centered = nbr_norm - nbr_norm.mean(dim=[2,3], keepdim=True)
        ref_centered = ref_norm - ref_norm.mean(dim=[2,3], keepdim=True)

        ref_unfold = F.unfold(ref_centered, kernel_size=K, padding=radius)  # (B, C*K^2, H*W)
        ref_unfold = ref_unfold.view(B, C, K * K, H * W)                 # (B, C, K^2, H*W)
        nbr_flat = nbr_norm.view(B, C, 1, H * W)                        # (B, C, 1, H*W)

        # 全图像素级别的点积求和 → 每个位移一个全局分数
        sim = (nbr_flat * ref_unfold).sum(dim=1)  # (B, K^2, H*W)
        sim = sim.mean(dim=2)                      # (B, K^2)  全图平均相似度

        # Top-3 峰值
        top3_val, top3_idx = sim.topk(min(3, K*K), dim=1)  # (B, 3)

        # 偏移
        dx = (top3_idx % K).float() - radius   # (B, 3)
        dy = (top3_idx // K).float() - radius

        # 按相似度加权平均（softmax 归一化权重）
        w = F.softmax(top3_val, dim=1)  # (B, 3)
        dx_avg = (dx * w).sum(dim=1, keepdim=True)  # (B, 1)
        dy_avg = (dy * w).sum(dim=1, keepdim=True)

        flow = torch.cat([dx_avg, dy_avg], dim=1).unsqueeze(-1).unsqueeze(-1)  # (B, 2, 1, 1)
        flow = flow.expand(-1, -1, H, W)  # (B, 2, H, W)
        return flow

    def _warp(self, x, flow):
        """
        用 grid_sample 按 flow 对 x 做空间变换。
        flow 单位: 像素 (dx, dy)。
        """
        B, C, H, W = x.shape
        device = x.device

        # 基准网格
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij')
        base_grid = torch.stack([gx, gy], dim=-1)  # (H, W, 2)

        # 加上 flow 偏移
        flow_xy = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
        grid = base_grid.unsqueeze(0) + flow_xy  # (B, H, W, 2)

        # 归一化到 [-1, 1] (grid_sample 约定)
        grid[..., 0] = 2.0 * grid[..., 0] / max(W - 1, 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / max(H - 1, 1) - 1.0

        return F.grid_sample(x, grid, mode='bilinear',
                             padding_mode='border', align_corners=True)
