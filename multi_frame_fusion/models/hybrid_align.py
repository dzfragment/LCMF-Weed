"""
HybridAlign: PCD 金字塔粗对齐 + DeformAlignV2 精修

解决纯 DeformAlignV2 在大位移场景下感受野不足的问题：
  - PCD 金字塔 L3→L2→L1 逐层对齐，低分辨率层感受野覆盖大位移
  - PCD 使用 DCNv2 modulation mask（改进版）
  - PCD 输出再经 DeformAlignV2 做 DCNv2 mask + 3C 上下文精修

每层 PCD offset predictor 的 DCN 支持 modulation mask。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


# ═══════════════════════════════════════════════════════════
# DCNv2 wrapper + offset predictor chains
# ═══════════════════════════════════════════════════════════

class DeformConvV2(nn.Module):
    """DCNv2 wrapper（offset + modulation mask）"""
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        self.bias = nn.Parameter(torch.empty(out_ch))
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.k = k
        # ── 初始化：in_ch==out_ch 时恒等，否则 kaiming ──
        if in_ch == out_ch:
            nn.init.zeros_(self.weight)
            eye_ch = min(in_ch, out_ch)
            with torch.no_grad():
                center_idx = k // 2
                self.weight[:eye_ch, :eye_ch, center_idx, center_idx] = torch.eye(eye_ch)
            for dh in range(k):
                for dw in range(k):
                    if dh != center_idx or dw != center_idx:
                        nn.init.normal_(self.weight[:, :, dh, dw], mean=0, std=1e-4)
        else:
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = self.weight[0].numel()
        nn.init.uniform_(self.bias, -1 / (fan_in ** 0.5), 1 / (fan_in ** 0.5))

    def forward(self, x, offset, mask):
        return deform_conv2d(x, offset, self.weight, self.bias,
                             stride=self.stride, padding=self.padding,
                             mask=mask)


class DeformConvV1(nn.Module):
    """DCNv1 wrapper（仅 offset，无 mask）"""
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        self.bias = nn.Parameter(torch.empty(out_ch))
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.k = k
        # ── 初始化：in_ch==out_ch 时恒等，否则 kaiming ──
        if in_ch == out_ch:
            nn.init.zeros_(self.weight)
            eye_ch = min(in_ch, out_ch)
            with torch.no_grad():
                center_idx = k // 2
                self.weight[:eye_ch, :eye_ch, center_idx, center_idx] = torch.eye(eye_ch)
            for dh in range(k):
                for dw in range(k):
                    if dh != center_idx or dw != center_idx:
                        nn.init.normal_(self.weight[:, :, dh, dw], mean=0, std=1e-4)
        else:
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = self.weight[0].numel()
        nn.init.uniform_(self.bias, -1 / (fan_in ** 0.5), 1 / (fan_in ** 0.5))

    def forward(self, x, offset):
        return deform_conv2d(x, offset, self.weight, self.bias,
                             stride=self.stride, padding=self.padding)


def offset_mask_predictor(in_ch, hidden_ch, use_mask=True):
    """
    构建 offset（+mask）预测链：Conv → ReLU → Conv → ReLU → 1×1
    最后一层零初始化（训练初期 offset≈0, mask≈1）。
    """
    k = 3
    offset_ch = 2 * k * k  # 18
    mask_ch = k * k         # 9

    layers = nn.ModuleDict({
        'conv1': nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
        'conv2': nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
        'to_offset': nn.Conv2d(hidden_ch, offset_ch, 3, 1, 1),
    })
    if use_mask:
        layers['to_mask'] = nn.Conv2d(hidden_ch, mask_ch, 3, 1, 1)

    # 零初始化 offset 输出
    nn.init.zeros_(layers['to_offset'].weight)
    nn.init.zeros_(layers['to_offset'].bias)
    # mask 初始化为接近 1: sigmoid(2.2) ≈ 0.9
    if use_mask:
        nn.init.constant_(layers['to_mask'].weight, 0)
        nn.init.constant_(layers['to_mask'].bias, 2.2)

    return layers


def forward_predictor(layers, x):
    """前向 offset predictor，返回 (offset, mask_or_None)"""
    o = F.relu(layers['conv1'](x))
    o = F.relu(layers['conv2'](o))
    offset = layers['to_offset'](o)
    mask = layers['to_mask'](o).sigmoid() if 'to_mask' in layers else None
    return offset, mask


def build_pyramid(feat, downsample_conv):
    """
    从单层特征构建 3 层特征金字塔。
    feat: (B, C, H, W)
    返回: [L1, L2, L3]，L1 分辨率最高
    """
    L1 = feat
    L2 = downsample_conv(L1)
    L3 = downsample_conv(L2)
    return [L1, L2, L3]


# ═══════════════════════════════════════════════════════════
# PCDAlignV2: PCD 金字塔对齐 + DCNv2 mask
# ═══════════════════════════════════════════════════════════

class PCDAlignV2(nn.Module):
    """
    PCD 金字塔对齐，支持 DCNv2 modulation mask。

    流程（与原版 PCD 一致，增加 mask 支持）：
    1. L3（最粗糙）：concat(nbr, ref) → offset+mask → DCNv2 warp
    2. L2：concat(nbr, ref, offset_l3_up*2) → offset+mask → DCNv2 warp → 融合 feat_l3_up
    3. L1：concat(nbr, ref, offset_l2_up*2) → offset+mask → DCNv2 warp → 融合 feat_l2_up
    4. 级联精修：concat(feat_l1, ref) → offset+mask → DCNv2 warp

    参数:
        num_feat: 特征通道数（所有金字塔层统一）
        use_mask: 是否使用 DCNv2 modulation mask
    """
    def __init__(self, num_feat=64, use_mask=True):
        super().__init__()
        self.num_feat = num_feat
        self.use_mask = use_mask

        # ─── L3: 最粗糙层 ───
        self.pred_l3 = offset_mask_predictor(num_feat * 2, num_feat * 2, use_mask)
        self.dcn_l3 = (DeformConvV2 if use_mask else DeformConvV1)(num_feat, num_feat, 3, padding=1)

        # ─── L2: concat(nbr,ref) + offset_l3_up ───
        self.pred_l2 = offset_mask_predictor(num_feat * 2 + 18, num_feat, use_mask)
        self.dcn_l2 = (DeformConvV2 if use_mask else DeformConvV1)(num_feat, num_feat, 3, padding=1)
        self.fuse_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # ─── L1: 最精细层 ───
        self.pred_l1 = offset_mask_predictor(num_feat * 2 + 18, num_feat, use_mask)
        self.dcn_l1 = (DeformConvV2 if use_mask else DeformConvV1)(num_feat, num_feat, 3, padding=1)
        self.fuse_l1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # ─── 级联精修 ───
        self.pred_cas = offset_mask_predictor(num_feat * 2, num_feat, use_mask)
        self.dcn_cas = (DeformConvV2 if use_mask else DeformConvV1)(num_feat, num_feat, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def _dcn_forward(self, dcn, x, offset, mask):
        if self.use_mask:
            return dcn(x, offset, mask)
        else:
            return dcn(x, offset)

    def forward(self, nbr_feat_l, ref_feat_l):
        """
        nbr_feat_l: [L1, L2, L3] 邻居帧金字塔
        ref_feat_l: [L1, L2, L3] 参考帧金字塔
        返回: (B, num_feat, H_l1, W_l1) 对齐后特征
        """
        nbr_l3, nbr_l2, nbr_l1 = nbr_feat_l[2], nbr_feat_l[1], nbr_feat_l[0]
        ref_l3, ref_l2, ref_l1 = ref_feat_l[2], ref_feat_l[1], ref_feat_l[0]

        # ─── L3 ───
        cat_l3 = torch.cat([nbr_l3, ref_l3], dim=1)
        offset_l3, mask_l3 = forward_predictor(self.pred_l3, cat_l3)
        feat_l3 = self._dcn_forward(self.dcn_l3, nbr_l3, offset_l3, mask_l3)

        # ─── L2 ───
        offset_l3_up = self.upsample(offset_l3) * 2
        feat_l3_up = self.upsample(F.relu(feat_l3))
        cat_l2 = torch.cat([nbr_l2, ref_l2, offset_l3_up], dim=1)
        offset_l2, mask_l2 = forward_predictor(self.pred_l2, cat_l2)
        feat_l2 = self._dcn_forward(self.dcn_l2, nbr_l2, offset_l2, mask_l2)
        feat_l2 = F.relu(self.fuse_l2(torch.cat([feat_l2, feat_l3_up], dim=1)))

        # ─── L1 ───
        offset_l2_up = self.upsample(offset_l2) * 2
        feat_l2_up = self.upsample(feat_l2)
        cat_l1 = torch.cat([nbr_l1, ref_l1, offset_l2_up], dim=1)
        offset_l1, mask_l1 = forward_predictor(self.pred_l1, cat_l1)
        feat_l1 = self._dcn_forward(self.dcn_l1, nbr_l1, offset_l1, mask_l1)
        feat_l1 = F.relu(self.fuse_l1(torch.cat([feat_l1, feat_l2_up], dim=1)))

        # ─── 级联精修 ───
        cat_cas = torch.cat([feat_l1, ref_l1], dim=1)
        offset_cas, mask_cas = forward_predictor(self.pred_cas, cat_cas)
        feat_out = F.relu(self._dcn_forward(self.dcn_cas, feat_l1, offset_cas, mask_cas))

        return feat_out


# ═══════════════════════════════════════════════════════════
# HybridAlign: PCD 粗对齐 + DeformAlignV2 精修
# ═══════════════════════════════════════════════════════════

class HybridAlign(nn.Module):
    """
    PCD 金字塔粗对齐 + DeformAlignV2 精修。

    解决纯 DeformAlignV2 在大位移（>50px）下感受野不足的问题：
      - PCD 金字塔 L3 层感受野 ~320px（图像空间），覆盖全位移范围
      - PCD 输出再经 DeformAlignV2 做 mask + 3C 上下文精修
      - 保留 DeformAlignV2 的初始化策略（offset 零初始化，接近恒等启动）

    参数:
        in_ch: 输入通道数（backbone 输出）
        num_feat: PCD 金字塔内部通道数（默认 64）
        hidden_ch: DeformAlignV2 offset 预测隐藏通道数
        use_pcd_mask: PCD 是否使用 DCNv2 mask
    """
    def __init__(self, in_ch, num_feat=64, hidden_ch=None, use_pcd_mask=True):
        super().__init__()
        if hidden_ch is None:
            hidden_ch = min(96, in_ch * 2)

        # 金字塔构建：stride 2 下采样
        self.down = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        # 输入投影到 PCD 工作通道
        self.proj_in = nn.Conv2d(in_ch, num_feat, 1, 1, 0)

        # PCD 粗对齐
        self.pcd = PCDAlignV2(num_feat, use_mask=use_pcd_mask)

        # 投影回原通道：
        # - 同维度时初始化为近恒等映射，让 PCD 输出直接传给 DeformAlignV2
        # - 不同维度时用小权重 + 零偏置，训练早期接近恒等
        self.proj_mid = nn.Conv2d(num_feat, in_ch, 1, 1, 0)
        if num_feat == in_ch:
            # 初始化为恒等映射：输出 ≈ 输入
            nn.init.zeros_(self.proj_mid.weight)
            eye_ch = min(num_feat, in_ch)
            with torch.no_grad():
                self.proj_mid.weight[:eye_ch, :eye_ch] = torch.eye(eye_ch).unsqueeze(-1).unsqueeze(-1)
            nn.init.zeros_(self.proj_mid.bias)
        else:
            # 小权重初始化，训练早期 PCD 贡献很小，让 DeformAlignV2 从恒等启动
            nn.init.normal_(self.proj_mid.weight, mean=0, std=0.001)
            nn.init.zeros_(self.proj_mid.bias)

        # DeformAlignV2 精修（复用现有实现）
        from .modules import DeformAlignV2
        self.refine = DeformAlignV2(in_ch, hidden_ch)

    def forward(self, nbr, ref):
        """
        nbr: (B, C, H, W) 邻居帧特征
        ref: (B, C, H, W) 参考帧特征（key frame）
        返回: (B, C, H, W) 对齐后的特征
        """
        # 投影 + 构建金字塔
        nbr_proj = self.proj_in(nbr)
        ref_proj = self.proj_in(ref)
        nbr_pyr = build_pyramid(nbr_proj, self.down)
        ref_pyr = build_pyramid(ref_proj, self.down)

        # PCD 粗对齐
        aligned_coarse = self.pcd(nbr_pyr, ref_pyr)

        # 投影回原通道
        aligned_mid = self.proj_mid(aligned_coarse)

        # DeformAlignV2 精修
        aligned_fine = self.refine(aligned_mid, ref)

        return aligned_fine


# ═══════════════════════════════════════════════════════════
# 简化的 PCD-only 对齐（用于对比实验）
# ═══════════════════════════════════════════════════════════

class PCDOnlyAlign(nn.Module):
    """
    纯 PCD 对齐（不含 DeformAlignV2 精修），用于消融对比。

    参数:
        in_ch: 输入通道数
        num_feat: PCD 内部通道数
        use_mask: 是否使用 DCNv2 mask
    """
    def __init__(self, in_ch, num_feat=64, use_mask=True):
        super().__init__()
        self.proj_in = nn.Conv2d(in_ch, num_feat, 1, 1, 0)
        self.down = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.pcd = PCDAlignV2(num_feat, use_mask=use_mask)
        self.proj_out = nn.Conv2d(num_feat, in_ch, 1, 1, 0)

    def forward(self, nbr, ref):
        nbr_proj = self.proj_in(nbr)
        ref_proj = self.proj_in(ref)
        nbr_pyr = build_pyramid(nbr_proj, self.down)
        ref_pyr = build_pyramid(ref_proj, self.down)
        aligned = self.pcd(nbr_pyr, ref_pyr)
        return self.proj_out(aligned)
