"""
DeformAlignV2 + AdaptiveFusion 模块
用于多帧特征对齐和融合（升级版）

DeformAlignV2 相比 V1 的改进:
  - DCNv2 modulation mask: 每个采样点学习 [0,1] 权重
  - Stage 2 3C 输入: cat(aligned1, nbr, ref) 感知原始运动
  - Mask 初始化为 0.9（训练早期近似 DCNv1）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


# ═══════════════════════════════════════════════════════════
# FeatureSharpener — 特征锐化（增强对空间位移的敏感度）
# ═══════════════════════════════════════════════════════════

class FeatureSharpener(nn.Module):
    """
    轻量特征锐化层，用于 DeformAlign 的 offset predictor 输入前端。

    问题：冻结 backbone 输出的 P3/P4 特征有 stride 8/16 的平移不变性，
    特征本身对亚像素位移的敏感度较低，限制 offset predictor 的学习上限。

    解法：在特征进入 offset predictor 之前，过一个极轻的 depthwise 卷积，
    初始化为近似 Laplacian 高通滤波（中心=4，四邻=-1）。
    这会增强特征图的高频分量，让 offset predictor 更容易感知空间位移。

    参数量：C × 3 × 3，每组约 576 参数（对 C=64），几乎零开销。
    """
    def __init__(self, channels):
        super().__init__()
        self.sharpen = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        # Laplacian: 中心 +4，四邻 -1
        nn.init.constant_(self.sharpen.weight, 0)
        with torch.no_grad():
            for c in range(channels):
                self.sharpen.weight[c, 0, 1, 1] = 4.0
                self.sharpen.weight[c, 0, 0, 1] = -1.0
                self.sharpen.weight[c, 0, 2, 1] = -1.0
                self.sharpen.weight[c, 0, 1, 0] = -1.0
                self.sharpen.weight[c, 0, 1, 2] = -1.0
        nn.init.zeros_(self.sharpen.bias)
        # 输出缩放，防止特征量级爆炸
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # 残差连接：sharpened = scale * sharpen(x) + original
        # scale 从 0.1 开始学习，训练早期接近恒等
        return self.scale * self.sharpen(x) + x


# ═══════════════════════════════════════════════════════════
# DeformConvPackV2 — DCNv2 (offset + modulation mask)
# ═══════════════════════════════════════════════════════════

class _NoOp(nn.Module):
    """两输入直通模块：(ref, key) → ref（消融实验中替代对齐模块）"""
    def forward(self, x, _key=None):
        return x


class GlobalShiftHead(nn.Module):
    """Predict one global (dx, dy) residual per reference frame.

    Inputs are the key/ref features at a single pyramid level, after the FFT
    coarse shift.  The module is deliberately tiny: it predicts **2 numbers**
    from a concatenated local feature pair, then spatially averages the
    prediction.  Output is in *feature pixels* of the input level; the caller
    converts it to image pixels with the level stride.

    `out` is zero-initialised, so a freshly built model is exactly FFT-only.
    """
    def __init__(self, in_ch, hidden=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch * 2, hidden, 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, 1, 1)
        self.out = nn.Conv2d(hidden, 2, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, key, ref):
        x = torch.cat([key, ref], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.out(x).mean(dim=(2, 3))  # (B, 2) in feature pixels


class DeformConvPackV2(nn.Module):
    """DCNv2 wrapper（offset + modulation mask）"""
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        self.bias = nn.Parameter(torch.empty(out_ch))
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.k = k
        # ── 初始化 ──
        # 当 in_ch==out_ch 时，初始化为近似恒等映射：
        #   中心位置=scaled identity, 其余位置=微小噪声
        # 这样 offset=0 时 DCN ≈ identity，训练从「不对齐」基准启动。
        # 当 in_ch!=out_ch 时，保持 kaiming 初始化。
        if in_ch == out_ch:
            nn.init.zeros_(self.weight)
            eye_ch = min(in_ch, out_ch)
            with torch.no_grad():
                center_idx = k // 2
                self.weight[:eye_ch, :eye_ch, center_idx, center_idx] = torch.eye(eye_ch)
            # 非中心位置加微小噪声破对称
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


# ═══════════════════════════════════════════════════════════
# DeformAlignV2 — DCNv2 + mask + Stage 2 nbr context
# ═══════════════════════════════════════════════════════════

class DeformAlignV2(nn.Module):
    """
    改进版 DeformAlign：DCNv2 mask + Stage 2 含原始 nbr 上下文。

    相比 DeformAlign (DCNv1) 的改进:
    1. DCNv2 modulation mask: 网络可以为每个采样点学习 [0,1] 权重，
       在运动边界和遮挡区域主动抑制无效采样点
    2. Stage 2 看到 cat(aligned1, nbr, ref)，不只是 cat(aligned1, ref)，
       让 refinement 阶段能感知原始运动幅度

    参数:
        in_ch: 输入通道数
        hidden_ch: offset/mask 预测的隐藏通道数
        use_sharpener: 是否在 offset predictor 前加特征锐化层
    """
    def __init__(self, in_ch, hidden_ch=None, use_sharpener=False):
        super().__init__()
        if hidden_ch is None:
            hidden_ch = min(96, in_ch * 2)
        k = 3
        offset_ch = 2 * k * k  # 18
        mask_ch = k * k         # 9
        self.use_sharpener = use_sharpener

        # 可选：特征锐化层（增强对空间位移的敏感度）
        if use_sharpener:
            self.sharpener_nbr = FeatureSharpener(in_ch)
            self.sharpener_ref = FeatureSharpener(in_ch)

        # Stage 1
        self.s1_conv1 = nn.Conv2d(in_ch * 2, hidden_ch, 3, 1, 1)
        self.s1_conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1)
        self.s1_offset = nn.Conv2d(hidden_ch, offset_ch, 3, 1, 1)
        self.s1_mask = nn.Conv2d(hidden_ch, mask_ch, 3, 1, 1)
        self.s1_dcn = DeformConvPackV2(in_ch, in_ch, 3, padding=1)

        # Stage 2 (3C 输入: aligned1 + 原始nbr + ref)
        self.s2_conv1 = nn.Conv2d(in_ch * 3, hidden_ch, 3, 1, 1)
        self.s2_conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1)
        self.s2_offset = nn.Conv2d(hidden_ch, offset_ch, 3, 1, 1)
        self.s2_mask = nn.Conv2d(hidden_ch, mask_ch, 3, 1, 1)
        self.s2_dcn = DeformConvPackV2(in_ch, in_ch, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # offset 零初始化（训练初期 offset≈0）
        for m in [self.s1_offset, self.s2_offset]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        # mask 初始化为接近 1（sigmoid(2.2) ≈ 0.9），训练初期近似 DCNv1
        for m in [self.s1_mask, self.s2_mask]:
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 2.2)

    def forward(self, nbr, ref):
        """
        nbr: (B, C, H, W) 邻居帧特征
        ref: (B, C, H, W) 参考帧特征（key frame）
        返回: (B, C, H, W) 对齐后的特征
        """
        # 可选：特征锐化
        if self.use_sharpener:
            nbr_s = self.sharpener_nbr(nbr)
            ref_s = self.sharpener_ref(ref)
        else:
            nbr_s = nbr
            ref_s = ref

        # Stage 1
        cat1 = torch.cat([nbr_s, ref_s], dim=1)
        o1 = self.relu(self.s1_conv1(cat1))
        o1 = self.relu(self.s1_conv2(o1))
        offset1 = self.s1_offset(o1)
        mask1 = self.s1_mask(o1).sigmoid()
        aligned1 = self.s1_dcn(nbr, offset1, mask1)

        # Stage 2: 同时看到 aligned1、原始 nbr、ref
        cat2 = torch.cat([aligned1, nbr, ref], dim=1)
        o2 = self.relu(self.s2_conv1(cat2))
        o2 = self.relu(self.s2_conv2(o2))
        offset2 = self.s2_offset(o2)
        mask2 = self.s2_mask(o2).sigmoid()
        aligned2 = self.s2_dcn(aligned1, offset2, mask2)

        # 供跨层 offset 复用（OffsetReuseAlign）读取；两份都是计算图节点，
        # 目标层复用后梯度会一并回传到本层的 offset predictor。
        self._last_offset1 = offset1
        self._last_mask1 = mask1
        return aligned2


class OffsetReuseAlign(nn.Module):
    """跨层复用 DCN offset/mask 的轻量对齐模块。

    只做一次 DCN（不训练 offset predictor）。offset/mask 来自共享层
    (shared_offset_level) 的 DeformAlignV2 第一级输出，按目标层特征图
    尺寸双线性缩放，并按层间像素尺度缩放 offset 幅值：

        offset_target = offset_source * (H_target / H_source)

    因为 stride 越小特征图越大，同样的图像位移在细层对应更大的
    feature-pixel 偏移。
    """

    def __init__(self, in_ch, reuse_mode='bilinear'):
        super().__init__()
        self.reuse_mode = reuse_mode
        self.dcn = DeformConvPackV2(in_ch, in_ch, 3, padding=1)

    def _offset_mask(self, nbr, src_module=None):
        """Build the reused (offset, mask) pair for this level."""
        B, C, H, W = nbr.shape
        off_s = getattr(src_module, '_last_offset1', None) if src_module is not None else None
        mask_s = getattr(src_module, '_last_mask1', None) if src_module is not None else None
        if off_s is None or mask_s is None:
            # 兜底：没有共享 offset 时退化为普通 conv（offset=0）
            off = nbr.new_zeros((B, 18, H, W))
            mask = nbr.new_ones((B, 9, H, W))
        else:
            hs, ws = off_s.shape[-2:]
            # 共享层若是 5x5 DCN（dcn_variant=k5），目标层仍是 3x3：
            # 取中心 3x3 的 offset/mask，保持初值与 V2 完全一致。
            if off_s.shape[1] == 50:
                k = 5
                centre = [(r + 1) * k + (c + 1)
                          for r in range(3) for c in range(3)]
                oidx = torch.cat([torch.tensor([2 * t, 2 * t + 1])
                                  for t in centre]).to(off_s.device)
                midx = torch.tensor(centre, device=off_s.device)
                off_s = off_s.index_select(1, oidx)
                mask_s = mask_s.index_select(1, midx)
            if self.reuse_mode == 'global_mean':
                # 每帧一个全局 (dy,dx) 向量，广播到目标层
                off = off_s.mean(dim=(2, 3), keepdim=True).expand(-1, -1, H, W)
                mask = mask_s.mean(dim=(2, 3), keepdim=True).expand(-1, -1, H, W)
            elif self.reuse_mode == 'nearest':
                off = F.interpolate(off_s, size=(H, W), mode='nearest')
                mask = F.interpolate(mask_s, size=(H, W), mode='nearest')
            elif self.reuse_mode == 'avg_pool':
                off = F.adaptive_avg_pool2d(off_s, (H, W))
                mask = F.adaptive_avg_pool2d(mask_s, (H, W))
            else:  # bilinear per-cell
                off = F.interpolate(off_s, size=(H, W), mode='bilinear',
                                    align_corners=True)
                mask = F.interpolate(mask_s, size=(H, W), mode='bilinear',
                                     align_corners=True)
            # 层间像素尺度：P3(128) -> P2(256) 放大 2 倍，P3 -> P4(64) 缩小 0.5
            off = off * (float(H) / float(hs))
        return off, mask

    def forward(self, nbr, ref=None, src_module=None):
        off, mask = self._offset_mask(nbr, src_module)
        return self.dcn(nbr, off, mask)


class DeformAlignV2Triple(DeformAlignV2):
    """三阶段 DAV2：在标准两阶段之后追加一次独立的 offset/mask/DCN 精修。"""
    def __init__(self, in_ch, hidden_ch=None, use_sharpener=False):
        super().__init__(in_ch, hidden_ch=hidden_ch, use_sharpener=use_sharpener)
        if hidden_ch is None:
            hidden_ch = min(96, in_ch * 2)
        k = 3
        offset_ch = 2 * k * k
        mask_ch = k * k
        self.s3_conv1 = nn.Conv2d(in_ch * 3, hidden_ch, 3, 1, 1)
        self.s3_conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1)
        self.s3_offset = nn.Conv2d(hidden_ch, offset_ch, 3, 1, 1)
        self.s3_mask = nn.Conv2d(hidden_ch, mask_ch, 3, 1, 1)
        self.s3_dcn = DeformConvPackV2(in_ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.s3_offset.weight)
        nn.init.zeros_(self.s3_offset.bias)
        nn.init.constant_(self.s3_mask.weight, 0)
        nn.init.constant_(self.s3_mask.bias, 2.2)

    def forward(self, nbr, ref):
        if self.use_sharpener:
            nbr_s = self.sharpener_nbr(nbr)
            ref_s = self.sharpener_ref(ref)
        else:
            nbr_s = nbr
            ref_s = ref

        cat1 = torch.cat([nbr_s, ref_s], dim=1)
        o1 = self.relu(self.s1_conv1(cat1))
        o1 = self.relu(self.s1_conv2(o1))
        offset1 = self.s1_offset(o1)
        mask1 = self.s1_mask(o1).sigmoid()
        aligned1 = self.s1_dcn(nbr, offset1, mask1)
        self._last_offset1 = offset1
        self._last_mask1 = mask1

        cat2 = torch.cat([aligned1, nbr, ref], dim=1)
        o2 = self.relu(self.s2_conv1(cat2))
        o2 = self.relu(self.s2_conv2(o2))
        aligned2 = self.s2_dcn(
            aligned1, self.s2_offset(o2), self.s2_mask(o2).sigmoid())

        cat3 = torch.cat([aligned2, nbr, ref], dim=1)
        o3 = self.relu(self.s3_conv1(cat3))
        o3 = self.relu(self.s3_conv2(o3))
        return self.s3_dcn(
            aligned2, self.s3_offset(o3), self.s3_mask(o3).sigmoid())


class DeformAlignV2Single(nn.Module):
    """
    单阶段 DeformAlignV2：仅 Stage 1，无 Stage 2 精修。

    用于消融实验，对比两阶段级联对齐的增益。
    参数量约为 DeformAlignV2 的 55%。
    """
    def __init__(self, in_ch, hidden_ch=None, use_sharpener=False):
        super().__init__()
        if hidden_ch is None:
            hidden_ch = min(96, in_ch * 2)
        k = 3
        offset_ch = 2 * k * k
        mask_ch = k * k
        self.use_sharpener = use_sharpener

        if use_sharpener:
            self.sharpener_nbr = FeatureSharpener(in_ch)
            self.sharpener_ref = FeatureSharpener(in_ch)

        self.s1_conv1 = nn.Conv2d(in_ch * 2, hidden_ch, 3, 1, 1)
        self.s1_conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1)
        self.s1_offset = nn.Conv2d(hidden_ch, offset_ch, 3, 1, 1)
        self.s1_mask = nn.Conv2d(hidden_ch, mask_ch, 3, 1, 1)
        self.s1_dcn = DeformConvPackV2(in_ch, in_ch, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        nn.init.zeros_(self.s1_offset.weight)
        nn.init.zeros_(self.s1_offset.bias)
        nn.init.constant_(self.s1_mask.weight, 0)
        nn.init.constant_(self.s1_mask.bias, 2.2)

    def forward(self, nbr, ref):
        if self.use_sharpener:
            nbr_s = self.sharpener_nbr(nbr)
            ref_s = self.sharpener_ref(ref)
        else:
            nbr_s = nbr
            ref_s = ref

        cat1 = torch.cat([nbr_s, ref_s], dim=1)
        o1 = self.relu(self.s1_conv1(cat1))
        o1 = self.relu(self.s1_conv2(o1))
        offset1 = self.s1_offset(o1)
        mask1 = self.s1_mask(o1).sigmoid()
        self._last_offset1 = offset1
        self._last_mask1 = mask1
        return self.s1_dcn(nbr, offset1, mask1)


# ═══════════════════════════════════════════════════════════
# AdaptiveFusion — Per-Channel Gated Residual (STF-style)
# ═══════════════════════════════════════════════════════════

class AdaptiveFusion(nn.Module):
    """
    自适应多帧融合模块（参考 STF 2024 的门控残差设计）。

    每个 ref 帧独立学习 per-channel gate，替代原来的 spatial+channel 双分支相加：
        gate_i = sigmoid(FC(ReLU(FC(GAP(cat(key, aligned_ref_i))))))
        output = key + proj(gate_1 * aligned_ref1 + gate_2 * aligned_ref2)

    关键设计原则：
      - 单分支门控，不搞「两个注意力相加」
      - gate ∈ [0,1] per-channel，天然有界
      - 训练早期 gate≈0.5（FC 零初始化），proj 零初始化 → output≈key（恒等启动）
      - proj 是 1×1 卷积，学习如何缩放融合后的 ref 信息

    参数:
        channels: 特征通道数
        reduction: gate 网络隐层压缩比
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels

        hidden = max(channels // reduction, 16)

        # Gate network for ref1: cat(key_gap, ref1_gap) → hidden → channels
        self.gate_r1_fc1 = nn.Linear(channels * 2, hidden)
        self.gate_r1_fc2 = nn.Linear(hidden, channels)

        # Gate network for ref2（独立权重，允许不对称门控）
        self.gate_r2_fc1 = nn.Linear(channels * 2, hidden)
        self.gate_r2_fc2 = nn.Linear(hidden, channels)

        # 输出投影：学习缩放融合后的 ref 信息
        self.proj = nn.Conv2d(channels, channels, 1, 1, 0)

        # ── 初始化：训练早期恒等映射 ──
        # gate FC2 零初始化 → sigmoid(0) = 0.5，均等融合
        for fc in [self.gate_r1_fc2, self.gate_r2_fc2]:
            nn.init.zeros_(fc.weight)
            nn.init.zeros_(fc.bias)
        # proj 零初始化 → proj(·)=0，output = key + 0 = key
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, key, ref1, ref2):
        """
        key:  (B, C, H, W) 关键帧特征
        ref1: (B, C, H, W) 已对齐的参考帧1特征
        ref2: (B, C, H, W) 已对齐的参考帧2特征
        返回: (B, C, H, W) 融合后特征
        """
        B, C, H, W = key.shape

        # GAP：将空间信息压缩为每通道一个标量
        gap_key = key.mean([2, 3])    # (B, C)
        gap_r1  = ref1.mean([2, 3])
        gap_r2  = ref2.mean([2, 3])

        # Gate for ref1
        g1 = F.relu(self.gate_r1_fc1(torch.cat([gap_key, gap_r1], dim=1)))
        g1 = self.gate_r1_fc2(g1).sigmoid()   # (B, C), ∈ [0, 1]

        # Gate for ref2
        g2 = F.relu(self.gate_r2_fc1(torch.cat([gap_key, gap_r2], dim=1)))
        g2 = self.gate_r2_fc2(g2).sigmoid()

        # 逐通道门控 + 投影 + 残差
        g1 = g1.view(B, C, 1, 1)
        g2 = g2.view(B, C, 1, 1)
        fused = g1 * ref1 + g2 * ref2          # 门控 ref 信息
        output = key + self.proj(fused)        # 残差连接
        return output

    def get_gate_weights(self, key, ref1, ref2):
        """
        （调试/可视化用）返回两个 ref 帧的 per-channel gate 权重。
        不会修改模型状态。
        """
        with torch.no_grad():
            gap_key = key.mean([2, 3])
            gap_r1  = ref1.mean([2, 3])
            gap_r2  = ref2.mean([2, 3])
            g1 = F.relu(self.gate_r1_fc1(torch.cat([gap_key, gap_r1], dim=1)))
            g1 = self.gate_r1_fc2(g1).sigmoid()
            g2 = F.relu(self.gate_r2_fc1(torch.cat([gap_key, gap_r2], dim=1)))
            g2 = self.gate_r2_fc2(g2).sigmoid()
        return g1, g2  # 各 (B, C)


# ═══════════════════════════════════════════════════════════
# AdaptiveFusionTSA — Temporal Self-Attention (帧间嵌入相似度)
# ═══════════════════════════════════════════════════════════

class AdaptiveFusionTSA(nn.Module):
    """Temporal Self-Attention 融合：帧间嵌入相似度驱动的逐像素加权。
    
    核心思路：将 key 和 ref 投影到不同 embedding 空间，
    逐像素点积计算相似度，sigmoid 得到权重，再拼接融合。
    
    与 spatial gate 的区别：
      - spatial gate 只看自己 (Conv([key,ref,|key-ref|]) → sigmoid)
      - TSA 看的是 key 和 ref 的关系 (emb_key · emb_ref → sigmoid)
      有 key 作为锚点，信号更强。
    """
    def __init__(self, channels, embed_ch=None, nonneg=False):
        super().__init__()
        if embed_ch is None:
            embed_ch = channels
        self.embed_ch = embed_ch
        self.nonneg = nonneg

        # key 投影：把自己映射到"匹配锚点"空间
        self.key_proj = nn.Conv2d(channels, embed_ch, 3, 1, 1)
        # ref 投影：把邻居帧映射到"可匹配"空间（ref1/ref2 共享）
        self.ref_proj = nn.Conv2d(channels, embed_ch, 3, 1, 1)

        # 融合卷积：3 个加权帧拼接 → 原通道
        self.feat_fusion = nn.Conv2d(channels * 3, channels, 1, 1, 0)
        
        # 初始化：投影近似恒等，融合零初始化（保持残差 identity 启动）
        nn.init.zeros_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        with torch.no_grad():
            for i in range(min(channels, embed_ch)):
                self.key_proj.weight[i, i, 1, 1] = 1.0
        
        nn.init.zeros_(self.ref_proj.weight)
        nn.init.zeros_(self.ref_proj.bias)
        with torch.no_grad():
            for i in range(min(channels, embed_ch)):
                self.ref_proj.weight[i, i, 1, 1] = 1.0
        
        # 非零初始化：避免 relu 在 0 处阻断梯度。正偏置保证初始时有少量正向贡献。
        nn.init.kaiming_normal_(self.feat_fusion.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.feat_fusion.bias, 0.1)

    def forward(self, key, ref1, ref2):
        B, C, H, W = key.shape
        
        # Step 1: 投影到 embedding 空间
        emb_key = self.key_proj(key)      # temporal_attn1: 参考帧→匹配锚点
        emb_r1  = self.ref_proj(ref1)     # temporal_attn2: 邻居帧→可匹配
        emb_r2  = self.ref_proj(ref2)
        emb_key_nbr = self.ref_proj(key)  # key 也作为"邻居帧"过 temporal_attn2
        
        # Step 2: 逐像素点积 + sigmoid（key 自身也在邻居侧参与点积）
        corr_key = torch.sum(emb_key * emb_key_nbr, dim=1)  # key↔key
        corr_r1  = torch.sum(emb_key * emb_r1, dim=1)       # key↔ref1
        corr_r2  = torch.sum(emb_key * emb_r2, dim=1)       # key↔ref2
        
        w_key = torch.sigmoid(corr_key).unsqueeze(1)  # (B,1,H,W)
        w_r1  = torch.sigmoid(corr_r1).unsqueeze(1)
        w_r2  = torch.sigmoid(corr_r2).unsqueeze(1)
        
        # Step 3: 逐帧加权 + 拼接 + 融合卷积
        key_w = key * w_key
        r1_w  = ref1 * w_r1
        r2_w  = ref2 * w_r2
        
        fused = torch.cat([key_w, r1_w, r2_w], dim=1)  # (B, 3C, H, W)
        feat_out = self.feat_fusion(fused)
        if self.nonneg:
            feat_out = F.relu(feat_out)                # 只做加法，不做减法
        output = key + feat_out                         # 残差
        return output

    def get_gate_weights(self, key, ref1, ref2):
        """返回 temporal attention 权重（兼容接口）。"""
        with torch.no_grad():
            emb_key = self.key_proj(key)
            emb_r1  = self.ref_proj(ref1)
            emb_r2  = self.ref_proj(ref2)
            w_r1 = torch.sigmoid(torch.sum(emb_key * emb_r1, dim=1, keepdim=True))
            w_r2 = torch.sigmoid(torch.sum(emb_key * emb_r2, dim=1, keepdim=True))
        return w_r1, w_r2


# ═══════════════════════════════════════════════════════════
# AdaptiveFusionSpatial — Per-Pixel Spatial Gate (遮挡场景)
# ═══════════════════════════════════════════════════════════

class AdaptiveFusionAvg(nn.Module):
    """Parameter-free arithmetic mean of key and the two aligned references.

    Supplementary avg experiment: unlike the historical residual-projection
    module with this name, the definition is exactly (key + ref1 + ref2) / 3.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, key, ref1, ref2):
        return (key + ref1 + ref2) / 3.0

    def get_gate_weights(self, key, ref1, ref2):
        B, _, H, W = key.shape
        weight = key.new_full((B, 1, H, W), 1.0 / 3.0)
        return weight, weight.clone()


class AdaptiveFusionConcat(nn.Module):
    """拼接融合：将 key, ref1, ref2 沿通道拼起来，1×1 投影回原维度。
    
    最朴素的多帧融合：不做任何门控，纯靠 1×1 conv 学怎么拼。
    """
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels * 3, channels, 1, 1, 0)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, key, ref1, ref2):
        fused = torch.cat([key, ref1, ref2], dim=1)
        return key + self.proj(fused)

    def get_gate_weights(self, key, ref1, ref2):
        """无 gate，返回虚拟权重。"""
        B, _, H, W = key.shape
        return (torch.ones(B, 1, H, W, device=key.device) * 0.5,
                torch.ones(B, 1, H, W, device=key.device) * 0.5)


class AdaptiveFusionSpatial(nn.Module):
    """
    空间门控融合模块：每像素独立 gate，应对局部遮挡。

    与 AdaptiveFusion (per-channel gate) 的区别：
      - gate predictor 输出 (B, 1, H, W)，每个像素独立决策
      - 初始化负偏置（bias=-2），训练早期 gate≈0.12，比 channel gate 更保守
      - 每个 ref 帧独立的 gate predictor（不对称门控）

    forward:
        s_i = sigmoid(Conv(ReLU(Conv(cat(key, ref_i)))))  ∈ [0,1]^H×W
        output = key + proj(s_1 ⊙ ref_1 + s_2 ⊙ ref_2)
    """
    def __init__(self, channels, gate_hidden=32, gate_init_bias=-2.0, use_diff=True,
                 fusion_mode='gate'):
        super().__init__()
        self.use_diff = use_diff
        self.fusion_mode = fusion_mode
        in_ch = channels * (3 if use_diff else 2)

        if fusion_mode == 'avg':
            # 简单平均模式：不需要 gate predictor，只保留 proj
            self.gate_r1_conv1 = None
            self.gate_r1_conv2 = None
            self.gate_r2_conv1 = None
            self.gate_r2_conv2 = None
        else:
            # Gate predictor for ref1: cat(key, ref1, [|key-ref1|]) → hidden → 1 → sigmoid
            self.gate_r1_conv1 = nn.Conv2d(in_ch, gate_hidden, 3, 1, 1)
            self.gate_r1_conv2 = nn.Conv2d(gate_hidden, 1, 3, 1, 1)

            # Gate predictor for ref2（独立权重）
            self.gate_r2_conv1 = nn.Conv2d(in_ch, gate_hidden, 3, 1, 1)
            self.gate_r2_conv2 = nn.Conv2d(gate_hidden, 1, 3, 1, 1)

            # ── 初始化 ──
            for conv in [self.gate_r1_conv1, self.gate_r2_conv1]:
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(conv.bias)
            for conv in [self.gate_r1_conv2, self.gate_r2_conv2]:
                nn.init.zeros_(conv.weight)
                nn.init.constant_(conv.bias, gate_init_bias)

        # 输出投影
        self.proj = nn.Conv2d(channels, channels, 1, 1, 0)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, key, ref1, ref2):
        B, C, H, W = key.shape

        if self.fusion_mode == 'avg':
            # 简单平均融合: (ref1 + ref2) / 2
            fused = (ref1 + ref2) / 2.0
        else:
            if self.use_diff:
                s1 = F.relu(self.gate_r1_conv1(torch.cat([key, ref1, (key-ref1).abs()], dim=1)))
                s2 = F.relu(self.gate_r2_conv1(torch.cat([key, ref2, (key-ref2).abs()], dim=1)))
            else:
                s1 = F.relu(self.gate_r1_conv1(torch.cat([key, ref1], dim=1)))
                s2 = F.relu(self.gate_r2_conv1(torch.cat([key, ref2], dim=1)))
            s1 = self.gate_r1_conv2(s1).sigmoid()
            s2 = self.gate_r2_conv2(s2).sigmoid()
            fused = s1 * ref1 + s2 * ref2

        output = key + self.proj(fused)
        return output

    def get_gate_weights(self, key, ref1, ref2):
        if self.fusion_mode == 'avg':
            B, _, H, W = key.shape
            half = torch.full((B, 1, H, W), 0.5, device=key.device)
            return half, half
        with torch.no_grad():
            if self.use_diff:
                s1 = F.relu(self.gate_r1_conv1(torch.cat([key, ref1, (key-ref1).abs()], dim=1)))
                s2 = F.relu(self.gate_r2_conv1(torch.cat([key, ref2, (key-ref2).abs()], dim=1)))
            else:
                s1 = F.relu(self.gate_r1_conv1(torch.cat([key, ref1], dim=1)))
                s2 = F.relu(self.gate_r2_conv1(torch.cat([key, ref2], dim=1)))
            s1 = self.gate_r1_conv2(s1).sigmoid()
            s2 = self.gate_r2_conv2(s2).sigmoid()
        return s1, s2


# ═══════════════════════════════════════════════════════════
# MultiFrameFusionV2: 每个特征层独立的 DeformAlignV2 + AdaptiveFusion
# ═══════════════════════════════════════════════════════════

FEATURE_LEVELS = ('p2', 'p3', 'p4')
LEVEL_TO_IDX = {level: idx for idx, level in enumerate(FEATURE_LEVELS)}
FUSION_CHANNELS = {
    'p2': 32,
    'p3': 64,
    'p4': 128,
}

# 不同单帧基座的特征层级/通道配置
#   p2l:    DarkNetP2_lite  → (p2, p3, p4) = (32, 64, 128), P3+P4 双尺度头
#           (P2 只经 FPN 融合, 不做 P2 头旁路; 与单帧 PAD p2l 一致)
#   p2l_p4: DarkNetP2_lite  → (p2, p3, p4) = (32, 64, 128), P4 单尺度头
#           (旧 YOLO 头配置; PAD 头下与 p2l 同构)
#   p34l:   DarkNet_lite    → (p3, p4)     = (64, 128),     P3+P4 双尺度头
BASE_SETTINGS = {
    'p2l': {
        'feature_levels': ('p2', 'p3', 'p4'),
        'fusion_channels': {'p2': 32, 'p3': 64, 'p4': 128},
        'detect_levels': ('p3', 'p4'),
    },
    'p2l_p4': {
        'feature_levels': ('p2', 'p3', 'p4'),
        'fusion_channels': {'p2': 32, 'p3': 64, 'p4': 128},
        'detect_levels': ('p4',),
    },
    'p34l': {
        'feature_levels': ('p3', 'p4'),
        'fusion_channels': {'p3': 64, 'p4': 128},
        'detect_levels': ('p3', 'p4'),
    },
}


def validate_fusion_levels(fusion_levels):
    """允许 P3、P4、P3+P4，以及 P2/P3/P4 全层融合。

    P2 特征图较大 (stride4)，默认不开启；`p2,p3,p4` 用于显式实验
    （DCN 同时作用于 P2/P3/P4）。
    """
    levels = list(fusion_levels)
    allowed = (['p2'], ['p3'], ['p4'], ['p2', 'p3'], ['p3', 'p4'],
               ['p2', 'p4'], ['p2', 'p3', 'p4'])
    if levels not in allowed:
        raise ValueError(
            f'不支持的 fusion_levels={levels}')
    return levels


def _alignment_loss_impl(fusion_levels, align_ref1, align_ref2,
                          last_aligned_r1, last_aligned_r2,
                          key_feats, ref1_feats, ref2_feats, level_to_idx=None):
    """get_alignment_loss 共享实现，供 MultiFrameFusionV2/V3 复用。"""
    if level_to_idx is None:
        level_to_idx = LEVEL_TO_IDX
    total_loss = 0.0
    count = 0

    for level in fusion_levels:
        idx = level_to_idx[level]

        if level in last_aligned_r1 and level in last_aligned_r2:
            aligned1 = last_aligned_r1[level]
            aligned2 = last_aligned_r2[level]
        else:
            if key_feats is None or ref1_feats is None or ref2_feats is None:
                raise RuntimeError(
                    'get_alignment_loss: no cached aligned features and no feats provided')
            key = key_feats[idx]
            ref1 = ref1_feats[idx]
            ref2 = ref2_feats[idx]
            aligned1 = align_ref1[level](ref1, key)
            aligned2 = align_ref2[level](ref2, key)

        if key_feats is not None:
            key = key_feats[idx]
        else:
            raise RuntimeError('get_alignment_loss: key_feats required for L1 computation')

        total_loss += F.l1_loss(aligned1, key) + F.l1_loss(aligned2, key)
        count += 2

    return total_loss / count


class MultiFrameFusionV2(nn.Module):
    """
    多帧融合模块 V2：使用 DeformAlignV2 (DCNv2 + mask)。

    每个特征层独立的对齐+融合实例，可配置 fusion_levels。
    forward() 会缓存对齐后特征，get_alignment_loss() 直接复用，
    避免训练时双重前向。
    """
    def __init__(self, fusion_levels=None, align_hidden_ch=None, use_sharpener=False,
                 fusion_gate_type='channel', gate_init_bias=-2.0,
                 gate_use_diff=True, fusion_mode='gate', no_align=False,
                 cascade_align=False, convgru_kernel=3, convgru_layers=1,
                 feature_levels=None, fusion_channels=None,
                 shared_offset_level=None, offset_reuse_mode='bilinear'):
        super().__init__()
        if fusion_levels is None:
            fusion_levels = ['p4']
        self.fusion_levels = validate_fusion_levels(fusion_levels)
        self.feature_levels = tuple(feature_levels) if feature_levels else FEATURE_LEVELS
        self.level_to_idx = {level: idx for idx, level in enumerate(self.feature_levels)}
        self.fusion_channels = fusion_channels if fusion_channels else FUSION_CHANNELS
        self.fusion_gate_type = fusion_gate_type
        self.no_align = no_align
        self.cascade_align = cascade_align
        if shared_offset_level in ('', 'none'):
            shared_offset_level = None
        if shared_offset_level is not None and shared_offset_level not in self.fusion_levels:
            raise ValueError(
                f'shared_offset_level={shared_offset_level!r} 不在 fusion_levels='
                f'{self.fusion_levels} 中')
        self.shared_offset_level = shared_offset_level
        if offset_reuse_mode not in ('bilinear', 'nearest', 'global_mean', 'avg_pool'):
            raise ValueError(f'unknown offset_reuse_mode={offset_reuse_mode!r}')
        self.offset_reuse_mode = offset_reuse_mode

        if fusion_gate_type == 'concat':
            fusion_cls = AdaptiveFusionConcat
            fusion_kwargs = {'channels': 0}
        elif fusion_gate_type == 'avg':
            fusion_cls = AdaptiveFusionAvg
            fusion_kwargs = {'channels': 0}
        elif fusion_gate_type == 'tsa':
            fusion_cls = AdaptiveFusionTSA
            fusion_kwargs = {'channels': 0}
        elif fusion_gate_type == 'tsa_add':
            # TSA + ReLU 约束 proj 输出 ≥0，只做加法不做减法
            fusion_cls = AdaptiveFusionTSA
            fusion_kwargs = {'channels': 0, 'nonneg': True}
        elif fusion_gate_type == 'convgru':
            from .convgru_fusion import ConvGRUFusion, get_gate_weights_for_convgru
            fusion_cls = ConvGRUFusion
            self._convgru_get_gate_weights = get_gate_weights_for_convgru
            fusion_kwargs = {'channels': 0, 'kernel_size': convgru_kernel,
                             'num_layers': convgru_layers}
        elif fusion_gate_type == 'spatial':
            fusion_cls = AdaptiveFusionSpatial
            fusion_kwargs = {'channels': 0, 'gate_init_bias': gate_init_bias,
                             'use_diff': gate_use_diff, 'fusion_mode': fusion_mode}
        else:
            fusion_cls = AdaptiveFusion
            fusion_kwargs = {'channels': 0}

        self.align_ref1 = nn.ModuleDict()
        self.align_ref2 = nn.ModuleDict()
        self.fusion = nn.ModuleDict()

        # 缓存最近一次 forward 的对齐特征，供 get_alignment_loss 复用
        self._last_aligned_r1 = {}
        self._last_aligned_r2 = {}

        for level in fusion_levels:
            ch = self.fusion_channels[level]
            if no_align:
                self.align_ref1[level] = _NoOp()
                self.align_ref2[level] = _NoOp()
            elif shared_offset_level is not None and level != shared_offset_level:
                # 只在该层做一次轻量 DCN，offset/mask 从共享层复用
                self.align_ref1[level] = OffsetReuseAlign(ch, reuse_mode=offset_reuse_mode)
                self.align_ref2[level] = OffsetReuseAlign(ch, reuse_mode=offset_reuse_mode)
            else:
                self.align_ref1[level] = DeformAlignV2(ch, align_hidden_ch, use_sharpener=use_sharpener)
                self.align_ref2[level] = DeformAlignV2(ch, align_hidden_ch, use_sharpener=use_sharpener)
            self.fusion[level] = fusion_cls(ch, **{k: v for k, v in fusion_kwargs.items() if k != 'channels'})

    def forward(self, key_feats, ref1_feats, ref2_feats):
        """
        key_feats/ref1_feats/ref2_feats: 按 self.feature_levels 顺序的特征元组
        返回: 同序元组，未选层保持关键帧特征
        副作用: 缓存所选融合层的对齐后特征
        """
        fused = list(key_feats)

        # 共享 offset 时先算共享层，再算复用层
        order = list(self.fusion_levels)
        if self.shared_offset_level is not None:
            order = [self.shared_offset_level] + [
                l for l in order if l != self.shared_offset_level]
        src1 = src2 = None
        for level in order:
            idx = self.level_to_idx[level]
            key = key_feats[idx]
            ref1 = ref1_feats[idx]
            ref2 = ref2_feats[idx]

            if isinstance(self.align_ref1[level], OffsetReuseAlign):
                aligned1 = self.align_ref1[level](ref1, key, src_module=src1)
                aligned2 = self.align_ref2[level](ref2, key, src_module=src2)
            else:
                aligned1 = self.align_ref1[level](ref1, key)
                aligned2 = self.align_ref2[level](ref2, key)
            if level == self.shared_offset_level:
                src1, src2 = self.align_ref1[level], self.align_ref2[level]
            self._last_aligned_r1[level] = aligned1
            self._last_aligned_r2[level] = aligned2
            fused[idx] = self.fusion[level](key, aligned1, aligned2)

        return tuple(fused)

    def get_alignment_loss(self, key_feats=None, ref1_feats=None, ref2_feats=None):
        """对齐 L1 辅助 loss。优先复用 forward() 缓存，缓存为空时回退重建。"""
        return _alignment_loss_impl(
            self.fusion_levels, self.align_ref1, self.align_ref2,
            self._last_aligned_r1, self._last_aligned_r2,
            key_feats, ref1_feats, ref2_feats, level_to_idx=self.level_to_idx)

    def get_aligned_features(self):
        """
        返回最近一次 forward() 缓存的对齐特征。
        """
        return dict(self._last_aligned_r1), dict(self._last_aligned_r2)

    def get_gate_weights(self, level, key, ref1, ref2):
        """
        兼容接口：获取指定层的门控权重。
        ConvGRU 模式下返回虚拟权重（该模式由时序状态管理，无显式gate）。
        Spatial/Channel 模式下委托给融合模块。
        """
        fusion_mod = self.fusion[level]
        if self.fusion_gate_type == 'convgru':
            return self._convgru_get_gate_weights(key, ref1, ref2)
        return fusion_mod.get_gate_weights(key, ref1, ref2)


# ═══════════════════════════════════════════════════════════
# MultiFrameFusionV3: HybridAlign (PCD + DeformAlignV2) 可选
# ═══════════════════════════════════════════════════════════

class MultiFrameFusionV3(nn.Module):
    """
    多帧融合模块 V3：支持 HybridAlign（PCD粗对齐 + DCNv2精修）。

    每个特征层可独立选择对齐方式：
      - 'hybrid': PCD 金字塔粗对齐 + DeformAlignV2 精修（大位移场景）
      - 'dav2': 纯 DeformAlignV2（小位移场景）

    用法:
        fusion = MultiFrameFusionV3(
            fusion_levels=['p4'],
            align_mode={'p4': 'hybrid'},
        )
    """
    def __init__(self, fusion_levels=None, align_mode=None,
                 align_hidden_ch=None, pcd_num_feat=64, use_sharpener=False,
                 fusion_gate_type='channel', gate_init_bias=-2.0,
                 gate_use_diff=True, fusion_mode='gate', no_align=False,
                 feature_levels=None, fusion_channels=None):
        super().__init__()
        if fusion_levels is None:
            fusion_levels = ['p4']
        fusion_levels = validate_fusion_levels(fusion_levels)
        if align_mode is None:
            align_mode = {l: 'hybrid' for l in fusion_levels}
        self.fusion_levels = fusion_levels
        self.feature_levels = tuple(feature_levels) if feature_levels else FEATURE_LEVELS
        self.level_to_idx = {level: idx for idx, level in enumerate(self.feature_levels)}
        self.fusion_channels = fusion_channels if fusion_channels else FUSION_CHANNELS
        self.align_mode = align_mode
        self.fusion_gate_type = fusion_gate_type
        self.no_align = no_align

        fusion_cls = AdaptiveFusionSpatial if fusion_gate_type == 'spatial' else AdaptiveFusion
        fusion_kwargs = {}
        if fusion_gate_type == 'spatial':
            fusion_kwargs = {'gate_init_bias': gate_init_bias, 'use_diff': gate_use_diff,
                             'fusion_mode': fusion_mode}

        from .hybrid_align import HybridAlign

        self.align_ref1 = nn.ModuleDict()
        self.align_ref2 = nn.ModuleDict()
        self.fusion = nn.ModuleDict()

        self._last_aligned_r1 = {}
        self._last_aligned_r2 = {}

        for level in fusion_levels:
            ch = self.fusion_channels[level]
            mode = align_mode.get(level, 'dav2')
            if no_align:
                self.align_ref1[level] = _NoOp()
                self.align_ref2[level] = _NoOp()
            elif mode == 'hybrid':
                self.align_ref1[level] = HybridAlign(ch, num_feat=pcd_num_feat,
                                                      hidden_ch=align_hidden_ch)
                self.align_ref2[level] = HybridAlign(ch, num_feat=pcd_num_feat,
                                                      hidden_ch=align_hidden_ch)
            elif mode == 'blockmatch':
                from .pyramid_match import PyramidMatchAlign
                self.align_ref1[level] = PyramidMatchAlign(ch)
                self.align_ref2[level] = PyramidMatchAlign(ch)
            elif mode == 'translation':
                from .translation_align import RigidAlign
                self.align_ref1[level] = RigidAlign(ch)
                self.align_ref2[level] = RigidAlign(ch)
            elif mode == 'blockmatch_dav2':
                from .pyramid_match import PyramidMatchAlign
                class BlockMatchDAV2(nn.Module):
                    def __init__(self, ch, ahc):
                        super().__init__()
                        self.bm = PyramidMatchAlign(ch)
                        self.dav2 = DeformAlignV2(ch, ahc)
                    def forward(self, nbr, ref):
                        coarse = self.bm(nbr, ref)
                        return self.dav2(coarse, ref)
                self.align_ref1[level] = BlockMatchDAV2(ch, align_hidden_ch)
                self.align_ref2[level] = BlockMatchDAV2(ch, align_hidden_ch)
            else:
                self.align_ref1[level] = DeformAlignV2(ch, align_hidden_ch, use_sharpener=use_sharpener)
                self.align_ref2[level] = DeformAlignV2(ch, align_hidden_ch, use_sharpener=use_sharpener)
            self.fusion[level] = fusion_cls(ch, **fusion_kwargs)

    def forward(self, key_feats, ref1_feats, ref2_feats):
        fused = list(key_feats)

        for level in self.fusion_levels:
            idx = self.level_to_idx[level]
            key = key_feats[idx]
            ref1 = ref1_feats[idx]
            ref2 = ref2_feats[idx]

            aligned1 = self.align_ref1[level](ref1, key)
            aligned2 = self.align_ref2[level](ref2, key)
            self._last_aligned_r1[level] = aligned1
            self._last_aligned_r2[level] = aligned2
            fused[idx] = self.fusion[level](key, aligned1, aligned2)

        return tuple(fused)

    def get_alignment_loss(self, key_feats=None, ref1_feats=None, ref2_feats=None):
        """对齐 L1 辅助 loss。优先复用 forward() 缓存，缓存为空时回退重建。"""
        return _alignment_loss_impl(
            self.fusion_levels, self.align_ref1, self.align_ref2,
            self._last_aligned_r1, self._last_aligned_r2,
            key_feats, ref1_feats, ref2_feats, level_to_idx=self.level_to_idx)

    def get_aligned_features(self):
        """
        返回最近一次 forward() 缓存的对齐特征。
        用于可视化和调试，返回 dict 的浅拷贝。
        """
        return dict(self._last_aligned_r1), dict(self._last_aligned_r2)


# 向后兼容别名
DeformAlign = DeformAlignV2
MultiFrameFusion = MultiFrameFusionV2
