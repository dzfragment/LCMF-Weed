"""
MultiFrameYOLO: P2L-P4 多帧融合检测模型。

结构:
    3帧图像 → 共享 DarkNetP2_lite → 每帧 (P2, P3, P4)
    可在 P3、P4 或 P3+P4 执行多帧对齐与融合，P2 始终使用关键帧特征
    (key_P2, temporal_P3, temporal_P4) → DarkFPN_P2_lite
    仅将 N4 送入单尺度 Head → boxes + keypoints
"""
import torch
import torch.nn as nn
import sys
import os

# 添加 YOLOv8-pose 路径
_YOLO_PATH = os.path.join(os.path.dirname(__file__), '../../YOLOv8-pose')
if _YOLO_PATH not in sys.path:
    sys.path.insert(0, _YOLO_PATH)

from nets.nn import DarkNetP2_lite, DarkFPN_P2_lite, DarkNet_lite, DarkFPN_lite, Head
from pad_core import build_point_head, normalize_head_cfg
from .modules import (FEATURE_LEVELS, LEVEL_TO_IDX, BASE_SETTINGS,
                      MultiFrameFusion, MultiFrameFusionV3)


class MultiFrameYOLO(nn.Module):
    """
    多帧融合 YOLO 检测模型

    参数:
        num_classes: 类别数
        kpt_shape: 关键点形状 (num_kpt, dim)，默认 (1, 2) 表示1个关键点2维坐标
        fusion_levels: 支持 ['p3']、['p4'] 或 ['p3','p4']；P2 禁止融合
        pretrained_path: YOLO_P2_lite_P4 的 best.pt
    """
    def __init__(self, num_classes=1, kpt_shape=(1, 2),
                 fusion_levels=None, pretrained_path=None,
                 fusion_version=2, align_mode=None, use_sharpener=False,
                 align_hidden_ch=None, fusion_gate_type='channel',
                 gate_init_bias=-2.0, convgru_kernel=3, convgru_layers=1,
                 global_align_stage='none', fft_downsample=4,
                 fft_chain_align=True, fft_mse_check=True, fft_normalize=True,
                 consistent_align=False, base_model='p2l_p4',
                 pad_head=False, pad_head_cfg=None, offset_reuse_mode='bilinear',
                 global_head=False, global_head_level='p3',
                 global_head_hidden=32, global_head_max_shift=32.0):
        super().__init__()
        if base_model not in BASE_SETTINGS:
            raise ValueError(
                f'base_model 仅支持 {sorted(BASE_SETTINGS)}，收到 {base_model!r}')
        self.base_model = base_model
        base_cfg = BASE_SETTINGS[base_model]
        if fusion_levels is None:
            fusion_levels = ['p4']
        fusion_levels = list(fusion_levels)
        if fusion_levels not in (['p2'], ['p3'], ['p4'], ['p2', 'p3'],
                                 ['p3', 'p4'], ['p2', 'p4'],
                                 ['p2', 'p3', 'p4']):
            raise ValueError(
                '多帧模型支持 p2/p3/p4 的单层、双、三层融合，'
                f'收到 fusion_levels={fusion_levels}')
        self.fusion_levels = fusion_levels
        self.feature_levels = base_cfg['feature_levels']
        self.level_to_idx = {level: idx for idx, level in enumerate(self.feature_levels)}
        self.fusion_channels = base_cfg['fusion_channels']
        self.detect_levels = base_cfg['detect_levels']
        self.kpt_shape = tuple(kpt_shape)
        self.global_head_enabled = bool(global_head)
        self.global_head_level = global_head_level
        self.global_head_hidden = int(global_head_hidden)
        self.global_head_max_shift = float(global_head_max_shift)
        if global_head:
            if global_head_level not in self.feature_levels:
                raise ValueError(
                    f'global_head_level={global_head_level!r} 不在 '
                    f'feature_levels={self.feature_levels} 中')
            from .modules import GlobalShiftHead
            ch = self.fusion_channels[global_head_level]
            self.global_head_r1 = GlobalShiftHead(ch, global_head_hidden)
            self.global_head_r2 = GlobalShiftHead(ch, global_head_hidden)
        else:
            self.global_head_r1 = None
            self.global_head_r2 = None
        self.fusion_version = fusion_version
        self.use_sharpener = use_sharpener
        self.align_hidden_ch = align_hidden_ch
        self.fusion_gate_type = fusion_gate_type
        self.gate_init_bias = gate_init_bias
        if global_align_stage not in ('none', 'feature'):
            raise ValueError(
                "MultiFrameYOLO 原生 global_align_stage 仅支持 'none'/'feature'; "
                "旧 image 模式继续使用 apply_fft_image_align()")
        self.global_align_stage = global_align_stage
        self.fft_downsample = fft_downsample
        self.fft_chain_align = fft_chain_align
        self.fft_mse_check = fft_mse_check
        self.fft_normalize = fft_normalize
        self.consistent_align = consistent_align

        # feature 模式下 FFT 只估计 RGB 全局位移，真正平移发生在 P2/P3/P4。
        # Sobel buffer persistent=False，不改变 checkpoint 参数结构。
        if global_align_stage == 'feature':
            if consistent_align:
                # 一致性管线: 三帧 top-K + 矢量自洽约束估计位移 (推荐)
                from fft_align import ConsistentAligner
                self.consistent_aligner = ConsistentAligner(
                    downsample=fft_downsample, k=5)
                self.fft_align_r1 = None
                self.fft_align_r2 = None
            else:
                from .fft_align import FFTAlign
                self.consistent_aligner = None
                self.fft_align_r1 = FFTAlign(
                    downsample=fft_downsample, do_mse_check=fft_mse_check,
                    normalize=fft_normalize)
                self.fft_align_r2 = FFTAlign(
                    downsample=fft_downsample, do_mse_check=fft_mse_check,
                    normalize=fft_normalize)
        else:
            self.consistent_aligner = None
            self.fft_align_r1 = None
            self.fft_align_r2 = None

        # 缓存最近一次 forward 的特征，供 AlignLoss 和分析脚本复用。
        # _last_ref* 表示真正进入 DCN 的特征；feature 模式下即 FFT 粗对齐后特征。
        self._last_key_feats = None
        self._last_ref1_feats = None
        self._last_ref2_feats = None
        self._last_raw_ref1_feats = None
        self._last_raw_ref2_feats = None
        self._last_fft_displacements = None
        self._last_fft_valid_masks = None
        self._last_global_shift = None

        # 与单帧 YOLO_P2_lite_P4 相同的架构超参
        width = [3, 16, 32, 64, 128, 256]
        depth = [1, 2, 2]

        # 多帧融合模块
        if fusion_version == 3:
            self.fusion = MultiFrameFusionV3(fusion_levels, align_mode=align_mode,
                                              align_hidden_ch=align_hidden_ch,
                                              use_sharpener=use_sharpener,
                                              fusion_gate_type=fusion_gate_type,
                                              gate_init_bias=gate_init_bias,
                                              convgru_kernel=convgru_kernel,
                                              convgru_layers=convgru_layers,
                                              feature_levels=self.feature_levels,
                                              fusion_channels=self.fusion_channels)
        else:
            self.fusion = MultiFrameFusion(fusion_levels, align_hidden_ch=align_hidden_ch,
                                            use_sharpener=use_sharpener,
                                            fusion_gate_type=fusion_gate_type,
                                            gate_init_bias=gate_init_bias,
                                            convgru_kernel=convgru_kernel,
                                            convgru_layers=convgru_layers,
                                            feature_levels=self.feature_levels,
                                            fusion_channels=self.fusion_channels,
                                            offset_reuse_mode=offset_reuse_mode)

        # 共享 backbone + FPN（按基座选择），输出 N3/N4
        if base_model in ('p2l', 'p2l_p4'):
            self.backbone = DarkNetP2_lite(width, depth)
            self.fpn = DarkFPN_P2_lite(width, depth)
        else:  # p34l
            self.backbone = DarkNet_lite(width, depth)
            self.fpn = DarkFPN_lite(width, depth)

        # 检测头：p2l_p4 仅 P4 单尺度；p2l/p34l 为 P3+P4 双尺度
        # pad_head=True 时替换为 PAD 点热图头（点锚定，不使用 TAL/DFL）
        self.pad_head = bool(pad_head)
        self.pad_head_cfg = normalize_head_cfg(pad_head_cfg) if self.pad_head else None
        if self.pad_head:
            if self.pad_head_cfg['use_p2'] and base_model not in ('p2l', 'p2l_p4'):
                raise ValueError(
                    'PAD use_p2 需要 backbone 的 P2 特征，仅支持 '
                    'base_model=p2l/p2l_p4')
            head_ch = (width[3], width[4])
            self.head = build_point_head(head_ch, self.pad_head_cfg)
            self.stride = torch.tensor(
                [4.0 if self.pad_head_cfg['out_stride4'] else 8.0])
            self.head.stride = self.stride
        else:
            head_ch = (width[4],) if base_model == 'p2l_p4' else (width[3], width[4])
            self.head = Head(num_classes, kpt_shape, head_ch)

            # 计算 stride（与 detect_forward 的选层一致）
            img_dummy = torch.zeros(1, 3, 256, 256)
            detect_feats = self._select_detect_feats(self.fpn(self.backbone(img_dummy)))
            self.head.stride = torch.tensor([256 / x.shape[-2] for x in detect_feats])
            self.stride = self.head.stride
            self.head.initialize_biases()

            # 严格冻结 DFL 卷积权重（与原始 YOLO 一致）
            if hasattr(self.head, 'dfl') and hasattr(self.head.dfl, 'conv'):
                self.head.dfl.conv.weight.requires_grad_(False)

        # 加载预训练权重
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def _extract_multiframe_features(self, frames):
        """提取三帧 P2/P3/P4 特征，并按配置施加 FFT 全局位移。"""
        key_img = frames[:, 0]
        ref1_img = frames[:, 1]
        ref2_img = frames[:, 2]
        image_hw = key_img.shape[-2:]

        # 位移估计不参与反向传播；输入显式转 float32，避免 AMP 下 FFT/Sobel
        # 的半精度兼容和数值稳定性问题。
        displacement = None
        if self.global_align_stage == 'feature':
            with torch.no_grad():
                if self.consistent_align:
                    ca = self.consistent_aligner
                    # 对齐器随模型 device 移动 (非 nn.Module, model.to 不覆盖)
                    if ca.aligner._sobel_x.device != ref1_img.device:
                        ca.aligner = ca.aligner.to(ref1_img.device)
                    _, _, dx1, dy1, dx2, dy2 = ca.align_batch(
                        key_img.float(), ref1_img.float(), ref2_img.float(),
                        return_displacement=True)
                    dx2_rel = dx2 - dx1
                    dy2_rel = dy2 - dy1
                    snr1 = snr2 = None
                elif self.fft_chain_align:
                    dx1, dy1, snr1 = self.fft_align_r1.find_displacement(
                        ref1_img.float(), key_img.float())
                    dx2_rel, dy2_rel, snr2 = self.fft_align_r2.find_displacement(
                        ref2_img.float(), ref1_img.float())
                    dx2 = dx1 + dx2_rel
                    dy2 = dy1 + dy2_rel
                else:
                    dx1, dy1, snr1 = self.fft_align_r1.find_displacement(
                        ref1_img.float(), key_img.float())
                    dx2, dy2, snr2 = self.fft_align_r2.find_displacement(
                        ref2_img.float(), key_img.float())
                    dx2_rel = dy2_rel = None
            displacement = {
                'ref1': {'dx': dx1, 'dy': dy1, 'snr': snr1},
                'ref2': {'dx': dx2, 'dy': dy2, 'snr': snr2},
                'ref2_relative': {'dx': dx2_rel, 'dy': dy2_rel}
                    if dx2_rel is not None else None,
            }

        # 所有帧都以未经平移的完整 RGB 图像进入共享 backbone。
        key_feats = self.backbone(key_img)
        raw_ref1_feats = self.backbone(ref1_img)
        raw_ref2_feats = self.backbone(ref2_img)
        ref1_feats = raw_ref1_feats
        ref2_feats = raw_ref2_feats
        valid_masks = None

        if displacement is not None:
            from .fft_align import apply_feature_shift
            shifted_r1, shifted_r2 = [], []
            masks_r1, masks_r2 = [], []
            for feat1, feat2 in zip(raw_ref1_feats, raw_ref2_feats):
                feat1_s, mask1 = apply_feature_shift(
                    feat1, dx1, dy1, image_hw, return_mask=True)
                feat2_s, mask2 = apply_feature_shift(
                    feat2, dx2, dy2, image_hw, return_mask=True)
                shifted_r1.append(feat1_s)
                shifted_r2.append(feat2_s)
                masks_r1.append(mask1)
                masks_r2.append(mask2)
            ref1_feats = tuple(shifted_r1)
            ref2_feats = tuple(shifted_r2)
            valid_masks = {'ref1': tuple(masks_r1), 'ref2': tuple(masks_r2)}

        # Optional learned global residual: predict one (dx,dy) per reference
        # from the post-FFT feature pair and apply it to every fused level.
        # This is deliberately separate from DCN's per-pixel local offsets.
        global_shift = None
        if self.global_head_r1 is not None:
            from .fft_align import apply_feature_shift
            idx = self.level_to_idx[self.global_head_level]
            key_l = key_feats[idx]
            Hf, Wf = ref1_feats[idx].shape[-2:]
            stride_x = float(image_hw[1]) / float(Wf)
            stride_y = float(image_hw[0]) / float(Hf)
            d1 = self.global_head_r1(key_l, ref1_feats[idx])  # (B,2) feat px
            d2 = self.global_head_r2(key_l, ref2_feats[idx])
            mx = self.global_head_max_shift
            dx1_g = torch.clamp(d1[:, 0] * stride_x, -mx, mx)
            dy1_g = torch.clamp(d1[:, 1] * stride_y, -mx, mx)
            dx2_g = torch.clamp(d2[:, 0] * stride_x, -mx, mx)
            dy2_g = torch.clamp(d2[:, 1] * stride_y, -mx, mx)
            new_r1, new_r2 = [], []
            for f in ref1_feats:
                sf, _ = apply_feature_shift(f, dx1_g, dy1_g, image_hw,
                                            return_mask=True)
                new_r1.append(sf)
            for f in ref2_feats:
                sf, _ = apply_feature_shift(f, dx2_g, dy2_g, image_hw,
                                            return_mask=True)
                new_r2.append(sf)
            ref1_feats = tuple(new_r1)
            ref2_feats = tuple(new_r2)
            global_shift = {
                'ref1': {'dx': dx1_g, 'dy': dy1_g},
                'ref2': {'dx': dx2_g, 'dy': dy2_g},
            }

        self._last_key_feats = tuple(key_feats)
        self._last_raw_ref1_feats = tuple(raw_ref1_feats)
        self._last_raw_ref2_feats = tuple(raw_ref2_feats)
        self._last_ref1_feats = tuple(ref1_feats)
        self._last_ref2_feats = tuple(ref2_feats)
        self._last_fft_displacements = displacement
        self._last_fft_valid_masks = valid_masks
        self._last_global_shift = global_shift
        return (self._last_key_feats, self._last_ref1_feats, self._last_ref2_feats,
                self._last_raw_ref1_feats, self._last_raw_ref2_feats)

    def _select_detect_feats(self, fpn_feats):
        """从 FPN 输出 (n3, n4) 中选出送入检测头的层（按基座配置）。"""
        if self.detect_levels == ('p4',):
            return [fpn_feats[1]]
        return [fpn_feats[0], fpn_feats[1]]

    def detect_forward(self, fused, p2=None):
        """融合特征 → FPN → 检测头（p2l_p4 用 N4，p2l/p34l 用 N3+N4）。

        PAD 头使用 FPN 的 N3+N4 两个尺度（可选 key 帧的 P2 细节）。
        """
        n3, n4 = self.fpn(fused)
        if self.pad_head:
            if self.pad_head_cfg['head_type'] == 'ms':
                return self.head([n3, n4], p2)
            return self.head(n3)
        return self.head(self._select_detect_feats((n3, n4)))

    def forward(self, frames):
        """
        frames: (B, 3, 3, H, W) — batch × 3帧 × 3通道 × 高 × 宽
        返回: Head 输出（与原始 YOLO 一致）
        """
        key_feats, ref1_feats, ref2_feats, _, _ = \
            self._extract_multiframe_features(frames)
        fused = self.fusion(key_feats, ref1_feats, ref2_feats)
        p2 = None
        if self.pad_head and self.pad_head_cfg['use_p2']:
            p2 = key_feats[self.level_to_idx['p2']]
        return self.detect_forward(fused, p2)

    def forward_with_features(self, frames):
        """
        前向传播并返回中间特征（用于可视化和调试）。

        调用 self.fusion() 保证逻辑与主 forward 一致，
        对齐特征从 fusion 的内部缓存读取。

        返回:
            outputs: Head 输出
            features: dict 包含 backbone 特征、对齐后特征、融合后特征
        """
        (key_feats, ref1_feats, ref2_feats,
         raw_ref1_feats, raw_ref2_feats) = self._extract_multiframe_features(frames)
        levels = self.feature_levels
        key_by_level = dict(zip(levels, key_feats))
        ref1_by_level = dict(zip(levels, ref1_feats))
        ref2_by_level = dict(zip(levels, ref2_feats))
        raw_ref1_by_level = dict(zip(levels, raw_ref1_feats))
        raw_ref2_by_level = dict(zip(levels, raw_ref2_feats))

        # 统一调用 fusion.forward()，避免逻辑漂移
        fused = self.fusion(key_feats, ref1_feats, ref2_feats)

        # 从 fusion 公开接口读取对齐后特征（避免直接访问私有缓存）
        aligned_r1, aligned_r2 = self.fusion.get_aligned_features()

        fused_feats = dict(zip(levels, fused))

        n3, n4 = self.fpn(fused)
        if self.pad_head:
            p2 = (key_feats[self.level_to_idx['p2']]
                  if self.pad_head_cfg['use_p2'] else None)
            if self.pad_head_cfg['head_type'] == 'ms':
                outputs = self.head([n3, n4], p2)
            else:
                outputs = self.head(n3)
            detect_feats = [n3, n4]
        else:
            detect_feats = self._select_detect_feats((n3, n4))
            outputs = self.head(detect_feats)

        features = {
            'key': key_by_level,
            # ref1/ref2 表示进入 DCN 前的各层特征。
            'ref1': ref1_by_level,
            'ref2': ref2_by_level,
            'raw_ref1': raw_ref1_by_level,
            'raw_ref2': raw_ref2_by_level,
            'fft_shifted_ref1': ref1_by_level
                if self.global_align_stage == 'feature' else None,
            'fft_shifted_ref2': ref2_by_level
                if self.global_align_stage == 'feature' else None,
            'fft_displacement': self._last_fft_displacements,
            'fft_valid_masks': self._last_fft_valid_masks,
            'aligned_r1': aligned_r1,
            'aligned_r2': aligned_r2,
            'fused': fused_feats,
            'fpn': {'p3': n3, 'p4': n4},
            'detect': dict(zip(self.detect_levels, detect_feats)),
        }
        return outputs, features

    def fuse(self):
        """融合 Conv+BN 层（推理加速）"""
        from nets.nn import fuse_conv, Conv
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self

    def load_pretrained(self, checkpoint_path):
        """
        从 YOLO_P2_lite_P4 的 best.pt 加载预训练权重。

        checkpoint 的 key 前缀映射：
            net.  → backbone.
            fpn.  → fpn.
            head. → head.
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in ckpt:
            model_data = ckpt['model']
            state_dict = model_data.state_dict() if hasattr(model_data, 'state_dict') else model_data
        else:
            state_dict = ckpt

        model_dict = self.state_dict()
        loaded = {}

        for ckpt_key, v in state_dict.items():
            # 前缀映射：net. → backbone.
            model_key = ckpt_key
            if ckpt_key.startswith('net.'):
                model_key = 'backbone.' + ckpt_key[4:]

            if model_key in model_dict and v.shape == model_dict[model_key].shape:
                loaded[model_key] = v.float()

        base_keys = {
            k for k in model_dict
            if k.startswith(('backbone.', 'fpn.', 'head.'))
        }
        missing_base = sorted(base_keys - loaded.keys())
        unexpected_base = sorted(
            k for k in state_dict
            if k.startswith(('net.', 'fpn.', 'head.'))
            and ('backbone.' + k[4:] if k.startswith('net.') else k) not in base_keys)
        if missing_base or unexpected_base:
            raise RuntimeError(
                f'预训练权重与 {self.base_model} 基座不匹配: '
                f'missing={missing_base[:5]} ({len(missing_base)}), '
                f'unexpected={unexpected_base[:5]} ({len(unexpected_base)})')
        self.load_state_dict(loaded, strict=False)
        print(f'[MultiFrameYOLO] {self.base_model} 基座权重完整加载: {len(loaded)}/{len(base_keys)}')

    def get_alignment_loss(self, frames=None):
        """
        计算对齐 L1 辅助 loss。
        优先复用 forward() 缓存的 backbone 特征，避免重复计算。
        frames 参数保留用于兼容独立调用（如 smoke_test），
        传入时会重新跑 backbone。
        """
        if self._last_key_feats is not None:
            # 复用 forward() 缓存的 backbone 特征
            key_feats = self._last_key_feats
            ref1_feats = self._last_ref1_feats
            ref2_feats = self._last_ref2_feats
        elif frames is not None:
            with torch.no_grad():
                key_feats = self.backbone(frames[:, 0])
                ref1_feats = self.backbone(frames[:, 1])
                ref2_feats = self.backbone(frames[:, 2])
        else:
            raise RuntimeError('get_alignment_loss: no cached features and no frames provided')

        return self.fusion.get_alignment_loss(key_feats, ref1_feats, ref2_feats)

    def freeze_backbone(self):
        """冻结 backbone（阶段2训练）"""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def freeze_fpn_head(self):
        """冻结 FPN 和 Head（阶段2训练）"""
        for p in self.fpn.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
            p.requires_grad_(False)
        # DFL 卷积始终冻结
        if hasattr(self.head, 'dfl') and hasattr(self.head.dfl, 'conv'):
            self.head.dfl.conv.weight.requires_grad_(False)

    def unfreeze_all(self):
        """解冻所有参数（阶段3训练）"""
        for p in self.parameters():
            p.requires_grad_(True)
        # DFL 卷积始终冻结
        if hasattr(self.head, 'dfl') and hasattr(self.head.dfl, 'conv'):
            self.head.dfl.conv.weight.requires_grad_(False)


def create_model(num_classes=1, kpt_shape=(1, 2), fusion_levels=None,
                 pretrained_path=None, device='cuda',
                 fusion_version=2, align_mode=None, use_sharpener=False,
                 align_hidden_ch=None, fusion_gate_type='channel',
                 global_align_stage='none', fft_downsample=4,
                 base_model='p2l_p4'):
    """工厂函数：创建 MultiFrameYOLO 模型并移到指定设备

    fusion_version: 2 (DeformAlignV2) 或 3 (HybridAlign)
    align_mode: V3 时生效，如 {'p3': 'hybrid', 'p4': 'dav2'}
    use_sharpener: 是否在 DeformAlignV2 输入前加特征锐化层
    align_hidden_ch: DeformAlignV2 offset predictor 隐藏通道数（默认 min(96, in_ch*2)）
    fusion_gate_type: 'channel'（per-channel gate）或 'spatial'（per-pixel gate）
    base_model: 'p2l'（P2/P3/P4 backbone + P3/P4 头）、
                'p2l_p4'（同 backbone + P4 单头）或 'p34l'（P3/P4 backbone + P3/P4 头）
    """
    model = MultiFrameYOLO(
        num_classes=num_classes,
        kpt_shape=kpt_shape,
        fusion_levels=fusion_levels,
        pretrained_path=pretrained_path,
        fusion_version=fusion_version,
        align_mode=align_mode,
        use_sharpener=use_sharpener,
        align_hidden_ch=align_hidden_ch,
        fusion_gate_type=fusion_gate_type,
        global_align_stage=global_align_stage,
        fft_downsample=fft_downsample,
        base_model=base_model,
    )
    model = model.to(device)
    return model


def create_model_v3(num_classes=1, kpt_shape=(1, 2), fusion_levels=None,
                    pretrained_path=None, device='cuda',
                    align_mode=None):
    """便捷函数：创建使用 HybridAlign (V3) 的模型"""
    return create_model(
        num_classes=num_classes,
        kpt_shape=kpt_shape,
        fusion_levels=fusion_levels,
        pretrained_path=pretrained_path,
        device=device,
        fusion_version=3,
        align_mode=align_mode,
    )


# ═══════════════════════════════════════════════════════════
# 流式推理模式：历史帧特征缓存复用
# ═══════════════════════════════════════════════════════════

class StreamingInference:
    """
    流式推理缓存（环形缓冲区）。

    重要：输入图像必须是 [0, 1] 范围的 float32 tensor（已除以 255.0），
    与训练时的归一化一致。如果图像是 [0, 255] uint8 或 float，请先做 img.float() / 255.0。

    用法:
        stream = StreamingInference(model)
        for new_frame in video:
            output = stream.infer(new_frame)
            stream.update_cache(new_frame)

    每次推理只过一次 backbone（新帧），历史帧特征从缓存取。
    缓存维持最近 3 帧的 backbone 特征，infer 时自动取 key/ref1/ref2。
    """
    def __init__(self, model, window_size=3):
        self.model = model
        self.window_size = window_size
        self.buffer = []  # list of (p2, p3, p4) backbone 特征，新帧追加在末尾

    def reset(self):
        self.buffer = []

    @torch.no_grad()
    def infer(self, key_img):
        """
        推理当前帧。

        key_img: (1, 3, H, W) 或 (3, H, W)，值域 [0, 1] float32
        返回: Head 输出
        """
        if key_img.dim() == 3:
            key_img = key_img.unsqueeze(0)

        if self.model.global_align_stage == 'feature':
            raise RuntimeError(
                'StreamingInference 尚未缓存 FFT 所需的历史 RGB/梯度图；'
                '请使用普通三帧推理，避免静默跳过 feature FFT 对齐。')

        key_feats = self.model.backbone(key_img)
        p2 = None
        if self.model.pad_head and self.model.pad_head_cfg['use_p2']:
            p2 = key_feats[self.model.level_to_idx['p2']]

        # 从缓存取历史特征：ref1=前1帧, ref2=前2帧
        if len(self.buffer) >= 2:
            ref1_feats = self.buffer[-1]
            ref2_feats = self.buffer[-2]
        elif len(self.buffer) == 1:
            ref1_feats = self.buffer[-1]
            ref2_feats = key_feats  # 不足时 fallback 到 key
        else:
            ref1_feats = key_feats
            ref2_feats = key_feats

        fused = self.model.fusion(key_feats, ref1_feats, ref2_feats)
        return self.model.detect_forward(fused, p2)

    def update_cache(self, img):
        """
        将新帧的 backbone 特征推入缓存。

        img: (1, 3, H, W) 或 (3, H, W)，值域 [0, 1] float32
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            feats = self.model.backbone(img)
        self.buffer.append(feats)
        # 只保留最近 window_size 帧
        while len(self.buffer) > self.window_size:
            self.buffer.pop(0)


def enable_consistent_align(model):
    """把 feature 模式模型的位移估计切换为一致性管线 (不改变任何权重)。

    用于 eval/推理脚本: 从 checkpoint 读到 consistent_align 标志后调用,
    等价于构造时传 consistent_align=True。

    Args:
        model: MultiFrameYOLO 实例 (global_align_stage='feature')
    Returns:
        model (原地修改并返回)
    """
    from fft_align import ConsistentAligner
    if model.global_align_stage != 'feature':
        raise ValueError('enable_consistent_align 仅适用于 global_align_stage="feature" 的模型')
    model.consistent_align = True
    model.consistent_aligner = ConsistentAligner(
        downsample=model.fft_downsample, k=5)
    # 旧 feature-FFT 对齐器不再需要；置 None 使模型与构造时 consistent_align=True 的结构一致
    model.fft_align_r1 = None
    model.fft_align_r2 = None
    return model
