#!/usr/bin/env python3
"""PAD (Point-Anchored Detector) 组件 — 从 few_shot_weed_stem 移植到 exp3 基座.

设计 (与 few_shot_weed_stem/fs_work/scripts/pad_lib.py 一致):
  - 复用 exp3 YOLOv8-pose 的 DarkNet/DarkFPN 权重;
  - FPN 输出 (n3 stride8, n4 stride16) 融合后预测茎基点热图 + 亚像素 offset + 框大小;
  - 每个 GT 只有一个正样本 (最近的特征格), 不使用 IoU^beta 的 TAL 分配;
  - 推理用 3x3 max-pool 峰 + top-k, 点即检测锚点, 框由点 + size 派生.

与 few_shot 版的差异:
  - 对齐 exp3 的 lite 基座 (p34l: P3/P4 无 P5; p2l_p4: P2/P3/P4), head 支持 2~3 个 FPN 层;
  - 单帧模型属性名保持 net/fpn, 便于 MultiFrameYOLO.load_pretrained 的 net.→backbone. 映射;
  - 评估接口改为接受任意 DataLoader (单帧 Dataset 或 MultiFrameDataset).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from nets.nn import (Conv, DarkNet, DarkFPN, DarkNetP2, DarkFPN_P2,
                     DarkNetP2_lite, DarkFPN_P2_lite, DarkNet_lite, DarkFPN_lite)
from utils import util


def _conv(cin, cout, k=3, dw=False):
    """3x3 Conv, optionally depthwise-separable (dw 3x3 -> 1x1 pointwise)."""
    if dw and k > 1:
        return torch.nn.Sequential(Conv(cin, cin, k, g=cin), Conv(cin, cout, 1))
    return Conv(cin, cout, k)


class PointHead(torch.nn.Module):
    """P3 -> heatmap/offset/size. out_stride4=False keeps the stride-8 grid."""

    def __init__(self, ch=64, mid=64, out_stride4=True, dw=False, use_size=True):
        super().__init__()
        self.out_stride4 = out_stride4
        self.use_size = use_size
        self.up = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(ch, ch, 3),
        ) if out_stride4 else torch.nn.Identity()
        self.hm = torch.nn.Sequential(
            _conv(ch, mid, 3, dw), _conv(mid, mid, 3, dw),
            torch.nn.Conv2d(mid, 1, 1))
        self.off = torch.nn.Sequential(
            _conv(ch, mid, 3, dw), _conv(mid, mid, 3, dw),
            torch.nn.Conv2d(mid, 2, 1))
        self.size = torch.nn.Sequential(
            _conv(ch, mid, 3, dw), _conv(mid, mid, 3, dw),
            torch.nn.Conv2d(mid, 2, 1)) if use_size else None
        # CenterNet 式初始化: 初始热图概率 ~0.01
        self.hm[-1].bias.data.fill_(-4.595)
        self.off[-1].bias.data.zero_()
        if self.size is not None:
            self.size[-1].bias.data.zero_()

    def forward(self, p3):
        x = self.up(p3)
        if self.size is None:
            return self.hm(x), self.off(x)
        return self.hm(x), self.off(x), self.size(x)


class PointHeadMS(torch.nn.Module):
    """多尺度点热图头: FPN 各层 1x1 降维后自顶向下融合.

    levels: 由细到粗的 FPN 特征 (如 p34l 的 (n3 stride8, n4 stride16)).
    out_stride4: True 时上采样到 stride4, False 保持最细层原 stride.
    use_p2: 把 backbone stride4 的 P2 细节经 1x1 + stride2 conv 接进最细层融合.
    """

    def __init__(self, chs=(64, 128), mid=32, out_stride4=False, dw=False,
                 use_p2=False, p2_ch=32, p2_mid=32, use_size=True):
        super().__init__()
        self.n_levels = len(chs)
        self.out_stride4 = out_stride4
        self.use_p2 = use_p2
        self.use_size = use_size
        self.lat = torch.nn.ModuleList(
            torch.nn.Conv2d(c, mid, 1) for c in chs)
        if use_p2:
            self.lat2 = torch.nn.Conv2d(p2_ch, p2_mid, 1)
            self.down2 = Conv(p2_mid, mid, 3, 2)
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.smooth = _conv(mid, mid, 3, dw)
        self.up4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            _conv(mid, mid, 3, dw)) if out_stride4 else torch.nn.Identity()
        self.hm = torch.nn.Sequential(
            _conv(mid, mid, 3, dw), _conv(mid, mid, 3, dw),
            torch.nn.Conv2d(mid, 1, 1))
        self.off = torch.nn.Sequential(
            _conv(mid, mid, 3, dw), _conv(mid, mid, 3, dw),
            torch.nn.Conv2d(mid, 2, 1))
        self.size = torch.nn.Sequential(
            _conv(mid, mid, 3, dw), _conv(mid, mid, 3, dw),
            torch.nn.Conv2d(mid, 2, 1)) if use_size else None
        self.hm[-1].bias.data.fill_(-4.595)
        self.off[-1].bias.data.zero_()
        if self.size is not None:
            self.size[-1].bias.data.zero_()

    def forward(self, feats, p2=None):
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        f = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            f = self.lat[i](feats[i]) + self.up(f)
        if self.use_p2:
            assert p2 is not None, 'use_p2 head needs the backbone P2 feature'
            f = f + self.down2(self.lat2(p2))
        x = self.up4(self.smooth(f))
        if self.size is None:
            return self.hm(x), self.off(x)
        return self.hm(x), self.off(x), self.size(x)


def normalize_head_cfg(head_cfg=None):
    cfg = {
        'head_type': 'ms',
        'mid': 32,
        'out_stride4': False,
        'dw': False,
        'use_p2': False,
        'p2_mid': 32,
        'use_size': True,
    }
    if head_cfg:
        unknown = set(head_cfg) - set(cfg)
        if unknown:
            raise ValueError(f'unknown head_cfg keys: {sorted(unknown)}')
        cfg.update(head_cfg)
    return cfg


def build_point_head(chs, head_cfg=None):
    cfg = normalize_head_cfg(head_cfg)
    head_type = cfg.pop('head_type')
    if head_type == 'ms':
        return PointHeadMS(chs=chs, **cfg)
    use_p2 = cfg.pop('use_p2')
    cfg.pop('p2_mid')
    if use_p2:
        raise ValueError('use_p2 only supported with head_type=ms')
    return PointHead(ch=chs[0], **cfg)


# ─────────────────────────────────────────────────────────────
# Single-frame model on exp3 lite backbones
# ─────────────────────────────────────────────────────────────
# 单帧基座注册表: name -> (backbone 类, FPN 类, 头输入层, P2 通道或 None).
# 头输入层沿用旧单帧消融 (single_frame_baseline/train.py MODEL_REGISTRY) 的头层级,
# 只换头不改各基座的检测层级 (p2l_p4 为兼容名, 与 p2l 同构, 故实际 7 种配置):
#   v8n    = yolo_v8_n            (SF-ORIG,  完整 P3/P4/P5 backbone+FPN, 头 P3/P4/P5)
#   p3     = yolo_v8_n_p3         (SF-P3,    同上, 头仅 P3)
#   p34    = yolo_v8_n_p34        (SF-P34,   同上, 头 P3/P4, 无 P5 头)
#   p34l   = yolo_v8_n_p34_lite   (SF-P34L,  无 P5 backbone, 头 P3/P4)
#   p2     = yolo_v8_n_p2         (SF-P2,    P2/P3/P4/P5, P2-FPN, 头 P3/P4)
#   p2l    = yolo_v8_n_p2_lite    (SF-P2L,   P2/P3/P4 无 P5, 头 P3/P4)
#   p2l_p4 = yolo_v8_n_p2_lite_p4 (SF-P2L-P4,同上, 头仅 P4)
# 所有 n 规格基座共用 WIDTH=[3,16,32,64,128,256] / DEPTH=[1,2,2].
PAD_SF_BASES = {
    'v8n':    (DarkNet,        DarkFPN,        ('p3', 'p4', 'p5'), None),
    'p3':     (DarkNet,        DarkFPN,        ('p3',),            None),
    'p34':    (DarkNet,        DarkFPN,        ('p3', 'p4'),       None),
    'p34l':   (DarkNet_lite,   DarkFPN_lite,   ('p3', 'p4'),       None),
    'p2':     (DarkNetP2,      DarkFPN_P2,     ('p3', 'p4'),       32),
    'p2l':    (DarkNetP2_lite, DarkFPN_P2_lite,('p3', 'p4'),       32),
    # 兼容名: 已有 pad_sf_p2l_p4_p2* checkpoint 的 (p3,p4) 头, 结构与 p2l 相同
    'p2l_p4': (DarkNetP2_lite, DarkFPN_P2_lite,('p3', 'p4'),       32),
    # 旧消融 SF-P2L-P4 的本义: 单尺度 P4 头 (上采样回 stride8, 与其他基座解码网格一致)
    'p2l_p4_only': (DarkNetP2_lite, DarkFPN_P2_lite, ('p4',),      32),
}


class PADSingleFrame(torch.nn.Module):
    """PAD 单帧模型: exp3 旧单帧消融的 7 种基座 + PAD 点热图头 (8 个注册名).

    属性名保持 net/fpn/head, 与 YOLOv8-pose checkpoint 的键名一致, 便于
    MultiFrameYOLO.load_pretrained 做 net.→backbone. 映射.
    """

    WIDTH = [3, 16, 32, 64, 128, 256]
    DEPTH = [1, 2, 2]

    def __init__(self, base_model='p34l', head_cfg=None):
        super().__init__()
        if base_model not in PAD_SF_BASES:
            raise ValueError(f'base_model 仅支持 {sorted(PAD_SF_BASES)}, 收到 {base_model!r}')
        self.base_model = base_model
        net_cls, fpn_cls, head_levels, p2_ch = PAD_SF_BASES[base_model]
        self.net = net_cls(self.WIDTH, self.DEPTH)
        self.fpn = fpn_cls(self.WIDTH, self.DEPTH)
        self.head_levels = head_levels
        self.p2_ch = p2_ch
        self.head_cfg = normalize_head_cfg(head_cfg)
        if self.head_cfg['use_p2'] and self.p2_ch is None:
            raise ValueError(f'{base_model} 基座没有 P2 特征, 不能 use_p2')
        ch_by_level = {'p2': self.WIDTH[2], 'p3': self.WIDTH[3],
                       'p4': self.WIDTH[4], 'p5': self.WIDTH[5]}
        fpn_chs = tuple(ch_by_level[lv] for lv in head_levels)
        # FPN 输出恒为 (p3, p4[, p5]) 由细到粗; 记录每个头层在输出中的下标
        self.fpn_idx = tuple({'p3': 0, 'p4': 1, 'p5': 2}[lv] for lv in head_levels)
        self.head = build_point_head(fpn_chs, self.head_cfg)
        self.stride = 4.0 if self.head_cfg['out_stride4'] else 8.0
        # 最细头层比目标 stride 更粗时, 头前用最近邻上采样补齐, 保证所有基座
        # 输出可共用同一解码/评估逻辑. 注意头内部 out_stride4 还会再做一次
        # 2x 上采样, pre_up 只需把最细层补到头的输入 stride, 不能多补
        fine_stride = {'p3': 8.0, 'p4': 16.0, 'p5': 32.0}[head_levels[0]]
        head_in_stride = self.stride * (2.0 if self.head_cfg['out_stride4'] else 1.0)
        n_up = max(0, int(round(math.log2(fine_stride / head_in_stride))))
        self.pre_up = torch.nn.Sequential(
            *[torch.nn.Upsample(scale_factor=2, mode='nearest')
              for _ in range(n_up)]) if n_up > 0 else None

    def forward(self, x):
        p2 = None
        if self.p2_ch is not None:
            outs = self.net(x)
            p2 = outs[0]
            fpn_outs = self.fpn(outs)
        else:
            fpn_outs = self.fpn(self.net(x))
        # 旧 checkpoint (基座扩展前) 实例上没有 fpn_idx/pre_up, 回退旧行为:
        # ms 头恒接 (p3,p4[,p5] 前缀), simple 头接 feats[0]
        fpn_idx = getattr(self, 'fpn_idx', None)
        if fpn_idx is None:
            n_levels = len(self.head.lat) if hasattr(self.head, 'lat') else 1
            fpn_idx = tuple(range(n_levels))
        feats = [fpn_outs[i] for i in fpn_idx]
        pre_up = getattr(self, 'pre_up', None)
        if pre_up is not None:
            feats = [pre_up(f) for f in feats]
        if self.head_cfg['head_type'] == 'ms':
            return self.head(feats, p2)
        return self.head(feats[0])


# ─────────────────────────────────────────────────────────────
# Target building
# ─────────────────────────────────────────────────────────────
def gaussian_radius(det_size, min_overlap=0.7):
    """CenterNet gaussian radius for (h, w) in feature cells."""
    h, w = det_size
    a1, b1, c1 = 1.0, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0))) / (2 * a1)
    a2, b2, c2 = 4.0, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 - math.sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0))) / (2 * a2)
    a3, b3, c3 = 4.0 * (1 - min_overlap), 2 * (1 - min_overlap) * (h + w), \
        (1 - min_overlap) ** 2 * w * h
    r3 = (b3 - math.sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0))) / (2 * a3)
    return max(0, int(min(r1, r2, r3)))


def _kpt_xy(kpt, n):
    """targets['kpt'] 可能是 (N,1,2) 或 (N,1,3)."""
    return float(kpt[n, 0, 0]), float(kpt[n, 0, 1])


def build_targets(targets, batch, hf, wf, device, min_overlap=0.7, sigma_div=3.0,
                  off_window=True, fixed_radius=None, skip_invisible=False):
    """targets from exp3 Dataset/MultiFrameDataset:
    box (N,4) cxcywh normalized, kpt (N,1,2|3) normalized.

    Returns heatmap (B,1,hf,wf), offset (B,2,hf,wf), size (B,2,hf,wf), pos (B,hf,wf).
    Point = GT keypoint; size = GT box w/h.
    """
    hm = torch.zeros((batch, 1, hf, wf), device=device)
    off = torch.zeros((batch, 2, hf, wf), device=device)
    size = torch.zeros((batch, 2, hf, wf), device=device)
    pos = torch.zeros((batch, hf, wf), device=device, dtype=torch.bool)
    off_mask = torch.zeros((batch, hf, wf), device=device, dtype=torch.bool)

    box = targets['box'].to(device).float()
    kpt = targets['kpt'].to(device).float()
    idx = targets['idx'].to(device).long()
    for n in range(box.shape[0]):
        b = int(idx[n].item())
        if b < 0 or b >= batch:
            continue
        if skip_invisible and kpt.shape[-1] >= 3 and float(kpt[n, 0, 2]) <= 0:
            # 增广后越界的关键点 vis=0; 与标准 YOLO 的 kpt_mask 一致, 不监督
            continue
        kx, ky = _kpt_xy(kpt, n)
        # keypoint must be inside the image
        if not (0.0 <= kx < 1.0 and 0.0 <= ky < 1.0):
            continue
        bw, bh = float(box[n, 2]), float(box[n, 3])
        gx, gy = kx * wf, ky * hf
        ix, iy = int(gx), int(gy)
        if not (0 <= ix < wf and 0 <= iy < hf):
            continue
        if fixed_radius:
            # 框尺寸不可信时使用固定半径 (单位: feature cells), 不因坏框丢点
            w_cells = h_cells = 1.0
            radius = max(1, int(fixed_radius))
        else:
            if bw <= 1e-3 or bh <= 1e-3:
                continue
            w_cells = max(bw * wf, 0.5)
            h_cells = max(bh * hf, 0.5)
            radius = max(1, gaussian_radius((h_cells, w_cells), min_overlap))
        sigma = max(radius / sigma_div, 1e-3)
        for dy in range(-radius, radius + 1):
            yy = iy + dy
            if yy < 0 or yy >= hf:
                continue
            for dx in range(-radius, radius + 1):
                xx = ix + dx
                if xx < 0 or xx >= wf:
                    continue
                g = math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
                if g > float(hm[b, 0, yy, xx]):
                    hm[b, 0, yy, xx] = g
                    # The max-Gaussian owner must also own offset/size. A later
                    # neighbouring GT must not overwrite a stronger peak's
                    # regression target while leaving its heatmap unchanged.
                    if off_window:
                        off_mask[b, yy, xx] = True
                        off[b, 0, yy, xx] = gx - (xx + 0.5)
                        off[b, 1, yy, xx] = gy - (yy + 0.5)
                        size[b, 0, yy, xx] = math.log(max(w_cells, 0.5))
                        size[b, 1, yy, xx] = math.log(max(h_cells, 0.5))
        pos[b, iy, ix] = True
        off_mask[b, iy, ix] = True
        off[b, 0, iy, ix] = gx - (ix + 0.5)
        off[b, 1, iy, ix] = gy - (iy + 0.5)
        size[b, 0, iy, ix] = math.log(max(w_cells, 0.5))
        size[b, 1, iy, ix] = math.log(max(h_cells, 0.5))
    return hm, off, size, pos, off_mask


def point_loss(pred, targets, params=None, weights=(1.0, 1.0, 0.5),
               min_overlap=0.7, sigma_div=3.0, size_iou_w=0.0,
               off_window=True, off_mag_w=0.0, fixed_radius=None,
               skip_invisible=False):
    """CenterNet focal heatmap + L1 offset/size.

    ``pred`` may be (hm, off) for a point-only head (use_size=False);
    size loss is then skipped.
    """
    use_size = len(pred) >= 3
    hm_logit, off = pred[0], pred[1]
    size = pred[2] if use_size else None
    B, _, hf, wf = hm_logit.shape
    hm_t, off_t, size_t, pos, off_mask = build_targets(
        targets, B, hf, wf, hm_logit.device,
        min_overlap=min_overlap, sigma_div=sigma_div, off_window=off_window,
        fixed_radius=fixed_radius, skip_invisible=skip_invisible)
    p = torch.sigmoid(hm_logit)
    pos_f = pos.float()
    neg_f = 1.0 - pos_f
    # (1-p)^2 log(p) for positives; (1-y)^4 p^2 log(1-p) for negatives
    # All factors are B,H,W. B,1,H,W * B,H,W would broadcast to B,B,H,W,
    # coupling one image's focal weight to every other image in the batch.
    pos_loss = -((1.0 - p[:, 0]) ** 2) * F.logsigmoid(hm_logit[:, 0]) * pos_f
    neg_loss = -((1.0 - hm_t[:, 0]) ** 4) * (p[:, 0] ** 2) * \
        F.logsigmoid(-hm_logit[:, 0]) * neg_f
    n_pos = pos_f.sum().clamp(min=1.0)
    loss_hm = (pos_loss.sum() + neg_loss.sum()) / n_pos

    if off_mask.any():
        n_off = off_mask.sum().clamp(min=1.0)
        off_p = off.permute(0, 2, 3, 1)[off_mask]
        off_gt = off_t.permute(0, 2, 3, 1)[off_mask]
        if off_mag_w > 0:
            wgt = 1.0 + off_mag_w * torch.linalg.vector_norm(off_gt, dim=1)
            loss_off = ((off_p - off_gt).abs().sum(1) * wgt).sum() / wgt.sum()
        else:
            loss_off = F.l1_loss(off_p, off_gt, reduction='sum') / n_off
        if use_size and fixed_radius is None:
            size_p = size.permute(0, 2, 3, 1)[off_mask]
            loss_size = F.l1_loss(size_p, size_t.permute(0, 2, 3, 1)[off_mask],
                                  reduction='sum') / n_off
            if size_iou_w > 0:
                gt_s = size_t.permute(0, 2, 3, 1)[off_mask]
                pw, ph = torch.exp(size_p[:, 0]), torch.exp(size_p[:, 1])
                gw, gh = torch.exp(gt_s[:, 0]), torch.exp(gt_s[:, 1])
                inter = torch.min(pw, gw) * torch.min(ph, gh)
                union = pw * ph + gw * gh - inter
                loss_size = loss_size + size_iou_w * (1.0 - inter / (union + 1e-7)).mean()
        else:
            loss_size = torch.zeros((), device=hm_logit.device)
    else:
        loss_off = torch.zeros((), device=hm_logit.device)
        loss_size = torch.zeros((), device=hm_logit.device)

    loss = weights[0] * loss_hm + weights[1] * loss_off + weights[2] * loss_size
    return loss, (float(loss_hm), float(loss_off), float(loss_size)), int(pos.sum())


# ─────────────────────────────────────────────────────────────
# Inference / evaluation
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def decode_predictions(pred, input_size=1024, conf=0.001, topk=300, blend=0.0):
    """pred: (hm_logit (B,1,H,W), off (B,2,H,W), size (B,2,H,W)).

    Returns list of np arrays [x1,y1,x2,y2,score,kx,ky].
    """
    use_size = len(pred) >= 3
    hm_logit, off = pred[0], pred[1]
    size = pred[2] if use_size else None
    B, _, hf, wf = hm_logit.shape
    stride_x, stride_y = input_size / wf, input_size / hf
    hm = torch.sigmoid(hm_logit)
    # 3x3 max-pool peak NMS
    hmax = F.max_pool2d(hm, 3, stride=1, padding=1)
    keep = (hm >= hmax - 1e-6).float()
    scores = (hm * keep).view(B, -1)
    k = min(topk, scores.shape[1])
    top_scores, top_idx = torch.topk(scores, k, dim=1)
    top_scores = top_scores.cpu().numpy()
    top_idx = top_idx.cpu().numpy()
    off_np = off.cpu().numpy()
    size_np = size.cpu().numpy() if use_size else None
    hm_np = hm.cpu().numpy()
    from numpy.lib.stride_tricks import sliding_window_view
    gx = np.array([-1.0, 0.0, 1.0], np.float32)[None, None, :]
    gy = np.array([-1.0, 0.0, 1.0], np.float32)[None, :, None]

    out = []
    for b in range(B):
        ys = (top_idx[b] // wf).astype(np.float32)
        xs = (top_idx[b] % wf).astype(np.float32)
        sc = top_scores[b]
        dx = off_np[b, 0].reshape(-1)[top_idx[b]]
        dy = off_np[b, 1].reshape(-1)[top_idx[b]]
        if blend:
            hp = np.pad(hm_np[b, 0], 1, mode='edge')
            win = sliding_window_view(hp, (3, 3))[ys.astype(int), xs.astype(int)]
            swin = win.sum((1, 2)) + 1e-9
            dx = dx + blend * (win * gx).sum((1, 2)) / swin
            dy = dy + blend * (win * gy).sum((1, 2)) / swin
        px = (xs + 0.5 + dx) * stride_x
        py = (ys + 0.5 + dy) * stride_y
        if use_size:
            sw = size_np[b, 0].reshape(-1)[top_idx[b]]
            sh = size_np[b, 1].reshape(-1)[top_idx[b]]
            bw = np.exp(np.clip(sw, -4.0, 6.0)) * stride_x
            bh = np.exp(np.clip(sh, -4.0, 6.0)) * stride_y
        else:
            # 点-only: 不输出框; 用 2px 占位框只是为了维持统一的 7 列格式
            bw = np.full_like(px, 2.0)
            bh = np.full_like(py, 2.0)
        x1, y1 = px - bw / 2.0, py - bh / 2.0
        x2, y2 = px + bw / 2.0, py + bh / 2.0
        m = (sc >= conf) & (px >= 0) & (px < input_size) & (py >= 0) & (py < input_size) \
            & (bw > 1.0) & (bh > 1.0)
        arr = np.stack([x1[m], y1[m], x2[m], y2[m], sc[m], px[m], py[m]], 1).astype(np.float32)
        out.append(arr)
    return out


def _gt_arrays(targets, i, w, h):
    idx = targets['idx'] == i
    cls = targets['cls'][idx]
    box = targets['box'][idx]
    kpt = targets['kpt'][idx]
    if not cls.shape[0]:
        return np.zeros((0, 6), np.float32)
    b = box.cpu().numpy().copy()
    k = kpt.cpu().numpy().copy()
    # normalized cxcywh -> pixel xyxy
    cx, cy, bw, bh = b[:, 0] * w, b[:, 1] * h, b[:, 2] * w, b[:, 3] * h
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    kx, ky = k[:, 0, 0] * w, k[:, 0, 1] * h
    return np.concatenate([xyxy, np.stack([kx, ky], 1)], 1).astype(np.float32)


@torch.no_grad()
def pad_infer(model, loader, input_size=1024, conf=0.001, topk=300, blend=0.0):
    model.eval()
    preds = {}
    k = 0
    for samples, targets in loader:
        samples = samples.to(next(model.parameters()).device).float() / 255
        h, w = samples.shape[-2], samples.shape[-1]
        out = model(samples)
        batch_pred = decode_predictions(out, input_size, conf=conf, topk=topk,
                                        blend=blend)
        for i, arr in enumerate(batch_pred):
            idx = k + i
            preds[idx] = {'pred': arr, 'gt': _gt_arrays(targets, i, w, h)}
        k += len(batch_pred)
    return preds


def box_iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, None, 0], a[:, None, 1], a[:, None, 2], a[:, None, 3]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], b[None, :, 2], b[None, :, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-7)


def match_and_score(preds, iou_th=0.5, conf=0.25):
    n_gt = n_pr = n_tp = 0
    errs, errs_px = [], []
    for d in preds.values():
        pr, gt = d['pred'], d['gt']
        if len(pr):
            pr = pr[pr[:, 4] >= conf]
        n_gt += len(gt)
        n_pr += len(pr)
        if not len(pr) or not len(gt):
            continue
        iou = box_iou_matrix(pr[:, :4], gt[:, :4])
        order = np.argsort(-pr[:, 4])
        taken = set()
        for pi in order:
            gi = int(np.argmax(iou[pi]))
            if iou[pi, gi] >= iou_th and gi not in taken:
                taken.add(gi)
                n_tp += 1
                gx1, gy1, gx2, gy2 = gt[gi, :4]
                diag = float(np.hypot(gx2 - gx1, gy2 - gy1)) + 1e-7
                err = float(np.hypot(pr[pi, 5] - gt[gi, 4], pr[pi, 6] - gt[gi, 5]))
                errs.append(err / diag)
                errs_px.append(err)
    errs = np.array(errs)
    errs_px = np.array(errs_px)
    return {
        'n_gt': n_gt, 'n_pred': n_pr, 'n_tp': n_tp,
        'recall@0.5': n_tp / max(n_gt, 1),
        'precision@0.5': n_tp / max(n_pr, 1),
        'nPE_mean': float(errs.mean()) if len(errs) else float('nan'),
        'nPE_median': float(np.median(errs)) if len(errs) else float('nan'),
        'PCK@0.25': float((errs < 0.25).mean()) if len(errs) else float('nan'),
        'PCK@0.5': float((errs < 0.5).mean()) if len(errs) else float('nan'),
        'kpt_dist_mean_px': float(errs_px.mean()) if len(errs_px) else float('nan'),
        'kpt_dist_median_px': float(np.median(errs_px)) if len(errs_px) else float('nan'),
        'kpt_dist_p90_px': float(np.percentile(errs_px, 90)) if len(errs_px) else float('nan'),
        **{f'PCK@{t:g}px': float((errs_px <= t).mean()) if len(errs_px) else float('nan')
           for t in (2.0, 5.0, 10.0)},
    }


def match_points(preds, tau_px=10.0, conf=0.25):
    """无框点匹配: 预测点与最近未匹配 GT 点的距离 <= tau_px 记为 TP.

    与 match_and_score 不同, 完全不使用检测框; 用于 point-only 头或在
    框标注不可信的数据集上评估. tau_px 是 1024 输入空间的绝对像素.
    """
    n_gt = n_pr = n_tp = 0
    errs = []
    for d in preds.values():
        pr, gt = d['pred'], d['gt']
        if len(pr):
            pr = pr[pr[:, 4] >= conf]
        n_gt += len(gt)
        n_pr += len(pr)
        if not len(pr) or not len(gt):
            continue
        gx, gy = gt[:, 4], gt[:, 5]
        order = np.argsort(-pr[:, 4])
        taken = set()
        for pi in order:
            if len(taken) == len(gt):
                break
            dist = np.hypot(gx - pr[pi, 5], gy - pr[pi, 6])
            if taken:
                dist[list(taken)] = 1e9
            gi = int(np.argmin(dist))
            if dist[gi] <= tau_px:
                taken.add(gi)
                n_tp += 1
                errs.append(float(dist[gi]))
    errs = np.array(errs)
    return {
        'n_gt': n_gt, 'n_pred': n_pr, 'n_tp': n_tp, 'tau_px': tau_px,
        'point_recall': n_tp / max(n_gt, 1),
        'point_precision': n_tp / max(n_pr, 1),
        'nPE_mean_px': float(errs.mean()) if len(errs) else float('nan'),
        'nPE_median_px': float(np.median(errs)) if len(errs) else float('nan'),
        **{f'PCK@{t:g}px': float((errs <= t).mean()) if len(errs) else float('nan')
           for t in (2.0, 5.0, 10.0)},
    }


def point_ap(preds, tau_px=10.0, conf_floor=0.001):
    """点级 AP: 按 conf 排序的预测点, 距离 tau_px 内匹配为 TP, VOC 积分."""
    scores, tps = [], []
    n_gt = 0
    for d in preds.values():
        pr, gt = d['pred'], d['gt']
        n_gt += len(gt)
        if not len(pr):
            continue
        pr = pr[pr[:, 4] >= conf_floor]
        if not len(pr):
            continue
        order = np.argsort(-pr[:, 4])
        taken = set()
        tp = np.zeros(len(pr), bool)
        if len(gt):
            gx, gy = gt[:, 4], gt[:, 5]
            for pi in order:
                if len(taken) == len(gt):
                    break
                dist = np.hypot(gx - pr[pi, 5], gy - pr[pi, 6])
                if taken:
                    dist[list(taken)] = 1e9
                gi = int(np.argmin(dist))
                if dist[gi] <= tau_px:
                    tp[pi] = True
                    taken.add(gi)
        scores.append(pr[:, 4])
        tps.append(tp)
    if not scores:
        return float('nan')
    scores = np.concatenate(scores)
    tps = np.concatenate(tps)
    order = np.argsort(-scores)
    tps = tps[order]
    cum_tp = np.cumsum(tps)
    cum_fp = np.arange(1, len(tps) + 1) - cum_tp
    recall = cum_tp / max(n_gt, 1)
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    return float(np.sum(np.diff(mrec) * mpre[1:]))


def evaluate_points(model, loader, input_size=1024, conf=0.25, topk=300,
                    tau_px=10.0, blend=0.0):
    """point-only 评估: 点距离匹配 + 点级 AP (不依赖任何框)."""
    preds = pad_infer(model, loader, input_size, conf=0.001, topk=topk, blend=blend)
    out = match_points(preds, tau_px=tau_px, conf=conf)
    out['point_ap_10px'] = point_ap(preds, tau_px=10.0)
    out['point_ap_20px'] = point_ap(preds, tau_px=20.0)
    out['conf'] = conf
    return out, preds


def _ap_from_preds(preds, iou_v):
    box_metrics, kpt_metrics = [], []
    for d in preds.values():
        pr, gt = d['pred'], d['gt']
        n, m = len(pr), len(gt)
        if n == 0:
            if m:
                box_metrics.append((np.zeros((0, len(iou_v)), bool), np.zeros(0),
                                    np.zeros(0), np.zeros(m, np.float32)))
                kpt_metrics.append((np.zeros((0, len(iou_v)), bool), np.zeros(0),
                                    np.zeros(0), np.zeros(m, np.float32)))
            continue
        output = torch.from_numpy(np.concatenate(
            [pr[:, :4], pr[:, 4:5], np.zeros((n, 1), np.float32)], 1)).to(iou_v.device)
        target = torch.from_numpy(np.concatenate(
            [np.zeros((m, 1), np.float32), gt[:, :4]], 1)).to(iou_v.device) if m else torch.zeros((0, 5)).to(iou_v.device)
        correct_box = util.compute_metric(output, target, iou_v)
        p_kpt = torch.from_numpy(pr[:, 5:7]).to(iou_v.device).view(n, 1, 2)
        t_kpt = torch.from_numpy(np.concatenate(
            [gt[:, 4:6], np.ones((m, 1), np.float32)], 1)).to(iou_v.device).view(m, 1, 3) \
            if m else torch.zeros((0, 1, 3)).to(iou_v.device)
        correct_kpt = util.compute_metric(output, target, iou_v, p_kpt, t_kpt)
        box_metrics.append((correct_box.cpu().numpy(), pr[:, 4], np.zeros(n), np.zeros(m)))
        kpt_metrics.append((correct_kpt.cpu().numpy(), pr[:, 4], np.zeros(n), np.zeros(m)))

    res = {'box_ap50': 0.0, 'box_ap': 0.0, 'pose_ap50': 0.0, 'pose_ap': 0.0}
    for key, metrics in (('box', box_metrics), ('pose', kpt_metrics)):
        if not metrics:
            continue
        cat = [np.concatenate(x, 0) for x in zip(*metrics)]
        if cat[0].any():
            _, _, _, _, ap50, ap = util.compute_ap(*cat)
            res[f'{key}_ap50'], res[f'{key}_ap'] = float(ap50), float(ap)
    return res


def evaluate_pad(model, loader, input_size=1024, conf=0.25, topk=300, blend=0.0):
    if getattr(getattr(model, 'head', None), 'size', object()) is None:
        raise ValueError('this is a point-only (use_size=False) model; '
                         'use evaluate_points() instead of evaluate_pad()')
    preds = pad_infer(model, loader, input_size, conf=0.001, topk=topk, blend=blend)
    out = match_and_score(preds, conf=conf)
    out.update(_ap_from_preds(preds, torch.linspace(0.5, 0.95, 10, device=next(model.parameters()).device)))
    return out, preds
