"""Structural variants of the single-stage DCN alignment module (exp4).

All variants are built as *functional supersets* of ``DeformAlignV2Single``:
initialised from a V2 checkpoint they reproduce V2's forward pass exactly, so a
fine-tuning screen isolates the effect of the new structure rather than the
effect of a different random initialisation.

Variants
--------
corr    : local (sub-pixel) correlation volume concatenated to the offset
          predictor input.  Extra input channels are zero-weighted at init.
k5      : 5x5 deformable sampling grid (25 taps) with the V2 3x3 weights
          transplanted into the centre; outer taps start at zero.
coarse  : additive offsets predicted at half resolution (smooth, low-frequency
          residual field), zero-initialised output.
dil2    : additive offsets from a dilation-2 offset-predictor branch,
          zero-initialised output.
p2refine: p2-native residual offsets added on top of the offsets reused from
          p3 (OffsetReuseAlignRefine).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from .modules import DeformConvPackV2, OffsetReuseAlign


def _corr_grid(h, w, device, dtype):
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([gx, gy], dim=-1)          # (H, W, 2) as (x, y)


class DeformAlignV2Adaptive(nn.Module):
    """Single-stage DCN alignment with a structural variant of the sampler."""

    def __init__(self, in_ch, hidden_ch=None, variant='corr',
                 corr_shifts=(-0.75, 0.0, 0.75), use_sharpener=False):
        super().__init__()
        if hidden_ch is None:
            hidden_ch = min(96, in_ch * 2)
        self.variant = variant
        self.use_sharpener = use_sharpener
        self.use_corr = 'corr' in variant
        self.use_k5 = 'k5' in variant
        self.use_coarse = 'coarse' in variant
        self.use_dil = 'dil' in variant

        self.corr_shifts = tuple((float(dy), float(dx))
                                 for dy in corr_shifts for dx in corr_shifts)
        corr_ch = len(self.corr_shifts) if self.use_corr else 0
        self.corr_ch = corr_ch

        self.s1_conv1 = nn.Conv2d(in_ch * 2 + corr_ch, hidden_ch, 3, 1, 1)
        self.s1_conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1)

        k = 5 if self.use_k5 else 3
        self.k = k
        offset_ch = 2 * k * k
        mask_ch = k * k
        self.s1_offset = nn.Conv2d(hidden_ch, offset_ch, 3, 1, 1)
        self.s1_mask = nn.Conv2d(hidden_ch, mask_ch, 3, 1, 1)
        self.s1_dcn = DeformConvPackV2(in_ch, in_ch, k, padding=k // 2)

        self.relu = nn.ReLU(inplace=True)

        if self.use_coarse:
            ch = max(hidden_ch // 2, 16)
            self.coarse_conv1 = nn.Conv2d(in_ch * 2, ch, 3, 1, 1)
            self.coarse_conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
            self.coarse_offset = nn.Conv2d(ch, 18, 3, 1, 1)
            nn.init.zeros_(self.coarse_offset.weight)
            nn.init.zeros_(self.coarse_offset.bias)
        if self.use_dil:
            ch = max(hidden_ch // 2, 16)
            self.dil_conv1 = nn.Conv2d(in_ch * 2, ch, 3, 1, 2, dilation=2)
            self.dil_conv2 = nn.Conv2d(ch, ch, 3, 1, 2, dilation=2)
            self.dil_offset = nn.Conv2d(ch, 18, 3, 1, 2, dilation=2)
            nn.init.zeros_(self.dil_offset.weight)
            nn.init.zeros_(self.dil_offset.bias)

        # Offset predictor starts at zero.  Mask starts at sigmoid(2.2)=0.9.
        nn.init.zeros_(self.s1_offset.weight)
        nn.init.zeros_(self.s1_offset.bias)
        nn.init.constant_(self.s1_mask.weight, 0)
        if self.use_k5:
            # outer taps get mask 0.5 (sigmoid(0)); the centre 3x3 gets 0.9.
            bias = torch.zeros(mask_ch)
            for r in range(3):
                for c in range(3):
                    bias[(r + 1) * 5 + (c + 1)] = 2.2
            nn.init.constant_(self.s1_mask.bias, 0)
            with torch.no_grad():
                self.s1_mask.bias.copy_(bias)
        else:
            nn.init.constant_(self.s1_mask.bias, 2.2)

    # ---- correlation features -------------------------------------------------
    def _correlation(self, nbr, ref):
        n = F.normalize(nbr, dim=1)
        r = F.normalize(ref, dim=1)
        b, c, h, w = nbr.shape
        base = _corr_grid(h, w, nbr.device, nbr.dtype)          # (H,W,2)
        maps = []
        for dy, dx in self.corr_shifts:
            shift = torch.tensor(
                [2.0 * dx / max(w - 1, 1), 2.0 * dy / max(h - 1, 1)],
                device=nbr.device, dtype=nbr.dtype)
            grid = (base + shift).unsqueeze(0).expand(b, -1, -1, -1)
            warped = F.grid_sample(n, grid, mode='bilinear',
                                   padding_mode='border', align_corners=True)
            maps.append((warped * r).sum(dim=1, keepdim=True))
        return torch.cat(maps, dim=1)

    def forward(self, nbr, ref):
        if self.use_corr:
            inp = torch.cat([nbr, ref, self._correlation(nbr, ref)], dim=1)
        else:
            inp = torch.cat([nbr, ref], dim=1)

        o1 = self.relu(self.s1_conv1(inp))
        o1 = self.relu(self.s1_conv2(o1))
        offset1 = self.s1_offset(o1)
        mask1 = self.s1_mask(o1).sigmoid()

        if self.use_coarse:
            pooled = F.avg_pool2d(inp, 2, ceil_mode=True)
            d = self.relu(self.coarse_conv1(pooled))
            d = self.relu(self.coarse_conv2(d))
            d = F.interpolate(self.coarse_offset(d), size=offset1.shape[-2:],
                              mode='bilinear', align_corners=True) * 2.0
            offset1 = offset1 + d
        if self.use_dil:
            d = self.relu(self.dil_conv1(inp))
            d = self.relu(self.dil_conv2(d))
            offset1 = offset1 + self.dil_offset(d)

        self._last_offset1 = offset1
        self._last_mask1 = mask1
        return self.s1_dcn(nbr, offset1, mask1)


class OffsetReuseAlignRefine(OffsetReuseAlign):
    """OffsetReuseAlign + p2-native residual offset predictor.

    The reused p3 offsets provide the coarse field; this branch predicts a
    zero-initialised residual from the p2 feature pair, so training starts from
    exactly the baseline OffsetReuseAlign behaviour.
    """

    def __init__(self, in_ch, reuse_mode='bilinear', hidden=64):
        super().__init__(in_ch, reuse_mode=reuse_mode)
        self.refine_conv1 = nn.Conv2d(in_ch * 2, hidden, 3, 1, 1)
        self.refine_conv2 = nn.Conv2d(hidden, hidden, 3, 1, 1)
        self.refine_offset = nn.Conv2d(hidden, 18, 3, 1, 1)
        nn.init.zeros_(self.refine_offset.weight)
        nn.init.zeros_(self.refine_offset.bias)

    def forward(self, nbr, ref=None, src_module=None):
        if ref is None:
            raise ValueError('OffsetReuseAlignRefine needs ref features')
        off, mask = self._offset_mask(nbr, src_module)
        d = F.relu(self.refine_conv1(torch.cat([nbr, ref], dim=1)))
        d = F.relu(self.refine_conv2(d))
        off = off + self.refine_offset(d)
        return self.dcn(nbr, off, mask)


# ---------------------------------------------------------------------------
# Checkpoint migration: make variants start functionally equal to V2
# ---------------------------------------------------------------------------

_V2_TAPS = [(r, c) for r in range(3) for c in range(3)]
_K5_IDX = {(r, c): (r + 1) * 5 + (c + 1) for r in range(3) for c in range(3)}


def _migrate_k5(mod, sd, prefix):
    """Transplant V2 3x3 weights/offsets/masks into a 5x5 module."""
    # DCN kernel
    w = sd[prefix + 's1_dcn.weight']
    new = torch.zeros_like(mod.s1_dcn.weight)
    for r, c in _V2_TAPS:
        new[:, :, r + 1, c + 1] = w[:, :, r, c]
    with torch.no_grad():
        mod.s1_dcn.weight.copy_(new)
        mod.s1_dcn.bias.copy_(sd[prefix + 's1_dcn.bias'])

    # offsets: 9 taps -> 25 taps (2 channels per tap, (y, x))
    ow = sd[prefix + 's1_offset.weight']
    ob = sd[prefix + 's1_offset.bias']
    new_w = torch.zeros_like(mod.s1_offset.weight)
    new_b = torch.zeros_like(mod.s1_offset.bias)
    for t, (r, c) in enumerate(_V2_TAPS):
        t5 = _K5_IDX[(r, c)]
        new_w[2 * t5:2 * t5 + 2] = ow[2 * t:2 * t + 2]
        new_b[2 * t5:2 * t5 + 2] = ob[2 * t:2 * t + 2]
    with torch.no_grad():
        mod.s1_offset.weight.copy_(new_w)
        mod.s1_offset.bias.copy_(new_b)

    # masks
    mw = sd[prefix + 's1_mask.weight']
    mb = sd[prefix + 's1_mask.bias']
    new_w = torch.zeros_like(mod.s1_mask.weight)
    new_b = torch.zeros_like(mod.s1_mask.bias)
    for t, (r, c) in enumerate(_V2_TAPS):
        new_w[_K5_IDX[(r, c)]] = mw[t]
        new_b[_K5_IDX[(r, c)]] = mb[t]
    with torch.no_grad():
        mod.s1_mask.weight.copy_(new_w)
        mod.s1_mask.bias.copy_(new_b)


def migrate_from_v2(model, sd):
    """Copy a V2 checkpoint's DCN tensors into variant modules in-place.

    Called before the shape-filtered state-dict load.  Handles the two variants
    whose tensors change shape (corr / k5); all other variants have matching
    core shapes and are covered by the filtered load.
    """
    for side in ('1', '2'):
        align = getattr(model.fusion, f'align_ref{side}')
        for level, mod in align.items():
            prefix = f'fusion.align_ref{side}.{level}.'
            if isinstance(mod, DeformAlignV2Adaptive):
                if mod.use_corr:
                    old = sd[prefix + 's1_conv1.weight']
                    with torch.no_grad():
                        mod.s1_conv1.weight.zero_()
                        mod.s1_conv1.weight[:, :old.shape[1]].copy_(old)
                        mod.s1_conv1.bias.copy_(sd[prefix + 's1_conv1.bias'])
                if mod.use_k5:
                    _migrate_k5(mod, sd, prefix)


def load_state_dict_shape_filtered(model, sd, verbose=True):
    """Load tensors with identical shape; leave the rest at module init.

    Returns (missing, mismatched) key lists.  Used when a variant is
    fine-tuned from a V2 checkpoint.
    """
    own = model.state_dict()
    missing, mismatched = [], []
    with torch.no_grad():
        for k, v in own.items():
            if k not in sd:
                missing.append(k)
                continue
            s = sd[k]
            if s.shape != v.shape:
                mismatched.append(k)
                continue
            v.copy_(s.to(device=v.device, dtype=v.dtype))
    if verbose:
        print(f'[dcn-variant] loaded {len(own) - len(missing) - len(mismatched)}'
              f' tensors; {len(missing)} new, {len(mismatched)} reshaped')
    return missing, mismatched
