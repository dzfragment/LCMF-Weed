#!/usr/bin/env python3
"""
P2L-P4 fusion level comparison: 参数量 + GFLOPS (FlopCounterMode)。

3 组实验：
  01_fuse_p4     — fusion_levels=['p4']
  02_fuse_p3     — fusion_levels=['p3']
  03_fuse_p3_p4  — fusion_levels=['p3', 'p4']

统一使用：ConsistentAlign (feature FFT), DAV2, Concat 融合。
"""
from __future__ import annotations
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXP2_DIR = _SCRIPT_DIR.parent.parent  # 仓库根
sys.path.insert(0, str(_EXP2_DIR / 'YOLOv8-pose'))
sys.path.insert(0, str(_EXP2_DIR / 'multi_frame_fusion'))

from models.mf_yolo import MultiFrameYOLO
from models.modules import (DeformAlignV2, DeformAlignV2Single,
                            AdaptiveFusion, AdaptiveFusionSpatial, _NoOp)
from train import apply_ablation

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAINED = str(_EXP2_DIR / 'experiments' / 'single_frame_baseline' / 'results' /
                 'weed_merged_1024' / 'yolo_v8_n_p2_lite_p4' / 'seed42' / 'best.pt')
RESULTS_ROOT = _SCRIPT_DIR / 'results_concat'
INPUT_SIZE = 1024

EXPERIMENTS = OrderedDict({
    '01_fuse_p4':    {'label': 'P4 only',   'fusion_levels': ['p4']},
    '02_fuse_p3':    {'label': 'P3 only',   'fusion_levels': ['p3']},
    '03_fuse_p3_p4': {'label': 'P3 + P4',   'fusion_levels': ['p3', 'p4']},
})


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_params_by_component(model):
    """按组件统计参数量。"""
    comps = OrderedDict()

    def params_of(mod):
        return {id(p): p.numel() for p in mod.parameters()}

    def sump(pd):
        return sum(pd.values())

    backbone = params_of(model.backbone)
    fpn = params_of(model.fpn)
    head = params_of(model.head)
    comps['backbone'] = sump(backbone)
    comps['fpn'] = sump(fpn)
    comps['head'] = sump(head)

    # Fusion internals
    fusion_params = sump(params_of(model.fusion))
    comps['fusion+align'] = fusion_params

    dcn_p = gate_p = proj_p = 0
    counted_ids = set()
    for mod in model.fusion.modules():
        if mod is model.fusion:
            continue
        mod_ids = set(id(p) for p in mod.parameters())
        if mod_ids <= counted_ids:
            continue
        mod_p = sum(p.numel() for p in mod.parameters())
        if isinstance(mod, (DeformAlignV2, DeformAlignV2Single)):
            dcn_p += mod_p; counted_ids |= mod_ids
        elif isinstance(mod, _NoOp):
            pass
        elif isinstance(mod, AdaptiveFusionSpatial):
            for cn, gc in mod.named_children():
                gc_ids = set(id(p) for p in gc.parameters())
                gc_p = sum(p.numel() for p in gc.parameters())
                if cn == 'proj':
                    proj_p += gc_p
                else:
                    gate_p += gc_p
                counted_ids |= gc_ids
        elif hasattr(mod, 'named_children'):
            for cn, gc in mod.named_children():
                gc_ids = set(id(p) for p in gc.parameters())
                gc_p = sum(p.numel() for p in gc.parameters())
                if cn == 'proj':
                    proj_p += gc_p
                else:
                    gate_p += gc_p
                counted_ids |= gc_ids

    comps['dcn'] = dcn_p
    comps['gate'] = gate_p
    comps['proj'] = proj_p
    return comps


def measure_gflops(model, frames):
    """PyTorch FlopCounterMode — per-operator real FLOPs."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad():
            with FlopCounterMode(display=False) as fcm:
                _ = model(frames)
        return fcm.get_total_flops() / 1e9
    except Exception as e:
        print(f'    [WARN] FlopCounterMode: {e}')
        return float('nan')


def build_model(cfg):
    model = MultiFrameYOLO(
        num_classes=1, kpt_shape=(1, 2),
        fusion_levels=cfg['fusion_levels'],
        pretrained_path=None,
        fusion_version=2,
        fusion_gate_type='concat',
        global_align_stage='feature',
        fft_downsample=4,
    ).to(DEVICE)

    apply_ablation(model, no_align=False, fusion_mode='gate_diff')

    ckpt_path = RESULTS_ROOT / cfg['exp_id'] / 'mf_stage3_best.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sd = ckpt.get('model')
        if isinstance(sd, dict):
            model.load_state_dict(sd, strict=False)
            if ckpt.get('consistent_align', False) and getattr(model, 'global_align_stage', '') == 'feature':
                from models.mf_yolo import enable_consistent_align
                enable_consistent_align(model)
        else:
            model = sd.float().to(DEVICE)
        print(f'  Loaded {ckpt_path.name}  epoch={ckpt.get("epoch")}  '
              f'best_pose_ap={ckpt.get("best_pose_ap", 0):.4f}')
    else:
        print(f'  [WARN] No checkpoint at {ckpt_path}')

    model.eval()
    return model


def read_stage3_best(exp_id):
    path = RESULTS_ROOT / exp_id / 'stage3_log.csv'
    if not path.exists():
        return None
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    valid = [r for r in rows if r.get('PoseAP', '').strip()]
    if not valid:
        return None
    best = max(valid, key=lambda r: float(r['PoseAP']))
    return {
        'epoch': int(float(best['epoch'])),
        'BoxAP': float(best['BoxAP']),
        'PoseAP': float(best['PoseAP']),
        'val_loss': float(best['val_loss']),
    }


def main():
    dummy = torch.randn(1, 3, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)

    rows = []

    for exp_id, cfg in EXPERIMENTS.items():
        cfg['exp_id'] = exp_id
        print(f'\n{"=" * 60}')
        print(f'[{exp_id}] {cfg["label"]}  fusion_levels={cfg["fusion_levels"]}')
        print(f'{"=" * 60}')

        model = build_model(cfg)

        total_p, _ = count_params(model)
        comps = count_params_by_component(model)
        print(f'  Params: {total_p:,}')
        for k, v in comps.items():
            if v > 0:
                print(f'    {k}: {v:,} ({100*v/total_p:.1f}%)')

        gflops = measure_gflops(model, dummy)
        print(f'  GFLOPS: {gflops:.2f}')

        stage3 = read_stage3_best(exp_id)

        row = {
            'experiment': exp_id,
            'label': cfg['label'],
            'fusion_levels': ','.join(cfg['fusion_levels']),
            'params_total': total_p,
            'gflops': round(gflops, 2),
        }
        for k, v in comps.items():
            row[f'params_{k}'] = v
        if stage3:
            row['epoch'] = stage3['epoch']
            row['BoxAP'] = stage3['BoxAP']
            row['PoseAP'] = stage3['PoseAP']
            row['val_loss'] = stage3['val_loss']

        rows.append(row)

        del model
        torch.cuda.empty_cache()

    # ── delta vs P4 ──
    p4_pose_ap = None
    for r in rows:
        if 'PoseAP' in r and r['experiment'] == '01_fuse_p4':
            p4_pose_ap = r['PoseAP']
            break
    if p4_pose_ap is not None:
        for r in rows:
            if 'PoseAP' in r:
                r['delta_pose_ap'] = round(r['PoseAP'] - p4_pose_ap, 4)

    # ── table ──
    print('\n')
    print(f'{"Model":<20s} | {"Params":>8s} | {"GFLOPS":>7s} | {"BoxAP":>7s} | {"PoseAP":>7s} | {"ΔPoseAP":>8s} | {"Epoch":>5s}')
    print('-' * 80)
    for r in rows:
        pm = r.get('params_total', 0) / 1e6
        g = r.get('gflops', '')
        ba = r.get('BoxAP', '')
        pa = r.get('PoseAP', '')
        dp = r.get('delta_pose_ap', '')
        ep = r.get('epoch', '')
        print(f'{r["label"]:<20s} | {pm:>7.2f}M | {str(g):>7s} | {str(ba):>7s} | {str(pa):>7s} | {str(dp):>8s} | {str(ep):>5s}')

    # ── CSV ──
    csv_path = _SCRIPT_DIR / 'model_gflops.csv'
    fieldnames = ['experiment', 'label', 'fusion_levels', 'epoch',
                  'params_total', 'gflops',
                  'BoxAP', 'PoseAP', 'delta_pose_ap', 'val_loss',
                  'params_backbone', 'params_fpn', 'params_head',
                  'params_fusion+align', 'params_dcn', 'params_gate', 'params_proj']
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved: {csv_path}')


if __name__ == '__main__':
    main()
