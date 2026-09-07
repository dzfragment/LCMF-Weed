#!/usr/bin/env python3
"""
results/ 下全部模型的参数量 + FLOPs 统计。
默认统计每个实验的 mf_stage3_best.pt，输出 model_profiles_all.csv
"""
from __future__ import annotations
import csv, os, sys, time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXP1_DIR = _SCRIPT_DIR.parent.parent  # 仓库根
sys.path.insert(0, str(_EXP1_DIR / 'YOLOv8-pose'))
sys.path.insert(0, str(_EXP1_DIR / 'multi_frame_fusion'))

from models.mf_yolo import MultiFrameYOLO
from models.modules import (DeformAlignV2, DeformAlignV2Single,
                            AdaptiveFusion, AdaptiveFusionAvg,
                            AdaptiveFusionConcat, AdaptiveFusionSpatial, _NoOp)
from models.fft_align import apply_consistent_image_align, FFTAlign
from train import apply_ablation, apply_dav2_single, apply_dav2_triple

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAINED = os.environ.get('PRETRAINED') or str(
    _EXP1_DIR / 'experiments' / 'single_frame_baseline' / 'results' /
    'yolo_v8_n_p2_lite_p4' / 'seed42' / 'best.pt')
RESULTS_ROOT = Path(os.environ.get('RESULTS_ROOT', _SCRIPT_DIR / 'results'))
INPUT_SIZE = 1024

# ── 全部 12 组配置 ────────────────────────────────────────────────
EXPERIMENTS = OrderedDict({
    # 主实验 (00-06)
    '00_full_fft_dcn_spatial': {
        'label': 'FFT + DCN + Spatial', 'group': '架构组件',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'spatial', 'fusion_mode': 'gate_diff',
        'no_align': False, 'dav2_single': False, 'dav2_triple': False,
        'rgb_fft': False,
    },
    '01_remove_fft': {
        'label': '去除 FFT', 'group': '架构组件',
        'global_align_stage': 'none', 'fft_downsample': 4,
        'fusion_gate_type': 'spatial', 'fusion_mode': 'gate_diff',
        'no_align': False, 'dav2_single': False, 'dav2_triple': False,
        'rgb_fft': False,
    },
    '02_remove_dcn': {
        'label': '去除 DCN', 'group': '架构组件',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'spatial', 'fusion_mode': 'gate_diff',
        'no_align': True, 'dav2_single': False, 'dav2_triple': False,
        'rgb_fft': False,
    },
    '03_remove_spatial': {
        'label': '去除 Spatial (Avg)', 'group': '架构组件',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'avg', 'fusion_mode': 'avg',
        'no_align': False, 'dav2_single': False, 'dav2_triple': False,
        'rgb_fft': False,
    },
    '05_ablate_concat_fusion': {
        'label': 'FFT + DCN + Concat', 'group': '架构组件',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'concat', 'fusion_mode': 'gate_diff',
        'no_align': False, 'dav2_single': False, 'dav2_triple': False,
        'rgb_fft': False,
    },
    '06_ablate_dav2_single': {
        'label': 'FFT + Single DAV2', 'group': '架构组件',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'spatial', 'fusion_mode': 'gate_diff',
        'no_align': False, 'dav2_single': True, 'dav2_triple': False,
        'rgb_fft': False,
    },
    # 补充实验 (07-12, 跳过 11)
    '07_ablate_dav2_triple': {
        'label': 'DAV2 3-stage', 'group': 'DCN深度',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'spatial', 'fusion_mode': 'gate_diff',
        'no_align': False, 'dav2_single': False, 'dav2_triple': True,
        'rgb_fft': False,
    },
    '08_fusion_channel': {
        'label': 'Channel Gate', 'group': '融合方式',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'channel', 'fusion_mode': 'gate_diff',
        'no_align': False, 'dav2_single': False, 'dav2_triple': False,
        'rgb_fft': False,
    },
    '09_fusion_convgru': {
        'label': 'ConvGRU', 'group': '融合方式',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'convgru', 'fusion_mode': 'gate_diff',
        'no_align': False, 'dav2_single': False, 'dav2_triple': False,
        'rgb_fft': False,
    },
    '10_fusion_tsa': {
        'label': 'TSA', 'group': '融合方式',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'tsa', 'fusion_mode': 'gate_diff',
        'no_align': False, 'dav2_single': False, 'dav2_triple': False,
        'rgb_fft': False,
    },
    '12_train_stage3_only': {
        'label': 'Stage3 Only', 'group': '训练策略',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'spatial', 'fusion_mode': 'gate_diff',
        'no_align': False, 'dav2_single': False, 'dav2_triple': False,
        'rgb_fft': False,
    },
    '13_remove_fft_dcn_concat': {
        'label': 'Concat (无 FFT/DCN)', 'group': '组合消融',
        'global_align_stage': 'none', 'fft_downsample': 4,
        'fusion_gate_type': 'concat', 'fusion_mode': 'gate_diff',
        'no_align': True, 'dav2_single': False, 'dav2_triple': False,
        'rgb_fft': False,
    },
    '14_fft_dcn_concat_seed123': {
        'label': 'FFT + DCN + Concat (seed123)', 'group': '种子鲁棒性',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'concat', 'fusion_mode': 'gate_diff',
        'no_align': False, 'dav2_single': False, 'dav2_triple': False,
        'rgb_fft': False,
    },
    '15_fft_dav2_single_concat': {
        'label': 'FFT + Single DAV2 + Concat', 'group': '组合消融',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'concat', 'fusion_mode': 'gate_diff',
        'no_align': False, 'dav2_single': True, 'dav2_triple': False,
        'rgb_fft': False,
    },
    '16_fft_dav2_triple_concat': {
        'label': 'FFT + Triple DAV2 + Concat', 'group': '组合消融',
        'global_align_stage': 'feature', 'fft_downsample': 4,
        'fusion_gate_type': 'concat', 'fusion_mode': 'gate_diff',
        'no_align': False, 'dav2_single': False, 'dav2_triple': True,
        'rgb_fft': False,
    },
})


# ── 统计函数 ───────────────────────────────────────────────────────
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_params_by_component(model):
    """按组件统计参数量，使用 param id 集合避免重复计数。"""
    comps = OrderedDict()

    def params_of(module):
        return {id(p): p.numel() for p in module.parameters()}

    def sum_params(param_dict):
        return sum(param_dict.values())

    # Backbone / FPN / Head — 互不重叠
    backbone = params_of(model.backbone)
    fpn = params_of(model.fpn)
    head = params_of(model.head)
    comps['backbone'] = sum_params(backbone)
    comps['fpn'] = sum_params(fpn)
    comps['head'] = sum_params(head)

    # FFT (零参数，仅占位)
    fft_p = 0
    if model.fft_align_r1 is not None:
        fft_p += sum_params(params_of(model.fft_align_r1))
    if model.fft_align_r2 is not None:
        fft_p += sum_params(params_of(model.fft_align_r2))
    comps['fft'] = fft_p

    # Fusion 内部细分 — 找出所有子模块，按类型分类
    fusion_params = sum_params(params_of(model.fusion))
    comps['fusion+align'] = fusion_params

    dcn_p = 0
    gate_p = 0
    proj_p = 0
    convgru_p = 0
    tsa_p = 0

    # 遍历 fusion 的所有子模块，用 id 集合追踪已计入的参数避免重复
    counted_ids = set()
    for mod_name, mod in model.fusion.named_modules():
        if mod is model.fusion:
            continue
        mod_ids = set(id(p) for p in mod.parameters())
        if mod_ids <= counted_ids:
            continue  # 该模块参数已全部被父模块计入
        mod_p = sum(p.numel() for p in mod.parameters())
        mod_cls = type(mod).__name__.lower()

        if isinstance(mod, (DeformAlignV2, DeformAlignV2Single)):
            dcn_p += mod_p
            counted_ids |= mod_ids
        elif isinstance(mod, _NoOp):
            pass
        elif isinstance(mod, AdaptiveFusionSpatial):
            # Spatial gate: 按子模块细分 gate/proj
            for cn, gc in mod.named_children():
                gc_ids = set(id(p) for p in gc.parameters())
                gc_p = sum(p.numel() for p in gc.parameters())
                if cn == 'proj':
                    proj_p += gc_p
                else:
                    gate_p += gc_p
                counted_ids |= gc_ids
        elif isinstance(mod, (AdaptiveFusion, AdaptiveFusionConcat, AdaptiveFusionAvg)):
            has_proj = False
            for cn, gc in mod.named_children():
                gc_ids = set(id(p) for p in gc.parameters())
                gc_p = sum(p.numel() for p in gc.parameters())
                if cn == 'proj':
                    proj_p += gc_p; has_proj = True
                else:
                    gate_p += gc_p
                counted_ids |= gc_ids
            # If the module IS the fusion block and has no children (channel gate),
            # its params go to gate
            if not has_proj and mod_p > 0 and not any(
                isinstance(gc, (DeformAlignV2, DeformAlignV2Single))
                for _, gc in mod.named_children()):
                pass  # already counted by children
        elif 'convgru' in mod_cls:
            convgru_p += mod_p
            counted_ids |= mod_ids
        elif 'tsa' in mod_cls or 'temporal' in mod_cls:
            tsa_p += mod_p
            counted_ids |= mod_ids
        # else: skip (intermediate container modules)

    comps['dcn'] = dcn_p
    comps['gate'] = gate_p
    comps['proj'] = proj_p
    if convgru_p > 0:
        comps['convgru'] = convgru_p
    if tsa_p > 0:
        comps['tsa'] = tsa_p

    return comps


def measure_gflops(model, frames):
    """Measure GFLOPS via PyTorch FlopCounterMode (per-operator, real FLOPs)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad():
            with FlopCounterMode(display=False) as fcm:
                _ = model(frames)
        return fcm.get_total_flops() / 1e9
    except Exception as e:
        print(f'    [WARN] FlopCounterMode failed: {e}')
        return float('nan')


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


def build_model(cfg):
    model = MultiFrameYOLO(
        num_classes=1, kpt_shape=(1, 2),
        fusion_levels=['p4'],
        pretrained_path=None,
        fusion_version=2,
        fusion_gate_type=cfg['fusion_gate_type'],
        global_align_stage=cfg['global_align_stage'],
        fft_downsample=cfg['fft_downsample'],
    ).to(DEVICE)

    apply_ablation(model, no_align=cfg['no_align'], fusion_mode=cfg['fusion_mode'])

    if cfg['dav2_triple']:
        apply_dav2_triple(model)
    elif cfg['dav2_single']:
        apply_dav2_single(model)

    if cfg['rgb_fft']:
        apply_consistent_image_align(model, downsample=cfg['fft_downsample'])

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
    else:
        print(f'    [WARN] No checkpoint at {ckpt_path}')

    model.eval()
    return model


# ── Main ───────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', type=str, default='', help='Single experiment ID')
    args = ap.parse_args()

    run_experiments = EXPERIMENTS
    if args.exp:
        run_experiments = {args.exp: EXPERIMENTS[args.exp]}

    dummy = torch.randn(1, 3, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)

    rows = []
    control_pose_ap = None

    for exp_id, cfg in run_experiments.items():
        cfg['exp_id'] = exp_id
        print(f'\n{"=" * 70}')
        print(f'[{exp_id}] {cfg["label"]}  ({cfg["group"]})')
        print(f'{"=" * 70}')

        model = build_model(cfg)

        # Params
        total_p, trainable_p = count_params(model)
        comps = count_params_by_component(model)
        print(f'  Params: {total_p:,} total  ({trainable_p:,} trainable)')
        for k, v in comps.items():
            if v > 0:
                print(f'    {k}: {v:,} ({100*v/total_p:.1f}%)')

        # FLOPs
        gflops = measure_gflops(model, dummy)
        print(f'  GFLOPS: {gflops:.2f}')

        # Val metrics from stage3 log
        stage3 = read_stage3_best(exp_id)
        if stage3:
            if control_pose_ap is None and exp_id.startswith('00_'):
                control_pose_ap = stage3['PoseAP']

        row = {
            'experiment': exp_id,
            'label': cfg['label'],
            'group': cfg['group'],
            'params_total': total_p,
            'params_trainable': trainable_p,
            'gflops': round(gflops, 2) if not np.isnan(gflops) else '',
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

    # delta
    if control_pose_ap is not None:
        for r in rows:
            if 'PoseAP' in r:
                r['delta_pose_ap'] = round(r['PoseAP'] - control_pose_ap, 4)

    # ── 输出表格 ──────────────────────────────────────────────
    print('\n')
    print('=' * 140)
    print(f'{"模型":<30s} | {"类别":<8s} | {"Params(M)":>9s} | {"GFLOPS":>7s} | {"BoxAP(val)":>10s} | {"PoseAP(val)":>10s} | {"ΔPoseAP":>8s} | {"Val Loss":>8s} | {"Epoch":>5s}')
    print('-' * 140)
    for r in rows:
        pm = r.get('params_total', 0) / 1e6
        g = r.get('gflops', '')
        ba = r.get('BoxAP', '')
        pa = r.get('PoseAP', '')
        dp = r.get('delta_pose_ap', '')
        vl = r.get('val_loss', '')
        ep = r.get('epoch', '')
        print(f'{r["label"]:<30s} | {r["group"]:<8s} | {pm:>9.2f} | {str(g):>7s} | {str(ba):>10s} | {str(pa):>10s} | {str(dp):>8s} | {str(vl):>8s} | {str(ep):>5s}')

    # ── 参数量分解 ──────────────────────────────────────────
    comp_keys = ['backbone', 'fpn', 'head', 'fft', 'dcn', 'gate', 'proj', 'convgru', 'tsa']
    print('\n=== 参数量分解 ===')
    hdr = f'{"模型":<30s} |'
    for ck in comp_keys:
        hdr += f' {ck:>9s} |'
    hdr += f' {"总计":>9s}'
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        row = f'{r["label"]:<30s} |'
        for ck in comp_keys:
            v = r.get(f'params_{ck}', 0)
            if v > 0:
                row += f' {v/1e3:>8.1f}K |'
            else:
                row += f' {"—":>9s} |'
        row += f' {r["params_total"]/1e6:>8.2f}M'
        print(row)

    # ── Save CSV ────────────────────────────────────────────
    csv_path = _SCRIPT_DIR / 'results_tables' / 'model_profiles_all.csv'
    fieldnames = ['experiment', 'label', 'group', 'epoch',
                  'params_total', 'params_trainable', 'gflops',
                  'BoxAP', 'PoseAP', 'delta_pose_ap', 'val_loss']
    for ck in comp_keys + ['fusion+align']:
        fieldnames.append(f'params_{ck}')
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved: {csv_path}')


if __name__ == '__main__':
    main()
