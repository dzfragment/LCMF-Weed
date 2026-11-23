#!/usr/bin/env python3
"""多帧 PAD checkpoint 的点级部署工作点评测.

主指标 (用户口径):
  - recall@10px  (= 20px 直径光斑的命中率)
  - nPE median / mean / p90 (px, 1024 输入空间)
  - FP/img, R@FP=1, R@precision>=0.90
  - AP10 作为排序质量辅助指标

用法:
  python eval_mf_point_ops.py [--ckpt name=path ...] [--out results.json]

默认扫描 experiments/pad_mf_* 下的 stage3 best/last.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_ROOT, 'YOLOv8-pose'))
sys.path.insert(0, os.path.join(_ROOT, 'multi_frame_fusion'))

from mf_dataset import MultiFrameDataset  # noqa: E402
from pad_core import pad_infer, point_ap  # noqa: E402
import train_pad_mf as tpm  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = os.path.join(os.getenv('LCMF_DATA_ROOT', os.path.join(_ROOT, 'dataset')),
                    'weed_sliding_merged')
TAU = 10.0
CONF = 0.25


def _args(ck):
    hc = ck.get('head_cfg') or {}
    return argparse.Namespace(
        head=hc.get('head_type', 'ms'), head_mid=hc.get('mid', 32),
        out_stride4=int(hc.get('out_stride4', False)),
        dw=int(hc.get('dw', False)), use_p2=int(hc.get('use_p2', False)),
        p2_mid=hc.get('p2_mid', 32), no_size=not hc.get('use_size', False),
        fusion_levels=','.join(ck.get('fusion_levels', ['p4'])),
        fusion_version=ck.get('fusion_version', 2),
        fusion_gate_type=ck.get('fusion_gate_type', 'spatial'),
        consistent_align=bool(ck.get('consistent_align', True)),
        fft_downsample=ck.get('fft_downsample', 4),
        no_align=bool(ck.get('no_align', False)), fusion_mode='gate_diff', cascade_align=False,
        iterative_fusion=False, two_frame=False, shared_cascade=False,
        base_model=ck.get('base_model', 'p34l'),
        shared_offset_level=ck.get('shared_offset_level') or 'none',
        offset_reuse_mode=ck.get('offset_reuse_mode', 'bilinear'),
        dav2_single=bool(ck.get('dav2_single', False)),
        dav2_triple=bool(ck.get('dav2_triple', False)),
        global_head=bool(ck.get('global_head', False)),
        global_head_level=ck.get('global_head_level', 'p3'),
        global_head_hidden=ck.get('global_head_hidden', 32),
        global_head_max_shift=ck.get('global_head_max_shift', 32.0),
    )


def _point_ops(preds):
    """Per-image greedy point matching at tau=10, all scores >= .001."""
    imgs = []
    n_gt = 0
    for d in preds.values():
        pr, gt = d['pred'], d['gt']
        n_gt += len(gt)
        if len(pr):
            pr = pr[pr[:, 4] >= 0.001]
        s = np.zeros(len(pr))
        tp = np.zeros(len(pr), bool)
        dist = np.full(len(pr), np.nan)
        if len(pr):
            order = np.argsort(-pr[:, 4], kind='stable')
            pr = pr[order]
            s = pr[:, 4].astype(np.float64)
            if len(gt):
                gx, gy = gt[:, 4].astype(np.float64), gt[:, 5].astype(np.float64)
                taken = set()
                for pi in range(len(pr)):
                    if len(taken) == len(gt):
                        break
                    dd = np.hypot(gx - pr[pi, 5], gy - pr[pi, 6])
                    if taken:
                        dd[list(taken)] = 1e9
                    gi = int(np.argmin(dd))
                    if dd[gi] <= TAU:
                        tp[pi] = True
                        dist[pi] = dd[gi]
                        taken.add(gi)
        imgs.append((s, tp, dist))

    # threshold-free AP10 (pad_core.point_ap uses the same greedy rule)
    ap10 = point_ap(preds, tau_px=TAU)

    S = np.concatenate([x[0] for x in imgs]) if imgs else np.zeros(0)
    T = np.concatenate([x[1] for x in imgs]) if imgs else np.zeros(0, bool)
    D = np.concatenate([x[2] for x in imgs]) if imgs else np.zeros(0)
    order = np.argsort(-S, kind='stable')
    S, T, D = S[order], T[order], D[order]
    cum = np.cumsum(T)
    fp = np.arange(1, len(T) + 1) - cum
    rec = cum / max(n_gt, 1)
    prec = cum / (cum + fp + 1e-9)
    fpi = fp / max(len(preds), 1)

    def at_conf(cf):
        m = S >= cf
        n_pred, n_tp = int(m.sum()), int(T[m].sum())
        ds = D[m & ~np.isnan(D)]
        return {
            'conf': float(cf), 'n_pred': n_pred, 'n_tp': n_tp,
            'recall': n_tp / max(n_gt, 1),
            'precision': n_tp / max(n_pred, 1),
            'fp_per_img': (n_pred - n_tp) / max(len(preds), 1),
            'nPE_median_px': float(np.median(ds)) if len(ds) else None,
            'nPE_mean_px': float(np.mean(ds)) if len(ds) else None,
            'nPE_p90_px': float(np.percentile(ds, 90)) if len(ds) else None,
        }

    def at_fp(budget):
        ok = np.where(fpi <= budget)[0]
        if not len(ok):
            return None
        return at_conf(float(S[ok[-1]]))

    def at_prec(target):
        ok = np.where(prec >= target)[0]
        if not len(ok):
            return None
        return at_conf(float(S[ok[np.argmax(rec[ok])]]))

    out = {'n_gt': int(n_gt), 'n_img': len(preds), 'point_ap_10px': ap10,
           'at_conf_0.25': at_conf(CONF),
           'at_fp_per_img_1': at_fp(1.0),
           'at_precision_0.90': at_prec(0.90)}
    return out


def _load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model, _ = tpm.load_ckpt(path, _args(ck), DEV)
    return model, ck


def _build_loader():
    ds = MultiFrameDataset(DATA, split='test', input_size=1024, augment=False)
    return torch.utils.data.DataLoader(
        ds, 4, False, num_workers=4, pin_memory=True,
        collate_fn=MultiFrameDataset.collate_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', default=[],
                    help='name=path; 可重复. 默认扫描 pad_mf_* 的 stage3 best/last')
    ap.add_argument('--out', default=os.path.join(_SCRIPT_DIR, 'mf_point_ops.json'))
    args = ap.parse_args()

    ckpts = []
    if args.ckpt:
        for s in args.ckpt:
            name, path = s.split('=', 1)
            ckpts.append((name, path))
    else:
        for pat in ('pad_mf_p34l_nosize', 'pad_mf_p2l_p4_p2_nosize',
                    'pad_mf_p2l_nosize', 'pad_mf_p2l_nosize_s1',
                    'pad_mf_p34l_nosize_s1', 'pad_mf_p34l'):
            d = os.path.join(_SCRIPT_DIR, pat)
            for tag in ('best', 'last'):
                p = os.path.join(d, f'pad_stage3_{tag}.pt')
                if os.path.exists(p):
                    ckpts.append((f'{pat}_{tag}', p))

    loader = _build_loader()
    results = {}
    for name, path in ckpts:
        print(f'\n=== {name} ===\n{path}')
        model, ck = _load_model(path)
        preds = pad_infer(model, loader, 1024, conf=0.001, topk=300)
        r = _point_ops(preds)
        r['base_model'] = ck.get('base_model')
        r['epoch'] = ck.get('epoch')
        r['head_cfg'] = ck.get('head_cfg')
        results[name] = r
        a = r['at_conf_0.25']
        print(f"R@10={a['recall']:.4f} P={a['precision']:.4f} "
              f"nPE={a['nPE_median_px']:.2f}/{a['nPE_mean_px']:.2f}/"
              f"{a['nPE_p90_px']:.2f} FP/img={a['fp_per_img']:.2f} "
              f"AP10={r['point_ap_10px']:.4f}")
        del model
        torch.cuda.empty_cache()

    with open(args.out, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
