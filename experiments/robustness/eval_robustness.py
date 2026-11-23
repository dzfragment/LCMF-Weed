#!/usr/bin/env python3
"""在 clean / 退化变体上做配对点级评测.

主指标:
  - R@10px @ FP/img=1  (匹配工作点)
  - AP10 (阈值无关)
  - nPE median/mean/p90 (TP 上)
辅助:
  - conf=0.25 的 R/P/FP/img
  - 遮挡: 关键点可见 vs 被遮住的命中率
  - 模糊: per-frame Laplacian 方差

用法:
  python eval_robustness.py --models mf --variants all
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

LCMF = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LCMF / 'YOLOv8-pose'))
sys.path.insert(0, str(LCMF / 'multi_frame_fusion'))
sys.path.insert(0, str(LCMF / 'experiments'))

from mf_dataset import MultiFrameDataset  # noqa: E402
from pad_core import PADSingleFrame, pad_infer, _ap_from_preds  # noqa: E402
import train_pad_mf as tpm  # noqa: E402
from eval_mf_point_ops import _point_ops  # noqa: E402

DATASET = Path(os.environ.get('LCMF_DATA_ROOT', LCMF / 'dataset')).resolve()
CLEAN = DATASET / 'robustness' / 'clean'
VARIANTS = DATASET / 'robustness' / 'variants'
OUT = LCMF / 'experiments' / 'robustness' / 'results'
TAU = 10.0
CONF = 0.25


def load_mf(path, device='cuda'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    # `--no-align` 的 checkpoint 不含任何 align_ref* 参数；旧 checkpoint 也没有
    # 保存 no_align 字段，因此从 state_dict 推断，避免 strict load 失败。
    sd = ck.get('model') or {}
    has_align = any('align_ref' in k for k in sd)
    no_align = bool(ck.get('no_align', not has_align))
    shared = ck.get('shared_offset_level') or 'none'
    args = argparse.Namespace(
        head=ck.get('head_cfg', {}).get('head_type', 'ms'),
        head_mid=ck.get('head_cfg', {}).get('mid', 32),
        out_stride4=int(ck.get('head_cfg', {}).get('out_stride4', False)),
        dw=int(ck.get('head_cfg', {}).get('dw', False)),
        use_p2=int(ck.get('head_cfg', {}).get('use_p2', False)),
        p2_mid=ck.get('head_cfg', {}).get('p2_mid', 32),
        no_size=not ck.get('head_cfg', {}).get('use_size', False),
        fusion_levels=','.join(ck.get('fusion_levels', ['p4'])),
        fusion_version=ck.get('fusion_version', 2),
        fusion_gate_type=ck.get('fusion_gate_type', 'spatial'),
        consistent_align=bool(ck.get('consistent_align', True)),
        fft_downsample=ck.get('fft_downsample', 4),
        no_align=no_align, fusion_mode='gate_diff', cascade_align=False,
        iterative_fusion=False, two_frame=False, shared_cascade=False,
        base_model=ck.get('base_model', 'p34l'),
        shared_offset_level=shared,
        offset_reuse_mode=ck.get('offset_reuse_mode', 'bilinear'),
        convgru_kernel=ck.get('convgru_kernel', 3),
        convgru_layers=ck.get('convgru_layers', 1),
        dav2_single=bool(ck.get('dav2_single', False)),
        dav2_triple=bool(ck.get('dav2_triple', False)),
        global_head=bool(ck.get('global_head', False)),
        global_head_level=ck.get('global_head_level', 'p3'),
        global_head_hidden=ck.get('global_head_hidden', 32),
        global_head_max_shift=ck.get('global_head_max_shift', 32.0),
    )
    model, _ = tpm.load_ckpt(path, args, device)
    return model, ck


def load_sf(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = PADSingleFrame(ck.get('base_model', 'p2l'), ck.get('head_cfg')).to(device)
    sd = ck['model'].float().state_dict() if hasattr(ck['model'], 'state_dict') else ck['model']
    m.load_state_dict(sd, strict=True)
    return m, ck


class Frame0View(torch.utils.data.Dataset):
    """单帧模型用: 从 3 帧序列里只取 frame_0."""
    def __init__(self, ds):
        self.ds = ds
        self.seq_dirs = ds.seq_dirs

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        frames, cls, box, kpt, idx = self.ds[i]
        return frames[0], cls, box, kpt, idx


class NoRefsView(torch.utils.data.Dataset):
    """无互补对照: 把 Frame 1/2 替换成 Frame 0 (与训练 frame-dropout 同做法)."""
    def __init__(self, ds):
        self.ds = ds
        self.seq_dirs = ds.seq_dirs

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        frames, cls, box, kpt, idx = self.ds[i]
        frames = frames.clone()
        frames[1] = frames[0]
        frames[2] = frames[0]
        return frames, cls, box, kpt, idx


def build_loader(root, single=False, no_refs=False, batch_size=8, num_workers=4):
    ds = MultiFrameDataset(str(root), split='test', input_size=1024,
                           params=None, augment=False)
    if no_refs and not single:
        ds = NoRefsView(ds)
    if single:
        ds = Frame0View(ds)
    loader = torch.utils.data.DataLoader(
        ds, batch_size, False, num_workers=num_workers, pin_memory=True,
        collate_fn=MultiFrameDataset.collate_fn)
    return ds, loader


def match_image(pr, gt):
    """返回 (scores, tp, dist, matched_gt_idx). 与 _point_ops 的贪心匹配一致."""
    if len(pr):
        pr = pr[pr[:, 4] >= 0.001]
    order = np.argsort(-pr[:, 4], kind='stable') if len(pr) else np.zeros(0, int)
    pr = pr[order] if len(pr) else pr
    s = pr[:, 4].astype(np.float64) if len(pr) else np.zeros(0)
    tp = np.zeros(len(pr), bool)
    dist = np.full(len(pr), np.nan)
    mgi = np.full(len(pr), -1, int)
    if len(pr) and len(gt):
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
                mgi[pi] = gi
                taken.add(gi)
    return s, tp, dist, mgi


def per_seq_records(preds, ds, meta, thresholds):
    """thresholds: {'conf25': .25, 'fp1': t}. 返回 per-seq list."""
    rows = []
    for idx, d in preds.items():
        seq = os.path.basename(ds.seq_dirs[idx])
        pr, gt = d['pred'], d['gt']
        s, tp, dist, mgi = match_image(pr, gt)
        cov = None
        sharp = None
        if meta:
            cov = meta.get('coverage', {}).get(seq, {}).get('frame0')
            sharp = meta.get('sharpness', {}).get(seq)
        rec = {'seq': seq, 'n_gt': int(len(gt)), 'n_pred': int(len(pr)),
               'n_tp_all': int(tp.sum()),
               'n_pred_all': int(len(pr)),
               'errs_all': dist[tp].tolist() if tp.any() else [],
               'sharpness': sharp}
        cov_ok = False
        if cov is not None:
            cov = np.asarray(cov, bool)
            if len(cov) == len(gt):
                cov_ok = True
                rec['n_gt_vis'] = int((~cov).sum())
                rec['n_gt_cov'] = int(cov.sum())
        for name, th in thresholds.items():
            if th is None:
                continue
            m = s >= th
            nt = int(tp[m].sum())
            ds_ = dist[m & ~np.isnan(dist)]
            rec[f'n_tp_{name}'] = nt
            rec[f'n_pred_{name}'] = int(m.sum())
            rec[f'errs_{name}'] = ds_.tolist() if len(ds_) else []
            if cov_ok:
                matched_gt = set(mgi[m & tp].tolist())
                rec[f'n_tp_vis_{name}'] = int(sum(
                    1 for i in range(len(gt)) if not cov[i] and i in matched_gt))
                rec[f'n_tp_cov_{name}'] = int(sum(
                    1 for i in range(len(gt)) if cov[i] and i in matched_gt))
        rows.append(rec)
    return rows


def eval_one(model, root, single, meta, device, batch_size, no_refs=False):
    ds, loader = build_loader(root, single=single, no_refs=no_refs,
                              batch_size=batch_size)
    preds = pad_infer(model, loader, 1024, conf=0.001, topk=300)
    agg = _point_ops(preds)
    iou_v = torch.linspace(0.5, 0.95, 10, device=next(model.parameters()).device)
    ap = _ap_from_preds(preds, iou_v)
    th25 = CONF
    fp1 = agg.get('at_fp_per_img_1') or {}
    th_fp1 = fp1.get('conf')
    rows = per_seq_records(preds, ds, meta, {'conf25': th25, 'fp1': th_fp1})
    return agg, rows, th_fp1, ap


def summarize_rows(rows, th_name):
    """聚合 per-seq records (含 bootstrap 用的原始 counts)."""
    n_gt = sum(r['n_gt'] for r in rows)
    n_tp = sum(r.get(f'n_tp_{th_name}', 0) for r in rows)
    errs = [e for r in rows for e in r.get(f'errs_{th_name}', [])]
    out = {'n_gt': n_gt, 'n_tp': n_tp,
           'recall': n_tp / max(n_gt, 1),
           'nPE_mean': float(np.mean(errs)) if errs else None,
           'nPE_median': float(np.median(errs)) if errs else None,
           'nPE_p90': float(np.percentile(errs, 90)) if errs else None}
    # visibility strata at the same operating point (only if keys exist)
    if rows and f'n_tp_vis_{th_name}' in rows[0]:
        out['n_gt_vis'] = int(sum(r.get('n_gt_vis', 0) for r in rows))
        out['n_tp_vis'] = int(sum(r.get(f'n_tp_vis_{th_name}', 0) for r in rows))
        out['n_gt_cov'] = int(sum(r.get('n_gt_cov', 0) for r in rows))
        out['n_tp_cov'] = int(sum(r.get(f'n_tp_cov_{th_name}', 0) for r in rows))
        out['recall_vis'] = out['n_tp_vis'] / max(out['n_gt_vis'], 1)
        out['hit_cov'] = out['n_tp_cov'] / max(out['n_gt_cov'], 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(LCMF / 'experiments/concat_ablation12_20261106/results/01_main/pad_stage3_last.pt'))
    ap.add_argument('--sf-ckpt', default=os.environ.get('LCMF_SF_CHECKPOINT', str(LCMF / 'experiments/single_frame_baseline/results/pad_sf_clean/p2l/best.pt')))
    ap.add_argument('--models', default='mf')
    ap.add_argument('--variants', default='all')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-seqs', type=int, default=0)
    ap.add_argument('--no-refs', action='store_true',
                    help='把 Frame 1/2 替换成 Frame 0 (无互补对照)')
    ap.add_argument('--out', default=str(OUT / 'robustness.json'))
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    variants = []
    if args.variants == 'all':
        variants = ['clean'] + sorted(os.listdir(VARIANTS))
    else:
        variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    models = [m.strip() for m in args.models.split(',') if m.strip()]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded = {}
    if 'mf' in models:
        loaded['mf'] = load_mf(args.ckpt, device)
    if 'sf' in models:
        loaded['sf'] = load_sf(args.sf_ckpt, device)

    all_out = {}
    for v in variants:
        if v == 'clean':
            root = CLEAN
        elif v == 'standard':
            root = DATASET / 'weed_sliding_merged'
        else:
            root = VARIANTS / v
        if not root.exists():
            print(f'[skip] missing {root}')
            continue
        meta = None
        if v != 'clean' and (root / 'meta.json').exists():
            meta = json.load(open(root / 'meta.json'))
        print(f'\n=== {v} ===')
        for mname, (model, ck) in loaded.items():
            single = (mname == 'sf')
            use_no_refs = args.no_refs and not single
            agg, rows, th_fp1, ap = eval_one(model, root, single, meta, device,
                                             args.batch_size, no_refs=use_no_refs)
            s25 = summarize_rows(rows, 'conf25')
            s_fp1 = summarize_rows(rows, 'fp1')
            key = f'{v}|{mname}' + ('_norefs' if use_no_refs else '')
            all_out[key] = {
                'variant': v, 'model': mname,
                'no_refs': bool(use_no_refs),
                'meta': {k: meta.get(k) for k in ('factor', 'mode', 'value',
                                                  'area_ratio', 'occluder', 'seed')}
                if meta else {},
                'agg': agg, 'ap': ap, 'conf25': s25, 'fp1': s_fp1,
                'fp1_threshold': th_fp1,
                'per_seq': rows,
            }
            print(f"  {mname}: AP10={agg['point_ap_10px']:.4f} "
                  f"R@FP1={s_fp1['recall']:.4f} nPE={s_fp1['nPE_median']} "
                  f"R@.25={s25['recall']:.4f} "
                  f"pose50={ap['pose_ap50']:.4f} pose={ap['pose_ap']:.4f}")
    with open(args.out, 'w') as f:
        json.dump(all_out, f, ensure_ascii=False, indent=1)
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
