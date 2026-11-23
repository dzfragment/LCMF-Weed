#!/usr/bin/env python3
"""单帧 PAD (Point-Anchored Detector) 训练 — exp3 基座.

把 few_shot_weed_stem 的新检测头 (点热图 + offset + size, 无 TAL/DFL) 移植到
exp3 的 lite 基座 (p34l / p2l_p4), 在单帧关键帧数据集上训练.

训练协议与 exp3 其他单帧模型对齐: COCO v8_n_pose.pt 部分加载, batch 8,
accumulate 8, lr0 0.01, 500 epochs, warmup 3 epochs, mosaic 最后 10 epoch 关闭.
额外: 前 --freeze-epochs 冻结 backbone/FPN, 只训练新 PAD 头.

用法:
  python train_pad.py --epochs 500 --batch-size 8 --accumulate 8 --lr0 0.01 \
      --backbone-mult 1.0 --pretrained ../../YOLOv8-pose/v8_n_pose.pt \
      --data-dir /path/to/weed_sliding_merged_keyframe \
      --out-dir results/pad_sf_p34l
"""
import argparse
import copy
import csv
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import yaml

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
YOLO_DIR = os.path.join(REPO_ROOT, 'YOLOv8-pose')
MF_DIR = os.path.join(REPO_ROOT, 'multi_frame_fusion')
sys.path.insert(0, YOLO_DIR)
sys.path.insert(0, MF_DIR)

from utils import util  # noqa: E402
from utils.dataset import Dataset  # noqa: E402
from pad_core import (PADSingleFrame, evaluate_pad, evaluate_points,  # noqa: E402
                       match_points, point_loss)

DEFAULT_SF_CKPT = os.path.join(
    _SCRIPT_DIR, 'results', 'weed_sliding_merged_keyframe',
    'yolo_v8_n_p34_lite', 'seed42', 'best.pt')


def build_file_list(data_dir, split):
    split_dir = os.path.join(data_dir, split)
    img_dir = os.path.join(split_dir, 'images')
    return sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                   if f.endswith(('.jpg', '.jpeg', '.png'))])


def build_params(args):
    with open(os.path.join(YOLO_DIR, 'utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)
    params.update({
        'nc': 1, 'names': {0: 'weed'}, 'kpt_shape': [1, 2], 'flip_index': [0],
        'mosaic': args.mosaic, 'mix_up': args.mix_up, 'degrees': 10.0,
        'scale': 0.5, 'shear': 0.0, 'translate': 0.1, 'flip_ud': 0.0,
        'flip_lr': 0.5, 'lr0': args.lr0, 'lrf': 0.01,
    })
    return params


def load_backbone_weights(model, path):
    """Partial-load net/fpn from a YOLOv8-pose or PAD checkpoint."""
    if not path or not os.path.exists(path):
        print(f'[warn] pretrained not found: {path}')
        return 0
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        model_data = ckpt['model']
        sd = model_data.state_dict() if hasattr(model_data, 'state_dict') else model_data
    else:
        sd = ckpt
    model_dict = model.state_dict()
    loaded = {}
    for k, v in sd.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            loaded[k] = v.float()
    missing = [k for k in model_dict if k.startswith(('net.', 'fpn.')) and k not in loaded]
    model.load_state_dict(loaded, strict=False)
    print(f'[init] loaded {len(loaded)} tensors from {os.path.basename(path)} '
          f'(missing net/fpn: {len(missing)})')
    return len(loaded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=os.path.join(os.getenv('LCMF_DATA_ROOT', os.path.join(REPO_ROOT, 'dataset')),
                                              'weed_sliding_merged_keyframe'))
    ap.add_argument('--out-dir', default=os.path.join(_SCRIPT_DIR, 'results', 'pad_sf_p34l'))
    ap.add_argument('--base-model', default='p34l',
                    choices=['v8n', 'p3', 'p34', 'p34l', 'p2', 'p2l',
                             'p2l_p4', 'p2l_p4_only'],
                    help='单帧基座: v8n=SF-ORIG(P3/4/5) p3=SF-P3 p34=SF-P34 '
                         'p34l=SF-P34L p2=SF-P2 p2l=SF-P2L '
                         'p2l_p4=SF-P2L-P4兼容名(=p2l) p2l_p4_only=单尺度P4头; '
                         '统一接 PAD 点热图头')
    ap.add_argument('--pretrained', default=DEFAULT_SF_CKPT,
                    help='backbone/FPN 初始化权重 (默认 SF-P34L 关键帧基线)')
    ap.add_argument('--head', default='ms', choices=['ms', 'simple'])
    ap.add_argument('--head-mid', type=int, default=32)
    ap.add_argument('--out-stride4', type=int, default=0)
    ap.add_argument('--dw', type=int, default=0)
    ap.add_argument('--use-p2', type=int, default=0)
    ap.add_argument('--p2-mid', type=int, default=32)
    ap.add_argument('--no-size', action='store_true',
                    help='去掉 size 分支, 只输出点 (heatmap+offset); 评估用点距离匹配')
    ap.add_argument('--input-size', type=int, default=1024)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--accumulate', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--lr0', type=float, default=0.01)
    ap.add_argument('--backbone-mult', type=float, default=1.0)
    ap.add_argument('--freeze-epochs', type=int, default=3,
                    help='前 N 个 epoch 冻结 backbone/FPN, 只训 PAD 头')
    ap.add_argument('--resume', action='store_true',
                    help='从 out-dir/last.pt 恢复 (含 optimizer/scheduler/EMA)')
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--topk', type=int, default=300)
    ap.add_argument('--patience', type=int, default=100,
                    help='连续 N 个 eval 无提升则早停 (0=关闭)')
    ap.add_argument('--min-delta', type=float, default=0.003,
                    help='提升超过该值才算改进')
    ap.add_argument('--min-epochs', type=int, default=100,
                    help='至少训练到该 epoch 才允许早停')
    ap.add_argument('--mosaic', type=float, default=0.5)
    ap.add_argument('--mix-up', type=float, default=0.1)
    ap.add_argument('--loss-hm', type=float, default=1.0)
    ap.add_argument('--loss-off', type=float, default=1.0)
    ap.add_argument('--loss-size', type=float, default=0.5)
    ap.add_argument('--gauss-overlap', type=float, default=0.7)
    ap.add_argument('--gauss-sigma-div', type=float, default=3.0)
    ap.add_argument('--gauss-fixed-radius', type=int, default=0,
                    help='>0 时用固定高斯半径(cells), 不依赖框尺寸 (坏框数据集用)')
    ap.add_argument('--skip-invisible', action='store_true',
                    help='跳过增广后 vis=0 的越界关键点 (标准 YOLO 的 kpt_mask 行为)')
    ap.add_argument('--size-iou-w', type=float, default=0.0)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--select-metric', default='pose_ap50',
                    choices=['pose_ap50', 'point_error'],
                    help='checkpoint 选择指标: pose_ap50=旧行为; '
                         'point_error=20px 直径召回 >= min-recall 约束下取平均点误差最小')
    ap.add_argument('--min-recall', type=float, default=0.85,
                    help='point_error 选点的最低 20px 直径召回约束')
    ap.add_argument('--min-delta-err', type=float, default=0.005,
                    help='point_error 选点: 平均误差至少改善该值(px)才算更好')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda'
    params = build_params(args)
    # 与 exp3 单帧训练一致的 weight decay 缩放
    params['weight_decay'] *= args.batch_size * args.accumulate / 64.0

    head_cfg = {
        'head_type': args.head, 'mid': args.head_mid,
        'out_stride4': bool(args.out_stride4), 'dw': bool(args.dw),
        'use_p2': bool(args.use_p2), 'p2_mid': args.p2_mid,
        'use_size': not args.no_size,
    }
    model = PADSingleFrame(args.base_model, head_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'PAD single-frame: base={args.base_model} head={head_cfg} '
          f'stride={model.stride} params={n_params:,}')

    resume_state = None
    start_epoch = 0
    best_pose = -1.0
    best_err = float('inf')
    best_recall = -1.0
    best_epoch = 0
    last_path = os.path.join(args.out_dir, 'last.pt')
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location='cpu', weights_only=False)
        sd = ck['model'].state_dict() if hasattr(ck['model'], 'state_dict') else ck['model']
        model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
        start_epoch = int(ck.get('epoch', 0))
        best_pose = float(ck.get('best_pose_ap50', -1.0))
        best_err = float(ck.get('best_npe_mean_px', float('inf')))
        best_recall = float(ck.get('best_recall20D', -1.0))
        best_epoch = int(ck.get('best_epoch', start_epoch))
        resume_state = ck
        print(f'[resume] {last_path} @ epoch {start_epoch}, '
              f'best_pose={best_pose:.4f} best_err={best_err:.4f} '
              f'best_recall={best_recall:.4f}')
    else:
        load_backbone_weights(model, args.pretrained)

    train_files = build_file_list(args.data_dir, 'train')
    val_files = build_file_list(args.data_dir, 'val')
    test_files = build_file_list(args.data_dir, 'test')
    print(f'Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}')

    train_dset = Dataset(train_files, args.input_size, params, True)
    val_dset = Dataset(val_files, args.input_size, params, False)
    test_dset = Dataset(test_files, args.input_size, params, False)
    train_loader = torch.utils.data.DataLoader(
        train_dset, args.batch_size, True, num_workers=args.num_workers,
        pin_memory=True, collate_fn=Dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dset, args.batch_size, False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=Dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test_dset, args.batch_size, False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=Dataset.collate_fn)

    # param groups: net 用 backbone_mult 缩放; bias / BN weight / 其余 weight 分开.
    # 注意: backbone 的 BN weight 必须一起进优化器 (漏掉会静默冻结).
    def make_optimizer():
        groups = {(loc, kind): [] for loc in ('net', 'rest')
                  for kind in ('bias', 'bn', 'w')}
        seen = set()
        wd = params['weight_decay']  # 已在 main 中按 batch*accumulate/64 缩放
        for name, v in model.named_modules():
            loc = 'net' if (name == 'net' or name.startswith('net.')) else 'rest'
            if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter) \
                    and v.bias.requires_grad and id(v.bias) not in seen:
                groups[(loc, 'bias')].append(v.bias)
                seen.add(id(v.bias))
            if isinstance(v, torch.nn.BatchNorm2d):
                if v.weight.requires_grad and id(v.weight) not in seen:
                    groups[(loc, 'bn')].append(v.weight)
                    seen.add(id(v.weight))
            elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter) \
                    and v.weight.requires_grad and id(v.weight) not in seen:
                groups[(loc, 'w')].append(v.weight)
                seen.add(id(v.weight))
        og = []
        for loc in ('net', 'rest'):
            lr = args.lr0 * (args.backbone_mult if loc == 'net' else 1.0)
            for kind in ('bias', 'bn', 'w'):
                ps = groups[(loc, kind)]
                if not ps:
                    continue
                g = {'params': ps, 'initial_lr': lr, 'lr': lr,
                     'is_bias': kind == 'bias'}
                if kind == 'w':
                    g['weight_decay'] = wd
                else:
                    g['weight_decay'] = 0.0
                og.append(g)
        n_total = sum(len(g['params']) for g in og)
        n_train = sum(1 for p in model.parameters() if p.requires_grad)
        print(f'[optimizer] grouped {n_total}/{n_train} trainable tensors, '
              f'wd={wd:.2e}')
        if n_total != n_train:
            raise RuntimeError(
                f'optimizer dropped {n_train - n_total} trainable tensors '
                '(would be silently frozen)')
        return torch.optim.SGD(og, args.lr0, params['momentum'], nesterov=True)

    optimizer = make_optimizer()

    def lr_fn(epoch):
        return (1 - epoch / max(args.epochs, 1)) * (1.0 - params['lrf']) + params['lrf']

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn, last_epoch=-1)
    ema = util.EMA(model)
    scaler = torch.cuda.amp.GradScaler()

    scheduler_loaded = False
    if resume_state is not None:
        for key, obj in (('optimizer', optimizer), ('scheduler', scheduler),
                         ('scaler', scaler)):
            st = resume_state.get(key)
            if st is None:
                continue
            try:
                obj.load_state_dict(st)
                scheduler_loaded = scheduler_loaded or key == 'scheduler'
                print(f'  {key} state restored')
            except Exception as e:
                print(f'  {key} state skipped: {e}')
        if resume_state.get('ema_state') is not None:
            ema.ema.load_state_dict(
                {k: v.float() for k, v in resume_state['ema_state'].items()})
            ema.updates = int(resume_state.get('ema_updates', 0))
            print(f'  EMA state restored (updates={ema.updates})')
    if not scheduler_loaded:
        for _ in range(start_epoch):
            scheduler.step()

    def set_backbone_frozen(frozen):
        for p in list(model.net.parameters()) + list(model.fpn.parameters()):
            p.requires_grad_(not frozen)
        if frozen:
            model.net.eval()
            model.fpn.eval()

    log_mode = 'a' if resume_state is not None else 'w'
    log_f = open(os.path.join(args.out_dir, 'step.csv'), log_mode, newline='')
    writer = csv.DictWriter(log_f, fieldnames=[
        'epoch', 'loss', 'loss_hm', 'loss_off', 'loss_size',
        'point_recall_20D', 'point_nPE_mean_px', 'point_nPE_median_px',
        'box_ap50', 'pose_ap50', 'pose_ap', 'recall', 'precision',
        'nPE_median', 'kpt_dist_mean_px', 'PCK@5px', 'PCK@10px',
        'point_ap_10px', 'point_ap_20px', 'point_precision',
        'point_PCK5'])
    if log_mode == 'w':
        writer.writeheader()

    nb = len(train_loader)
    num_warmup = max(round(params['warmup_epochs'] * nb), 100)
    epochs_no_improve = max(0, start_epoch - best_epoch) if best_epoch else 0
    early_stop = False
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        if epoch < args.freeze_epochs:
            set_backbone_frozen(True)
        elif epoch == args.freeze_epochs:
            set_backbone_frozen(False)
            for g in optimizer.param_groups:
                g['lr'] = g['initial_lr'] * lr_fn(epoch)
        model.train()
        if epoch < args.freeze_epochs:
            model.net.eval()
            model.fpn.eval()
        # 与其他单帧模型一致: 最后 10 epoch 关闭 mosaic
        if args.epochs - epoch == 10:
            train_dset.mosaic = False

        m_loss = util.AverageMeter()
        m_hm = util.AverageMeter()
        m_off = util.AverageMeter()
        m_size = util.AverageMeter()
        micro = 0
        for i, (samples, targets) in enumerate(train_loader):
            x = i + nb * epoch
            samples = samples.to(device).float() / 255
            if x <= num_warmup:
                for g in optimizer.param_groups:
                    base_lr = g['initial_lr'] * lr_fn(epoch)
                    fp = [params['warmup_bias_lr'], base_lr] if g.get('is_bias') \
                        else [0.0, base_lr]
                    g['lr'] = np.interp(x, [0, num_warmup], fp)
                    if 'momentum' in g:
                        g['momentum'] = np.interp(
                            x, [0, num_warmup],
                            [params['warmup_momentum'], params['momentum']])
            with torch.cuda.amp.autocast():
                out = model(samples)
                loss, (lhm, loff, lsz), _ = point_loss(
                    out, targets, params,
                    weights=(args.loss_hm, args.loss_off, args.loss_size),
                    min_overlap=args.gauss_overlap, sigma_div=args.gauss_sigma_div,
                    size_iou_w=args.size_iou_w,
                    fixed_radius=args.gauss_fixed_radius or None,
                    skip_invisible=args.skip_invisible)
            m_loss.update(loss.item(), samples.size(0))
            m_hm.update(lhm, samples.size(0))
            m_off.update(loff, samples.size(0))
            m_size.update(lsz, samples.size(0))
            scaler.scale(loss / args.accumulate).backward()
            micro += 1
            if micro % args.accumulate == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
        # 处理 epoch 末尾不足一个 accumulate 的梯度
        if micro % args.accumulate != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)
        scheduler.step()

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            if args.no_size:
                point_metrics, preds = evaluate_points(
                    ema.ema, val_loader, args.input_size,
                    conf=args.conf, topk=args.topk, tau_px=10.0)
                metrics = None
            else:
                metrics, preds = evaluate_pad(ema.ema, val_loader, args.input_size,
                                              conf=args.conf, topk=args.topk)
                # 复用同一次前向的 preds, 避免重复推理
                point_metrics = match_points(preds, tau_px=10.0, conf=args.conf)
            row = {
                'epoch': epoch + 1, 'loss': f'{m_loss.avg:.3f}',
                'loss_hm': f'{m_hm.avg:.3f}', 'loss_off': f'{m_off.avg:.3f}',
                'loss_size': f'{m_size.avg:.3f}',
                'point_recall_20D': f'{point_metrics["point_recall"]:.4f}',
                'point_nPE_mean_px': f'{point_metrics["nPE_mean_px"]:.3f}',
                'point_nPE_median_px': f'{point_metrics["nPE_median_px"]:.3f}',
            }
            if metrics is not None:
                row.update({
                    'box_ap50': f'{metrics["box_ap50"]:.4f}',
                    'pose_ap50': f'{metrics["pose_ap50"]:.4f}',
                    'pose_ap': f'{metrics["pose_ap"]:.4f}',
                    'recall': f'{metrics["recall@0.5"]:.4f}',
                    'precision': f'{metrics["precision@0.5"]:.4f}',
                    'nPE_median': f'{metrics["nPE_median"]:.4f}',
                    'kpt_dist_mean_px': f'{metrics["kpt_dist_mean_px"]:.2f}',
                    'PCK@5px': f'{metrics["PCK@5px"]:.4f}',
                    'PCK@10px': f'{metrics["PCK@10px"]:.4f}',
                })
            else:
                row.update({
                    'point_ap_10px': f'{point_metrics["point_ap_10px"]:.4f}',
                    'point_ap_20px': f'{point_metrics["point_ap_20px"]:.4f}',
                    'point_precision': f'{point_metrics["point_precision"]:.4f}',
                    'point_PCK5': f'{point_metrics["PCK@5px"]:.4f}',
                })
            writer.writerow(row)
            log_f.flush()

            # ── 选点 ────────────────────────────────────────────
            if args.select_metric == 'point_error':
                err = float(point_metrics['nPE_mean_px'])
                rec = float(point_metrics['point_recall'])
                eligible = np.isfinite(err) and rec >= args.min_recall
                if eligible:
                    if not np.isfinite(best_err):
                        is_best = True
                    else:
                        is_best = (
                            err < best_err - args.min_delta_err
                            or (abs(err - best_err) <= args.min_delta_err
                                and rec > best_recall + 1e-6))
                else:
                    is_best = False
                if is_best:
                    best_err, best_recall = err, rec
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if metrics is not None:
                    best_pose = max(best_pose, float(metrics['pose_ap50']))
            else:
                best_score = (point_metrics['point_ap_10px'] if args.no_size
                              else metrics['pose_ap50'])
                if not np.isfinite(best_score):
                    best_score = -1.0
                is_best = best_score > best_pose + args.min_delta
                if is_best:
                    best_pose = best_score
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if args.no_size:
                    best_err, best_recall = point_metrics['nPE_mean_px'], point_metrics['point_recall']
            print(f'[E{epoch + 1}/{args.epochs}] loss {m_loss.avg:.2f} | '
                  f'R20D {point_metrics["point_recall"]:.4f} '
                  f'nPE {point_metrics["nPE_mean_px"]:.3f}px'
                  + (f' box50 {metrics["box_ap50"]:.3f} '
                     f'pose50 {metrics["pose_ap50"]:.3f}' if metrics else '')
                  + (' | BEST' if is_best else ''))
            ckpt = {
                'model': copy.deepcopy(ema.ema).half(),
                'epoch': epoch + 1,
                'head_cfg': model.head_cfg,
                'base_model': args.base_model,
                'best_pose_ap50': best_pose,
                'best_npe_mean_px': best_err if np.isfinite(best_err) else -1.0,
                'best_recall20D': best_recall,
                'best_metric': ('point_error' if args.select_metric == 'point_error'
                                else ('point_ap_10px' if args.no_size else 'pose_ap50')),
                'best_epoch': best_epoch,
                'val_metrics': metrics,
                'val_point_metrics': point_metrics,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'ema_state': {k: v.detach().cpu().clone()
                              for k, v in ema.ema.state_dict().items()},
                'ema_updates': ema.updates,
            }
            torch.save(ckpt, os.path.join(args.out_dir, 'last.pt'))
            if is_best:
                torch.save(ckpt, os.path.join(args.out_dir, 'best.pt'))
            if args.patience > 0 and (epoch + 1) >= args.min_epochs \
                    and epochs_no_improve >= args.patience:
                print(f'[early-stop] no improvement for {epochs_no_improve} evals '
                      f'(best E{best_epoch} err={best_err:.3f}px rec={best_recall:.4f})')
                early_stop = True
        else:
            print(f'[E{epoch + 1}/{args.epochs}] loss {m_loss.avg:.2f} (no eval)')
        if early_stop:
            break

    log_f.close()
    elapsed = time.time() - t0

    # point_error 选点若始终没达到 min_recall, 至少保证 best.pt 存在
    if not os.path.exists(os.path.join(args.out_dir, 'best.pt')) \
            and os.path.exists(os.path.join(args.out_dir, 'last.pt')):
        import shutil
        shutil.copy2(os.path.join(args.out_dir, 'last.pt'),
                     os.path.join(args.out_dir, 'best.pt'))
        print('[warn] no epoch met min_recall; best.pt = last.pt')

    # test: best + last
    for tag in ('best', 'last'):
        p = os.path.join(args.out_dir, f'{tag}.pt')
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location=device, weights_only=False)
        m = ck['model'].float().to(device)
        point_met, _ = evaluate_points(m, test_loader, args.input_size,
                                       conf=args.conf, topk=args.topk)
        if args.no_size:
            met = point_met
            print(f'TEST {tag}: pointAP10 {met["point_ap_10px"]:.4f} '
                  f'R20D {met["point_recall"]:.4f} P {met["point_precision"]:.4f} '
                  f'nPEmean {met["nPE_mean_px"]:.3f}px '
                  f'nPEmed {met["nPE_median_px"]:.2f}px PCK5 {met["PCK@5px"]:.4f}')
        else:
            met, _ = evaluate_pad(m, test_loader, args.input_size,
                                  conf=args.conf, topk=args.topk)
            print(f'TEST {tag}: box50 {met["box_ap50"]:.4f} pose50 {met["pose_ap50"]:.4f} '
                  f'R {met["recall@0.5"]:.4f} P {met["precision@0.5"]:.4f} '
                  f'nPE {met["nPE_median"]:.4f} PCK5 {met["PCK@5px"]:.4f} | '
                  f'pointAP10 {point_met["point_ap_10px"]:.4f} '
                  f'R20D {point_met["point_recall"]:.4f} '
                  f'point-nPEmean {point_met["nPE_mean_px"]:.3f}px '
                  f'point-nPEmed {point_met["nPE_median_px"]:.2f}px')
        ck['test_metrics'] = met
        ck['test_point_metrics'] = point_met
        torch.save(ck, p)
        del m
        torch.cuda.empty_cache()

    if args.select_metric == 'point_error':
        best_name = f'point_error(min_nPE, recall>={args.min_recall})'
        print(f'done in {elapsed / 60:.1f} min, best val nPE={best_err:.3f}px '
              f'recall20D={best_recall:.4f} @E{best_epoch}')
    else:
        best_name = 'point_ap_10px' if args.no_size else 'pose_ap50'
        print(f'done in {elapsed / 60:.1f} min, best val {best_name}={best_pose:.4f} '
              f'@E{best_epoch}')
    with open(os.path.join(args.out_dir, 'run_info.txt'), 'w') as f:
        f.write(f'base_model={args.base_model}\nhead_cfg={model.head_cfg}\n'
                f'data={args.data_dir}\npretrained={args.pretrained}\n'
                f'epochs={args.epochs} lr0={args.lr0} backbone_mult={args.backbone_mult}\n'
                f'best_metric={best_name} best_npe_mean_px={best_err:.4f} '
                f'best_recall20D={best_recall:.4f} best_pose_ap50={best_pose:.4f} '
                f'best_epoch={best_epoch}\n'
                f'elapsed_min={elapsed / 60:.1f}\n')


if __name__ == '__main__':
    main()
