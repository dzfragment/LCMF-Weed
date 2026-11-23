#!/usr/bin/env python3
"""多帧 PAD 训练 — 在 exp3 MultiFrameYOLO 融合框架上换用 PAD 点热图头.

流程与 exp3 的 train.py 对齐:
  stage 2: 冻结 backbone/FPN/PAD 头, 只训练对齐+融合模块;
  stage 3: 解冻 backbone/FPN/fusion (可选 --unfreeze-head), 端到端微调.

初始化:
  stage 2: --pretrained 指向单帧 PAD checkpoint (net./fpn./head. 完整加载);
  stage 3: --weights 指向 stage 2 的 best checkpoint.

用法:
  python train_pad_mf.py --stage 2 --epochs 20 --pretrained .../pad_sf/best.pt \
      --output-dir results/pad_mf --consistent-align --fft-downsample 4
  python train_pad_mf.py --stage 3 --epochs 50 --weights results/pad_mf/pad_stage2_best.pt \
      --output-dir results/pad_mf --consistent-align --fft-downsample 4 --unfreeze-head
"""
import argparse
import csv
import os
import random
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_WORKSPACE, 'YOLOv8-pose'))
sys.path.insert(0, _SCRIPT_DIR)

from models.mf_yolo import MultiFrameYOLO  # noqa: E402
from mf_dataset import MultiFrameDataset, create_dataloaders  # noqa: E402
from pad_core import (evaluate_pad, evaluate_points, match_points,  # noqa: E402
                      normalize_head_cfg, point_loss)
from train import AlignLoss, apply_ablation, apply_cascade, apply_iterative_fusion, \
    apply_shared_cascade, apply_two_frame, apply_dav2_single, apply_dav2_triple, \
    apply_shared_offset, apply_dcn_variant  # noqa: E402
from utils import util  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-test-eval', action='store_true')
    p.add_argument('--weights-source', choices=['model', 'ema'], default='model')
    p.add_argument('--stage', type=int, default=2, choices=[2, 3])
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--input-size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lrf', type=float, default=0.01)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--align-loss-weight', type=float, default=0.5)
    p.add_argument('--align-cos-weight', type=float, default=1.0)
    p.add_argument('--dataset-root',
                   default=os.path.join(os.getenv('LCMF_DATA_ROOT', os.path.join(_WORKSPACE, 'dataset')),
                                        'weed_sliding_merged'))
    p.add_argument('--pretrained', type=str, default=None,
                   help='stage 2 的单帧 PAD checkpoint')
    p.add_argument('--weights', type=str, default=None,
                   help='stage 3 从 stage 2 checkpoint 初始化 (仅模型权重)')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--output-dir', type=str, default=os.path.join(_SCRIPT_DIR, 'results', 'pad_mf'))
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--fusion-levels', type=str, default='p4')
    p.add_argument('--shared-offset-level', type=str, default='none',
                   choices=['none', 'p2', 'p3', 'p4'],
                   help='只在该融合层计算 DCN offset，其余层复用并缩放 offset')
    p.add_argument('--offset-reuse-mode', type=str, default='bilinear',
                   choices=['bilinear', 'nearest', 'global_mean', 'avg_pool'],
                   help='复用 offset 的重采样方式')
    p.add_argument('--base-model', type=str, default='p34l',
                   choices=['p34l', 'p2l', 'p2l_p4'])
    p.add_argument('--fusion-version', type=int, default=2, choices=[2, 3])
    p.add_argument('--fusion-gate-type', type=str, default='spatial')
    p.add_argument('--fusion-mode', type=str, default='gate_diff')
    p.add_argument('--convgru-kernel', type=int, default=3)
    p.add_argument('--convgru-layers', type=int, default=1)
    dav2 = p.add_mutually_exclusive_group()
    dav2.add_argument('--dav2-single', action='store_true',
                      help='单阶段 DAV2（旧 06 消融）')
    dav2.add_argument('--dav2-triple', action='store_true',
                      help='三阶段 DAV2（旧 07 消融）')
    p.add_argument('--dcn-variant', type=str, default=None,
                   help='共享 DCN 层的结构变体（见 models/dcn_variants.py；'
                        '可用 + 组合，如 coarse+dil2+p2refine）')
    p.add_argument('--no-align', action='store_true')
    p.add_argument('--cascade-align', action='store_true')
    p.add_argument('--iterative-fusion', action='store_true')
    p.add_argument('--two-frame', action='store_true')
    p.add_argument('--shared-cascade', action='store_true')
    p.add_argument('--consistent-align', action='store_true')
    p.add_argument('--fft-downsample', type=int, default=4)
    p.add_argument('--global-head', action='store_true',
                   help='在 FFT 之后加一个 2 维全局平移残差头')
    p.add_argument('--global-head-level', default='p3',
                   help='global shift head 读取的特征层，默认 p3')
    p.add_argument('--global-head-hidden', type=int, default=32)
    p.add_argument('--global-head-max-shift', type=float, default=32.0,
                   help='global head 预测的图像空间位移上限 (px)')
    p.add_argument('--global-head-lr-mult', type=float, default=5.0,
                   help='global head 参数的学习率倍率（stage2/3）')
    p.add_argument('--init-global-head', type=str, default=None,
                   help='从另一个 checkpoint 复制 global_head_r1/r2 权重')
    p.add_argument('--freeze-global-head', action='store_true',
                   help='整个训练过程中冻结 global shift head')
    p.add_argument('--frame-dropout', type=float, default=0.2)
    p.add_argument('--unfreeze-head', action='store_true')
    p.add_argument('--no-preload', action='store_true', default=True)
    p.add_argument('--seed', type=int, default=42)
    # PAD head
    p.add_argument('--head', default='ms', choices=['ms', 'simple'])
    p.add_argument('--head-mid', type=int, default=32)
    p.add_argument('--out-stride4', type=int, default=0)
    p.add_argument('--dw', type=int, default=0)
    p.add_argument('--use-p2', type=int, default=0)
    p.add_argument('--p2-mid', type=int, default=32)
    p.add_argument('--no-size', action='store_true',
                   help='去掉 size 分支, 只输出点; 评估用点距离匹配')
    # loss
    p.add_argument('--loss-hm', type=float, default=1.0)
    p.add_argument('--loss-off', type=float, default=1.0)
    p.add_argument('--loss-size', type=float, default=0.5)
    p.add_argument('--gauss-overlap', type=float, default=0.7)
    p.add_argument('--gauss-sigma-div', type=float, default=3.0)
    p.add_argument('--gauss-fixed-radius', type=int, default=0,
                   help='>0 时用固定高斯半径(cells), 不依赖框尺寸 (坏框数据集用)')
    p.add_argument('--skip-invisible', action='store_true',
                   help='跳过增广后 vis=0 的越界关键点')
    p.add_argument('--size-iou-w', type=float, default=0.0)
    # eval
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--topk', type=int, default=300)
    p.add_argument('--patience', type=int, default=100,
                   help='连续 N 个 eval 无提升则早停 (0=关闭)')
    p.add_argument('--min-delta', type=float, default=0.003,
                   help='提升超过该值才算改进')
    p.add_argument('--min-epochs', type=int, default=100,
                   help='至少训练到该 epoch 才允许早停')
    p.add_argument('--select-metric', default='pose_ap50',
                   choices=['pose_ap50', 'point_error'],
                   help='checkpoint 选择: pose_ap50=旧行为; '
                        'point_error=20px 直径召回 >= min-recall 下取平均点误差最小')
    p.add_argument('--min-recall', type=float, default=0.85)
    p.add_argument('--min-delta-err', type=float, default=0.005)
    return p.parse_args()


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def reconcile_resume_files(output_dir, stage, ckpt_state, last_path):
    """Recover from interruption between CSV/last/best atomic file writes."""
    import shutil
    epoch = int(ckpt_state['epoch'])
    log_path = os.path.join(output_dir, f'stage{stage}_log.csv')
    if os.path.exists(log_path):
        with open(log_path, newline='') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = [row for row in reader if int(row['epoch']) <= epoch]
        with open(log_path + '.tmp', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(log_path + '.tmp', log_path)
    best_epoch = int(ckpt_state.get('best_epoch') or 0)
    best_path = os.path.join(output_dir, f'pad_stage{stage}_best.pt')
    if best_epoch == epoch and best_epoch > 0:
        # last is committed first. If best was not committed, last is exactly
        # the selected raw model and can safely reconstruct the missing best.
        needs_repair = not os.path.exists(best_path)
        if not needs_repair:
            best = torch.load(best_path, map_location='cpu', weights_only=False)
            needs_repair = int(best['epoch']) != best_epoch
        if needs_repair:
            shutil.copy2(last_path, best_path + '.tmp')
            os.replace(best_path + '.tmp', best_path)
    elif best_epoch > 0:
        if not os.path.exists(best_path):
            raise RuntimeError(f'Missing selected best checkpoint E{best_epoch}: {best_path}')
        best = torch.load(best_path, map_location='cpu', weights_only=False)
        if int(best['epoch']) != best_epoch:
            raise RuntimeError(f'Best checkpoint epoch does not match last state: {best_path}')


def _validate_sf_checkpoint(path, base_model, head_cfg):
    """stage2 初始化前校验单帧 checkpoint 与 CLI 基座/头配置一致.

    以前只靠 load_pretrained 的 shape 匹配兜底, base_model 写错或
    size/use_p2 不一致时错误信息不直观. 这里提前给出明确报错.
    """
    ck = torch.load(path, map_location='cpu', weights_only=False)
    ck_base = ck.get('base_model')
    if ck_base is not None and ck_base != base_model:
        raise ValueError(
            f'单帧 checkpoint base_model={ck_base} 与 --base-model={base_model} 不一致: {path}')
    saved = ck.get('head_cfg')
    if saved:
        saved_n = normalize_head_cfg(saved)
        cli_n = normalize_head_cfg(head_cfg)
        diffs = {k: (saved_n[k], cli_n[k]) for k in cli_n
                 if saved_n.get(k) != cli_n[k]}
        if diffs:
            raise ValueError(
                f'单帧 checkpoint head_cfg 与 CLI 不一致: {diffs}\n  ckpt={path}')
    return ck


def _load_global_head_from_ckpt(model, path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('model', ck)
    keys = {k: v for k, v in sd.items() if k.startswith('global_head')}
    if not keys:
        raise ValueError(f'{path} 中没有 global_head 参数')
    missing, unexpected = model.load_state_dict(keys, strict=False)
    print(f'[global-head] loaded {len(keys)} tensors from {path}')
    return missing, unexpected


def build_model(args, device, pretrained_path=None):
    head_cfg = {
        'head_type': args.head, 'mid': args.head_mid,
        'out_stride4': bool(args.out_stride4), 'dw': bool(args.dw),
        'use_p2': bool(args.use_p2), 'p2_mid': args.p2_mid,
        'use_size': not args.no_size,
    }
    if pretrained_path is not None:
        _validate_sf_checkpoint(pretrained_path, args.base_model, head_cfg)
    fusion_levels = args.fusion_levels.split(',')
    model = MultiFrameYOLO(
        num_classes=1, kpt_shape=(1, 2),
        fusion_levels=fusion_levels,
        pretrained_path=pretrained_path,
        fusion_version=args.fusion_version,
        fusion_gate_type=args.fusion_gate_type,
        convgru_kernel=args.convgru_kernel,
        convgru_layers=args.convgru_layers,
        global_align_stage='feature' if args.consistent_align else 'none',
        fft_downsample=args.fft_downsample,
        consistent_align=args.consistent_align,
        base_model=args.base_model,
        pad_head=True, pad_head_cfg=head_cfg,
        offset_reuse_mode=args.offset_reuse_mode,
        global_head=args.global_head,
        global_head_level=args.global_head_level,
        global_head_hidden=args.global_head_hidden,
        global_head_max_shift=args.global_head_max_shift,
    ).to(device)
    apply_ablation(model, no_align=args.no_align, fusion_mode=args.fusion_mode)
    if args.shared_offset_level != 'none':
        if args.no_align:
            raise ValueError('--shared-offset-level 不能与 --no-align 同时使用')
        apply_shared_offset(model, args.shared_offset_level,
                            offset_reuse_mode=args.offset_reuse_mode)
    if args.cascade_align:
        apply_cascade(model)
    if args.iterative_fusion:
        apply_iterative_fusion(model)
    if args.two_frame:
        apply_two_frame(model)
    if args.shared_cascade:
        apply_shared_cascade(model)
    if args.dav2_single:
        apply_dav2_single(model)
    elif args.dav2_triple:
        apply_dav2_triple(model)
    model.dav2_single = bool(args.dav2_single)
    model.dav2_triple = bool(args.dav2_triple)
    apply_dcn_variant(model, getattr(args, 'dcn_variant', None) or 'v2')
    model.convgru_kernel = args.convgru_kernel
    model.convgru_layers = args.convgru_layers
    if getattr(args, 'init_global_head', None):
        _load_global_head_from_ckpt(model, args.init_global_head)
    return model


def save_ckpt(model, optimizer, scheduler, scaler, ema, epoch, best_pose, path,
              stage=None, best_epoch=None, best_err=None, best_recall=None,
              select_metric=None):
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'best_pose_ap50': best_pose,
        'best_npe_mean_px': best_err,
        'best_recall20D': best_recall,
        'best_metric': select_metric,
        'best_epoch': best_epoch,
        'stage': stage,
        'ema_updates': (ema.updates if ema is not None else 0),
        'rng_state': capture_rng_state(),
        'fusion_version': model.fusion_version,
        'fusion_gate_type': model.fusion_gate_type,
        'convgru_kernel': getattr(model, 'convgru_kernel', 3),
        'convgru_layers': getattr(model, 'convgru_layers', 1),
        'dav2_single': getattr(model, 'dav2_single', False),
        'dav2_triple': getattr(model, 'dav2_triple', False),
        'dcn_variant': getattr(model, 'dcn_variant', 'v2'),
        'consistent_align': model.consistent_align,
        'no_align': bool(getattr(model.fusion, 'no_align', False)),
        'global_head': bool(getattr(model, 'global_head_enabled', False)),
        'global_head_level': getattr(model, 'global_head_level', 'p3'),
        'global_head_hidden': getattr(model, 'global_head_hidden', 32),
        'global_head_max_shift': getattr(model, 'global_head_max_shift', 32.0),
        'fft_downsample': model.fft_downsample,
        'base_model': model.base_model,
        'fusion_levels': list(model.fusion_levels),
        'shared_offset_level': getattr(model.fusion, 'shared_offset_level', None),
        'offset_reuse_mode': getattr(model.fusion, 'offset_reuse_mode', 'bilinear'),
        'feature_levels': list(model.feature_levels),
        'detect_levels': list(model.detect_levels),
        'pad_head': True,
        'head_cfg': model.pad_head_cfg,
    }
    if ema is not None:
        ckpt['ema_model'] = {k: v.half() for k, v in ema.ema.state_dict().items()}
    temp_path = str(path) + '.tmp'
    with open(temp_path, 'wb') as f:
        torch.save(ckpt, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)
    directory_fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_ckpt(path, args, device, prefer_ema=False):
    ck = torch.load(path, map_location=device, weights_only=False)
    if not ck.get('pad_head', False):
        raise ValueError(f'{path} is not a PAD checkpoint')
    saved_cfg = dict(ck.get('head_cfg') or {})
    # CLI 的 head 配置优先（允许显式覆盖），否则用 checkpoint 内配置
    cli_cfg = {
        'head_type': args.head, 'mid': args.head_mid,
        'out_stride4': bool(args.out_stride4), 'dw': bool(args.dw),
        'use_p2': bool(args.use_p2), 'p2_mid': args.p2_mid,
        'use_size': not args.no_size,
    }
    for k, v in list(saved_cfg.items()):
        if k in cli_cfg and cli_cfg[k] != v:
            print(f'[warn] head_cfg {k}: ckpt={v} cli={cli_cfg[k]}, use ckpt')
    cfg = normalize_head_cfg(saved_cfg or cli_cfg)
    fusion_levels = ck.get('fusion_levels', args.fusion_levels.split(','))
    ck_mode = ck.get('offset_reuse_mode', args.offset_reuse_mode)
    if ck_mode != args.offset_reuse_mode:
        raise ValueError(
            f'checkpoint offset_reuse_mode={ck_mode!r} 与 CLI '
            f'--offset-reuse-mode={args.offset_reuse_mode!r} 不一致')
    model = MultiFrameYOLO(
        num_classes=1, kpt_shape=(1, 2),
        fusion_levels=fusion_levels,
        pretrained_path=None,
        fusion_version=ck.get('fusion_version', args.fusion_version),
        fusion_gate_type=ck.get('fusion_gate_type', args.fusion_gate_type),
        convgru_kernel=ck.get('convgru_kernel', args.convgru_kernel),
        convgru_layers=ck.get('convgru_layers', args.convgru_layers),
        global_align_stage='feature' if ck.get('consistent_align', args.consistent_align) else 'none',
        fft_downsample=ck.get('fft_downsample', args.fft_downsample),
        consistent_align=bool(ck.get('consistent_align', args.consistent_align)),
        base_model=ck.get('base_model', args.base_model),
        pad_head=True, pad_head_cfg=cfg,
        offset_reuse_mode=args.offset_reuse_mode,
        global_head=bool(ck.get('global_head',
                                getattr(args, 'global_head', False))),
        global_head_level=ck.get('global_head_level',
                                 getattr(args, 'global_head_level', 'p3')),
        global_head_hidden=ck.get('global_head_hidden',
                                  getattr(args, 'global_head_hidden', 32)),
        global_head_max_shift=ck.get('global_head_max_shift',
                                     getattr(args, 'global_head_max_shift', 32.0)),
    ).to(device)
    apply_ablation(model, no_align=args.no_align, fusion_mode=args.fusion_mode)
    ck_shared = ck.get('shared_offset_level', args.shared_offset_level)
    if ck_shared in (None, ''):
        ck_shared = 'none'
    if ck_shared != args.shared_offset_level:
        raise ValueError(
            f'checkpoint shared_offset_level={ck_shared!r} 与 CLI '
            f'--shared-offset-level={args.shared_offset_level!r} 不一致')
    if ck_shared != 'none':
        apply_shared_offset(model, ck_shared, offset_reuse_mode=args.offset_reuse_mode)
    if args.cascade_align:
        apply_cascade(model)
    if args.iterative_fusion:
        apply_iterative_fusion(model)
    if args.two_frame:
        apply_two_frame(model)
    if args.shared_cascade:
        apply_shared_cascade(model)
    dav2_single = bool(ck.get('dav2_single', args.dav2_single))
    dav2_triple = bool(ck.get('dav2_triple', args.dav2_triple))
    if dav2_single:
        apply_dav2_single(model)
    elif dav2_triple:
        apply_dav2_triple(model)
    model.dav2_single = dav2_single
    model.dav2_triple = dav2_triple
    # DCN structural variant: CLI request wins; otherwise use the checkpoint's.
    ck_variant = ck.get('dcn_variant', 'v2') or 'v2'
    target_variant = getattr(args, 'dcn_variant', None) or ck_variant
    apply_dcn_variant(model, target_variant)
    model.dcn_variant = target_variant
    model.convgru_kernel = ck.get('convgru_kernel', args.convgru_kernel)
    model.convgru_layers = ck.get('convgru_layers', args.convgru_layers)
    sd = ck.get('ema_model') if (prefer_ema and ck.get('ema_model') is not None) \
        else ck['model']
    if target_variant == ck_variant:
        model.load_state_dict(sd, strict=True)
    else:
        from models.dcn_variants import migrate_from_v2, load_state_dict_shape_filtered
        migrate_from_v2(model, sd)
        load_state_dict_shape_filtered(model, sd)
    state = {
        'optimizer': ck.get('optimizer'), 'scheduler': ck.get('scheduler'),
        'scaler': ck.get('scaler'), 'epoch': ck.get('epoch', 0),
        'best_pose_ap50': ck.get('best_pose_ap50', -1.0),
        'best_npe_mean_px': ck.get('best_npe_mean_px', float('inf')),
        'best_recall20D': ck.get('best_recall20D', -1.0),
        'best_epoch': ck.get('best_epoch', 0),
        'stage': ck.get('stage'), 'ema_model': ck.get('ema_model'),
        'ema_updates': ck.get('ema_updates', 0),
        'rng_state': ck.get('rng_state'),
    }
    return model, state


def build_test_loader(args):
    dset = MultiFrameDataset(
        args.dataset_root, split='test', input_size=args.input_size,
        params=None, augment=False)
    loader = torch.utils.data.DataLoader(
        dset, args.batch_size, False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=MultiFrameDataset.collate_fn)
    return loader


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, stage: {args.stage}')

    ckpt_state = None
    if args.resume:
        print(f'Resume from {args.resume}')
        model, ckpt_state = load_ckpt(args.resume, args, device)
        if ckpt_state['stage'] not in (None, args.stage):
            raise ValueError(
                f'resume stage={ckpt_state["stage"]} 与 --stage={args.stage} 不一致')
        start_epoch = ckpt_state['epoch']
        best_pose = ckpt_state['best_pose_ap50']
        best_err = ckpt_state.get('best_npe_mean_px', float('inf'))
        best_recall = ckpt_state.get('best_recall20D', -1.0)
    elif args.weights:
        print(f'Load stage2 weights from {args.weights}')
        model, _ = load_ckpt(args.weights, args, device,
                             prefer_ema=(args.weights_source == 'ema'))
        start_epoch = 0
        best_pose = -1.0
        best_err = float('inf')
        best_recall = -1.0
    else:
        if not args.pretrained:
            raise ValueError('stage2 需要 --pretrained 单帧 PAD checkpoint')
        print(f'Init from single-frame PAD: {args.pretrained}')
        model = build_model(args, device, pretrained_path=args.pretrained)
        start_epoch = 0
        best_pose = -1.0
        best_err = float('inf')
        best_recall = -1.0

    best_epoch = int(ckpt_state.get('best_epoch', 0)) if ckpt_state else 0
    print(f'head cfg: {model.pad_head_cfg}, point stride: {model.stride}, '
          f'params: {sum(p.numel() for p in model.parameters()):,}')

    # 阶段冻结
    if args.stage == 2:
        model.freeze_backbone()
        for p in model.fpn.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(False)
        print('Stage 2: backbone/FPN/PAD head frozen, fusion trainable')
    else:
        model.unfreeze_all()
        if not args.unfreeze_head:
            for p in model.head.parameters():
                p.requires_grad_(False)
            print('Stage 3: PAD head frozen')
        else:
            print('Stage 3: PAD head unfrozen')

    if getattr(args, 'freeze_global_head', False):
        if getattr(model, 'global_head_r1', None) is None:
            raise ValueError('--freeze-global-head 但模型没有 global head')
        for p in model.global_head_r1.parameters():
            p.requires_grad_(False)
        for p in model.global_head_r2.parameters():
            p.requires_grad_(False)
        print('global shift head frozen')

    # Keep the data RNG independent of architecture initialization.
    setup_seed(args.seed)
    train_loader, val_loader = create_dataloaders(
        args.dataset_root, input_size=args.input_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        no_preload=args.no_preload,
    )

    # optimizer groups (同 exp3: backbone 0.1x, fusion 1x, other 0.5x)
    # global head 零初始化且输出只有 2 维，给它单独一个更高的 lr。
    gh_params = [p for n, p in model.named_parameters()
                 if n.startswith('global_head') and p.requires_grad]
    gh_lr = args.lr * args.global_head_lr_mult
    if args.stage == 2:
        if gh_params:
            params = [p for n, p in model.named_parameters()
                      if p.requires_grad and not n.startswith('global_head')]
            groups = [{'params': params, 'lr': args.lr},
                      {'params': gh_params, 'lr': gh_lr}]
            optimizer = torch.optim.SGD(groups, lr=args.lr, momentum=0.937,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
        else:
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.937,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
    else:
        bb, fus, oth = [], [], []
        gh_set = {id(p) for p in gh_params}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in gh_set:
                continue
            if n.startswith('backbone'):
                bb.append(p)
            elif n.startswith('fusion'):
                fus.append(p)
            else:
                oth.append(p)
        groups = []
        if bb:
            groups.append({'params': bb, 'lr': args.lr * 0.1})
        if fus:
            groups.append({'params': fus, 'lr': args.lr})
        if oth:
            groups.append({'params': oth, 'lr': args.lr * 0.5})
        if gh_params:
            groups.append({'params': gh_params,
                           'lr': args.lr * 0.5 * args.global_head_lr_mult})
        optimizer = torch.optim.SGD(groups, lr=args.lr, momentum=0.937,
                                    weight_decay=args.weight_decay, nesterov=True)
    for g in optimizer.param_groups:
        g['initial_lr'] = g['lr']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * args.lrf)
    ema = util.EMA(model)
    scaler = torch.cuda.amp.GradScaler()

    if ckpt_state is not None:
        if ckpt_state.get('ema_model') is not None:
            ema.ema.load_state_dict(
                {k: v.float() for k, v in ckpt_state['ema_model'].items()})
            ema.updates = int(ckpt_state.get('ema_updates', 0))
            print(f'  EMA state restored (updates={ema.updates})')
        if ckpt_state['optimizer'] is not None:
            optimizer.load_state_dict(ckpt_state['optimizer'])
            print('  optimizer state restored')
        if ckpt_state['scheduler'] is not None:
            scheduler.load_state_dict(ckpt_state['scheduler'])
        if ckpt_state['scaler'] is not None:
            scaler.load_state_dict(ckpt_state['scaler'])

    align_loss_fn = AlignLoss(model)
    nb = len(train_loader)
    nw = max(round(3.0 * nb), 100)
    if ckpt_state is not None:
        if ckpt_state.get('rng_state') is None:
            raise RuntimeError('Resume checkpoint lacks RNG state; cannot reproduce the data/augmentation stream')
        reconcile_resume_files(args.output_dir, args.stage, ckpt_state, args.resume)
        # Restore AFTER model/EMA/loader construction, before the next iterator.
        restore_rng_state(ckpt_state['rng_state'])
        print('  Python/NumPy/Torch/CUDA RNG state restored')
    log_mode = 'a' if args.resume else 'w'
    log_f = open(os.path.join(args.output_dir, f'stage{args.stage}_log.csv'),
                 log_mode, newline='')
    writer = csv.DictWriter(log_f, fieldnames=[
        'epoch', 'loss', 'det_loss', 'align_loss', 'loss_hm', 'loss_off', 'loss_size',
        'point_recall_20D', 'point_nPE_mean_px', 'point_nPE_median_px',
        'box_ap50', 'pose_ap50', 'pose_ap', 'recall', 'precision',
        'nPE_median', 'kpt_dist_mean_px', 'PCK@5px', 'PCK@10px',
        'point_ap_10px', 'point_ap_20px', 'point_precision', 'point_PCK5'])
    if log_mode == 'w':
        writer.writeheader()

    epochs_no_improve = max(0, start_epoch - best_epoch) if best_epoch else 0
    early_stop = False
    completed_epochs = start_epoch
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        completed_epochs = epoch + 1
        model.train()
        if args.stage == 2:
            model.backbone.eval()
            model.fpn.eval()
            model.head.eval()
        mloss = util.AverageMeter()
        mdet = util.AverageMeter()
        malign = util.AverageMeter()
        mhm = util.AverageMeter()
        moff = util.AverageMeter()
        msz = util.AverageMeter()

        import tqdm
        pbar = tqdm.tqdm(enumerate(train_loader), total=nb,
                         desc=f'E{epoch + 1}/{args.epochs}')
        for i, (frames, targets) in pbar:
            x = i + nb * epoch
            frames = frames.to(device).float() / 255.0

            if args.stage == 3 and args.frame_dropout > 0:
                mask = torch.rand(frames.shape[0], device=frames.device) < args.frame_dropout
                if mask.any():
                    frames[mask, 1] = frames[mask, 0]
                    frames[mask, 2] = frames[mask, 0]

            if x <= nw:
                for g in optimizer.param_groups:
                    g['lr'] = np.interp(x, [0, nw], [0.0, g['initial_lr']])

            with torch.cuda.amp.autocast():
                outputs = model(frames)
                det_loss, (lhm, loff, lsz), _ = point_loss(
                    outputs, targets,
                    weights=(args.loss_hm, args.loss_off, args.loss_size),
                    min_overlap=args.gauss_overlap, sigma_div=args.gauss_sigma_div,
                    size_iou_w=args.size_iou_w,
                    fixed_radius=args.gauss_fixed_radius or None,
                    skip_invisible=args.skip_invisible)
                if args.align_loss_weight != 0:
                    align_l1, align_cos = align_loss_fn.compute()
                    align_loss = align_l1 + args.align_cos_weight * align_cos
                else:
                    # A disabled auxiliary objective must not introduce NaN via 0*loss.
                    align_loss = det_loss.new_zeros(())
                loss = det_loss + args.align_loss_weight * align_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f'Non-finite loss at stage={args.stage}, epoch={epoch+1}, batch={i}: '
                    f'detection={float(det_loss)}, alignment={float(align_loss)}')

            mloss.update(loss.item(), frames.size(0))
            mdet.update(det_loss.item(), frames.size(0))
            malign.update(float(align_loss), frames.size(0))
            mhm.update(lhm, frames.size(0))
            moff.update(loff, frames.size(0))
            msz.update(lsz, frames.size(0))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)
            pbar.set_description(
                f'E{epoch + 1}/{args.epochs} L={mloss.avg:.2f} D={mdet.avg:.2f} '
                f'A={malign.avg:.3f}')

        scheduler.step()

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            # exp3 经验: 用 model 而非 EMA (EMA BN stats 滞后)
            if args.no_size:
                point_metrics, _ = evaluate_points(
                    model, val_loader, args.input_size,
                    conf=args.conf, topk=args.topk, tau_px=10.0)
                metrics = None
            else:
                metrics, preds = evaluate_pad(model, val_loader, args.input_size,
                                              conf=args.conf, topk=args.topk)
                point_metrics = match_points(preds, tau_px=10.0, conf=args.conf)
            row = {
                'epoch': epoch + 1, 'loss': f'{mloss.avg:.3f}',
                'det_loss': f'{mdet.avg:.3f}', 'align_loss': f'{malign.avg:.4f}',
                'loss_hm': f'{mhm.avg:.3f}', 'loss_off': f'{moff.avg:.3f}',
                'loss_size': f'{msz.avg:.3f}',
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
            os.fsync(log_f.fileno())
            print(f'[E{epoch + 1}] train L={mloss.avg:.2f} D={mdet.avg:.2f} | '
                  f'val R20D {point_metrics["point_recall"]:.4f} '
                  f'nPE {point_metrics["nPE_mean_px"]:.3f}px'
                  + (f' box50 {metrics["box_ap50"]:.3f} '
                     f'pose50 {metrics["pose_ap50"]:.3f}' if metrics else ''))

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

            sel_metric = ('point_error' if args.select_metric == 'point_error'
                          else ('point_ap_10px' if args.no_size else 'pose_ap50'))
            # Commit last first. Resume can reconstruct an uncommitted best
            # from last when this epoch was selected, never the reverse.
            save_ckpt(model, optimizer, scheduler, scaler, ema, epoch + 1,
                      best_pose,
                      os.path.join(args.output_dir, f'pad_stage{args.stage}_last.pt'),
                      stage=args.stage, best_epoch=best_epoch,
                      best_err=best_err, best_recall=best_recall,
                      select_metric=sel_metric)
            if is_best:
                save_ckpt(model, optimizer, scheduler, scaler, ema, epoch + 1,
                          best_pose,
                          os.path.join(args.output_dir, f'pad_stage{args.stage}_best.pt'),
                          stage=args.stage, best_epoch=best_epoch,
                          best_err=best_err, best_recall=best_recall,
                          select_metric=sel_metric)
            if args.patience > 0 and (epoch + 1) >= args.min_epochs \
                    and epochs_no_improve >= args.patience:
                print(f'[early-stop] no improvement for {epochs_no_improve} evals '
                      f'(best E{best_epoch} err={best_err:.3f}px '
                      f'rec={best_recall:.4f})')
                early_stop = True
        else:
            print(f'[E{epoch + 1}] train L={mloss.avg:.2f} D={mdet.avg:.2f} (no eval)')
        if early_stop:
            break

    log_f.close()
    elapsed = time.time() - t0
    if not os.path.exists(os.path.join(args.output_dir, f'pad_stage{args.stage}_best.pt')) \
            and os.path.exists(os.path.join(args.output_dir, f'pad_stage{args.stage}_last.pt')):
        import shutil
        shutil.copy2(os.path.join(args.output_dir, f'pad_stage{args.stage}_last.pt'),
                     os.path.join(args.output_dir, f'pad_stage{args.stage}_best.pt'))
        print('[warn] no epoch met min_recall; best = last')
    if args.select_metric == 'point_error':
        best_name = f'point_error(min_nPE, recall>={args.min_recall})'
        print(f'\nStage {args.stage} done in {elapsed / 60:.1f} min, '
              f'best val nPE={best_err:.3f}px recall20D={best_recall:.4f}')
    else:
        best_name = 'point_ap_10px' if args.no_size else 'pose_ap50'
        print(f'\nStage {args.stage} done in {elapsed / 60:.1f} min, '
              f'best val {best_name}={best_pose:.4f}')

    # test 评估 (best + last)
    test_loader = None if args.skip_test_eval else build_test_loader(args)
    results = {}
    for tag in (() if args.skip_test_eval else ('best', 'last')):
        p = os.path.join(args.output_dir, f'pad_stage{args.stage}_{tag}.pt')
        if not os.path.exists(p):
            continue
        m, _ = load_ckpt(p, args, device)
        point_met, _ = evaluate_points(m, test_loader, args.input_size,
                                       conf=args.conf, topk=args.topk)
        if args.no_size:
            met = None
            print(f'TEST stage{args.stage} {tag}: pointAP10 '
                  f'{point_met["point_ap_10px"]:.4f} '
                  f'R {point_met["point_recall"]:.4f} '
                  f'P {point_met["point_precision"]:.4f} '
                  f'nPE {point_met["nPE_median_px"]:.2f}px '
                  f'PCK5 {point_met["PCK@5px"]:.4f}')
        else:
            met, _ = evaluate_pad(m, test_loader, args.input_size,
                                  conf=args.conf, topk=args.topk)
            print(f'TEST stage{args.stage} {tag}: box50 {met["box_ap50"]:.4f} '
                  f'pose50 {met["pose_ap50"]:.4f} R {met["recall@0.5"]:.4f} '
                  f'P {met["precision@0.5"]:.4f} nPE {met["nPE_median"]:.4f} '
                  f'PCK5 {met["PCK@5px"]:.4f} | '
                  f'pointAP10 {point_met["point_ap_10px"]:.4f} '
                  f'R20D {point_met["point_recall"]:.4f} '
                  f'point-nPEmean {point_met["nPE_mean_px"]:.3f}px '
                  f'point-nPEmed {point_met["nPE_median_px"]:.2f}px')
        results[tag] = {'box_metrics': met, 'point_metrics': point_met}
        del m
        torch.cuda.empty_cache()
    with open(os.path.join(args.output_dir, f'stage{args.stage}_results.json'), 'w') as f:
        import json
        json.dump({'stage': args.stage,
                   'best_metric': ('point_error' if args.select_metric == 'point_error'
                                   else ('point_ap_10px' if args.no_size else 'pose_ap50')),
                   'best_val': best_pose,
                   'best_npe_mean_px': best_err,
                   'best_recall20D': best_recall,
                   'test': results,
                   'epochs': args.epochs, 'epochs_completed': completed_epochs,
                   'seed': args.seed, 'align_loss_weight': args.align_loss_weight,
                   'align_cos_weight': args.align_cos_weight, 'lr': args.lr,
                   'head_cfg': model.pad_head_cfg,
                   'fusion_levels': list(model.fusion_levels),
                   'shared_offset_level': getattr(model.fusion, 'shared_offset_level', None),
                   'offset_reuse_mode': getattr(model.fusion, 'offset_reuse_mode', 'bilinear'),
                   'data': args.dataset_root,
                   'elapsed_min': round(elapsed / 60, 1)}, f, indent=1)


if __name__ == '__main__':
    train(get_args())
