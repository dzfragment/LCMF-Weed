"""
多帧融合检测训练脚本 v3

分阶段训练:
  阶段2: 预训练对齐+融合（冻结 backbone/FPN/Head）
  阶段3: backbone + fusion + FPN 联合微调（Head 保持冻结，backbone 小学习率）

用法:
  python train.py --stage 2 --epochs 80 --batch-size 4 --input-size 512
"""
import argparse
import csv
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_WORKSPACE, 'YOLOv8-pose'))
sys.path.insert(0, _SCRIPT_DIR)

from models.mf_yolo import MultiFrameYOLO
from models.modules import LEVEL_TO_IDX
from mf_dataset import create_dataloaders


def apply_ablation(model, no_align=False, fusion_mode='gate_diff'):
    """在模型创建后应用消融设置：砍对齐 / 改融合模式。"""
    import torch.nn as nn
    from models.modules import _NoOp
    fusion = model.fusion

    # 1. 砍对齐
    if no_align:
        fusion.no_align = True
        for l in fusion.fusion_levels:
            fusion.align_ref1[l] = _NoOp()
            fusion.align_ref2[l] = _NoOp()

    # 2. 改融合模式
    if fusion_mode == 'gate_nodiff':
        for l in fusion.fusion_levels:
            fmod = fusion.fusion[l]
            if hasattr(fmod, 'use_diff') and fmod.use_diff:
                # 重建 gate conv 为无 diff 版本
                ch = fmod.gate_r1_conv1.in_channels // 3  # 原来 3C → 现在 2C
                fmod.use_diff = False
                fmod.gate_r1_conv1 = nn.Conv2d(ch * 2, fmod.gate_r1_conv1.out_channels, 3, 1, 1).to(fmod.gate_r1_conv1.weight.device)
                fmod.gate_r1_conv2 = nn.Conv2d(fmod.gate_r1_conv2.in_channels, 1, 3, 1, 1).to(fmod.gate_r1_conv2.weight.device)
                fmod.gate_r2_conv1 = nn.Conv2d(ch * 2, fmod.gate_r2_conv1.out_channels, 3, 1, 1).to(fmod.gate_r2_conv1.weight.device)
                fmod.gate_r2_conv2 = nn.Conv2d(fmod.gate_r2_conv2.in_channels, 1, 3, 1, 1).to(fmod.gate_r2_conv2.weight.device)
                nn.init.kaiming_normal_(fmod.gate_r1_conv1.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(fmod.gate_r1_conv1.bias)
                nn.init.zeros_(fmod.gate_r1_conv2.weight)
                nn.init.constant_(fmod.gate_r1_conv2.bias, 0)
                nn.init.kaiming_normal_(fmod.gate_r2_conv1.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(fmod.gate_r2_conv1.bias)
                nn.init.zeros_(fmod.gate_r2_conv2.weight)
                nn.init.constant_(fmod.gate_r2_conv2.bias, 0)
    elif fusion_mode == 'avg':
        for l in fusion.fusion_levels:
            fmod = fusion.fusion[l]
            if hasattr(fmod, 'fusion_mode'):
                fmod.fusion_mode = 'avg'
                fmod.gate_r1_conv1 = None
                fmod.gate_r1_conv2 = None
                fmod.gate_r2_conv1 = None
                fmod.gate_r2_conv2 = None

    return model


def apply_shared_offset(model, level, offset_reuse_mode=None):
    """跨层复用 DCN offset：只在 level 层训练 offset predictor，
    其余融合层用 OffsetReuseAlign（一次 DCN + 复用/缩放 offset）。

    必须在加载 checkpoint 权重之前调用（会替换 align 模块）。
    """
    from models.modules import OffsetReuseAlign
    if level in (None, '', 'none'):
        return model
    fusion = model.fusion
    if getattr(fusion, 'fusion_version', 2) != 2:
        raise ValueError('shared offset 仅支持 fusion_version=2')
    if level not in fusion.fusion_levels:
        raise ValueError(
            f'shared offset level={level!r} 不在 fusion_levels={fusion.fusion_levels}')
    if offset_reuse_mode is None:
        offset_reuse_mode = getattr(fusion, 'offset_reuse_mode', 'bilinear')
    fusion.shared_offset_level = level
    fusion.offset_reuse_mode = offset_reuse_mode
    device = next(model.parameters()).device
    for l in fusion.fusion_levels:
        if l == level:
            continue
        ch = fusion.fusion_channels[l]
        fusion.align_ref1[l] = OffsetReuseAlign(
            ch, reuse_mode=offset_reuse_mode).to(device)
        fusion.align_ref2[l] = OffsetReuseAlign(
            ch, reuse_mode=offset_reuse_mode).to(device)
    return model


def apply_cascade(model):
    """级联对齐: ref2 先对齐 ref1，ref1 再对齐 key。"""
    fusion = model.fusion

    def cascade_forward(key_feats, ref1_feats, ref2_feats):
        fused = list(key_feats)
        for level in fusion.fusion_levels:
            idx = fusion.level_to_idx[level]
            key = key_feats[idx]
            ref1 = ref1_feats[idx]
            ref2 = ref2_feats[idx]
            aligned1 = fusion.align_ref1[level](ref1, key)
            aligned2 = fusion.align_ref2[level](ref2, aligned1)
            fusion._last_aligned_r1[level] = aligned1
            fusion._last_aligned_r2[level] = aligned2
            fused[idx] = fusion.fusion[level](key, aligned1, aligned2)
        return tuple(fused)

    fusion.forward = cascade_forward
    fusion.cascade_align = True
    return model


def apply_iterative_fusion(model):
    """
    迭代融合: Step1 ref2→ref1→ref1*, Step2 ref1*→key→output。
    两次对齐和融合共用同一组 DA 和 gate。
    """
    fusion = model.fusion

    def iter_forward(key_feats, ref1_feats, ref2_feats):
        fused = list(key_feats)
        for level in fusion.fusion_levels:
            idx = fusion.level_to_idx[level]
            key = key_feats[idx]; ref1 = ref1_feats[idx]; ref2 = ref2_feats[idx]

            # Step 1: ref2 → ref1，增强 ref1
            a_r2_to_r1 = fusion.align_ref2[level](ref2, ref1)
            ref1_star = fusion.fusion[level](ref1, ref1, a_r2_to_r1)

            # Step 2: ref1* → key，最终融合
            a_r1_to_key = fusion.align_ref1[level](ref1_star, key)
            out = fusion.fusion[level](key, a_r1_to_key, ref1_star)
            fused[idx] = out

            # 注意: 缓存对齐后特征仅供 L1 loss 使用
            # r2 对齐到 ref1（非 key），cosine loss 会通过 recompute 路径正确计算
            fusion._last_aligned_r1[level] = a_r1_to_key
            # 不缓存 r2：让 cosine loss 回退到 ref2→key 的标准对齐
        return tuple(fused)

    fusion.forward = iter_forward
    return model


def apply_shared_cascade(model):
    """
    共用 DA/gate 的级联:
    Step1: align_ref1(ref2, ref1) → fusion → ref1*
    Step2: align_ref1(ref1*, key) → fusion → output
    """
    fusion = model.fusion
    def fwd(key_feats, ref1_feats, ref2_feats):
        fused = list(key_feats)
        for level in fusion.fusion_levels:
            idx = fusion.level_to_idx[level]
            key = key_feats[idx]; ref1 = ref1_feats[idx]; ref2 = ref2_feats[idx]
            a2 = fusion.align_ref1[level](ref2, ref1)
            # Step1: 纯 gate 加权，不用 proj，避免特征膨胀
            r1s = ref1 + (fusion.fusion[level](ref1, ref1, a2) - ref1)
            a1 = fusion.align_ref1[level](r1s, key)
            out = fusion.fusion[level](key, a1, r1s)
            fused[idx] = out
            fusion._last_aligned_r1[level] = a1
        return tuple(fused)
    fusion.forward = fwd
    return model


def apply_image_align(model):
    """图像层全局平移对齐: ref→key 在 RGB 层对齐后再喂 backbone。"""
    from models.image_align import ImageAlign
    model._img_align_r1 = ImageAlign()
    model._img_align_r2 = ImageAlign()
    orig_fwd = model.forward

    def aligned_forward(frames):
        key = frames[:, 0]; ref1 = frames[:, 1]; ref2 = frames[:, 2]
        ref1_a = model._img_align_r1(ref1, key)
        ref2_a = model._img_align_r2(ref2, key)
        return orig_fwd(torch.stack([key, ref1_a, ref2_a], dim=1))

    model.forward = aligned_forward
    return model


def apply_two_frame(model):
    """仅用 key+ref1 两帧：ref2 路径被替换为 ref1 副本（贡献为 0）。"""
    fusion = model.fusion

    def two_frame_forward(key_feats, ref1_feats, ref2_feats):
        fused = list(key_feats)
        for level in fusion.fusion_levels:
            idx = fusion.level_to_idx[level]
            key = key_feats[idx]
            ref1 = ref1_feats[idx]
            # ref1 同时作为两个 ref 输入（ref2 路径等同 ref1）
            aligned1 = fusion.align_ref1[level](ref1, key)
            aligned2 = fusion.align_ref2[level](ref1, key)
            fusion._last_aligned_r1[level] = aligned1
            fusion._last_aligned_r2[level] = aligned2
            fused[idx] = fusion.fusion[level](key, aligned1, aligned2)
        return tuple(fused)

    fusion.forward = two_frame_forward
    # 对齐 loss 也要按两帧口径: forward 已把 ref2 路径替换成 ref1,
    # 但 AlignLoss 仍会拿原始 ref2 特征算 loss; 这里同步覆盖, 避免训练
    # 一个 forward 根本不用的 ref2 分支.
    _orig_align_loss = fusion.get_alignment_loss

    def two_frame_alignment_loss(key_feats=None, ref1_feats=None, ref2_feats=None):
        return _orig_align_loss(key_feats, ref1_feats, ref1_feats)

    fusion.get_alignment_loss = two_frame_alignment_loss
    return model

from utils import util


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=int, default=2, choices=[0, 2, 3])
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--input-size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lrf', type=float, default=0.01)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--align-loss-weight', type=float, default=0.5,
                   help='对齐 loss 在总 loss 中的权重（默认 0.5，训练 early stage 建议 0.5~1.0）')
    p.add_argument('--align-cos-weight', type=float, default=1.0,
                   help='对齐 loss 中 cosine similarity 项相对于 L1 的权重。'
                        'cosine 对特征量级不敏感，关注结构对齐质量。'
                        'align-only 模式下建议调大到 2.0~5.0')
    p.add_argument('--align-only', action='store_true',
                   help='仅训练对齐模块（只用 align loss，跳过 det loss）')
    p.add_argument('--resume', type=str, default=None,
                   help='从 checkpoint 恢复训练（含 optimizer/scheduler 状态）。'
                        '断电后直接用同一 stage 的 last.pt resume，无需改任何参数')
    p.add_argument('--weights', type=str, default=None,
                   help='仅加载模型权重（不含 optimizer/scheduler）。'
                        '用于 Stage 2→3 跨阶段接力，epoch 从 0 开始')
    p.add_argument('--dataset-root', type=str,
                   default=os.environ.get('DATASET_ROOT') or os.path.join(_WORKSPACE, 'datasets', 'weed_sliding_merged'))
    p.add_argument('--pretrained', type=str,
                   default=os.path.join(_WORKSPACE,
                       'experiments/single_frame_baseline/results/yolo_v8_n_p2_lite_p4/seed42/best.pt'))
    p.add_argument('--output-dir', type=str,
                   default=os.path.join(_SCRIPT_DIR, 'weights'))
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--fusion-levels', type=str, default='p4',
                   help='融合层：p3、p4 或 p3,p4；P2 因特征图过大不参与融合')
    p.add_argument('--base-model', type=str, default='p2l_p4',
                   choices=['p2l', 'p2l_p4', 'p34l'],
                   help='单帧基座: p2l=P2/P3/P4 backbone+P3/P4双头; '
                        'p2l_p4=P2/P3/P4 backbone+P4单头(原版); '
                        'p34l=P3/P4 backbone+P3/P4双头')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fusion-version', type=int, default=2, choices=[2, 3],
                   help='2=DeformAlignV2, 3=HybridAlign (PCD+DCNv2)')
    p.add_argument('--use-sharpener', action='store_true',
                   help='在 DeformAlignV2 输入前加特征锐化层（Laplacian 初始化，~1K params/level）')
    p.add_argument('--align-mode', type=str, default=None,
                   help='V3 时每层的对齐方式，如 p3:hybrid,p4:dav2')
    p.add_argument('--align-hidden-ch', type=int, default=None,
                   help='DAV2 offset predictor 隐藏通道数 (默认 min(96, in_ch*2))')
    p.add_argument('--fusion-gate-type', type=str, default='channel',
                   choices=['channel', 'spatial', 'convgru', 'concat', 'tsa', 'tsa_add', 'avg'],
                   help='融合门控: channel/spatial/convgru/concat/tsa/tsa_add(加法约束)/avg=简单平均')
    p.add_argument('--unfreeze-head', action='store_true',
                   help='Stage 3 时一并解冻 head（让 head 适配融合特征）')
    p.add_argument('--gate-init-bias', type=float, default=-2.0,
                   help='spatial gate 输出层偏置初始值 (sigmoid(bias): -2→0.12, 0→0.5, 2→0.88)')
    p.add_argument('--convgru-kernel', type=int, default=3,
                   help='ConvGRU 卷积核大小 (默认 3)')
    p.add_argument('--convgru-layers', type=int, default=1,
                   help='ConvGRU 层数 (默认 1)')
    p.add_argument('--fusion-mode', type=str, default='gate_diff',
                   choices=['gate_diff', 'gate_nodiff', 'avg'],
                   help='融合模式: gate_diff=对齐+diff gate, gate_nodiff=对齐+gate(无差异图), avg=对齐+简单平均')
    p.add_argument('--no-align', action='store_true',
                   help='跳过 DA 对齐，直接对 raw 特征做融合')
    p.add_argument('--cascade-align', action='store_true',
                   help='级联对齐: ref2→ref1→key，减小远帧对齐步长')
    p.add_argument('--rigid-weight', type=float, default=0.0,
                   help='刚体先验正则化权重 (0=关闭, 建议 0.05~0.2)')
    p.add_argument('--gate-sparse-weight', type=float, default=0.0,
                   help='gate_r2 稀疏正则权重 (0=关闭, 建议 0.01~0.1), 惩罚 gate 均匀化')
    p.add_argument('--iterative-fusion', action='store_true',
                   help='迭代融合: ref2→ref1 融合, 再 ref1*→key 融合, 共用 DA/gate')
    p.add_argument('--two-frame', action='store_true',
                   help='仅用 key+ref1 两帧训练 (消融 ref2 的贡献)')
    p.add_argument('--shared-cascade', action='store_true',
                   help='共用 DA/gate 的级联: ref2→ref1→ref1*, ref1*→key, 同组参数')
    p.add_argument('--image-align', action='store_true',
                   help='图像层全局平移对齐 (grid_sample 点积搜索), backbone 前完成')
    p.add_argument('--fft-align', action='store_true',
                   help='旧模式：FFT 互相关估计位移并在 RGB 层平移，随后进入 backbone')
    p.add_argument('--consistent-align', action='store_true',
                   help='推荐模式：三帧一致性约束位移估计 (ConsistentAligner, top-K + 矢量自洽 + 期望注入)，\n'
                        '取代 --fft-feature-align；RGB 只估计位移，实际平移仍在特征层；\n'
                        'weed_sliding_merged 三集 40px 口径 100%%')
    p.add_argument('--fft-feature-align', action='store_true',
                   help='FFT 在 RGB 上估计位移，在 backbone 后的 P2/P3/P4 特征层执行分数位移')
    p.add_argument('--fft-downsample', type=int, default=8,
                   help='FFT 对齐降采样倍数 (默认 8, 可选 4 获得 ±2px 精度)')
    # 归一化互相关 (cross-power spectrum) 项目统一始终开启，不再提供开关参数。
    p.set_defaults(fft_normalize=True)
    p.add_argument('--frame-dropout', type=float, default=0.2,
                   help='S3 训练时以概率 p 将 ref1/ref2 替换为 key, 迫使 backbone 学习单帧检测 (默认 0.2, 0=关闭)')
    p.add_argument('--no-preload', action='store_true',
                   help='跳过数据集解码缓存预载，改为惰性磁盘读取（大幅降低常驻内存，'
                        '避免共享机器 OOM；代价是每 epoch 增加少量 NVMe I/O）')
    dav2_group = p.add_mutually_exclusive_group()
    dav2_group.add_argument('--dav2-single', action='store_true',
                            help='使用单阶段 DAV2（无第二阶段精修），用于层数消融')
    dav2_group.add_argument('--dav2-triple', action='store_true',
                            help='使用三阶段 DAV2，在标准两阶段后追加一次精修')
    return p.parse_args()


def setup_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def save_ckpt(model, optimizer, scheduler, scaler, ema, epoch,
              best_val, best_pose_ap, path, fusion_version=2, align_mode=None,
              use_sharpener=False, cascade_align=False,
              fft_align=False, fft_feature_align=False, fft_downsample=None,
              consistent_align=False,
              dav2_single=False, dav2_triple=False, fusion_gate_type='channel'):
    """保存完整训练状态：原始模型（可训练）+ EMA 模型（推理）。"""
    ckpt = {
        'model': model.state_dict(),                     # 原始模型 state_dict (可训练)
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'best_val': best_val,
        'best_pose_ap': best_pose_ap,
        'fusion_version': fusion_version,
        'align_mode': align_mode,
        'use_sharpener': use_sharpener,
        'cascade_align': cascade_align,
        'fft_align': fft_align,
        'fft_feature_align': fft_feature_align,
        'consistent_align': consistent_align,
        'global_align_stage': ('feature' if (fft_feature_align or consistent_align) else
                               'image' if fft_align else 'none'),
        'fft_downsample': fft_downsample,
        'dav2_single': dav2_single,
        'dav2_triple': dav2_triple,
        'dav2_stages': 1 if dav2_single else 3 if dav2_triple else 2,
        'fusion_gate_type': fusion_gate_type,
        'fft_chain_align': True,
        'fft_mse_check': True,
        'base_model': model.base_model,
        'feature_levels': list(model.feature_levels),
        'fusion_levels': list(model.fusion_levels),
        'detect_levels': list(model.detect_levels),
    }
    if ema is not None:
        # 直接 half + state_dict，避免 deepcopy
        ema_state = {k: v.half() for k, v in ema.ema.state_dict().items()}
        ckpt['ema_model'] = ema_state
    torch.save(ckpt, path)


def load_ckpt(path, device='cuda', fusion_version=2, align_mode=None, fusion_levels=None,
              use_sharpener=False, align_hidden_ch=None, fusion_gate_type='channel',
              no_align=False, fusion_mode_ablation='gate_diff',
              convgru_kernel=3, convgru_layers=1,
              fft_feature_align=False, fft_downsample=4,
              fft_normalize=True, consistent_align=False,
              dav2_single=False, dav2_triple=False, base_model='p2l_p4'):
    """加载 checkpoint：重建模型并恢复训练状态。state_dict 加载保证了 requires_grad=True。
    no_align / fusion_mode_ablation: 消融实验参数，必须在 load_state_dict 前应用。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # 一致性对齐标志优先读 checkpoint（resume/评估场景调用方可能不显式传入；
    # 若缺失会按 feature FFT 重建模型，导致 consistent 训练的 checkpoint
    # 在 resume/评估时静默丢失全局对齐）。
    consistent_align = consistent_align or bool(ckpt.get('consistent_align', False))

    model_data = ckpt.get('model')
    # 优先用 checkpoint 中的 fusion_version/align_mode，其次用参数传入的
    ckpt_fv = ckpt.get('fusion_version')
    ckpt_am = ckpt.get('align_mode')
    ckpt_sp = ckpt.get('use_sharpener', False)
    fv = ckpt_fv if ckpt_fv is not None else fusion_version
    am = ckpt_am if ckpt_am is not None else align_mode
    sp = ckpt_sp if ckpt_sp else use_sharpener
    if fusion_levels is None:
        fusion_levels = ['p4']
    fusion_levels = list(fusion_levels)
    if fusion_levels not in (['p2'], ['p3'], ['p4'], ['p2', 'p3'],
                             ['p3', 'p4'], ['p2', 'p4'],
                             ['p2', 'p3', 'p4']):
        raise ValueError(f'不支持的 fusion_levels={fusion_levels}')
    ckpt_levels = ckpt.get('fusion_levels')
    if ckpt_levels != fusion_levels:
        raise ValueError(
            f'Checkpoint fusion_levels={ckpt_levels} 与 CLI fusion_levels={fusion_levels} 不一致')
    ckpt_base = ckpt.get('base_model', 'p2l_p4')
    if ckpt_base != base_model:
        raise ValueError(
            f'Checkpoint base_model={ckpt_base} 与 CLI base_model={base_model} 不一致')

    # 安全检查：checkpoint 版本与 CLI 不一致时始终警告
    if ckpt_fv is not None and ckpt_fv != fusion_version:
        import warnings
        warnings.warn(
            f'Checkpoint fusion_version={ckpt_fv} 与 CLI fusion_version={fusion_version} 不一致。'
            f'将使用 checkpoint 的版本 {ckpt_fv}。')
        fv = ckpt_fv
        am = ckpt_am if ckpt_am is not None else align_mode

    # 兼容旧格式：旧 checkpoint 可能存的是完整 Module 对象
    if isinstance(model_data, dict):
        # 新格式：state_dict
        model = MultiFrameYOLO(
            num_classes=1, kpt_shape=(1, 2),
            fusion_levels=fusion_levels,
            pretrained_path=None,
            fusion_version=fv,
            align_mode=am,
            use_sharpener=sp,
            align_hidden_ch=align_hidden_ch,
            fusion_gate_type=fusion_gate_type,
            convgru_kernel=convgru_kernel,
            convgru_layers=convgru_layers,
            global_align_stage='feature' if (fft_feature_align or consistent_align) else 'none',
            fft_downsample=fft_downsample,
            fft_normalize=fft_normalize,
            consistent_align=consistent_align,
            base_model=base_model,
        )
        model = model.to(device)
        apply_ablation(model, no_align=no_align, fusion_mode=fusion_mode_ablation)
        ckpt_dav2_single = ckpt.get('dav2_single', False)
        ckpt_dav2_triple = ckpt.get('dav2_triple', False)
        if ckpt_dav2_single != dav2_single or ckpt_dav2_triple != dav2_triple:
            raise ValueError(
                'Checkpoint DAV2 层数与 CLI 不一致: '
                f'single={ckpt_dav2_single}/{dav2_single}, '
                f'triple={ckpt_dav2_triple}/{dav2_triple}')
        ckpt_gate = ckpt.get('fusion_gate_type')
        if ckpt_gate is not None and ckpt_gate != fusion_gate_type:
            raise ValueError(
                f'Checkpoint fusion_gate_type={ckpt_gate} 与 CLI '
                f'fusion_gate_type={fusion_gate_type} 不一致')
        if dav2_single:
            apply_dav2_single(model)
        elif dav2_triple:
            apply_dav2_triple(model)
        model.load_state_dict(model_data, strict=True)
        if ckpt.get('cascade_align', False):
            apply_cascade(model)
    else:
        raise ValueError('不再支持保存完整 Module 对象的旧 checkpoint 格式')

    state = {
        'optimizer': ckpt.get('optimizer'),
        'scheduler': ckpt.get('scheduler'),
        'scaler': ckpt.get('scaler'),
        'epoch': ckpt.get('epoch', 0),
        'best_val': ckpt.get('best_val', float('inf')),
        'best_pose_ap': ckpt.get('best_pose_ap', 0.0),
    }
    return model, state


# ═══════════════════════════════════════════════════════════
# 定制 Loss: 处理 kpt_shape=[1,2]（无visibility维度）
# ═══════════════════════════════════════════════════════════

class DetLossMF:
    """
    包装原始 ComputeLoss，处理 kpt_shape=[1,2] 的 kpt_mask 问题。

    原始 ComputeLoss 对 kpt 最后一维取 gt_kpt[..., 2] 作为 visibility mask，
    但如果 targets 的 kpt 只有 (x,y) 没有 visibility 维度会报错。
    这里确保 kpt 至少有 3 维（补 visibility=2.0）。
    """
    def __init__(self, model, params):
        self.inner = util.ComputeLoss(model, params)
        self.device = self.inner.device
        self.params = self.inner.params
        self.kpt_shape = model.head.kpt_shape
        self.has_kpt_vis = (self.kpt_shape[1] == 3)

    def __call__(self, outputs, targets):
        # 确保 kpt targets 有 visibility 维度
        kpt = targets.get('kpt')
        if kpt is not None and kpt.dim() == 3 and kpt.shape[-1] == 2:
            # (N, nk, 2) → (N, nk, 3)，补 visibility=2 (labeled & visible)
            vis = torch.full((*kpt.shape[:-1], 1), 2.0, device=kpt.device, dtype=kpt.dtype)
            targets = {**targets, 'kpt': torch.cat([kpt, vis], dim=-1)}
        return self.inner(outputs, targets)


class RigidRegularizer:
    """
    刚体先验正则化：从 DAV2 per-pixel 偏移场拟合全局刚体 (tx, ty, θ)，
    惩罚各单位像素偏移偏离刚体模型的残差。

    用法：
        reg = RigidRegularizer(model, weight=0.1)
        loss = reg.compute()  # 在 forward 后调用
    """
    def __init__(self, model, weight=0.1):
        self.model = model
        self.weight = weight
        # 注册 hooks 捕获 DAV2 offset 输出
        self._offset_cache = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        fusion = self.model.fusion
        for level in fusion.fusion_levels:
            for side in ['align_ref1', 'align_ref2']:
                mod = getattr(fusion, side)[level]
                if hasattr(mod, 's1_offset'):
                    h = mod.s1_offset.register_forward_hook(
                        lambda m, inp, out, l=level, s=side:
                            self._offset_cache.setdefault(f'{l}_{s}_s1', []).append(out.detach()))
                    self._hooks.append(h)
                if hasattr(mod, 's2_offset'):
                    h = mod.s2_offset.register_forward_hook(
                        lambda m, inp, out, l=level, s=side:
                            self._offset_cache.setdefault(f'{l}_{s}_s2', []).append(out.detach()))
                    self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute(self):
        """在 model.forward() 之后调用。返回刚体正则化 loss。"""
        total_loss = 0.0
        count = 0

        for key, offset_list in self._offset_cache.items():
            if not offset_list:
                continue
            off = offset_list[-1]  # (B, 18, H, W) for 3×3 kernel
            # 取 kernel 中心的偏移作为该像素的 flow
            k = 3
            cx = off[:, 4, :, :]      # (B, H, W) x-offset at center
            cy = off[:, 4 + k*k, :, :]  # (B, H, W) y-offset at center

            B, H, W = cx.shape
            device = cx.device

            # 构建像素坐标
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij')
            gx = gx.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
            gy = gy.unsqueeze(0).expand(B, -1, -1)

            # 目标位置: (gx + cx, gy + cy)
            tx = gx + cx
            ty = gy + cy

            # ── 最小二乘拟合刚体:  x' = R_theta * x + t ──
            # 对每组 batch 做加权最小二乘
            for b in range(B):
                src_x = gx[b].flatten()   # (N,)
                src_y = gy[b].flatten()
                dst_x = tx[b].flatten()
                dst_y = ty[b].flatten()
                N = src_x.numel()

                # 刚体拟合: [dst_x, dst_y] = [cos, -sin, tx; sin, cos, ty] @ [src_x, src_y, 1]
                # 构造: A * p = b, 其中 p = [cos, sin, tx, ty]
                # 但 cos/sin 有约束 cos²+sin²=1，简化为线性近似 cos≈1, sin≈θ (小角度)
                # 更准确的：用线性最小二乘拟合仿射，然后提取刚体分量
                # 简化版：拟合全局平移 tx,ty 和旋转 θ

                # 中心化坐标
                mx_s = src_x.mean(); my_s = src_y.mean()
                mx_d = dst_x.mean(); my_d = dst_y.mean()
                sx = src_x - mx_s; sy = src_y - my_s
                dx = dst_x - mx_d; dy = dst_y - my_d

                # 估计旋转 θ: θ = mean(dx*sy - dy*sx) / mean(sx² + sy²)  (小角度近似)
                denom = (sx*sx + sy*sy).mean() + 1e-8
                theta = (dx*sy - dy*sx).mean() / denom

                # 估计平移
                tx_hat = mx_d - (mx_s * 1.0 - my_s * theta)
                ty_hat = my_d - (mx_s * theta + my_s * 1.0)

                # 刚体预测的目标位置
                rigid_dx = src_x - theta * src_y + tx_hat
                rigid_dy = theta * src_x + src_y + ty_hat

                # 残差
                res_x = dst_x - rigid_dx
                res_y = dst_y - rigid_dy
                res = (res_x.abs() + res_y.abs()).mean()
                total_loss += res
                count += 1

        self._offset_cache.clear()
        if count == 0:
            return 0.0
        return self.weight * total_loss / count


class GateSparseLoss:
    """
    Gate 稀疏正则：惩罚 gate_r2 均值过高，鼓励选择性开门。
    loss = mean(gate_r2)
    当前 gate_r2 ≈ 0.82，正则化推动模型降��不需要的区域。
    """
    def __init__(self, model, weight=0.05):
        self.model = model
        self.weight = weight
        # 注册 hook 捕获 gate_r2
        self._gate_cache = []
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        fusion = self.model.fusion
        for level in fusion.fusion_levels:
            mod = fusion.fusion[level]
            if hasattr(mod, 'gate_r2_conv2') and mod.gate_r2_conv2 is not None:
                # Hook 在 sigmoid 之前/之后取决于是否要未激活值
                # 直接用 forward hook 捕获整个模块输出中的 gate
                pass

    def compute(self):
        """在 model.forward() 后调用，需要 gate 已被计算。"""
        # 直接用 get_gate_weights 会双重前向，改用 hook 方案
        # 简单方案：利用 forward 缓存的特征重新计算 gate
        total_loss = 0.0
        count = 0
        if self.model._last_key_feats is not None:
            key_feats = self.model._last_key_feats
            ref1_feats = self.model._last_ref1_feats
            ref2_feats = self.model._last_ref2_feats
            fusion = self.model.fusion
            for level in fusion.fusion_levels:
                idx = fusion.level_to_idx[level]
                key = key_feats[idx]
                ref1 = ref1_feats[idx]
                ref2 = ref2_feats[idx]
                aligned1 = fusion.align_ref1[level](ref1, key)
                aligned2 = fusion.align_ref2[level](ref2, key)
                fmod = fusion.fusion[level]
                if hasattr(fmod, 'get_gate_weights'):
                    with torch.no_grad():
                        _, s2 = fmod.get_gate_weights(key, aligned1, aligned2)
                    total_loss += s2.mean()
                    count += 1
        if count == 0:
            return 0.0
        return self.weight * total_loss / count


class AlignLoss:
    """对齐辅助 loss：L1 + Cosine Similarity，复用 forward() 缓存的 backbone 和对齐特征"""
    def __init__(self, model):
        self.model = model

    def compute(self, frames=None):
        """
        计算对齐 loss。优先复用 model.forward() 中缓存的 backbone 特征
        和 fusion.forward() 中缓存的对齐特征，避免双重前向。
        frames 仅在缓存缺失时使用（向后兼容）。

        返回: (l1_loss, cos_sim_loss) 或仅 l1_loss（向后兼容）
        """
        if self.model._last_key_feats is not None:
            key_feats = self.model._last_key_feats
            ref1_feats = self.model._last_ref1_feats
            ref2_feats = self.model._last_ref2_feats
        elif frames is not None:
            with torch.no_grad():
                key_feats = self.model.backbone(frames[:, 0])
                ref1_feats = self.model.backbone(frames[:, 1])
                ref2_feats = self.model.backbone(frames[:, 2])
        else:
            raise RuntimeError('AlignLoss.compute: no cached features and no frames')

        l1_loss = self.model.fusion.get_alignment_loss(key_feats, ref1_feats, ref2_feats)

        # Cosine similarity 辅助指标：对齐后特征与 key 的余弦距离
        cos_loss = self._compute_cosine_loss(key_feats)

        return l1_loss, cos_loss

    def _compute_cosine_loss(self, key_feats):
        """
        计算对齐后特征与 key 的 1 - CosineSimilarity。
        余弦相似度对特征量级不敏感，是 L1 的互补指标。
        """
        total = 0.0
        count = 0
        aligned_r1, aligned_r2 = self.model.fusion.get_aligned_features()
        for level in self.model.fusion.fusion_levels:
            idx = self.model.fusion.level_to_idx[level]
            key = key_feats[idx]
            key_flat = F.normalize(key.view(key.size(0), key.size(1), -1), dim=2)

            for aligned_dict in [aligned_r1, aligned_r2]:
                if level in aligned_dict:
                    al = aligned_dict[level]
                    al_flat = F.normalize(al.view(al.size(0), al.size(1), -1), dim=2)
                    cos_sim = (al_flat * key_flat).sum(dim=2).mean()
                    total += 1.0 - cos_sim  # 余弦距离
                    count += 1
        return total / count if count > 0 else torch.tensor(0.0)


# ═══════════════════════════════════════════════════════════
# 训练主流程
# ═══════════════════════════════════════════════════════════

def _dav2_replace_levels(model):
    """共享 offset 时只替换共享层；否则替换所有融合层。"""
    shared = getattr(model.fusion, 'shared_offset_level', None)
    return [shared] if shared else list(model.fusion.fusion_levels)


def apply_dav2_triple(model):
    """将所选融合层的标准 DAV2 替换为三阶段版本；须在严格加载权重前调用。"""
    from models.modules import DeformAlignV2Triple
    for level in _dav2_replace_levels(model):
        align1 = model.fusion.align_ref1[level]
        if isinstance(align1, DeformAlignV2Triple) or not hasattr(align1, 's1_conv1'):
            continue
        ch = align1.s1_conv1.in_channels // 2
        hidden_ch = align1.s1_conv1.out_channels
        use_sharpener = getattr(align1, 'use_sharpener', False)
        device = next(model.parameters()).device
        model.fusion.align_ref1[level] = DeformAlignV2Triple(
            ch, hidden_ch=hidden_ch, use_sharpener=use_sharpener).to(device)
        model.fusion.align_ref2[level] = DeformAlignV2Triple(
            ch, hidden_ch=hidden_ch, use_sharpener=use_sharpener).to(device)


def apply_dav2_single(model):
    """将所选融合层的双阶段 DAV2 替换为单阶段版本；须在严格加载权重前调用。"""
    from models.modules import DeformAlignV2Single
    for level in _dav2_replace_levels(model):
        align1 = model.fusion.align_ref1[level]
        if isinstance(align1, DeformAlignV2Single) or not hasattr(align1, 's1_conv1'):
            continue
        ch = align1.s1_conv1.in_channels // 2
        device = next(model.parameters()).device
        model.fusion.align_ref1[level] = DeformAlignV2Single(ch).to(device)
        model.fusion.align_ref2[level] = DeformAlignV2Single(ch).to(device)


def apply_dcn_variant(model, variant='v2'):
    """Replace the shared DCN module with a structural variant (see dcn_variants).

    Must run after apply_dav2_single / apply_shared_offset and before loading
    weights.  Variants are functional supersets of V2 at initialisation.
    """
    if variant in (None, '', 'v2', 'none'):
        return model
    from models.dcn_variants import DeformAlignV2Adaptive, OffsetReuseAlignRefine

    # Numeric aliases for common combinations.
    alias = {'ms': 'coarse+dil2', 'ms_p2': 'coarse+dil2+p2refine'}
    expr = alias.get(variant, variant)
    tokens = [t for t in expr.split('+') if t]
    align_tokens = [t for t in tokens if t in ('corr', 'k5', 'coarse', 'dil2', 'dil')]
    want_p2 = 'p2refine' in tokens or 'p2' in tokens
    unknown = set(tokens) - set(align_tokens) - {'p2refine', 'p2'}
    if unknown:
        raise ValueError(f'unknown dcn_variant={variant!r} (tokens={tokens})')
    if not align_tokens and not want_p2:
        raise ValueError(f'empty dcn_variant={variant!r}')

    fusion = model.fusion
    shared = getattr(fusion, 'shared_offset_level', None)
    levels = [shared] if shared else list(fusion.fusion_levels)
    device = next(model.parameters()).device

    if align_tokens:
        dag_variant = '+'.join(align_tokens)
        for level in levels:
            align1 = fusion.align_ref1[level]
            ch = align1.s1_conv1.in_channels // 2
            hidden = align1.s1_conv1.out_channels
            fusion.align_ref1[level] = DeformAlignV2Adaptive(
                ch, hidden_ch=hidden, variant=dag_variant).to(device)
            fusion.align_ref2[level] = DeformAlignV2Adaptive(
                ch, hidden_ch=hidden, variant=dag_variant).to(device)
    if want_p2:
        if 'p2' not in fusion.fusion_levels:
            raise ValueError('p2refine requires p2 in fusion_levels')
        for side in ('1', '2'):
            align = getattr(fusion, f'align_ref{side}')
            ch = fusion.fusion_channels['p2']
            align['p2'] = OffsetReuseAlignRefine(
                ch, reuse_mode=fusion.offset_reuse_mode).to(device)
    model.dcn_variant = variant
    return model


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_seed(args.seed)
    if args.fft_align and args.fft_feature_align:
        raise ValueError('--fft-align 与 --fft-feature-align 不能同时开启')
    if args.image_align and args.fft_feature_align:
        raise ValueError('--image-align 与 --fft-feature-align 不能同时开启')
    if args.consistent_align and (args.fft_align or args.fft_feature_align or args.image_align):
        raise ValueError('--consistent-align 与 --fft-align/--fft-feature-align/--image-align 不能同时开启')
    if args.cascade_align and (args.fft_align or args.fft_feature_align or
                               args.consistent_align or args.image_align):
        raise ValueError(
            '--cascade-align 是普通特征级链式对齐，不能与 FFT/consistent/image '
            '全局预对齐同时开启')

    # 默认 P3/P4；显式实验支持单层、双层、三层融合。
    fusion_levels = args.fusion_levels.split(',') if args.fusion_levels else ['p4']
    if fusion_levels not in (['p2'], ['p3'], ['p4'], ['p2', 'p3'],
                             ['p3', 'p4'], ['p2', 'p4'],
                             ['p2', 'p3', 'p4']):
        raise ValueError(f'不支持的 --fusion-levels {args.fusion_levels!r}')
    align_mode = None
    if args.align_mode:
        align_mode = {}
        for pair in args.align_mode.split(','):
            k, v = pair.strip().split(':')
            align_mode[k.strip()] = v.strip()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, Stage: {args.stage}')

    # 模型
    if args.resume:
        print(f'Resuming from {args.resume}')
        model, ckpt_state = load_ckpt(args.resume, device,
                                       fusion_version=args.fusion_version,
                                       align_mode=align_mode,
                                       fusion_levels=fusion_levels,
                                       use_sharpener=args.use_sharpener,
                                       align_hidden_ch=args.align_hidden_ch,
                                       fusion_gate_type=args.fusion_gate_type,
                                       no_align=args.no_align,
                                       fusion_mode_ablation=args.fusion_mode,
                                       convgru_kernel=args.convgru_kernel,
                                       convgru_layers=args.convgru_layers,
                                       fft_feature_align=args.fft_feature_align,
                                       fft_downsample=args.fft_downsample,
                                       fft_normalize=args.fft_normalize,
                                       consistent_align=args.consistent_align,
                                       dav2_single=args.dav2_single,
                                       dav2_triple=args.dav2_triple,
                                       base_model=args.base_model)
        # Resume 后重新应用 cascade / image-align / fft-align（不在 checkpoint 中）
        if args.cascade_align:
            apply_cascade(model)
        if args.consistent_align:
            print(f'Resume: 一致性约束位移估计 (ds={args.fft_downsample}x)')
        elif args.fft_align:
            from models.fft_align import apply_fft_image_align
            apply_fft_image_align(model, downsample=args.fft_downsample)
            print(f'Resume: 重新应用 FFT 图像层对齐 (ds={args.fft_downsample}x)')
        elif args.image_align:
            apply_image_align(model)
            print('Resume: 重新应用 grid_sample 图像层对齐')
        start_epoch = ckpt_state['epoch']
        best_val = ckpt_state['best_val']
        best_pose_ap = ckpt_state['best_pose_ap']
    elif args.weights:
        # 仅加载模型权重，epoch 从 0 开始（Stage 2→3 跨阶段接力用）
        print(f'Loading weights from {args.weights}')
        model = MultiFrameYOLO(
            num_classes=1, kpt_shape=(1, 2),
            fusion_levels=fusion_levels,
            pretrained_path=args.pretrained if args.pretrained else None,
            fusion_version=args.fusion_version,
            align_mode=align_mode,
            use_sharpener=args.use_sharpener,
            align_hidden_ch=args.align_hidden_ch,
            fusion_gate_type=args.fusion_gate_type,
            gate_init_bias=args.gate_init_bias,
            convgru_kernel=args.convgru_kernel,
            convgru_layers=args.convgru_layers,
            global_align_stage='feature' if (args.fft_feature_align or args.consistent_align) else 'none',
            fft_downsample=args.fft_downsample,
            fft_normalize=args.fft_normalize,
            consistent_align=args.consistent_align,
            base_model=args.base_model,
        )
        model = model.to(device)
        apply_ablation(model, no_align=args.no_align, fusion_mode=args.fusion_mode)
        if args.cascade_align:
            apply_cascade(model)
        if args.iterative_fusion:
            apply_iterative_fusion(model)
        if args.two_frame:
            apply_two_frame(model)
        if args.shared_cascade:
            apply_shared_cascade(model)
        if args.consistent_align:
            print(f'一致性约束位移估计: ConsistentAligner 三帧自洽 (ds={args.fft_downsample}x), 特征层平移')
        elif args.fft_align:
            from models.fft_align import apply_fft_image_align
            apply_fft_image_align(model, downsample=args.fft_downsample)
            print(f'FFT 图像层对齐 (ds={args.fft_downsample}x)')
        elif args.image_align:
            apply_image_align(model)
        if args.dav2_single:
            apply_dav2_single(model)
            print('单阶段 DAV2: 仅 Stage 1, 无精修级联')
        elif args.dav2_triple:
            apply_dav2_triple(model)
            print('三阶段 DAV2: 在标准两阶段后追加 Stage 3 精修')
        # 仅加载 model weights（EMA 优先），跳过 optimizer/scheduler/epoch
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        if ckpt.get('fusion_levels') != fusion_levels:
            raise ValueError(
                f'Checkpoint fusion_levels={ckpt.get("fusion_levels")} 与 CLI '
                f'fusion_levels={fusion_levels} 不一致')
        if (ckpt.get('dav2_single', False) != args.dav2_single or
                ckpt.get('dav2_triple', False) != args.dav2_triple):
            raise ValueError('Checkpoint DAV2 层数与 CLI 不一致')
        ckpt_gate = ckpt.get('fusion_gate_type')
        if ckpt_gate is not None and ckpt_gate != args.fusion_gate_type:
            raise ValueError(
                f'Checkpoint fusion_gate_type={ckpt_gate} '
                f'与 CLI fusion_gate_type={args.fusion_gate_type} 不一致')
        sd = ckpt.get('ema_model') or ckpt.get('model')
        if isinstance(sd, dict):
            model.load_state_dict(sd, strict=True)
        print(f'[weights] {len(sd)} keys loaded, starting from epoch 0')
        start_epoch = 0
        ckpt_state = None
        best_val = float('inf')
        best_pose_ap = 0.0
    else:
        model = MultiFrameYOLO(
            num_classes=1, kpt_shape=(1, 2),
            fusion_levels=fusion_levels,
            pretrained_path=args.pretrained if args.pretrained else None,
            fusion_version=args.fusion_version,
            align_mode=align_mode,
            use_sharpener=args.use_sharpener,
            align_hidden_ch=args.align_hidden_ch,
            fusion_gate_type=args.fusion_gate_type,
            gate_init_bias=args.gate_init_bias,
            convgru_kernel=args.convgru_kernel,
            convgru_layers=args.convgru_layers,
            global_align_stage='feature' if (args.fft_feature_align or args.consistent_align) else 'none',
            fft_downsample=args.fft_downsample,
            fft_normalize=args.fft_normalize,
            consistent_align=args.consistent_align,
            base_model=args.base_model,
        )
        model = model.to(device)
        start_epoch = 0
        ckpt_state = None
        best_val = float('inf')
        best_pose_ap = 0.0
        apply_ablation(model, no_align=args.no_align, fusion_mode=args.fusion_mode)
        if args.cascade_align:
            apply_cascade(model)
            print('级联对齐: ref2→ref1→key')
        if args.iterative_fusion:
            apply_iterative_fusion(model)
            print('迭代融合: ref2→ref1*, ref1*→key')
        if args.two_frame:
            apply_two_frame(model)
            print('两帧模式: 仅用 key+ref1')
        if args.shared_cascade:
            apply_shared_cascade(model)
            print('共用级联: ref2→ref1→ref1*, ref1*→key, 共用 DA/gate')
        if args.consistent_align:
            print(f'一致性约束位移估计: top-K + 三帧自洽 + 期望注入 (ds={args.fft_downsample}x), 特征层平移 + DAV2 精修')
        elif args.fft_align:
            from models.fft_align import apply_fft_image_align
            apply_fft_image_align(model, downsample=args.fft_downsample)
            print(f'FFT 图像层对齐: FFT互相关 (ds={args.fft_downsample}x) + backbone + DAV2 精修')
        elif args.fft_feature_align:
            print(f'FFT 特征层对齐: RGB估计位移 (ds={args.fft_downsample}x) + raw P2/P3/P4 分数位移 + P4 DAV2')
        elif args.image_align:
            apply_image_align(model)
            print('图像层对齐: grid_sample 点积搜索 + backbone + DAV2')
        if args.dav2_single:
            apply_dav2_single(model)
            print('单阶段 DAV2: 仅 Stage 1, 无精修级联')
        elif args.dav2_triple:
            apply_dav2_triple(model)
            print('三阶段 DAV2: 在标准两阶段后追加 Stage 3 精修')

    # 阶段参数控制
    if args.stage == 0:
        # Joint training: backbone/fusion/FPN 全可训练，Head 冻结
        model.unfreeze_all()
        for p in model.head.parameters():
            p.requires_grad_(False)
        if hasattr(model.head, 'dfl') and hasattr(model.head.dfl, 'conv'):
            model.head.dfl.conv.weight.requires_grad_(False)
        print('Stage 0 (joint): backbone + fusion + FPN trainable, Head frozen')
    elif args.stage == 2:
        model.freeze_backbone()
        model.freeze_fpn_head()
        print('Stage 2: backbone/FPN/Head frozen')
    else:
        # 阶段 3：解冻 backbone + fusion + FPN，Head 和 DFL 冻结（可选用 --unfreeze-head 解冻 head）
        model.unfreeze_all()
        if not args.unfreeze_head:
            for p in model.head.parameters():
                p.requires_grad_(False)
            if hasattr(model.head, 'dfl') and hasattr(model.head.dfl, 'conv'):
                model.head.dfl.conv.weight.requires_grad_(False)
        print(f'Stage 3: end-to-end fine-tuning (head {"unfrozen" if args.unfreeze_head else "frozen"})')

    # 数据
    train_loader, val_loader = create_dataloaders(
        args.dataset_root, input_size=args.input_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        no_preload=args.no_preload,
    )

    # Loss
    det_loss_fn = DetLossMF(model, {
        'cls': 0.5, 'box': 7.5, 'dfl': 1.5, 'kpt': 12.0, 'obj': 1.0,
        'kpt_shape': [1, 2], 'names': {0: 'weed'},
    })
    align_loss_fn = AlignLoss(model)
    rigid_reg = RigidRegularizer(model, weight=args.rigid_weight) if args.rigid_weight > 0 else None
    gate_sparse = GateSparseLoss(model, weight=args.gate_sparse_weight) if args.gate_sparse_weight > 0 else None

    # Optimizer（纯对齐模式用 Adam，联合训练用 SGD）
    if args.align_only:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=0)
    elif args.stage == 2:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=args.lr, momentum=0.937,
                                     weight_decay=args.weight_decay, nesterov=True)
    else:
        bb, fus, oth = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith('backbone'):
                bb.append(p)
            elif n.startswith('fusion'):
                fus.append(p)
            else:
                oth.append(p)
        optimizer = torch.optim.SGD([
            {'params': bb, 'lr': args.lr * 0.1},
            {'params': fus, 'lr': args.lr},
            {'params': oth, 'lr': args.lr * 0.5},
        ], lr=args.lr, momentum=0.937, weight_decay=args.weight_decay, nesterov=True)

    if args.align_only:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        # 使用 CosineAnnealingLR 避免 LambdaLR 在多 group 下的兼容问题
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * args.lrf)
    ema = util.EMA(model) if not args.align_only else None
    scaler = torch.cuda.amp.GradScaler()

    # 恢复训练状态（optimizer/scheduler/scaler）
    # 跨优化器/分组切换时跳过
    if ckpt_state is not None:
        if ckpt_state['optimizer'] is not None:
            cur_pg = optimizer.state_dict()['param_groups']
            ckpt_pg = ckpt_state['optimizer']['param_groups']
            cur_keys = set(cur_pg[0].keys())
            ckpt_keys = set(ckpt_pg[0].keys())
            if cur_keys == ckpt_keys and len(cur_pg) == len(ckpt_pg):
                optimizer.load_state_dict(ckpt_state['optimizer'])
                print('  已恢复 optimizer 状态')
            else:
                print('  优化器分组/类型不匹配，使用新优化器')
        if ckpt_state['scheduler'] is not None:
            try:
                scheduler.load_state_dict(ckpt_state['scheduler'])
                if len(scheduler.base_lrs) != len(optimizer.param_groups):
                    raise ValueError('base_lrs count mismatch')
            except (KeyError, ValueError):
                print('  调度器不兼容，重建新调度器')
                if args.align_only:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=args.epochs, eta_min=args.lr * args.lrf)
        if ckpt_state['scaler'] is not None:
            scaler.load_state_dict(ckpt_state['scaler'])

    nb = len(train_loader)
    nw = max(round(3.0 * nb), 100)

    log_mode = 'a' if args.resume else 'w'
    log_f = open(os.path.join(args.output_dir, f'stage{args.stage}_log.csv'), log_mode)
    w = csv.DictWriter(log_f, fieldnames=['epoch', 'loss', 'det_loss', 'align_loss',
                                           'align_l1', 'align_cos',
                                           'val_loss', 'BoxAP', 'PoseAP'])
    if not args.resume:
        w.writeheader()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        mloss = util.AverageMeter()
        mdet = util.AverageMeter()
        malign = util.AverageMeter()
        ml1 = util.AverageMeter()
        mcos = util.AverageMeter()

        pbar = tqdm.tqdm(enumerate(train_loader), total=nb, desc=f'E{epoch+1}/{args.epochs}')

        for i, (frames, targets) in pbar:
            x = i + nb * epoch
            frames = frames.to(device).float() / 255.0

            # Frame Dropout: 随机将 ref1/ref2 替换为 key，迫使 backbone 学习单帧检测
            if args.stage == 3 and args.frame_dropout > 0:
                mask = torch.rand(frames.shape[0], device=frames.device) < args.frame_dropout
                if mask.any():
                    frames[mask, 1] = frames[mask, 0]  # ref1 ← key
                    frames[mask, 2] = frames[mask, 0]  # ref2 ← key

            if x <= nw and not args.align_only:
                # Warmup: 线性插值从 0 到 initial_lr
                xp = [0, nw]
                for g in optimizer.param_groups:
                    g['lr'] = np.interp(x, xp, [0.0, g['initial_lr']])

            with torch.cuda.amp.autocast():
                outputs = model(frames)
                # 纯对齐模式：跳过检测 loss，对齐 loss 权重强制为 1.0
                if args.align_only:
                    det_loss = torch.tensor(0.0, device=device)
                    align_weight = 1.0
                else:
                    det_loss = det_loss_fn(outputs, targets)
                    align_weight = args.align_loss_weight
                align_l1, align_cos = align_loss_fn.compute()
                # 对齐 loss = L1 + λ_cos * cos_distance
                align_loss = align_l1 + args.align_cos_weight * align_cos
                loss = det_loss + align_weight * align_loss

                if rigid_reg is not None:
                    rigid_loss = rigid_reg.compute()
                    loss = loss + rigid_loss

                if gate_sparse is not None:
                    loss = loss + gate_sparse.compute()

            mloss.update(loss.item(), frames.size(0))
            mdet.update(det_loss.item(), frames.size(0))
            malign.update(align_loss.item(), frames.size(0))
            ml1.update(align_l1.item(), frames.size(0))
            mcos.update(align_cos.item(), frames.size(0))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

            pbar.set_description(
                f'E{epoch+1}/{args.epochs} L={mloss.avg:.2f} D={mdet.avg:.2f} '
                f'A={malign.avg:.4f}(L1={ml1.avg:.4f} C={mcos.avg:.4f})')

        scheduler.step()

        # Validation: loss + mAP (纯对齐模式下跳过验证)
        if args.align_only:
            val_loss = float('nan')
            box_ap, pose_ap = 0.0, 0.0
        else:
            val_loss = validate(model, val_loader, det_loss_fn, align_loss_fn, args, device)
            # 评估时直接用 model (EMA 的 BN stats 滞后会导致 AP=0)
            box_ap, pose_ap = test(args, model, val_loader)

        print(f'[E{epoch+1}] train: L={mloss.avg:.2f} D={mdet.avg:.2f} '
              f'A={malign.avg:.4f}(L1={ml1.avg:.4f} Cos={mcos.avg:.4f}) | '
              f'val_loss={val_loss:.2f} '
              f'BoxAP={box_ap:.3f} PoseAP={pose_ap:.3f}')
        w.writerow({
            'epoch': epoch + 1, 'loss': f'{mloss.avg:.2f}',
            'det_loss': f'{mdet.avg:.2f}', 'align_loss': f'{malign.avg:.4f}',
            'align_l1': f'{ml1.avg:.4f}', 'align_cos': f'{mcos.avg:.4f}',
            'val_loss': f'{val_loss:.2f}',
            'BoxAP': f'{box_ap:.3f}', 'PoseAP': f'{pose_ap:.3f}',
        })
        log_f.flush()

        if val_loss < best_val:
            best_val = val_loss
        if pose_ap > best_pose_ap:
            best_pose_ap = pose_ap
            save_ckpt(model, optimizer, scheduler, scaler, ema, epoch + 1,
                      best_val, best_pose_ap,
                      os.path.join(args.output_dir, f'mf_stage{args.stage}_best.pt'),
                      fusion_version=args.fusion_version, align_mode=align_mode,
                      use_sharpener=args.use_sharpener, cascade_align=args.cascade_align,
                      fft_align=args.fft_align,
                      fft_feature_align=args.fft_feature_align,
                      consistent_align=args.consistent_align,
                      fft_downsample=args.fft_downsample,
                      dav2_single=args.dav2_single,
                      dav2_triple=args.dav2_triple,
                      fusion_gate_type=args.fusion_gate_type)
        save_ckpt(model, optimizer, scheduler, scaler, ema, epoch + 1,
                  best_val, best_pose_ap,
                  os.path.join(args.output_dir, f'mf_stage{args.stage}_last.pt'),
                  fusion_version=args.fusion_version, align_mode=align_mode,
                  use_sharpener=args.use_sharpener, cascade_align=args.cascade_align,
                  fft_align=args.fft_align,
                  fft_feature_align=args.fft_feature_align,
                  consistent_align=args.consistent_align,
                  fft_downsample=args.fft_downsample,
                  dav2_single=args.dav2_single,
                  dav2_triple=args.dav2_triple,
                  fusion_gate_type=args.fusion_gate_type)

    log_f.close()
    print(f'\nStage {args.stage} done. Best val_loss: {best_val:.2f}, Best PoseAP: {best_pose_ap:.3f}')


@torch.no_grad()
def validate(model, loader, det_loss_fn, align_loss_fn, args, device):
    """
    验证集 loss 评估。
    注意：ComputeLoss 依赖训练模式的 head 输出分布，
    所以这里用 model.train() 做前向。BN 的 momentum 被临时设为 0.0
    （即 running stats 冻结），以防止验证数据污染 running stats。
    """
    mloss = util.AverageMeter()

    # BN 的 running stats 更新公式：
    #   new = (1-momentum) × old + momentum × batch
    # momentum=0 → new=old，即 running stats 保持不变
    bn_momentum = {}
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_momentum[m] = m.momentum
            m.momentum = 0.0

    try:
        for frames, targets in loader:
            frames = frames.to(device).float() / 255.0
            # ComputeLoss 需要训练模式输出 → 用 model.train() 做前向
            # BN 已被冻结（momentum=0.0），所以 running stats 不会变
            model.train()
            outputs = model(frames)

            try:
                det_loss = det_loss_fn(outputs, targets)
                align_l1, align_cos = align_loss_fn.compute()
                align_loss = align_l1 + args.align_cos_weight * align_cos
                loss = det_loss + args.align_loss_weight * align_loss
                mloss.update(loss.item(), frames.size(0))
            except Exception as e:
                print(f'[WARN] validate batch failed: {e}')
                mloss.update(1e9, frames.size(0))
    finally:
        # 恢复 BN momentum
        for m, mom in bn_momentum.items():
            m.momentum = mom

    return mloss.avg


@torch.no_grad()
def test(args, model, val_loader):
    """
    多帧模型验证集 mAP 评估。
    与单帧训练 test() 逻辑一致：COCO 风格 mAP@0.5:0.95，BoxAP + PoseAP。
    三帧输入，标注仅绑定 Frame 0。

    使用 autocast(dtype=float16) 做混合精度推理，避免 deepcopy 模型。

    返回: (box_mean_ap, kpt_mean_ap)
    """
    device = next(model.parameters()).device

    # 保存原始 training 状态，验证期间用 eval 模式 + autocast
    was_training = model.training
    model.eval()

    # BN 的 running stats 更新：momentum=0 保持原值不变
    bn_momentum = {}
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_momentum[m] = m.momentum
            m.momentum = 0.0

    try:
        iou_v = torch.linspace(0.5, 0.95, 10, device=device)
        n_iou = iou_v.numel()
        kpt_shape = model.head.kpt_shape

        box_mean_ap = 0.
        kpt_mean_ap = 0.
        box_metrics = []
        kpt_metrics = []

        p_bar = tqdm.tqdm(val_loader, desc=('%10s' * 2) % ('BoxAP', 'PoseAP'))
        for frames, targets in p_bar:
            frames = frames.to(device).float() / 255.0

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(frames)

            # Head 输出格式: train→tuple(list, tensor), eval→tensor
            # 适应所有可能的输出格式
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(outputs, list):
                outputs = torch.cat([o.flatten(2) for o in outputs], dim=2)

            outputs = util.non_max_suppression(
                outputs.float(),  # NMS 需要 float32
                0.001, 0.7, model.head.nc,
            )
            _, _, _, h, w = frames.shape  # (B, 3, 3, H, W)
            scale = torch.tensor((w, h, w, h), device=device)

            for i, output in enumerate(outputs):
                idx = targets['idx'] == i
                cls = targets['cls'][idx]
                t_box = targets['box'][idx]
                t_kpt = targets['kpt'][idx]

                cls = cls.to(device)
                t_box = t_box.to(device)
                t_kpt = t_kpt.to(device)

                correct_box = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=device)
                correct_kpt = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=device)

                if output.shape[0] == 0:
                    if cls.shape[0]:
                        box_metrics.append((correct_box, *torch.zeros((2, 0), device=device), cls.squeeze(-1)))
                        kpt_metrics.append((correct_kpt, *torch.zeros((2, 0), device=device), cls.squeeze(-1)))
                    continue

                pred = output.clone()
                p_kpt = pred[:, 6:].view(output.shape[0], kpt_shape[0], -1)

                if cls.shape[0]:
                    t_box_xyxy = util.wh2xy(t_box)
                    t_kpt_px = t_kpt.clone()
                    t_kpt_px[..., 0] *= w
                    t_kpt_px[..., 1] *= h

                    target = torch.cat((cls, t_box_xyxy * scale), 1)
                    correct_box = util.compute_metric(pred[:, :6], target, iou_v)
                    correct_kpt = util.compute_metric(pred[:, :6], target, iou_v, p_kpt, t_kpt_px)

                box_metrics.append((correct_box, output[:, 4], output[:, 5], cls.squeeze(-1)))
                kpt_metrics.append((correct_kpt, output[:, 4], output[:, 5], cls.squeeze(-1)))

        # 聚合计算
        box_metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*box_metrics)]
        kpt_metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*kpt_metrics)]
        if len(box_metrics) and box_metrics[0].any():
            tp, fp, m_pre, m_rec, map50, box_mean_ap = util.compute_ap(*box_metrics)
        if len(kpt_metrics) and kpt_metrics[0].any():
            tp, fp, m_pre, m_rec, map50, kpt_mean_ap = util.compute_ap(*kpt_metrics)

    finally:
        # 恢复 BN momentum 和 training 状态
        for m, mom in bn_momentum.items():
            m.momentum = mom
        if was_training:
            model.train()

    print('%10.3g' * 2 % (box_mean_ap, kpt_mean_ap))
    return box_mean_ap, kpt_mean_ap


if __name__ == '__main__':
    train(get_args())
