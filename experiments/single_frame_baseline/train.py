#!/usr/bin/env python3
"""
EXP-10: Single-frame YOLO backbone ablation — unified training script.
Supports all existing model variants with multi-seed training.

Usage:
    python train.py --model yolo_v8_n          # SF-ORIG (P3/P4/P5)
    python train.py --model yolo_v8_n_p3       # P3 only
    python train.py --model yolo_v8_n_p34      # SF-P34
    python train.py --model yolo_v8_n_p34_lite # SF-P34L
    python train.py --model yolo_v8_n_p2       # P2 backbone + P3/P4 heads
    python train.py --model yolo_v8_n_p2_lite  # P2 lite + P3/P4 heads
    python train.py --model yolo_v8_n_p2_lite_p4  # P2 lite + P4 head only
    python train.py --model all --seed 42      # train all models with seed 42
"""

import argparse
import copy
import csv
import os
import sys
import time
import warnings

import numpy
import torch
import tqdm
import yaml

# Add YOLOv8-pose to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'YOLOv8-pose'))
from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(PROJECT_ROOT))  # 仓库根
YOLO_DIR = os.path.join(REPO_ROOT, 'YOLOv8-pose')
PRETRAINED = os.path.join(YOLO_DIR, 'v8_n_pose.pt')  # 可选：COCO 预训练主干，不存在时从零训练

# 数据集注册表。默认查找仓库外的 datasets/ 目录（与仓库同级），
# 也可通过 DATASETS_ROOT 环境变量指向任意位置；目录名需与 key 一致。
DATASETS_ROOT = os.environ.get('DATASETS_ROOT') or os.path.join(os.path.dirname(REPO_ROOT), 'datasets')
DATASET_REGISTRY = {
    'weed1024': os.path.join(DATASETS_ROOT, 'weed1024'),
    'weed_merged_1024': os.path.join(DATASETS_ROOT, 'weed_merged_1024'),
    # 3帧数据集仅关键帧版（frame_0），由 convert_keyframe_dataset.py 生成，
    # 用于训练与多帧模型对比的单帧关键帧基线。
    'weed_sliding_merged_keyframe': os.path.join(DATASETS_ROOT, 'weed_sliding_merged_keyframe'),
}

# ── Model registry ───────────────────────────────────────────────
MODEL_REGISTRY = {
    'yolo_v8_n': {
        'name': 'SF-ORIG',
        'description': 'Full YOLO P3/P4/P5 backbone + FPN + P3/P4/P5 heads',
        'builder': lambda: nn.yolo_v8_n(num_classes=1, kpt_shape=(1, 2)),
    },
    'yolo_v8_n_p3': {
        'name': 'SF-P3',
        'description': 'Full P3/P4/P5 backbone + FPN, P3 head only',
        'builder': lambda: nn.yolo_v8_n_p3(num_classes=1, kpt_shape=(1, 2)),
    },
    'yolo_v8_n_p34': {
        'name': 'SF-P34',
        'description': 'Full P3/P4/P5 backbone + FPN, P3/P4 heads (no P5 head)',
        'builder': lambda: nn.yolo_v8_n_p34(num_classes=1, kpt_shape=(1, 2)),
    },
    'yolo_v8_n_p34_lite': {
        'name': 'SF-P34L',
        'description': 'P3/P4 backbone + lite FPN, P3/P4 heads (no P5 at all)',
        'builder': lambda: nn.yolo_v8_n_p34_lite(num_classes=1, kpt_shape=(1, 2)),
    },
    'yolo_v8_n_p2': {
        'name': 'SF-P2',
        'description': 'P2/P3/P4/P5 backbone + P2 FPN, P3/P4 heads',
        'builder': lambda: nn.yolo_v8_n_p2(num_classes=1, kpt_shape=(1, 2)),
    },
    'yolo_v8_n_p2_lite': {
        'name': 'SF-P2L',
        'description': 'P2/P3/P4 backbone + lite FPN (no P5), P3/P4 heads',
        'builder': lambda: nn.yolo_v8_n_p2_lite(num_classes=1, kpt_shape=(1, 2)),
    },
    'yolo_v8_n_p2_lite_p4': {
        'name': 'SF-P2L-P4',
        'description': 'P2/P3/P4 backbone + lite FPN (no P5), P4 head only',
        'builder': lambda: nn.yolo_v8_n_p2_lite_p4(num_classes=1, kpt_shape=(1, 2)),
    },
}

ALL_MODELS = list(MODEL_REGISTRY.keys())


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
        'mosaic': 0.5, 'mix_up': 0.1, 'degrees': 10.0, 'scale': 0.5, 'shear': 0.0,
        'translate': 0.1, 'flip_ud': 0.0, 'flip_lr': 0.5,
        'lr0': args.lr0, 'lrf': 0.01, 'epochs': args.epochs,
    })
    return params


def train(args):
    model_key = args.model
    seed = args.seed
    info = MODEL_REGISTRY[model_key]
    data_dir = args.data_dir
    dataset_name = os.path.basename(data_dir)

    # ── Set seed ─────────────────────────────────────────────
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── Output dir ───────────────────────────────────────────
    out_dir = os.path.join(RESULTS_DIR, dataset_name, model_key, f'seed{seed}')
    os.makedirs(out_dir, exist_ok=True)

    # ── Model ────────────────────────────────────────────────
    model = info['builder']().cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {info['name']} ({model_key})")
    print(f"Description: {info['description']}")
    print(f"Parameters: {n_params:,}  Seed: {seed}")

    # Load pretrained backbone from v8_n_pose.pt
    if args.resume and os.path.exists(f'{out_dir}/last.pt'):
        ckpt = torch.load(f'{out_dir}/last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'].float().state_dict(), strict=True)
        print(f"Resumed from {out_dir}/last.pt")
    else:
        pretrained_path = args.pretrained if args.pretrained else PRETRAINED
        if os.path.exists(pretrained_path):
            pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=False)['model'].float().state_dict()
            model_dict = model.state_dict()
            loaded = {}
            for k, v in pretrained.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    loaded[k] = v
            model.load_state_dict(state_dict=loaded, strict=False)
            if args.pretrained is None:
                # 仅默认 COCO 预训练时重置 cls bias；自定义基座 fine-tune 时保留已学 bias
                for m in model.head.cls:
                    m[-1].bias.data[:] = -1.5
                print(f"Loaded {len(loaded)} pretrained layers, cls bias=-1.5")
            else:
                print(f"Loaded {len(loaded)} pretrained layers from {pretrained_path} (保留原 bias)")
        else:
            print(f"WARNING: Pretrained weights not found at {pretrained_path}")

    params = build_params(args)

    # ── Optimizer ────────────────────────────────────────────
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    p0, p1, p2 = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p2.append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p1.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p0.append(v.weight)

    optimizer = torch.optim.SGD(p2, params['lr0'], params['momentum'], nesterov=True)
    optimizer.add_param_group({'params': p0, 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': p1})
    del p0, p1, p2

    def lr_fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    # Resume state
    start_epoch = 0
    best = 0
    if args.resume and os.path.exists(f'{out_dir}/step.csv'):
        with open(f'{out_dir}/step.csv', 'r') as rf:
            rows = list(csv.DictReader(rf))
            start_epoch = len(rows)
            best = max(float(r['PoseAP']) for r in rows)
        print(f"Resume: start_epoch={start_epoch}, best PoseAP={best:.4f}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn, last_epoch=-1)
    for _ in range(start_epoch):
        scheduler.step()
    ema = util.EMA(model) if args.local_rank == 0 else None

    # ── Data ─────────────────────────────────────────────────
    train_files = build_file_list(data_dir, 'train')
    val_files = build_file_list(data_dir, 'val')
    print(f"Train: {len(train_files)}  Val: {len(val_files)}")

    dataset = Dataset(train_files, args.input_size, params, True)
    sampler = None
    if args.world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    loader = torch.utils.data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                                         num_workers=4, pin_memory=True, collate_fn=Dataset.collate_fn)

    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # ── Training loop ────────────────────────────────────────
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 100)

    csv_mode = 'a' if (args.resume and start_epoch > 0) else 'w'
    with open(f'{out_dir}/step.csv', csv_mode) as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'BoxAP', 'PoseAP'])
        if csv_mode == 'w':
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            model.train()
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            m_loss = util.AverageMeter()
            if args.world_size > 1:
                sampler.set_epoch(epoch)
            p_bar = enumerate(loader)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                p_bar = tqdm.tqdm(p_bar, total=num_batch)

            optimizer.zero_grad()

            for i, (samples, targets) in p_bar:
                x_step = i + num_batch * epoch
                samples = samples.cuda().float() / 255

                if x_step <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, numpy.interp(x_step, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params['warmup_bias_lr'], y['initial_lr'] * lr_fn(epoch)]
                        else:
                            fp = [0.0, y['initial_lr'] * lr_fn(epoch)]
                        y['lr'] = numpy.interp(x_step, xp, fp)
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = numpy.interp(x_step, xp, fp)

                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                loss = criterion(outputs, targets)

                m_loss.update(loss.item(), samples.size(0))
                loss *= args.batch_size * args.world_size

                amp_scale.scale(loss).backward()

                if x_step % accumulate == 0:
                    amp_scale.unscale_(optimizer)
                    util.clip_gradients(model)
                    amp_scale.step(optimizer)
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                if args.local_rank == 0:
                    p_bar.set_description(
                        ('%10s' + '%10.4g') % (f'{epoch + 1}/{args.epochs}', m_loss.avg))
                del loss, outputs

            scheduler.step()

            if args.local_rank == 0:
                box_ap, pose_ap = evaluate(args, params, val_files, ema.ema)
                writer.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'BoxAP': str(f'{box_ap:.3f}'),
                                 'PoseAP': str(f'{pose_ap:.3f}')})
                f.flush()

                if pose_ap > best:
                    best = pose_ap

                ckpt = {'model': copy.deepcopy(ema.ema).half(),
                        'epoch': epoch + 1,
                        'best_pose_ap': best,
                        'seed': seed}
                torch.save(ckpt, f'{out_dir}/last.pt')
                if best == pose_ap:
                    torch.save(ckpt, f'{out_dir}/best.pt')
                del ckpt

    if args.local_rank == 0:
        util.strip_optimizer(f'{out_dir}/best.pt')
        util.strip_optimizer(f'{out_dir}/last.pt')

    torch.cuda.empty_cache()
    print(f"\nTraining complete. Best val PoseAP: {best:.4f}")

    # ── Save model info ──────────────────────────────────────
    if args.local_rank == 0:
        with open(f'{out_dir}/model_info.txt', 'w') as f:
            f.write(f"model_key: {model_key}\n")
            f.write(f"name: {info['name']}\n")
            f.write(f"description: {info['description']}\n")
            f.write(f"params: {n_params}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"epochs: {args.epochs}\n")
            f.write(f"best_val_pose_ap: {best:.4f}\n")

    return best


@torch.no_grad()
def evaluate(args, params, file_list, model=None):
    numpy.random.shuffle(file_list)
    dataset = Dataset(file_list, args.input_size, params, False)
    loader = torch.utils.data.DataLoader(dataset, 4, False, num_workers=4,
                                         pin_memory=True, collate_fn=Dataset.collate_fn)

    if model is None:
        model = torch.load(f'{args.out_dir}/best.pt', map_location='cuda', weights_only=False)['model'].float()

    model.half()
    model.eval()

    iou_v = torch.linspace(0.5, 0.95, 10).cuda()
    n_iou = iou_v.numel()
    box_metrics, kpt_metrics = [], []

    for samples, targets in tqdm.tqdm(loader, desc=('%10s' * 2) % ('BoxAP', 'PoseAP')):
        samples = samples.cuda().half() / 255
        _, _, h, w = samples.shape
        scale = torch.tensor((w, h, w, h)).cuda()
        outputs = model(samples)
        outputs = util.non_max_suppression(outputs, 0.001, 0.7, model.head.nc)

        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx].cuda()
            box = targets['box'][idx].cuda()
            kpt = targets['kpt'][idx].cuda()

            correct_box = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()
            correct_kpt = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    box_metrics.append((correct_box, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                    kpt_metrics.append((correct_kpt, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue

            pred = output.clone()
            p_kpt = pred[:, 6:].view(output.shape[0], model.head.kpt_shape[0], -1)

            if cls.shape[0]:
                t_box = util.wh2xy(box)
                t_kpt = kpt.clone()
                t_kpt[..., 0] *= w
                t_kpt[..., 1] *= h
                target = torch.cat((cls, t_box * scale), 1)
                correct_box = util.compute_metric(pred[:, :6], target, iou_v)
                correct_kpt = util.compute_metric(pred[:, :6], target, iou_v, p_kpt, t_kpt)

            box_metrics.append((correct_box, output[:, 4], output[:, 5], cls.squeeze(-1)))
            kpt_metrics.append((correct_kpt, output[:, 4], output[:, 5], cls.squeeze(-1)))

    box_ap = kpt_ap = 0.0
    box_metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*box_metrics)]
    kpt_metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*kpt_metrics)]
    if len(box_metrics) and box_metrics[0].any():
        _, _, _, _, _, box_ap = util.compute_ap(*box_metrics)
    if len(kpt_metrics) and kpt_metrics[0].any():
        _, _, _, _, _, kpt_ap = util.compute_ap(*kpt_metrics)

    print('%10.3g' * 2 % (box_ap, kpt_ap))
    model.float()
    return box_ap, kpt_ap


@torch.no_grad()
def test_final(args):
    """Final test-set evaluation for a trained model."""
    model_key = args.model
    seed = args.seed
    data_dir = args.data_dir
    dataset_name = os.path.basename(data_dir)
    info = MODEL_REGISTRY[model_key]
    out_dir = os.path.join(RESULTS_DIR, dataset_name, model_key, f'seed{seed}')

    print("\n" + "=" * 60)
    print(f"TEST SET: {info['name']} ({model_key}) seed={seed}  dataset={dataset_name}")
    print("=" * 60)
    test_files = build_file_list(data_dir, 'test')
    print(f"Test images: {len(test_files)}")

    model = torch.load(f'{out_dir}/best.pt', map_location='cuda', weights_only=False)['model'].float()
    box_ap, pose_ap = evaluate(args, build_params(args), test_files, model)

    print(f"\nTest Box  mAP@0.5:0.95: {box_ap:.4f}")
    print(f"Test Pose mAP@0.5:0.95: {pose_ap:.4f}")

    with open(f'{out_dir}/test_result.txt', 'w') as f:
        f.write(f"model: {info['name']} ({model_key})\n")
        f.write(f"seed: {seed}\n")
        f.write(f"Box_mAP: {box_ap:.4f}\n")
        f.write(f"Pose_mAP: {pose_ap:.4f}\n")

    return box_ap, pose_ap


def main():
    parser = argparse.ArgumentParser(description='EXP-10 Single-frame YOLO backbone ablation')
    parser.add_argument('--model', default='yolo_v8_n',
                        choices=ALL_MODELS + ['all'],
                        help='Model variant to train')
    parser.add_argument('--data', default='weed1024',
                        choices=list(DATASET_REGISTRY.keys()),
                        help='Dataset registry key (see DATASET_REGISTRY)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--input-size', default=1024, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--pretrained', default=None,
                        help='自定义预训练权重（默认 COCO v8_n_pose.pt）。'
                             'fine-tune 多帧基座时传入其 best.pt')
    parser.add_argument('--lr0', default=0.01, type=float,
                        help='初始学习率（对齐多帧 Stage3 fine-tune 用 1e-4）')
    parser.add_argument('--smoke', action='store_true', help='Smoke test: 2 epochs only')
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 2

    args.data_dir = DATASET_REGISTRY[args.data]
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    # ── Results dir scoped by dataset ─────────────────────────
    global RESULTS_DIR
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    util.setup_multi_processes()

    models_to_train = ALL_MODELS if args.model == 'all' else [args.model]

    for model_key in models_to_train:
        args.model = model_key
        dataset_name = os.path.basename(args.data_dir)
        info = MODEL_REGISTRY[model_key]
        out_dir = os.path.join(RESULTS_DIR, dataset_name, model_key, f'seed{args.seed}')
        args.out_dir = out_dir

        if args.local_rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Training: {info['name']} ({model_key}) | Seed: {args.seed}"
                  f" | Dataset: {dataset_name} | Epochs: {args.epochs}")
            print(f"Output: {out_dir}")
            print(f"{'=' * 60}")

        if args.test_only:
            test_final(args)
        else:
            train(args)
            if args.local_rank == 0:
                test_final(args)


if __name__ == "__main__":
    import random
    main()
