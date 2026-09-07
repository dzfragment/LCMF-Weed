#!/usr/bin/env python3
"""
单帧关键帧基线 (SF-P2L-P4, weed_sliding_merged_keyframe seed42) 在
weed_public_sliding_v2_test_variants 的 10 个测试集 variant 上的评测。
仅用每个序列的 frame_0 关键帧作为输入。

输出: 每个 variant 的 BoxAP/PoseAP，以及跨 10 个 variant 的均值 ± 标准差。
"""
import os, sys
import numpy as np
import torch
from torch.utils import data

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP2_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # 仓库根
sys.path.insert(0, os.path.join(_EXP2_DIR, 'YOLOv8-pose'))
from nets import nn
from utils import util

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = 1024

# 10 个 public v2 test variant
VARIANTS_ROOT = os.environ.get('PUBLIC_TEST_ROOT', os.path.join(_EXP2_DIR, 'weed_public_sliding_v2_test_variants'))

DEFAULT_CKPT = os.path.join(
    _SCRIPT_DIR, 'results', 'weed_sliding_merged_keyframe',
    'yolo_v8_n_p2_lite_p4', 'seed42', 'best.pt')


class SingleFrameDataset(data.Dataset):
    """读取给定序列根目录下每个序列的 frame_0.jpg 和 labels_frame0.txt"""
    def __init__(self, seq_root):
        self.dirs = sorted([
            os.path.join(seq_root, d) for d in os.listdir(seq_root)
            if os.path.isdir(os.path.join(seq_root, d))
        ])

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        import cv2
        img = cv2.imread(os.path.join(self.dirs[i], 'frame_0.jpg'))
        if img is None:
            raise FileNotFoundError(f'{self.dirs[i]}/frame_0.jpg')
        h, w = img.shape[:2]
        r = min(INPUT_SIZE / h, INPUT_SIZE / w, 1.0)
        pw, ph = int(round(w * r)), int(round(h * r))
        dw, dh = (INPUT_SIZE - pw) / 2, (INPUT_SIZE - ph) / 2

        if (w, h) != (pw, ph):
            img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_LINEAR)
        t = int(round(dh - 0.1)); b = int(round(dh + 0.1))
        l = int(round(dw - 0.1)); r = int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        sample = torch.from_numpy(np.ascontiguousarray(
            img.transpose((2, 0, 1))[::-1]))

        lp = os.path.join(self.dirs[i], 'labels_frame0.txt')
        if os.path.exists(lp):
            with open(lp) as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
            if lines:
                label = np.array([list(map(float, l.split())) for l in lines], dtype=np.float32)
                box, kpt = label[:, :5], label[:, 5:]
            else:
                box = np.zeros((0, 5), dtype=np.float32)
                kpt = np.zeros((0, 0), dtype=np.float32)
        else:
            box = np.zeros((0, 5), dtype=np.float32)
            kpt = np.zeros((0, 0), dtype=np.float32)

        if kpt.size > 0:
            kpt = np.concatenate([kpt, np.full((kpt.shape[0], 1), 2.0,
                                  dtype=np.float32)], axis=1).reshape(-1, 1, 3)
        else:
            kpt = np.zeros((0, 1, 3), dtype=np.float32)

        nl = len(box)
        if nl:
            box[:, 1:] = _wh2xy(box[:, 1:], pw, ph, dw, dh)
            kpt[..., 0] = kpt[..., 0] * pw + dw
            kpt[..., 1] = kpt[..., 1] * ph + dh
            box[:, 1:] = _xy2wh(box[:, 1:], INPUT_SIZE, INPUT_SIZE)
            kpt[..., 0] /= INPUT_SIZE; kpt[..., 1] /= INPUT_SIZE

        cls = torch.from_numpy(box[:, 0:1]).float() if nl else torch.zeros((0, 1))
        tb = torch.from_numpy(box[:, 1:5]).float() if nl else torch.zeros((0, 4))
        tk = torch.from_numpy(kpt).float() if nl else torch.zeros((0, 1, 3))
        return sample, cls, tb, tk, torch.zeros(nl, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, kpt, indices = zip(*batch)
        samples = torch.stack(samples, 0); cls = torch.cat(cls, 0)
        box = torch.cat(box, 0); kpt = torch.cat(kpt, 0)
        ni = list(indices)
        for i in range(len(ni)): ni[i] += i
        return samples, {'cls': cls, 'box': box, 'kpt': kpt, 'idx': torch.cat(ni, 0)}


def _wh2xy(x, w, h, dw, dh):
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + dw
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + dh
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + dw
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + dh
    return y


def _xy2wh(x, w, h):
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1e-3)
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1e-3)
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y


@torch.no_grad()
def evaluate(model, dataloader):
    iou_v = torch.linspace(0.5, 0.95, 10, device=DEVICE)
    ni = iou_v.numel()
    ks = model.head.kpt_shape
    bm, km = [], []
    for samples, targets in dataloader:
        samples = samples.to(DEVICE).half() / 255.0
        outputs = model(samples)
        outputs = util.non_max_suppression(outputs, 0.001, 0.7, model.head.nc)
        _, _, h, w = samples.shape
        scale = torch.tensor((w, h, w, h), device=DEVICE)
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx].to(DEVICE)
            tb = targets['box'][idx].to(DEVICE)
            tk = targets['kpt'][idx].to(DEVICE)
            cb = torch.zeros(output.shape[0], ni, dtype=torch.bool, device=DEVICE)
            ck = torch.zeros(output.shape[0], ni, dtype=torch.bool, device=DEVICE)
            if output.shape[0] == 0:
                if cls.shape[0]:
                    bm.append((cb, *torch.zeros((2, 0), device=DEVICE), cls.squeeze(-1)))
                    km.append((ck, *torch.zeros((2, 0), device=DEVICE), cls.squeeze(-1)))
                continue
            pred = output.clone()
            pk = pred[:, 6:].view(output.shape[0], ks[0], -1)
            if cls.shape[0]:
                target = torch.cat((cls, util.wh2xy(tb) * scale), 1)
                tkp = tk.clone(); tkp[..., 0] *= w; tkp[..., 1] *= h
                cb = util.compute_metric(pred[:, :6], target, iou_v)
                ck = util.compute_metric(pred[:, :6], target, iou_v, pk, tkp)
            bm.append((cb, output[:, 4], output[:, 5], cls.squeeze(-1)))
            km.append((ck, output[:, 4], output[:, 5], cls.squeeze(-1)))

    def ap(ml):
        arr = [torch.cat(x, 0).cpu().numpy() for x in zip(*ml)]
        if len(arr) and arr[0].any():
            *_, a = util.compute_ap(*arr)
            return float(a)
        return 0.0
    return ap(bm), ap(km)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ckpt.get('model')
    if isinstance(model, dict):
        m = nn.yolo_v8_n_p2_lite_p4(num_classes=1, kpt_shape=(1, 2)).to(DEVICE)
        m.load_state_dict(model, strict=False)
        model = m
    else:
        model = model.float().to(DEVICE)
    model.half()
    model.eval()
    return model, ckpt.get('epoch', '?')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', default=DEFAULT_CKPT,
                        help='单帧模型 checkpoint 路径（默认关键帧基线）')
    parser.add_argument('--tag', default='SF-P2L-P4',
                        help='模型标签，用于输出文件名/打印')
    parser.add_argument('--out', default=None,
                        help='结果保存路径（默认 sf_keyframe_public_v2_<tag>.txt）')
    args = parser.parse_args()

    CKPT_PATH = args.ckpt
    MODEL_LABEL = args.tag
    VARIANTS = sorted([
        d for d in os.listdir(VARIANTS_ROOT)
        if os.path.isdir(os.path.join(VARIANTS_ROOT, d))
    ])

    print(f'Device: {DEVICE}')
    print(f'Checkpoint: {CKPT_PATH}')
    print(f'Model: {MODEL_LABEL}')
    print(f'Variants: {len(VARIANTS)}')
    print()

    model, epoch = load_model(CKPT_PATH)
    print(f'Loaded model (epoch={epoch})')
    print()

    results = []
    for v in VARIANTS:
        seq_root = os.path.join(VARIANTS_ROOT, v, 'test')
        dataset = SingleFrameDataset(seq_root)
        loader = data.DataLoader(dataset, batch_size=8, shuffle=False,
                                 num_workers=4, pin_memory=True,
                                 collate_fn=SingleFrameDataset.collate_fn)
        print(f'[{v}] {len(dataset)} 序列, evaluating...')
        box_ap, pose_ap = evaluate(model, loader)
        results.append({'variant': v, 'n': len(dataset), 'BoxAP': box_ap, 'PoseAP': pose_ap})
        print(f'  BoxAP={box_ap:.4f}  PoseAP={pose_ap:.4f}')

    box = np.array([r['BoxAP'] for r in results])
    pose = np.array([r['PoseAP'] for r in results])

    print()
    print('=' * 62)
    print(f'{"Variant":<14s} {"N":>5s} {"BoxAP":>8s} {"PoseAP":>8s}')
    print('-' * 62)
    for r in results:
        print(f'{r["variant"]:<14s} {r["n"]:>5d} {r["BoxAP"]:>8.4f} {r["PoseAP"]:>8.4f}')
    print('-' * 62)
    print(f'{"mean":<14s} {"":>5s} {box.mean():>8.4f} {pose.mean():>8.4f}')
    print(f'{"std":<14s} {"":>5s} {box.std(ddof=1):>8.4f} {pose.std(ddof=1):>8.4f}')
    print(f'{"mean±std":<14s} {"":>5s} {box.mean():.4f}±{box.std(ddof=1):.4f}  {pose.mean():.4f}±{pose.std(ddof=1):.4f}')
    print('=' * 62)

    # 保存
    if args.out is None:
        tag_safe = MODEL_LABEL.replace('/', '_').replace(' ', '_')
        out = os.path.join(_SCRIPT_DIR, f'sf_keyframe_public_v2_{tag_safe}.txt')
    else:
        out = args.out
    with open(out, 'w') as f:
        f.write(f"model: {MODEL_LABEL}\n")
        f.write(f"checkpoint: {CKPT_PATH}\n")
        f.write(f"variants_root: {VARIANTS_ROOT}\n")
        f.write(f"input: frame_0 keyframe only\n")
        f.write(f"variants: {len(VARIANTS)}\n")
        f.write("\nper_variant:\n")
        for r in results:
            f.write(f"  {r['variant']}: n={r['n']} BoxAP={r['BoxAP']:.4f} PoseAP={r['PoseAP']:.4f}\n")
        f.write(f"\nBoxAP  mean±std: {box.mean():.4f} ± {box.std(ddof=1):.4f}\n")
        f.write(f"PoseAP mean±std: {pose.mean():.4f} ± {pose.std(ddof=1):.4f}\n")
    print(f'\nSaved: {out}')
