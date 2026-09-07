#!/usr/bin/env python3
"""Test completed fusion models on test set (does not stop training)."""
import os, sys
import numpy as np
import torch, tqdm
from torch.utils import data

_EXP2_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根
sys.path.insert(0, os.path.join(_EXP2_DIR, 'YOLOv8-pose'))
sys.path.insert(0, os.path.join(_EXP2_DIR, 'multi_frame_fusion'))

from models.mf_yolo import MultiFrameYOLO
from utils import util

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAINED = os.environ.get('PRETRAINED') or os.path.join(
    _EXP2_DIR, 'experiments', 'single_frame_baseline', 'results', 'weed_merged_1024',
    'yolo_v8_n_p2_lite_p4', 'seed42', 'best.pt')
try:
    DS_ROOT = os.environ['DATASET_ROOT']
except KeyError:
    raise SystemExit('请设置环境变量 DATASET_ROOT（weed_sliding_merged 数据集根目录）')
INPUT_SIZE = 1024

EXPERIMENTS = [
    ('01_fuse_p4', 'P4 only', ['p4']),
    ('02_fuse_p3', 'P3 only', ['p3']),
    ('03_fuse_p3_p4', 'P3+P4', ['p3', 'p4']),
]


class EvalDataset(data.Dataset):
    def __init__(self, split='test'):
        self.split_dir = os.path.join(DS_ROOT, split)
        self.seq_dirs = sorted([
            os.path.join(self.split_dir, d)
            for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        ])
        self._load_labels()
        print(f'  {split}: {len(self.seq_dirs)} sequences')

    def _load_labels(self):
        self.boxes, self.kpts = [], []
        for seq_dir in self.seq_dirs:
            label_path = os.path.join(seq_dir, 'labels_frame0.txt')
            if os.path.exists(label_path):
                with open(label_path) as f:
                    lines = [x.strip() for x in f.readlines() if x.strip()]
                if lines:
                    label = np.array([list(map(float, l.split())) for l in lines], dtype=np.float32)
                    box = label[:, :5]
                    kpt = label[:, 5:]
                else:
                    box = np.zeros((0, 5), dtype=np.float32)
                    kpt = np.zeros((0, 0), dtype=np.float32)
            else:
                box = np.zeros((0, 5), dtype=np.float32)
                kpt = np.zeros((0, 0), dtype=np.float32)
            nk = 1
            if kpt.size > 0:
                vis = np.full((kpt.shape[0], 1), 2.0, dtype=np.float32)
                kpt = np.concatenate([kpt, vis], axis=1).reshape(-1, nk, 3)
            else:
                kpt = np.zeros((0, nk, 3), dtype=np.float32)
            self.boxes.append(box)
            self.kpts.append(kpt)

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, index):
        import cv2
        seq_dir = self.seq_dirs[index]
        images = []
        for i in range(3):
            img_path = os.path.join(seq_dir, f'frame_{i}.jpg')
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f'Cannot read {img_path}')
            images.append(img)

        h, w = images[0].shape[:2]
        r = min(INPUT_SIZE / h, INPUT_SIZE / w)
        r = min(r, 1.0)
        pad_w = int(round(w * r))
        pad_h = int(round(h * r))
        dw = (INPUT_SIZE - pad_w) / 2
        dh = (INPUT_SIZE - pad_h) / 2

        samples = []
        for img in images:
            if (w, h) != (pad_w, pad_h):
                img = cv2.resize(img, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
            top = int(round(dh - 0.1))
            bottom = int(round(dh + 0.1))
            left = int(round(dw - 0.1))
            right = int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
            s = img.transpose((2, 0, 1))[::-1]
            s = np.ascontiguousarray(s)
            samples.append(torch.from_numpy(s))

        frames_native = torch.stack(samples, dim=0)

        box = self.boxes[index].copy()
        kpt = self.kpts[index].copy()
        nl = len(box)
        if nl:
            box[:, 1:] = self._wh2xy(box[:, 1:], pad_w, pad_h, dw, dh)
            kpt[..., 0] = kpt[..., 0] * pad_w + dw
            kpt[..., 1] = kpt[..., 1] * pad_h + dh
            box[:, 1:] = self._xy2wh(box[:, 1:], INPUT_SIZE, INPUT_SIZE)
            kpt[..., 0] /= INPUT_SIZE
            kpt[..., 1] /= INPUT_SIZE

        cls = torch.from_numpy(box[:, 0:1]).float() if nl else torch.zeros((0, 1))
        target_box = torch.from_numpy(box[:, 1:5]).float() if nl else torch.zeros((0, 4))
        target_kpt = torch.from_numpy(kpt).float() if nl else torch.zeros((0, 1, 3))
        return frames_native, cls, target_box, target_kpt, torch.zeros(nl, dtype=torch.long)

    @staticmethod
    def _wh2xy(x, w, h, dw, dh):
        y = np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + dw
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + dh
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + dw
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + dh
        return y

    @staticmethod
    def _xy2wh(x, w, h):
        x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1e-3)
        x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1e-3)
        y = np.copy(x)
        y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
        y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
        y[:, 2] = (x[:, 2] - x[:, 0]) / w
        y[:, 3] = (x[:, 3] - x[:, 1]) / h
        return y

    @staticmethod
    def collate_fn(batch):
        frames, cls, box, kpt, indices = zip(*batch)
        frames = torch.stack(frames, 0)
        cls = torch.cat(cls, 0)
        box = torch.cat(box, 0)
        kpt = torch.cat(kpt, 0)
        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, 0)
        targets = {'cls': cls, 'box': box, 'kpt': kpt, 'idx': indices}
        return frames, targets


def load_model(ckpt_path, fusion_levels):
    print(f'  Loading: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = MultiFrameYOLO(
        num_classes=1, kpt_shape=(1, 2),
        fusion_levels=fusion_levels,
        pretrained_path=PRETRAINED,
        fusion_version=2,
        fusion_gate_type='spatial',
        global_align_stage='feature',
        fft_downsample=4,
    )
    sd = ckpt.get('model')
    if isinstance(sd, dict):
        model.load_state_dict(sd, strict=False)
        if ckpt.get('consistent_align', False) and getattr(model, 'global_align_stage', '') == 'feature':
            from models.mf_yolo import enable_consistent_align
            enable_consistent_align(model)
    model = model.to(DEVICE).eval()
    return model


@torch.no_grad()
def evaluate(model, loader, desc='eval'):
    iou_v = torch.linspace(0.5, 0.95, 10, device=DEVICE)
    n_iou = iou_v.numel()
    kpt_shape = model.head.kpt_shape
    box_metrics, kpt_metrics = [], []

    pbar = tqdm.tqdm(loader, desc=f'{desc:>25s}')
    for frames_native, targets in pbar:
        frames = frames_native.to(DEVICE).float() / 255.0

        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = model(frames)

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(outputs, list):
            outputs = torch.cat([o.flatten(2) for o in outputs], dim=2)

        outputs = util.non_max_suppression(outputs.float(), 0.001, 0.7, model.head.nc)
        _, _, _, h, w = frames.shape
        scale = torch.tensor((w, h, w, h), device=DEVICE)

        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx].to(DEVICE)
            t_box = targets['box'][idx].to(DEVICE)
            t_kpt = targets['kpt'][idx].to(DEVICE)

            correct_box = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=DEVICE)
            correct_kpt = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=DEVICE)

            if output.shape[0] == 0:
                if cls.shape[0]:
                    box_metrics.append((correct_box, *torch.zeros((2, 0), device=DEVICE), cls.squeeze(-1)))
                    kpt_metrics.append((correct_kpt, *torch.zeros((2, 0), device=DEVICE), cls.squeeze(-1)))
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

    def compute_ap(metrics_list):
        arr = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics_list)]
        if len(arr) and arr[0].any():
            *_, ap = util.compute_ap(*arr)
            return ap
        return 0.0

    return compute_ap(box_metrics), compute_ap(kpt_metrics)


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    print(f'Dataset: {DS_ROOT}')
    print()

    dataset = EvalDataset('test')
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False,
                             num_workers=4, pin_memory=True,
                             collate_fn=EvalDataset.collate_fn)

    results = {}
    for exp_id, label, fusion_levels in EXPERIMENTS:
        ckpt = os.path.join(_EXP2_DIR, 'results_concat', exp_id,
                'mf_stage3_best.pt')
        print(f'\n{"="*50}')
        print(f'Testing: {exp_id} ({label}), fusion_levels={fusion_levels}')
        model = load_model(ckpt, fusion_levels)
        box_ap, pose_ap = evaluate(model, loader, f'{exp_id} test')
        results[exp_id] = (label, box_ap, pose_ap)
        print(f'  BoxAP: {box_ap:.4f}  PoseAP: {pose_ap:.4f}')

    print(f'\n{"="*60}')
    print(f'{"Test Set Summary":^60s}')
    print(f'{"="*60}')
    print(f'{"Experiment":<20s} {"BoxAP":>10s} {"PoseAP":>10s}')
    print(f'{"-"*60}')
    p4_pose = None
    for exp_id, (label, box_ap, pose_ap) in results.items():
        if exp_id == '01_fuse_p4':
            p4_pose = pose_ap
        delta = f'{pose_ap - p4_pose:+.4f}' if p4_pose is not None else '-'
        print(f'{label:<20s} {box_ap:10.4f} {pose_ap:10.4f}')
    if p4_pose is not None:
        print(f'\n  ΔPoseAP vs P4-only:')
        for exp_id, (label, box_ap, pose_ap) in results.items():
            if exp_id != '01_fuse_p4':
                print(f'  {label}: {pose_ap - p4_pose:+.4f}')
    print(f'{"="*60}')
