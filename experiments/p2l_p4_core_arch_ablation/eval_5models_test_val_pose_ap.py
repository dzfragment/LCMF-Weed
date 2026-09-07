#!/usr/bin/env python3
"""
5 模型 (00 主模型 / 01 DCN+Spatial / 02 FFT+Spatial / 05 FFT+DCN+Concat / 13 仅Concat)
在 weed_sliding_merged 的 test 和 val 上评估 pose AP (mAP@50:95)，4 种帧模式。
模型构建参数与 viz_5model_compare 完全一致。
"""
import os, sys
from collections import OrderedDict
import numpy as np
import torch
from torch.utils import data

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP1_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # 仓库根
sys.path.insert(0, os.path.join(_EXP1_DIR, 'YOLOv8-pose'))
sys.path.insert(0, os.path.join(_EXP1_DIR, 'multi_frame_fusion'))

from models.mf_yolo import MultiFrameYOLO
from train import apply_ablation
from utils import util

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_ROOT = os.environ.get('RESULTS_ROOT') or os.path.join(_SCRIPT_DIR, 'results')
DS_ROOT = os.environ.get('DATASET_ROOT') or os.path.join(_EXP1_DIR, 'weed_sliding_merged')
INPUT_SIZE = 1024

# ── 5 模型配置 (与 viz_5model_compare 一致) ───────────────────────
MODEL_SPECS = OrderedDict({
    '00_full_fft_dcn_spatial': dict(
        label='FFT+DCN+Spatial (主模型)', gate_type='spatial', gas='feature',
        fft_ds=4, no_align=False, fusion_mode='gate_diff'),
    '01_remove_fft': dict(
        label='DCN+Spatial (去FFT)', gate_type='spatial', gas='none',
        fft_ds=8, no_align=False, fusion_mode='gate_diff'),
    '02_remove_dcn': dict(
        label='FFT+Spatial (去DCN)', gate_type='spatial', gas='feature',
        fft_ds=4, no_align=True, fusion_mode='gate_diff'),
    '05_ablate_concat_fusion': dict(
        label='FFT+DCN+Concat', gate_type='concat', gas='feature',
        fft_ds=4, no_align=False, fusion_mode='gate_diff'),
    '13_remove_fft_dcn_concat': dict(
        label='仅Concat融合', gate_type='concat', gas='none',
        fft_ds=8, no_align=True, fusion_mode='gate_diff'),
})

FRAME_MODES = OrderedDict({
    '1f(key3)':          lambda f: torch.stack([f[:, 0], f[:, 0], f[:, 0]], dim=1),
    '2f(+ref1)':         lambda f: torch.stack([f[:, 0], f[:, 1], f[:, 1]], dim=1),
    '2f(+ref2)':         lambda f: torch.stack([f[:, 0], f[:, 2], f[:, 2]], dim=1),
    '3f(key+r1+r2)':     lambda f: f,
})


class EvalDataset(data.Dataset):
    def __init__(self, ds_root, split):
        self.dirs = sorted([
            os.path.join(ds_root, split, d) for d in os.listdir(os.path.join(ds_root, split))
            if os.path.isdir(os.path.join(ds_root, split, d))
        ])

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        import cv2
        imgs = []
        for j in range(3):
            img = cv2.imread(os.path.join(self.dirs[i], f'frame_{j}.jpg'))
            if img is None:
                raise FileNotFoundError(f'{self.dirs[i]}/frame_{j}.jpg')
            imgs.append(img)
        h, w = imgs[0].shape[:2]
        r = min(INPUT_SIZE / h, INPUT_SIZE / w, 1.0)
        pw, ph = int(round(w * r)), int(round(h * r))
        dw, dh = (INPUT_SIZE - pw) / 2, (INPUT_SIZE - ph) / 2
        samples = []
        for img in imgs:
            if (w, h) != (pw, ph):
                img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_LINEAR)
            t = int(round(dh - 0.1)); b = int(round(dh + 0.1))
            l = int(round(dw - 0.1)); r = int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            samples.append(torch.from_numpy(np.ascontiguousarray(
                img.transpose((2, 0, 1))[::-1])))
        fn = torch.stack(samples, dim=0)
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
        return fn, cls, tb, tk, torch.zeros(nl, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        frames, cls, box, kpt, indices = zip(*batch)
        frames = torch.stack(frames, 0); cls = torch.cat(cls, 0)
        box = torch.cat(box, 0); kpt = torch.cat(kpt, 0)
        ni = list(indices)
        for i in range(len(ni)): ni[i] += i
        return frames, {'cls': cls, 'box': box, 'kpt': kpt, 'idx': torch.cat(ni, 0)}


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
def evaluate_mode(model, dataloader, mode_fn):
    iou_v = torch.linspace(0.5, 0.95, 10, device=DEVICE)
    ni = iou_v.numel()
    ks = model.head.kpt_shape
    bm, km = [], []
    for frames_native, targets in dataloader:
        frames = mode_fn(frames_native.to(DEVICE).float() / 255.0)
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


def build_model(spec):
    ckpt = torch.load(os.path.join(RESULTS_ROOT, spec['ckpt'], 'mf_stage3_best.pt'),
                      map_location=DEVICE, weights_only=False)
    fv = ckpt.get('fusion_version', 2)
    model = MultiFrameYOLO(
        num_classes=1, kpt_shape=(1, 2), fusion_levels=['p4'],
        pretrained_path=None, fusion_version=fv,
        fusion_gate_type=spec['gate_type'],
        global_align_stage=spec['gas'],
        fft_downsample=spec['fft_ds'],
    ).to(DEVICE)
    apply_ablation(model, no_align=spec['no_align'], fusion_mode=spec['fusion_mode'])
    sd = ckpt.get('model')
    if isinstance(sd, dict):
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if ckpt.get('consistent_align', False) and getattr(model, 'global_align_stage', '') == 'feature':
            from models.mf_yolo import enable_consistent_align
            enable_consistent_align(model)
        if missing:
            print(f'    [WARN] missing keys: {len(missing)}')
        if unexpected:
            print(f'    [WARN] unexpected keys: {len(unexpected)}')
    else:
        model = sd.float().to(DEVICE)
    model.eval()
    return model, ckpt.get('epoch', '?'), ckpt.get('best_pose_ap', 0)


def run_split(split):
    dataset = EvalDataset(DS_ROOT, split)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False,
                             num_workers=4, pin_memory=True,
                             collate_fn=EvalDataset.collate_fn)
    print(f'\n{"#" * 100}')
    print(f'# {split} 集: {len(dataset)} 序列')
    print(f'{"#" * 100}')
    results = {}
    for exp_id, spec in MODEL_SPECS.items():
        spec = dict(spec)
        spec['ckpt'] = exp_id
        print(f'\n  [{exp_id}] {spec["label"]}')
        model, epoch, best_ap = build_model(spec)
        print(f'    epoch={epoch}  best_pose_ap(训练val)={best_ap:.4f}')
        mr = OrderedDict()
        for mn, mf in FRAME_MODES.items():
            ba, pa = evaluate_mode(model, loader, mf)
            mr[mn] = (round(ba, 4), round(pa, 4))
        results[spec['label']] = mr
        p1 = mr['1f(key3)'][1]
        p3 = mr['3f(key+r1+r2)'][1]
        print(f'    1f={p1:.4f}  2f+r1={mr["2f(+ref1)"][1]:.4f}  '
              f'2f+r2={mr["2f(+ref2)"][1]:.4f}  3f={p3:.4f} (Δ{p3 - p1:+.4f})')
        del model
        torch.cuda.empty_cache()
    return results


def print_table(results, split):
    print(f'\n{"=" * 120}')
    print(f'=== PoseAP ({split}, mAP@50:95) ===')
    hdr = f'{"模型":<24s} |'
    for m in FRAME_MODES:
        hdr += f' {m:>11s} |'
    hdr += f' {"Δ(3f-1f)":>9s}'
    print(hdr)
    print('-' * len(hdr))
    for label, mr in results.items():
        p1 = mr['1f(key3)'][1]
        p3 = mr['3f(key+r1+r2)'][1]
        row = f'{label:<24s} |'
        for m in FRAME_MODES:
            row += f' {mr[m][1]:11.4f} |'
        row += f' {p3 - p1:+9.4f}'
        print(row)
    print(f'\n=== BoxAP ({split}, mAP@50:95) ===')
    hdr = f'{"模型":<24s} |'
    for m in FRAME_MODES:
        hdr += f' {m:>11s} |'
    print(hdr)
    print('-' * len(hdr))
    for label, mr in results.items():
        row = f'{label:<24s} |'
        for m in FRAME_MODES:
            row += f' {mr[m][0]:11.4f} |'
        print(row)


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    r_test = run_split('test')
    r_val = run_split('val')
    print_table(r_test, 'test')
    print_table(r_val, 'val')
