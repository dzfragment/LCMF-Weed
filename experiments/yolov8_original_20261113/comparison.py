"""Read-only point evaluation shared by official YOLOv8n-pose and the fixed PAD SF."""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(HERE / 'ultralytics_settings'))
os.environ['ULTRALYTICS_OFFLINE'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['COMET_MODE'] = 'DISABLED'
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'YOLOv8-pose'), str(ROOT / 'multi_frame_fusion')]

import hashlib
import json
import math
import pickle
import tempfile
import numpy as np
import torch
import yaml
from utils.dataset import Dataset
from pad_core import PADSingleFrame, _gt_arrays, pad_infer, match_points, point_ap
from ultralytics import YOLO, settings
from ultralytics.utils.nms import non_max_suppression

settings.update({k: False for k in ['sync', 'wandb', 'comet', 'clearml', 'mlflow',
                                  'neptune', 'dvc', 'tensorboard'] if k in settings})
SF = Path(os.environ.get('LCMF_SF_CHECKPOINT',
    HERE.parent / 'single_frame_baseline/results/pad_sf_clean/p2l/best.pt')).resolve()
SF_LAST = Path(os.environ.get('LCMF_SF_LAST_CHECKPOINT', SF.with_name('last.pt'))).resolve()
DATA = Path(os.environ.get('LCMF_DATA_ROOT', ROOT / 'dataset')).resolve() / 'weed_merged_1024_clean'
COCO_WEIGHTS = Path(os.environ.get('LCMF_YOLO_COCO_WEIGHTS', HERE / 'initial/yolov8n-pose.pt')).resolve()


class EvaluationDataset(Dataset):
    """Read exactly the pinned files; historical .cache files contain foreign paths/subsets."""
    @staticmethod
    def load_label(filenames, params):
        labels = {}
        for filename in filenames:
            p = Path(filename)
            label = p.parent.parent / 'labels' / (p.stem + '.txt')
            text = label.read_text().strip()
            a = np.array([list(map(float, line.split())) for line in text.splitlines()], np.float32).reshape(-1, 7)
            assert not len(a) or (np.isfinite(a).all() and (a[:, 0] == 0).all()), label
            if len(a):
                _, indices = np.unique(a, axis=0, return_index=True)
                if len(indices) < len(a):
                    a = a[indices]
            keypoints = a[:, 5:].reshape(-1, 1, 2)
            visible = (keypoints >= 0).all(-1, keepdims=True).astype(np.float32)
            labels[filename] = [a[:, :5], np.concatenate([keypoints, visible], axis=-1)]
        return labels


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = f.name
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def dump(path, value):
    atomic_bytes(path, (json.dumps(clean(value), indent=2, allow_nan=False) + '\n').encode())


def loader(split='val', data=DATA, size=1024, batch=8, workers=4, limit=None):
    params = yaml.safe_load((ROOT / 'YOLOv8-pose/utils/args.yaml').read_text())
    params.update(nc=1, names={0: 'weed'}, kpt_shape=[1, 2], flip_index=[0])
    files = sorted(str(p) for p in (Path(data) / split / 'images').iterdir()
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if limit is not None:
        files = files[:limit]
    ds = EvaluationDataset(files, size, params, False)
    assert ds.filenames == files
    return torch.utils.data.DataLoader(ds, batch, False, num_workers=workers,
        pin_memory=workers > 0, collate_fn=Dataset.collate_fn)


@torch.inference_mode()
def yolo_infer(model, batches, conf=.001):
    model.eval()
    device = next(model.parameters()).device
    preds = {}
    for images, targets in batches:
        images = images.to(device).float() / 255
        h, w = images.shape[-2:]
        detections = non_max_suppression(model(images), conf_thres=conf, iou_thres=.7,
            nc=1, max_det=300, max_time_img=10.)
        assert len(detections) == len(images)
        start = len(preds)
        for i, det in enumerate(detections):
            # Standard NMS: xyxy, score, class, keypoint_x, keypoint_y.
            assert det.shape[1] == 8, det.shape
            arr = det[:, [0, 1, 2, 3, 4, 6, 7]].cpu().numpy().astype(np.float32)
            if len(arr):
                keep = ((arr[:, 5] >= 0) & (arr[:, 5] < w) &
                        (arr[:, 6] >= 0) & (arr[:, 6] < h) &
                        (arr[:, 2] - arr[:, 0] > 1) & (arr[:, 3] - arr[:, 1] > 1))
                arr = arr[keep]
            preds[start + i] = dict(pred=arr, gt=_gt_arrays(targets, i, w, h))
    return preds


def fixed_fp(preds, fp_per_image=1.):
    rows = []
    n_gt = sum(len(d['gt']) for d in preds.values())
    for d in preds.values():
        pr, gt = d['pred'], d['gt']
        taken = set()
        for j in np.argsort(-pr[:, 4]):
            dist = np.hypot(gt[:, 4] - pr[j, 5], gt[:, 5] - pr[j, 6])
            if taken:
                dist[list(taken)] = np.inf
            ok = bool(len(dist) and dist.min() <= 10)
            if ok:
                taken.add(int(dist.argmin()))
            rows.append((float(pr[j, 4]), ok))
    rows.sort(key=lambda x: -x[0])
    tp = fp = 0
    best = dict(point_recall=0., conf=None, n_tp=0, n_fp=0)
    i = 0
    while i < len(rows):
        score = rows[i][0]
        while i < len(rows) and rows[i][0] == score:
            tp += rows[i][1]
            fp += not rows[i][1]
            i += 1
        if fp > fp_per_image * len(preds):
            break
        best = dict(point_recall=tp / max(n_gt, 1), conf=score, n_tp=tp, n_fp=fp)
    return best | dict(fp_per_image_budget=fp_per_image, threshold_scope='test PR diagnostic; not checkpoint selection')


def metrics(preds, ap=True):
    result = match_points(preds, tau_px=10., conf=.25)
    if ap:
        result.update(point_ap_10px=point_ap(preds, 10., .001),
                      point_ap_20px=point_ap(preds, 20., .001),
                      recall_at_fp1=fixed_fp(preds))
    return result


def should_select(metric, previous):
    err, rec = metric['nPE_mean_px'], metric['point_recall']
    if not math.isfinite(err) or rec < .85:
        return False
    if not previous or previous.get('nPE_mean_px') is None:
        return True
    old = previous['nPE_mean_px']
    return err < old - .005 or (abs(err - old) <= .005 and rec > previous['point_recall'] + 1e-6)


def evaluate_checkpoint(kind, checkpoint, split, device='cuda:0'):
    if kind == 'yolo':
        m = YOLO(str(checkpoint)).model.to(device).float().eval()
    else:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        m = PADSingleFrame(ck['base_model'], ck['head_cfg'])
        m.load_state_dict(ck['model'].float().state_dict(), strict=True)
        m = m.to(device).eval()
    batches = loader(split)
    preds = yolo_infer(m, batches) if kind == 'yolo' else pad_infer(m, batches, 1024, .001, 300)
    result = metrics(preds) | dict(model=kind, split=split, checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint), data=str(DATA), n_images=len(preds),
        radius_px=10, conf=.25, conf_floor=.001, max_candidates=300,
        parameters=sum(p.numel() for p in m.parameters()), inference_precision='fp32 from fp16 EMA checkpoint')
    del m
    torch.cuda.empty_cache()
    return result, preds


@torch.inference_mode()
def profile_checkpoint(kind, checkpoint):
    import time
    from copy import deepcopy
    import thop
    if kind == 'yolo':
        model = YOLO(str(checkpoint)).model.to('cuda:0').float().eval()
    else:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = PADSingleFrame(ck['base_model'], ck['head_cfg'])
        model.load_state_dict(ck['model'].float().state_dict(), strict=True)
        model = model.to('cuda:0').eval()
    x = torch.zeros(1, 3, 1024, 1024, device='cuda:0')
    copy = deepcopy(model)
    macs, _ = thop.profile(copy, inputs=(x,), verbose=False)
    del copy
    for _ in range(30):
        model(x)
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        started = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000)
    result = dict(model=kind, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        parameters=sum(p.numel() for p in model.parameters()), input=[1, 3, 1024, 1024],
        macs=float(macs), gflops_2mac=float(2 * macs / 1e9), forward_ms_mean=float(np.mean(times)),
        forward_ms_p95=float(np.percentile(times, 95)), precision='fp32', fused=False,
        warmup=30, repetitions=100, excludes='preprocessing, decode, NMS, data loading',
        device=torch.cuda.get_device_name(0))
    del model, x
    torch.cuda.empty_cache()
    return result


def final_comparison(best, last):
    records = []
    for kind, checkpoints in [('pad', [('best', SF), ('last', SF_LAST)]),
                              ('yolo', [('best', best), ('last', last)])]:
        for tag, checkpoint in checkpoints:
            for split in ['val', 'test']:
                stem = f'{kind}_{split}_{tag}'
                path = HERE / 'evaluation' / (stem + '.json')
                if path.exists() and json.loads(path.read_text())['checkpoint_sha256'] == digest(checkpoint):
                    record = json.loads(path.read_text())
                else:
                    record, preds = evaluate_checkpoint(kind, checkpoint, split)
                    record['tag'] = tag
                    atomic_bytes(HERE / 'evaluation' / (stem + '_preds.pkl'), pickle.dumps(preds))
                    dump(path, record)
                records.append(record)
    dump(HERE / 'summary.json', records)
    profiles = []
    for kind, ck in [('pad', SF), ('yolo', best)]:
        path = HERE / 'evaluation' / f'{kind}_profile.json'
        if path.exists() and json.loads(path.read_text())['checkpoint_sha256'] == digest(ck):
            profile = json.loads(path.read_text())
        else:
            profile = profile_checkpoint(kind, ck)
            dump(path, profile)
        profiles.append(profile)
    dump(HERE / 'profile.json', profiles)
    lines = ['# 单帧 P2L+PAD 与官方 YOLOv8n-pose', '',
             '相同 clean 数据：1696/398/334；seed42，300轮。以下为 test334，点匹配半径10px，conf=.25。', '',
             '| 模型 | 权重 | R@10 | AP10 | 平均匹配误差 px | 参数 |',
             '|---|---|---:|---:|---:|---:|']
    for r in records:
        if r['split'] == 'test':
            def fmt(v, digits):
                return 'NA' if v is None or not math.isfinite(v) else f'{v:.{digits}f}'
            lines.append(f"| {r['model']} | {r['tag']} | {fmt(r['point_recall'],4)} | {fmt(r['point_ap_10px'],4)} | {fmt(r['nPE_mean_px'],3)} | {r['parameters']:,} |")
    lines += ['', '| 模型 | GFLOPs (2×MACs) | FP32 batch1 forward ms | p95 ms |', '|---|---:|---:|---:|']
    for r in profiles:
        lines.append(f"| {r['model']} | {r['gflops_2mac']:.3f} | {r['forward_ms_mean']:.3f} | {r['forward_ms_p95']:.3f} |")
    lines += ['', 'PAD 为既有权重，YOLO 为新训；两个模型实现、损失和 COCO 权重迁移覆盖率不同，因此这是完整模型对照，不能将全部差异归因于某一个模块。',
              '主 best 按验证集 R@10>=.85 后最小点误差选择；若 YOLO 全程未达阈值，则按预先规则使用 last。官方原生按 fitness 选择的 best.pt 单独保留。',
              '固定 FP/图的召回见 summary.json；从测试 PR 曲线读取，仅作诊断，不用于选权重。']
    atomic_bytes(HERE / 'RESULTS.md', ('\n'.join(lines) + '\n').encode())
    return records
