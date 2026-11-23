"""Measure Exp4 model robustness to reproducible blur on real three-frame data."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import batched
import math
import os
from pathlib import Path
import pickle
import re
import sys

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
DATA = Path(os.environ.get('LCMF_REAL_TEST_ROOT',
    Path(os.environ.get('LCMF_DATA_ROOT', HERE.parents[1] / 'dataset')) / 'real_testset')).resolve()
ABLATION = EXPERIMENTS / 'concat_ablation12_20261106'
YOLO_SUITE = EXPERIMENTS / 'yolov8_original_20261113'
MAIN_CKPT = ABLATION / 'results/01_main/pad_stage3_last.pt'
YOLO_CKPT = YOLO_SUITE / 'training/seed42/weights/best.pt'
SIZE = 1024
SEED = 20261119
REF_PROBABILITY = .5
CONDITIONS = [('original', 0), *[('gaussian', s) for s in (2, 4, 8, 12)],
              *[('motion', s) for s in (7, 15, 29, 43)]]

sys.path.insert(0, str(ABLATION))
from common import ARMS, architecture_flags, args_for, assert_checkpoint, digest, trainer


@dataclass(frozen=True)
class Sequence:
    name: str
    cohort: str
    images: tuple
    labels: np.ndarray
    manifest: dict


def read_sequence(folder):
    if folder.name.startswith('group_'):
        cohort = 'group'
        paths = [folder / f'{i:03d}.png' for i in (1, 2, 3)]
    else:
        cohort = 'current'
        paths = []
        for pattern in ('00_current_*.png', '01_prev_n_*.png', '02_prev_2n_*.png'):
            matches = list(folder.glob(pattern))
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one {pattern} in {folder}')
            paths.append(matches[0])
    label_path = paths[0].with_suffix('.txt')
    rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
    labels = np.array(rows, dtype=np.float32) if rows else np.empty((0, 7), dtype=np.float32)
    if (labels.shape[1] != 7 or not np.isfinite(labels).all()
            or (labels[:, 0] != 0).any() or ((labels[:, 1:] < 0) | (labels[:, 1:] > 1)).any()
            or (labels[:, 3:5] <= 0).any()):
        raise ValueError(f'Invalid label rows in {label_path}')

    images = tuple(cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths)
    if any(image is None or image.shape != (SIZE, SIZE, 3) for image in images):
        raise ValueError(f'Expected three {SIZE}×{SIZE} images in {folder}')
    hashes = [digest(path) for path in paths]
    # Identical source key frames receive identical draws, including across directories.
    rng = np.random.default_rng(np.random.SeedSequence([SEED, int(hashes[0][:16], 16)]))
    draw = float(rng.random())
    selected_ref = int(rng.integers(1, 3)) if draw < REF_PROBABILITY else None
    angle = int(rng.integers(0, 180))
    meta = dict(sequence=folder.name, cohort=cohort, n_gt=len(labels),
                frames=[path.name for path in paths], frame_sha256=hashes,
                labels=label_path.name, label_sha256=digest(label_path),
                reference_draw=draw, blurred_reference_slot=selected_ref, motion_angle_deg=angle)
    return Sequence(folder.name, cohort, images, labels, meta)


def motion_kernel(length, angle, *, exact_travel=False):
    """Rasterize a uniform line. Legacy conditions specify the kernel width."""
    half_travel = length / 2 if exact_travel else length // 2
    radius = math.ceil(half_travel)
    kernel_size = 2 * radius + 1
    t = np.linspace(-half_travel, half_travel, 8 * int(2 * half_travel) + 1)
    radians = math.radians(angle)
    x, y = radius + t * math.cos(radians), radius + t * math.sin(radians)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    for xi in (x0, x0 + 1):
        for yi in (y0, y0 + 1):
            weights = (1 - np.abs(x - xi)) * (1 - np.abs(y - yi))
            valid = (xi >= 0) & (xi < kernel_size) & (yi >= 0) & (yi < kernel_size)
            np.add.at(kernel, (yi[valid], xi[valid]), weights[valid])
    kernel = (kernel + kernel[::-1, ::-1]) * .5
    return (kernel / kernel.sum()).astype(np.float32)


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def prepare(sequence, kind, strength):
    frames = list(sequence.images)
    mask = [False, False, False]
    if kind != 'original':
        ref = sequence.manifest['blurred_reference_slot']
        mask[0] = True
        if ref is not None:
            mask[ref] = True
        gaussian_sigma, motion_length = None, None
        if kind == 'gaussian':
            gaussian_sigma = strength
        elif kind in ('motion', 'motion_travel'):
            motion_length = strength
        elif kind == 'gaussian_motion':
            gaussian_sigma, motion_length = strength
        else:
            raise ValueError(f'Unknown blur condition {kind}')
        if motion_length is not None:
            kernel = motion_kernel(motion_length, sequence.manifest['motion_angle_deg'],
                                   exact_travel=kind != 'motion')
        for i, blurred in enumerate(mask):
            if not blurred:
                continue
            if gaussian_sigma is not None:
                frames[i] = cv2.GaussianBlur(frames[i], (0, 0), gaussian_sigma,
                                             borderType=cv2.BORDER_REFLECT_101)
            if motion_length is not None:
                frames[i] = cv2.filter2D(frames[i], -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    expected = [not np.array_equal(frame, original) for frame, original in zip(frames, sequence.images, strict=True)]
    if expected != mask or all(mask[1:]):
        raise ValueError(f'Unexpected changed frames {expected} for {sequence.name}, {kind}, {strength}')

    scores = [sharpness(frame) for frame in frames]
    rgb = np.stack([frame[:, :, ::-1].transpose(2, 0, 1) for frame in frames])
    labels = sequence.labels
    kpt = np.concatenate([labels[:, 5:7], np.full((len(labels), 1), 2, dtype=np.float32)], axis=1)
    sample = (torch.from_numpy(rgb), torch.from_numpy(labels[:, 0:1]),
              torch.from_numpy(labels[:, 1:5]), torch.from_numpy(kpt[:, None, :]),
              torch.zeros(len(labels), dtype=torch.long))
    return sample, mask, scores


def load_evaluators(output):
    settings_dir = str(output / 'ultralytics_settings')
    Path(settings_dir).mkdir()
    os.environ.update(YOLO_CONFIG_DIR=settings_dir, ULTRALYTICS_OFFLINE='true',
                      WANDB_MODE='disabled', COMET_MODE='DISABLED')
    training = trainer()
    sys.path[:0] = [str(YOLO_SUITE),
                    str(EXPERIMENTS / 'concat_core_multiseed_20261111/pose_ap')]
    import ultralytics
    from comparison import atomic_bytes, dump, metrics, yolo_infer
    from evaluate_saved import score_pose
    from pad_core import pad_infer
    os.environ['YOLO_CONFIG_DIR'] = settings_dir
    return training, ultralytics, pad_infer, yolo_infer, metrics, score_pose, dump, atomic_bytes


def metadata(sequences):
    by_frame = defaultdict(list)
    for seq in sequences:
        for i, hash_ in enumerate(seq.manifest['frame_sha256']):
            by_frame[hash_].append([seq.name, i])
    return dict(n_sequences=len(sequences), n_gt=sum(len(s.labels) for s in sequences),
                sequence_cohorts=dict(Counter(s.cohort for s in sequences)),
                assigned_reference_slots=dict(Counter(str(s.manifest['blurred_reference_slot']) for s in sequences)),
                identical_source_frames=[rows for rows in by_frame.values() if len(rows) > 1],
                sequences=[s.manifest for s in sequences])


def scopes(sequences):
    return dict(all=list(range(len(sequences))),
                group=[i for i, s in enumerate(sequences) if s.cohort == 'group'],
                current=[i for i, s in enumerate(sequences) if s.cohort == 'current'],
                ref0=[i for i, s in enumerate(sequences) if s.manifest['blurred_reference_slot'] is None],
                ref1=[i for i, s in enumerate(sequences) if s.manifest['blurred_reference_slot'] is not None])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--out', type=Path, default=HERE / 'results')
    parser.add_argument('--scope', choices=('all', 'current'), default='all')
    blur = parser.add_mutually_exclusive_group()
    blur.add_argument('--gaussian-motion', type=int, nargs=2, metavar=('SIGMA', 'TRAVEL'),
                      help='Evaluate one combined condition, Gaussian first and then linear motion')
    blur.add_argument('--motion-travel', type=int, metavar='TRAVEL',
                      help='Evaluate one linear motion blur with this exact travel distance in pixels')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Use only the first N sequences for smoke testing')
    args = parser.parse_args()
    conditions = CONDITIONS
    if args.gaussian_motion is not None:
        sigma, travel = args.gaussian_motion
        if sigma < 1 or travel < 1:
            parser.error('--gaussian-motion expects a positive sigma and travel length')
        conditions = [('gaussian_motion', (sigma, travel))]
    elif args.motion_travel is not None:
        if args.motion_travel < 1:
            parser.error('--motion-travel expects a positive travel length')
        conditions = [('motion_travel', args.motion_travel)]
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    device = 'cuda:0'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the Exp4 multi-frame checkpoint')
    folders = sorted(path for path in args.data.iterdir() if path.is_dir())
    if args.scope == 'current':
        folders = [path for path in folders if re.fullmatch(r'\d+_f\d+_x\d+_y\d+', path.name)]
    if args.limit is not None:
        folders = folders[:args.limit]
    if not folders:
        raise ValueError(f'No test sequences under {args.data}')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    training, ultralytics, pad_infer, yolo_infer, metrics, score_pose, dump, atomic_bytes = load_evaluators(args.out)
    arm = ARMS[0]
    ck_meta = assert_checkpoint(MAIN_CKPT, arm)
    model_args = args_for(architecture_flags(arm))
    main_model, _ = training.load_ckpt(str(MAIN_CKPT), model_args, device, prefer_ema=False)
    yolo_model = ultralytics.YOLO(str(YOLO_CKPT)).model.to(device).float().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pose_thresholds = torch.linspace(.50, .95, 10)
    result = dict(data=str(args.data.resolve()), scope=args.scope, seed=SEED, reference_blur_probability=REF_PROBABILITY,
                  batch_size=args.batch_size, input_size=SIZE, point_match_radius_px=10,
                  conf=.25, conf_floor=.001, max_candidates=300, yolo_nms_iou=.7,
                  pose_oks_sigma=1., pose_gt_area_factor=.53, precision='FP32',
                  runtime=dict(torch=torch.__version__, opencv=cv2.__version__,
                               ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0)),
                  models=dict(main=dict(checkpoint=str(MAIN_CKPT), checkpoint_sha256=digest(MAIN_CKPT),
                         seed=42, weights='stage3 last raw', metadata=ck_meta),
                     yolo=dict(checkpoint=str(YOLO_CKPT), checkpoint_sha256=digest(YOLO_CKPT),
                         seed=42, weights='native validation fitness best EMA')),
                  conditions=[])

    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        sequences = list(workers.map(read_sequence, folders))
        manifest = metadata(sequences)
        sequence_scopes = scopes(sequences)
        dump(args.out / 'manifest.json', manifest)
        print(f"{manifest['n_sequences']} sequences, {manifest['n_gt']} GT, "
              f"reference assignments {manifest['assigned_reference_slots']}", flush=True)

        for kind, strength in conditions:
            if kind == 'gaussian_motion':
                sigma, travel = strength
                key = f'gaussian_{sigma}_motion_{travel}'
            else:
                key = f'{kind}_{strength}' if strength else kind
            predictions = dict(main={}, yolo={})
            changed = []
            sharpness_rows = []
            for batch in batched(workers.map(lambda s: prepare(s, kind, strength), sequences), args.batch_size):
                samples, masks, scores = zip(*batch, strict=True)
                frames, targets = training.MultiFrameDataset.collate_fn(samples)
                main_preds = pad_infer(main_model, [(frames, targets)], SIZE)
                yolo_preds = yolo_infer(yolo_model, [(frames[:, 0].contiguous(), targets)])
                start = len(changed)
                for i in range(len(batch)):
                    if not np.array_equal(main_preds[i]['gt'], yolo_preds[i]['gt']):
                        raise ValueError(f'Models received inconsistent labels for {sequences[start + i].name}')
                    predictions['main'][start + i] = main_preds[i]
                    predictions['yolo'][start + i] = yolo_preds[i]
                changed.extend(masks)
                sharpness_rows.extend(scores)
            if len(changed) != len(sequences):
                raise ValueError(f'Incomplete condition {key}')
            condition = dict(name=key, kind=kind, strength=strength, changed_slots=changed,
                             laplacian_variance=sharpness_rows, metrics={})
            if kind == 'gaussian_motion':
                condition.update(gaussian_sigma_px=sigma, motion_travel_px=travel,
                                 motion_kernel_width_px=2 * math.ceil(travel / 2) + 1,
                                 filter_order=['gaussian', 'motion'])
            elif kind == 'motion_travel':
                condition.update(motion_travel_px=strength,
                                 motion_kernel_width_px=2 * math.ceil(strength / 2) + 1,
                                 filter_order=['motion'])
            for model, records in predictions.items():
                condition['metrics'][model] = {}
                for scope, indices in sequence_scopes.items():
                    if not indices:
                        continue
                    subset = {i: records[i] for i in indices}
                    values = metrics(subset)
                    ap = (score_pose(subset, pose_thresholds)[0] if any(len(r['pred']) for r in subset.values())
                          else np.zeros(len(pose_thresholds)))
                    values.update(n_sequences=len(subset),
                                  fp_per_image=(values['n_pred'] - values['n_tp']) / len(subset),
                                  pose_ap50=float(ap[0]), pose_ap75=float(ap[5]), pose_ap50_95=float(ap.mean()))
                    condition['metrics'][model][scope] = values
            atomic_bytes(args.out / f'{key}_predictions.pkl', pickle.dumps(predictions))
            result['conditions'].append(condition)
            dump(args.out / 'summary.json', result)
            print(key, {model: {field: round(values['all'][field], 4)
                               for field in ('point_ap_10px', 'point_recall', 'pose_ap50_95')}
                        for model, values in condition['metrics'].items()}, flush=True)


if __name__ == '__main__':
    main()
