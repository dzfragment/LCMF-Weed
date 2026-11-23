"""Evaluate single-frame YOLOv8n-pose with DIS optical-flow point fusion."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
import json
from pathlib import Path
import pickle
from time import perf_counter

import cv2
import numpy as np
import torch

from evaluate import DATA, HERE, SIZE, YOLO_CKPT, digest, load_evaluators, prepare, read_sequence


RESULTS = HERE / 'results'
COMBINED = RESULTS / 'gaussian_12_motion_30_current'
FLOW_SIZE = SIZE // 2
FLOW_SCALE = FLOW_SIZE / SIZE
ROUND_TRIP_PX = 3.
POINT_NMS_PX = 8.
MAX_POINTS = 300
CONF = .25


def sample_displacement(flow, points):
    """Sample half-resolution flow at full-resolution points and restore its units."""
    xy = np.asarray((points + .5) * FLOW_SCALE - .5, dtype=np.float32)
    vectors = cv2.remap(flow, xy[:, :1].copy(), xy[:, 1:].copy(), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    return vectors[:, 0] / FLOW_SCALE


def project(predictions, forward, backward):
    """Return consistent reference predictions in keyframe coordinates."""
    if not len(predictions):
        return predictions.copy(), 0

    points = predictions[:, 5:7]
    shift = sample_displacement(forward, points)
    mapped = points + shift
    inside = np.flatnonzero((np.isfinite(mapped) & (mapped >= 0) & (mapped < SIZE)).all(axis=1))
    if not len(inside):
        return predictions[:0].copy(), 0

    returned = mapped[inside] + sample_displacement(backward, mapped[inside])
    error = np.linalg.norm(returned - points[inside], axis=1)
    keep = inside[error <= ROUND_TRIP_PX]
    projected = predictions[keep].copy()
    projected[:, :4] += np.tile(shift[keep], 2)
    projected[:, 5:7] = mapped[keep]
    return projected, len(inside)


def point_nms(predictions):
    """Score-ordered 8 px suppression, matching the previous late-fusion radius."""
    candidates = np.concatenate(predictions)
    slots = np.repeat(np.arange(len(predictions)), [len(view) for view in predictions])
    order = np.argsort(-candidates[:, 4], kind='stable')
    keep = []
    while len(order) and len(keep) < MAX_POINTS:
        winner, order = order[0], order[1:]
        keep.append(winner)
        delta = candidates[order, 5:7] - candidates[winner, 5:7]
        order = order[np.sum(delta**2, axis=1) > POINT_NMS_PX**2]
    return candidates[keep], slots[keep]


def to_gray(frame):
    # Resize first to keep the flow preprocessing at half resolution.
    rgb = frame.permute(1, 2, 0).contiguous().numpy()
    small = cv2.resize(rgb, (FLOW_SIZE, FLOW_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def fuse(frames, views, dis):
    tick = perf_counter()
    key = to_gray(frames[0])
    aligned = [views[0]]
    ref_diagnostics = []
    for frame, pred in zip(frames[1:], views[1:], strict=True):
        inside = 0
        projected = pred
        if len(pred):
            reference = to_gray(frame)
            if not np.array_equal(reference, key):
                forward = dis.calc(reference, key, None)
                backward = dis.calc(key, reference, None)
                projected, inside = project(pred, forward, backward)
            else:
                inside = len(pred)
        aligned.append(projected)
        ref_diagnostics.append(dict(candidates=len(pred), inside_keyframe=inside,
                                    kept_after_flow=len(projected)))
    fused, source = point_nms(aligned)
    elapsed_ms = (perf_counter() - tick) * 1000
    counts = np.bincount(source[fused[:, 4] >= CONF], minlength=3)
    return fused, source, dict(refs=ref_diagnostics, selected_conf25_by_slot=counts.tolist(),
                               flow_and_fusion_ms=elapsed_ms)


def load_comparisons(combined_run=COMBINED):
    """Resolve the 42 current-frame sequences in each of the two prior runs."""
    original = json.loads((RESULTS / 'summary.json').read_text())
    original_sequences = json.loads((RESULTS / 'manifest.json').read_text())['sequences']
    indices = [i for i, row in enumerate(original_sequences) if row['cohort'] == 'current']
    sequences = [original_sequences[i] for i in indices]
    combined = json.loads((combined_run / 'summary.json').read_text())
    if json.loads((combined_run / 'manifest.json').read_text())['sequences'] != sequences:
        raise ValueError('The additional blur run used a different sequence manifest')
    for field in ('models', 'conf', 'point_match_radius_px', 'input_size', 'max_candidates', 'conf_floor'):
        if original[field] != combined[field]:
            raise ValueError(f'Mismatched comparison settings: {field}')
    if original['models']['yolo']['checkpoint_sha256'] != digest(YOLO_CKPT):
        raise ValueError(f'Unexpected YOLO checkpoint: {YOLO_CKPT}')
    conditions = [(RESULTS, cond, indices) for cond in original['conditions']]
    conditions += [(combined_run, cond, list(range(len(sequences)))) for cond in combined['conditions']]
    return original, sequences, conditions


def summarize(records, metrics, score_pose):
    values = metrics(records)
    counts = values['n_pred'], values['n_tp']
    values['fp_per_image'] = (counts[0] - counts[1]) / len(records)
    thresholds = torch.linspace(.5, .95, 10)
    ap = (score_pose(records, thresholds)[0] if any(len(row['pred']) for row in records.values())
          else np.zeros(len(thresholds)))
    return values | dict(pose_ap50=float(ap[0]), pose_ap75=float(ap[5]), pose_ap50_95=float(ap.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RESULTS / 'dis_yolo_late_fusion_current')
    parser.add_argument('--additional-run', '--combined-run', type=Path, default=COMBINED,
                        help='Saved additional blur run to include in the comparison')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--condition', action='append', help='Run only the named condition; may be repeated')
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    base, manifest, conditions = load_comparisons(args.additional_run)
    if args.condition:
        unknown = set(args.condition) - {c['name'] for _, c, _ in conditions}
        if unknown:
            parser.error(f'Unknown condition(s): {sorted(unknown)}')
        conditions = [(folder, c, indices) for folder, c, indices in conditions
                      if c['name'] in args.condition]

    args.out.mkdir(parents=True, exist_ok=False)
    training, ultralytics, _, yolo_infer, metrics, score_pose, dump, atomic_bytes = load_evaluators(args.out)
    model = ultralytics.YOLO(str(YOLO_CKPT)).model.to('cuda:0').float().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    result = dict(
        data=str(DATA), scope='current', n_sequences=len(manifest), n_gt=sum(row['n_gt'] for row in manifest),
        weights=base['models'], conf=CONF, conf_floor=base['conf_floor'], yolo_nms_iou=base['yolo_nms_iou'],
        point_match_radius_px=base['point_match_radius_px'], resolution=SIZE,
        algorithm=dict(flow='OpenCV DIS PRESET_FAST', flow_resolution=FLOW_SIZE,
                       finest_scale=dis.getFinestScale(), round_trip_max_px=ROUND_TRIP_PX,
                       point_nms_radius_px=POINT_NMS_PX, max_points=MAX_POINTS,
                       source_scores='unchanged detector confidence', cv2_threads=cv2.getNumThreads(),
                       timing_scope='grayscale resize, up to four flow fields, projection and point NMS; excludes YOLO'),
        runtime=dict(torch=torch.__version__, opencv=cv2.__version__, ultralytics=ultralytics.__version__),
        conditions=[],
    )
    dump(args.out / 'manifest.json', manifest)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        sequences = list(executor.map(read_sequence, (DATA / row['sequence'] for row in manifest)))
        if [seq.manifest for seq in sequences] != manifest:
            raise ValueError('Original sequence files or the reference draw changed')

        for folder, condition, indices in conditions:
            name, kind, strength = (condition[k] for k in ('name', 'kind', 'strength'))
            with (folder / f'{name}_predictions.pkl').open('rb') as stream:
                baseline = pickle.load(stream)
            records = dict(main={}, yolo={}, yolo_point_nms={}, dis_yolo={})
            provenance = []
            rows = executor.map(lambda seq: prepare(seq, kind, strength), sequences)
            for batch in batched(rows, args.batch_size):
                samples, masks, scores = zip(*batch, strict=True)
                frames, targets = training.MultiFrameDataset.collate_fn(samples)
                outputs = [yolo_infer(model, [(frames[:, slot].contiguous(), targets)])
                           for slot in range(frames.shape[1])]
                start = len(provenance)
                for j, (mask, sharp) in enumerate(zip(masks, scores, strict=True)):
                    i = start + j
                    base_idx = indices[i]
                    if mask != condition['changed_slots'][base_idx]:
                        raise ValueError(f'Blur mask mismatch for {sequences[i].name}')
                    np.testing.assert_allclose(sharp, condition['laplacian_variance'][base_idx], rtol=1e-6)
                    key = outputs[0][j]
                    gt = key['gt']
                    main = baseline['main'][base_idx]
                    if not np.array_equal(gt, main['gt']):
                        raise ValueError(f'Label mismatch for {sequences[i].name}')
                    views = [output[j]['pred'] for output in outputs]
                    fused, source, info = fuse(frames[j], views, dis)
                    key_point_nms, _ = point_nms(views[:1])
                    records['main'][i] = main
                    records['yolo'][i] = key
                    records['yolo_point_nms'][i] = dict(pred=key_point_nms, gt=gt)
                    records['dis_yolo'][i] = dict(pred=fused, gt=gt)
                    provenance.append(dict(sequence=sequences[i].name, views=views,
                                           fused_source_slots=source, **info))
            if len(provenance) != len(sequences):
                raise ValueError(f'Incomplete evaluation: {name}')
            info = [dict(sequence=p['sequence'], refs=p['refs'],
                         selected_conf25_by_slot=p['selected_conf25_by_slot'],
                         flow_and_fusion_ms=p['flow_and_fusion_ms']) for p in provenance]
            per_model = {key: summarize(pred, metrics, score_pose) for key, pred in records.items()}
            if not np.isclose(per_model['main']['point_ap_10px'], condition['metrics']['main']['current']['point_ap_10px']):
                raise ValueError(f'Main-model rescoring differs for {name}')
            times = [row['flow_and_fusion_ms'] for row in info]
            result['conditions'].append(dict(name=name, kind=kind, strength=strength, metrics=per_model,
                reference_candidates=sum(r['candidates'] for row in info for r in row['refs']),
                references_passing_flow=sum(r['kept_after_flow'] for row in info for r in row['refs']),
                selected_conf25_by_slot=np.array([row['selected_conf25_by_slot'] for row in info]).sum(axis=0).tolist(),
                flow_and_fusion_ms_median=float(np.median(times)),
                flow_and_fusion_ms_p90=float(np.percentile(times, 90)),
                previous_keyframe_ap10=condition['metrics']['yolo']['current']['point_ap_10px'],
                sequences=info))
            atomic_bytes(args.out / f'{name}_predictions.pkl', pickle.dumps(dict(records=records, provenance=provenance)))
            dump(args.out / 'summary.json', result)
            print(name, ' '.join(f'{key}: AP10={m["point_ap_10px"]:.3f}, R10={m["point_recall"]:.3f}'
                                 for key, m in per_model.items()), flush=True)


if __name__ == '__main__':
    main()
