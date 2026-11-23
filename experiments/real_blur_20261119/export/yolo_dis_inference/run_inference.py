"""Run the bundled model on one triplet and optionally benchmark repeated inference."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

from yolo_dis import CONF, SOURCE_NAMES, YoloDIS, prepare_frames


def read_image(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def summarize(samples):
    return dict(count=len(samples), median=float(np.median(samples)),
                p95=float(np.percentile(samples, 95)))


def write_overlay(image, predictions, slots, path):
    colors = ((55, 190, 75), (50, 176, 245), (238, 123, 98))
    annotated = image.copy()
    for record, slot in zip(predictions, slots, strict=True):
        x, y = (int(round(v)) for v in record[5:7])
        cv2.circle(annotated, (x, y), 8, (12, 20, 24), 3, cv2.LINE_AA)
        cv2.circle(annotated, (x, y), 8, colors[slot], 2, cv2.LINE_AA)
        cv2.putText(annotated, f'{slot}:{record[4]:.2f}', (x + 9, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, colors[slot], 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or '.png', annotated)
    if not ok:
        raise ValueError(f'Cannot encode overlay: {path}')
    encoded.tofile(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keyframe', type=Path, required=True, help='Current target image')
    parser.add_argument('--ref1', type=Path, required=True, help='First reference image')
    parser.add_argument('--ref2', type=Path, required=True, help='Second reference image')
    parser.add_argument('--weights', type=Path, required=True, help='Trained one-keypoint YOLOv8 pose checkpoint')
    parser.add_argument('--device', default='auto', help='auto, cpu, or cuda:0')
    parser.add_argument('--confidence', type=float, default=CONF)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--output', type=Path, help='Write predictions as JSON')
    parser.add_argument('--overlay', type=Path, help='Write a keyframe overlay, e.g. overlay.png')
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error('repeat must be positive and warmup cannot be negative.')
    try:
        keyframe, reference_1, reference_2 = (read_image(path) for path in (args.keyframe, args.ref1, args.ref2))
        key = YoloDIS(args.device, args.cpu_threads, weights=args.weights)
        start = perf_counter()
        frames, geometry = prepare_frames(keyframe, reference_1, reference_2)
        input_preparation_ms = (perf_counter() - start) * 1000
        for _ in range(args.warmup):
            key.infer_prepared(frames, geometry, confidence=args.confidence)
        elapsed, flow = [], []
        for _ in range(args.repeat):
            result = key.infer_prepared(frames, geometry, confidence=args.confidence)
            elapsed.append(result.inference_ms)
            flow.append(result.flow_and_fusion_ms)
        records = [dict(x=float(row[5]), y=float(row[6]), confidence=float(row[4]),
                        bbox_xyxy=row[:4].tolist(), source=SOURCE_NAMES[slot], source_slot=int(slot))
                   for row, slot in zip(result.predictions, result.source_slots, strict=True)]
        payload = dict(device=str(key.device), keyframe_size=dict(width=geometry.width, height=geometry.height),
                       model_input_size=1024, confidence_threshold=args.confidence,
                       input_paths=dict(keyframe=str(args.keyframe), reference_1=str(args.ref1),
                                        reference_2=str(args.ref2)),
                       timing_ms=dict(input_preparation=input_preparation_ms, inference=summarize(elapsed),
                                      flow_and_fusion=summarize(flow)),
                       reference_diagnostics=result.references, n_points=len(records), keypoints=records)
        content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding='utf-8')
        else:
            sys.stdout.write(content)
        if args.overlay:
            write_overlay(keyframe, result.predictions, result.source_slots, args.overlay)
    except (OSError, ValueError, RuntimeError, cv2.error) as err:
        parser.exit(1, f'Inference failed: {err}\n')


if __name__ == '__main__':
    main()
