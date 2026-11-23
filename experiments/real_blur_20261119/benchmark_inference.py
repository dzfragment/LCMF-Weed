"""Benchmark the existing three Exp4 inference pipelines at batch one."""

import argparse
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import cv2
import numpy as np
import torch

from evaluate import (ARMS, DATA, HERE, MAIN_CKPT, SIZE, YOLO_CKPT, architecture_flags,
                      args_for, assert_checkpoint, digest, load_evaluators, prepare, read_sequence)
from evaluate_dis_fusion import fuse


RUN = HERE / 'results/dis_yolo_gaussian_4_motion_7_current'


def statistics(values):
    return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                p90=float(np.percentile(values, 90)), p95=float(np.percentile(values, 95)),
                stdev=float(np.std(values, ddof=1)))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--out', type=Path, default=RUN / 'inference_latency.json')
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=25)
    parser.add_argument('--save-flow-inputs', type=Path, help='Save frames for separate operation counting')
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 1:
        parser.error('repetitions and warmup must be positive')
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'
    summary = json.loads((args.run / 'summary.json').read_text())
    condition, = summary['conditions']
    manifest = json.loads((args.run / 'manifest.json').read_text())
    for name, checkpoint in [('main', MAIN_CKPT), ('yolo', YOLO_CKPT)]:
        if digest(checkpoint) != summary['weights'][name]['checkpoint_sha256']:
            raise ValueError(f'Mismatched checkpoint for {name}')
    samples = []
    for row in manifest:
        seq = read_sequence(DATA / row['sequence'])
        if seq.manifest != row:
            raise ValueError(f'Sequence changed: {seq.name}')
        sample, _, _ = prepare(seq, condition['kind'], condition['strength'])
        samples.append(sample[0])
    if args.save_flow_inputs is not None:
        np.save(args.save_flow_inputs, torch.stack(samples).numpy())

    with TemporaryDirectory(prefix='exp4-inference-profile-') as tmp:
        training, ultralytics, pad_infer, yolo_infer, _, _, dump, _ = load_evaluators(Path(tmp))
        yolo = ultralytics.YOLO(str(YOLO_CKPT)).model.to(device).float().eval()
        arm = ARMS[0]
        assert_checkpoint(MAIN_CKPT, arm)
        main_model, _ = training.load_ckpt(str(MAIN_CKPT), args_for(architecture_flags(arm)),
                                          device, prefer_ema=False)
        main_model = main_model.eval()
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        # Reuse production decoders without timing any conversion of test labels.
        targets = dict(idx=torch.empty(0, dtype=torch.long), cls=torch.empty((0, 1)),
                       box=torch.empty((0, 4)), kpt=torch.empty((0, 1, 3)))

        def single(frames):
            yolo_infer(yolo, [(frames[0:1], targets)])

        def flow_fusion(frames):
            preds = [yolo_infer(yolo, [(frame.unsqueeze(0), targets)])[0]['pred']
                     for frame in frames]
            _, _, details = fuse(frames, preds, dis)
            return details['flow_and_fusion_ms']

        def multi(frames):
            pad_infer(main_model, [(frames.unsqueeze(0), targets)], SIZE)

        pipelines = dict(single_yolo=single, yolo_dis=flow_fusion, exp4_main=multi)
        labels = tuple(pipelines)
        for i in range(args.warmup):
            frames = samples[i % len(samples)]
            for label in labels:
                pipelines[label](frames)
        torch.cuda.synchronize()

        rng = random.Random(20261119)
        rows = {label: [] for label in labels}
        flow_times = []
        for repetition in range(args.repetitions):
            order = list(range(len(samples)))
            rng.shuffle(order)
            for i in order:
                run_order = list(labels)
                rng.shuffle(run_order)
                for label in run_order:
                    torch.cuda.synchronize()
                    tick = perf_counter_ns()
                    cpu_ms = pipelines[label](samples[i])
                    torch.cuda.synchronize()
                    elapsed_ms = (perf_counter_ns() - tick) / 1e6
                    row = dict(repetition=repetition, sequence=manifest[i]['sequence'], ms=elapsed_ms)
                    rows[label].append(row)
                    if cpu_ms is not None:
                        flow_times.append(row | {'ms': cpu_ms})
            print(f'Finished repetition {repetition + 1}/{args.repetitions}', flush=True)

        cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                   if line.startswith('model name'))
        result = dict(condition=condition['name'], input_size=SIZE, precision='fp32',
                      batch_sequences=1, cache='none', n_sequences=len(samples),
                      warmup_per_pipeline=args.warmup, repetitions=args.repetitions,
                      hardware=dict(cpu=cpu, gpu=torch.cuda.get_device_name(0)),
                      software=dict(torch=torch.__version__, opencv=cv2.__version__,
                                    ultralytics=ultralytics.__version__, torch_cpu_threads=torch.get_num_threads(),
                                    opencv_threads=cv2.getNumThreads(), tf32=False),
                      timing_scope=dict(start='CPU resident RGB uint8 CHW tensors',
                                        end='decoded keyframe points with filtering',
                                        includes='CPU→GPU copy, normalization, forward, decode/filtering, CPU DIS and point NMS',
                                        excludes='file reads, synthetic blur generation, model initialization, metric scoring'),
                      pipelines={label: statistics([r['ms'] for r in traces]) for label, traces in rows.items()},
                      dis_and_fusion=statistics([r['ms'] for r in flow_times]),
                      measurements_ms=rows, dis_measurements_ms=flow_times)
        dump(args.out, result)
        print(json.dumps(result['pipelines'], indent=2), flush=True)


if __name__ == '__main__':
    main()
