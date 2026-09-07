#!/usr/bin/env python3
"""Measure inference time: 100 rounds per model, report mean/std/FPS."""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'YOLOv8-pose'))
from nets import nn

MODELS = [
    ('yolo_v8_n',             'SF-ORIG'),
    ('yolo_v8_n_p3',          'SF-P3'),
    ('yolo_v8_n_p34',         'SF-P34'),
    ('yolo_v8_n_p34_lite',    'SF-P34L'),
    ('yolo_v8_n_p2',          'SF-P2'),
    ('yolo_v8_n_p2_lite',     'SF-P2L'),
    ('yolo_v8_n_p2_lite_p4',  'SF-P2L-P4'),
]

BUILDERS = {
    'yolo_v8_n':             lambda: nn.yolo_v8_n(1, (1, 2)),
    'yolo_v8_n_p3':          lambda: nn.yolo_v8_n_p3(1, (1, 2)),
    'yolo_v8_n_p34':         lambda: nn.yolo_v8_n_p34(1, (1, 2)),
    'yolo_v8_n_p34_lite':    lambda: nn.yolo_v8_n_p34_lite(1, (1, 2)),
    'yolo_v8_n_p2':          lambda: nn.yolo_v8_n_p2(1, (1, 2)),
    'yolo_v8_n_p2_lite':     lambda: nn.yolo_v8_n_p2_lite(1, (1, 2)),
    'yolo_v8_n_p2_lite_p4':  lambda: nn.yolo_v8_n_p2_lite_p4(1, (1, 2)),
}

ROUNDS = 100
WARMUP = 20
INPUT_SIZE = 1024

torch.cuda.empty_cache()

print(f"{'Model':<12} {'Warmup':>6} {'Rounds':>6} {'Mean(ms)':>9} {'Std(ms)':>8} {'Min(ms)':>8} {'Max(ms)':>8} {'FPS':>7}")
print("-" * 68)

for mk, name in MODELS:
    model = BUILDERS[mk]().cuda().eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).cuda()

    # Warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = model(dummy)
    torch.cuda.synchronize()

    # Timed rounds
    times = []
    for _ in range(ROUNDS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    fps = 1000.0 / times.mean()
    print(f"{name:<12} {WARMUP:>6} {ROUNDS:>6} {times.mean():>8.3f} {times.std():>7.3f} {times.min():>7.3f} {times.max():>7.3f} {fps:>6.1f}")

    del model
    torch.cuda.empty_cache()
