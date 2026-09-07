#!/usr/bin/env python3
"""
EXP-10: FLOPs, parameter count, and latency benchmark for all YOLO variants.
Outputs: benchmark_results.json and benchmark_results.csv
"""

import argparse
import json
import os
import sys
import time

import numpy
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'YOLOv8-pose'))
from nets import nn

# ── Model registry (must match train.py) ──────────────────────────
MODEL_REGISTRY = {
    'yolo_v8_n':            ('SF-ORIG',   lambda: nn.yolo_v8_n(num_classes=1, kpt_shape=(1, 2))),
    'yolo_v8_n_p3':         ('SF-P3',     lambda: nn.yolo_v8_n_p3(num_classes=1, kpt_shape=(1, 2))),
    'yolo_v8_n_p34':        ('SF-P34',    lambda: nn.yolo_v8_n_p34(num_classes=1, kpt_shape=(1, 2))),
    'yolo_v8_n_p34_lite':   ('SF-P34L',   lambda: nn.yolo_v8_n_p34_lite(num_classes=1, kpt_shape=(1, 2))),
    'yolo_v8_n_p2':         ('SF-P2',     lambda: nn.yolo_v8_n_p2(num_classes=1, kpt_shape=(1, 2))),
    'yolo_v8_n_p2_lite':    ('SF-P2L',    lambda: nn.yolo_v8_n_p2_lite(num_classes=1, kpt_shape=(1, 2))),
    'yolo_v8_n_p2_lite_p4': ('SF-P2L-P4', lambda: nn.yolo_v8_n_p2_lite_p4(num_classes=1, kpt_shape=(1, 2))),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def count_params(model):
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_gflops(model, input_size=1024):
    """Measure GFLOPS via PyTorch FlopCounterMode (per-operator, real FLOPs)."""
    from torch.utils.flop_counter import FlopCounterMode
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fcm:
            _ = model(dummy)
    return fcm.get_total_flops() / 1e9


def measure_latency(model, input_size=1024, warmup=100, runs=500, amp=False):
    """Measure inference latency (batch=1) in milliseconds."""
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, input_size, input_size).to(device)

    model.eval()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            if amp:
                with torch.cuda.amp.autocast():
                    _ = model(dummy)
            else:
                _ = model(dummy)
    torch.cuda.synchronize()

    # Timed runs
    timings = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if amp:
                with torch.cuda.amp.autocast():
                    _ = model(dummy)
            else:
                _ = model(dummy)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)  # ms

    timings = numpy.array(timings)
    return {
        'mean_ms': float(timings.mean()),
        'std_ms': float(timings.std()),
        'median_ms': float(numpy.median(timings)),
        'p90_ms': float(numpy.percentile(timings, 90)),
        'p95_ms': float(numpy.percentile(timings, 95)),
        'p99_ms': float(numpy.percentile(timings, 99)),
        'fps': float(1000.0 / timings.mean()),
    }


def measure_memory(model, input_size=1024):
    """Measure peak GPU memory for a single forward pass (batch=1)."""
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, input_size, input_size).to(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    with torch.no_grad():
        _ = model(dummy)

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return peak_mb


def benchmark_all(output_dir=None):
    """Run full benchmark on all models. Save CSV + JSON."""
    results = []

    for model_key, (name, builder) in MODEL_REGISTRY.items():
        print(f"\n{'=' * 50}")
        print(f"Benchmarking: {name} ({model_key})")
        print(f"{'=' * 50}")

        model = builder().cuda()
        model.eval()

        # Params
        total_p, trainable_p = count_params(model)
        print(f"  Params: {total_p:,} total, {trainable_p:,} trainable")

        # GFLOPS
        try:
            gflops = measure_gflops(model)
            print(f"  GFLOPS: {gflops:.2f}")
        except Exception as e:
            print(f"  FLOPs: ERROR - {e}")
            gflops = None

        # Latency FP32
        lat_fp32 = measure_latency(model, amp=False)
        print(f"  Latency (FP32): mean={lat_fp32['mean_ms']:.2f}ms, "
              f"P95={lat_fp32['p95_ms']:.2f}ms, FPS={lat_fp32['fps']:.1f}")

        # Latency FP16/AMP
        lat_fp16 = measure_latency(model, amp=True)
        print(f"  Latency (AMP):  mean={lat_fp16['mean_ms']:.2f}ms, "
              f"P95={lat_fp16['p95_ms']:.2f}ms, FPS={lat_fp16['fps']:.1f}")

        # Memory
        peak_mem = measure_memory(model)
        print(f"  Peak memory: {peak_mem:.1f} MB")

        # Model file size estimation
        tmp_path = '/tmp/_bench_model.pt'
        torch.save({'model': model.half()}, tmp_path)
        file_size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
        os.remove(tmp_path)
        print(f"  Model file size (fp16): {file_size_mb:.2f} MB")

        results.append({
            'model_key': model_key,
            'name': name,
            'params_total': total_p,
            'params_trainable': trainable_p,
            'gflops': round(gflops, 2) if gflops else None,
            'latency_fp32_mean_ms': round(lat_fp32['mean_ms'], 3),
            'latency_fp32_p95_ms': round(lat_fp32['p95_ms'], 3),
            'latency_fp32_fps': round(lat_fp32['fps'], 1),
            'latency_amp_mean_ms': round(lat_fp16['mean_ms'], 3),
            'latency_amp_p95_ms': round(lat_fp16['p95_ms'], 3),
            'latency_amp_fps': round(lat_fp16['fps'], 1),
            'peak_memory_mb': round(peak_mem, 1),
            'file_size_mb': round(file_size_mb, 2),
        })

        del model
        torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────
    if output_dir is None:
        output_dir = SCRIPT_DIR

    # JSON
    json_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # CSV
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    fieldnames = ['model_key', 'name', 'params_total', 'params_trainable', 'gflops',
                  'latency_fp32_mean_ms', 'latency_fp32_p95_ms', 'latency_fp32_fps',
                  'latency_amp_mean_ms', 'latency_amp_p95_ms', 'latency_amp_fps',
                  'peak_memory_mb', 'file_size_mb']
    with open(csv_path, 'w') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {csv_path}")

    # ── Print summary table ──────────────────────────────────
    print(f"\n{'─' * 90}")
    print(f"{'Model':<14} {'Params':>10} {'GFLOPs':>8} {'FP32(ms)':>9} {'AMP(ms)':>8} {'FPS':>7} {'Mem(MB)':>8}")
    print(f"{'─' * 90}")
    for r in results:
        gflops_str = f"{r['gflops']:.1f}" if r['gflops'] else 'N/A'
        print(f"{r['name']:<14} {r['params_total']:>10,} {gflops_str:>8} "
              f"{r['latency_fp32_mean_ms']:>8.2f} {r['latency_amp_mean_ms']:>8.2f} "
              f"{r['latency_fp32_fps']:>6.0f} {r['peak_memory_mb']:>8.1f}")
    print(f"{'─' * 90}")

    return results


def main():
    parser = argparse.ArgumentParser(description='EXP-10 Model Benchmark')
    parser.add_argument('--model', default='all', help='Model key or "all"')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    if args.model != 'all':
        # Benchmark single model
        global MODEL_REGISTRY
        if args.model not in MODEL_REGISTRY:
            print(f"Unknown model: {args.model}. Choices: {list(MODEL_REGISTRY.keys())}")
            sys.exit(1)
        MODEL_REGISTRY = {args.model: MODEL_REGISTRY[args.model]}

    benchmark_all(args.output_dir)


if __name__ == "__main__":
    main()
