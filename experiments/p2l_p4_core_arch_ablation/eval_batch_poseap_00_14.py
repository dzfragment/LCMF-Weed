#!/usr/bin/env python3
"""Batch PoseAP evaluation for experiments 00 and 14.

The evaluation target is a split directory (e.g. $DATASET_ROOT/test).  Each sequence
contains frame_0, frame_1, frame_2 and labels_frame0.txt; frame_0 is the
key-frame target.  The metric implementation is reused from the project's
existing eval_5models_test_val_pose_ap module.

Examples:
    python eval_batch_poseap_00_14.py --dry-run
    python eval_batch_poseap_00_14.py
    python eval_batch_poseap_00_14.py --models 00,14 --checkpoint mf_stage3_last.pt
    python eval_batch_poseap_00_14.py --dataset /path/to/test --batch-size 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils import data

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = Path(os.environ.get("DATASET_ROOT") or "")
DEFAULT_OUTPUT = SCRIPT_DIR / "poseap_batch_results"
RESULTS_ROOT = Path(os.environ.get("BATCH_RESULTS_ROOT", SCRIPT_DIR / "results"))

# Import the project's already validated dataset, frame modes and PoseAP code.
sys.path.insert(0, str(SCRIPT_DIR))
from eval_5models_test_val_pose_ap import (  # noqa: E402
    EvalDataset,
    FRAME_MODES,
    evaluate_mode,
)

MODEL_SPECS = OrderedDict({
    "00": {
        "exp_dir": "00_full_fft_dcn_spatial",
        "label": "00 FFT+DCN+Spatial",
        "gate_type": "spatial", "gas": "feature", "fft_ds": 4,
        "no_align": False, "fusion_mode": "gate_diff",
        "dav2": None,
    },
    "01": {
        "exp_dir": "01_remove_fft",
        "label": "01 DCN+Spatial (去FFT)",
        "gate_type": "spatial", "gas": "none", "fft_ds": 8,
        "no_align": False, "fusion_mode": "gate_diff",
        "dav2": None,
    },
    "02": {
        "exp_dir": "02_remove_dcn",
        "label": "02 FFT+Spatial (去DCN)",
        "gate_type": "spatial", "gas": "feature", "fft_ds": 4,
        "no_align": True, "fusion_mode": "gate_diff",
        "dav2": None,
    },
    "03": {
        "exp_dir": "03_remove_spatial",
        "label": "03 FFT+DCN (去Spatial,Avg)",
        "gate_type": "avg", "gas": "feature", "fft_ds": 4,
        "no_align": False, "fusion_mode": "avg",
        "dav2": None,
    },
    "05": {
        "exp_dir": "05_ablate_concat_fusion",
        "label": "05 FFT+DCN+Concat",
        "gate_type": "concat", "gas": "feature", "fft_ds": 4,
        "no_align": False, "fusion_mode": "gate_diff",
        "dav2": None,
    },
    "06": {
        "exp_dir": "06_ablate_dav2_single",
        "label": "06 FFT+DCN+Spatial (DAV2单阶段)",
        "gate_type": "spatial", "gas": "feature", "fft_ds": 4,
        "no_align": False, "fusion_mode": "gate_diff",
        "dav2": "single",
    },
    "07": {
        "exp_dir": "07_ablate_dav2_triple",
        "label": "07 FFT+DCN+Spatial (DAV2三阶段)",
        "gate_type": "spatial", "gas": "feature", "fft_ds": 4,
        "no_align": False, "fusion_mode": "gate_diff",
        "dav2": "triple",
    },
    "08": {
        "exp_dir": "08_fusion_channel",
        "label": "08 Channel gate",
        "gate_type": "channel", "gas": "feature", "fft_ds": 4,
        "no_align": False, "fusion_mode": "gate_diff",
        "dav2": None,
    },
    "09": {
        "exp_dir": "09_fusion_convgru",
        "label": "09 ConvGRU",
        "gate_type": "convgru", "gas": "feature", "fft_ds": 4,
        "no_align": False, "fusion_mode": "gate_diff",
        "dav2": None,
        "convgru_kernel": 3, "convgru_layers": 1,
    },
    "10": {
        "exp_dir": "10_fusion_tsa",
        "label": "10 TSA",
        "gate_type": "tsa", "gas": "feature", "fft_ds": 4,
        "no_align": False, "fusion_mode": "gate_diff",
        "dav2": None,
    },
    "12": {
        "exp_dir": "12_train_stage3_only",
        "label": "12 Stage3Only",
        "gate_type": "spatial", "gas": "feature", "fft_ds": 4,
        "no_align": False, "fusion_mode": "gate_diff",
        "dav2": None,
    },
    "13": {
        "exp_dir": "13_remove_fft_dcn_concat",
        "label": "13 仅Concat融合",
        "gate_type": "concat", "gas": "none", "fft_ds": 8,
        "no_align": True, "fusion_mode": "gate_diff",
        "dav2": None,
    },
    "14": {
        "exp_dir": "14_fft_dcn_concat_seed123",
        "label": "14 FFT+DCN+Concat seed123",
        "gate_type": "concat", "gas": "feature", "fft_ds": 4,
        "no_align": False, "fusion_mode": "gate_diff",
        "dav2": None,
    },
    "15": {
        "exp_dir": "15_fft_dav2_single_concat",
        "label": "15 FFT+DCN+Concat (DAV2单阶段)",
        "gate_type": "concat", "gas": "feature", "fft_ds": 4,
        "no_align": False, "fusion_mode": "gate_diff",
        "dav2": "single",
    },
    "16": {
        "exp_dir": "16_fft_dav2_triple_concat",
        "label": "16 FFT+DCN+Concat (DAV2三阶段)",
        "gate_type": "concat", "gas": "feature", "fft_ds": 4,
        "no_align": False, "fusion_mode": "gate_diff",
        "dav2": "triple",
    },
})

# Ordered so --models 00,14 default keeps the original pairing, while
# other models follow a stable display order.
MODEL_DISPLAY_ORDER = list(MODEL_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help=f"Directory containing seq_* folders (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--seed-variants-root", type=Path, default=None,
        help="Root containing seed_*/test folders; evaluates each seed as a separate "
             "dataset (3f by default) and writes cross-seed stats.",
    )
    parser.add_argument(
        "--models", default=",".join(MODEL_SPECS),
        help=f"Comma-separated experiment IDs (default: all {len(MODEL_SPECS)} models)",
    )
    parser.add_argument(
        "--checkpoint", default="mf_stage3_best.pt",
        help="Checkpoint filename inside each results/<experiment> directory",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--frame-modes", default="all",
        help="all, or comma-separated mode keys: 1f,2f+r1,2f+r2,3f",
    )
    parser.add_argument(
        "--align", default="consistent", choices=["consistent", "chain-fft", "none"],
        help="Alignment pipeline used at inference. 'consistent' (training default) uses the "
             "consistent aligner; 'chain-fft' uses native chain FFT (ref1->key, ref2->ref1, "
             "cumulative ref2->key) on the same weights; 'none' disables feature alignment. "
             "Default: consistent",
    )
    parser.add_argument(
        "--weights-root", type=Path, default=None,
        help="Directory containing per-model <exp_dir>/<checkpoint> folders. Defaults to "
             "BATCH_RESULTS_ROOT env or the script's ./results",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate dataset/checkpoints and print the planned evaluation only",
    )
    return parser.parse_args()


def selected_models(value: str) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in ids if item not in MODEL_SPECS]
    if not ids or unknown:
        raise SystemExit(f"Invalid --models: {value!r}; choose from {list(MODEL_SPECS)}")
    return ids


def selected_modes(value: str) -> OrderedDict:
    if value == "all":
        return FRAME_MODES
    aliases = {
        "1f": "1f(key3)",
        "2f+r1": "2f(+ref1)",
        "2f+r2": "2f(+ref2)",
        "3f": "3f(key+r1+r2)",
    }
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in aliases]
    if not requested or unknown:
        raise SystemExit(f"Invalid --frame-modes: {value!r}; choose from {list(aliases)}")
    return OrderedDict((aliases[item], FRAME_MODES[aliases[item]]) for item in requested)


def checkpoint_metadata(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "epoch": ckpt.get("epoch", "?"),
        "best_pose_ap": float(ckpt.get("best_pose_ap", 0.0)),
        "fusion_version": ckpt.get("fusion_version", 2),
        "fusion_gate_type": ckpt.get("fusion_gate_type"),
        "global_align_stage": ckpt.get("global_align_stage"),
        "fft_downsample": ckpt.get("fft_downsample"),
        "consistent_align": bool(ckpt.get("consistent_align", False)),
    }


def build_model(model_id: str, checkpoint: Path, device: torch.device, align="consistent"):
    from models.mf_yolo import MultiFrameYOLO, enable_consistent_align
    from train import apply_ablation, apply_dav2_single, apply_dav2_triple

    spec = MODEL_SPECS[model_id]
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    fusion_version = ckpt.get("fusion_version", 2)
    # chain-fft must construct the feature-aligned model with consistent_align=False
    # so its forward path naturally uses FFTAlign (ref1->key + ref2->ref1 cumulative).
    use_consistent = (align == "consistent" and ckpt.get("consistent_align", False))
    model = MultiFrameYOLO(
        num_classes=1,
        kpt_shape=(1, 2),
        fusion_levels=list(ckpt.get("fusion_levels", ["p4"])),
        pretrained_path=None,
        fusion_version=fusion_version,
        fusion_gate_type=spec["gate_type"],
        global_align_stage=spec["gas"],
        fft_downsample=spec["fft_ds"],
        convgru_kernel=spec.get("convgru_kernel", 3),
        convgru_layers=spec.get("convgru_layers", 1),
        consistent_align=use_consistent,
    ).to(device)
    apply_ablation(model, no_align=spec["no_align"], fusion_mode=spec["fusion_mode"])
    if spec.get("dav2") == "single":
        apply_dav2_single(model)
    elif spec.get("dav2") == "triple":
        apply_dav2_triple(model)

    state_dict = ckpt.get("model")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint model payload in {checkpoint}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if use_consistent:
        enable_consistent_align(model)
    if missing:
        print(f"  [WARN] {model_id}: missing state keys={len(missing)}")
    if unexpected:
        print(f"  [WARN] {model_id}: unexpected state keys={len(unexpected)}")
    model.eval()
    return model, ckpt


def validate_dataset(dataset_dir: Path) -> tuple[int, int]:
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    seq_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    incomplete = []
    for seq_dir in seq_dirs:
        required = [seq_dir / f"frame_{i}.jpg" for i in range(3)]
        if any(not path.is_file() for path in required):
            incomplete.append(seq_dir.name)
    if incomplete:
        preview = ", ".join(incomplete[:5])
        raise SystemExit(f"Incomplete sequences ({len(incomplete)}), e.g. {preview}")
    return len(seq_dirs), sum(1 for path in seq_dirs if (path / "labels_frame0.txt").is_file())


def write_outputs(rows: list[dict], metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "poseap_summary.csv"
    fields = [
        "model", "label", "checkpoint", "epoch", "dataset", "sequences",
        "frame_mode", "box_ap", "pose_ap", "delta_pose_ap_vs_1f",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "poseap_summary.json"
    json_path.write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


def evaluate_dataset_from_args(dataset_dir, output_dir, model_ids, modes, args, device, label=None):
    """Run 3f-only PoseAP evaluation on one dataset dir; return (rows, metadata)."""
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    dataset_label = label or dataset_dir.name or dataset_dir.parent.name
    # namespace by align mode so consistent / chain-fft / none runs don't collide
    output_dir = output_dir / args.align / dataset_label
    sequence_count, label_count = validate_dataset(dataset_dir)

    plans = []
    for model_id in model_ids:
        checkpoint = RESULTS_ROOT / MODEL_SPECS[model_id]["exp_dir"] / args.checkpoint
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
        plans.append((model_id, checkpoint, checkpoint_metadata(checkpoint)))

    print(f"\nDevice: {device}")
    print(f"Dataset: {dataset_dir}")
    print(f"Sequences: {sequence_count}; label files: {label_count}")
    print(f"Models: {', '.join(model_ids)}")
    print(f"Frame modes: {', '.join(modes)}")
    for model_id, checkpoint, metadata in plans:
        print(
            f"  {model_id}: {checkpoint.name}, epoch={metadata['epoch']}, "
            f"train_best_pose_ap={metadata['best_pose_ap']:.4f}, "
            f"consistent_align={metadata['consistent_align']}"
        )

    # EvalDataset expects the dataset root plus split name; accepting a direct
    # split directory keeps the CLI convenient for arbitrary test folders.
    dataset = EvalDataset(str(dataset_dir.parent), dataset_dir.name)
    if len(dataset) != sequence_count:
        raise RuntimeError(f"Dataset validation mismatch: {sequence_count} vs loader {len(dataset)}")
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=EvalDataset.collate_fn,
    )

    run_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_dir),
        "sequences": sequence_count,
        "label_files": label_count,
        "device": str(device),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "checkpoint": args.checkpoint,
        "align": args.align,
        "weights_root": str(RESULTS_ROOT),
        "models": model_ids,
        "frame_modes": list(modes),
    }
    rows: list[dict] = []
    for model_id, checkpoint, ckpt_info in plans:
        print(f"\n=== {model_id}: {MODEL_SPECS[model_id]['label']} ===")
        started = time.perf_counter()
        model, ckpt = build_model(model_id, checkpoint, device, align=args.align)
        model_results = OrderedDict()
        try:
            for mode_name, mode_fn in modes.items():
                box_ap, pose_ap = evaluate_mode(model, loader, mode_fn)
                model_results[mode_name] = (float(box_ap), float(pose_ap))
                print(f"  {mode_name}: BoxAP={box_ap:.4f} PoseAP={pose_ap:.4f}")
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        baseline = model_results.get("1f(key3)", (None, None))[1]
        for mode_name, (box_ap, pose_ap) in model_results.items():
            rows.append({
                "model": model_id,
                "label": MODEL_SPECS[model_id]["label"],
                "checkpoint": str(checkpoint),
                "epoch": ckpt_info["epoch"],
                "dataset": str(dataset_dir),
                "sequences": sequence_count,
                "frame_mode": mode_name,
                "box_ap": round(box_ap, 6),
                "pose_ap": round(pose_ap, 6),
                "delta_pose_ap_vs_1f": (
                    round(pose_ap - baseline, 6)
                    if baseline is not None else ""
                ),
            })
        elapsed = time.perf_counter() - started
        print(f"  elapsed={elapsed:.1f}s")

    if not args.dry_run:
        write_outputs(rows, run_metadata, output_dir)
    return rows, run_metadata


def ensure_models_available(model_ids, checkpoint):
    for model_id in model_ids:
        path = RESULTS_ROOT / MODEL_SPECS[model_id]["exp_dir"] / checkpoint
        if not path.is_file():
            raise SystemExit(f"Checkpoint does not exist: {path}")
    return [
        (model_id, RESULTS_ROOT / MODEL_SPECS[model_id]["exp_dir"] / checkpoint,
         checkpoint_metadata(RESULTS_ROOT / MODEL_SPECS[model_id]["exp_dir"] / checkpoint))
        for model_id in model_ids
    ]


def write_seed_stats(all_rows, seed_names, output_dir, frame_modes):
    """Write cross-seed PoseAP/BoxAP statistics for each frame mode."""
    import statistics as st

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # rows carry fields 'seed','model','pose_ap','box_ap','frame_mode'
    def slug(mode):
        return {'1f(key3)': '1f', '2f(+ref1)': '2f_ref1', '2f(+ref2)': '2f_ref2',
                '3f(key+r1+r2)': '3f'}.get(mode, mode.replace('(', '').replace(')', '').replace('+', '').replace(' ', '_'))

    for fm in frame_modes:
        rows = [r for r in all_rows if r["frame_mode"] == fm]
        per_model = {}
        for r in rows:
            per_model.setdefault(r["model"], []).append(r)

        stats_rows = []
        for model_id, records in per_model.items():
            paps = [r["pose_ap"] for r in records]
            baps = [r["box_ap"] for r in records]
            stats_rows.append({
                "model": model_id,
                "label": records[0]["label"],
                "n_seeds": len(records),
                "mean_pose_ap": round(st.mean(paps), 6),
                "min_pose_ap": round(min(paps), 6),
                "max_pose_ap": round(max(paps), 6),
                "range_pose_ap": round(max(paps) - min(paps), 6),
                "std_pose_ap": round(st.stdev(paps), 6) if len(paps) > 1 else 0.0,
                "mean_box_ap": round(st.mean(baps), 6),
                "std_box_ap": round(st.stdev(baps), 6) if len(baps) > 1 else 0.0,
            })
        stats_rows.sort(key=lambda r: r["mean_pose_ap"], reverse=True)
        rank = {r["model"]: i + 1 for i, r in enumerate(stats_rows)}
        for r in stats_rows:
            r["mean_rank"] = rank[r["model"]]

        fields = ["model", "label", "n_seeds", "mean_pose_ap", "min_pose_ap", "max_pose_ap",
                  "range_pose_ap", "std_pose_ap", "mean_box_ap", "std_box_ap", "mean_rank"]
        csv_path = output_dir / f"seed_stats_{slug(fm)}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(stats_rows)

        json_path = output_dir / f"seed_stats_{slug(fm)}.json"
        matrix = {
            "frame_mode": fm,
            "seeds": seed_names,
            "models": {m: [None] * len(seed_names) for m in per_model},
        }
        for r in rows:
            idx = seed_names.index(r["seed"])
            matrix["models"][r["model"]][idx] = r["pose_ap"]
        json_path.write_text(
            json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print(f"Saved seed stats: {csv_path}  (n_seeds={len(seed_names)})")
        print(f"Saved seed matrix: {json_path}")
    return frame_modes


def main() -> None:
    args = parse_args()
    model_ids = selected_models(args.models)
    modes = selected_modes(args.frame_modes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global RESULTS_ROOT
    if args.weights_root is not None:
        RESULTS_ROOT = args.weights_root.expanduser().resolve()
    if not RESULTS_ROOT.is_dir():
        raise SystemExit(f"Weights root does not exist: {RESULTS_ROOT}")

    if args.seed_variants_root is not None:
        root = args.seed_variants_root.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"Seed variants root does not exist: {root}")
        seed_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        datasets = []
        for sd in seed_dirs:
            test_dir = sd / "test"
            if test_dir.is_dir():
                datasets.append((sd.name, test_dir))
        if not datasets:
            raise SystemExit(f"No seed_*/test found under {root}")
        print(f"Seed variants root: {root}")
        print(f"Found {len(datasets)} seed datasets")

        output_root = args.output_dir.expanduser().resolve() / "seed_variants"
        all_rows = []
        seed_names = []
        for seed_name, test_dir in datasets:
            label = f"{seed_name}"
            print(f"\n{'#'*70}\n# SEED {seed_name}\n{'#'*70}")
            if args.dry_run:
                n, l = validate_dataset(test_dir)
                print(f"  {test_dir}: {n} seq, {l} labels (modes={list(modes)})")
                continue
            rows, meta = evaluate_dataset_from_args(
                test_dir, output_root, model_ids, modes, args, device, label=seed_name)
            added = False
            for r in rows:
                if r["model"] in model_ids and r["frame_mode"] in modes:
                    new = dict(r)
                    new["seed"] = seed_name
                    all_rows.append(new)
                    added = True
            if added:
                seed_names.append(seed_name)

        if args.dry_run:
            print("Dry run complete for seed variants.")
            return

        # Track seeds that produced data (some may be filtered by mode)
        used_seeds = list(dict.fromkeys(seed_names))
        stats_root = output_root / args.align
        write_seed_stats(all_rows, used_seeds, stats_root, list(modes))
        return

    # single dataset path
    dataset_dir = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    evaluate_dataset_from_args(dataset_dir, output_dir, model_ids, modes, args, device)


if __name__ == "__main__":
    main()
