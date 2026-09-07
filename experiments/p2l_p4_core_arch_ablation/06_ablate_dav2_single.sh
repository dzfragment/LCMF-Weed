#!/usr/bin/env bash
# Alignment capacity ablation: feature FFT + single-stage DAV2 + Spatial gate.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--consistent-align --fft-downsample 4 --dav2-single --fusion-gate-type spatial --fusion-mode gate_diff)
S3_ARGS=(--consistent-align --fft-downsample 4 --dav2-single --fusion-gate-type spatial --fusion-mode gate_diff)
source "$SCRIPT_DIR/run_experiment.sh" \
  "06_ablate_dav2_single" \
  "对齐容量消融：Feature FFT + 单阶段 DAV2 + Spatial gate 融合"
