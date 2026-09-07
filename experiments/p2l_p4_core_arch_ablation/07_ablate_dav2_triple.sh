#!/usr/bin/env bash
# Module study: three-stage DAV2, keeping Feature FFT and Spatial fusion unchanged.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--consistent-align --fft-downsample 4 --dav2-triple --fusion-gate-type spatial --fusion-mode gate_diff)
S3_ARGS=(--consistent-align --fft-downsample 4 --dav2-triple --fusion-gate-type spatial --fusion-mode gate_diff)
source "$SCRIPT_DIR/run_experiment.sh" \
  "07_ablate_dav2_triple" \
  "DAV2 层数实验：Feature FFT + 三阶段 DAV2 + Spatial gate"
