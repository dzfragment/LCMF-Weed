#!/usr/bin/env bash
# Remove FFT only; retain DCN + Spatial gate.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--fusion-gate-type spatial --fusion-mode gate_diff)
S3_ARGS=(--fusion-gate-type spatial --fusion-mode gate_diff)
source "$SCRIPT_DIR/run_experiment.sh" \
  "01_remove_fft" \
  "去除 FFT：仅 P4 DCN 精对齐 + Spatial gate 融合"
