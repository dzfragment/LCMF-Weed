#!/usr/bin/env bash
# Remove DCN only; retain feature FFT + Spatial gate.
# --no-align bypasses DAV2 after FFT-shifted features enter the fusion module.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--consistent-align --fft-downsample 4 --no-align --fusion-gate-type spatial --fusion-mode gate_diff)
S3_ARGS=(--consistent-align --fft-downsample 4 --no-align --fusion-gate-type spatial --fusion-mode gate_diff)
source "$SCRIPT_DIR/run_experiment.sh" \
  "02_remove_dcn" \
  "去除 DCN：Feature FFT 粗对齐后直接用 Spatial gate 融合"
