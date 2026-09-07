#!/usr/bin/env bash
# Control: FFT + DCN + Spatial gate.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type spatial --fusion-mode gate_diff)
S3_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type spatial --fusion-mode gate_diff)
source "$SCRIPT_DIR/run_experiment.sh" \
  "00_full_fft_dcn_spatial" \
  "完整模型：Feature FFT 粗对齐 + P4 DCN 精对齐 + Spatial gate 融合"
