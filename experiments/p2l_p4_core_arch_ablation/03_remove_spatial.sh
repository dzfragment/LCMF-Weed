#!/usr/bin/env bash
# Remove Spatial gate only; retain feature FFT + DCN.
# Use parameter-free averaging as the neutral replacement for learned gating.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type avg --fusion-mode avg)
S3_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type avg --fusion-mode avg)
source "$SCRIPT_DIR/run_experiment.sh" \
  "03_remove_spatial" \
  "去除 Spatial gate：Feature FFT + P4 DCN 后使用无参数平均融合"
