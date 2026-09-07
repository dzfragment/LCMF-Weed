#!/usr/bin/env bash
# Fusion study: temporal self-attention with non-negative residual constraint.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type tsa_add)
S3_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type tsa_add)
source "$SCRIPT_DIR/run_experiment.sh" \
  "11_fusion_tsa_add" \
  "融合模块实验：Feature FFT + 双阶段 DAV2 + 非负残差 TSA-add"
