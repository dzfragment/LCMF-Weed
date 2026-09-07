#!/usr/bin/env bash
# Fusion study: temporal self-attention.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type tsa)
S3_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type tsa)
source "$SCRIPT_DIR/run_experiment.sh" \
  "10_fusion_tsa" \
  "融合模块实验：Feature FFT + 双阶段 DAV2 + Temporal Self-Attention"
