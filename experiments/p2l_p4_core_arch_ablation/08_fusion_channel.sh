#!/usr/bin/env bash
# Fusion study: per-channel gate.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type channel)
S3_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type channel)
source "$SCRIPT_DIR/run_experiment.sh" \
  "08_fusion_channel" \
  "融合模块实验：Feature FFT + 双阶段 DAV2 + Channel gate"
