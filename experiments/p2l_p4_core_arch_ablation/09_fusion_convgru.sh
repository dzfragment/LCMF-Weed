#!/usr/bin/env bash
# Fusion study: ConvGRU temporal recurrence.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type convgru --convgru-kernel 3 --convgru-layers 1)
S3_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type convgru --convgru-kernel 3 --convgru-layers 1)
source "$SCRIPT_DIR/run_experiment.sh" \
  "09_fusion_convgru" \
  "融合模块实验：Feature FFT + 双阶段 DAV2 + 单层 ConvGRU"
