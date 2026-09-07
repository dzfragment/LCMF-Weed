#!/usr/bin/env bash
# FFT + 三阶段 DAV2 + Concat 融合（结合 07 的 DCN 容量消融与 05 的 Concat 融合）。
# consistent-align 保留（FFT 全局对齐）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--consistent-align --fft-downsample 4 --dav2-triple --fusion-gate-type concat)
S3_ARGS=(--consistent-align --fft-downsample 4 --dav2-triple --fusion-gate-type concat)
source "$SCRIPT_DIR/run_experiment.sh" \
  "16_fft_dav2_triple_concat" \
  "FFT + 三阶段 DAV2 + Concat 融合"
