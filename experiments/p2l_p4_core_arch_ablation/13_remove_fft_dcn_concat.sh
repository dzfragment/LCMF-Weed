#!/usr/bin/env bash
# Remove FFT + DCN: no global alignment, no DCN, concat feature fusion only.
# S2/S3 args match exp1's 13_remove_fft_dcn_concat (纯无对齐 + concat).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--no-align --fusion-gate-type concat --fusion-mode gate_diff)
S3_ARGS=(--no-align --fusion-gate-type concat --fusion-mode gate_diff)
source "$SCRIPT_DIR/run_experiment.sh" \
  "13_remove_fft_dcn_concat" \
  "去除 FFT + DCN：无对齐 + Concat 特征拼接融合"
