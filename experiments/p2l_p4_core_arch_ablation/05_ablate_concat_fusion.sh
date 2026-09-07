#!/usr/bin/env bash
# Fusion mechanism ablation: feature FFT + P4 DCN + learned concat projection.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type concat)
S3_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type concat)
source "$SCRIPT_DIR/run_experiment.sh" \
  "05_ablate_concat_fusion" \
  "融合机制消融：Feature FFT + P4 DCN 后，以 Concat 投影替代 Spatial gate"
