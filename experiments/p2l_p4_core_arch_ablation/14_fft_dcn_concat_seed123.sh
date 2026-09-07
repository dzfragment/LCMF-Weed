#!/usr/bin/env bash
# FFT + DCN + Concat 融合，seed=123（种子鲁棒性验证，独立于 05 的 seed=42）。
# 架构与 05_ablate_concat_fusion 一致：FFT(consistent-align) + 双阶段 DAV2 + Concat 投影融合。
# 仅 seed 不同（123 vs 42），结果写入独立目录，不影响已有模型权重。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type concat)
S3_ARGS=(--consistent-align --fft-downsample 4 --fusion-gate-type concat)
SEED=123 source "$SCRIPT_DIR/run_experiment.sh" \
  "14_fft_dcn_concat_seed123" \
  "FFT+DCN+Concat 融合 (seed=123, 种子鲁棒性)"
