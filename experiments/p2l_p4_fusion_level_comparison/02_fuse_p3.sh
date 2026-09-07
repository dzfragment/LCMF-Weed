#!/usr/bin/env bash
# Temporal alignment/fusion only at P3; P4 remains the key-frame backbone feature.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ID="02_fuse_p3"
FUSION_LEVELS="p3"
DESCRIPTION="仅 P3 多帧融合；P2/P4 使用关键帧特征，融合后的 P3 经 FPN 影响最终 P4 Head"
export EXP_ID FUSION_LEVELS DESCRIPTION
source "$SCRIPT_DIR/run_experiment.sh"
