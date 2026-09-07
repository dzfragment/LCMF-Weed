#!/usr/bin/env bash
# P3+P4 多帧融合 + 单层 DCN (DeformAlignV2Single)；P2 用关键帧特征，检测头 P4。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ID="03_fuse_p3_p4"
FUSION_LEVELS="p3,p4"
DESCRIPTION="P3+P4 融合 + 单层 DCN；P2 用关键帧特征，检测头 P4"
EXTRA_ARGS=(--dav2-single)
export EXP_ID FUSION_LEVELS DESCRIPTION EXTRA_ARGS
source "$SCRIPT_DIR/run_experiment.sh"
