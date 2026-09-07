#!/usr/bin/env bash
# 仅 P3 多帧融合 + 单层 DCN (DeformAlignV2Single)；检测头固定 P4。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ID="02_fuse_p3"
FUSION_LEVELS="p3"
DESCRIPTION="仅 P3 融合 + 单层 DCN；P2/P4 用关键帧特征，检测头 P4"
EXTRA_ARGS=(--dav2-single)
export EXP_ID FUSION_LEVELS DESCRIPTION EXTRA_ARGS
source "$SCRIPT_DIR/run_experiment.sh"
