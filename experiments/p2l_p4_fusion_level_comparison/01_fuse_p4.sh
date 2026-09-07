#!/usr/bin/env bash
# Control: temporal alignment/fusion only at P4.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ID="01_fuse_p4"
FUSION_LEVELS="p4"
DESCRIPTION="仅 P4 多帧融合；P2/P3 使用关键帧特征，检测头固定为 P4"
export EXP_ID FUSION_LEVELS DESCRIPTION
source "$SCRIPT_DIR/run_experiment.sh"
