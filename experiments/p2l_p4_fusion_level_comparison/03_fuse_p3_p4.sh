#!/usr/bin/env bash
# Temporal alignment/fusion at both P3 and P4; P2 remains key-frame only.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ID="03_fuse_p3_p4"
FUSION_LEVELS="p3,p4"
DESCRIPTION="P3 与 P4 同时多帧融合；P2 使用关键帧特征，检测头仍固定为 P4"
export EXP_ID FUSION_LEVELS DESCRIPTION
source "$SCRIPT_DIR/run_experiment.sh"
