#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} == --help ]]; then
    echo "Usage: $0 {clean|natocc} YOLOV8_BACKBONE_WEIGHTS [train_pad.py options]"
    echo "Set LCMF_DATA_ROOT to the directory containing the prepared datasets."
    exit 0
fi
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 {clean|natocc} YOLOV8_BACKBONE_WEIGHTS [train_pad.py options]" >&2
    exit 2
fi

dataset=${LCMF_DATA_ROOT:-"$root/dataset"}
split=$1
weights=$2
shift 2
case $split in
    clean) data="$dataset/weed_merged_1024_clean" ;;
    natocc) data="$dataset/weed_sliding_merged_keyframe" ;;
    *) echo "Expected clean or natocc" >&2; exit 2 ;;
esac
if [[ ! -f $weights ]]; then
    echo "Missing backbone weights: $weights" >&2
    exit 2
fi
exec "${PYTHON:-python}" "$root/experiments/single_frame_baseline/train_pad.py" \
    --data-dir "$data" --out-dir "$root/experiments/single_frame_baseline/results/pad_sf_$split/p2l" \
    --base-model p2l --pretrained "$weights" --head ms --epochs 300 \
    --batch-size 8 --accumulate 8 --lr0 0.01 --select-metric point_error \
    --min-recall 0.85 --min-delta-err 0.005 --num-workers 4 --seed 42 "$@"
