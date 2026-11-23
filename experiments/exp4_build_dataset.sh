#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} == --help ]]; then
    echo "Usage: $0 [--generate]"
    echo "Set LCMF_DATA_ROOT to the dataset directory. --generate builds sliding data from source images."
    exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --generate ) ]]; then
    echo "Usage: $0 [--generate]" >&2
    exit 2
fi

cd "$root"
python_bin=${PYTHON:-python}
dataset=${LCMF_DATA_ROOT:-"$root/dataset"}
export LCMF_DATA_ROOT="$dataset" WEED_SRC_ROOT="$dataset" WEED_EXP_ROOT="$dataset"
if [[ ${1:-} == --generate ]]; then
    "$python_bin" dataset_tools/build_occluder_bank.py
    export OCCLUDER_BANK="$dataset/patch_bank_train.npy"
    "$python_bin" dataset_tools/stage2_sliding.py
    "$python_bin" dataset_tools/gen_public_weed_sliding.py
fi
"$python_bin" dataset_tools/merge_sliding.py
DATASET_ROOT="$dataset/weed_sliding_merged" KEYFRAME_ROOT="$dataset/weed_sliding_merged_keyframe" \
    "$python_bin" experiments/single_frame_baseline/convert_keyframe_dataset.py
