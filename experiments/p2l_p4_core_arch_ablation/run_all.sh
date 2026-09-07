#!/usr/bin/env bash
# Run the complete seven-group core architecture ablation sequentially.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for script in \
  00_full_fft_dcn_spatial.sh \
  01_remove_fft.sh \
  02_remove_dcn.sh \
  03_remove_spatial.sh \
  05_ablate_concat_fusion.sh \
  06_ablate_dav2_single.sh; do
  echo "################################################################"
  echo "Running $script"
  echo "################################################################"
  bash "$SCRIPT_DIR/$script"
done

python "$SCRIPT_DIR/summarize_results.py"
