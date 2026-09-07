#!/usr/bin/env bash
# Run only the new supplement experiments. Historical 00-06 are intentionally excluded.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for script in \
  07_ablate_dav2_triple.sh \
  08_fusion_channel.sh \
  09_fusion_convgru.sh \
  10_fusion_tsa.sh \
  11_fusion_tsa_add.sh \
  12_train_stage3_only.sh; do
  echo "################################################################"
  echo "Running supplement: $script"
  echo "################################################################"
  bash "$SCRIPT_DIR/$script"
done

python "$SCRIPT_DIR/summarize_supplement_results.py"
