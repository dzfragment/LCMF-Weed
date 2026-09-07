#!/usr/bin/env bash
# Short 3-epoch integration pilot for only the six new experiments.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export S2_EPOCHS="${S2_EPOCHS:-3}"
export S3_EPOCHS="${S3_EPOCHS:-3}"
export RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/supplement_pilot_results}"
export PILOT_S3_EPOCHS="${PILOT_S3_EPOCHS:-3}"

# The first five use the shared two-stage runner.
for script in \
  07_ablate_dav2_triple.sh \
  08_fusion_channel.sh \
  09_fusion_convgru.sh \
  10_fusion_tsa.sh \
  11_fusion_tsa_add.sh; do
  echo "################################################################"
  echo "Pilot: $script"
  echo "################################################################"
  bash "$SCRIPT_DIR/$script"
done

# Stage3-only has its own one-stage schedule.
export S3_EPOCHS="$PILOT_S3_EPOCHS"
echo "################################################################"
echo "Pilot: 12_train_stage3_only.sh"
echo "################################################################"
bash "$SCRIPT_DIR/12_train_stage3_only.sh"

python "$SCRIPT_DIR/summarize_supplement_results.py"
