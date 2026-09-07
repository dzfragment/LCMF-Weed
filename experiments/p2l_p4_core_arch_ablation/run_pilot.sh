#!/usr/bin/env bash
# Run all seven groups for a short integration pilot.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export S2_EPOCHS="${S2_EPOCHS:-3}"
export S3_EPOCHS="${S3_EPOCHS:-3}"
export RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/pilot_results}"
bash "$SCRIPT_DIR/run_all.sh"
