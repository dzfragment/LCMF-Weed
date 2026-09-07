#!/usr/bin/env bash
# Run all three fusion-level experiments sequentially.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for script in 01_fuse_p4.sh 02_fuse_p3.sh 03_fuse_p3_p4.sh; do
  echo "################################################################"
  echo "Running $script"
  echo "################################################################"
  bash "$SCRIPT_DIR/$script"
done
python "$SCRIPT_DIR/summarize_results.py"
