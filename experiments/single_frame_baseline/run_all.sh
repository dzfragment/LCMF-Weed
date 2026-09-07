#!/bin/bash
# EXP-10: Sequential training of all YOLO variants
# Each model: 3 seeds [42, 123, 456], 500 epochs
# Logs written to train.log in each seed directory

set -e

cd "$(dirname "$0")"

MODELS=(
    yolo_v8_n
    yolo_v8_n_p3
    yolo_v8_n_p34
    yolo_v8_n_p34_lite
    yolo_v8_n_p2
    yolo_v8_n_p2_lite
    yolo_v8_n_p2_lite_p4
)

SEEDS=(42 123 456)
TOTAL=$(( ${#MODELS[@]} * ${#SEEDS[@]} ))
CURRENT=0
START_TIME=$(date +%s)

echo "============================================================"
echo "EXP-10: Single-frame YOLO Backbone Ablation"
echo "Models: ${#MODELS[@]}  |  Seeds per model: ${#SEEDS[@]}  |  Total runs: ${TOTAL}"
echo "Start: $(date)"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        RUN_START=$(date +%s)

        echo ""
        echo "============================================================"
        echo "[${CURRENT}/${TOTAL}] MODEL=${MODEL}  SEED=${SEED}"
        echo "Started at: $(date)"
        echo "============================================================"

        python train.py \
            --model "${MODEL}" \
            --seed "${SEED}" \
            --batch-size 8 \
            --epochs 500 \
            2>&1 | tee -a "results/${MODEL}/seed${SEED}/train.log"

        RUN_END=$(date +%s)
        RUN_ELAPSED=$((RUN_END - RUN_START))
        TOTAL_ELAPSED=$((RUN_END - START_TIME))

        echo "[${CURRENT}/${TOTAL}] Done in $((RUN_ELAPSED / 60))m $((RUN_ELAPSED % 60))s"
        echo "Total elapsed: $((TOTAL_ELAPSED / 3600))h $(((TOTAL_ELAPSED % 3600) / 60))m"
    done
done

echo ""
echo "============================================================"
echo "ALL DONE. Total time: $((( $(date +%s) - START_TIME) / 3600))h"
echo "============================================================"
