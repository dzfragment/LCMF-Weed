#!/usr/bin/env bash
# Training-stage study: skip Stage 2 and train the full main model directly with Stage 3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP1_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # 仓库根目录
PROJECT_DIR="$EXP1_DIR/multi_frame_fusion"
EXP_ID="12_train_stage3_only"
DESCRIPTION="训练阶段实验：跳过 Stage 2，主模型从单帧基座初始化直接执行 Stage 3"

PRETRAINED="${PRETRAINED:-$EXP1_DIR/experiments/single_frame_baseline/results/yolo_v8_n_p2_lite_p4/seed42/best.pt}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
OUT="$RESULTS_ROOT/$EXP_ID"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
INPUT_SIZE="${INPUT_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
S3_EPOCHS="${S3_EPOCHS:-500}"
S3_LR="${S3_LR:-1e-4}"
FRAME_DROPOUT="${FRAME_DROPOUT:-0.1}"

[[ -f "$PROJECT_DIR/train.py" ]] || { echo "Missing train.py: $PROJECT_DIR/train.py" >&2; exit 1; }
[[ -n "$DATASET_ROOT" ]] || { echo "请先设置 DATASET_ROOT（weed_sliding_merged 数据集根目录）" >&2; exit 1; }
[[ -d "$DATASET_ROOT" ]] || { echo "Missing dataset: $DATASET_ROOT" >&2; exit 1; }
[[ -f "$PRETRAINED" ]] || { echo "Missing pretrained: $PRETRAINED" >&2; exit 1; }
mkdir -p "$OUT"

{
  echo "experiment=$EXP_ID"
  echo "description=$DESCRIPTION"
  echo "dataset=$DATASET_ROOT"
  echo "pretrained=$PRETRAINED"
  echo "training_schedule=stage3_only"
  echo "stage2_epochs=0"
  echo "stage3_epochs=$S3_EPOCHS"
  echo "stage3_lr=$S3_LR"
  echo "frame_dropout=$FRAME_DROPOUT"
  echo "head=unfrozen"
  echo "seed=$SEED"
  echo "input_size=$INPUT_SIZE"
  echo "batch_size=$BATCH_SIZE"
  echo "model=feature_fft+dav2_2stage+spatial"
} > "$OUT/run_config.txt"

START=()
TEE_ARGS=()
if [[ -f "$OUT/mf_stage3_last.pt" ]]; then
  START=(--resume "$OUT/mf_stage3_last.pt")
  TEE_ARGS=(-a)
fi

cd "$PROJECT_DIR"
echo "============================================================"
echo "[$EXP_ID] $DESCRIPTION"
echo "Stage 2 is intentionally skipped; fusion starts from identity/random initialization."
echo "============================================================"
python train.py \
  --stage 3 --epochs "$S3_EPOCHS" --lr "$S3_LR" \
  --batch-size "$BATCH_SIZE" --input-size "$INPUT_SIZE" \
  --fusion-levels p4 \
  --no-preload \
  --consistent-align --fft-downsample 4 \
  --fusion-gate-type spatial --fusion-mode gate_diff \
  --frame-dropout "$FRAME_DROPOUT" --unfreeze-head \
  --dataset-root "$DATASET_ROOT" --pretrained "$PRETRAINED" \
  --output-dir "$OUT" --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" --seed "$SEED" \
  "${START[@]}" \
  2>&1 | tee "${TEE_ARGS[@]}" "$OUT/stage3_console.log"

echo "[$EXP_ID] completed: $OUT"
