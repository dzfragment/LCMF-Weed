#!/usr/bin/env bash
# Shared runner for P2L-P4 core architecture ablation.
# Usage: source run_experiment.sh <experiment_id> <description>
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: source run_experiment.sh <experiment_id> <description>" >&2
  return 2 2>/dev/null || exit 2
fi

EXP_ID="$1"
DESCRIPTION="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP1_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # 仓库根目录
WORKSPACE_DIR="$(cd "$EXP1_DIR/.." && pwd)"
PROJECT_DIR="$EXP1_DIR/multi_frame_fusion"

DATASET_ROOT="${DATASET_ROOT:-}"
PRETRAINED="${PRETRAINED:-$EXP1_DIR/experiments/single_frame_baseline/results/weed_merged_1024/yolo_v8_n_p2_lite_p4/seed42/best.pt}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
OUT="$RESULTS_ROOT/$EXP_ID"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
INPUT_SIZE="${INPUT_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
S2_EPOCHS="${S2_EPOCHS:-100}"
S3_EPOCHS="${S3_EPOCHS:-500}"
S2_LR="${S2_LR:-1e-3}"
S3_LR="${S3_LR:-1e-4}"
FRAME_DROPOUT="${FRAME_DROPOUT:-0.1}"

: "${S2_ARGS:=}"
: "${S3_ARGS:=}"
if ! declare -p S2_ARGS 2>/dev/null | grep -q 'declare -a'; then S2_ARGS=(); fi
if ! declare -p S3_ARGS 2>/dev/null | grep -q 'declare -a'; then S3_ARGS=(); fi

[[ -f "$PROJECT_DIR/train.py" ]] || { echo "Missing train.py: $PROJECT_DIR/train.py" >&2; exit 1; }
[[ -n "$DATASET_ROOT" ]] || { echo "请先设置 DATASET_ROOT（weed_sliding_merged 数据集根目录）" >&2; exit 1; }
[[ -d "$DATASET_ROOT" ]] || { echo "Missing dataset: $DATASET_ROOT" >&2; exit 1; }
[[ -f "$PRETRAINED" ]] || { echo "缺少单帧基座权重: $PRETRAINED" >&2; echo "请先按 README 训练 single_frame_baseline 基线，或用 PRETRAINED 环境变量指定" >&2; exit 1; }
mkdir -p "$OUT"

BASE_ARGS=(
  --batch-size "$BATCH_SIZE"
  --input-size "$INPUT_SIZE"
  --fusion-levels p4
  --no-preload  # 流式读盘，避免 38GB 全量预加载触发 OOM
  --dataset-root "$DATASET_ROOT"
  --pretrained "$PRETRAINED"
  --output-dir "$OUT"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --seed "$SEED"
)

{
  echo "experiment=$EXP_ID"
  echo "description=$DESCRIPTION"
  echo "architecture=p2l_p4_fusion_p4_head"
  echo "project=$PROJECT_DIR"
  echo "dataset=$DATASET_ROOT"
  echo "pretrained=$PRETRAINED"
  echo "output=$OUT"
  echo "seed=$SEED"
  echo "input_size=$INPUT_SIZE"
  echo "batch_size=$BATCH_SIZE"
  echo "stage2_epochs=$S2_EPOCHS"
  echo "stage3_epochs=$S3_EPOCHS"
  printf 'stage2_args='; printf '%q ' "${S2_ARGS[@]}"; printf '\n'
  printf 'stage3_args='; printf '%q ' "${S3_ARGS[@]}"; printf '\n'
} > "$OUT/run_config.txt"

cd "$PROJECT_DIR"

S2_START=()
if [[ -f "$OUT/mf_stage2_last.pt" ]]; then
  S2_START=(--resume "$OUT/mf_stage2_last.pt")
fi

echo "============================================================"
echo "[$EXP_ID] $DESCRIPTION"
echo "Stage 2: train P4 alignment/fusion; freeze Backbone/FPN/Head"
echo "============================================================"
python train.py \
  --stage 2 --epochs "$S2_EPOCHS" --lr "$S2_LR" \
  "${BASE_ARGS[@]}" "${S2_ARGS[@]}" "${S2_START[@]}" \
  2>&1 | tee -a "$OUT/stage2_console.log"

[[ -f "$OUT/mf_stage2_best.pt" ]] || {
  echo "Stage 2 did not produce mf_stage2_best.pt" >&2
  exit 1
}

S3_START=(--weights "$OUT/mf_stage2_best.pt")
TEE_ARGS=()
if [[ -f "$OUT/mf_stage3_last.pt" ]]; then
  S3_START=(--resume "$OUT/mf_stage3_last.pt")
  TEE_ARGS=(-a)
fi

echo "============================================================"
echo "[$EXP_ID] Stage 3: end-to-end fine-tuning, Head unfrozen"
echo "============================================================"
python train.py \
  --stage 3 --epochs "$S3_EPOCHS" --lr "$S3_LR" \
  "${BASE_ARGS[@]}" --frame-dropout "$FRAME_DROPOUT" --unfreeze-head \
  "${S3_ARGS[@]}" "${S3_START[@]}" \
  2>&1 | tee "${TEE_ARGS[@]}" "$OUT/stage3_console.log"

echo "[$EXP_ID] completed: $OUT"
