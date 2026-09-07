#!/usr/bin/env bash
# Shared runner for P3/P4 fusion-level comparison.
# Source with EXP_ID, FUSION_LEVELS and DESCRIPTION.
set -euo pipefail

: "${EXP_ID:?EXP_ID is required}"
: "${FUSION_LEVELS:?FUSION_LEVELS is required}"
: "${DESCRIPTION:?DESCRIPTION is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP2_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # 仓库根目录
WORKSPACE_DIR="$(cd "$EXP2_DIR/.." && pwd)"
PROJECT_DIR="$EXP2_DIR/multi_frame_fusion"

DATASET_ROOT="${DATASET_ROOT:-}"
# 统一使用 single_frame_baseline 产出的 SF-P2L-P4 基座 (weed_merged_1024, seed42)
PRETRAINED="${PRETRAINED:-$EXP2_DIR/experiments/single_frame_baseline/results/weed_merged_1024/yolo_v8_n_p2_lite_p4/seed42/best.pt}"
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

[[ -f "$PROJECT_DIR/train.py" ]] || { echo "Missing train.py: $PROJECT_DIR/train.py" >&2; exit 1; }
[[ -n "$DATASET_ROOT" ]] || { echo "请先设置 DATASET_ROOT（weed_sliding_merged 数据集根目录）" >&2; exit 1; }
[[ -d "$DATASET_ROOT" ]] || { echo "Missing dataset: $DATASET_ROOT" >&2; exit 1; }
[[ -f "$PRETRAINED" ]] || { echo "Missing pretrained: $PRETRAINED" >&2; exit 1; }
mkdir -p "$OUT"

# 可选追加参数（实验脚本可设置 EXTRA_ARGS=(--dav2-single) 等）
: "${EXTRA_ARGS:=}"
if ! declare -p EXTRA_ARGS 2>/dev/null | grep -q 'declare -a'; then EXTRA_ARGS=(); fi

COMMON_ARGS=(
  --batch-size "$BATCH_SIZE"
  --input-size "$INPUT_SIZE"
  --fusion-levels "$FUSION_LEVELS"
  --consistent-align
  --fft-downsample 4
  --fusion-gate-type concat
  --no-preload
  --dataset-root "$DATASET_ROOT"
  --pretrained "$PRETRAINED"
  --output-dir "$OUT"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --seed "$SEED"
  "${EXTRA_ARGS[@]}"
)

{
  echo "experiment=$EXP_ID"
  echo "description=$DESCRIPTION"
  echo "architecture=p2l_p4_fpn_p4_head"
  echo "feature_levels=p2,p3,p4"
  echo "fusion_levels=$FUSION_LEVELS"
  echo "detect_levels=p4"
  echo "global_alignment=consistent_feature_downsample4"
  echo "local_alignment=dav2"
  echo "fusion_gate=concat"
  echo "dataset=$DATASET_ROOT"
  echo "pretrained=$PRETRAINED"
  echo "seed=$SEED"
  echo "input_size=$INPUT_SIZE"
  echo "batch_size=$BATCH_SIZE"
  echo "stage2_epochs=$S2_EPOCHS"
  echo "stage3_epochs=$S3_EPOCHS"
  echo "stage2_lr=$S2_LR"
  echo "stage3_lr=$S3_LR"
  echo "frame_dropout=$FRAME_DROPOUT"
  echo "stage3_head=unfrozen"
  printf 'extra_args='; printf '%q ' "${EXTRA_ARGS[@]}"; printf '\n'
} > "$OUT/run_config.txt"

cd "$PROJECT_DIR"

# 若 Stage 3 已有进度（best 或 last 任一存在），跳过 Stage 2 直接续训 Stage 3。
# 用于 OOM 被杀后自动重启时避免无意义地重跑已完成/无效果的 Stage 2。
if [[ -f "$OUT/mf_stage3_last.pt" || -f "$OUT/mf_stage3_best.pt" ]]; then
  echo "[$EXP_ID] Stage 3 已有进度，跳过 Stage 2，直接续训 Stage 3"
else
  S2_START=()
  if [[ -f "$OUT/mf_stage2_last.pt" ]]; then
    S2_START=(--resume "$OUT/mf_stage2_last.pt")
  fi

  echo "============================================================"
  echo "[$EXP_ID] $DESCRIPTION"
  echo "Fusion levels: $FUSION_LEVELS; detection level remains P4"
  echo "Stage 2"
  echo "============================================================"
  python train.py \
    --stage 2 --epochs "$S2_EPOCHS" --lr "$S2_LR" \
    "${COMMON_ARGS[@]}" "${S2_START[@]}" \
    2>&1 | tee -a "$OUT/stage2_console.log"

  [[ -f "$OUT/mf_stage2_best.pt" ]] || {
    echo "Stage 2 did not produce mf_stage2_best.pt" >&2
    exit 1
  }
fi

S3_START=(--weights "$OUT/mf_stage2_best.pt")
TEE_ARGS=()
if [[ -f "$OUT/mf_stage3_last.pt" ]]; then
  S3_START=(--resume "$OUT/mf_stage3_last.pt")
  TEE_ARGS=(-a)
fi

echo "============================================================"
echo "[$EXP_ID] Stage 3"
echo "============================================================"
python train.py \
  --stage 3 --epochs "$S3_EPOCHS" --lr "$S3_LR" \
  "${COMMON_ARGS[@]}" --frame-dropout "$FRAME_DROPOUT" --unfreeze-head \
  "${S3_START[@]}" \
  2>&1 | tee "${TEE_ARGS[@]}" "$OUT/stage3_console.log"

echo "[$EXP_ID] completed: $OUT"
