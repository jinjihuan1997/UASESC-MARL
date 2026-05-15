#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-quick}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-../logs/26-04-17/26-04-17-MST/SCI_M/v_1/checkpoints/model_best.pth}"
DATA_ROOT="${DATA_ROOT:-../data}"
DATASET_NAME="${DATASET_NAME:-nfs_block_rgb_256_8f}"
LOG_ROOT="${LOG_ROOT:-../logs}"
LOG_GROUP_DIR="${LOG_GROUP_DIR:-}"
DECODER="${DECODER:-MST}"
SHUTTER="${SHUTTER:-lsvpe}"
INTERP="${INTERP:-none}"
FREEZE_FRONTEND="${FREEZE_FRONTEND:-1}"

MLR="${MLR:-5e-5}"
RLR="${RLR:-1e-3}"
RAN_LOSS_WEIGHT="${RAN_LOSS_WEIGHT:-1.0}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"

mu_override=""
snr_override=""

usage() {
  cat <<'EOF'
Usage:
  ./run_semcom_sweep.sh [single|quick|full]

Modes:
  single  Train one short SemCom run.
  quick   Sweep a small set of mu/SNR points for fast screening.
  full    Sweep a larger set of mu/SNR points.

Useful overrides:
  BASE_CHECKPOINT=...      SCI checkpoint to initialize from
  DATASET_NAME=...         Dataset name under data_root
  LOG_GROUP_DIR=...        Fixed output directory; disable date-based folders
  MU_LIST_OVERRIDE="..."   Space-separated mu list, e.g. "5e-4 1e-3"
  SNR_LIST_OVERRIDE="..."  Space-separated SNR list, e.g. "5 10 15"
  MAX_EPOCHS=...           Override epochs
  BATCH_SIZE=...           Override batch size
  NUM_WORKERS=...          Override dataloader workers
  FREEZE_FRONTEND=0        Jointly fine-tune shutter/decoder

Examples:
  ./run_semcom_sweep.sh single
  ./run_semcom_sweep.sh quick
  MU_LIST_OVERRIDE="5e-4" SNR_LIST_OVERRIDE="10" MAX_EPOCHS=20 ./run_semcom_sweep.sh single
  FREEZE_FRONTEND=0 ./run_semcom_sweep.sh quick
EOF
}

case "$MODE" in
  single)
    mu_override="${MU_LIST_OVERRIDE:-5e-4}"
    snr_override="${SNR_LIST_OVERRIDE:-10}"
    MAX_EPOCHS="${MAX_EPOCHS:-15}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
    STEPS_TIL_SUMMARY="${STEPS_TIL_SUMMARY:-1500}"
    VAL_INTERVAL="${VAL_INTERVAL:-1500}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1500}"
    WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
    WARMUP_RATE_LEVEL="${WARMUP_RATE_LEVEL:-4}"
    EXP_PREFIX="${EXP_PREFIX:-SemCom_single}"
    ;;
  quick)
    mu_override="${MU_LIST_OVERRIDE:-5e-4 1e-3}"
    snr_override="${SNR_LIST_OVERRIDE:-10}"
    MAX_EPOCHS="${MAX_EPOCHS:-20}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
    STEPS_TIL_SUMMARY="${STEPS_TIL_SUMMARY:-1500}"
    VAL_INTERVAL="${VAL_INTERVAL:-1500}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1500}"
    WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
    WARMUP_RATE_LEVEL="${WARMUP_RATE_LEVEL:-4}"
    EXP_PREFIX="${EXP_PREFIX:-SemCom_quick}"
    ;;
  full)
    mu_override="${MU_LIST_OVERRIDE:-1e-4 5e-4 1e-3 2e-3}"
    snr_override="${SNR_LIST_OVERRIDE:-0 5 10 15 20}"
    MAX_EPOCHS="${MAX_EPOCHS:-50}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
    STEPS_TIL_SUMMARY="${STEPS_TIL_SUMMARY:-1000}"
    VAL_INTERVAL="${VAL_INTERVAL:-1000}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
    WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
    WARMUP_RATE_LEVEL="${WARMUP_RATE_LEVEL:-4}"
    EXP_PREFIX="${EXP_PREFIX:-SemCom_full}"
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 1
    ;;
esac

read -r -a MU_LIST <<< "$mu_override"
read -r -a SNR_LIST <<< "$snr_override"

freeze_args=()
if [[ "$FREEZE_FRONTEND" == "1" ]]; then
  freeze_args+=(--freeze_shutter --freeze_decoder)
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Mode: $MODE"
echo "Base checkpoint: $BASE_CHECKPOINT"
echo "Dataset: $DATASET_NAME"
echo "Log group dir: ${LOG_GROUP_DIR:-<date-based default>}"
echo "MU list: ${MU_LIST[*]}"
echo "SNR list: ${SNR_LIST[*]}"
echo "Freeze frontend: $FREEZE_FRONTEND"
echo "Epochs: $MAX_EPOCHS, batch_size: $BATCH_SIZE, num_workers: $NUM_WORKERS"
echo

for mu in "${MU_LIST[@]}"; do
  for snr in "${SNR_LIST[@]}"; do
    mu_tag="${mu//./p}"
    exp_name="${EXP_PREFIX}_mu${mu_tag}_snr${snr}"
    echo "=== Training $exp_name ==="
    python train_semcom_x.py \
      --data_root "$DATA_ROOT" \
      --dataset_name "$DATASET_NAME" \
      --log_root "$LOG_ROOT" \
      --log_group_dir "$LOG_GROUP_DIR" \
      --decoder "$DECODER" \
      --shutter "$SHUTTER" \
      --interp "$INTERP" \
      --base_checkpoint "$BASE_CHECKPOINT" \
      --exp_name "$exp_name" \
      "${freeze_args[@]}" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --steps_til_summary "$STEPS_TIL_SUMMARY" \
      --val_interval "$VAL_INTERVAL" \
      --save_interval "$SAVE_INTERVAL" \
      --max_epochs "$MAX_EPOCHS" \
      --mlr "$MLR" \
      --rlr "$RLR" \
      --mu_comm "$mu" \
      --ran_loss_weight "$RAN_LOSS_WEIGHT" \
      --grad_clip "$GRAD_CLIP" \
      --warmup_epochs "$WARMUP_EPOCHS" \
      --warmup_rate_level "$WARMUP_RATE_LEVEL" \
      --comm_snr_db "$snr"
    echo
  done
done
