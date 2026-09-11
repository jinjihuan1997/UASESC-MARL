#!/usr/bin/env bash
# Train exactly four selectable semantic modes for TON RL:
#   SCI ratio choices 0,3  x  RAN rate choices 1,4  => 2x2 = 4 modes.
#
# This trains one shared SemCom model with deterministic forced mode evaluation.
# The inner RAN policy is not used as the mode selector; TON/HARL RL selects
# (ratio_level, rate_level) from the four rows later.
set -euo pipefail

CRL=${CRL:-/home/king/Downloads/Projects/TON/CRL-SemCom-VidCI}
cd "$CRL/experiment_scripts"

PYTHON_BIN=${PYTHON_BIN:-/home/king/miniconda3/envs/crl-semcom-vidci/bin/python3.10}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

LOG_GROUP_DIR=${LOG_GROUP_DIR:-$CRL/logs/semantic_modes_2x2_adaptive_snr}
EXP_NAME=${EXP_NAME:-SemCom_2x2_modes_adaptive_snr}

# Paper-level indices kept available; training only samples the two choices below.
LAT=${LAT:-48}
RATE_LEVELS=4
RATIO_LEVELS=4
RATE_CHOICES=${RATE_CHOICES:-1,4}
RATIO_CHOICES=${RATIO_CHOICES:-0,3}

SNR_MIN=${SNR_MIN:-0}
SNR_MAX=${SNR_MAX:-20}
SNR_REF=${SNR_REF:-10}
MAX_EPOCHS=${MAX_EPOCHS:-50}
MAX_BATCHES=${MAX_BATCHES:-64}
BATCH=${BATCH:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
MLR=${MLR:-5e-5}

mkdir -p "$LOG_GROUP_DIR"

echo "[2x2-modes] log_group_dir=$LOG_GROUP_DIR"
echo "[2x2-modes] exp_name=$EXP_NAME"
echo "[2x2-modes] SCI ratio choices=$RATIO_CHOICES, RAN rate choices=$RATE_CHOICES, random SNR=${SNR_MIN}-${SNR_MAX} dB"

"$PYTHON_BIN" train_semcom_x.py --local \
  --dataset_name nfs_block_rgb_256_8f -b 8,256,256 \
  --log_group_dir "$LOG_GROUP_DIR" \
  --exp_name "$EXP_NAME" \
  --comm_latent_channels "$LAT" \
  --comm_rate_levels "$RATE_LEVELS" \
  --comm_ratio_levels "$RATIO_LEVELS" \
  --random_rate_level --random_rate_choices "$RATE_CHOICES" \
  --random_ratio_level --random_ratio_choices "$RATIO_CHOICES" \
  --random_snr --comm_snr_min "$SNR_MIN" --comm_snr_max "$SNR_MAX" \
  --comm_snr_db "$SNR_REF" \
  --eval_forced_rate_level 4 --eval_forced_ratio_level 3 \
  --warmup_epochs 0 \
  --ran_loss_weight 0 \
  --max_epochs "$MAX_EPOCHS" \
  --max_train_batches "$MAX_BATCHES" \
  --batch_size "$BATCH" \
  --num_workers "$NUM_WORKERS" \
  --mlr "$MLR"

echo "[2x2-modes] done. Evaluate four modes with test_semcom_x.py using:"
echo "  ratio/rate pairs: (0,1), (0,4), (3,1), (3,4)"
