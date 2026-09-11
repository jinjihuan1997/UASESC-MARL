#!/usr/bin/env bash
# PDF-aligned SemCom reproduction with adaptive-SNR training.
#
# Keeps the main paper settings:
#   * NfS 16-frame 256x256 blocks.
#   * C_m=48 and four semantic rate levels f=1..4.
#   * Prefix mask keeps the first 12f symbols at each latent location.
#   * First 80 epochs force f=4 for SCE/SCD warmup; then RAN samples f and
#     participates in training through the REINFORCE loss.
#   * SCI uses the original five-action ratio space:
#       -1,0,1,2,3 -> 0, 1/T, 2/T, 4/T, 8/T.
#
# Deliberate deviation from the PDF experiment:
#   * SNR is sampled uniformly from 0..20 dB during training instead of fixed
#     at 10 dB, so the trained codec is robust for downstream mode profiling.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LAT=48
RATE_LEVELS=4
RATIO_LEVELS=5
SNR_MIN=0
SNR_MAX=20
SNR_REF=10
MAX_EPOCHS=180
MAX_BATCHES=0
BATCH=4
MLR=5e-5
RLR=5e-3
MU_COMM=1e-3
RAN_LOSS_WEIGHT=1.0
WARMUP_EPOCHS=80
WARMUP_RATE_LEVEL=4
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[paper-strict-adaptive-snr] C_m=${LAT}, rates=1..${RATE_LEVELS}, SCI ratios={0,1/T,2/T,4/T,8/T}, SNR=${SNR_MIN}-${SNR_MAX}dB"

"${PYTHON_BIN}" train_semcom_x.py --local \
  --dataset_name nfs_block_rgb_256_8f -b 8,256,256 \
  --exp_name "SemCom_paper_strict_lat${LAT}_adaptive_snr" \
  --legacy_action_space \
  --comm_latent_channels "${LAT}" --comm_rate_levels "${RATE_LEVELS}" \
  --comm_ratio_levels "${RATIO_LEVELS}" \
  --random_ratio_level \
  --random_snr --comm_snr_min "${SNR_MIN}" --comm_snr_max "${SNR_MAX}" \
  --comm_snr_db "${SNR_REF}" \
  --eval_forced_rate_level "${RATE_LEVELS}" --eval_forced_ratio_level 3 \
  --warmup_epochs "${WARMUP_EPOCHS}" --warmup_rate_level "${WARMUP_RATE_LEVEL}" \
  --max_epochs "${MAX_EPOCHS}" --max_train_batches "${MAX_BATCHES}" \
  --mu_comm "${MU_COMM}" --ran_loss_weight "${RAN_LOSS_WEIGHT}" \
  --mlr "${MLR}" --rlr "${RLR}" --batch_size "${BATCH}"

echo "[paper-strict-adaptive-snr] done."
