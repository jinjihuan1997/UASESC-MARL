#!/usr/bin/env bash
# Quality ladder = SEPARATE fixed-SCI-ratio models (author's proven recipe).
# Each model fixes one SCI acquisition ratio (--comm_forced_ratio_level) so it
# CANNOT pull a low ratio up to the ceiling -> the quality ladder is preserved.
# Within each model the semantic rate (random) gives the COST axis, and random
# SNR gives the SNR axis. After training, sweep (rate x SNR) per model and stack
# the rows into the semantic-mode configuration table.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LAT=${LAT:-16}
RATE_LEVELS=4
RATIO_LEVELS=4
RATIO_CHOICES=${RATIO_CHOICES:-"0 1 2 3"}   # one model per SCI ratio level
SNR_MIN=${SNR_MIN:-0}; SNR_MAX=${SNR_MAX:-20}; SNR_REF=${SNR_REF:-10}
# Author used full data x 100+ epochs. A tiny subset will NOT separate the
# ladder, so use a larger budget here than the 64-batch debug runs.
MAX_EPOCHS=${MAX_EPOCHS:-60}
MAX_BATCHES=${MAX_BATCHES:-256}
BATCH=${BATCH:-4}
MLR=${MLR:-5e-5}

for R in ${RATIO_CHOICES}; do
  echo "=================================================================="
  echo "[sci-ladder] SEPARATE model, FIXED SCI ratio level ${R} (measure $((1<<R)))"
  echo "=================================================================="
  python train_semcom_x.py --local \
    --dataset_name nfs_block_rgb_256_8f -b 8,256,256 \
    --exp_name "SemCom_sci_ratio${R}_lat${LAT}" \
    --comm_latent_channels "${LAT}" --comm_rate_levels "${RATE_LEVELS}" \
    --comm_ratio_levels "${RATIO_LEVELS}" \
    --comm_forced_ratio_level "${R}" \
    --random_rate_level \
    --random_snr --comm_snr_min "${SNR_MIN}" --comm_snr_max "${SNR_MAX}" \
    --comm_snr_db "${SNR_REF}" \
    --eval_forced_rate_level 4 --eval_forced_ratio_level "${R}" \
    --warmup_epochs 0 --max_epochs "${MAX_EPOCHS}" --max_train_batches "${MAX_BATCHES}" \
    --ran_loss_weight 0 \
    --mlr "${MLR}" --batch_size "${BATCH}"
done

echo "[sci-ladder] done. Build the config table with: python /tmp/eval_sci_ladder_table.py"
