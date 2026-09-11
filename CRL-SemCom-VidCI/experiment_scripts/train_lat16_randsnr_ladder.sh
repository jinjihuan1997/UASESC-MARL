#!/usr/bin/env bash
# Train 4 fixed-rate semantic codec models (rate level 1..4) with a 16-channel
# latent bottleneck and RANDOM SNR (0-20 dB) so each model is SNR-robust.
# RAN is bypassed via --comm_forced_rate_level (recon-only, no REINFORCE).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

LAT=16
LEVELS=4
SNR_MIN=0
SNR_MAX=20
SNR_REF=10            # validation reference SNR only
MAX_EPOCHS=50
MAX_BATCHES=64
BATCH=4
MLR=5e-5

for R in 1 2 3 4; do
  echo "=================================================================="
  echo "[ladder] training rate level ${R}  (lat=${LAT}, snr=${SNR_MIN}-${SNR_MAX})"
  echo "=================================================================="
  python train_semcom_x.py --local \
    --dataset_name nfs_block_rgb_256_8f -b 8,256,256 \
    --exp_name "SemCom_lat${LAT}_r${R}_randsnr" \
    --comm_latent_channels "${LAT}" --comm_rate_levels "${LEVELS}" \
    --comm_forced_rate_level "${R}" --eval_forced_rate_level "${R}" \
    --random_snr --comm_snr_min "${SNR_MIN}" --comm_snr_max "${SNR_MAX}" \
    --comm_snr_db "${SNR_REF}" \
    --warmup_epochs 0 --max_epochs "${MAX_EPOCHS}" --max_train_batches "${MAX_BATCHES}" \
    --ran_loss_weight 0 \
    --mlr "${MLR}" --batch_size "${BATCH}"
done

echo "[ladder] all 4 fixed-rate models done."
