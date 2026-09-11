#!/usr/bin/env bash
# Paper [50]-aligned reproduction: train ONE SHARED SCE/SCD codec.
#
# Key alignment with "Compression Ratio Learning and Semantic Communications
# for Video Imaging" (Zhang et al., JSTSP 2024):
#   * a SINGLE shared encoder/decoder (NOT one model per rate)
#   * the first-k prefix mask (build_rate_mask) keeps the first f*(C_m/levels)
#     channels -> identical masking MECHANISM to the paper's "first 12f symbols"
#   * training visits every rate level (--random_rate_level) so the latent
#     becomes ordered and forced-rate eval traces ONE rate-distortion curve
#   * random SNR per step makes the same codec SNR-robust (for the mode x SNR
#     offline profile)
#
# The paper's online adaptive RAN + REINFORCE (mu) is intentionally replaced by
# deterministic uniform mode selection: in the UAV-ESCS system the MARL agent
# selects the mode (rate level), exactly as ton.tex abstracts it. So RAN is
# bypassed at deployment and we disable its REINFORCE loss (--ran_loss_weight 0).
#
# C_m note: the paper uses C_m=48 for grey, T=16 NfS. Our RGB blocks saturate
# the reconstruction earlier, so we use a smaller C_m to place the 4 UNIFORM
# rate levels inside the steep R-D region (the masking rule is unchanged).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LAT=8           # C_m : levels map to 2/4/6/8 kept channels
LEVELS=4
SNR_MIN=0
SNR_MAX=20
SNR_REF=10
MAX_EPOCHS=50
MAX_BATCHES=64
BATCH=4
MLR=5e-5

echo "[paper-aligned] ONE shared codec, C_m=${LAT}, levels=${LEVELS} -> kept channels 2/4/6/8"
echo "[paper-aligned] random rate level + random SNR ${SNR_MIN}-${SNR_MAX} dB per step"

python train_semcom_x.py --local \
  --dataset_name nfs_block_rgb_256_8f -b 8,256,256 \
  --exp_name "SemCom_paper_shared_lat${LAT}" \
  --comm_latent_channels "${LAT}" --comm_rate_levels "${LEVELS}" \
  --random_rate_level \
  --random_snr --comm_snr_min "${SNR_MIN}" --comm_snr_max "${SNR_MAX}" \
  --comm_snr_db "${SNR_REF}" \
  --eval_forced_rate_level "${LEVELS}" \
  --warmup_epochs 0 --max_epochs "${MAX_EPOCHS}" --max_train_batches "${MAX_BATCHES}" \
  --ran_loss_weight 0 \
  --mlr "${MLR}" --batch_size "${BATCH}"

echo "[paper-aligned] done. Sweep the ladder with: python /tmp/eval_shared_ladder.py"
