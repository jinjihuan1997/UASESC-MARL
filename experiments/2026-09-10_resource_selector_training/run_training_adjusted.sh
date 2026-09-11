#!/usr/bin/env bash
set -euo pipefail
selector_training_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/king/miniconda3/envs/harl_sionna/bin/python "$selector_training_root/supervisor_adjusted.py" start --resume
