#!/usr/bin/env bash
set -euo pipefail
reward_pilot_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/king/miniconda3/envs/harl_sionna/bin/python "$reward_pilot_root/supervisor.py" "$@"
