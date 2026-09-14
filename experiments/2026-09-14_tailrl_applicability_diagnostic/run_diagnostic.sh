#!/usr/bin/env bash
# Recompute the completed diagnostic only; no new rollouts or training.
set -euo pipefail
diag_dir=$(dirname "$(realpath "$0")")
python_exec=${PYTHON_EXECUTABLE:-/home/king/miniconda3/envs/harl_sionna/bin/python}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$python_exec" "$diag_dir/reproduce.py"
