#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WEIGHT_PYTHON=${WEIGHT_PYTHON:-/home/king/miniconda3/envs/harl_sionna/bin/python}
"$WEIGHT_PYTHON" run_evaluation.py
"$WEIGHT_PYTHON" finalize.py
