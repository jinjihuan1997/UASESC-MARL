#!/usr/bin/env bash
set -euo pipefail
RECOMPOSITION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "${PYTHON:-/home/king/miniconda3/envs/harl_sionna/bin/python}" "$RECOMPOSITION_DIR/run_evaluation.py"
