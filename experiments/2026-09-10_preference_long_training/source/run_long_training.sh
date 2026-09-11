#!/usr/bin/env bash
set -euo pipefail
LONG_TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LONG_TRAIN_PY=/home/king/miniconda3/envs/harl_sionna/bin/python
LONG_TRAIN_RUN="$LONG_TRAIN_ROOT/runs/three_seed_sc_20260909"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LONG_TRAIN_COMMAND="${1:-status}"
if (($#)); then shift; fi
exec "$LONG_TRAIN_PY" "$LONG_TRAIN_ROOT/three_seed_train.py" "$LONG_TRAIN_COMMAND" --output "$LONG_TRAIN_RUN" "$@"
