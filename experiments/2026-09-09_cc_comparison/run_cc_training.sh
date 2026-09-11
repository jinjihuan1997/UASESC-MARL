#!/usr/bin/env bash
set -euo pipefail
CC_TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CC_TRAIN_PY=/home/king/miniconda3/envs/harl_sionna/bin/python
CC_TRAIN_RUN="$CC_TRAIN_ROOT/runs/three_seed_cc_20260909"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=
CC_TRAIN_COMMAND="${1:-status}"
if (($#)); then shift; fi
exec "$CC_TRAIN_PY" "$CC_TRAIN_ROOT/three_seed_train.py" "$CC_TRAIN_COMMAND" --output "$CC_TRAIN_RUN" "$@"
