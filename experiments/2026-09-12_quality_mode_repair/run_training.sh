#!/usr/bin/env bash
set -euo pipefail
repair_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
repair_python="${REPAIR_PYTHON:-/home/king/miniconda3/envs/harl_sionna/bin/python}"
exec "$repair_python" "$repair_dir/supervise.py"
