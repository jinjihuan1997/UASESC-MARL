#!/usr/bin/env bash
set -euo pipefail
LIGHT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LIGHT_PYTHON="${LIGHT_PYTHON:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["executable"])' "$LIGHT_DIR/software_environment.json")}"
[[ -x "$LIGHT_PYTHON" ]] || { echo 'Set LIGHT_PYTHON to the existing compatible interpreter.' >&2; exit 1; }
"$LIGHT_PYTHON" "$LIGHT_DIR/run_evaluation.py" "$@"
