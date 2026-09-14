#!/usr/bin/env bash
set -euo pipefail
STUDY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
STUDY_PYTHON="${STUDY_PYTHON:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["executable"])' "$STUDY_DIR/software_environment.json")}"
if [[ ! -x "$STUDY_PYTHON" ]]; then
  echo "Set STUDY_PYTHON to the already provisioned compatible Python interpreter." >&2
  exit 1
fi
"$STUDY_PYTHON" "$STUDY_DIR/orchestrate.py"
