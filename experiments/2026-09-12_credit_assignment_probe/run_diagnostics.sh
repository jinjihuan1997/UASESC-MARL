#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$RUN_DIR/tmp"
PROBE_PYTHON="${PROBE_PYTHON:-/home/king/miniconda3/envs/harl_sionna/bin/python}"
cd "$RUN_DIR"
exec 9> "$RUN_DIR/run.lock"
flock -n 9 || { echo 'A diagnostic process already owns this run.'; exit 1; }
trap 'rc=$?; if [ "$rc" -ne 0 ]; then "$PROBE_PYTHON" record_failure.py "$rc"; fi' EXIT
if [[ "${1:-}" == "--aggregate-only" ]]; then
    "$PROBE_PYTHON" aggregate.py
    exit
fi
"$PROBE_PYTHON" -c 'from support import *; m=verify_inputs(full=True); assert read(HERE/"preflight.json")["state"]=="PASS"'
"$PROBE_PYTHON" evaluate_execution_variants.py
"$PROBE_PYTHON" collect_value_probe.py
"$PROBE_PYTHON" fit_value_probe.py
"$PROBE_PYTHON" aggregate.py
