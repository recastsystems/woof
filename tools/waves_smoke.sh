#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; echo "waves_smoke_exit=$rc"' EXIT
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
LANE=$(dirname -- "$ROOT")
export PYTHONPATH="$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export CUPY_CACHE_DIR="$LANE/cupy-cache"
export WOOF_WAVES_CACHE="$LANE/wave-cache"
export GPUWM_RW_WAVES_COUPLED_PREPARE="$LANE/bin/rw_waves_coupled_prepare"
export GPUWM_RW_WAVES_DATA="$LANE/bin/rw_waves_data"
export RAYON_NUM_THREADS=$(getconf _NPROCESSORS_ONLN)
export OMP_NUM_THREADS=$RAYON_NUM_THREADS
PYTHON=${WAVES_PYTHON:?point to the existing release interpreter}
cd "$ROOT"
date -u +%FT%TZ
git rev-parse HEAD
timeout 60s "$PYTHON" -m compileall -q woof/ocean tools/waves_coupling_smoke.py
timeout 45s "$PYTHON" -m woof run --help
timeout 45s "$PYTHON" -m woof go --help
timeout 45s "$PYTHON" -m woof ocean waves --help
timeout 480s "$PYTHON" tools/waves_coupling_smoke.py
date -u +%FT%TZ
