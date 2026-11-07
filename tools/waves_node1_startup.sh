#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
LANE=$(dirname -- "$ROOT")
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 GPUWM_NO_LOCAL_GPU=0
export CUPY_CACHE_DIR="$LANE/cupy-cache" XDG_CACHE_HOME="$LANE/cache" TMPDIR="$LANE/tmp"
export WOOF_WAVES_CACHE="$LANE/wave-cache"
export GPUWM_RW_WAVES_COUPLED_PREPARE="$LANE/bin/rw_waves_coupled_prepare"
export GPUWM_RW_WAVES_DATA="$LANE/bin/rw_waves_data"
export WOOF_CPU_PREPROCESS_BRIDGE="$LANE/bin/libgpuwm_preprocess_cpu.so"
export WOOF_NCWRITE_BRIDGE="$LANE/bin/libnetcdf_writer.so"
export WOOF_WAVES_BATHYMETRY="$LANE/shared-inputs/etopo2022-bed-60s.nc"
export WOOF_WAVES_STATIONS="$LANE/shared-inputs/stations.txt"
export WOOF_RW_WRFBATCH="$LANE/bin/rw_wrfbatch"
export RAYON_NUM_THREADS=24 OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=1
cd "$ROOT"
trap 'rc=$?; echo "full_grid_startup_exit=$rc"; date -u +%FT%TZ' EXIT
date -u +%FT%TZ
git rev-parse HEAD
timeout 420s "$LANE/venv/bin/python" -u tools/waves_startup_smoke.py \
    --prepared-root "$LANE/preparation/milton-native/prepared" \
    --experiment-config "$LANE/preparation/milton-native/authority/experiment.toml" \
    --wps-namelist "$LANE/preparation/milton-native/authority/namelist.wps" \
    --outdir "${WAVES_STARTUP_OUTPUT:-$LANE/startup-check-hour}" --seconds 3600
