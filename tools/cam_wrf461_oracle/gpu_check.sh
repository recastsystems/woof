#!/usr/bin/env bash
set -euo pipefail
L=$(realpath "$1")
T="$L/src"
source /work/sprint/env.sh
export CUPY_CACHE_DIR="$L/cupy-cache"
cd "$T"
bash tools/cam_wrf461_oracle/build.sh "$L/public-wrf" "$L/oracle-build" > "$L/oracle-build.log" 2>&1
GPUWM_WRF_EXACT=0 python tools/cam_wrf461_oracle/check_paths.py "$L/oracle-build" --out "$L/paths-default.json"
GPUWM_WRF_EXACT=1 python tools/cam_wrf461_oracle/check_paths.py "$L/oracle-build" --out "$L/paths-strict.json"
