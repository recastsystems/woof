#!/usr/bin/env bash
set -euo pipefail
L=$(realpath "$1")
T="$L/src"
if [[ "$L" == /work/sprint/* ]]; then source /work/sprint/env.sh; wrf="$L/public-wrf"; else source /work/env.sh; source /work/wrf/env.sh; wrf="$WRF_SOURCE_ROOT"; fi
export PYTHONPATH="$T"
export CUPY_CACHE_DIR="$L/cupy-cache"
cd "$T"
bash tools/cam_wrf461_oracle/build.sh "$wrf" "$L/oracle-build" > "$L/oracle-build.log" 2>&1
python tools/cam_wrf461_oracle/make_emissivity_driver.py "$wrf/phys/module_ra_cam.F" "$wrf/phys/module_ra_cam_support.F" woof/data/cam3/state-layout.json "$L/oracle-build/driver.F90"
(cd "$L/oracle-build"; gfortran -O0 -g -fno-fast-math -ffp-contract=off -fno-tree-vectorize -ffree-form -ffree-line-length-none -fconvert=big-endian -fcheck=bounds -o cam_emissivity stubs.o module_ra_cam_support.o module_ra_cam.o driver.F90; ./cam_emissivity) >> "$L/oracle-build.log" 2>&1
set +e
status=0
for mode in 0 1; do
 for spectrum in emissivity lw sw; do
  GPUWM_WRF_EXACT=$mode python tools/cam_wrf461_oracle/check_flux_gpu.py "$L/oracle-build" --spectrum "$spectrum" --out "$L/$spectrum-mode-$mode.json"
  result=$?
  echo "$spectrum mode=$mode status=$result"
  ((status|=result))
 done
done
python tools/cam_wrf461_oracle/check_math_gpu.py --out "$L/math-probe.json"
exit "$status"
