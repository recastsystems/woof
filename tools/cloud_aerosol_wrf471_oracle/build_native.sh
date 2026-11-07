#!/bin/bash
set -euo pipefail
cd "${1:?usage: build_native.sh OWNED_SCRATCH_ROOT}"
python3 - <<'PY'
import os
load = os.getloadavg()[0]
print(f'test host A load before build: {load:.2f}; jobs=2; cap=24', flush=True)
if load >= 20:
    raise SystemExit('load leaves less than four units of headroom; build deferred')
PY
cd native
OMP_NUM_THREADS=1 J='-j 2' ./compile -j 2 em_real > ../evidence/native-build.log 2>&1
test main/wrf.exe -nt phys/module_radiation_driver.F
nm main/wrf.exe | grep lane_cloud_before_ > /dev/null
nm main/wrf.exe | grep lane_farms_capture_ > /dev/null
sha256sum main/wrf.exe phys/module_radiation_driver.o phys/module_ra_farms.o > ../evidence/native-built.sha256
