#!/usr/bin/env bash
set -euo pipefail
L=$(realpath "$1");T="$L/src"
if [[ "$L" == /work/sprint/* ]]; then source /work/sprint/env.sh; wrf="$L/public-wrf"; else source /work/env.sh; source /work/wrf/env.sh; wrf="$WRF_SOURCE_ROOT"; fi
export PYTHONPATH="$T" CUPY_CACHE_DIR="$L/cupy-cache"
cd "$T"
bash tools/cam_wrf461_oracle/build.sh "$wrf" "$L/driver-oracle" > "$L/driver-oracle.log" 2>&1
python tools/cam_wrf461_oracle/make_emissivity_driver.py "$wrf/phys/module_ra_cam.F" "$wrf/phys/module_ra_cam_support.F" woof/data/cam3/state-layout.json "$L/driver-oracle/driver.F90"
python tools/cam_wrf461_oracle/make_camrad_driver.py "$wrf/phys/module_ra_cam.F" "$L/driver-oracle/driver.F90"
python - "$L/driver-oracle" <<'PY'
import re,sys
from pathlib import Path
p=Path(sys.argv[1]);s=Path('tools/cam_wrf461_oracle/stubs.F90').read_text();s=re.sub(r'(?is)module module_ra_clwrf_support\b.*?end module','',s);s+='\ninteger function get_unused_unit()\nget_unused_unit=37\nend function\n';(p/'serial-stubs.F90').write_text(s)
PY
(cd "$L/driver-oracle"
flags=(-O0 -g -ffree-form -fno-fast-math -ffp-contract=off -fno-tree-vectorize -fconvert=big-endian -ffree-line-length-none -fcheck=bounds)
gfortran "${flags[@]}" -c serial-stubs.F90 "$wrf/phys/module_ra_clWRF_support.F"
gfortran "${flags[@]}" -o cam_driver serial-stubs.o module_ra_clWRF_support.o module_ra_cam.o module_ra_cam_support.o driver.F90
ln -sfn "$wrf/run/CAMtr_volume_mixing_ratio" CAMtr_volume_mixing_ratio
ln -sfn "$wrf/run/ozone_plev.formatted" ozone_plev.formatted
./cam_driver) >> "$L/driver-oracle.log" 2>&1
set +e
status=0
for mode in 0 1; do
 GPUWM_WRF_EXACT=$mode python tools/cam_wrf461_oracle/check_camrad_gpu.py "$wrf/phys/module_ra_cam.F" "$L/driver-oracle" --out "$L/camrad-mode-$mode.json"
 result=$?;echo "camrad mode=$mode status=$result";((status|=result))
done
exit "$status"
