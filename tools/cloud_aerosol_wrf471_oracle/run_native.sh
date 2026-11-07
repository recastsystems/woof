#!/bin/bash
set -euo pipefail
oracle_dir=$(cd "$(dirname "$0")" && pwd)
cd "${1:?usage: run_native.sh OWNED_SCRATCH_ROOT}"
lane_root=$PWD
python3 - <<'PY'
import os
load=os.getloadavg()[0]
print(f'test host A load before native run: {load:.2f}; MPI ranks=2; cap=24',flush=True)
if load>=20: raise SystemExit('native run deferred for CPU headroom')
PY
mkdir -p native-case native-capture
cd native-case
for source_path in "$lane_root"/native/run/*; do
  filename=$(basename "$source_path")
  if [[ "$filename" != namelist.input && "$filename" != wrf.exe && "$filename" != real.exe ]]; then
    [[ -e "$filename" ]] || ln -s "$source_path" "$filename"
  fi
done
for source_path in /workspace/shared-prepared/wrf-jan00-30km/met_em.d01.*.nc; do
  [[ -e $(basename "$source_path") ]] || ln -s "$source_path" .
done
cp "$oracle_dir/native_namelist.input" namelist.input
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$lane_root/native-deps/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
mpirun --oversubscribe -np 2 "$lane_root/native/main/real.exe" > "$lane_root/evidence/native-real.stdout.log" 2>&1
cp rsl.error.0000 "$lane_root/evidence/native-real.rsl.log"
grep -q 'SUCCESS COMPLETE REAL' "$lane_root/evidence/native-real.rsl.log"
sha256sum wrfinput_d01 wrfbdy_d01 > "$lane_root/evidence/native-input.sha256"
WOOF_CAPTURE_DIR="$lane_root/native-capture" mpirun --oversubscribe -np 2 "$lane_root/native/main/wrf.exe" > "$lane_root/evidence/native-forecast.stdout.log" 2>&1
cp rsl.error.0000 "$lane_root/evidence/native-forecast.rsl.log"
grep -q 'SUCCESS COMPLETE WRF' "$lane_root/evidence/native-forecast.rsl.log"
sha256sum "$lane_root"/native-capture/*.bin > "$lane_root/evidence/native-capture.sha256"
