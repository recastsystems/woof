#!/usr/bin/env bash
set -euo pipefail
test "$#" = 2 || { echo 'usage: build.sh WRF_SOURCE_ROOT BUILD_DIR' >&2; exit 2; }
wrf=$(realpath "$1")
out=$(realpath -m "$2")
here=$(cd "$(dirname "$0")" && pwd)
echo 'b0484d38338d9c7925d8011889f5a57280870febe451cfdce843b1e9ae669e32  phys/module_ra_cam.F
a7f12628b075ed4c2d45e6f3a933149007a5d1a8a0d628111e48b3b7ff394aa5  phys/module_ra_cam_support.F' | (cd "$wrf"; sha256sum -c -)
mkdir -p "$out"
cd "$out"
gfortran --version | head -1 > compiler.txt
flags=(-O0 -g -cpp -ffree-form -ffree-line-length-none -fno-fast-math -ffp-contract=off -fno-tree-vectorize -fallow-invalid-boz -fallow-argument-mismatch -fcheck=bounds -fconvert=big-endian)
gfortran "${flags[@]}" -c "$here/stubs.F90"
gfortran "${flags[@]}" -c "$wrf/phys/module_ra_cam_support.F" "$wrf/phys/module_ra_cam.F"
nm -u module_ra_cam.o module_ra_cam_support.o > undefined.txt
if grep -q _ZGV undefined.txt; then echo 'vector math in scalar oracle' >&2; exit 3; fi
"${PYTHON:-python3}" "$here/make_driver.py" "$wrf/phys/module_ra_cam.F" driver.F90
gfortran "${flags[@]}" -o cam_column stubs.o module_ra_cam_support.o module_ra_cam.o driver.F90
ln -sfn "$wrf/run/CAM_ABS_DATA" CAM_ABS_DATA
ln -sfn "$wrf/run/CAM_AEROPT_DATA" CAM_AEROPT_DATA
./cam_column
sha256sum "$wrf/phys/module_ra_cam.F" "$wrf/phys/module_ra_cam_support.F" "$wrf/run/CAM_ABS_DATA" "$wrf/run/CAM_AEROPT_DATA" columns.bin columns.tsv > SHA256SUMS
