#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 || $# == 3 ]] || { echo 'usage: build_driver.sh WRF_SOURCE_ROOT BUILD_DIR [LEVELS]' >&2;exit 2; }
wrf=$(realpath "$1");out=$(realpath -m "$2");levels=${3:-32};here=$(cd "$(dirname "$0")" && pwd)
[[ $levels =~ ^[0-9]+$ && $levels -ge 2 ]] || { echo 'levels must be an integer >= 2' >&2;exit 2; }
echo 'b249f472550dd2d8e594fb5d2474e280f423e5e339a9ff393d42c261098d5182  phys/module_ra_clWRF_support.F' | (cd "$wrf"; sha256sum -c -)
bash "$here/build.sh" "$wrf" "$out"
cd "$out"
flags=(-O0 -g -cpp -ffree-form -ffree-line-length-none -fno-fast-math -ffp-contract=off -fno-tree-vectorize -fallow-invalid-boz -fallow-argument-mismatch -fcheck=bounds -fconvert=big-endian)
gfortran "${flags[@]}" -c "$here/serial_stubs.F90" "$wrf/phys/module_ra_clWRF_support.F" "$wrf/phys/module_ra_cam_support.F" "$wrf/phys/module_ra_cam.F"
"${PYTHON:-python3}" "$here/make_emissivity_driver.py" "$wrf/phys/module_ra_cam.F" "$wrf/phys/module_ra_cam_support.F" "$here/../../woof/data/cam3/state-layout.json" driver.F90
"${PYTHON:-python3}" "$here/make_camrad_driver.py" "$wrf/phys/module_ra_cam.F" driver.F90 "$levels"
gfortran "${flags[@]}" -o cam_driver serial_stubs.o module_ra_clWRF_support.o module_ra_cam_support.o module_ra_cam.o driver.F90
ln -sfn "$wrf/run/CAMtr_volume_mixing_ratio.SSP245" CAMtr_volume_mixing_ratio
ln -sfn "$wrf/run/ozone_plev.formatted" ozone_plev.formatted
./cam_driver
sha256sum "$wrf/phys/module_ra_clWRF_support.F" columns.bin columns.tsv state.bin > DRIVER-SHA256SUMS
