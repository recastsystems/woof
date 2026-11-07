#!/usr/bin/env bash
set -euo pipefail
root=$(realpath "$1"); out=$(realpath -m "$2"); here=$(cd "$(dirname "$0")" && pwd)
bash "$here/../noah_wrf461_oracle/build.sh" "$root" "$out"
cd "$out"
gfortran -O0 -ffree-form -ffree-line-length-none -c "$root/phys/module_irrigation.F"
gfortran -O0 -ffree-line-length-none -I . -c "$here/run_soil.F90"
gfortran -o run_soil stub_wrf.o module_model_constants.o module_wrf_error.o module_sf_noahlsm.o module_sf_noahlsm_glacial_only.o module_sf_noahdrv.o module_irrigation.o run_soil.o
./run_soil channel 1 F F F 1 > channel-stdout.txt
./run_soil drip 1 F F F 2 > drip-stdout.txt
sha256sum "$here/run_soil.F90" "$root/phys/module_irrigation.F" channel-step*.csv drip-step*.csv > soil-SHA256SUMS
