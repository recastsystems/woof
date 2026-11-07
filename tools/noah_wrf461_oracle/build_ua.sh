#!/usr/bin/env bash
# Independent WRF 4.6.1 UA oracle. The original 42-column harness stays unchanged.
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
ROOT=$(realpath "$1")
B=$(realpath -m "$2")
bash "$S/build.sh" "$ROOT" "$B"
cd "$B"
# The original builder pins the Noah driver and its parameter tables.
# Pin the additional diagnostic source against the same WRF commit.
test "$(git -C "$ROOT" hash-object phys/module_sf_sfcdiags.F)" = "$(git -C "$ROOT" rev-parse HEAD:phys/module_sf_sfcdiags.F)"
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none "$ROOT/phys/module_sf_sfcdiags.F"
gfortran -c -O0 -ffree-form -ffree-line-length-none -I "$B" "$S/run_lsm_ua.F90"
gfortran -o run_lsm_ua stub_wrf.o module_model_constants.o module_wrf_error.o module_sf_noahlsm.o module_sf_noahlsm_glacial_only.o module_sf_noahdrv.o module_sf_sfcdiags.o run_lsm_ua.o
./run_lsm_ua noah-lsm-extended.csv 1 F F F F > run-extended.log
./run_lsm_ua noah-lsm-ua.csv 1 F F F T > run-ua.log
./run_lsm_ua noah-lsm-ua-monalb.csv 1 F T T T > run-ua-monalb.log
sha256sum "$ROOT/phys/module_sf_sfcdiags.F" "$S/run_lsm_ua.F90" noah-lsm-extended.csv noah-lsm-ua.csv noah-lsm-ua-monalb.csv > ua-sha256sums.txt


test "$(git -C "$ROOT" hash-object phys/module_ra_gfdleta.F)" = "$(git -C "$ROOT" rev-parse HEAD:phys/module_ra_gfdleta.F)"
sed '/^subroutine cal_mon_day()/,/^end subroutine cal_mon_day/d' "$S/stub_wrf.F90" > stub_mosaic.F90
awk '/^ *SUBROUTINE CAL_MON_DAY/,/^ *END SUBROUTINE CAL_MON_DAY/' "$ROOT/phys/module_ra_gfdleta.F" > cal_mon_day.F90
gfortran -c -O0 -ffree-form -ffree-line-length-none stub_mosaic.F90 cal_mon_day.F90
ulimit -c 0

gfortran -c -O0 -ffree-form -ffree-line-length-none -I "$B" "$S/run_mosaic_ua.F90"
gfortran -o run_mosaic_ua stub_mosaic.o cal_mon_day.o module_model_constants.o module_wrf_error.o module_sf_noahlsm.o module_sf_noahlsm_glacial_only.o module_sf_noahdrv.o module_sf_sfcdiags.o run_mosaic_ua.o
./run_mosaic_ua noah-mosaic-ua.csv 1 F F F T > run-mosaic-ua.log
./run_mosaic_ua noah-mosaic-false.csv 1 F F F F > run-mosaic-false.log
sha256sum "$S/run_mosaic_ua.F90" noah-mosaic-ua.csv noah-mosaic-false.csv >> ua-sha256sums.txt

# Explicit control for WRF's undefined PC after complete canopy burial.
# The untouched referee above remains the qualification authority.
python - "$ROOT" "$B" <<'CONTROL'
from pathlib import Path
import sys
root,b=map(Path,sys.argv[1:])
s=(root/"phys/module_sf_noahlsm.F").read_text()
needle="         ELSE\n            RC = 0.0\n         END IF"
assert s.count(needle)==1
s=s.replace(needle,"         ELSE\n            RC = 0.0\n            IF(UA_PHYS) PC = 0.0\n         END IF")
(b/"noah_pc_control.F").write_text(s)
CONTROL
gfortran -c -O0 -cpp -Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4 -ffree-form -ffree-line-length-none -fallow-argument-mismatch -I "$B" noah_pc_control.F
gfortran -o run_pc_control stub_wrf.o module_model_constants.o module_wrf_error.o noah_pc_control.o module_sf_noahlsm_glacial_only.o module_sf_noahdrv.o module_sf_sfcdiags.o run_lsm_ua.o
./run_pc_control noah-lsm-ua-pc-control.csv 1 F F F T > run-pc-control.log
./run_pc_control noah-lsm-ua-monalb-pc-control.csv 1 F T T T > run-monalb-pc-control.log
gfortran -o run_mosaic_pc_control stub_mosaic.o cal_mon_day.o module_model_constants.o module_wrf_error.o noah_pc_control.o module_sf_noahlsm_glacial_only.o module_sf_noahdrv.o module_sf_sfcdiags.o run_mosaic_ua.o
./run_mosaic_pc_control noah-mosaic-ua-pc-control.csv 1 F F F T > run-mosaic-pc-control.log
sha256sum noah_pc_control.F noah*-pc-control.csv >> ua-sha256sums.txt

gfortran -c -O0 -ffree-form -ffree-line-length-none -I "$B" "$S/run_mosaic_two_ua.F90"
gfortran -o run_mosaic_two stub_mosaic.o cal_mon_day.o module_model_constants.o module_wrf_error.o module_sf_noahlsm.o module_sf_noahlsm_glacial_only.o module_sf_noahdrv.o module_sf_sfcdiags.o run_mosaic_two_ua.o
./run_mosaic_two noah-mosaic-ua-two.csv 1 F F F T > run-mosaic-two.log
gfortran -o run_mosaic_two_control stub_mosaic.o cal_mon_day.o module_model_constants.o module_wrf_error.o noah_pc_control.o module_sf_noahlsm_glacial_only.o module_sf_noahdrv.o module_sf_sfcdiags.o run_mosaic_two_ua.o
./run_mosaic_two_control noah-mosaic-ua-two-pc-control.csv 1 F F F T > run-mosaic-two-control.log
sha256sum "$S/run_mosaic_two_ua.F90" noah-mosaic-ua-two*.csv >> ua-sha256sums.txt
