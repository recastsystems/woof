#!/usr/bin/env bash
# Dump WRF v4.6.1 thompson_init's collision seeds as hex words.
# usage: build.sh WRF_PHYS_DIR TABLE_DIR BUILD_DIR   (writes BUILD_DIR/constants.txt)
set -euo pipefail
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
oracle=$(CDPATH= cd -- "$here/../thompson_wrf461_oracle" && pwd)
phys=$(realpath "$1"); tables=$(realpath "$2"); build=$(realpath -m "$3")
echo "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802  $phys/module_mp_thompson.F" | sha256sum -c -
mkdir -p "$build"; cd "$build"
python3 "$here/dump_seeds.py" "$phys/module_mp_thompson.F" module_mp_thompson_dump.F
f="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -ffree-form -ffree-line-length-none"
gfortran -c $f "$oracle/stub_wrf.F90"
gfortran -c $f "$phys/module_mp_radar.F"
gfortran -c $f -cpp -DWRF_CHEM=0 module_mp_thompson_dump.F -o module_mp_thompson.o
gfortran -c $f "$here/run_dump.F90"
gfortran $f -o run_dump stub_wrf.o module_mp_radar.o module_mp_thompson.o run_dump.o
for t in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat; do ln -sf "$tables/$t" "$t"; done
GFORTRAN_CONVERT_UNIT='big_endian:20' ./run_dump | grep -E '^ ?[RDI] ' > constants.txt
wc -l constants.txt

python3 "$here/make_header.py" constants.txt collision_constants.cuh
cmp constants.txt "$here/seeds-wrf461.txt"
