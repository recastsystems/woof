#!/usr/bin/env bash
# build_oracle.sh MODULE_SOURCE BUILD_DIR [extra gfortran flags]
set -euo pipefail
src=$(realpath "$1"); bd=$(realpath -m "$2"); shift 2
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$bd"; cd "$bd"
F="-O2 -fno-fast-math -ffp-contract=off -ffree-form -ffree-line-length-none $*"
gfortran -c $F "$here/stub_wrf_error.F90"
gfortran -c $F -cpp -DRWORDSIZE=4 -DDWORDSIZE=8 -o module_mp_ntu.o "$src"
gfortran -c $F "$here/run_ntu.F90"
gfortran -o run_ntu stub_wrf_error.o module_mp_ntu.o run_ntu.o
if nm -u module_mp_ntu.o | grep -q _ZGV; then echo "libmvec symbols in reference" >&2; exit 4; fi
echo built "$bd/run_ntu" with "$F"
