#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: build.sh WRF_SOURCE_ROOT BUILD_DIR' >&2; exit 2; }
R=$(realpath "$1"); B=$(realpath -m "$2"); S=$(cd "$(dirname "$0")" && pwd)
P=d66e442fccc04111067e29274c9f9eaccc3cef28
[[ $(git -C "$R" rev-parse HEAD) == "$P" ]] || { echo 'WRF revision mismatch' >&2; exit 3; }
files=(phys/module_ra_gfdleta.F phys/module_mp_etanew.F share/module_model_constants.F run/tr49t85 run/tr49t67 run/tr67t85 run/co2_trans)
git -C "$R" diff --quiet HEAD -- "${files[@]}" || { echo 'WRF reference source or tables modified' >&2; exit 3; }
mkdir -p "$B"; cd "$B"
flags=(-O0 -DRWORDSIZE=4 -DIWORDSIZE=4 -ffp-contract=off -fno-tree-vectorize -fcheck=all -fbacktrace -fconvert=big-endian -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch -ffunction-sections -fdata-sections)
gfortran "${flags[@]}" -c "$S/stub_runtime.F90"
gfortran "${flags[@]}" -c "$R/share/module_model_constants.F"
gfortran "${flags[@]}" -c "$R/phys/module_mp_etanew.F"
gfortran "${flags[@]}" -c "$R/phys/module_ra_gfdleta.F"
gfortran "${flags[@]}" -c "$S/run_gfdleta.F90"
gfortran -Wl,--gc-sections -o run_gfdleta stub_runtime.o module_model_constants.o module_mp_etanew.o module_ra_gfdleta.o run_gfdleta.o
for f in tr49t85 tr49t67 tr67t85; do ln -sfn "$R/run/$f" "$B/$f"; done
gfortran "${flags[@]}" -c "$S/run_primitives.F90"
gfortran -Wl,--gc-sections -o run_primitives stub_runtime.o module_model_constants.o module_mp_etanew.o module_ra_gfdleta.o run_primitives.o
gfortran "${flags[@]}" -c "$S/run_lw_helpers.F90"
gfortran -Wl,--gc-sections -o run_lw_helpers stub_runtime.o module_model_constants.o module_mp_etanew.o module_ra_gfdleta.o run_lw_helpers.o
printf '%s\n' "$P" > wrf-commit.txt
gfortran --version | head -1 > compiler.txt
sha256sum "$R/phys/module_ra_gfdleta.F" "$R/phys/module_mp_etanew.F" "$R/share/module_model_constants.F" "$R"/run/{tr49t85,tr49t67,tr67t85,co2_trans} "$S"/build.sh "$S"/stub_runtime.F90 "$S"/run_gfdleta.F90 "$S"/run_primitives.F90 "$S"/run_lw_helpers.F90 "$S"/run_suite.sh "$S"/validate_gfdleta.py > oracle-source-sha256.txt
