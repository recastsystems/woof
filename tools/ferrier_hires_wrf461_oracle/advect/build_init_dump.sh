#!/usr/bin/env bash
# Dump fer_hires_init's tables and constants (and FER_HIRES's per-call
# constants) from WRF v4.6.1 for the port's init to be graded against.
# usage: build_init_dump.sh WRF_SOURCE_ROOT BUILD_DIR
set -euo pipefail
src=$(realpath "$1"); out=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); harness=$(dirname "$here")
[[ $(git -C "$src" rev-parse HEAD) == d66e442fccc04111067e29274c9f9eaccc3cef28 ]]
git -C "$src" diff --quiet HEAD -- phys/module_mp_fer_hires.F run/ETAMPNEW_DATA.expanded_rain
mkdir -p "$out"; cd "$out"
fer="$src/phys/module_mp_fer_hires.F"
# Insert the dump subroutine before END MODULE; nothing else changes.
n=$(grep -n '^ *END MODULE module_mp_fer_hires' "$fer" | cut -d: -f1)
head -n $((n - 1)) "$fer" > module_mp_fer_hires_dump.F
cat "$here/dump_sub.F90" >> module_mp_fer_hires_dump.F
tail -n +"$n" "$fer" >> module_mp_fer_hires_dump.F
common=(-O0 -cpp -ffree-form -ffree-line-length-none -DRWORDSIZE=4 -DIWORDSIZE=4 -fallow-argument-mismatch)
gfortran -c "${common[@]}" "$harness/stub_wrf.F90"
gfortran -c "${common[@]}" module_mp_fer_hires_dump.F 2> compile.stderr
gfortran -c "${common[@]}" -fconvert=big-endian "$here/run_init_dump.F90"
gfortran -o run_init_dump stub_wrf.o module_mp_fer_hires_dump.o run_init_dump.o
ln -sf "$src/run/ETAMPNEW_DATA.expanded_rain" ETAMPNEW_DATA.expanded_rain
./run_init_dump > run.stdout 2> run.stderr
sha256sum init_dump.csv
