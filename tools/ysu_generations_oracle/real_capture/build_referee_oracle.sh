#!/usr/bin/env bash
# Link the column driver (tools/ysu_wrf461_oracle/run_bl_ysu_wide.F90) against
# the bl_ysu.o and ccpp_kind_types.o objects the native WRF 4.8.0 strict
# referee build itself compiled, so the standalone oracle runs the referee's
# own machine code.  Run inside localhost/wrfref2404:1 (GNU Fortran 13.3.0).
#   build_referee_oracle.sh WRF_BUILD ENGINE_ROOT OUT
set -euo pipefail
W=$(realpath "$1"); T=$(realpath "$2"); O=$(realpath -m "$3")
mkdir -p "$O"; cd "$O"
obj=$W/phys/physics_mmm/bl_ysu.o
kind=""; for k in "$W"/phys/ccpp_kind_types.o "$W"/phys/physics_mmm/ccpp_kind_types.o; do [ -f "$k" ] && { kind=$k; break; }; done
test -f "$obj" && test -n "$kind"
flags=(-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -ffree-form -ffree-line-length-none)
gfortran "${flags[@]}" -I"$W/phys" -I"$W/phys/physics_mmm" -c "$T/tools/ysu_wrf461_oracle/run_bl_ysu_wide.F90"
gfortran -o oracle "$kind" "$obj" run_bl_ysu_wide.o
{
  gfortran --version | head -1
  echo "flags=${flags[*]}"
  sha256sum "$obj" "$kind" "$W/main/wrf.exe" "$T/tools/ysu_wrf461_oracle/run_bl_ysu_wide.F90" oracle
} > receipt.txt
cat receipt.txt
