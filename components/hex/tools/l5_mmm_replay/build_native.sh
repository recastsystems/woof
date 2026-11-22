#!/bin/bash
# Standalone native referees (test tools, never on a run path): pristine NCAR/MMM-physics bl_ysu.F90 and
# sf_sfclayrev.F90 at a4baf7f3243d and 550b5b494758, MPAS v8.4.2 physics_wrf/module_bl_ysu.F, MPAS strict GNU
# single-precision flags (tools/native_physics_reference/build_reference.sh gnu-strict).
# Usage: build_native.sh WORK SRC   SRC holds {a4baf7f,550b5b4}_bl_ysu.F90, {a4baf7f,550b5b4}_sf_sfclayrev.F90 and
# module_bl_ysu.F (fetched from github.com/NCAR/MMM-physics and MPAS-Dev/MPAS-Model v8.4.2; hashes in README.md).
set -euo pipefail
WORK=$(realpath "$1"); SRC=$(realpath "$2")
HERE=$(dirname "$(realpath "$0")")
T=$HERE/../../tests/mpas_ysu_v842
F="-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -fconvert=big-endian -ffree-form -fPIC -cpp -Dmpas"
for v in a4baf7f 550b5b4; do
  d=$WORK/build-$v; rm -rf "$d"; mkdir -p "$d"; cd "$d"
  cp "$T/kinds.F90" "$T/native_bridge.F90" "$SRC/module_bl_ysu.F" "$HERE/mm5_bridge.F90" .
  cp "$SRC/${v}_bl_ysu.F90" bl_ysu.F90; cp "$SRC/${v}_sf_sfclayrev.F90" sf_sfclayrev.F90
  D=""; [ "$v" = 550b5b4 ] && D="-DMMM550"
  gfortran $F -c kinds.F90 bl_ysu.F90 module_bl_ysu.F native_bridge.F90
  gfortran -shared -o libysu.so kinds.o bl_ysu.o module_bl_ysu.o native_bridge.o
  gfortran $F $D -c sf_sfclayrev.F90 mm5_bridge.F90
  gfortran -shared -o libmm5.so kinds.o sf_sfclayrev.o mm5_bridge.o
  sha256sum bl_ysu.F90 sf_sfclayrev.F90
done
gcc -O2 -shared -fPIC -o "$HERE/ftz/l5ftz.so" "$HERE/ftz/l5ftz.c"
