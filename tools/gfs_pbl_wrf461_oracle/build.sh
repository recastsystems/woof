#!/usr/bin/env bash
# Build the WRF v4.6.1 GFS PBL (bl_pbl_physics=3) column oracle.
#
#   bash tools/gfs_pbl_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
#
# Compiles the byte-unmodified phys/module_gfs_machine.F,
# phys/module_gfs_physcons.F and phys/module_bl_gfs.F plus
# share/module_model_constants.F, preprocessed with WRF's own cpp line for
# an ARW build (configure.wrf CPP/COREDEFS: no HWRF, NMM_CORE=0), at the
# flags of the scalar WRF referee build (configure.wrf FCOPTIM:
# -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize).  Refuses a
# source whose SHA-256 is not the v4.6.1 file's, so a fixture can never
# come from an edited copy, and refuses an object that calls a vector libm
# entry (_ZGV*), so the reference is the scalar libm it claims to be.
set -euo pipefail
SRC=${1:?WRF source root}
OUT=${2:?build directory}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
declare -A PIN=(
  [share/module_model_constants.F]=5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062
  [phys/module_bl_gfs.F]=a3898642ee386d60a74af2ef45867e6b5f7df518eef9543ae91586d00372b71c
  [phys/module_gfs_machine.F]=300ee4f75663e89d34c8a9b8733cc07a5dd48c3d973da019d31cb76524babe25
  [phys/module_gfs_physcons.F]=2b8ef99663ba62748b57c7002f5e5b8c9338f2a825c1d32c113bd4e3bff2eb52
)
for f in "${!PIN[@]}"; do
  got=$(sha256sum "$SRC/$f" | cut -d' ' -f1)
  if [ "$got" != "${PIN[$f]}" ]; then
    echo "refusing: $f is $got, not the WRF v4.6.1 file ${PIN[$f]}" >&2
    exit 2
  fi
done
CPPDEFS="-DEM_CORE=1 -DNMM_CORE=0 -DNMM_MAX_DIM=2600 -DDA_CORE=0 -DWRFPLUS=0 -DCOAMPS_CORE=0 -DEXP_CORE=0 -DIWORDSIZE=4 -DDWORDSIZE=8 -DRWORDSIZE=4 -DLWORDSIZE=4"
FFLAGS="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w -ffree-form -ffree-line-length-none"
cd "$OUT"
for m in share/module_model_constants phys/module_gfs_machine phys/module_gfs_physcons phys/module_bl_gfs; do
  b=$(basename "$m")
  /lib/cpp -P -nostdinc -traditional-cpp $CPPDEFS "$SRC/$m.F" > "$b.f90"
  gfortran $FFLAGS -c "$b.f90" -o "$b.o"
done
gfortran $FFLAGS -c "$HERE/run_bl_gfs.F90" -o run_bl_gfs.o
gfortran -o run_bl_gfs run_bl_gfs.o module_bl_gfs.o module_gfs_physcons.o \
  module_gfs_machine.o module_model_constants.o -lm
if nm module_bl_gfs.o | grep -q '_ZGV'; then
  echo "refusing: module_bl_gfs.o calls a vector libm entry" >&2
  nm module_bl_gfs.o | grep '_ZGV' >&2
  exit 3
fi
{
  echo "gfortran: $(gfortran --version | head -1)"
  echo "glibc: $(ldd --version | head -1)"
  echo "flags: $FFLAGS"
  echo "cpp: /lib/cpp -P -nostdinc -traditional-cpp $CPPDEFS"
  echo "libm calls in module_bl_gfs.o: $(nm -u module_bl_gfs.o | awk '{print $2}' | grep -v '^_gfortran' | tr '\n' ' ')"
  for f in "${!PIN[@]}"; do echo "sha256 $f ${PIN[$f]}"; done
  echo "sha256 share/module_model_constants.F $(sha256sum "$SRC/share/module_model_constants.F" | cut -d' ' -f1)"
} > toolchain.txt
cat toolchain.txt
