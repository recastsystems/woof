#!/usr/bin/env bash
# Link the column driver against the native WRF v4.8.0 Shin-Hong objects of
# the strict referee build, unmodified, and run the column corpus.
#
#   build.sh WRF_480_BUILD OUT_DIR [IN.bin ...]     (inside wrfref2404:1)
#
# WRF_480_BUILD is the referee tree (a development machine ~/wrf480-port/build/
# WRF-4.8.0-serial-pristine: GNU Fortran 13.3, FCOPTIM -O2 -fno-fast-math
# -ffp-contract=off -fno-tree-vectorize).  The objects are the ones its
# wrf.exe links: phys/module_bl_shinhong.o, phys/physics_mmm/bl_shinhong.o and
# phys/ccpp_kind_types.o.  Their sources are pinned by sha256 below; the
# driver is compiled -O0 (it only copies words).  For each IN.bin the
# native outputs are written beside it as IN.wrf-out.bin.
set -euo pipefail
B=$(realpath "$1")
O=$(realpath -m "$2")
shift 2
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$O"
test "$(sha256sum "$B/phys/physics_mmm/bl_shinhong.F90" | cut -d" " -f1)" = 7d2588364d95519024d4914273e6a4953f1baa393261e0035f9a7198fbc9d961
test "$(sha256sum "$B/phys/module_bl_shinhong.F" | cut -d" " -f1)" = c7bfe36353a9db6f4fd471a5ca13287efb4b92d5c12ff2797af685eb536cb5dc
grep -q -- '-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize' "$B/configure.wrf"
cd "$O"
gfortran -O0 -ffree-form -ffree-line-length-none -I "$B/phys" \
    -c "$here/run_shinhong480_columns.F90" -o run_shinhong480_columns.o
gfortran -o run_shinhong480_columns run_shinhong480_columns.o \
    "$B/phys/module_bl_shinhong.o" "$B/phys/physics_mmm/bl_shinhong.o" \
    "$B/phys/ccpp_kind_types.o"
{
  gfortran --version | head -1
  ldd --version | head -1
  sha256sum "$B/phys/physics_mmm/bl_shinhong.F90" "$B/phys/module_bl_shinhong.F" \
      "$B/phys/physics_mmm/bl_shinhong.o" "$B/phys/module_bl_shinhong.o" \
      "$B/main/wrf.exe" "$here/run_shinhong480_columns.F90" run_shinhong480_columns
  grep -E '^(FCOPTIM|FCBASEOPTS_NO_G|PROMOTION)' "$B/configure.wrf"
  echo "libm call sites in bl_shinhong.o:"
  objdump -dr "$B/phys/physics_mmm/bl_shinhong.o" | awk '/R_X86_64_PLT32/ {print $3}' \
      | sed 's/-0x4$//' | grep -E 'pow|exp|tanh|log|sqrt' | sort | uniq -c
  echo "fma instructions in bl_shinhong.o: $(objdump -d "$B/phys/physics_mmm/bl_shinhong.o" | grep -c -E 'vfmadd|vfmsub|vfnm' || true)"
} > receipt.txt
# The labelled-PATCHED arm: the same objects with the wrapper's TKE copy-in
# restored (patch_tke_copyin.py), compiled with the referee's own wrapper
# flags (compile.log of the referee tree).
mkdir -p patched
python3 "$here/patch_tke_copyin.py" "$B/phys/module_bl_shinhong.f90" patched/module_bl_shinhong.f90   || /usr/bin/env python3 "$here/patch_tke_copyin.py" "$B/phys/module_bl_shinhong.f90" patched/module_bl_shinhong.f90
( cd patched && gfortran -o module_bl_shinhong.o -c -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w     -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch     -fallow-invalid-boz -I "$B/phys" module_bl_shinhong.f90   && gfortran -O0 -ffree-form -ffree-line-length-none -I . -I "$B/phys"     -c "$here/run_shinhong480_columns.F90" -o run_shinhong480_columns.o   && gfortran -o run_shinhong480_columns run_shinhong480_columns.o module_bl_shinhong.o     "$B/phys/physics_mmm/bl_shinhong.o" "$B/phys/ccpp_kind_types.o" )
sha256sum patched/module_bl_shinhong.f90 patched/module_bl_shinhong.o patched/run_shinhong480_columns >> receipt.txt
diff "$B/phys/module_bl_shinhong.f90" patched/module_bl_shinhong.f90 >> receipt.txt || true
for f in "$@"; do
  ./run_shinhong480_columns "$f" "${f%.bin}.wrf-out.bin"
  patched/run_shinhong480_columns "$f" "${f%.bin}.wrf-patched-out.bin"
done
cat receipt.txt
