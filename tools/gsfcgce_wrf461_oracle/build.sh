#!/usr/bin/env bash
# Build the WRF v4.6.1 Goddard GCE (mp_physics=97) column oracle.
#
#   bash tools/gsfcgce_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR> [WRF_BUILD_ROOT]
#
# WRF_BUILD_ROOT is a completed serial WRF 4.6.1 build of the same pinned
# source (default /work/wrf/build/scalar/WRF-4.6.1, the 0 ULP referee:
# gfortran 13.3.0, FCOPTIM -O2 -fno-fast-math -ffp-contract=off
# -fno-tree-vectorize).  Four arms, one driver (run_gsfcgce.F90):
#
#   ref  the referee's OWN objects: phys/module_mp_gsfcgce.o,
#        phys/module_mp_radar.o and frame/module_wrf_error.o exactly as
#        wrf.exe links them.  The arithmetic of record.
#   src  the referee's preprocessed module_mp_gsfcgce.f90 recompiled here
#        with the referee's flags.  Its disassembly must equal ref's, which
#        proves the fix arm below is built the way wrf.exe is.
#   fix  src with WOOF's three declared repairs (make_fixed_source.py) and
#        nothing else, same flags.  The referee for the configurations the
#        repairs touch; equal to ref everywhere else (compare.py checks).
#   o0   src at -O0, information only (gcc folds consat_s's constant
#        powers with correct rounding at -O2; -O0 calls powf instead).
set -euo pipefail
WRF=$(realpath "${1:?WRF_SOURCE_ROOT}")
OUT=$(realpath -m "${2:?BUILD_DIR}")
REFB=$(realpath "${3:-/work/wrf/build/scalar/WRF-4.6.1}")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PINNED=d66e442fccc04111067e29274c9f9eaccc3cef28
EXPECT_SRC=43e841f78e97a35e261f7d91d83c6ad54c7bf30a74625e933603d2e8d6126ef1
head=$(git -C "$WRF" rev-parse HEAD)
[ "$head" = "$PINNED" ] || { echo "WRF tree at $head, not $PINNED" >&2; exit 3; }
git -C "$WRF" diff --quiet HEAD -- phys/module_mp_gsfcgce.F \
    phys/module_mp_radar.F frame/module_wrf_error.F \
  || { echo "a compiled WRF GCE dependency differs from pinned HEAD" >&2; exit 3; }
got=$(sha256sum "$WRF/phys/module_mp_gsfcgce.F" | cut -d' ' -f1)
[ "$got" = "$EXPECT_SRC" ] || { echo "module_mp_gsfcgce.F sha256 $got" >&2; exit 3; }
got=$(sha256sum "$REFB/phys/module_mp_gsfcgce.F" | cut -d' ' -f1)
[ "$got" = "$EXPECT_SRC" ] || { echo "referee build compiled other GCE source" >&2; exit 3; }
for f in phys/module_mp_radar.F frame/module_wrf_error.F; do
  cmp -s "$WRF/$f" "$REFB/$f" || { echo "referee $f differs from pinned" >&2; exit 3; }
done

FLAGS="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w -ffree-form \
-ffree-line-length-none -fconvert=big-endian -frecord-marker=4 \
-fallow-argument-mismatch -fallow-invalid-boz"
# Never remove an existing build tree. Use a fresh owned output directory.
for arm in ref src fix o0; do
  if [ -e "$OUT/$arm" ]; then
    echo "output already exists: $OUT/$arm; choose a fresh BUILD_DIR" >&2
    exit 2
  fi
done
mkdir -p "$OUT/ref" "$OUT/src" "$OUT/fix" "$OUT/o0"

# Shared referee pieces: the radar and error modules (objects + .mod).
cp "$REFB/phys/module_mp_radar.o" "$REFB/phys/module_mp_radar.mod" \
   "$REFB/frame/module_wrf_error.o" "$REFB/frame/module_wrf_error.mod" "$OUT/ref/"
cp "$REFB/phys/module_mp_gsfcgce.o" "$REFB/phys/module_mp_gsfcgce.mod" "$OUT/ref/"
cp "$REFB/phys/module_mp_gsfcgce.f90" "$OUT/src/"
python3 "$HERE/make_fixed_source.py" "$OUT/src/module_mp_gsfcgce.f90" \
    "$OUT/fix/module_mp_gsfcgce.f90"
cp "$OUT/src/module_mp_gsfcgce.f90" "$OUT/o0/"

link () {   # arm dir, module object dir
  local d=$1 m=$2
  ( cd "$d"
    gfortran -O0 -c "$HERE/services.F90" -o services.o
    gfortran -O0 -I "$m" -I "$OUT/ref" -c "$HERE/run_gsfcgce.F90" -o run_gsfcgce.o
    gfortran -o run_gsfcgce run_gsfcgce.o "$m/module_mp_gsfcgce.o" \
        "$OUT/ref/module_mp_radar.o" "$OUT/ref/module_wrf_error.o" services.o )
}
link "$OUT/ref" "$OUT/ref"
for arm in src fix; do
  ( cd "$OUT/$arm"; gfortran -o module_mp_gsfcgce.o -c $FLAGS -I "$OUT/ref" module_mp_gsfcgce.f90 )
done
( cd "$OUT/o0"; gfortran -o module_mp_gsfcgce.o -c ${FLAGS/-O2/-O0} -I "$OUT/ref" module_mp_gsfcgce.f90 )
for arm in src fix o0; do
  mkdir -p "$OUT/$arm/mod"; mv "$OUT/$arm/module_mp_gsfcgce.mod" "$OUT/$arm/mod/"
  cp "$OUT/$arm/module_mp_gsfcgce.o" "$OUT/$arm/mod/"
  link "$OUT/$arm" "$OUT/$arm/mod"
done

# src must be the referee's code, instruction for instruction.
dis () { objdump -dr --no-show-raw-insn "$1" | sed -n '/^Disassembly/,$p' | tail -n +2; }
dis "$OUT/ref/module_mp_gsfcgce.o" > "$OUT/ref.dis"
dis "$OUT/src/mod/module_mp_gsfcgce.o" > "$OUT/src.dis"
if ! cmp -s "$OUT/ref.dis" "$OUT/src.dis"; then
  echo "recompiled source does not reproduce the referee object" >&2; exit 4
fi
{
  gfortran --version | sed -n 1p
  ldd --version | sed -n 1p
  echo "wrf $PINNED"
  echo "module_mp_gsfcgce.F $EXPECT_SRC"
  echo "referee object $(sha256sum "$REFB/phys/module_mp_gsfcgce.o" | cut -d' ' -f1)"
  echo "src arm reproduces referee disassembly: yes ($(wc -l < "$OUT/ref.dis") lines)"
  echo "fix source $(sha256sum "$OUT/fix/module_mp_gsfcgce.f90" | cut -d' ' -f1)"
  echo "flags $FLAGS"
} | tee "$OUT/TOOLCHAIN.txt"
