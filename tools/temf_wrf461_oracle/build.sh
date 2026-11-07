#!/usr/bin/env bash
# Build the WRF v4.6.1 TEMF column oracle (bl_pbl_physics = 10 with
# sf_sfclay_physics = 10) from a pristine WRF source tree.
#
#   bash tools/temf_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
#
# Both modules are pinned by sha256 to the v4.6.1 files.  They have no USE
# statement; module_sf_temfsfclay.F calls WRF_DEBUG from its init routine,
# which stub_wrf.F90 supplies, so nothing else of WRF is linked.
#
# Arms (every arm runs the same driver, run_temf.F90):
#   stock     the byte-unmodified modules at WRF's own optimisation on this
#             box (-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize,
#             /work/wrf/provenance/flags-scalar.txt), atanf/atan2f/log10f
#             correctly rounded (cr_shim.c).  Proves the transcription.
#   fixed     the modules with WOOF's three declared fixes (apply_fixes.py),
#             same flags and shim.  THE ORACLE for the 0 ULP claim on the
#             shipped kernel.
#   stock-O0  stock at -O0: shows the -O2 build folds nothing the port
#             does not.
#   stock-raw stock without the shim: what glibc 2.39's own atanf, atan2f
#             and log10f give.  Reported, not graded.
#   only-Fn   stock plus the single declared fix Fn (n = 1..6): what each
#             fix moves on its own, for the divergence table.
set -euo pipefail
WRF=${1:?usage: build.sh WRF_SOURCE_ROOT BUILD_DIR}
OUT=${2:?usage: build.sh WRF_SOURCE_ROOT BUILD_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
PBL="$WRF/phys/module_bl_temf.F"
SFC="$WRF/phys/module_sf_temfsfclay.F"
EXPECT_PBL=0c4d731ca3124198342a3a2f90e0a43f34ddf5731b545460fb27bf794ed77475
EXPECT_SFC=70f2ed6bc48aad5b1bf68be4e01753a132f2ecdb4fa6d95c0ecce5d5554fcd94
for pair in "$PBL:$EXPECT_PBL" "$SFC:$EXPECT_SFC"; do
  f=${pair%%:*}; want=${pair##*:}
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$got" = "$want" ] || { echo "FATAL: $f sha256 $got != $want" >&2; exit 1; }
done

mkdir -p "$OUT/src-fixed"
python3 "$HERE/apply_fixes.py" "$WRF" "$OUT/src-fixed"
( cd "$OUT" && for f in module_bl_temf.F module_sf_temfsfclay.F; do
    diff -u --label "a/phys/$f" --label "b/phys/$f" "$WRF/phys/$f" \
        "src-fixed/$f" || true
  done ) > "$OUT/declared-fixes.diff"

FREE="-cpp -ffree-form -ffree-line-length-none"
WRFOPT="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize"
gcc -O2 -c "$HERE/cr_shim.c" -o "$OUT/cr_shim.o"

build () {
  local arm=$1 srcdir=$2 opt=$3 shim=$4
  local d="$OUT/$arm"
  rm -rf "$d"; mkdir -p "$d"
  ( cd "$d"
    gfortran $FREE $opt -c "$srcdir/module_sf_temfsfclay.F" -o module_sf_temfsfclay.o
    gfortran $FREE $opt -c "$srcdir/module_bl_temf.F" -o module_bl_temf.o
    gfortran $FREE -O0 -c "$HERE/stub_wrf.F90" -o stub_wrf.o
    gfortran $FREE -O0 -I . -c "$HERE/run_temf.F90" -o run_temf.o
    if [ "$shim" = yes ]; then
      gfortran run_temf.o module_bl_temf.o module_sf_temfsfclay.o stub_wrf.o \
          "$OUT/cr_shim.o" -lquadmath -o run_temf
    else
      gfortran run_temf.o module_bl_temf.o module_sf_temfsfclay.o stub_wrf.o \
          -o run_temf
    fi
    nm -u module_bl_temf.o module_sf_temfsfclay.o | grep -v ':$' \
        | sed 's/^ *U *//' | sort -u > undefined.txt )
  echo "built arm $arm ($opt, shim=$shim)"
}
build stock     "$WRF/phys"        "$WRFOPT" yes
build fixed     "$OUT/src-fixed"   "$WRFOPT" yes
build stock-O0  "$WRF/phys"        "-O0"     yes
build stock-raw "$WRF/phys"        "$WRFOPT" no
for fx in F1 F2 F3 F4 F5 F6; do
  mkdir -p "$OUT/src-only-$fx"
  python3 "$HERE/apply_fixes.py" "$WRF" "$OUT/src-only-$fx" "$fx" > /dev/null
  build "only-$fx" "$OUT/src-only-$fx" "$WRFOPT" yes
done

# The kernel's constant table (woof/core/temf_constants.py) against
# gfortran's own folding of the same Fortran text, word for word.
TREE=$(cd "$HERE/../.." && pwd)
if [ -f "$TREE/woof/core/temf_constants.py" ]; then
( cd "$TREE" && python3 -m woof.core.temf_constants --fortran-probe ) \
    > "$OUT/constants_probe.f90"
gfortran -O0 -ffree-line-length-none "$OUT/constants_probe.f90" \
    -o "$OUT/constants_probe"
"$OUT/constants_probe" | sed 's/^ *//' > "$OUT/constants-gfortran.txt"
( cd "$TREE" && python3 -m woof.core.temf_constants --table ) \
    > "$OUT/constants-table.txt"
if ! diff "$OUT/constants-gfortran.txt" "$OUT/constants-table.txt"; then
  echo "FATAL: a TEMFC word is not gfortran's folding of its text" >&2; exit 6
fi
echo "constant table: $(wc -l < "$OUT/constants-table.txt") words equal gfortran's"
fi

if grep -q '_ZGV' "$OUT/stock/undefined.txt"; then
  echo "FATAL: the stock arm calls a libmvec vector routine" >&2; exit 4
fi
{
  gfortran --version | sed -n 1p
  ldd --version | sed -n 1p
  echo "module_bl_temf.F $EXPECT_PBL"
  echo "module_sf_temfsfclay.F $EXPECT_SFC"
  echo "WRF FCOPTIM on this box: $WRFOPT"
  echo "libm entries the modules call: $(tr '\n' ' ' < "$OUT/stock/undefined.txt")"
  for arm in stock fixed stock-O0 stock-raw; do
    echo "fma instructions in arm $arm: $(objdump -d "$OUT/$arm/run_temf" | grep -c 'vfmadd\|vfmsub\|vfnmadd' || true)"
  done
  sha256sum "$HERE/run_temf.F90" "$HERE/cr_shim.c" "$HERE/apply_fixes.py" \
      "$HERE/stub_wrf.F90" "$OUT/declared-fixes.diff" \
      | sed "s|$HERE/||; s|$OUT/||"
} | tee "$OUT/TOOLCHAIN.txt"
