#!/usr/bin/env bash
# Build the WRF 4.6.1 P3 column oracle from a pristine WRF source tree.
#
#   bash tools/p3_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR> [50|51|52|53]
#
# The optional third argument picks the call shape (default 50).  53 builds
# tools/p3_wrf461_oracle/p3_ref_driver_3mom.f90 instead, which calls the
# same unmodified module with WRF's P3_1CAT_3MOM shape (nc_3d and qzi1_3d
# present, p3_init trplMomI=.true.) against run/p3_lookupTable_1.dat-v5.4_3momI.
#
# module_mp_p3.F is compiled UNMODIFIED (sha256 pinned below; the same bytes
# ship in WRF v4.6.1, v4.7.1 and v4.8.0).  The driver is the existing
# evidence/p3-fortran-oracle-20260828/harness/p3_ref_driver.f90, which calls
# mp_p3_wrapper_wrf with the mp_physics=50 call shape; p3_nc_ref_driver.f90
# here is the same harness with the mp_physics=51 call shape (nc_3d passed,
# so log_predictNc is on).  Both link the SAME module_mp_p3.o per arm.
#
# Arms:
#   A  -O0 -ffp-contract=off -fno-unsafe-math-optimizations
#      statement order as written, against this box's C library (glibc
#      2.39, whose tgammaf and log10f round incorrectly).  Reported.
#   CR arm A with tgammaf/log10f correctly rounded (crlibm_shim.c), which
#      is what WRF computes on a C library that rounds them correctly.
#      THE ORACLE for the 0 ULP claim; byte-identical to arm A built on
#      glibc 2.43 for all thirteen p3-fortref fixtures.
#   B  WRF 4.6.1's own GNU FCOPTIM: -O2 -ftree-vectorize -funroll-loops
#      (what a stock wrf.exe on this box would compute; vectorised loops
#      may call libmvec).  Reported, not the claim.
set -euo pipefail
WRF=${1:?usage: build.sh WRF_SOURCE_ROOT BUILD_DIR}
OUT=${2:?usage: build.sh WRF_SOURCE_ROOT BUILD_DIR}
VARIANT=${3:-50}
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
SRC="$WRF/phys/module_mp_p3.F"
DRV="$TREE/evidence/p3-fortran-oracle-20260828/harness/p3_ref_driver.f90"
DRVNC="$HERE/p3_nc_ref_driver.f90"
# The mp_physics = 52 driver: mp_p3_wrapper_wrf_2cat, nCat = 2, predicted Nc.
DRV2="$HERE/p3_ref_driver_2cat.f90"
TAB="$WRF/run/p3_lookupTable_1.dat-v5.4_2momI"
# Table 2 (ice-ice collection between categories), read by p3_init only when
# nCat > 1 (module_mp_p3.F:474-519).  WOOF packages WRF's own bytes (LF).
TAB2="$WRF/run/p3_lookupTable_2.dat-v5.3"
EXPECT_TAB2=c9103041ead719cf9e2e215cf846e0ef81b2004f29386156444a079af6b24d97

EXPECT_SRC=716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6
case "$VARIANT" in
  50|51|52)
    DRV="$TREE/evidence/p3-fortran-oracle-20260828/harness/p3_ref_driver.f90"
    TAB="$WRF/run/p3_lookupTable_1.dat-v5.4_2momI"
    # WRF's run/ copy is LF; WOOF's packaged copy is the same text with CRLF
    # (woof/data/p3/tables/MANIFEST.sha256 pins 5cb47f2a...).  Both hash to
    # be1ab6fb... once carriage returns are removed.
    EXPECT_TAB=be1ab6fb03481e376e47c6c79d808af5d8ab069f2b242931e9c54801bad4ae84
    woof_text () { tr -d '\r' < "$TREE/woof/data/p3/tables/p3_lookupTable_1.dat-v5.4_2momI"; } ;;
  53)
    DRV="$HERE/p3_ref_driver_3mom.f90"
    TAB="$WRF/run/p3_lookupTable_1.dat-v5.4_3momI"
    # WOOF packages WRF's own LF text, xz-compressed (MANIFEST.sha256 pins
    # the .xz bytes; this pins the text inside it).
    EXPECT_TAB=9a3c57ecc09498802c8d7cb3931dbb0200dcf0f51466b3e288120c080271e6dc
    woof_text () { xz -dc "$TREE/woof/data/p3/tables/p3_lookupTable_1.dat-v5.4_3momI.xz"; } ;;
  *) echo "FATAL: variant must be 50, 51, 52 or 53, got $VARIANT" >&2; exit 2 ;;
esac
if [ "$VARIANT" = 51 ]; then DRV="$DRVNC"; fi
if [ "$VARIANT" = 52 ]; then DRV="$DRV2"; fi

got=$(sha256sum "$SRC" | cut -d' ' -f1)
[ "$got" = "$EXPECT_SRC" ] || { echo "FATAL: $SRC sha256 $got != $EXPECT_SRC" >&2; exit 1; }
got=$(tr -d '\r' < "$TAB" | sha256sum | cut -d' ' -f1)
[ "$got" = "$EXPECT_TAB" ] || { echo "FATAL: $TAB (LF) sha256 $got != $EXPECT_TAB" >&2; exit 1; }
if [ "${P3_ORACLE_SKIP_WOOF_TABLE:-0}" != 1 ]; then
  got=$(woof_text | sha256sum | cut -d' ' -f1)
  [ "$got" = "$EXPECT_TAB" ] || { echo "FATAL: WOOF table text differs from WRF's" >&2; exit 1; }
fi

# Every driver is built, including category two, so validate its table too.
got=$(sha256sum "$TAB2" | cut -d' ' -f1)
[ "$got" = "$EXPECT_TAB2" ] || { echo "FATAL: $TAB2 sha256 $got != $EXPECT_TAB2" >&2; exit 1; }
if [ "${P3_ORACLE_SKIP_WOOF_TABLE:-0}" != 1 ]; then
  got=$(sha256sum "$TREE/woof/data/p3/tables/p3_lookupTable_2.dat-v5.3" | cut -d' ' -f1)
  [ "$got" = "$EXPECT_TAB2" ] || { echo "FATAL: packaged table 2 differs from WRF" >&2; exit 1; }
fi
mkdir -p "$OUT/tables"
cp "$TAB" "$TAB2" "$OUT/tables/"
FREE="-cpp -ffree-form -ffree-line-length-none"
build () {
  local arm=$1; shift
  local d="$OUT/build/$arm"
  rm -rf "$d"; mkdir -p "$d"
  ( cd "$d"
    gfortran $FREE "$@" -c "$SRC" -o module_mp_p3.o
    gfortran $FREE "$@" -c "$DRV" -o p3_ref_driver.o
    gfortran "$@" module_mp_p3.o p3_ref_driver.o -o p3_ref_driver
    gfortran $FREE "$@" -c "$DRVNC" -o p3_nc_ref_driver.o
    gfortran "$@" module_mp_p3.o p3_nc_ref_driver.o -o p3_nc_ref_driver
    gfortran $FREE "$@" -c "$DRV2" -o p3_ref_driver_2cat.o
    gfortran "$@" module_mp_p3.o p3_ref_driver_2cat.o -o p3_ref_driver_2cat )
  echo "built arm $arm: $*"
}
build A -O0 -ffp-contract=off -fno-unsafe-math-optimizations
build B -O2 -ftree-vectorize -funroll-loops
# Arm CR: arm A's flags and objects, with crlibm_shim.o linked ahead of libm
# so tgammaf/log10f are correctly rounded (current-glibc behaviour).
d="$OUT/build/CR"; rm -rf "$d"; mkdir -p "$d"
gcc -O2 -c "$HERE/crlibm_shim.c" -o "$d/crlibm_shim.o"
gfortran "$OUT/build/A/module_mp_p3.o" "$OUT/build/A/p3_ref_driver.o" "$d/crlibm_shim.o" -lquadmath -o "$d/p3_ref_driver"
gfortran "$OUT/build/A/module_mp_p3.o" "$OUT/build/A/p3_nc_ref_driver.o" "$d/crlibm_shim.o" -lquadmath -o "$d/p3_nc_ref_driver"
gfortran "$OUT/build/A/module_mp_p3.o" "$OUT/build/A/p3_ref_driver_2cat.o" "$d/crlibm_shim.o" -lquadmath -o "$d/p3_ref_driver_2cat"
echo "built arm CR: arm A objects + crlibm_shim.o"
{
  gfortran --version | sed -n 1p
  ldd --version | sed -n 1p
  echo "variant $VARIANT"
  echo "module_mp_p3.F $EXPECT_SRC"
  echo "driver $(sha256sum "$DRV" | cut -d' ' -f1)"
  echo "nc driver $(sha256sum "$DRVNC" | cut -d' ' -f1)"
  echo "driver_2cat $(sha256sum "$DRV2" | cut -d' ' -f1)"
  echo "fma instructions in arm A: $(objdump -d "$OUT/build/A/p3_ref_driver" | grep -c 'vfmadd\|vfmsub\|vfnmadd' || true)"
  echo "fma instructions in arm B: $(objdump -d "$OUT/build/B/p3_ref_driver" | grep -c 'vfmadd\|vfmsub\|vfnmadd' || true)"
  echo "libmvec calls in arm B: $(objdump -d "$OUT/build/B/p3_ref_driver" | grep -c 'call.*_ZGV' || true)"
} | tee "$OUT/build/TOOLCHAIN.txt"
