#!/usr/bin/env bash
# Build and run the QNSE PBL oracle (bl_pbl_physics = 4, with and without
# the EDMF shallow mass flux mfshconv = 1) against the byte-unmodified
# WRF v4.6.1 phys/module_bl_qnsepbl.F and phys/module_bl_mfshconvpbl.F.
#
#     bash build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# The three sources are pinned by sha256 (the files of the v4.6.1 tag,
# d66e442fccc04111067e29274c9f9eaccc3cef28).  Nothing is stubbed: both
# modules USE only module_model_constants, compiled here from the same tree.
#
# The reference build uses WRF's own flags for this sprint's builds
# (/work/wrf/build-all.sh: -O2 -fno-fast-math -ffp-contract=off, scalar arm
# -fno-tree-vectorize).  It is NOT built at -O0, on purpose: at -O0
# gfortran 13 lowers a REAL exponent of 2.0 (ZKIC**2., (1.-ZKIC)**2.,
# module_bl_mfshconvpbl.F:985-986, :1001-1002) to a powf(x, 2.0) call,
# which differs from x*x in the last bit for about 0.08% of arguments,
# while at -O2 the same source folds to x*x -- which is what wrf.exe runs.
# pow-fold-probe.txt is the measurement; the -O0 build is kept beside the
# reference and its word differences are recorded (o0-vs-ref.txt).
#
# -finit-logical=false: COMPUTE_ENTR_DETR reads GTEST_LOCAL_LCL2 at
# module_bl_mfshconvpbl.F:997 on paths that never assigned it (:730-731,
# :922-936), so stock WRF reads stack garbage there.  The flag gives it the
# value every other logical of the routine starts at.  init_probe proves
# the flag reaches an automatic logical array on this toolchain, and the
# -finit-logical=true arm is run as evidence of how much the garbage moves.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
src=$(realpath "$1")
out=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

declare -A pin=(
  [phys/module_bl_qnsepbl.F]=737a235b9febb01004d5bfe7a9364a0fc57e8af79cbced8239d68ab623e80e66
  [phys/module_bl_mfshconvpbl.F]=cb678bb818adda9906ab6479cf04e665285042e5bfe9f39c7ddef2e9221c7a34
  [share/module_model_constants.F]=5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062
)
for f in "${!pin[@]}"; do
    got=$(sha256sum "$src/$f" | awk '{print $1}')
    if [[ "$got" != "${pin[$f]}" ]]; then
        echo "$f is $got, not the pinned ${pin[$f]}" >&2
        exit 3
    fi
done

mkdir -p "$out"
cd "$out"
FF="-cpp -DEM_CORE=1 -ffree-form -ffree-line-length-none"

build() {  # dir flags
    local d=$1; shift
    mkdir -p "$d"
    (cd "$d"
     gfortran -c $FF "$@" "$src/share/module_model_constants.F" -o module_model_constants.o
     gfortran -c $FF "$@" "$src/phys/module_bl_qnsepbl.F" -o module_bl_qnsepbl.o
     gfortran -c $FF "$@" "$src/phys/module_bl_mfshconvpbl.F" -o module_bl_mfshconvpbl.o
     gfortran -c $FF "$@" "$here/run_qnse.F90" -o run_qnse.o
     gfortran -o run_qnse run_qnse.o module_bl_qnsepbl.o module_bl_mfshconvpbl.o \
         module_model_constants.o
     gfortran -c $FF "$@" "$here/const_dump.F90" -o const_dump.o
     gfortran -o const_dump const_dump.o module_bl_qnsepbl.o \
         module_bl_mfshconvpbl.o module_model_constants.o
     nm -u module_bl_qnsepbl.o module_bl_mfshconvpbl.o | sed 's/^ *//' | sort -u \
         > undefined.txt)
}

WRFFLAGS="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize"
build ref $WRFFLAGS -finit-logical=false
build o0 -O0 -finit-logical=false
build garbage_true $WRFFLAGS -finit-logical=true

if grep -q '_ZGV' ref/undefined.txt; then
    echo "reference object pulled in a vector libm symbol" >&2
    exit 4
fi

# --- does -finit-logical reach an automatic logical array? -----------------
gfortran -O0 -finit-logical=true -o probe_t "$here/init_probe.F90"
gfortran -O0 -finit-logical=false -o probe_f "$here/init_probe.F90"
pt=$(./probe_t); pf=$(./probe_f)
echo "init_probe -finit-logical=true: $pt; =false: $pf" | tee init-probe.txt
if [[ "$pt" != "true_count=64" || "$pf" != "true_count=0" ]]; then
    echo "-finit-logical does not reach automatic logical arrays here" >&2
    exit 5
fi

"${PYTHON:-python3}" "$here/make_cases.py" "$out/cases"
for d in ref o0 garbage_true; do
    mkdir -p "$out/$d/run"
    for arm in mf0 mf1; do
        "$out/$d/run_qnse" "$out/cases/cases-$arm.bin" "$out/$d/run/$arm"
    done
done
./ref/const_dump > consts.txt

# x**2. lowering on this toolchain, -O0 against WRF's flags.
gfortran -O0 -o pow_fold_o0 "$here/pow_fold_probe.F90"
gfortran $WRFFLAGS -o pow_fold_o2 "$here/pow_fold_probe.F90"
{ echo "-O0:       $(./pow_fold_o0)"; echo "WRF flags: $(./pow_fold_o2)"; }     | tee pow-fold-probe.txt
"${PYTHON:-python3}" "$here/pack_fixture.py" --diff ref/run o0/run > o0-vs-ref.txt || true
echo "-O0 build against the reference (WRF flags):"; cat o0-vs-ref.txt
for f in ref/run/*.bin; do
    cmp -s "$f" "garbage_true/run/$(basename "$f")" \
        && echo "same    $(basename "$f")" || echo "differs $(basename "$f")"
done > garbage-sensitivity.txt
cat garbage-sensitivity.txt

{
  echo "# gfortran: $(gfortran --version | head -1)"
  echo "# glibc:    $(ldd --version | head -1)"
  sha256sum "$src/phys/module_bl_qnsepbl.F" "$src/phys/module_bl_mfshconvpbl.F" \
      "$src/share/module_model_constants.F" "$here"/*.F90 "$here"/*.py \
      "$here/build.sh" cases/*.bin ref/run/*.bin consts.txt \
      | sed "s|$src|<wrf-4.6.1-checkout>|; s|$here|<oracle-dir>|; s|$out/||"
} > oracle-sha256sums.txt
echo "oracle built in $out"
