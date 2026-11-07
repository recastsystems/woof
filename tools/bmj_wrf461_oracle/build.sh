#!/usr/bin/env bash
# Build and run the Betts-Miller-Janjic (cu_physics = 2) column oracle against
# the byte-unmodified WRF v4.6.1 phys/module_cu_bmj.F.
#
# THE FILE SET IS TWO FILES.  module_cu_bmj.F's only USE is
# MODULE_MODEL_CONSTANTS (share/module_model_constants.F), which itself uses
# nothing; both are compiled from the pinned tree and digest-checked below.
# `nm -u` on the scheme object is the receipt: libm and libgfortran only.
#
# TWO REFERENCE EXECUTABLES, ONE SOURCE.
#   run_bmj          the stock oracle: GAMMA() is the C library's tgammaf.
#   run_bmj_crgamma  the same objects linked with cr_tgammaf.c, which
#                    interposes a correctly rounded tgammaf.  WOOF's gamma is
#                    correctly rounded by owner ruling, so this is the arm the
#                    port is graded against at 0 ULP on every field; the stock
#                    arm measures the declared gamma divergence, which can only
#                    reach the bmj_rad_feedback outputs (module_cu_bmj.F:308).
#
# -DPRIVATE=PUBLIC.  The moist-adiabat tables TTBL/TTBLQ/STHE/THE0/... are
# PRIVATE module variables (module_cu_bmj.F:52-57).  cpp substitutes the
# token at compile time so the harness can dump them; the file's bytes are
# not edited and no arithmetic statement contains the token.
#
# -O0, no vectoriser, as every WRF column oracle in this tree.  A second
# build at the scalar WRF build's own flags (configure.wrf FCOPTIM:
# -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize) is run as
# evidence and its outputs must equal the -O0 outputs byte for byte.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}

declare -A pinned=(
  [LICENSE.txt]="b30611d35ad1922bc741e86d21e1a897234c1a9c9e2af28076e021f565dfbec6"
  [phys/module_cu_bmj.F]="62d2805aae152b58093d07100000b581458b536164d90f9cecc27a157e2bd92f"
  [share/module_model_constants.F]="5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
)
for rel in "${!pinned[@]}"; do
    actual=$(sha256sum "${source_root}/${rel}" | awk '{print $1}')
    if [[ "${actual}" != "${pinned[$rel]}" ]]; then
        echo "${rel} is ${actual}, not the pinned ${pinned[$rel]}" >&2
        exit 3
    fi
done

mkdir -p "${build_dir}"
cd "${build_dir}"

fflags=(-cpp -DEM_CORE=1 -DPRIVATE=PUBLIC -ffree-form -ffree-line-length-none -w)
build_variant() {   # $1 = tag, rest = optimisation flags
    local tag=$1; shift
    mkdir -p "${tag}"
    ( cd "${tag}"
      gfortran -c "$@" "${fflags[@]}" -o module_model_constants.o \
          "${source_root}/share/module_model_constants.F"
      gfortran -c "$@" "${fflags[@]}" -o module_cu_bmj.o \
          "${source_root}/phys/module_cu_bmj.F"
      gfortran -c "$@" -ffree-line-length-none -o run_bmj.o "${script_dir}/run_bmj.F90"
      gcc -c -O2 -o cr_tgammaf.o "${script_dir}/cr_tgammaf.c"
      gfortran -o run_bmj run_bmj.o module_cu_bmj.o module_model_constants.o
      gfortran -o run_bmj_crgamma run_bmj.o module_cu_bmj.o \
          module_model_constants.o cr_tgammaf.o -lm
      nm -u module_cu_bmj.o | sed 's/^ *//' | sort > undefined.txt )
}
build_variant O0 -O0
build_variant wrfscalar -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize

if grep -q '_ZGV' O0/undefined.txt; then
    echo "-O0 object pulled in a libmvec vector symbol" >&2
    exit 4
fi
grep -q 'tgammaf' O0/undefined.txt || { echo "module_cu_bmj.o calls no tgammaf; the pinned-gamma arm is vacuous" >&2; exit 5; }
if nm O0/run_bmj_crgamma | grep -q ' U tgammaf'; then
    echo "run_bmj_crgamma still imports tgammaf; interposition failed" >&2
    exit 6
fi

"${python}" "${script_dir}/gen_cases.py" cases.bin
for tag in O0 wrfscalar; do
    ( cd "${tag}"
      ./run_bmj ../cases.bin stock tables.bin
      ./run_bmj_crgamma ../cases.bin crgamma tables-crgamma.bin )
done

# -O0 is the reference; the scalar-WRF-flag build must agree byte for byte.
for f in stock-arm1.bin stock-arm2.bin crgamma-arm1.bin crgamma-arm2.bin tables.bin; do
    if ! cmp -s "O0/${f}" "wrfscalar/${f}"; then
        echo "WRF scalar-flag build differs from -O0 on ${f}" >&2
        exit 7
    fi
done
cp O0/stock-arm1.bin O0/stock-arm2.bin O0/crgamma-arm1.bin \
   O0/crgamma-arm2.bin O0/tables.bin .
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# libm symbols of module_cu_bmj.o at -O0 (THE REFERENCE):"
    grep -E 'expf|powf|logf|sqrtf|tgammaf|acosf|_ZGV' O0/undefined.txt || true
    echo "# -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize (scalar WRF build):"
    grep -E 'expf|powf|logf|sqrtf|tgammaf|acosf|_ZGV' wrfscalar/undefined.txt || true
    echo "# scalar-WRF-flag outputs equal the -O0 outputs byte for byte: yes"
} > libm-report.txt
cat libm-report.txt
sha256sum "${source_root}/phys/module_cu_bmj.F" \
    "${source_root}/share/module_model_constants.F" \
    "${script_dir}/run_bmj.F90" "${script_dir}/gen_cases.py" \
    "${script_dir}/cr_tgammaf.c" cases.bin \
    stock-arm1.bin stock-arm2.bin crgamma-arm1.bin crgamma-arm2.bin \
    tables.bin > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built in ${build_dir}"
