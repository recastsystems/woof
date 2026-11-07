#!/usr/bin/env bash
# Build and run the TEMF surface-layer oracle (sf_sfclay_physics=10) against
# the byte-unmodified WRF v4.6.1 phys/module_sf_temfsfclay.F.
#
# usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]
#
# The module has no USE statement and calls one external, wrf_debug (in
# TEMFSFCLAYINIT), which stub_wrf.F90 supplies.  `nm -u` on each object is
# the receipt.
#
# Three arms, one driver (run_temf_sfclay.F90), identical inputs:
#   A  WRF's own arithmetic.  With WRF_BUILD_ROOT, the object the WRF build
#      compiled (phys/module_sf_temfsfclay.o of a full gfortran build: the
#      scalar referee is -O2 -fno-fast-math -ffp-contract=off
#      -fno-tree-vectorize); without it, the pinned source compiled with
#      exactly those flags.
#   B  The pinned source at -O0.  Must equal arm A word for word: the
#      optimiser changes no float32 word of this scheme.
#   C  Arm A's object with the C library's log10f replaced by a correctly
#      rounded log10f (log10f_correctly_rounded.c, linked ahead of libm).
#      WOOF's float32 log10 is correctly rounded; the C library's is not
#      (log10-probe.txt measures how often it differs), so arm C is the arm
#      WOOF's strict mode is graded at 0 ULP against, and A-vs-C names
#      exactly the columns where the two log10 words differ.
#
# pow/exp/log semantics: the scheme's real-exponent powers are **(1./3.)
# and **1.5 (powf calls); **2. folds to a multiply.  The objects' undefined
# symbols are listed in undefined-*.txt.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
wrf_build_root=${3:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
module_source="${source_root}/phys/module_sf_temfsfclay.F"
pinned_sha256="70f2ed6bc48aad5b1bf68be4e01753a132f2ecdb4fa6d95c0ecce5d5554fcd94"
wrf_scalar_flags=(-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize)
free_form=(-w -ffree-form -ffree-line-length-none)

actual_sha256=$(sha256sum "${module_source}" | awk '{print $1}')
if [[ "${actual_sha256}" != "${pinned_sha256}" ]]; then
    echo "module_sf_temfsfclay.F is ${actual_sha256}, not the pinned" \
         "${pinned_sha256}" >&2
    exit 3
fi

mkdir -p "${build_dir}"/{a,b,c}
cd "${build_dir}"

gfortran -c -O0 "${free_form[@]}" -o stub_wrf.o "${script_dir}/stub_wrf.F90"
gcc -c -O2 -o log10f_correctly_rounded.o "${script_dir}/log10f_correctly_rounded.c"
gcc -O0 -o log10_probe "${script_dir}/log10_probe.c" -lm

# --- arm A: WRF's object, or the source at WRF's scalar flags --------------
if [[ -n "${wrf_build_root}" ]]; then
    wrf_object="${wrf_build_root}/phys/module_sf_temfsfclay.o"
    built_from="${wrf_build_root}/phys/module_sf_temfsfclay.F"
    test -f "${wrf_object}"
    built_sha=$(sha256sum "${built_from}" | awk '{print $1}')
    if [[ "${built_sha}" != "${pinned_sha256}" ]]; then
        echo "the WRF build compiled ${built_sha}, not the pinned source" >&2
        exit 3
    fi
    cp "${wrf_object}" a/module_sf_temfsfclay.o
    cp "${wrf_build_root}/phys/module_sf_temfsfclay.mod" a/
    echo "arm A object: ${wrf_object}" > arm-a-origin.txt
else
    cpp -P -traditional -nostdinc "${module_source}" > a/module_sf_temfsfclay.f90
    (cd a && gfortran -c "${wrf_scalar_flags[@]}" "${free_form[@]}" \
        module_sf_temfsfclay.f90)
    echo "arm A object: pinned source at ${wrf_scalar_flags[*]}" > arm-a-origin.txt
fi
(cd a && gfortran -c -O0 "${free_form[@]}" -I . "${script_dir}/run_temf_sfclay.F90" \
    && gfortran -o run_temf_sfclay run_temf_sfclay.o module_sf_temfsfclay.o \
        ../stub_wrf.o -lm)

# --- arm B: the pinned source at -O0 ----------------------------------------
(cd b && gfortran -c -O0 -cpp "${free_form[@]}" -o module_sf_temfsfclay.o \
        "${module_source}" \
    && gfortran -c -O0 "${free_form[@]}" -I . "${script_dir}/run_temf_sfclay.F90" \
    && gfortran -o run_temf_sfclay run_temf_sfclay.o module_sf_temfsfclay.o \
        ../stub_wrf.o -lm)

# --- arm C: arm A's object, correctly rounded log10f ------------------------
(cd c && gfortran -o run_temf_sfclay ../a/run_temf_sfclay.o \
        ../a/module_sf_temfsfclay.o ../stub_wrf.o \
        ../log10f_correctly_rounded.o -lm)
if ! nm c/run_temf_sfclay | grep -q ' T log10f$'; then
    echo "arm C did not interpose log10f" >&2
    exit 4
fi

for arm in a b; do
    nm -u "${arm}/module_sf_temfsfclay.o" | sed 's/^ *//' | sort \
        > "undefined-${arm}.txt"
done
if grep -q '_ZGV' undefined-a.txt undefined-b.txt; then
    echo "a vector libm symbol reached the oracle objects" >&2
    exit 5
fi

# --- run ----------------------------------------------------------------------
for arm in a b c; do
    "./${arm}/run_temf_sfclay" "temf-inputs-${arm}.csv" "temf-outputs-${arm}.csv" \
        > "run-${arm}.log"
done
# Arms A and C share every external input; the carried HFX and WM_TEMF
# columns of the input record follow each arm's own previous outputs.
cmp temf-inputs-a.csv temf-inputs-b.csv
if ! cmp -s temf-outputs-a.csv temf-outputs-b.csv; then
    echo "arm A (WRF's -O2 object) and arm B (-O0) disagree" >&2
    diff temf-outputs-a.csv temf-outputs-b.csv | head >&2
    exit 6
fi
cp temf-inputs-a.csv temf-inputs-wrf.csv
cp temf-outputs-a.csv temf-outputs-wrf.csv
cp temf-inputs-c.csv temf-inputs-wrf-crlog10.csv
cp temf-outputs-c.csv temf-outputs-wrf-crlog10.csv

# --- the C library's log10f, measured against the correctly rounded value ----
{
    echo "# $(ldd --version | head -1)"
    for range in "1 10" "10 1000" "1000 1e5" "1e5 1e7"; do
        ./log10_probe ${range} | tail -1
    done
} > log10-probe.txt

# --- WOOF's own log10 (log10_flt32.cuh, compiled as C) against MPFR -------
# Every positive finite binary32 word.  MPFR is the shared library gfortran
# itself links (no headers needed).  TEMF_LOG10_THREADS sets the width.
kernels_dir="${script_dir}/../../woof/core/kernels"
gcc -O2 -ffp-contract=off -fopenmp -I "${kernels_dir}" -o log10_exhaustive     "${script_dir}/log10_exhaustive.c" -l:libmpfr.so.6 -lm
./log10_exhaustive "${TEMF_LOG10_THREADS:-32}" > log10-exhaustive.txt
cat log10-exhaustive.txt

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    cat arm-a-origin.txt
    echo "# arm A undefined symbols:"; cat undefined-a.txt
} > compiler.txt

sha256sum "${module_source}" "${script_dir}"/run_temf_sfclay.F90 \
    "${script_dir}"/stub_wrf.F90 "${script_dir}"/log10f_correctly_rounded.c \
    temf-inputs-wrf.csv temf-outputs-wrf.csv     temf-inputs-wrf-crlog10.csv temf-outputs-wrf-crlog10.csv \
    log10-probe.txt log10-exhaustive.txt     "${kernels_dir}/log10_flt32.cuh" > oracle-sha256sums.txt
echo "oracle built in ${build_dir}"
