#!/usr/bin/env bash
# Build and run the Grell-Devenyi (cu_physics = 93) column oracle against
# WRF v4.6.1's own phys/module_cu_gd.F.
#
#   bash tools/gd_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_TREE ...]
#
# module_cu_gd.F has no USE statement and no #include: GRELLDRV and every
# routine under it are self-contained module procedures, so the module
# compiles alone and `nm -u` on the object is the receipt (libm's expf,
# logf and atan2f plus the gfortran runtime, nothing else).
#
# Arms, every one linked with the same driver program run_gd.F90:
#
#   stock-strict   pristine source through WRF's own preprocessing pipeline
#                  (sed, cpp with WRF's defines, tools/standard.exe, cpp),
#                  compiled with the strict build's flags
#                  (-O2 -fno-fast-math -ffp-contract=off);
#   stock-scalar   the same source with the scalar build's flags (adds
#                  -fno-tree-vectorize): the 0 ULP referee of this sprint;
#   fixed-strict / fixed-scalar
#                  the source with patches/gd-closure-count.patch applied --
#                  an optional closure-count experiment. The forecast adapter
#                  always uses the unmodified stock arm;
#   tree-<name>    for each WRF_BUILD_TREE given: that tree's own
#                  phys/module_cu_gd.o, the object its wrf.exe links.  The
#                  harness refuses a tree whose module_cu_gd.F is not the
#                  pinned source.
#
# validate_arms.py then requires stock-strict == stock-scalar == every tree
# arm and fixed-strict == fixed-scalar, word for word.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_TREE ...]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
shift 2
trees=("$@")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# WRF v4.6.1 (git d66e442fccc04111067e29274c9f9eaccc3cef28)
pinned=14948e3b6e82ba2d1529c2b31ff89c2e40e1596018d01898123bc2baa8eb81df
src="${source_root}/phys/module_cu_gd.F"
actual=$(sha256sum "${src}" | awk '{print $1}')
if [[ "${actual}" != "${pinned}" ]]; then
    echo "phys/module_cu_gd.F is ${actual}, not the pinned ${pinned}" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

# WRF's tools/standard.exe, built from the pinned tree's own source.
gcc -O2 -o standard.exe "${source_root}/tools/standard.c"

# The preprocessing pipeline exactly as WRF's phys/Makefile runs it
# (compile.log of the strict build tree, configure option 32).
cpp_defs="-DEM_CORE=1 -DNMM_CORE=0 -DNMM_MAX_DIM=2600 -DDA_CORE=0 -DWRFPLUS=0
 -DIWORDSIZE=4 -DDWORDSIZE=8 -DRWORDSIZE=4 -DLWORDSIZE=4
 -DNONSTANDARD_SYSTEM_SUBR -DWRF_USE_CLM -DUSE_NETCDF4_FEATURES
 -DWRFIO_NCD_LARGE_FILE_SUPPORT -DNETCDF -DLANDREAD_STUB=1 -DUSE_ALLOCATABLES
 -Dwrfmodel -DGRIB1 -DINTIO -DKEEP_INT_AROUND -DLIMIT_ARGS
 -DBUILD_RRTMG_FAST=0 -DBUILD_RRTMK=0 -DBUILD_SBM_FAST=1
 -DSHOW_ALL_VARS_USED=0 -DCONFIG_BUF_LEN=65536 -DMAX_DOMAINS_F=21
 -DMAX_HISTORY=25 -DNMM_NEST=0"
preprocess() {   # in.F out.f90
    sed -e "s/^\!.*'.*//" -e "s/^ *\!.*'.*//" "$1" > "$2.G"
    # shellcheck disable=SC2086
    /lib/cpp -P -nostdinc -I"${source_root}/inc" ${cpp_defs} -I. \
        -traditional-cpp -DUSE_NETCDF4_FEATURES \
        -DWRFIO_NCD_LARGE_FILE_SUPPORT "$2.G" > "$2.bb"
    ./standard.exe "$2.bb" | /lib/cpp -P -nostdinc -traditional-cpp \
        -DUSE_NETCDF4_FEATURES -DWRFIO_NCD_LARGE_FILE_SUPPORT > "$2"
    rm -f "$2.G" "$2.bb"
}

mkdir -p stock fixed
cp "${src}" stock/module_cu_gd.F
cp "${src}" fixed/module_cu_gd.F
(cd fixed && patch -s -p2 < "${script_dir}/patches/gd-closure-count.patch")
preprocess stock/module_cu_gd.F stock/module_cu_gd.f90
preprocess fixed/module_cu_gd.F fixed/module_cu_gd.f90

base="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz"
strict="-O2 -fno-fast-math -ffp-contract=off"
scalar="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize"

link_arm() {   # arm-name object moddir
    gfortran -c -O0 -ffree-form -ffree-line-length-none -I "$3" -J "$3" \
        -o "$3/run_gd.o" "${script_dir}/run_gd.F90"
    gfortran -o "run_gd_$1" "$3/run_gd.o" "$2"
}

: > undefined.txt
for variant in stock fixed; do
    for flags in strict scalar; do
        d="${variant}/${flags}"
        mkdir -p "${d}"
        # shellcheck disable=SC2086
        gfortran -c ${!flags} ${base} -J "${d}" -o "${d}/module_cu_gd.o" \
            "${variant}/module_cu_gd.f90"
        nm -u "${d}/module_cu_gd.o" | sed 's/^ *//' | sed "s|^|${variant}-${flags} |" >> undefined.txt
        link_arm "${variant}-${flags}" "${d}/module_cu_gd.o" "${d}"
    done
done
for tree in "${trees[@]}"; do
    name=$(basename "$(dirname "${tree}")")       # .../build/<name>/WRF-4.6.1
    tsum=$(sha256sum "${tree}/phys/module_cu_gd.F" | awk '{print $1}')
    if [[ "${tsum}" != "${pinned}" ]]; then
        echo "${tree}/phys/module_cu_gd.F is not the pinned source" >&2
        exit 3
    fi
    d="tree/${name}"
    mkdir -p "${d}"
    cp "${tree}/phys/module_cu_gd.o" "${tree}/phys/module_cu_gd.mod" "${d}/"
    nm -u "${d}/module_cu_gd.o" | sed 's/^ *//' | sed "s|^|tree-${name} |" >> undefined.txt
    link_arm "tree-${name}" "${d}/module_cu_gd.o" "${d}"
done
if grep -v -E ' (expf|logf|atan2f|malloc|free|memcpy|memset|_gfortran_[a-z_]+)$' undefined.txt; then
    echo "module_cu_gd.o imports something other than libm and the gfortran runtime" >&2
    exit 4
fi
if grep -q '_ZGV' undefined.txt; then
    echo "a GD object pulled in a libmvec vector symbol" >&2
    exit 4
fi

# --- cases and runs ---------------------------------------------------------
mkdir -p cases out
python3 "${script_dir}/gd_cases.py" cases > cases/manifest.txt
for input in cases/*.in; do
    stem=$(basename "${input}" .in)
    for exe in run_gd_*; do
        arm=${exe#run_gd_}
        "./${exe}" "${input}" "out/${stem}.${arm}.out"
    done
done
sha256sum out/*.out > out/SHA256SUMS
gfortran --version | head -1 > toolchain.txt
ldd --version | head -1 >> toolchain.txt
echo "build.sh done: $(ls out/*.out | wc -l) outputs"
