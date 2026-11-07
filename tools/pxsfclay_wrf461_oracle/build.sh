#!/usr/bin/env bash
# Build the WRF v4.6.1 Pleim-Xiu surface-layer column oracle (sf_sfclay_physics=7).
#
# The referee is WRF's own text compiled with WRF's own scalar flags:
#   * phys/module_sf_pxsfclay.F, unmodified, run through cpp as WRF's build
#     does (it carries no #if, so cpp changes nothing but the extension);
#   * share/module_model_constants.F, unmodified, for cp, g, rcp, r_d, xlv,
#     svp1-3, svpt0, ep_1, ep_2 and karman: the words WRF's surface driver
#     hands the scheme;
#   * pxsfclay_seaice_wrapper and get_local_ice_tsk copied byte for byte out
#     of phys/module_surface_driver.F by extract.py.
#
# Flags: the sprint's scalar WRF build (/work/wrf/provenance/flags-scalar.txt)
#   -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize
# A second -O0 build of the same text is run as a control; the two must be
# word-identical or the oracle refuses to report (the PX text has no loop a
# vectoriser could route through libmvec, and this proves it on the host).
#
# usage: build.sh WRF_SOURCE_ROOT BUILD_DIR
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
files=(phys/module_sf_pxsfclay.F phys/module_surface_driver.F
       share/module_model_constants.F)

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${files[@]}"; then
    echo "a PX oracle source differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"
python3 "${script_dir}/extract.py" "${source_root}" "${build_dir}" > /dev/null

cpp -P -traditional -DEM_CORE=1 \
    "${source_root}/share/module_model_constants.F" > module_model_constants.f90
cpp -P -traditional -DEM_CORE=1 \
    "${source_root}/phys/module_sf_pxsfclay.F" > module_sf_pxsfclay.f90

base="-w -ffree-form -ffree-line-length-none"
scalar="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize"
for variant in scalar O0; do
    mkdir -p "${variant}"
    if [[ "${variant}" == scalar ]]; then opts="${scalar}"; else opts="-O0"; fi
    (cd "${variant}"
     gfortran -c ${opts} ${base} ../module_model_constants.f90
     gfortran -c ${opts} ${base} ../module_sf_pxsfclay.f90
     gfortran -c ${opts} ${base} ../px_seaice_oracle.f90
     gfortran -c ${opts} ${base} "${script_dir}/run_pxsfclay.F90"
     gfortran -o run_pxsfclay module_model_constants.o module_sf_pxsfclay.o \
         px_seaice_oracle.o run_pxsfclay.o
     nm -u module_sf_pxsfclay.o px_seaice_oracle.o | sed 's/^ *//' \
         | sort -u > undefined.txt
     if grep -q '_ZGV' undefined.txt; then
         echo "${variant}: a libmvec vector symbol was linked" >&2
         exit 4
     fi)
done

{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "scalar flags: ${scalar} ${base}"
    echo "libm symbols (scalar):"
    grep -E 'expf|logf|powf|atanf|sqrtf' scalar/undefined.txt || true
} > toolchain.txt
sha256sum "${source_root}/phys/module_sf_pxsfclay.F" \
    "${source_root}/phys/module_surface_driver.F" \
    "${source_root}/share/module_model_constants.F" \
    module_sf_pxsfclay.f90 px_seaice_oracle.f90 \
    "${script_dir}/run_pxsfclay.F90" > sources.sha256
echo "oracle built in ${build_dir}"
