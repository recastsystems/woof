#!/usr/bin/env bash
# Build and run the slab (sf_surface_physics=1) column oracle against the
# byte-unmodified WRF v4.6.1 phys/module_sf_slab.F.
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# module_sf_slab.F has no USE statement and calls nothing outside itself, so
# it compiles as it ships.  The physical constants are NOT retyped here: the
# driver USEs WRF's own share/module_model_constants.F (compiled with
# -DEM_CORE=1, as WRF builds it) and writes the values it used into the
# output header, so the port is graded against WRF's constants, bit for bit.
#
# The byte pin is the FILE: module_sf_slab.F must hash to the v4.6.1 value
# recorded below (tree d66e442fccc0), or the build stops.
#
# -O0, no vectoriser: every REAL operation is one rounded SSE scalar op, the
# EXPs and REAL**REAL powers are scalar glibc expf/powf calls (`nm -u` below is
# the receipt), and the one integer power, TG0**4, is libgcc's __powisf2,
# which pow_probe.F90 measures as (x*x)*(x*x).  Any _ZGV (libmvec) symbol in
# the reference object fails the build.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
slab_source="${source_root}/phys/module_sf_slab.F"
constants_source="${source_root}/share/module_model_constants.F"
pinned_sha256="c44f26777b163234da249373b09a02fa92cb953209bd35a6fe3c30c2158200cb"

actual=$(sha256sum "${slab_source}" | awk '{print $1}')
if [[ "${actual}" != "${pinned_sha256}" ]]; then
    echo "module_sf_slab.F is ${actual}, not the pinned ${pinned_sha256}" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"
strict_flags=(-O0 -fno-fast-math -ffp-contract=off -fno-tree-vectorize)
gfortran -c "${strict_flags[@]}" -cpp -DEM_CORE=1 -ffree-form -ffree-line-length-none \
    -o module_model_constants.o "${constants_source}"
gfortran -c "${strict_flags[@]}" -cpp -ffree-form -ffree-line-length-none \
    -o module_sf_slab.o "${slab_source}"
if nm module_sf_slab.o | grep -q '_ZGV'; then
    echo "libmvec symbol in the -O0 reference object" >&2
    exit 4
fi
nm -u module_sf_slab.o > nm-u-module_sf_slab.txt
gfortran -c "${strict_flags[@]}" -ffree-form -ffree-line-length-none \
    "${here}/run_slab.F90"
gfortran -o run_slab run_slab.o module_sf_slab.o module_model_constants.o
gfortran -O0 -o pow_probe "${here}/pow_probe.F90"
./pow_probe > pow-probe.txt
grep -q '^match (x\*x)\*(x\*x) 4001$' pow-probe.txt || {
    echo "TG0**4 is no longer (x*x)*(x*x) on this toolchain" >&2; exit 5; }

python3 "${here}/generate_cases.py" "${build_dir}"
./run_slab slab-input.bin slab-output.bin
{
    echo "module_sf_slab.F sha256 ${pinned_sha256}"
    echo "module_model_constants.F sha256 $(sha256sum "${constants_source}" | awk '{print $1}')"
    gfortran --version | sed -n 1p
    ldd --version | sed -n 1p
    echo "slab-input.bin sha256 $(sha256sum slab-input.bin | awk '{print $1}')"
    echo "slab-output.bin sha256 $(sha256sum slab-output.bin | awk '{print $1}')"
} > provenance.txt
cat provenance.txt
