#!/usr/bin/env bash
# Build and run the BouLac column oracle (bl_pbl_physics = 8) against the
# byte-unmodified WRF v4.6.1 phys/module_bl_boulac.F.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_SCALAR_BUILD_ROOT]
#
# Variants, all driven by the same run_bl_boulac.F90 over the same grids:
#
# * wrfobj (THE FIXTURES OF RECORD, when WRF_SCALAR_BUILD_ROOT is given):
#   the driver linked against the scalar WRF build's own
#   phys/module_bl_boulac.o -- the very object inside the wrf.exe that is
#   the 0 ULP referee for full-model runs (gfortran 13.3.0, -O2
#   -fno-fast-math -ffp-contract=off -fno-tree-vectorize).
# * o2: the pinned source compiled here with those same flags.  Must be
#   byte-identical to wrfobj.
# * o0: the pinned source at -O0.  Evidence; the script reports whether it
#   is byte-identical.
# * snan: o2 with every local real initialised to a signalling NaN.  Must be
#   byte-identical: no uninitialised local reaches an output on these cases.
#
# Without WRF_SCALAR_BUILD_ROOT the o2 build is the fixture of record.
#
# module_bl_boulac.F has no USE statement and calls nothing outside itself;
# `nm -u` on the object is the receipt (powf and the allocator only).
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_SCALAR_BUILD_ROOT]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
scalar_root=${3:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src="${source_root}/phys/module_bl_boulac.F"
pinned_sha256="68285f1457fe62e37ad1d76680d3c69e4a48a6baca93348f6f8a15c5fb40d871"

actual=$(sha256sum "${src}" | awk '{print $1}')
if [[ "${actual}" != "${pinned_sha256}" ]]; then
    echo "module_bl_boulac.F is ${actual}, not the pinned ${pinned_sha256}" >&2
    exit 3
fi

wrf_flags="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize"
free="-ffree-form -ffree-line-length-none"
mkdir -p "${build_dir}"
cd "${build_dir}"
python3 "${script_dir}/make_cases.py" "${build_dir}/cases"

build_variant() {
    local name=$1 flags=$2
    mkdir -p "${name}"
    (cd "${name}" &&
        gfortran -c ${flags} -cpp ${free} -o module_bl_boulac.o "${src}" &&
        gfortran -c ${flags} ${free} -I . "${script_dir}/run_bl_boulac.F90" &&
        gfortran -o run_bl_boulac module_bl_boulac.o run_bl_boulac.o &&
        nm -u module_bl_boulac.o | sed 's/^ *//' | sort > undefined.txt)
}

build_variant o2 "${wrf_flags}"
build_variant o0 "-O0"
build_variant snan "${wrf_flags} -finit-real=snan -finit-integer=-77777"

variants="o2 o0 snan"
record=o2
if [[ -n "${scalar_root}" ]]; then
    obj="${scalar_root}/phys/module_bl_boulac.o"
    test -f "${obj}"
    mkdir -p wrfobj
    (cd wrfobj &&
        gfortran -c ${wrf_flags} ${free} -I "${scalar_root}/phys" \
            "${script_dir}/run_bl_boulac.F90" &&
        gfortran -o run_bl_boulac "${obj}" run_bl_boulac.o &&
        sha256sum "${obj}" > wrf-object-sha256.txt &&
        nm -u "${obj}" | sed 's/^ *//' | sort > undefined.txt)
    variants="wrfobj ${variants}"
    record=wrfobj
fi

if grep -Ev '^U (powf|malloc|free|memset|memcpy|_gfortran_.*)$' \
        "${record}/undefined.txt"; then
    echo "the reference object calls something beyond powf and the allocator" >&2
    exit 4
fi

grids=$(python3 -c "import json,sys; print(' '.join(g['grid'] for g in json.load(open('cases/cases-index.json'))))")
for v in ${variants}; do
    for gname in ${grids}; do
        "./${v}/run_bl_boulac" "cases/${gname}.in.bin" "${v}/${gname}.rec.bin"
    done
done

status=0
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# fixture of record: ${record}"
    for v in ${variants}; do
        [[ "${v}" == "${record}" ]] && continue
        same=yes
        for gname in ${grids}; do
            cmp -s "${record}/${gname}.rec.bin" "${v}/${gname}.rec.bin" || same=no
        done
        echo "${v} byte-identical to ${record}: ${same}"
    done
} > variants-report.txt
cat variants-report.txt
for v in o2 snan; do
    [[ "${v}" == "${record}" ]] && continue
    if grep -q "^${v} byte-identical to ${record}: no" variants-report.txt; then
        echo "variant ${v} must be byte-identical to ${record}" >&2
        status=5
    fi
done

mkdir -p fixtures
cp cases/cases-index.json fixtures/
for gname in ${grids}; do
    cp "${record}/${gname}.rec.bin" "fixtures/${gname}.rec.bin"
    cp "cases/${gname}.in.bin" "fixtures/${gname}.in.bin"
done
sha256sum "${src}" "${script_dir}/run_bl_boulac.F90" \
    "${script_dir}/make_cases.py" fixtures/* > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built in ${build_dir} (record: ${record})"
exit ${status}
