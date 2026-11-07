#!/usr/bin/env bash
# Build and run the WRF v4.6.1 Eta Ferrier (mp_physics=95, ETAMPNEW) column
# oracle from byte-unmodified WRF source.  CPU only.
#
# usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_SCALAR_BUILD_ROOT]
#
# Writes into BUILD_DIR:
#   columns_inputs.csv, columns.in     (make_columns.py)
#   O0/{levels,surface,init}.csv        the reference (-O0, scalar libm)
#   wrfflags/...                        the same source at the WRF scalar
#                                       referee's flags (-O2 -fno-fast-math
#                                       -ffp-contract=off -fno-tree-vectorize)
#   wrfobj/...                          (optional) WRF's own compiled
#                                       module_mp_etanew.o from a full build
#   cov/...                             -O0 --coverage twin: gcov line
#                                       coverage of the module
#   SHA256SUMS
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_SCALAR_BUILD_ROOT]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
wrf_build=${3:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
mp_source="${source_root}/phys/module_mp_etanew.F"
table="${source_root}/run/ETAMPNEW_DATA.expanded_rain"
test -f "${mp_source}"
test -f "${table}"

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- \
        phys/module_mp_etanew.F run/ETAMPNEW_DATA.expanded_rain; then
    echo "module_mp_etanew.F or its table differs from pinned HEAD" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"
python3 "${script_dir}/make_columns.py" "${build_dir}"

common=(-cpp -ffree-form -ffree-line-length-none -DRWORDSIZE=4
        -DIWORDSIZE=4 -fallow-argument-mismatch)

run_one() {
    local tag=$1
    (
        cd "${tag}"
        ln -sf "${table}" ETAMPNEW_DATA.expanded_rain
        ln -sf ../columns.in columns.in
        ./run_etanew > run.stdout 2> run.stderr
    )
}

build_from_source() {
    local tag=$1; shift
    local flags=("$@")
    mkdir -p "${tag}"
    (
        cd "${tag}"
        gfortran -c "${flags[@]}" "${common[@]}" "${script_dir}/stub_wrf.F90"
        gfortran -c "${flags[@]}" "${common[@]}" \
            -o module_mp_etanew.o "${mp_source}"
        # -fconvert only takes effect in the main program's unit; WRF reads
        # ETAMPNEW_DATA.expanded_rain big-endian.
        gfortran -c "${flags[@]}" "${common[@]}" -fconvert=big-endian \
            "${script_dir}/run_etanew.F90"
        gfortran "${flags[@]}" -o run_etanew stub_wrf.o module_mp_etanew.o \
            run_etanew.o
        nm -u module_mp_etanew.o | sed 's/^ *//' | sort \
            > undefined-symbols.txt
    )
    run_one "${tag}"
}

build_from_source O0 -O0
if grep -q '_ZGV' O0/undefined-symbols.txt; then
    echo "-O0 object pulled in a libmvec vector symbol" >&2
    exit 4
fi
build_from_source wrfflags -O2 -fno-fast-math -ffp-contract=off \
    -fno-tree-vectorize
build_from_source cov -O0 --coverage

if [[ -n "${wrf_build}" ]]; then
    obj="${wrf_build}/phys/module_mp_etanew.o"
    mod="${wrf_build}/phys/module_mp_etanew.mod"
    mkdir -p wrfobj
    (
        cd wrfobj
        cp "${obj}" "${mod}" .
        gfortran -c -O0 "${common[@]}" "${script_dir}/stub_wrf.F90"
        gfortran -c -O0 "${common[@]}" -fconvert=big-endian -I. \
            "${script_dir}/run_etanew.F90"
        gfortran -o run_etanew stub_wrf.o module_mp_etanew.o run_etanew.o
        nm -u module_mp_etanew.o | sed 's/^ *//' | sort \
            > undefined-symbols.txt
    )
    run_one wrfobj
fi

(
    cd cov
    gcov -o . module_mp_etanew.o > gcov.stdout 2>&1 || true
)

{
    echo "wrf_commit ${pinned_commit}"
    echo "gfortran $(gfortran -dumpfullversion)"
    echo "glibc $(ldd --version | head -1 | awk '{print $NF}')"
    sha256sum "${mp_source}" "${table}" | sed "s#${source_root}/##"
    for tag in O0 wrfflags wrfobj; do
        [[ -d ${tag} ]] || continue
        sha256sum ${tag}/levels.csv ${tag}/surface.csv ${tag}/init.csv
    done
    sha256sum columns_inputs.csv
} > SHA256SUMS
cat SHA256SUMS
for tag in wrfflags wrfobj; do
    [[ -d ${tag} ]] || continue
    echo "== O0 vs ${tag}"
    python3 "${script_dir}/compare_etanew.py" O0 ${tag} || true
done
