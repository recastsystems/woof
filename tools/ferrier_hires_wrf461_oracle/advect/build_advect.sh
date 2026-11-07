#!/usr/bin/env bash
# Build and run the WRF v4.6.1 Ferrier-Aligo advected-rime (mp_physics=15,
# FER_HIRES_ADVECT) column oracle from byte-unmodified WRF source.  CPU only.
#
# usage: build_advect.sh WRF_SOURCE_ROOT BUILD_DIR
#
# Writes into BUILD_DIR:
#   columns_inputs.csv, columns.in        (make_columns_advect.py)
#   O0/levels.csv, O0/surface.csv          the reference (-O0, scalar libm)
#   O2vec/levels.csv, O2vec/surface.csv    WRF's own GNU flags (-O2
#                                          -ftree-vectorize -funroll-loops),
#                                          evidence only
#   cov/advect.gcov                        gcov line coverage of the -O0
#                                          module over the column set
#   SHA256SUMS
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build_advect.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
harness=$(dirname "${here}")

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
fer_source="${source_root}/phys/module_mp_fer_hires.F"
table="${source_root}/run/ETAMPNEW_DATA.expanded_rain"
test -f "${fer_source}"
test -f "${table}"
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- \
        phys/module_mp_fer_hires.F run/ETAMPNEW_DATA.expanded_rain; then
    echo "module_mp_fer_hires.F or its table differs from pinned HEAD" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"
python3 "${here}/make_columns_advect.py" "${build_dir}"

common=(-cpp -ffree-form -ffree-line-length-none -DRWORDSIZE=4
        -DIWORDSIZE=4 -fallow-argument-mismatch)
build_one() {
    local tag=$1; shift
    mkdir -p "${tag}"
    (
        cd "${tag}"
        cp "${fer_source}" module_mp_fer_hires.F
        gfortran -c "$@" "${common[@]}" "${harness}/stub_wrf.F90"
        gfortran -c "$@" "${common[@]}" -o module_mp_fer_hires.o \
            module_mp_fer_hires.F 2> compile.stderr
        # -fconvert only takes effect in the main program's unit; WRF reads
        # ETAMPNEW_DATA.expanded_rain big-endian.
        gfortran -c "$@" "${common[@]}" -fconvert=big-endian \
            "${here}/run_ferrier_advect.F90"
        gfortran "$@" -o run_ferrier_advect stub_wrf.o \
            module_mp_fer_hires.o run_ferrier_advect.o
        nm -u module_mp_fer_hires.o | sed 's/^ *//' | sort \
            > undefined-symbols.txt
        ln -sf "${table}" ETAMPNEW_DATA.expanded_rain
        ln -sf ../columns.in columns.in
        ./run_ferrier_advect > run.stdout 2> run.stderr
    )
}

build_one O0 -O0
if grep -q '_ZGV' O0/undefined-symbols.txt; then
    echo "-O0 Ferrier object pulled in a libmvec vector symbol" >&2
    exit 4
fi
build_one O2vec -O2 -ftree-vectorize -funroll-loops
build_one cov -O0 --coverage
( cd cov && gcov module_mp_fer_hires.F > /dev/null 2>&1 \
    && mv module_mp_fer_hires.F.gcov advect.gcov )

{
    echo "wrf_commit ${pinned_commit}"
    echo "gfortran $(gfortran -dumpfullversion)"
    echo "glibc $(ldd --version | head -1 | awk '{print $NF}')"
    sha256sum "${fer_source}" "${table}" | sed "s#${source_root}/##"
    sha256sum columns_inputs.csv O0/levels.csv O0/surface.csv \
        O2vec/levels.csv O2vec/surface.csv cov/levels.csv cov/surface.csv
} > SHA256SUMS
cat SHA256SUMS
python3 "${harness}/compare_ferrier.py" O0 O2vec columns_inputs.csv || true
python3 "${harness}/compare_ferrier.py" O0 cov columns_inputs.csv || true
