#!/usr/bin/env bash
# Build and run the standalone SBU-YLin (mp_physics = 13) column oracle from
# WRF v4.6.1's byte-unmodified phys/module_mp_sbu_ylin.F, twice:
#   pristine  the file exactly as pinned
#   declared  the same file with WOOF's declared fixes applied
#             (apply_declared_fixes.py: D1 psmltevp abr, D2 av_s in the snow
#             collection kernel, D3 the qvmin repair order)
# and write both outputs for the same case file.
#
# Reference flags: the sprint's scalar referee build of WRF v4.6.1
# (/work/wrf/build/scalar: FCOPTIM = -O2 -fno-fast-math -ffp-contract=off
# -fno-tree-vectorize, plus WRF's free-form options), so the column oracle
# and the full referee wrf.exe compile this module identically.  Stock WRF
# compiles phys/ at -O2 -ftree-vectorize, where gfortran may route EXP
# through libmvec's vector variants; the referee must stay on the scalar
# float32 calls the port reproduces, so the script FAILS if any _ZGV symbol
# appears in an object it links, and it also fails if a vectorisable EXP
# positive control at the stock flags produces none (silence from nm must
# be evidence, not luck).
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
pinned_sha256="$(cat "${script_dir}/module_mp_sbu_ylin.F.sha256")"
scheme_source="${source_root}/phys/module_mp_sbu_ylin.F"
test -f "${scheme_source}"

# The reference is the pinned file, not merely one with the right name.
got_sha256=$(sha256sum "${scheme_source}" | cut -d' ' -f1)
if [[ "${got_sha256}" != "${pinned_sha256}" ]]; then
    echo "module_mp_sbu_ylin.F sha256 ${got_sha256} != pinned ${pinned_sha256}" >&2
    exit 3
fi
if git -C "${source_root}" rev-parse HEAD >/dev/null 2>&1; then
    head_commit=$(git -C "${source_root}" rev-parse HEAD)
    if [[ "${head_commit}" != "${pinned_commit}" ]]; then
        echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
        exit 3
    fi
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

fflags=(-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -cpp
        -DRWORDSIZE=4 -ffree-form -ffree-line-length-none)

python3 "${script_dir}/apply_declared_fixes.py" "${scheme_source}" \
    module_mp_sbu_ylin_declared.F

gfortran -c "${fflags[@]}" "${script_dir}/stub_wrf_error.F90"
for variant in pristine declared; do
    mkdir -p "${variant}"
    src="${scheme_source}"
    [[ "${variant}" == declared ]] && src="${build_dir}/module_mp_sbu_ylin_declared.F"
    gfortran -c "${fflags[@]}" -I "${build_dir}" -J "${variant}" \
        -o "${variant}/module_mp_sbu_ylin.o" "${src}"
    gfortran -c "${fflags[@]}" -I "${variant}" -J "${variant}" \
        -o "${variant}/run_sbu_ylin.o" "${script_dir}/run_sbu_ylin.F90"
    gfortran -o "run_sbu_ylin_${variant}" stub_wrf_error.o \
        "${variant}/module_mp_sbu_ylin.o" "${variant}/run_sbu_ylin.o"
    nm -u "${variant}/module_mp_sbu_ylin.o" | sed 's/^ *//' | sort \
        > "undefined-${variant}.txt"
    if grep -q '_ZGV' "undefined-${variant}.txt"; then
        echo "${variant} object pulled in a libmvec vector symbol" >&2
        exit 4
    fi
done

gfortran -c -O2 -ftree-vectorize -funroll-loops \
    -o libmvec_positive_control.o "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol" >&2
    exit 5
fi

# The probe pins the two compiler behaviours the port must reproduce rather
# than assume: MAX/MIN on signed zeros, and the integer-power lowering.
gfortran "${fflags[@]}" -o probe_lowering "${script_dir}/probe_lowering.F90"
./probe_lowering > probe-lowering.txt
nm -u pristine/module_mp_sbu_ylin.o | grep -E 'powisf2|powf|expf' \
    > libm-symbols.txt || true

python3 "${script_dir}/make_cases.py" sbu-ylin-cases.txt --names sbu-ylin-case-names.txt
./run_sbu_ylin_pristine sbu-ylin-cases.txt sbu-ylin-pristine.txt
./run_sbu_ylin_declared sbu-ylin-cases.txt sbu-ylin-declared.txt

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# flags:    ${fflags[*]}"
    echo "# scheme libm surface (pristine object):"
    cat libm-symbols.txt
    echo "# lowering probe:"
    cat probe-lowering.txt
} > oracle-report.txt
cat oracle-report.txt

# Exported in-file helper functions are tested even though the column never calls them.
gfortran "${fflags[@]}" -I pristine -J pristine -o run_sbu_leaves \
    "${script_dir}/run_sbu_leaves.F90" pristine/module_mp_sbu_ylin.o stub_wrf_error.o
./run_sbu_leaves

sha256sum "${scheme_source}" module_mp_sbu_ylin_declared.F \
    "${script_dir}"/*.F90 "${script_dir}"/*.py \
    sbu-ylin-cases.txt sbu-ylin-pristine.txt sbu-ylin-declared.txt sbu-ylin-leaves.txt \
    > oracle-sha256sums.txt
echo "${pinned_commit}" > wrf-commit.txt
echo "SBU-YLin oracle built in ${build_dir}"
