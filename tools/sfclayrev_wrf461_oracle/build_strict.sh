#!/usr/bin/env bash
# The revised MM5 oracle built with the pinned strict recipe: the flags every
# strict wrf.exe 4.6.1 referee is compiled with (scripts/build_wrf_strict_variant.sh
# lineage: GNU Fortran 13.3.0, FCOPTIM -O2 -fno-fast-math -ffp-contract=off
# -fno-tree-vectorize) on glibc's scalar libm.  This is the reference
# tests/test_sfclayrev_wrf461_parity.py grades the kernel against.
#
#   bash build_strict.sh WRF_SOURCE_DIR BUILD_DIR [INPUTS]
#
# WRF_SOURCE_DIR holds phys/ccpp_kind_types.F, phys/physics_mmm/sf_sfclayrev.F90
# and phys/module_sf_sfclayrev.F of WRF v4.6.1 (d66e442f); each is checked
# against its pinned SHA-256.  Writes BUILD_DIR/sfclayrev-outputs-strict.hex and
# strict-receipt.txt.
#
# Why not build.sh's -O0 object: at -O2 gfortran folds powf(x, 2.0) into x*x
# (the unstable similarity functions' ym**2., yh**2. and zolf**2.), so the -O0
# object is not the arithmetic the strict wrf.exe runs.  The strict wrf.exe's
# psim_unstable_full imports powf twice, not four times (objdump of the
# referee binary, 2026-10-29).
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build_strict.sh WRF_SOURCE_DIR BUILD_DIR [INPUTS]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
inputs=$(realpath "${3:-${repo_root}/tests/data/oracles/sfclayrev/sfclayrev-inputs.hex}")
kind_source="${source_root}/phys/ccpp_kind_types.F"
scheme_source="${source_root}/phys/physics_mmm/sf_sfclayrev.F90"
wrapper_source="${source_root}/phys/module_sf_sfclayrev.F"
FC=${FC:-gfortran}
FCOPTIM="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize"

check() {
    local got
    got=$(sha256sum "$1" | cut -d' ' -f1)
    [[ "${got}" == "$2" ]] || { echo "$1: sha256 ${got}, want the pinned $2" >&2; exit 3; }
}
check "${kind_source}" a76e1e5b52fc7cd40be9ccb506fde28b1ef2486b812d518f7f1c882766d484db
check "${scheme_source}" 5897bb51c289ecd3f156ccdba975deb72dd538a45a53c9018be42388b463bf8e
check "${wrapper_source}" 388633bac8063c8e773083ea4f280f69f25e607efb903ed221e112e239cb1cb2
case "$(${FC} --version | head -1)" in
    *" 13.3.0"*) ;;
    *) echo "REFUSED: the strict recipe pins GNU Fortran 13.3.0; found '$(${FC} --version | head -1)'" >&2
       exit 3 ;;
esac

mkdir -p "${build_dir}"
cd "${build_dir}"
# EM_CORE=1 is what WRF's ARW build defines (see build.sh).
${FC} -c ${FCOPTIM} -cpp -DRWORDSIZE=4 -ffree-form -ffree-line-length-none \
    "${kind_source}"
${FC} -c ${FCOPTIM} -cpp -ffree-form -ffree-line-length-none \
    -I "${build_dir}" -o sf_sfclayrev_strict.o "${scheme_source}"
${FC} -c ${FCOPTIM} -cpp -DEM_CORE=1 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" -o module_sf_sfclayrev_strict.o "${wrapper_source}"
${FC} -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_sfclayrev.F90"
${FC} -o run_sfclayrev_strict ccpp_kind_types.o sf_sfclayrev_strict.o \
    module_sf_sfclayrev_strict.o run_sfclayrev.o -lm

nm -u sf_sfclayrev_strict.o module_sf_sfclayrev_strict.o | sed 's/^ *//' \
    | grep -v ':$' | sort -u > undefined-strict.txt
if grep -q '_ZGV' undefined-strict.txt; then
    echo "the strict object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-strict.txt >&2
    exit 4
fi
./run_sfclayrev_strict "${inputs}" sfclayrev-outputs-strict.hex

{
    echo "# gfortran: $(${FC} --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# FCOPTIM:  ${FCOPTIM}"
    echo "# libm symbols the strict scheme and wrapper import:"
    grep -E 'expf|powf|logf|atanf|sqrtf|_ZGV' undefined-strict.txt || true
    echo "# powf calls in the unstable similarity functions (folded x**2. leaves 2 + 2):"
    objdump -dr --no-show-raw-insn sf_sfclayrev_strict.o \
        | awk '/^[0-9a-f]+ <.*>:$/{f=($2 ~ /psi[mh]_unstable_full/) ? $2 : ""} f && /R_X86_64_PLT32[ \t]+powf/{n[f]++} END{for (k in n) print "#   " k, n[k]}'
    sha256sum "${kind_source}" "${scheme_source}" "${wrapper_source}" \
        "${script_dir}/run_sfclayrev.F90" "${inputs}" \
        sfclayrev-outputs-strict.hex
} > strict-receipt.txt
cat strict-receipt.txt
echo "strict oracle run in ${build_dir}"
