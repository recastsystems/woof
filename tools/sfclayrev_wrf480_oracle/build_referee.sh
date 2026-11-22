#!/usr/bin/env bash
# The revised MM5 oracle for WRF v4.8.0 (kim_tofd off and on), linked against
# the native 4.8.0 strict serial referee build's OWN objects: the code its
# wrf.exe runs (the WRF 4.8.0 port plan, section 1:
# localhost/wrfref2404:1, GNU Fortran 13.3.0, FCOPTIM -O2 -fno-fast-math
# -ffp-contract=off -fno-tree-vectorize, glibc 2.39, WRF 06d4240ae989).
# Run it inside that container.
#
#   bash build_referee.sh REFEREE_BUILD_DIR OUT_DIR INPUTS [TOFD_FACTOR] [STEPS]
#
# Each pinned source and object is checked by SHA-256 before linking; only
# the driver (run_sfclayrev480.F90, data shuffling, -O0) is compiled here.
set -euo pipefail
[[ $# -ge 3 ]] || { echo "usage: build_referee.sh REFEREE_BUILD_DIR OUT_DIR INPUTS [TOFD_FACTOR] [STEPS]" >&2; exit 2; }
ref=$(realpath "$1"); out=$(realpath -m "$2"); inputs=$(realpath "$3")
factor=${4:-.003}; steps=${5:-3}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FC=${FC:-gfortran}
check() {
    local got; got=$(sha256sum "$1" | cut -d' ' -f1)
    [[ "${got}" == "$2" ]] || { echo "$1: sha256 ${got}, want the pinned $2" >&2; exit 3; }
}
# WRF v4.8.0 06d4240ae989 sources (physics_mmm 550b5b49).
check "${ref}/phys/ccpp_kind_types.F" 0cc7b2d71d8a93ff1f2b401324de8fd1ee70ba598f83d89254d6dbe8c31d0e4b
check "${ref}/phys/physics_mmm/sf_sfclayrev.F90" 1bff41ec6d8d2bf73a07a40c5a6dfac2565b46601c9ee3a9b0f6aa6fe85ed698
check "${ref}/phys/module_sf_sfclayrev.F" 64da720d495f6eb801bbd03c21ea69554ed3b9953a0cc2a6e1e9044111c9b691
grep -q '"wrf_exe_sha256": "f6f47f86de59afe8a848a67dcd612f5ac9988e7b2b174467283c7022b59f1a5c"' "${ref}/BUILD.json" \
    || { echo "${ref} is not the pinned 4.8.0 strict serial pristine referee" >&2; exit 3; }
case "$(${FC} --version | head -1)" in
    *" 13.3.0"*) ;;
    *) echo "REFUSED: the strict recipe pins GNU Fortran 13.3.0" >&2; exit 3 ;;
esac
mkdir -p "${out}"; cd "${out}"
${FC} -c -O0 -ffree-form -ffree-line-length-none -I "${ref}/phys" \
    -o run_sfclayrev480.o "${here}/run_sfclayrev480.F90"
${FC} -o run_sfclayrev480 run_sfclayrev480.o "${ref}/phys/ccpp_kind_types.o" \
    "${ref}/phys/physics_mmm/sf_sfclayrev.o" "${ref}/phys/module_sf_sfclayrev.o" -lm
./run_sfclayrev480 "${inputs}" sfclayrev480-outputs.hex "${factor}" "${steps}"
{
    echo "# gfortran: $(${FC} --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# referee:  ${ref} (BUILD.json wrf.exe f6f47f86de59...)"
    echo "# tofd_factor ${factor}, chained steps ${steps}"
    echo "# libm symbols the referee scheme object imports:"
    nm -u "${ref}/phys/physics_mmm/sf_sfclayrev.o" | grep -E 'expf|powf|logf|atanf|sqrtf|_ZGV' || true
    sha256sum "${ref}/phys/ccpp_kind_types.o" "${ref}/phys/physics_mmm/sf_sfclayrev.o" \
        "${ref}/phys/module_sf_sfclayrev.o" "${here}/run_sfclayrev480.F90" "${inputs}" \
        sfclayrev480-outputs.hex
} > referee-receipt.txt
cat referee-receipt.txt
