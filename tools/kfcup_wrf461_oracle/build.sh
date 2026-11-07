#!/usr/bin/env bash
# Build and run the KF-CuP (cu_physics = 10) column oracle against WRF v4.6.1.
#
#   bash tools/kfcup_wrf461_oracle/build.sh WRF_SOURCE_ROOT WRF_BUILD_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT  pristine v4.6.1 tree (the pins below are checked here)
# WRF_BUILD_ROOT   a compiled serial WRF v4.6.1 tree (configure 32) whose
#                  phys/module_cu_kfcup.F is byte-identical to the pinned one
# BUILD_DIR        scratch directory for this oracle
#
# TWO ARMS, ONE ANSWER.
#   arm "wrf"  links WRF's OWN compiled object, phys/module_cu_kfcup.o, as the
#              WRF build made it (FCOPTIM -O2 -fno-fast-math -ffp-contract=off),
#              against WRF's own module_model_constants.o and
#              module_state_description.o.  This is the artifact.
#   arm "o0"   recompiles WRF's own preprocessed phys/module_cu_kfcup.f90 at
#              -O0, the scalar reference every other oracle in this tree uses.
# The build FAILS unless the two arms write byte-identical CSVs, so the
# capture is the artifact's answer and the -O0 reading of it at once.
#
# libmvec: the build FAILS if either kfcup object references a _ZGV*
# vector-math symbol (none does on gfortran 13.3 / glibc 2.39).
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT WRF_BUILD_ROOT BUILD_DIR" >&2
    exit 2
fi
src=$(realpath "$1")
wrfb=$(realpath "$2")
out=$(realpath -m "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- pins: WRF v4.6.1 (d66e442), LF line endings -------------------------
declare -A pinned=(
  ["phys/module_cu_kfcup.F"]="d5773c66c9e353b7838bc14e59d66905091425ce449e84cea58f14e7725b074c"
  ["phys/module_cumulus_driver.F"]="d73eb4b670599e0bbf98a3661b5dbf17413aed01139d18fa1a84161578dc31e3"
  ["phys/module_radiation_driver.F"]="779875651512709c47d50ef6c1e1216ad94726f1996c5e45a0db8eb800733ffa"
  ["phys/module_physics_addtendc.F"]="ecfe34c06bb3e91a5398a616a69a69a5aa155e73dc082ef0745421df46331894"
  ["share/module_model_constants.F"]="5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
)
for rel in "${!pinned[@]}"; do
    for root in "${src}" "${wrfb}"; do
        actual=$(sha256sum "${root}/${rel}" | awk '{print $1}')
        if [[ "${actual}" != "${pinned[$rel]}" ]]; then
            echo "FATAL: ${root}/${rel} is ${actual}, not the pinned ${pinned[$rel]}" >&2
            exit 3
        fi
    done
done

mkdir -p "${out}/wrf" "${out}/o0" "${out}/extract"
cd "${out}"
{
  gfortran --version | sed -n 1p
  ldd --version | sed -n 1p
  echo "WRF build: ${wrfb}"
  grep -E '^(FCOPTIM|FCBASEOPTS_NO_G|FORMAT_FREE)' "${wrfb}/configure.wrf"
} > compiler.txt

# --- libmvec guard on WRF's own object ------------------------------------
if nm -u "${wrfb}/phys/module_cu_kfcup.o" | grep -q '_ZGV'; then
    echo "FATAL: WRF's module_cu_kfcup.o references libmvec" >&2; exit 5
fi
nm -u "${wrfb}/phys/module_cu_kfcup.o" > wrf-object-undefined-symbols.txt

# --- extracts for the radiation merge and the NCA clock -------------------
# Taken from WRF's own preprocessed .f90 between textual markers and
# checked: the marker lines must appear exactly once, and the extract's
# digest is recorded.  Nothing is retyped.
f90rad="${wrfb}/phys/module_radiation_driver.f90"
awk '/^   SUBROUTINE cal_cldfra1\(/{p=1} p{print} /^   END SUBROUTINE cal_cldfra1/{p=0}' \
    "${f90rad}" > extract/cal_cldfra1.inc
awk '/^     if\(present\(cldfra_cup\)\) then/{p=1} /^     IF\( shcu_physics == 5 \) THEN/{p=0} p{print}' \
    "${f90rad}" > extract/kfcup_merge_block.inc
f90adv="${wrfb}/phys/module_physics_addtendc.f90"
awk '/^   CASE \(KFETASCHEME, MSKFSCHEME, KFCUPSCHEME\)/{p=1;next} /^   CASE DEFAULT/{p=0} p{print}' \
    "${f90adv}" > extract/kfcup_advance_block.inc
for f in cal_cldfra1 kfcup_merge_block kfcup_advance_block; do
    n=$(wc -l < "extract/${f}.inc")
    if [[ "${n}" -lt 20 ]]; then
        echo "FATAL: extract ${f} is ${n} lines; a marker moved" >&2; exit 4
    fi
done
sha256sum extract/*.inc > extract-sha256sums.txt

build_arm() {
    local arm=$1
    local dir="${out}/${arm}"
    cd "${dir}"
    local inc=()
    local objs=()
    if [[ "${arm}" == "wrf" ]]; then
        inc=(-I "${wrfb}/phys" -I "${wrfb}/share" -I "${wrfb}/frame")
        objs=("${wrfb}/phys/module_cu_kfcup.o" "${wrfb}/share/module_model_constants.o"
              "${wrfb}/frame/module_state_description.o")
        gfortran -c -O0 -ffree-form -ffree-line-length-none -o stubs.o "${here}/kfcup_stubs.F90"
    else
        cp "${wrfb}/phys/module_cu_kfcup.f90" .
        cp "${wrfb}/share/module_model_constants.f90" .
        gfortran -c -O0 -ffree-form -ffree-line-length-none -o stubs.o "${here}/kfcup_stubs.F90"
        gfortran -c -O0 -ffree-form -ffree-line-length-none module_model_constants.f90
        # module_state_description: WRF's own object and .mod (num_chem and
        # the scheme selectors); only module_cu_kfcup itself is recompiled.
        cp "${wrfb}/frame/module_state_description.mod" .
        cp "${wrfb}/frame/module_wrf_error.mod" .
        gfortran -c -O0 -w -ffree-form -ffree-line-length-none module_cu_kfcup.f90
        if nm -u module_cu_kfcup.o | grep -q '_ZGV'; then
            echo "FATAL: -O0 module_cu_kfcup.o references libmvec" >&2; exit 5
        fi
        objs=(module_cu_kfcup.o module_model_constants.o "${wrfb}/frame/module_state_description.o")
        inc=(-I .)
    fi
    gfortran -c -O0 -ffree-form -ffree-line-length-none "${inc[@]}" "${here}/kfcup_fixture.F90"
    for prog in ${KFCUP_PROGS:-run_kfcup_cps run_kfcup_stages run_kfcup_rad}; do
        gfortran -c -O0 -ffree-form -ffree-line-length-none "${inc[@]}" -I "${out}/extract" \
            "${here}/${prog}.F90"
        gfortran -o "${prog}" "${prog}.o" kfcup_fixture.o stubs.o "${objs[@]}"
        "./${prog}" 2> "${prog}.stderr"
    done
    cd "${out}"
}

build_arm wrf
build_arm o0

# --- uninitialised-memory probe -------------------------------------------
# The -O0 recompilation twice more, identical but for gfortran's
# -finit-* seeds (signalling NaN and sentinels, then zeros).  Any output
# word that moves means WRF's answer depends on memory it never wrote, and
# no port could match it in principle.  (LNTERMS is that case, which is why
# run_kfcup_cps does not capture it.)
for mode in nan zero; do
    d="${out}/uninit-${mode}"
    mkdir -p "${d}"; cd "${d}"
    if [[ "${mode}" == nan ]]; then
        seeds=(-finit-real=snan -finit-integer=-77777 -finit-logical=true)
    else
        seeds=(-finit-real=zero -finit-integer=0 -finit-logical=false)
    fi
    cp "${out}/o0/"*.mod "${out}/o0/module_model_constants.o" "${out}/o0/stubs.o" .
    gfortran -c -O0 -w -ffree-form -ffree-line-length-none "${seeds[@]}" "${out}/o0/module_cu_kfcup.f90"
    gfortran -c -O0 -ffree-form -ffree-line-length-none -I . "${here}/kfcup_fixture.F90"
    for prog in run_kfcup_cps run_kfcup_stages; do
        gfortran -c -O0 -ffree-form -ffree-line-length-none -I . "${here}/${prog}.F90"
        gfortran -o "${prog}" "${prog}.o" kfcup_fixture.o stubs.o module_cu_kfcup.o             module_model_constants.o "${wrfb}/frame/module_state_description.o"
        "./${prog}" 2> "${prog}.stderr"
    done
    for f in *.csv; do
        if ! cmp -s "${f}" "${out}/o0/${f}"; then
            echo "FATAL: ${f} depends on uninitialised memory (${mode} seeds)" >&2; exit 8
        fi
    done
    cd "${out}"
done

# --- the two arms must agree word for word --------------------------------
for f in wrf/*.csv; do
    b=$(basename "$f")
    if ! cmp -s "wrf/${b}" "o0/${b}"; then
        echo "FATAL: arm wrf and arm o0 differ on ${b}" >&2; exit 6
    fi
done

# --- column independence ---------------------------------------------------
bad=$(awk -F, 'NR > 1 && $3 != 0' wrf/kfcup-isolation.csv | wc -l)
if [[ "${bad}" -ne 0 ]]; then
    echo "FATAL: ${bad} isolation rows differ" >&2; exit 7
fi

cp wrf/*.csv .
sha256sum *.csv compiler.txt extract-sha256sums.txt wrf-object-undefined-symbols.txt > oracle-sha256sums.txt
echo "built and ran in ${out}"
wc -l *.csv
