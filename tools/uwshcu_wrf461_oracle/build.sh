#!/usr/bin/env bash
# Build and run the WRF v4.6.1 UW shallow cumulus (shcu_physics=2) oracle.
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR CASES_DIR
#
# WRF_SOURCE_ROOT holds a pristine WRF v4.6.1 tree (tag v4.6.1, commit
# d66e442fccc04111067e29274c9f9eaccc3cef28); every file SOURCES.sha256 pins
# is checked against its pin before anything compiles, so a fixture cannot
# come from an edited file.  CASES_DIR holds what make_cases.py wrote.
# Builds, each in its own directory (Fortran module files would collide):
#
#   ref    THE REFERENCE: byte-unmodified WRF sources at the sprint's WRF
#          "scalar" build flags (/work/wrf/provenance/flags-scalar.txt:
#          -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize), the
#          WRF preprocessor defines of an EM, RWORDSIZE=4 build.  Its
#          fixtures are the fixtures of record.  At -O2 gcc lowers a real
#          exponent of exactly 2.0 to x*x (module_shcu_camuwshcu.F:2540's
#          wue**2._r8), as the port does; every other real exponent stays a
#          libm pow call.
#   strict WRF's "strict" flags (the same without -fno-tree-vectorize).
#          Evidence only: the script reports whether its outputs equal the
#          reference's byte for byte and which libmvec symbols it imports.
#   o0     -O0.  Evidence only: at -O0 wue**2._r8 is a pow call, so this
#          build may differ from the reference in the last bit where pow
#          and x*x round differently; the script reports it.
#   snan   the reference with every local real initialised to a signalling
#          NaN and every local integer to -999999.  Required identical to
#          the reference: no uninitialised local reaches an output on the
#          cases run.
#
# libmvec guard: the reference objects must not reference any _ZGV* vector
# symbol; a positive control proves the grep can fire on this toolchain.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR CASES_DIR" >&2
    exit 2
fi
src=$(realpath "$1")
out=$(realpath -m "$2")
cases=$(realpath "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

( cd "${src}" && grep -v '^#' "${here}/SOURCES.sha256" | sha256sum -c --quiet - )
echo "sources: all $(grep -vc '^#' "${here}/SOURCES.sha256") pins verified"

WRFDEFS="-DEM_CORE=1 -DNMM_CORE=0 -DCOAMPS_CORE=0 -DDA_CORE=0 -DEXP_CORE=0 \
-DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4 -Dwrfmodel -DGRIB1 \
-DINTIO -DLIMIT_ARGS -DCONFIG_BUF_LEN=65536 -DMAX_DOMAINS_F=21 -DMAX_HISTORY=25 \
-DNMM_NEST=0"
BASE="-cpp -ffree-form -ffree-line-length-none -fno-range-check -g ${WRFDEFS}"
REF="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize"
STRICT="-O2 -fno-fast-math -ffp-contract=off"

# compile order = USE order
CAM=(
  phys/module_cam_shr_kind_mod.F
  phys/module_cam_shr_const_mod.F
  phys/module_cam_physconst.F
  phys/module_cam_support.F
  phys/module_cam_constituents.F
  phys/module_cam_gffgch.F
  phys/module_cam_wv_saturation.F
  phys/module_cam_esinti.F
  phys/module_cam_error_function.F
  phys/module_shcu_camuwshcu.F
  phys/module_shcu_camuwshcu_driver.F
)
UWIO="${here}/../uwpbl_wrf471_oracle/uwio.F90"

build_variant() {
    local name=$1 opt=$2 extra=$3
    local dir="${out}/${name}"
    rm -rf "${dir}"
    mkdir -p "${dir}"
    (
        cd "${dir}"
        gfortran -c ${opt} ${BASE} ${extra} "${here}/stub_wrf.F90"
        objs=(stub_wrf.o)
        for f in "${CAM[@]}"; do
            local b
            b=$(basename "${f}" .F)
            gfortran -c ${opt} ${BASE} ${extra} -o "${b}.o" "${src}/${f}"
            objs+=("${b}.o")
        done
        gfortran -c ${opt} ${BASE} ${extra} "${UWIO}"
        gfortran -c ${opt} ${BASE} ${extra} "${here}/run_uwshcu.F90"
        gfortran -c ${opt} ${BASE} ${extra} "${here}/run_uwshcu_core.F90"
        gfortran -o run_uwshcu "${objs[@]}" uwio.o run_uwshcu.o
        gfortran -o run_uwshcu_core "${objs[@]}" uwio.o run_uwshcu_core.o
        gfortran ${opt} -o probe_constants "${here}/probe_constants.F90"
        for o in "${objs[@]}"; do
            nm -u "${o}" | sed 's/^ *U *//' | sed "s|^|${o}: |"
        done | sort > undefined.txt
    )
}

build_variant ref "${REF}" ""
build_variant strict "${STRICT}" ""
build_variant o0 "-O0" ""
build_variant snan "${REF}" "-finit-real=snan -finit-integer=-999999 -finit-derived"

# --- libmvec guard ------------------------------------------------------------
if grep -q '_ZGV' "${out}/ref/undefined.txt"; then
    echo "reference objects pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' "${out}/ref/undefined.txt" >&2
    exit 4
fi
cat > "${out}/libmvec_control.f90" <<'EOF'
subroutine control(x, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  x = exp(x)
end subroutine control
EOF
( cd "${out}" && gfortran -c -Ofast -ftree-vectorize -mavx2 -o libmvec_control.o libmvec_control.f90 )
if ! nm -u "${out}/libmvec_control.o" | grep -q '_ZGV'; then
    echo "libmvec positive control produced no _ZGV symbol; the guard is vacuous" >&2
    exit 5
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    for v in ref strict o0; do
        echo "# libm symbols the ${v} objects call:"
        grep -E ': (exp|log|pow|sqrt|log10|erfc|_ZGV)' "${out}/${v}/undefined.txt" || true
    done
} > "${out}/libm-report.txt"
cat "${out}/libm-report.txt"

# --- run every case file through every variant ----------------------------------
status=0
mkdir -p "${out}/fixtures"
for c in "${cases}"/cases-*.bin "${cases}"/core-*.bin; do
    stem=$(basename "${c}" .bin)
    prog=run_uwshcu
    [[ "${stem}" == core-* ]] && prog=run_uwshcu_core
    for v in ref strict o0 snan; do
        ( cd "${out}/${v}" && ./${prog} "${c}" "${out}/${v}/${stem}" )
        if [[ ! -s "${out}/${v}/${stem}.bin" || ! -s "${out}/${v}/${stem}.manifest" ]]; then
            echo "FAIL: ${v} wrote an empty fixture for ${stem}" >&2
            exit 7
        fi
    done
    cp "${out}/ref/${stem}.bin" "${out}/ref/${stem}.manifest" "${out}/fixtures/"
    # Value comparison: at -O2 gcc may emit maxsd/minsd with either operand
    # order for a Fortran MAX/MIN, so a tie between +0 and -0 can come out
    # with either sign depending on surrounding code generation (measured:
    # the snan build flips the sign of some exactly-zero NUMLIQ tendencies).
    # Signed-zero-only differences are counted and reported, never failed.
    for v in snan strict o0; do
        read -r nbad nsz < <(python3 -c "import sys; sys.path.insert(0, '${here}'); import summarize as s; print(*s.same_values('${out}/ref/${stem}', '${out}/${v}/${stem}'))")
        if [[ "${nbad}" != 0 ]]; then
            if [[ "${v}" == snan ]]; then
                echo "FAIL: snan build differs from the reference on ${stem} (${nbad} values)" >&2
                status=6
            else
                echo "${stem}: ${v} build DIFFERS from the reference in ${nbad} values (evidence, not a failure)"
            fi
        else
            echo "${stem}: ${v} build value-identical to the reference (${nsz} signed-zero-only words)"
        fi
    done
done
cp "${cases}/cases-index.json" "${out}/fixtures/" 2>/dev/null || true
"${out}/ref/probe_constants" > "${out}/fixtures/constants.txt"
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "reference flags: ${REF}"
} > "${out}/fixtures/toolchain.txt"
( cd "${out}/fixtures" && sha256sum ./*.bin ./*.manifest ./constants.txt > SHA256SUMS )
echo "oracle fixtures in ${out}/fixtures"
exit ${status}
