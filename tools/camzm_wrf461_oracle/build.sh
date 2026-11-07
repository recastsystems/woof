#!/usr/bin/env bash
# Build and run the WRF v4.6.1 CAM Zhang-McFarlane (cu_physics = 7) column
# oracle.
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR [CASES_DIR]
#
# WRF_SOURCE_ROOT is a WRF v4.6.1 tree; every file SOURCES.sha256 pins is
# checked before anything compiles, so a fixture cannot come from an edited
# file.  CASES_DIR defaults to BUILD_DIR/cases, written by make_cases.py.
# Variants, each in its own directory because Fortran module files collide:
#
#   stock    byte-unmodified WRF sources at the WRF 4.6.1 strict-scalar
#            build's own flags (/work/wrf/provenance flags-scalar:
#            -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize), the
#            WRF preprocessor defines of an EM RWORDSIZE=4 build.  This is
#            WRF as it ships.
#   fixed    the same sources with fixes.patch applied (the two WRF defects
#            WOOF does not reproduce; see README.md).  Same flags.  ITS
#            fixtures are altered-source attribution instruments only.
#   o0       fixed at -O0.  Required byte-identical to fixed: the arithmetic
#            does not depend on the optimiser.
#   snan     fixed with every local real initialised to a signalling NaN and
#            every local integer to -999999.  Required byte-identical: no
#            uninitialised local reaches an output on the cases run.
#
# The stock-versus-fixed difference is MEASURED and reported per field by
# compare_variants.py, so the declared divergence is a number, not a claim.
#
# libmvec guard: no object of record may reference a _ZGV* vector symbol;
# a positive control proves the grep can fire on this toolchain.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [CASES_DIR]" >&2
    exit 2
fi
src=$(realpath "$1")
out=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${out}"
cases=${3:-${out}/cases}
if [[ $# -lt 3 ]]; then
    python3 "${here}/make_cases.py" "${cases}"
fi
cases=$(realpath "${cases}")

( cd "${src}" && grep -v '^#' "${here}/SOURCES.sha256" | sha256sum -c --quiet - )
echo "sources: all $(grep -vc '^#' "${here}/SOURCES.sha256") pins verified"

WRFDEFS="-DEM_CORE=1 -DNMM_CORE=0 -DCOAMPS_CORE=0 -DDA_CORE=0 -DEXP_CORE=0 \
-DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4 -Dwrfmodel -DGRIB1 \
-DINTIO -DLIMIT_ARGS -DCONFIG_BUF_LEN=65536 -DMAX_DOMAINS_F=21 -DMAX_HISTORY=25 \
-DNMM_NEST=0 -DNONSTANDARD_SYSTEM_SUBR -DUSE_ALLOCATABLES"
BASE="-cpp -w -ffree-form -ffree-line-length-none -fallow-argument-mismatch \
-fallow-invalid-boz ${WRFDEFS}"
SCALAR="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize"

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
  phys/module_cam_cldwat.F
  phys/module_cu_camzm.F
  share/module_model_constants.F
  phys/module_cu_camzm_driver.F
)

# the patched tree the fixed/o0/snan variants compile
fixsrc="${out}/fixed-src"
rm -rf "${fixsrc}"
mkdir -p "${fixsrc}/phys" "${fixsrc}/share"
for f in "${CAM[@]}"; do cp "${src}/${f}" "${fixsrc}/${f}"; done
( cd "${fixsrc}" && patch -p1 --no-backup-if-mismatch < "${here}/fixes.patch" )

build_variant() {
    local name=$1 opt=$2 tree=$3 extra=$4
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
            gfortran -c ${opt} ${BASE} ${extra} -o "${b}.o" "${tree}/${f}"
            objs+=("${b}.o")
        done
        gfortran -c ${opt} ${BASE} ${extra} "${here}/zmio.F90"
        gfortran -c ${opt} ${BASE} ${extra} "${here}/run_camzm.F90"
        gfortran -o run_camzm "${objs[@]}" zmio.o run_camzm.o
        for o in "${objs[@]}"; do
            nm -u "${o}" | sed 's/^ *U *//' | sed "s|^|${o}: |"
        done | sort > undefined.txt
    )
}

build_variant stock "${SCALAR}" "${src}" ""
build_variant fixed "${SCALAR}" "${fixsrc}" ""
build_variant o0 "-O0" "${fixsrc}" ""
build_variant snan "${SCALAR}" "${fixsrc}" "-finit-real=snan -finit-integer=-999999 -finit-derived"

# --- libmvec guard ------------------------------------------------------------
for v in stock fixed; do
    if grep -q '_ZGV' "${out}/${v}/undefined.txt"; then
        echo "${v} objects pulled in a libmvec vector symbol:" >&2
        grep '_ZGV' "${out}/${v}/undefined.txt" >&2
        exit 4
    fi
done
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
    echo "# libm symbols the fixture-of-record objects call (fixed, scalar flags):"
    grep -E ': (exp|log|pow|sqrt|expf|logf|powf|sqrtf|__powi|_ZGV)' "${out}/fixed/undefined.txt" || true
} > "${out}/libm-report.txt"
cat "${out}/libm-report.txt"

# --- run every case file through every variant ----------------------------------
status=0
mkdir -p "${out}/fixtures"
for c in "${cases}"/cases-*.bin; do
    stem=zm-$(basename "${c}" .bin | sed 's/^cases-//')
    for v in stock fixed o0 snan; do
        ( cd "${out}/${v}" && ./run_camzm "${c}" "${out}/${v}/${stem}" )
        if [[ ! -s "${out}/${v}/${stem}.bin" || ! -s "${out}/${v}/${stem}.manifest" ]]; then
            echo "FAIL: ${v} wrote an empty fixture for ${stem}" >&2
            exit 7
        fi
    done
    cp "${out}/fixed/${stem}.bin" "${out}/fixed/${stem}.manifest" "${out}/fixtures/"
    cp "${out}/stock/${stem}.bin" "${out}/fixtures/${stem}-stock.bin"
    cp "${out}/stock/${stem}.manifest" "${out}/fixtures/${stem}-stock.manifest"
    # The landfrac fix is an exchange of the two arms of one IF on XLAND, and
    # camzm_driver reads XLAND nowhere else.  So stock WRF handed XLAND with
    # 1 and 2 exchanged must reproduce the fixed build word for word (the
    # recorded inputs keep the true XLAND).  If it does not, the fix moved
    # something besides that test, and the build fails.
    ( cd "${out}/stock" && ./run_camzm "${c}" "${out}/stock/${stem}-swapx" swap-xland )
    if cmp -s "${out}/fixed/${stem}.bin" "${out}/stock/${stem}-swapx.bin"; then
        echo "${stem}: stock WRF with XLAND 1<->2 byte-identical to the fixed build"
    else
        echo "FAIL: stock WRF with XLAND exchanged differs from the fixed build on ${stem}" >&2
        status=8
    fi
    for v in o0 snan; do
        if cmp -s "${out}/fixed/${stem}.bin" "${out}/${v}/${stem}.bin"; then
            echo "${stem}: ${v} byte-identical to the fixture of record"
        else
            echo "FAIL: ${v} build differs from the fixture of record on ${stem}" >&2
            status=6
        fi
    done
done
cp "${cases}/cases-index.json" "${out}/fixtures/"
for c in "${cases}"/cases-*.bin; do
    cp "${c}" "${out}/fixtures/$(basename "${c}" .bin).cases"
done
python3 "${here}/compare_variants.py" "${out}/fixtures" > "${out}/fixtures/stock-vs-fixed.txt"
cat "${out}/fixtures/stock-vs-fixed.txt"
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "flags: ${SCALAR}"
} > "${out}/fixtures/toolchain.txt"
cp "${out}/libm-report.txt" "${out}/fixtures/"
( cd "${out}/fixtures" && sha256sum ./*.bin ./*.manifest ./*.cases > SHA256SUMS )
echo "oracle fixtures in ${out}/fixtures"
exit ${status}
