#!/usr/bin/env bash
# Build and run the Pleim-Xiu LSM oracle against the byte-unmodified WRF
# v4.6.1/v4.7.1 phys/module_sf_pxlsm.F and module_sf_pxlsm_data.F.
#
# usage: build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# TWO OBJECT SETS, ONE REFERENCE.  The reference is WRF AS BUILT: the flags
# of the scalar wrf.exe the combo-sweep recordings were made with
# (-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize, configure.wrf
# FCOPTIM of /work/wrf/build/scalar).  An -O0 build of the same sources is
# made beside it purely as evidence, because the two are NOT the same
# program on this toolchain: gfortran lowers X**5 to __builtin_powif, which
# -O0 emits as a call to libgcc's __powisf2 (x*((x*x)*(x*x))) and -O2
# expands inline as an addition chain (x*x)*(x*(x*x)).  SURFPX's dry-layer
# thickness (1.-WG/WSAT)**5 (:1272) is the one such power in the scheme, so
# the two builds can differ there in the last bit.  run_pxlsm is run on
# every case with both and the word-level difference is written down
# (o0-vs-asbuilt.txt); the fixtures are the as-built words.
#
# libmvec.  -fno-tree-vectorize is WRF's own scalar setting, so no _ZGV*
# vector libm symbol may appear in the reference objects; the script fails
# if one does, and a positive control proves the grep can see one.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${build_dir}"
python3 "${script_dir}/verify_sources.py" "${source_root}" \
    > "${build_dir}/source-provenance.json"
cd "${build_dir}"

defines="-DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4"
defines="${defines} -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"
asbuilt="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize"

build_flavor() {
    local flavor=$1 opt=$2
    mkdir -p "${flavor}"
    (
        cd "${flavor}"
        gfortran -c ${opt} -cpp ${free} "${script_dir}/stub_wrf.F90"
        gfortran -c ${opt} -cpp ${defines} ${free} \
            "${source_root}/share/module_model_constants.F"
        gfortran -c ${opt} -cpp ${defines} ${free} -fallow-argument-mismatch \
            "${source_root}/frame/module_wrf_error.F"
        gfortran -c ${opt} -cpp ${defines} ${free} -I . \
            "${source_root}/phys/module_sf_pxlsm_data.F"
        gfortran -c ${opt} -cpp ${defines} ${free} -I . \
            "${source_root}/phys/module_sf_pxlsm.F"
        gfortran -c -O0 ${free} -I . "${script_dir}/run_pxlsm.F90"
        gfortran -o run_pxlsm stub_wrf.o module_model_constants.o \
            module_wrf_error.o module_sf_pxlsm_data.o module_sf_pxlsm.o \
            run_pxlsm.o
        gfortran -c -O0 ${free} -I . "${script_dir}/dump_tables.F90"
        gfortran -o dump_tables module_sf_pxlsm_data.o dump_tables.o
        nm -u module_sf_pxlsm.o | sed 's/^ *//' | sort > undefined-module_sf_pxlsm.txt
        if grep -q '_ZGV' undefined-module_sf_pxlsm.txt; then
            echo "${flavor} object pulled in a libmvec vector symbol" >&2
            exit 4
        fi
    )
}

build_flavor asbuilt "${asbuilt}"
build_flavor o0 "-O0"

# Positive control: the grep must be able to see a _ZGV symbol at all.
cat > control.f90 <<'EOF'
subroutine control(x, n)
  integer :: n, i
  real :: x(n)
  do i = 1, n
    x(i) = exp(x(i))
  end do
end subroutine control
EOF
gfortran -c -Ofast -ftree-vectorize -o control.o control.f90
nm -u control.o | sed 's/^ *//' | sort > undefined-control.txt
grep -q '_ZGV' undefined-control.txt || {
    echo "libmvec positive control emitted no _ZGV symbol" >&2; exit 5; }

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# as-built (${asbuilt}) module_sf_pxlsm.o libm/libgcc symbols:"
    grep -E 'expf|powf|logf|sqrtf|cosf|__powisf2|_ZGV' asbuilt/undefined-module_sf_pxlsm.txt || true
    echo "# -O0 module_sf_pxlsm.o libm/libgcc symbols:"
    grep -E 'expf|powf|logf|sqrtf|cosf|__powisf2|_ZGV' o0/undefined-module_sf_pxlsm.txt || true
    echo "# positive control:"
    grep -E '_ZGV' undefined-control.txt || true
} > libm-report.txt
cat libm-report.txt

# --- cases ------------------------------------------------------------------
python3 "${script_dir}/gen_cases.py" cases
for flavor in asbuilt o0; do
    mkdir -p "out-${flavor}"
    for case_file in cases/case-*.bin; do
        name=$(basename "${case_file}" .bin)
        name=${name#case-}
        "./${flavor}/run_pxlsm" "${case_file}" "out-${flavor}/out-${name}.bin" \
            > "out-${flavor}/stdout-${name}.txt"
    done
done
(cd asbuilt && ./dump_tables && mv pxlsm-tables.txt ..)
# Every constant the kernel reads, as gfortran makes it (constants.py).
python3 "${script_dir}/constants.py" --fortran > asbuilt/literals.F90
(cd asbuilt && gfortran ${asbuilt} ${free} -I . -o literals literals.F90     module_model_constants.o && ./literals && mv literals.txt ..)
python3 "${script_dir}/constants.py" --cuda literals.txt > pxk_table.cuh

# Word-level difference between the -O0 evidence build and the reference.
python3 - <<'EOF' > o0-vs-asbuilt.txt
import glob, os
import numpy as np
total = diff = 0
for path in sorted(glob.glob("out-asbuilt/out-*.bin")):
    other = path.replace("out-asbuilt", "out-o0")
    a = np.fromfile(path, dtype="<u4")
    b = np.fromfile(other, dtype="<u4")
    n = int((a != b).sum())
    total += a.size
    diff += n
    print(f"{os.path.basename(path)} words={a.size} differing={n}")
print(f"TOTAL words={total} differing={diff}")
EOF
cat o0-vs-asbuilt.txt

sha256sum "${source_root}/share/module_model_constants.F" \
    "${source_root}/frame/module_wrf_error.F" \
    "${source_root}/phys/module_sf_pxlsm_data.F" \
    "${source_root}/phys/module_sf_pxlsm.F" \
    "${script_dir}"/*.F90 "${script_dir}"/*.py "${script_dir}/build.sh" \
    cases/*.bin out-asbuilt/*.bin pxlsm-tables.txt literals.txt > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built in ${build_dir}"
