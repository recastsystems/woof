#!/usr/bin/env bash
# Build and run the Goddard 4-ice (mp_physics = 7) column oracle against
# byte-unmodified WRF v4.6.1.
#
#   build.sh WRF_SOURCE_ROOT WRF_BUILD_TREE BUILD_DIR
#
# WRF_SOURCE_ROOT  pristine WRF v4.6.1 checkout at d66e442 (never built in)
# WRF_BUILD_TREE   a compiled WRF-4.6.1 tree (configure 32, serial) whose
#                  phys/module_mp_gsfcgce_4ice_nuwrf.o and module_mp_radar.o
#                  were built by WRF's own make
#
# Two reference executables are linked from the SAME runner:
#
#   run_wrf  the objects WRF's own build produced, linked unchanged -- the
#            arithmetic a real wrf.exe runs.  THIS IS THE REFEREE: the
#            expected outputs are its words.
#   run_O0   the two WRF modules compiled here from the pristine source at
#            -O0.  Diagnostic only, and it does NOT agree with run_wrf:
#            at -O0 gfortran lowers an integer power (zs**5, zr**3, ...) to
#            libgcc's __powisf2, a square-and-multiply loop, while WRF's -O2
#            build expands the same power into the powi_table multiply chain
#            (x**5 = (x*x*x)*(x*x) rather than x*((x*x)*(x*x))).  The two
#            round differently.  -O0 also resolves MAX/MIN ties between +0
#            and -0 differently from -O2.  The module constants consat_s
#            builds are identical between the two (dumped and compared when
#            this lane was built), so the split is in the column arithmetic.
#            The disagreement count is written to o0-vs-wrf.txt.
#
# A second WRF build tree may be given (WRF_SCALAR_TREE, the same configure
# with -fno-tree-vectorize): its object must reproduce run_wrf word for word,
# so the referee does not depend on the vectorizer.
#
# The file set is two WRF files.  module_mp_gsfcgce_4ice_nuwrf.F USEs
# module_mp_radar (radar_init's xam_* inputs, set once on the first call) and
# nothing else outside WRF-Chem; module_mp_radar USEs module_wrf_error, which
# stub_wrf_services.F90 supplies as a service-only module.
set -euo pipefail

if [[ $# -ne 3 && $# -ne 4 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT WRF_BUILD_TREE BUILD_DIR"          "[WRF_SCALAR_TREE]" >&2
    exit 2
fi

source_root=$(realpath "$1")
wrf_build=$(realpath "$2")
build_dir=$(realpath -m "$3")
scalar_build=${4:+$(realpath "$4")}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
scheme_rel="phys/module_mp_gsfcgce_4ice_nuwrf.F"
radar_rel="phys/module_mp_radar.F"

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${scheme_rel}" "${radar_rel}"; then
    echo "a compiled WRF Goddard 4-ice dependency differs from pinned HEAD" >&2
    exit 3
fi
# The build tree's copy of the scheme must be the pinned bytes too: its
# object is what run_wrf links.
for rel in "${scheme_rel}" "${radar_rel}"; do
    if ! cmp -s "${source_root}/${rel}" "${wrf_build}/${rel}"; then
        echo "${wrf_build}/${rel} is not the pinned source" >&2
        exit 3
    fi
done

mkdir -p "${build_dir}/O0" "${build_dir}/wrf" "${build_dir}/cases"
fflags=(-O0 -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch)

# --- run_O0: pristine source at -O0 ----------------------------------------
cd "${build_dir}/O0"
gfortran -c "${fflags[@]}" "${script_dir}/stub_wrf_services.F90"
gfortran -c "${fflags[@]}" -I . "${source_root}/${radar_rel}"
gfortran -c "${fflags[@]}" -DWRF_CHEM=0 -I . \
    -o module_mp_gsfcgce_4ice_nuwrf.o "${source_root}/${scheme_rel}"
gfortran -c "${fflags[@]}" -I . "${script_dir}/run_gsfcgce4ice.F90"
gfortran -o ../run_O0 run_gsfcgce4ice.o module_mp_gsfcgce_4ice_nuwrf.o \
    module_mp_radar.o stub_wrf_services.o -lmvec -lm

# The -O0 scheme object must stay on scalar libm.
nm -u module_mp_gsfcgce_4ice_nuwrf.o | sed 's/^ *//' | sort > ../undefined-O0.txt
if grep -q '_ZGV' ../undefined-O0.txt; then
    echo "-O0 scheme object pulled in a libmvec vector symbol" >&2
    exit 4
fi

# --- run_wrf: WRF's own objects ----------------------------------------------
cd "${build_dir}/wrf"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -o stub_wrf_services.o "${script_dir}/stub_wrf_services.F90"
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
    -I "${wrf_build}/phys" "${script_dir}/run_gsfcgce4ice.F90"
gfortran -o ../run_wrf run_gsfcgce4ice.o \
    "${wrf_build}/phys/module_mp_gsfcgce_4ice_nuwrf.o" \
    "${wrf_build}/phys/module_mp_radar.o" stub_wrf_services.o -lmvec -lm
nm -u "${wrf_build}/phys/module_mp_gsfcgce_4ice_nuwrf.o" | sed 's/^ *//' \
    | sort > ../undefined-wrf.txt
if grep -q '_ZGV' ../undefined-wrf.txt; then
    echo "the WRF-built scheme object calls libmvec; the scalar oracle" \
         "would not be the artifact" >&2
    exit 4
fi

if [[ -n "${scalar_build}" ]]; then
    gfortran -o ../run_wrf_scalar run_gsfcgce4ice.o         "${scalar_build}/phys/module_mp_gsfcgce_4ice_nuwrf.o"         "${scalar_build}/phys/module_mp_radar.o" stub_wrf_services.o -lmvec -lm
fi

# --- cases --------------------------------------------------------------------
cd "${build_dir}"
export ORACLE_DIR="${script_dir}"
python3 "${script_dir}/gen_cases.py" cases
: > o0-vs-wrf.txt
for case in cases/case-*.bin; do
    name=$(basename "${case}" .bin)
    ./run_wrf "${case}" "cases/${name}.wrf.out" > "cases/${name}.wrf.log"
    ./run_O0 "${case}" "cases/${name}.O0.out" > "cases/${name}.O0.log"
    if [[ -n "${scalar_build}" ]]; then
        ./run_wrf_scalar "${case}" "cases/${name}.scalar.out"             > "cases/${name}.scalar.log"
        if ! cmp -s "cases/${name}.scalar.out" "cases/${name}.wrf.out"; then
            echo "${name}: the scalar WRF build disagrees with the strict" >&2
            exit 5
        fi
    fi
    python3 - "${case}" "cases/${name}.O0.out" "cases/${name}.wrf.out"         >> o0-vs-wrf.txt <<'PY'
import os
import sys
import numpy as np
sys.path.insert(0, os.environ["ORACLE_DIR"])
import layout
case = layout.read_input(sys.argv[1])
o0 = layout.read_output(sys.argv[2], case.ncol, case.nz, case.nstep)
wrf = layout.read_output(sys.argv[3], case.ncol, case.nz, case.nstep)
words = differ = signed_zero = 0
for a, b in zip(o0, wrf):
    for f in layout.GRADED_FIELDS:
        x, y = a[f].view(np.uint32), b[f].view(np.uint32)
        words += x.size
        bad = x != y
        differ += int(bad.sum())
        signed_zero += int((bad & (((x | y) & 0x7FFFFFFF) == 0)).sum())
print(f"{sys.argv[1]}: -O0 differs from WRF's object in {differ} of "
      f"{words} words ({signed_zero} of them +0 against -0)")
PY
    cp "cases/${name}.wrf.out" "cases/${name}.expected.out"
done
cat o0-vs-wrf.txt

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# WRF commit: ${pinned_commit}"
    echo "# WRF build flags:"
    grep -E '^(FCOPTIM|FCBASEOPTS_NO_G|FORMAT_FREE|FCCOMPAT)' \
        "${wrf_build}/configure.wrf" || true
    echo "# libm surface of the WRF-built scheme object:"
    grep -E 'expf|powf|logf|log10f|^log$|^sin$|_ZGV' undefined-wrf.txt || true
} > oracle-provenance.txt
sha256sum "${source_root}/${scheme_rel}" "${source_root}/${radar_rel}" \
    "${wrf_build}/phys/module_mp_gsfcgce_4ice_nuwrf.o" \
    "${wrf_build}/phys/module_mp_radar.o" \
    "${script_dir}/run_gsfcgce4ice.F90" "${script_dir}/stub_wrf_services.F90" \
    "${script_dir}/gen_cases.py" "${script_dir}/layout.py" \
    cases/*.bin cases/*.expected.out > oracle-sha256sums.txt
cat oracle-provenance.txt
echo "Goddard 4-ice oracle built in ${build_dir}"
