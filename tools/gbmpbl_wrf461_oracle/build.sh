#!/usr/bin/env bash
# Build and run the GBM PBL (bl_pbl_physics=12) column oracle.
#
#   bash build.sh WRF_SOURCE_ROOT WRF_SCALAR_BUILD BUILD_DIR
#
# WRF_SOURCE_ROOT is the pristine v4.6.1 tree; phys/module_bl_gbmpbl.F must
# hash to the pin below.  WRF_SCALAR_BUILD is a compiled WRF v4.6.1 tree
# whose configure.wrf carries FCOPTIM = -O2 -fno-fast-math -ffp-contract=off
# -fno-tree-vectorize (the sprint's scalar referee build).
#
# Two oracles, one driver (run_gbmpbl.F90):
#   * run_wrf links WRF's OWN compiled object phys/module_bl_gbmpbl.o from
#     the scalar build -- the words wrf.exe computes;
#   * run_o0 links a -O0 compile of the pristine source.
# The script fails unless both write byte-identical output, so the fixture
# is a property of the scheme's source, not of one optimisation level.
#
# The scheme USEs only module_model_constants (parameters, nothing linked)
# and calls only expf, logf and powf from libm (nm -u is the receipt): no
# stub is needed.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT WRF_SCALAR_BUILD BUILD_DIR" >&2
    exit 2
fi
src_root=$(realpath "$1")
wrf_build=$(realpath "$2")
build_dir=$(realpath -m "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
gbm_src="${src_root}/phys/module_bl_gbmpbl.F"
pinned="92d318297a54b3e89a2735da1bf6cfcef27b053c02339c3fb82dbbc4ca59a4cf"

actual=$(sha256sum "${gbm_src}" | awk '{print $1}')
if [[ "${actual}" != "${pinned}" ]]; then
    echo "module_bl_gbmpbl.F is ${actual}, not the pinned v4.6.1 ${pinned}" >&2
    exit 3
fi
grep -q -- '-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize' \
    "${wrf_build}/configure.wrf" || {
    echo "${wrf_build} is not the scalar referee build" >&2; exit 4; }

mkdir -p "${build_dir}/o0" "${build_dir}/wrf"
cd "${build_dir}"

# --- -O0 compile of the pristine source ------------------------------------
gfortran -c -O0 -cpp -DEM_CORE=1 -ffree-form -ffree-line-length-none -w \
    -J o0 -o o0/module_model_constants.o \
    "${src_root}/share/module_model_constants.F"
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none -w \
    -I o0 -J o0 -o o0/module_bl_gbmpbl.o "${gbm_src}"
gfortran -c -O0 -ffree-form -ffree-line-length-none -I o0 -J o0 \
    -o o0/run_gbmpbl.o "${here}/run_gbmpbl.F90"
gfortran -o run_o0 o0/run_gbmpbl.o o0/module_bl_gbmpbl.o \
    o0/module_model_constants.o

# --- WRF's own object --------------------------------------------------------
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${wrf_build}/phys" -J wrf -o wrf/run_gbmpbl.o "${here}/run_gbmpbl.F90"
gfortran -o run_wrf wrf/run_gbmpbl.o "${wrf_build}/phys/module_bl_gbmpbl.o"

nm -u "${wrf_build}/phys/module_bl_gbmpbl.o" | sed 's/^ *//' | sort \
    > undefined-wrf.txt
nm -u o0/module_bl_gbmpbl.o | sed 's/^ *//' | sort > undefined-o0.txt
if grep -q '_ZGV' undefined-wrf.txt undefined-o0.txt; then
    echo "a GBM object pulled in a libmvec vector symbol" >&2; exit 5
fi

# --- cases and runs ----------------------------------------------------------
python3 "${here}/make_cases.py" gbm-cases.bin --meta gbm-cases.json
./run_wrf gbm-cases.bin gbm-oracle-wrf.bin
./run_o0 gbm-cases.bin gbm-oracle-o0.bin
if ! cmp -s gbm-oracle-wrf.bin gbm-oracle-o0.bin; then
    echo "WRF's scalar object and the -O0 compile disagree" >&2
    cmp gbm-oracle-wrf.bin gbm-oracle-o0.bin | head -3 >&2 || true
    exit 6
fi

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# WRF object: $(sha256sum "${wrf_build}/phys/module_bl_gbmpbl.o")"
    echo "# libm symbols, WRF object:"; grep -E 'expf|logf|powf|sqrtf|_ZGV' undefined-wrf.txt || true
    echo "# libm symbols, -O0 object:"; grep -E 'expf|logf|powf|sqrtf|_ZGV' undefined-o0.txt || true
} > oracle-report.txt
sha256sum "${gbm_src}" "${here}/run_gbmpbl.F90" "${here}/make_cases.py" \
    gbm-cases.bin gbm-oracle-wrf.bin gbm-oracle-o0.bin > oracle-sha256sums.txt
cat oracle-report.txt
echo "WRF object == -O0 compile: byte-identical ($(stat -c %s gbm-oracle-wrf.bin) bytes)"

# roots is private and unused by GBM2D. Extract that exact source body into
# a standalone unit to exercise its branches without changing the WRF tree.
python3 - "${gbm_src}" <<'PYROOT'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
lo=s.index('  subroutine roots(')
hi=s.index('  end subroutine roots',lo)+len('  end subroutine roots')
Path('roots.F90').write_text(s[lo:hi]+'\n')
PYROOT
gfortran -O0 -ffp-contract=off -ffree-line-length-none -o run_roots \
    roots.F90 "${here}/run_roots.F90"
./run_roots roots-oracle.bin
sha256sum roots.F90 roots-oracle.bin > roots-sha256sums.txt
