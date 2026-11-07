#!/usr/bin/env bash
# Build and run the QNSE surface-layer column oracle against the
# byte-unmodified WRF v4.6.1 phys/module_sf_qnsesfc.F (sf_sfclay_physics=4).
#
#   bash tools/qnsesfc_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR [OUT.npz]
#
# WRF_SOURCE_ROOT must be a wrf-model/WRF checkout at the v4.6.1 commit with
# share/module_model_constants.F and phys/module_sf_qnsesfc.F unmodified; the
# script refuses otherwise.  module_sf_qnsesfc.F USEs MODULE_MODEL_CONSTANTS
# and MODULE_DM (WRF_DM_MAXVAL, called only in NMM builds); the second is
# satisfied by module_dm_stub.F90, which stops the run if it is ever reached.
#
# Reference build: WRF's own optimised phys/ flags as this box's WRF builds
# use them (-O2 -fno-fast-math -ffp-contract=off); stock configure's
# (-O2 -ftree-vectorize -funroll-loops) must write the same bytes
# (o2-equality.txt), and no object may call libmvec.  The reference is NOT
# -O0 here, unlike the MYJ oracle: at -O0 gfortran sends the integer powers
# (ZETA**2, ZETA**3, (ZETA-0.5)**5, USTAR**3) to libgcc's __powisf2, whose
# square-and-multiply order rounds (ZETA-0.5)**5 differently from the
# multiply chain every optimised build (and so every WRF executable) emits.
# The -O0 build is still run and its difference recorded (o0-difference.txt):
# 79 PSIH words per table, at most 2 ULP.
# Output: BUILD_DIR/out_<set>.bin and OUT.npz (default
# BUILD_DIR/qnsesfc-wrf461.npz), the fixture
# tests/test_qnsesfc_wrf461_parity.py and compare_qnsesfc.py read;
# coverage-qnsesfc.txt lists the executable lines of module_sf_qnsesfc.F the
# fixture never reached.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [OUT.npz]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out_npz=$(realpath -m "${3:-${build_dir}/qnsesfc-wrf461.npz}")
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
const_rel="share/module_model_constants.F"
sfc_rel="phys/module_sf_qnsesfc.F"

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${const_rel}" "${sfc_rel}"; then
    echo "${const_rel} or ${sfc_rel} differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

defines="-DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"
boxflags="-O2 -fno-fast-math -ffp-contract=off"
stockflags="-O2 -ftree-vectorize -funroll-loops"

# --- the framework pieces and the driver, and the -O0 comparison object ------
gfortran -c -O0 -cpp ${defines} ${free} -o module_model_constants.o \
    "${source_root}/${const_rel}"
gfortran -c -O0 ${free} -o module_dm_stub.o "${script_dir}/module_dm_stub.F90"
gfortran -c -O0 -cpp ${defines} ${free} -I "${build_dir}" \
    -o module_sf_qnsesfc_O0.o "${source_root}/${sfc_rel}"
gfortran -c -O0 ${free} -I "${build_dir}" -o run_qnsesfc.o \
    "${script_dir}/run_qnsesfc.F90"
gfortran -o run_qnsesfc_O0 module_model_constants.o module_dm_stub.o \
    module_sf_qnsesfc_O0.o run_qnsesfc.o

# --- THE REFERENCE: this box's WRF phys/ flags, and stock configure's -------
gfortran -c ${boxflags} -cpp ${defines} ${free} -I "${build_dir}" \
    -o module_sf_qnsesfc_O2box.o "${source_root}/${sfc_rel}"
gfortran -o run_qnsesfc module_model_constants.o module_dm_stub.o \
    module_sf_qnsesfc_O2box.o run_qnsesfc.o
gfortran -c ${stockflags} -cpp ${defines} ${free} -I "${build_dir}" \
    -o module_sf_qnsesfc_O2vec.o "${source_root}/${sfc_rel}"
gfortran -o run_qnsesfc_O2vec module_model_constants.o module_dm_stub.o \
    module_sf_qnsesfc_O2vec.o run_qnsesfc.o

# Exercise QNSESFCINIT's cold-start arm as well as the fixture's restart arm.
gfortran -c ${boxflags} ${free} -I "${build_dir}" -o init_state_probe.o \
    "${script_dir}/init_state_probe.F90"
gfortran -o init_state_probe module_model_constants.o module_dm_stub.o \
    module_sf_qnsesfc_O2box.o init_state_probe.o
./init_state_probe | tee init-state.txt

# --- receipts: what the objects call ----------------------------------------
for tag in O0 O2box O2vec; do
    nm -u "module_sf_qnsesfc_${tag}.o" | sed 's/^ *//' | sort > "undefined-${tag}.txt"
    if grep -q '_ZGV' "undefined-${tag}.txt"; then
        echo "undefined-${tag}.txt: a libmvec vector symbol is called" >&2
        grep '_ZGV' "undefined-${tag}.txt" >&2
        exit 4
    fi
done
objdump -d module_sf_qnsesfc_O0.o module_sf_qnsesfc_O2box.o module_sf_qnsesfc_O2vec.o \
    | grep -cE 'vfmadd|vfmsub|vfnmadd|vfnmsub' > fma-count.txt || true
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# cpu:      $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
    echo "# cpu fma:  $(grep -o -m1 -w fma /proc/cpuinfo || echo absent)"
    echo "# WRF:      ${head_commit} (clean: ${sfc_rel}, ${const_rel})"
    echo "# ${boxflags} undefined symbols (THE REFERENCE):"
    sed 's/^/#   /' undefined-O2box.txt
    echo "# -O0 undefined symbols (integer powers through __powisf2):"
    sed 's/^/#   /' undefined-O0.txt
    echo "# ${stockflags} undefined symbols:"
    sed 's/^/#   /' undefined-O2vec.txt
    echo "# FMA instructions in any object: $(cat fma-count.txt)"
} > compiler.txt
cat compiler.txt

# --- inputs and runs ----------------------------------------------------------
python3 "${script_dir}/make_cases.py" "${build_dir}"
for set in $(cat cases.txt); do
    ./run_qnsesfc "case_${set}.bin" "out_${set}.bin"
    ./run_qnsesfc_O2vec "case_${set}.bin" "O2vec_${set}.bin"
    if ! cmp -s "out_${set}.bin" "O2vec_${set}.bin"; then
        echo "stock ${stockflags} differs from the ${boxflags} reference on ${set}" >&2
        exit 6
    fi
    ./run_qnsesfc_O0 "case_${set}.bin" "O0_${set}.bin"
done
echo "receipt: ${stockflags} outputs byte-identical to the ${boxflags} reference on every set"     | tee o2-equality.txt
python3 "${script_dir}/pack_fixture.py" --o0-difference "${build_dir}"     | tee o0-difference.txt
python3 "${script_dir}/pack_fixture.py" "${build_dir}" "${out_npz}"
sha256sum case_*.bin out_*.bin "${out_npz}" > oracle-sha256sums.txt

# --- receipt: which lines of module_sf_qnsesfc.F the fixture executes -------
mkdir -p coverage
( cd coverage
  gfortran -c ${boxflags} --coverage -cpp ${defines} ${free} -I "${build_dir}" \
      -o module_sf_qnsesfc.o "${source_root}/${sfc_rel}"
  gfortran --coverage -o run_qnsesfc_cov ../module_model_constants.o \
      ../module_dm_stub.o module_sf_qnsesfc.o ../run_qnsesfc.o
  for set in $(cat ../cases.txt); do
      ./run_qnsesfc_cov "../case_${set}.bin" "cov_${set}.bin"
      cmp -s "../out_${set}.bin" "cov_${set}.bin" \
          || { echo "coverage build changed the words on ${set}" >&2; exit 7; }
  done
  gcov -o . module_sf_qnsesfc.o > gcov-summary.txt 2>&1 || true
)
python3 - "${build_dir}/coverage" > coverage-qnsesfc.txt <<'PY'
import pathlib, sys
d = pathlib.Path(sys.argv[1])
g = next(d.glob("module_sf_qnsesfc.F.gcov"))
never, ran = [], 0
for line in g.read_text().splitlines():
    count, lineno, text = (part.strip() for part in line.split(":", 2))
    if count == "-" or lineno == "0":
        continue
    if count.startswith("#####"):
        never.append(f"{lineno}: {text}")
    else:
        ran += 1
print(f"executable lines reached: {ran}; never reached: {len(never)}")
for row in never:
    print("  never", row)
PY
cat coverage-qnsesfc.txt
