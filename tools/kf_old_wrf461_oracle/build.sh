#!/usr/bin/env bash
# Build and run the WRF v4.6.1 old Kain-Fritsch (cu_physics = 99) column
# oracle against the byte-unmodified public source.
#
# usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_PRODUCTION_OBJECT]
#
# THE FILE SET IS TWO WRF FILES.  phys/module_cu_kf.F USEs module_wrf_error
# and nothing else; KFCPS takes every physical constant as an argument.  The
# constants the driver passes come from share/module_model_constants.F,
# which has no USE at all, so it is compiled from the pinned tree rather
# than restated: a restated EP_2 = R_d/R_v is exactly the kind of constant
# a port and its oracle can quietly share the same wrong value of.
# module_wrf_error is stood in for by stub_wrf.F90 (logging and abort
# only; frame/module_wrf_error.F needs ESMF).
#
# THREE ARMS, one harness:
#   A  module_cu_kf.F at -O0, the reference the parity test grades against.
#   N  the same source at -O0 with every uninitialised local poisoned
#      (-finit-real=snan -finit-integer=-77777 -finit-logical=true).  The
#      scheme reads several locals before it can be sure they were set on
#      every path (AINCM1, ISTOP, EE1/UD1, RL); arm N agreeing with arm A
#      word for word on every case is the receipt that no output of these
#      cases depends on stack contents, which is what lets a port that
#      zero-initialises its workspace be graded at 0 ULP.
#   B  (optional) WRF's OWN production object for module_cu_kf.F from a
#      configure/compile build of the same tree (-O2 -fno-fast-math
#      -ffp-contract=off on this box), linked into the same harness.  Arm B
#      agreeing with arm A is the receipt that the -O0 reference is the
#      model WRF actually runs, not a debug build of it.
#
# libmvec: the -O0 objects must carry no _ZGV* vector-math symbol (same
# guard as the YSU, GF and New Tiedtke oracles); the production object's
# undefined symbols are recorded, and it calls scalar expf/logf/powf.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_PRODUCTION_OBJECT]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
production_obj=${3:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

declare -A pinned=(
  ["phys/module_cu_kf.F"]="8042c90f7b7f5b14b36acd809f4f310b8fc3f75101039169e602e7df3b5d2f61"
  ["share/module_model_constants.F"]="5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
)
for rel in "${!pinned[@]}"; do
    actual=$(sha256sum "${source_root}/${rel}" | awk '{print $1}')
    if [[ "${actual}" != "${pinned[$rel]}" ]]; then
        echo "${rel} is ${actual}, not the pinned ${pinned[$rel]}" >&2
        exit 3
    fi
done

mkdir -p "${build_dir}"
cd "${build_dir}"

FFLAGS=(-O0 -cpp -ffree-form -ffree-line-length-none)
POISON=(-finit-real=snan -finit-integer=-77777 -finit-logical=true)

gfortran -c "${FFLAGS[@]}" -o module_model_constants.o \
    "${source_root}/share/module_model_constants.F"
gfortran -c -O0 -ffree-line-length-none -o stub_wrf.o \
    "${script_dir}/stub_wrf.F90"
gfortran -c "${FFLAGS[@]}" -o module_cu_kf_A.o \
    "${source_root}/phys/module_cu_kf.F"
mkdir -p poison
gfortran -c "${FFLAGS[@]}" "${POISON[@]}" -J poison \
    -o module_cu_kf_N.o "${source_root}/phys/module_cu_kf.F"
gfortran -c -O0 -ffree-line-length-none -o run_kf_old.o \
    "${script_dir}/run_kf_old.F90"

gfortran -o run_kf_old_A run_kf_old.o module_cu_kf_A.o \
    module_model_constants.o stub_wrf.o
gfortran -o run_kf_old_N run_kf_old.o module_cu_kf_N.o \
    module_model_constants.o stub_wrf.o
arms=(A N)
if [[ -n "${production_obj}" ]]; then
    cp "${production_obj}" module_cu_kf_B.o
    gfortran -o run_kf_old_B run_kf_old.o module_cu_kf_B.o \
        module_model_constants.o stub_wrf.o
    arms+=(B)
fi

for obj in module_cu_kf_A.o module_cu_kf_N.o module_model_constants.o; do
    if nm -u "${obj}" | grep -q '_ZGV'; then
        echo "FATAL: ${obj} carries a libmvec vector-math symbol" >&2
        exit 5
    fi
done
{
  for obj in module_cu_kf_*.o; do
      echo "## ${obj}"; nm -u "${obj}"
  done
} > kf-old-undefined-symbols.txt
gfortran --version | sed -n 1p > compiler.txt
ldd --version | sed -n 1p >> compiler.txt

python3 "${script_dir}/gen_cases.py" cases.bin \
    --real74 "${script_dir}/../../tests/data/kf_real74_12z_columns.npz"
ncase=$(python3 -c "import struct;print(struct.unpack('<3i',open('cases.bin','rb').read(12))[2])")

rm -rf out arms.txt && mkdir -p out
run_one() {  # arm index
    local arm=$1 i=$2 log
    log=$(./run_kf_old_${arm} cases.bin "${i}" "out/${arm}-${i}.bin" 2>&1) \
        && echo "${arm} ${i} 0" \
        || echo "${arm} ${i} $? $(echo "${log}" | grep -m1 'WRF-FATAL' || true)"
}
export -f run_one
for arm in "${arms[@]}"; do
    seq 0 $((ncase - 1)) | xargs -P 32 -I{} bash -c "run_one ${arm} {}"
done | sort -k1,1 -k2,2n > outcomes.txt

# Every arm must agree with A, word for word, including on which columns
# reach a WRF fatal.
status=0
for arm in "${arms[@]}"; do
    [[ "${arm}" == A ]] && continue
    diff <(awk '$1=="A"{ $1=""; print }' outcomes.txt) \
         <(awk -v a="${arm}" '$1==a{ $1=""; print }' outcomes.txt) \
        > "outcomes-A-vs-${arm}.diff" || { echo "arm ${arm} outcomes differ" >&2; status=6; }
    differing=0
    for ((i = 0; i < ncase; i++)); do
        if [[ -f "out/A-${i}.bin" ]] && ! cmp -s "out/A-${i}.bin" "out/${arm}-${i}.bin"; then
            differing=$((differing + 1))
        fi
    done
    echo "arm ${arm} vs A: ${differing} of ${ncase} case outputs differ" | tee -a arms.txt
    [[ ${differing} -eq 0 ]] || status=6
done

python3 "${script_dir}/collect.py" cases.bin outcomes.txt out \
    kf-old-wrf461-oracle.npz
sha256sum kf-old-wrf461-oracle.npz cases.bin compiler.txt \
    kf-old-undefined-symbols.txt > oracle-sha256sums.txt
cat arms.txt 2>/dev/null || true
exit ${status}
