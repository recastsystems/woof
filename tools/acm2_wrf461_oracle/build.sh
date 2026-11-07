#!/usr/bin/env bash
# Build and run the ACM2 (Pleim) PBL oracle against the byte-unmodified WRF
# v4.6.1 phys/module_bl_acm.F.
#
#   bash tools/acm2_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR [REFEREE_BUILD]
#
# The module goes through WRF's OWN preprocessing (the sed that strips
# quoted comment lines, cpp -P -nostdinc -traditional-cpp with WRF's define
# set, tools/standard.c, cpp again) and WRF's OWN compile line for phys/
# (configure.wrf FCOPTIM of the serial gfortran build:
# -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize plus
# FCBASEOPTS).  That is the build the combo-sweep referee wrf.exe links, so
# the fixture is the referee's arithmetic, not a debug build's.
#
# When REFEREE_BUILD (a configured, compiled WRF-4.6.1 tree) is given the
# script also proves it: the preprocessed .f90 must be byte-identical to the
# referee's phys/module_bl_acm.f90, and a second driver linked against the
# referee's own phys/module_bl_acm.o must write byte-identical fixtures.
# An -O0 build of the same .f90 is run as evidence (O0-report.txt): it is
# NOT WRF's program, because -O0 keeps one power as a powf call.
#
# The pin is the FILE: module_bl_acm.F must hash to the value below, the one
# in the pristine v4.6.1 tree (d66e442), so the fixture cannot come from an
# edited file.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [REFEREE_BUILD]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
referee=${3:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
acm_source="${source_root}/phys/module_bl_acm.F"
pinned_sha256="f18685f7ea091f567c3f37e983f26357dcd7e9ec2eea9a6e70635c506a66d60c"

actual_sha256=$(sha256sum "${acm_source}" | awk '{print $1}')
if [[ "${actual_sha256}" != "${pinned_sha256}" ]]; then
    echo "module_bl_acm.F is ${actual_sha256}, not the pinned" \
         "${pinned_sha256}" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

# --- WRF's preprocessing pipeline (phys/Makefile .F.o rule) ------------------
gcc -O2 -o standard.exe "${source_root}/tools/standard.c"
WRF_DEFINES="-DEM_CORE=1 -DNMM_CORE=0 -DNMM_MAX_DIM=2600 -DDA_CORE=0 -DWRFPLUS=0 \
-DIWORDSIZE=4 -DDWORDSIZE=8 -DRWORDSIZE=4 -DLWORDSIZE=4 -DNONSTANDARD_SYSTEM_SUBR \
-DWRF_USE_CLM -DUSE_NETCDF4_FEATURES -DWRFIO_NCD_LARGE_FILE_SUPPORT -DNETCDF \
-DLANDREAD_STUB=1 -DUSE_ALLOCATABLES -Dwrfmodel -DGRIB1 -DINTIO -DKEEP_INT_AROUND \
-DLIMIT_ARGS -DBUILD_RRTMG_FAST=0 -DBUILD_RRTMK=0 -DBUILD_SBM_FAST=1 \
-DSHOW_ALL_VARS_USED=0 -DCONFIG_BUF_LEN=65536 -DMAX_DOMAINS_F=21 -DMAX_HISTORY=25 \
-DNMM_NEST=0"
sed -e "s/^\!.*'.*//" -e "s/^ *\!.*'.*//" "${acm_source}" > module_bl_acm.G
# shellcheck disable=SC2086
/lib/cpp -P -nostdinc -I"${source_root}/inc" ${WRF_DEFINES} -I. -traditional-cpp \
    -DUSE_NETCDF4_FEATURES -DWRFIO_NCD_LARGE_FILE_SUPPORT module_bl_acm.G > module_bl_acm.bb
./standard.exe module_bl_acm.bb | /lib/cpp -P -nostdinc -traditional-cpp \
    -DUSE_NETCDF4_FEATURES -DWRFIO_NCD_LARGE_FILE_SUPPORT > module_bl_acm.f90
rm -f module_bl_acm.G module_bl_acm.bb

# --- WRF's compile line for phys/ (serial gfortran, scalar referee) ----------
FCOPTIM="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize"
FCBASE="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 \
-fallow-argument-mismatch -fallow-invalid-boz"
# shellcheck disable=SC2086
gfortran -o module_bl_acm.o -c ${FCOPTIM} ${FCBASE} module_bl_acm.f90
mkdir -p O0
: > O0-report.txt
# shellcheck disable=SC2086
gfortran -o O0/module_bl_acm.o -J O0 -c -O0 ${FCBASE} module_bl_acm.f90

gfortran -c -O0 -ffree-form -ffree-line-length-none "${script_dir}/stub_wrf.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none -I . "${script_dir}/run_acm2.F90"
gfortran -o run_acm2 run_acm2.o module_bl_acm.o stub_wrf.o
gfortran -o run_acm2_O0 run_acm2.o O0/module_bl_acm.o stub_wrf.o

# --- receipts ----------------------------------------------------------------
nm -u module_bl_acm.o | sed 's/^ *//' | sort > undefined.txt
if grep -q '_ZGV' undefined.txt; then
    echo "module_bl_acm.o pulled in a libmvec vector symbol" >&2
    exit 4
fi
# Real-exponent powers are the libm surface: count the call sites.  The
# source has six powers: (PBL/(KARMAN*|MOL|))**0.333333 and
# (UST**3+0.6*WST**3)**0.33333 in ACM2D, the two (1-GAMM*ZOVL)**(-0.25) arms
# of EDDYX, and (KARMAN/HOVL)**0.3333 in each of ACM and ACMM.  gfortran
# merges EDDYX's two arms into ONE call on the selected argument, so the
# object carries five.  (1-ZF/PBL)**P with the PARAMETER P = 2.0 must NOT be
# a call: gfortran folds it to one multiply, which the kernel transcribes.
objdump -dr module_bl_acm.o > module_bl_acm.objdump.txt
: > libm-report.txt
pow_calls=$(grep -cE 'R_X86_64_PLT32[[:space:]]+powf' module_bl_acm.objdump.txt || true)
awk '/^[0-9a-f]+ <.*>:/{fn=$2} /R_X86_64_PLT32[[:space:]]+powf/{print "powf call in " fn}'     module_bl_acm.objdump.txt >> libm-report.txt
echo "powf call sites: ${pow_calls}" >> libm-report.txt
grep -E 'powf|expf|logf|sqrtf' undefined.txt >> libm-report.txt || true
if [[ "${pow_calls}" != "5" ]]; then
    echo "expected 5 powf call sites in module_bl_acm.o, found ${pow_calls}" >&2
    exit 5
fi
if grep -qE 'R_X86_64_PLT32[[:space:]]+sqrtf' module_bl_acm.objdump.txt; then
    echo "SQRT lowered to a libm call rather than sqrtss" >> libm-report.txt
fi

# --- fixtures ----------------------------------------------------------------
python3 "${script_dir}/make_cases.py" "${build_dir}"
for name in acm2-l40 acm2-l60; do
    ./run_acm2 "${name}.in.bin" "${name}.out.bin"
    ./run_acm2_O0 "${name}.in.bin" "${name}.O0.out.bin"
    # Evidence only, never the fixture: at -O0 gfortran does NOT fold
    # (1-ZF/PBL)**P (P = 2.0) to a multiply but calls powf(x, 2.0), which is
    # not always the correctly rounded x*x, so the -O0 build is a different
    # program from WRF's.  The differing bytes are counted, not refused.
    echo "${name}: $(cmp -l "${name}.out.bin" "${name}.O0.out.bin" | wc -l) bytes differ at -O0" >> O0-report.txt
done
# WRF's fatal: the oracle must stop with status 3 naming column 2.
set +e
./run_acm2 fatal.in.bin fatal.out.bin 2> fatal.stderr.txt
fatal_rc=$?
set -e
echo "fatal fixture exit status: ${fatal_rc}" >> fatal.stderr.txt
if [[ "${fatal_rc}" != "3" ]] || ! grep -q 'RIBX never exceeds RIC' fatal.stderr.txt; then
    echo "the fatal fixture did not reach WRF's RIB fatal" >&2
    cat fatal.stderr.txt >&2
    exit 6
fi

# --- referee cross-check -----------------------------------------------------
if [[ -n "${referee}" ]]; then
    cmp module_bl_acm.f90 "${referee}/phys/module_bl_acm.f90"
    gfortran -o run_acm2_referee run_acm2.o "${referee}/phys/module_bl_acm.o" stub_wrf.o
    for name in acm2-l40 acm2-l60; do
        ./run_acm2_referee "${name}.in.bin" "${name}.referee.out.bin"
        cmp "${name}.out.bin" "${name}.referee.out.bin"
    done
    echo "referee: ${referee} (.f90 and fixtures byte-identical)" > referee.txt
    sha256sum "${referee}/phys/module_bl_acm.o" >> referee.txt
fi

sha256sum "${acm_source}" "${script_dir}/run_acm2.F90" \
    "${script_dir}/stub_wrf.F90" "${script_dir}/make_cases.py" \
    module_bl_acm.f90 \
    acm2-l40.in.bin acm2-l40.out.bin acm2-l60.in.bin acm2-l60.out.bin \
    fatal.in.bin > oracle-sha256sums.txt
{
    gfortran --version | head -1
    ldd --version | head -1
    echo "FCOPTIM ${FCOPTIM}"
} > toolchain.txt
echo "oracle built in ${build_dir}"
