#!/usr/bin/env bash
# Build and run the Multiscale Kain-Fritsch (cu_physics = 11) column oracle
# against WRF v4.6.1.
#
# Two builds of the same pinned source:
#
#   pristine  phys/module_cu_mskf.F byte-unmodified.  Its object code is
#             compared instruction by instruction with WRF's own scalar
#             build when WRF_SCALAR_BUILD is set (objdump-match.txt).
#   ref       the same file plus wrf461-mskf-defined-reads.patch, two lines
#             that give the aerosol path's two uninitialised reads a defined
#             value.  This is the fixture the port is graded on.
#
# Why the fixture is not the pristine build: on the aercu_opt > 0 path the
# pristine module reads two uninitialised locals (QSATu below and above the
# updraft, and pgam(k) where pgam(k-1) is meant), so its words are a function
# of whatever the stack held, not of the inputs.  This script MEASURES that
# instead of asserting it: the pristine module is rebuilt with
# -finit-real=nan/zero/inf and the words that move are counted
# (uninit-pristine.txt); the ref module is rebuilt the same way and must not
# move a single word (uninit-ref.txt, enforced), and it must also be
# column-isolated (isolation.txt, enforced).  aercu_opt = 0 columns are
# word-identical between the two builds; that is enforced too.
#
# Arithmetic flags: WRF's scalar-referee FCOPTIM (-O2 -fno-fast-math
# -ffp-contract=off -fno-tree-vectorize) and its FCBASEOPTS, not -O0.
# gfortran lowers a constant integer power to a different multiplication
# chain at -O0 (libgcc's square-and-multiply) than at -O2 (gcc's powi
# table), and the aerosol path forms x**4, x**5 and x**6, so the oracle must
# be the arithmetic the WRF executable runs.  An -O0 build is kept as
# evidence (o0-vs-o2.txt); lowering-probe.txt records the chains.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
sources=(phys/module_cam_shr_kind_mod.F phys/module_cam_error_function.F
         phys/module_cu_mskf.F)
declare -A pinned=(
  [phys/module_cu_mskf.F]="4bca54b294def08ffc98ef69a30650c943e2c394f34802e75e50c7f1a2896fcb"
  [phys/module_cam_error_function.F]="49fa938132d5d49da279a387315bca8ff23c5de332b939a1065e95a063371bf0"
  [phys/module_cam_shr_kind_mod.F]="369bad7e4b528953dc4047ad38bf0b0d88120190f75541667be635c55edf1783"
)

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${sources[@]}"; then
    echo "a compiled WRF MSKF source differs from pinned HEAD" >&2
    exit 3
fi
for src in "${sources[@]}"; do
    actual=$(sha256sum "${source_root}/${src}" | awk '{print $1}')
    if [[ "${actual}" != "${pinned[$src]}" ]]; then
        echo "${src} is ${actual}, not the pinned ${pinned[$src]}" >&2
        exit 3
    fi
done

mkdir -p "${build_dir}"
cd "${build_dir}"
rm -rf variant-* src-* standard.exe
sha256sum "${sources[@]/#/${source_root}/}" > source-sha256.txt

# The patched copy: the pristine file plus the two defined-read lines.
mkdir -p src-ref/phys
cp "${source_root}/phys/module_cu_mskf.F" src-ref/phys/
(cd src-ref && patch -s -p1 < "${script_dir}/wrf461-mskf-defined-reads.patch")

# WRF's preprocessing (the scalar build's compile.log, phys/Makefile rule).
wrf_defines=(-DEM_CORE=1 -DNMM_CORE=0 -DNMM_MAX_DIM=2600 -DDA_CORE=0
             -DWRFPLUS=0 -DIWORDSIZE=4 -DDWORDSIZE=8 -DRWORDSIZE=4
             -DLWORDSIZE=4 -DNONSTANDARD_SYSTEM_SUBR -DWRF_USE_CLM
             -DUSE_NETCDF4_FEATURES -DWRFIO_NCD_LARGE_FILE_SUPPORT -DNETCDF
             -DLANDREAD_STUB=1 -DUSE_ALLOCATABLES -Dwrfmodel -DGRIB1 -DINTIO
             -DKEEP_INT_AROUND -DLIMIT_ARGS -DBUILD_RRTMG_FAST=0
             -DBUILD_RRTMK=0 -DBUILD_SBM_FAST=1 -DSHOW_ALL_VARS_USED=0
             -DCONFIG_BUF_LEN=65536 -DMAX_DOMAINS_F=21 -DMAX_HISTORY=25
             -DNMM_NEST=0)
# tools/standard.exe is WRF's own post-cpp filter, built from the pinned
# tools/standard.c exactly as tools/Makefile builds it.
gcc -DIWORDSIZE=4 -DMAX_HISTORY=25 -DNMM_CORE=0 -O0 -o standard.exe \
    "${source_root}/tools/standard.c"
preprocess() {   # $1 = source .F path, $2 = output .f90
    sed -e "s/^\!.*'.*//" -e "s/^ *\!.*'.*//" "$1" > "$2.G"
    /lib/cpp -P -nostdinc "${wrf_defines[@]}" -I. -traditional-cpp \
        -DUSE_NETCDF4_FEATURES -DWRFIO_NCD_LARGE_FILE_SUPPORT \
        "$2.G" > "$2.bb"
    ./standard.exe "$2.bb" | /lib/cpp -P -nostdinc -traditional-cpp \
        -DUSE_NETCDF4_FEATURES -DWRFIO_NCD_LARGE_FILE_SUPPORT > "$2"
    rm -f "$2.G" "$2.bb"
}
preprocess "${source_root}/phys/module_cam_shr_kind_mod.F" \
    module_cam_shr_kind_mod.f90
preprocess "${source_root}/phys/module_cam_error_function.F" \
    module_cam_error_function.f90
preprocess "${source_root}/phys/module_cu_mskf.F" module_cu_mskf.f90
preprocess src-ref/phys/module_cu_mskf.F module_cu_mskf_ref.f90

wrf_fflags=(-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w
            -ffree-form -ffree-line-length-none -fconvert=big-endian
            -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz)

build_variant() {   # $1 = name, $2 = mskf .f90, rest = extra mskf flags
    local name=$1 src=$2; shift 2
    local dir="variant-${name}"
    mkdir -p "${dir}"
    (
      cd "${dir}"
      gfortran -c "${wrf_fflags[@]}" ../module_cam_shr_kind_mod.f90
      gfortran -c "${wrf_fflags[@]}" ../module_cam_error_function.f90
      gfortran -c "${wrf_fflags[@]}" "${script_dir}/stub_wrf_error.F90"
      gfortran -c "${wrf_fflags[@]}" "$@" -o module_cu_mskf.o "../${src}"
      gfortran -c -O0 -ffree-form -ffree-line-length-none \
          "${script_dir}/run_mskf.F90"
      gfortran -o run_mskf module_cam_shr_kind_mod.o \
          module_cam_error_function.o stub_wrf_error.o module_cu_mskf.o \
          run_mskf.o
      ln -sf ../cases_i32.bin ../cases_f32.bin .
      ./run_mskf "${ncase}"
    )
}
words_differ() {    # $1 $2 = out_f32.bin paths; prints per-case counts
    python3 - "$1" "$2" "${ncase}" <<'PY'
import sys, numpy as np
n = int(sys.argv[3])
a = np.fromfile(sys.argv[1], "<u4").reshape(n, -1)
b = np.fromfile(sys.argv[2], "<u4").reshape(n, -1)
d = (a != b).sum(1)
print(int(d.sum()), " ".join(f"{i+1}:{int(x)}" for i, x in enumerate(d) if x))
PY
}

python3 "${script_dir}/make_cases.py" "${build_dir}" > cases.log
ncase=$(python3 -c 'import json;print(json.load(open("cases.json"))["ncase"])')
first_aercu=$(python3 -c '
import json
c = json.load(open("cases.json"))["cases"]
print(min(i + 1 for i, x in enumerate(c) if x["aercu_opt"] > 0))')

# --- the two builds -----------------------------------------------------------
build_variant pristine module_cu_mskf.f90
build_variant ref module_cu_mskf_ref.f90
cp variant-ref/out_f32.bin out_f32.bin
cp variant-ref/isolation.txt isolation.txt
cp variant-pristine/isolation.txt isolation-pristine.txt
if awk '$2 != 0 {bad=1} END {exit !bad}' isolation.txt; then
    echo "a reference column's output depends on its neighbours:" >&2
    awk '$2 != 0' isolation.txt >&2
    exit 5
fi
for v in pristine ref; do
    nm -u "variant-${v}/module_cu_mskf.o" | sed 's/^ *//' | sort \
        > "undefined-${v}.txt"
    if grep -q '_ZGV' "undefined-${v}.txt"; then
        echo "the ${v} object pulled in a libmvec vector symbol" >&2
        exit 4
    fi
done

# The patch may move aerosol-path words only.
read -r total detail < <(words_differ variant-pristine/out_f32.bin out_f32.bin)
echo "pristine vs ref: ${total} words; cases ${detail}" > pristine-vs-ref.txt
for item in ${detail}; do
    if (( ${item%%:*} < first_aercu )); then
        echo "the defined-reads patch moved an aercu_opt = 0 column:" \
             "${item}" >&2
        exit 6
    fi
done

# --- receipt: the pristine object code equals WRF's own scalar build ----------
if [[ -n "${WRF_SCALAR_BUILD:-}" ]]; then
    wrf_obj="${WRF_SCALAR_BUILD}/phys/module_cu_mskf.o"
    objdump -d --no-show-raw-insn variant-pristine/module_cu_mskf.o \
        | sed '1,/^Disassembly/d' > ours.dis
    objdump -d --no-show-raw-insn "${wrf_obj}" \
        | sed '1,/^Disassembly/d' > wrf.dis
    if cmp -s ours.dis wrf.dis; then
        echo "pristine module_cu_mskf.o text identical to ${wrf_obj}" \
            > objdump-match.txt
    else
        echo "pristine module_cu_mskf.o text DIFFERS from ${wrf_obj}" \
            > objdump-match.txt
        diff ours.dis wrf.dis | head -40 >> objdump-match.txt || true
    fi
    rm -f ours.dis wrf.dis
fi

# --- uninitialised-memory probes ----------------------------------------------
: > uninit-pristine.txt
: > uninit-ref.txt
for fill in nan zero inf; do
    build_variant "pristine-init-${fill}" module_cu_mskf.f90 \
        -finit-real="${fill}" -finit-integer=-77777
    build_variant "ref-init-${fill}" module_cu_mskf_ref.f90 \
        -finit-real="${fill}" -finit-integer=-77777
    echo "${fill} $(words_differ variant-pristine/out_f32.bin \
        "variant-pristine-init-${fill}/out_f32.bin")" >> uninit-pristine.txt
    echo "${fill} $(words_differ out_f32.bin \
        "variant-ref-init-${fill}/out_f32.bin")" >> uninit-ref.txt
done
if awk '$2 != 0 {bad=1} END {exit !bad}' uninit-ref.txt; then
    echo "an uninitialised read still reaches a reference output:" >&2
    cat uninit-ref.txt >&2
    exit 7
fi

# --- evidence: the reference source at -O0 ------------------------------------
build_variant o0 module_cu_mskf_ref.f90 -O0
echo "O0 vs O2 reference: $(words_differ out_f32.bin variant-o0/out_f32.bin)" \
    > o0-vs-o2.txt

# --- probes: integer powers, MAX/MIN on signed zero, NINT, AINT ----------------
gfortran "${wrf_fflags[@]}" -o lowering_probe "${script_dir}/lowering_probe.F90"
./lowering_probe > lowering-probe.txt

sha256sum out_f32.bin cases_i32.bin cases_f32.bin cases.json \
    > oracle-sha256sums.txt
echo "isolation (ref): $(awk '{s+=$2} END {print s}' isolation.txt) words"
echo "isolation (pristine): $(awk '{s+=$2} END {print s}' \
    isolation-pristine.txt) words"
cat pristine-vs-ref.txt uninit-pristine.txt uninit-ref.txt o0-vs-o2.txt
cat objdump-match.txt 2>/dev/null | head -1 || true
