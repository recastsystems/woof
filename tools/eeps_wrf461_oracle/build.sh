#!/usr/bin/env bash
# Build and run the EEPS (bl_pbl_physics=16) column oracle against WRF v4.6.1
# phys/module_bl_eepsilon.F and share/module_model_constants.F.
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR [PYTHON]
#
# The module is compiled with the flags of the scalar WRF build that is the
# sprint's 0 ULP referee (configure.wrf FCOPTIM: -O2 -fno-fast-math
# -ffp-contract=off -fno-tree-vectorize, FORMAT_FREE), so the column runs the
# arithmetic a WRF forecast runs.
#
# ONE source edit, and it is a defect fix, not a tuning: eeps2d accumulates
# the cloud-top radiative flux into RADSUM(j) (module_bl_eepsilon.F:748) but
# never initialises it, so on a stratocumulus column the top-down TKE source
# starts from whatever the stack held.  The reference build initialises it to
# zero beside its siblings (MINRAD, KMINRAD, ... at :644-650).  The stock
# module is also built twice, with -finit-real=zero and -finit-real=snan, as
# the receipt that the stock result depends on uninitialised memory: the zero
# build must equal the fixed build word for word and the snan build must not.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [PYTHON]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
python=${3:-python3}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
eeps_source="${source_root}/phys/module_bl_eepsilon.F"
constants_source="${source_root}/share/module_model_constants.F"
eeps_sha256="b1ea6db55866af15cb3c6962b1a15fa512eda803078f7304e6a133462d7e12de"
constants_sha256="5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"

check_pin() {
    local file=$1 want=$2 got
    got=$(sha256sum "$file" | awk '{print $1}')
    if [[ "$got" != "$want" ]]; then
        echo "$file is $got, not the pinned v4.6.1/v4.7.1 $want" >&2
        exit 3
    fi
}
check_pin "$eeps_source" "$eeps_sha256"
check_pin "$constants_source" "$constants_sha256"

flags=(-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w
       -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1)

mkdir -p "$build_dir"
cd "$build_dir"

# --- the radsum fix, applied to a copy, checked to touch exactly one line ---
cp "$eeps_source" module_bl_eepsilon_stock.F
# (the v4.6.1 file has CRLF line ends; the inserted line keeps them)
sed 's/^\(         minrad(j)  = 100\.\)\(\r\{0,1\}\)$/\1\2\n         radsum(j)  = 0.\2/' \
    "$eeps_source" > module_bl_eepsilon_fixed.F
if [[ $(diff module_bl_eepsilon_stock.F module_bl_eepsilon_fixed.F \
        | grep -c '^>') -ne 1 ]]; then
    echo "radsum fix did not apply to exactly one line" >&2
    exit 4
fi
diff module_bl_eepsilon_stock.F module_bl_eepsilon_fixed.F \
    > radsum-fix.diff || true

build_variant() {   # name, module source, extra flags...
    local name=$1 src=$2
    shift 2
    mkdir -p "$name"
    (cd "$name"
     gfortran -c "${flags[@]}" "$@" -o module_model_constants.o \
         "$constants_source"
     gfortran -c "${flags[@]}" "$@" -o module_bl_eeps.o "../$src"
     gfortran -c "${flags[@]}" -o run_bl_eeps.o "$script_dir/run_bl_eeps.F90"
     gfortran -o run_bl_eeps run_bl_eeps.o module_bl_eeps.o \
         module_model_constants.o
     nm -u module_bl_eeps.o | sed 's/^ *//' | sort > undefined.txt)
}
build_variant fixed module_bl_eepsilon_fixed.F
build_variant stock-zero module_bl_eepsilon_stock.F -finit-real=zero
build_variant stock-snan module_bl_eepsilon_stock.F -finit-real=snan

# The libm calls the column makes: powf (REAL**REAL), expf, logf.  No
# vector variant and no cbrtf (x**(1./3.) stays a powf without fast-math).
grep -E 'powf|expf|logf|sqrtf|cbrtf|_ZGV' fixed/undefined.txt \
    > libm-symbols.txt || true
if grep -qE '_ZGV|cbrtf' libm-symbols.txt; then
    echo "unexpected libm symbol in the reference object" >&2
    exit 5
fi

# --- column sets -------------------------------------------------------------
# A: dt=60 (three TKE substeps), every optional moisture argument present.
# B: dt=180 (five substeps), QS3D/QG3D absent (a qv,qc,qr,qi package).
# C: restart-shaped: first call at itimestep=7 from carried, non-trivial
#    PEK_ADV/PEP_ADV (and PEK_PBL/PEP_PBL that must be ignored).
"$python" "$script_dir/make_columns.py" inputs-A.bin --dt 60 --nsteps 8
"$python" "$script_dir/make_columns.py" inputs-B.bin --dt 180 --nsteps 6 \
    --no-qs-qg
"$python" "$script_dir/make_columns.py" inputs-C.bin --dt 90 --nsteps 4 \
    --first-step 7
for set in A B C; do
    for variant in fixed stock-zero stock-snan; do
        "./$variant/run_bl_eeps" "inputs-$set.bin" "trace-$set-$variant.bin" \
            constants.txt
    done
    cmp "trace-$set-fixed.bin" "trace-$set-stock-zero.bin"
    if cmp -s "trace-$set-fixed.bin" "trace-$set-stock-snan.bin"; then
        echo "set $set: stock-snan equals fixed (no cloud-top column fired)"
    else
        echo "set $set: stock-snan differs from fixed (uninitialised RADSUM)"
    fi
done > radsum-receipt.txt
cat radsum-receipt.txt

{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "libc:     $(ldd --version | head -1)"
    echo "flags:    ${flags[*]}"
} > compiler.txt
sha256sum "$eeps_source" "$constants_source" module_bl_eepsilon_fixed.F \
    "$script_dir/run_bl_eeps.F90" "$script_dir/make_columns.py" \
    inputs-*.bin trace-*-fixed.bin constants.txt > oracle-sha256sums.txt
echo "oracle built in $build_dir"
