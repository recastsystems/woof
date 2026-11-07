#!/usr/bin/env bash
# Build and run the K-epsilon PBL oracle (bl_pbl_physics = 17) against WRF
# v4.6.1 phys/module_bl_keps.F.
#
#   bash tools/keps_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# Two references are built from the same pinned file:
#
#   pristine  the byte-unmodified module.  Kept as evidence of what WRF as
#             shipped does, including its four defects (below).
#   declared  the pinned module with declared-divergences.patch applied:
#             the four defects fixed and nothing else.  This is the fixture
#             of record the GPU port is graded against.
#
# The four defects, each a divergence WOOF declares rather than reproduces:
#   1. GET_PBLH compares against `ttke`, an undeclared, never-assigned
#      implicit REAL (module_bl_keps.F:822-823); the MYNN line it was copied
#      from compares against the TKE computed on the line above.
#   2. surface_bl_pra_ri accumulates `radsum` without ever setting it
#      (module_bl_keps.F:1065), so the cloud-top radiative velocity scale
#      starts from whatever the stack held.
#   3. the driver's first-level y-momentum flux reads the x wind
#      (module_bl_keps.F:363, `u1D`/`u_phy` on the `wv` line).
#   4. surface_bl_pra_ri forms z/L as br*psim*psim/psih (module_bl_keps.F:
#      1074), which is 0/0 in the surface layer's neutral regime, where WRF's
#      revised MM5 layer sets psim = psih = 0 exactly (sf_sfclayrev.F90:
#      450-451).  What WRF then does depends on how the compiler lowers MAX:
#      the -O2 scalar build takes the clamp (z/L = 1e-8), an -O0 build
#      carries the NaN into every tendency of the column.  The patch takes
#      the neutral regime's own z/L = 0 before the same clamps, which is the
#      -O2 build's answer by construction instead of by instruction choice.
#
# Not a defect, and not patched: the driver never copies the model-top face
# pressure into p8w1D (module_bl_keps.F:270 copies kts..kte only), but the
# radiative sum reads p8w1D(iz_pbl) at most, and GET_PBLH's exit safeguards
# hold iz_pbl - 1 <= kte - 3, so that word is never read.
#
# THE REFERENCE IS THE WRF SCALAR BUILD'S OWN FLAGS (/work/wrf/provenance/
# flags-scalar.txt: -O2 -fno-fast-math -ffp-contract=off
# -fno-tree-vectorize), not -O0.  Measured on gfortran 13.3: at -O0 every
# real power, including x**2., x**1. and x**(-1.), is a call to powf, which
# is not correctly rounded, while the compiled WRF folds exactly those
# three exponents (x*x, x, 1/x) and keeps powf for every other real
# exponent.  The two builds therefore disagree in last bits wherever a
# square is taken (shear, terms_tpe, the Prandtl profile), and the binary
# WRF users run is the -O2 one.  The -O0 builds are kept as evidence.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
keps_source="${source_root}/phys/module_bl_keps.F"
pinned_sha256="7e07e1125023e589b29ca688b2e9de69fbb3ee01794851d72b6143e5845d3584"

test -f "${keps_source}"
actual_sha256=$(sha256sum "${keps_source}" | awk '{print $1}')
if [[ "${actual_sha256}" != "${pinned_sha256}" ]]; then
    echo "module_bl_keps.F is ${actual_sha256}, not the pinned" \
         "${pinned_sha256} (WRF v4.6.1)" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"
cp "${keps_source}" module_bl_keps_pristine.F
patch -s -o module_bl_keps_declared.F "${keps_source}" \
    < "${script_dir}/declared-divergences.patch"

flags_ref="-O0 -cpp -ffree-form -ffree-line-length-none"
flags_wrf="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -cpp -ffree-form -ffree-line-length-none"

build_one() {   # tag module-source flags
    local tag=$1 src=$2 flags=$3
    mkdir -p "${tag}"
    (cd "${tag}" &&
     gfortran -c ${flags} -o module_bl_keps.o "../${src}" &&
     gfortran -c -O0 -ffree-form -ffree-line-length-none \
         -I . "${script_dir}/run_bl_keps.F90" &&
     gfortran -o run_bl_keps module_bl_keps.o run_bl_keps.o &&
     nm -u module_bl_keps.o | sed 's/^ *//' | sort > undefined.txt)
    if grep -q '_ZGV' "${tag}/undefined.txt"; then
        echo "${tag}: the module pulled in a vector libm symbol" >&2
        exit 4
    fi
}

build_one declared module_bl_keps_declared.F "${flags_wrf}"
build_one pristine module_bl_keps_pristine.F "${flags_wrf}"
build_one declared-O0 module_bl_keps_declared.F "${flags_ref}"
build_one pristine-O0 module_bl_keps_pristine.F "${flags_ref}"

sets="Acold Adev Abep Bcold Bdev"
for tag in declared pristine declared-O0 pristine-O0; do
    for set in ${sets}; do
        (cd "${tag}" && ./run_bl_keps "${set}" "keps-${set}.bin" "keps-${set}.manifest")
    done
done

# keps receives the BEP source terms and never reads them: every output
# word of the flag_bep set must equal the same columns' words without it.
compare="python3 ${script_dir}/compare_sets.py"
if command -v python >/dev/null; then compare="python ${script_dir}/compare_sets.py"; fi
status=0
if ! ${compare} declared/keps-Adev.bin declared/keps-Abep.bin --outputs-only \
        | tee bep-identity.txt | grep -q '^TOTAL differing words: 0$'; then
    echo "Abep differs from Adev: keps read a BEP source term" >&2
    status=6
fi
# Evidence: what each divergence and the -O0 lowering move, per set.
for set in ${sets}; do
    { echo "== ${set}: pristine (WRF as shipped) against declared";
      ${compare} declared/keps-${set}.bin pristine/keps-${set}.bin --outputs-only;
      echo "== ${set}: declared at -O0 against declared at WRF's flags";
      ${compare} declared/keps-${set}.bin declared-O0/keps-${set}.bin --outputs-only;
    } >> divergence-evidence.txt
done

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# reference object's undefined symbols (libm calls of record):"
    grep -E 'expf|powf|logf|sqrtf|tanhf|_ZGV' declared/undefined.txt || true
} > libm-report.txt
cat libm-report.txt

sha256sum "${keps_source}" "${script_dir}/run_bl_keps.F90" \
    "${script_dir}/declared-divergences.patch" module_bl_keps_declared.F \
    declared/keps-*.bin declared/keps-*.manifest \
    pristine/keps-*.bin > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built in ${build_dir}"
exit ${status}
