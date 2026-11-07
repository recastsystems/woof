#!/usr/bin/env bash
# Build and run the MRF PBL column oracle (bl_pbl_physics = 99) against the
# byte-unmodified WRF v4.6.1 phys/module_bl_mrf.F.
#
#   bash tools/mrf_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD]
#
# WRF_BUILD (default /work/wrf/build/scalar/WRF-4.6.1) is a compiled WRF
# v4.6.1 tree.  THE REFEREE ARM LINKS ITS phys/module_bl_mrf.o: the object
# WRF's own `compile em_real` produced with that tree's FCOPTIM, not a
# recompile of ours, so the oracle runs the exact machine code wrf.exe runs.
#
# The second arm ("rebuild") compiles the pinned source here with the FCOPTIM
# read from that tree's configure.wrf; its output must be byte-identical to
# the referee's or this script fails (the object is what the source says).
#
# A third arm compiles the pinned source at -O0 and is DIAGNOSTIC ONLY.  It
# is not WRF as built: at -O0 gfortran leaves ZFAC**PFAC (PFAC = 2.0,
# module_bl_mrf.F:978) as a fifth powf call, which -O1 and above fold to
# ZFAC*ZFAC, and powf(x, 2.0) is not always the correctly rounded square.
# The receipt records the call counts of every object and how many words the
# -O0 arm writes differently, so that difference is measured, not assumed.
#
# No stub: module_bl_mrf.F has no USE statement and calls only its own
# TRIDI2, so it compiles as it ships.  `nm -u` on each object is the receipt
# (libm and libgfortran only).
#
# Exponents in the referee object: (P1000mb/PSFCPA)**ROVCP,
# (P1000mb/(PL*1000.))**ROVCP, (1.-APHI16*HOL1)**(-1./4.) and **(-1./2.)
# remain powf calls; ZFAC**PFAC and every integer exponent are multiplies.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
wrf_build=$(realpath "${3:-/work/wrf/build/scalar/WRF-4.6.1}")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
mrf_source="${source_root}/phys/module_bl_mrf.F"
pinned_sha256="a1306d0063043d001fa4a53b9219031b931999c7d7daea172fd97518bc9b19eb"

actual=$(sha256sum "${mrf_source}" | awk '{print $1}')
if [[ "${actual}" != "${pinned_sha256}" ]]; then
    echo "module_bl_mrf.F is ${actual}, not the pinned ${pinned_sha256}" >&2
    exit 3
fi
ref_src_sha=$(sha256sum "${wrf_build}/phys/module_bl_mrf.F" | awk '{print $1}')
if [[ "${ref_src_sha}" != "${pinned_sha256}" ]]; then
    echo "the referee build compiled a different module_bl_mrf.F" >&2
    exit 3
fi

mkdir -p "${build_dir}"/{referee,rebuild,o0}
cd "${build_dir}"
driver_flags=(-O0 -ffp-contract=off -ffree-form -ffree-line-length-none)
wrf_flags=(-w -ffree-form -ffree-line-length-none -fallow-argument-mismatch
           -fallow-invalid-boz)

# --- referee arm: WRF's own object --------------------------------------------
cp "${wrf_build}/phys/module_bl_mrf.o" "${wrf_build}/phys/module_bl_mrf.mod" \
   referee/
# --- rebuild arm: the pinned source at the referee tree's FCOPTIM -------------
read -r -a fcoptim <<< "$(sed -n 's/^FCOPTIM[[:space:]]*=[[:space:]]*//p' \
                          "${wrf_build}/configure.wrf")"
gfortran -c "${fcoptim[@]}" "${wrf_flags[@]}" -J rebuild \
    -o rebuild/module_bl_mrf.o "${mrf_source}"
# --- diagnostic -O0 arm ----------------------------------------------------------
gfortran -c -O0 "${wrf_flags[@]}" -J o0 -o o0/module_bl_mrf.o "${mrf_source}"
for arm in referee rebuild o0; do
    gfortran -c "${driver_flags[@]}" -I "${arm}" -J "${arm}" \
        -o "${arm}/run_bl_mrf.o" "${script_dir}/run_bl_mrf.F90"
    gfortran -o "${arm}/run_bl_mrf" "${arm}/run_bl_mrf.o" \
        "${arm}/module_bl_mrf.o"
done

# --- receipts -------------------------------------------------------------------
{
    echo "# module_bl_mrf.F sha256 ${pinned_sha256}"
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# referee object: ${wrf_build}/phys/module_bl_mrf.o"
    echo "#   sha256 $(sha256sum referee/module_bl_mrf.o | awk '{print $1}')"
    grep -E '^(FCOPTIM|FCBASEOPTS_NO_G|FORMAT_FREE|FCCOMPAT|PROMOTION)' \
        "${wrf_build}/configure.wrf" | sed 's/^/#   /'
    echo "# referee undefined symbols:"
    nm -u referee/module_bl_mrf.o | sed 's/^ */#   /'
    for arm in referee rebuild o0; do
        echo "# ${arm} libm call sites (objdump -dr relocations):"
        for f in powf expf logf sqrtf; do
            n=$(objdump -dr "${arm}/module_bl_mrf.o" \
                | grep -c "R_X86_64_PLT32[[:space:]]*${f}" || true)
            echo "#   ${f}: ${n}"
        done
        echo "#   sqrtss: $(objdump -d "${arm}/module_bl_mrf.o" \
                            | grep -c sqrtss || true)"
    done
} > build-receipt.txt
if nm -u referee/module_bl_mrf.o rebuild/module_bl_mrf.o | grep -q '_ZGV'; then
    echo "an MRF object pulled in a libmvec vector symbol" >&2
    exit 4
fi

# --- run every arm on the same cases ----------------------------------------------
"${python}" "${script_dir}/gen_mrf_cases.py" "${build_dir}"
for arm in referee rebuild o0; do
    ( cd "${arm}" && cp ../mrf-cases.bin . && ./run_bl_mrf )
done
if ! cmp -s referee/mrf-oracle-out.bin rebuild/mrf-oracle-out.bin; then
    echo "the referee object and the rebuild at its FCOPTIM disagree" >&2
    cmp referee/mrf-oracle-out.bin rebuild/mrf-oracle-out.bin | head -3 >&2
    exit 5
fi
nwords=$(( $(stat -c %s referee/mrf-oracle-out.bin) / 4 ))
ndiff=$( (cmp -l referee/mrf-oracle-out.bin o0/mrf-oracle-out.bin || true) \
         | awk '{print int(($1-1)/4)}' | sort -u | wc -l)
{
    echo "# referee and rebuild outputs byte-identical:" \
         "$(sha256sum referee/mrf-oracle-out.bin | awk '{print $1}')"
    echo "# diagnostic -O0 arm: ${ndiff} of ${nwords} words differ from the" \
         "referee (the unfolded ZFAC**PFAC powf call)"
} >> build-receipt.txt
"${python}" "${script_dir}/pack_mrf_fixture.py" referee/mrf-oracle-out.bin \
    "${build_dir}/fixture"
cp build-receipt.txt mrf-cases.json "${build_dir}/fixture/"
( cd "${build_dir}/fixture" && sha256sum mrf-levels.csv mrf-surface.csv \
    mrf-cases.json build-receipt.txt > oracle-sha256sums.txt )
cat build-receipt.txt
echo "fixture in ${build_dir}/fixture"
