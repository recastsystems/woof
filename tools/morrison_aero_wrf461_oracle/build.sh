#!/usr/bin/env bash
# Build and run the mp_physics=40 (Morrison two-moment, CESM-NCSU aerosol)
# column oracle from WRF v4.6.1.
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR OUT_DIR
#
# Two references are built from the pinned source:
#   pinned  module_mp_morr_two_moment_aero.F byte-unmodified.
#   t0fix   the same file with the one documented correction WOOF ships
#           (t0fix.patch: the Abdul-Razzak-Ghan activation reads the melting
#           point SVPT0 = 273.15 K where WRF reads the dynamics base-state
#           temperature T0 = 300 K).  Every other byte is the pinned file.
# A third, constants-only build removes the PRIVATE attributes (and nothing
# else) so dump_constants can print the words MORR_TWO_MOMENT_INIT_AERO holds.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR OUT_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
out_dir=$(realpath -m "$3")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
constants_source="${source_root}/share/module_model_constants.F"
radar_source="${source_root}/phys/module_mp_radar.F"
aero_source="${source_root}/phys/module_mp_morr_two_moment_aero.F"
for source in "${constants_source}" "${radar_source}" "${aero_source}"; do
    test -f "${source}"
done
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- \
        share/module_model_constants.F phys/module_mp_radar.F \
        phys/module_mp_morr_two_moment_aero.F; then
    echo "a compiled WRF dependency differs from pinned HEAD" >&2
    exit 3
fi

mkdir -p "${build_dir}" "${out_dir}"
cd "${build_dir}"
ff=(-cpp -ffree-form -ffree-line-length-none)

# --- shared WRF dependencies, -O0 (scalar libm, no contraction) -------------
gfortran -c -O0 "${ff[@]}" "${script_dir}/stub_wrf_error.F90"
gfortran -c -O0 "${ff[@]}" "${constants_source}"
gfortran -c -O0 "${ff[@]}" -I "${build_dir}" "${radar_source}"

# --- the three copies of the aero module ------------------------------------
cp "${aero_source}" aero_pinned.F
cp "${aero_source}" aero_t0fix.F
patch -s aero_t0fix.F < "${script_dir}/t0fix.patch"
# constants-only copy: strip PRIVATE visibility, nothing else
sed -E -e 's/,[[:space:]]*[Pp][Rr][Ii][Vv][Aa][Tt][Ee]//' \
       -e 's/^([[:space:]]*)[Pp][Rr][Ii][Vv][Aa][Tt][Ee][[:space:]]*::.*$/\1! (visibility statement removed)/' \
       aero_pinned.F > aero_public.F
# The constants copy may differ from the pinned file only in lines that
# carried PRIVATE.  Refuse anything else.
diff aero_pinned.F aero_public.F | grep -E '^[<>]' | grep -v -i -E 'private|visibility statement removed' \
    && { echo "constants copy differs beyond PRIVATE attributes" >&2; exit 4; } || true
# The t0fix copy may differ only in the documented t0 lines.
diff aero_pinned.F aero_t0fix.F > t0fix-applied.diff || true
if grep -E '^[<>]' t0fix-applied.diff | grep -v -E 't0|svpt0' ; then
    echo "t0fix copy differs beyond the documented t0 lines" >&2; exit 4
fi

build_variant() {  # name source opt
    local name=$1 src=$2 opt=$3
    mkdir -p "${name}"
    gfortran -c ${opt} "${ff[@]}" -DWRF_CHEM=0 -fallow-argument-mismatch \
        -I "${build_dir}" -J "${name}" -o "${name}/aero.o" "${src}"
    gfortran -c -O0 "${ff[@]}" -I "${build_dir}" -I "${name}" -J "${name}" \
        -o "${name}/run.o" "${script_dir}/run_morrison_aero.F90"
    gfortran -o "${name}/run_morrison_aero" stub_wrf_error.o \
        module_model_constants.o module_mp_radar.o "${name}/aero.o" "${name}/run.o"
    nm -u "${name}/aero.o" | sed 's/^ *//' | sort > "${name}/undefined.txt"
    if grep -q '_ZGV' "${name}/undefined.txt"; then
        echo "${name}: libmvec vector symbol linked" >&2; exit 5
    fi
}
build_variant pinned aero_pinned.F -O0
build_variant t0fix aero_t0fix.F -O0
# Evidence only: WRF's own optimisation level (scalar, no contraction).
build_variant pinned_O2 aero_pinned.F "-O2 -fno-tree-vectorize -ffp-contract=off -fno-fast-math"

mkdir -p public
gfortran -c -O0 "${ff[@]}" -DWRF_CHEM=0 -fallow-argument-mismatch \
    -I "${build_dir}" -J public -o public/aero.o aero_public.F
gfortran -c -O0 "${ff[@]}" -I "${build_dir}" -I public -J public \
    -o public/dump.o "${script_dir}/dump_constants.F90"
gfortran -o public/dump_constants stub_wrf_error.o module_model_constants.o \
    module_mp_radar.o public/aero.o public/dump.o
mkdir -p public_t0fix
sed -E -e 's/,[[:space:]]*[Pp][Rr][Ii][Vv][Aa][Tt][Ee]//' \
       -e 's/^([[:space:]]*)[Pp][Rr][Ii][Vv][Aa][Tt][Ee][[:space:]]*::.*$/\1! (visibility statement removed)/' \
       aero_t0fix.F > aero_public_t0fix.F
gfortran -c -O0 "${ff[@]}" -DWRF_CHEM=0 -fallow-argument-mismatch \
    -I "${build_dir}" -J public_t0fix -o public_t0fix/aero.o aero_public_t0fix.F
gfortran -c -O0 "${ff[@]}" -I "${build_dir}" -I public_t0fix -J public_t0fix \
    -o public_t0fix/dump.o "${script_dir}/dump_constants.F90"
gfortran -o public_t0fix/dump_constants stub_wrf_error.o module_model_constants.o \
    module_mp_radar.o public_t0fix/aero.o public_t0fix/dump.o

gfortran -c -O0 "${ff[@]}" -I "${build_dir}" -I public -J public -o public/radar_dump.o "${script_dir}/dump_radar.F90"
gfortran -o public/dump_radar stub_wrf_error.o module_model_constants.o module_mp_radar.o public/aero.o public/radar_dump.o
for r in 0 1; do
    public/dump_radar $r > "${out_dir}/radar-rimed$r.txt"
    public/dump_constants $r > "${out_dir}/constants-pinned-rimed$r.txt"
    public_t0fix/dump_constants $r > "${out_dir}/constants-t0fix-rimed$r.txt"
done

# --- runs: (rimed, aercu_opt, dt) --------------------------------------------
modes=("1 0 20.0" "0 0 20.0" "1 0 60.0" "1 2 20.0" "0 2 30.0" "1 1 20.0")
for variant in pinned t0fix pinned_O2; do
    for m in "${modes[@]}"; do
        read -r rimed aercu dt <<< "${m}"
        tag="r${rimed}a${aercu}dt${dt%.*}"
        for layout in batch single; do
            "${variant}/run_morrison_aero" "${rimed}" "${aercu}" "${dt}" "${layout}" \
                "${variant}/${tag}-${layout}-levels.csv" \
                "${variant}/${tag}-${layout}-surface.csv"
        done
        # WRF's tile loop carries SAVEd state from column to column; the
        # oracle is only usable if one call per column gives the same words.
        cmp "${variant}/${tag}-batch-levels.csv" "${variant}/${tag}-single-levels.csv"
        cmp "${variant}/${tag}-batch-surface.csv" "${variant}/${tag}-single-surface.csv"
        if [[ "${variant}" != pinned_O2 ]]; then
            cp "${variant}/${tag}-batch-levels.csv" "${out_dir}/${variant}-${tag}-levels.csv"
            cp "${variant}/${tag}-batch-surface.csv" "${out_dir}/${variant}-${tag}-surface.csv"
        fi
    done
done

# -O0 against WRF's -O2 (scalar): a receipt, not a gate.
{
    echo "# pinned -O0 vs pinned -O2 -fno-tree-vectorize -ffp-contract=off"
    for m in "${modes[@]}"; do
        read -r rimed aercu dt <<< "${m}"
        tag="r${rimed}a${aercu}dt${dt%.*}"
        if cmp -s "pinned/${tag}-batch-levels.csv" "pinned_O2/${tag}-batch-levels.csv" \
           && cmp -s "pinned/${tag}-batch-surface.csv" "pinned_O2/${tag}-batch-surface.csv"; then
            echo "${tag} identical"
        else
            n=$(diff "pinned/${tag}-batch-levels.csv" "pinned_O2/${tag}-batch-levels.csv" | grep -c '^<' || true)
            echo "${tag} differ in ${n} level rows"
        fi
    done
} > "${out_dir}/O0-vs-O2.txt"
cat "${out_dir}/O0-vs-O2.txt"

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# libm symbols called by the -O0 pinned object:"
    grep -E 'expf|powf|logf|log10f|sqrtf|powisf2|_ZGV' pinned/undefined.txt || true
} > "${out_dir}/libm-report.txt"
cat "${out_dir}/libm-report.txt"

(cd "${out_dir}" && sha256sum "${constants_source}" "${radar_source}" "${aero_source}" \
    "${script_dir}/run_morrison_aero.F90" "${script_dir}/dump_constants.F90" \
    "${script_dir}/stub_wrf_error.F90" "${script_dir}/t0fix.patch" \
    *.csv constants-*.txt radar-*.txt > oracle-sha256sums.txt)
echo "${pinned_commit}" > "${out_dir}/wrf-commit.txt"
gfortran --version | head -1 > "${out_dir}/compiler.txt"
echo "mp_physics=40 oracle written to ${out_dir}"
