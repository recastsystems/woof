#!/usr/bin/env bash
# Build and run the WRF v4.6.1 SAS (cu_physics = 94) column oracle.
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]
#
# The harness (run_sas.f90 + sas_tables_dump.c) is LINKED AGAINST THE
# OBJECTS OF THE BUILT WRF TREES -- WRF_BUILD_ROOT/{strict,scalar}/WRF-4.6.1/
# phys/module_{cu_sas,gfs_funcphys,gfs_physcons,gfs_machine}.o -- so the
# arithmetic graded is exactly what that tree's wrf.exe executes (configure
# option 32, serial, FCOPTIM from the tree's own configure.wrf; scalar adds
# -fno-tree-vectorize and is the 0-ULP referee).  The harness itself is
# compiled with the same FCOPTIM so the stage-1 prep it transcribes from
# CU_SAS :312-406 rounds like the original; run_sas proves that by recombining
# the stages and matching CU_SAS word for word ("chained").
#
# Refuses: a WRF source tree not at the pinned commit, any edited byte in the
# four pinned files, a built tree whose copy of them differs from the pinned
# source, and any libmvec (_ZGV*) symbol in a linked object (vector libm
# would not be the scalar exp/pow the GPU port reproduces).
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
wrf_build_root=$(realpath "${3:-/work/wrf/build}")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
declare -A pinned=(
  ["phys/module_cu_sas.F"]="a59026ff2061293ce657e7d56fdcbe2381a615721e2801616997cd3f96456158"
  ["phys/module_gfs_funcphys.F"]="0d43dccf0510bd49ef07f81c357e4b96fd9391604f7c14d7e3cf8700ff383e74"
  ["phys/module_gfs_physcons.F"]="2b8ef99663ba62748b57c7002f5e5b8c9338f2a825c1d32c113bd4e3bff2eb52"
  ["phys/module_gfs_machine.F"]="300ee4f75663e89d34c8a9b8733cc07a5dd48c3d973da019d31cb76524babe25"
  # Not compiled here; pinned because the refusal of an adaptive step and
  # the absent shalconv/sas_mass_flux arguments are read from it (:1131-1154).
  ["phys/module_cumulus_driver.F"]="d73eb4b670599e0bbf98a3661b5dbf17413aed01139d18fa1a84161578dc31e3"
)

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 1
fi
for rel in "${!pinned[@]}"; do
    got=$(sha256sum "${source_root}/${rel}" | cut -d' ' -f1)
    if [[ "${got}" != "${pinned[$rel]}" ]]; then
        echo "${rel} digest ${got} is not the pinned ${pinned[$rel]}" >&2
        exit 1
    fi
done

mkdir -p "${build_dir}"
python3 "${script_dir}/gen_cases.py" "${build_dir}/cases.bin" > "${build_dir}/gen_cases.log"

for tree in strict scalar; do
    t="${wrf_build_root}/${tree}/WRF-4.6.1"
    for rel in phys/module_cu_sas.F phys/module_gfs_funcphys.F \
               phys/module_gfs_physcons.F phys/module_gfs_machine.F; do
        cmp -s "${source_root}/${rel}" "${t}/${rel}" || {
            echo "${t}/${rel} differs from the pinned source" >&2; exit 1; }
    done
    fcoptim=$(sed -n 's/^FCOPTIM[[:space:]]*=[[:space:]]*//p' "${t}/configure.wrf")
    out="${build_dir}/${tree}"
    mkdir -p "${out}"
    objs=()
    for m in cu_sas gfs_funcphys gfs_physcons gfs_machine; do
        objs+=("${t}/phys/module_${m}.o")
    done
    if nm "${objs[@]}" | grep -q '_ZGV'; then
        echo "${tree}: a linked WRF object calls libmvec" >&2; exit 1
    fi
    ( cd "${out}"
      # shellcheck disable=SC2086
      gfortran ${fcoptim} -w -ffree-form -ffree-line-length-none \
          -I"${t}/phys" -c "${script_dir}/run_sas.f90" -o run_sas.o
      gcc -O0 -c "${script_dir}/sas_tables_dump.c" -o sas_tables_dump.o
      gfortran -o run_sas run_sas.o sas_tables_dump.o "${objs[@]}" -lm
      if nm run_sas.o | grep -q '_ZGV'; then
          echo "${tree}: harness calls libmvec" >&2; exit 1
      fi
      echo "FCOPTIM ${fcoptim}" > flags.txt
      gfortran --version | head -1 > compiler.txt
      ln -sf ../cases.bin cases.bin
      ./run_sas . | tee run.log )
done

# The scalar tree is the referee; the vectorizing tree must agree with it.
for f in stage1.bin stage2.bin stage3.bin stage3_shaloff.bin tables.bin; do
    if cmp -s "${build_dir}/strict/${f}" "${build_dir}/scalar/${f}"; then
        echo "strict == scalar ${f}"
    else
        echo "strict != scalar ${f}"
    fi
done | tee "${build_dir}/strict-vs-scalar.txt"
