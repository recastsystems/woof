#!/usr/bin/env bash
set -euo pipefail
root=$(realpath "$1"); out=$(realpath -m "$2"); here=$(cd "$(dirname "$0")" && pwd)
bash "$here/build_soil.sh" "$root" "$out"
sources=(phys/ccpp_kind_types.F phys/physics_mmm/module_libmassv.F90 phys/physics_mmm/mp_radar.F90 phys/physics_mmm/mp_wsm6.F90)
git -C "$root" diff --quiet HEAD -- "${sources[@]}"
cd "$out"
flags=(-O0 -fno-tree-vectorize -ffp-contract=off -cpp -ffree-form -ffree-line-length-none)
gfortran "${flags[@]}" -DRWORDSIZE=4 -c "$root/phys/ccpp_kind_types.F"
gfortran "${flags[@]}" -c "$root/phys/physics_mmm/module_libmassv.F90"
gfortran "${flags[@]}" -c "$root/phys/physics_mmm/mp_radar.F90"
gfortran "${flags[@]}" -c "$root/phys/physics_mmm/mp_wsm6.F90"
gfortran "${flags[@]}" -c "$here/run_sprinkler_soil.F90"
gfortran -o run_sprinkler_soil stub_wrf.o module_model_constants.o module_wrf_error.o module_sf_noahlsm.o module_sf_noahlsm_glacial_only.o module_sf_noahdrv.o module_irrigation.o ccpp_kind_types.o module_libmassv.o mp_radar.o mp_wsm6.o run_sprinkler_soil.o
./run_sprinkler_soil sprinkler 1 T F F 3 > sprinkler-stdout.txt
sha256sum "$here/run_sprinkler_soil.F90" "$root/phys/physics_mmm/"{module_libmassv,mp_radar,mp_wsm6}.F90 sprinkler-step*.csv sprinkler-water*.txt > sprinkler-soil-SHA256SUMS
