#!/usr/bin/env bash
set -euo pipefail
[ "$#" = 4 ] || { echo 'usage: build_hail.sh WRF_ROOT BUILD_DIR CCN_FILE PREBUILT_TABLE_DIR' >&2; exit 2; }
script_dir=$(cd "$(dirname "$0")" && pwd)
wrf_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
ccn=$(realpath "$3")
prebuilt=$(realpath "$4")
test "$(git -C "$wrf_root" rev-parse HEAD)" = d66e442fccc04111067e29274c9f9eaccc3cef28
git -C "$wrf_root" diff --exit-code -- phys/module_mp_thompson.F phys/module_mp_radar.F
test ! -e "$build_dir"
mkdir -p "$build_dir/pristine" "$build_dir/tables" "$build_dir/column-oracle-hail"
cd "$build_dir/pristine"
fc=${FC:-gfortran}
flags=(-O2 -fno-tree-vectorize -ffree-form -ffree-line-length-none)
"$fc" -c "${flags[@]}" "$script_dir/stub_wrf_hail.F90"
"$fc" -c "${flags[@]}" "$wrf_root/phys/module_mp_radar.F"
"$fc" -c "${flags[@]}" -cpp -DWRF_CHEM=0 "$wrf_root/phys/module_mp_thompson.F"
for program in generate_tables_hail run_column_hail; do
  "$fc" -c "${flags[@]}" "$script_dir/$program.F90"
  "$fc" "${flags[@]}" -o "$program" stub_wrf_hail.o module_mp_radar.o module_mp_thompson.o "$program.o"
  if nm -D "$program" | grep -q _ZGV; then echo 'SIMD math is not the scalar referee' >&2; exit 3; fi
done
cd ../tables
for table in qr_acr_qg_mp38V1.dat qr_acr_qsV2.dat freezeH2O.dat; do ln "$prebuilt/$table" "$table"; done
cp "$ccn" CCN_ACTIVATE.BIN
../pristine/generate_tables_hail > ../readback.log
grep -q 'read qr_acr_qg_mp38V1.dat instead of computing' ../readback.log
export GFORTRAN_CONVERT_UNIT='big_endian:20'
for scenario in gh-convective gh-stratiform gh-hail gh-density-sweep gh-warm-melt; do
  ../pristine/run_column_hail "$scenario" ../column-oracle-hail > "../column-$scenario.log"
done
{
  printf 'source_commit='; git -C "$wrf_root" rev-parse HEAD
  printf 'compiler='; "$fc" --version | head -1
  printf 'libc='; ldd --version | head -1
  printf 'flags=%s\n' "${flags[*]}"
  sha256sum "$wrf_root/phys/module_mp_thompson.F" "$wrf_root/phys/module_mp_radar.F"
  sha256sum "$script_dir/build_hail.sh" "$script_dir/"{stub_wrf_hail,generate_tables_hail,run_column_hail}.F90
  sha256sum *.dat CCN_ACTIVATE.BIN ../column-oracle-hail/*.csv
} > ../PROVENANCE-HAIL.txt
cat ../PROVENANCE-HAIL.txt
