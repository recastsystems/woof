#!/usr/bin/env bash
# Build the unmodified public-domain WRF v4.6.1 module, one reference only.
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2; exit 2; }
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pin=d66e442fccc04111067e29274c9f9eaccc3cef28
expected=384aa324dc3664af1a8d0e6244f671a76ef888a0018feb50f511803b7732edbb
source="$source_root/phys/module_mp_wdm7.F"
echo "$expected  $source" | sha256sum -c --quiet
mkdir -p "$build_dir/stock"
cd "$build_dir/stock"
flags=(-O0 -cpp -ffree-form -ffree-line-length-none -DRWORDSIZE=4 -ffp-contract=off -fno-tree-vectorize)
gfortran -c "${flags[@]}" "$script_dir/../wdm5_wrf461_oracle/stub_wrf_error.F90"
gfortran -c "${flags[@]}" "$source_root/share/module_model_constants.F"
gfortran -c "${flags[@]}" "$source_root/phys/module_mp_radar.F"
gfortran -c -O0 -cpp -ffixed-form "$source_root/frame/libmassv.F"
gfortran -c "${flags[@]}" "$source"
gfortran -c "${flags[@]}" "$script_dir/run_wdm7.F90"
gfortran -o ../run_wdm7_stock ./*.o
nm -u module_mp_wdm7.o > undefined-O0.txt
if grep -q _ZGV undefined-O0.txt; then echo "unexpected vector libm reference" >&2; exit 4; fi
if objdump -d ../run_wdm7_stock | grep -E 'vfn?m(add|sub)' >/dev/null; then echo "unexpected FMA" >&2; exit 4; fi
sha256sum "$source" "$source_root/share/module_model_constants.F" "$source_root/phys/module_mp_radar.F" "$source_root/frame/libmassv.F" > ../oracle-sha256sums.txt
printf '%s\n' "$pin" > ../wrf-commit.txt
gfortran --version | head -1 > ../compiler.txt
