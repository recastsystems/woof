#!/usr/bin/env bash
# Build the WRF v4.6.1 NSSL variant column oracle from the unchanged official
# module_mp_nssl_2mom.F. This standalone strict build uses the recorded
# GNU Fortran flags below. It is not a whole native WRF forecast build.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 /path/to/WRF-v4.6.1 /new/build-directory" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
wrf_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
official_source="$wrf_root/phys/module_mp_nssl_2mom.F"
# Canonical-LF SHA-256 of WRF v4.6.1 d66e442 phys/module_mp_nssl_2mom.F (CR stripped).
expected_lf_sha=1eb1b138b75ff3b0cfe33c23779f4ec9b72e57a5455a53ef11c9e55ae0f42722
expected_licence_sha=b30611d35ad1922bc741e86d21e1a897234c1a9c9e2af28076e021f565dfbec6

if [ ! -f "$wrf_root/LICENSE.txt" ]; then
  echo "missing WRF public-domain statement: $wrf_root/LICENSE.txt" >&2
  exit 2
fi
actual_licence_sha=$(sed 's/\r$//' "$wrf_root/LICENSE.txt" | sha256sum | awk '{print $1}')
if [ "$actual_licence_sha" != "$expected_licence_sha" ]; then
  echo "unexpected WRF licence SHA-256: $actual_licence_sha" >&2
  exit 2
fi

if [ ! -f "$official_source" ]; then
  echo "missing official WRF source: $official_source" >&2
  exit 2
fi
actual_source_sha=$(sed 's/\r$//' "$official_source" | sha256sum | awk '{print $1}')
if [ "$actual_source_sha" != "$expected_lf_sha" ]; then
  echo "unexpected canonical-LF module_mp_nssl_2mom.F SHA-256: $actual_source_sha" >&2
  exit 2
fi
if [ -e "$build_dir" ]; then
  echo "refusing to reuse variant oracle build directory: $build_dir" >&2
  exit 2
fi

mkdir -p "$build_dir"
cp "$official_source" "$build_dir/module_mp_nssl_2mom.F"
cd "$build_dir"
sed -i 's/\r$//' module_mp_nssl_2mom.F
touch namelist.input
flags="-O2 -fno-tree-vectorize -fno-fast-math -ffp-contract=off -w -ffree-form -ffree-line-length-none"
gfortran -c $flags -cpp -DWRF_CHEM=0 module_mp_nssl_2mom.F
gfortran -c $flags "$script_dir/../nssl2_wrf461_composed_replay/stub_wrf.F90" \
  "$script_dir/variant_column.F90"
gfortran -O2 -o nssl2_variant_column \
  module_mp_nssl_2mom.o stub_wrf.o variant_column.o

printf '%s  %s\n' "$actual_source_sha" module_mp_nssl_2mom.F > SOURCE_SHA256
printf '%s  %s\n' "$actual_licence_sha" LICENSE.txt > LICENCE_SHA256
gfortran --version | head -n 1 > TOOLCHAIN.txt
printf '%s\n' "$flags" > FLAGS.txt
echo "NSSL2_VARIANT_ORACLE_BUILT $build_dir/nssl2_variant_column"
