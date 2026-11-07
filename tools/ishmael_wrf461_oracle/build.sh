#!/usr/bin/env bash
# Build one pristine, strict WRF mp=55 replay executable. No campaign runs.
# build.sh WRF_SOURCE_ROOT BUILD_DIR [CAPTURED_INPUTS DT]
# The source and tables are identical at WRF v4.6.1 and v4.7.1.
set -euo pipefail
if [[ $# != 2 && $# != 4 ]]; then
    echo 'usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [CAPTURED_INPUTS DT]' >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ $# == 4 ]]; then
    captured_inputs=$(realpath "$3")
fi
module_source="${source_root}/phys/module_mp_jensen_ishmael.F"
module_sha='4cefa1a2f28c2bc36061100bd4bef10371643ccf1e80da43be96f8ba31843326'
printf '%s  %s\n' "$module_sha" "$module_source" | sha256sum -c -
mkdir -p "${build_dir}"
cd "${build_dir}"
base=(-cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch
      -fallow-invalid-boz -fconvert=big-endian -frecord-marker=4
      -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize)
gfortran -c "${base[@]}" "${script_dir}/stub_wrf_error.F90"
gfortran -c "${base[@]}" "${module_source}"
gfortran -c "${base[@]}" "${script_dir}/run_ishmael.F90"
gfortran -o run_ishmael stub_wrf_error.o module_mp_jensen_ishmael.o run_ishmael.o
nm -u module_mp_jensen_ishmael.o > undefined.txt
if rg -q '_ZGV' undefined.txt; then
    echo 'strict native object pulled in vector libm symbols' >&2
    exit 4
fi
sha256sum "${module_source}" "${script_dir}/run_ishmael.F90" > source-sha256sums.txt
if [[ $# == 4 ]]; then
    # The files must come from capture_native.py on a real forecast. The
    # caller retains that run's log and capture-instrumentation receipt.
    for table in ishmael-gamma-tab.bin ishmael-qi-qc.bin ishmael-qi-qr.bin; do
        ln -s "${source_root}/run/${table}" "${table}"
    done
    printf '%s  %s\n' \
      d85e4960053024a73034fb962ff52bd62aeae1b975dd5b4f62244ec82d622daa ishmael-gamma-tab.bin \
      b8e4cea138b78668f027934bb8e39c6ea2cb62c1d277c17249c72003c8f4646c ishmael-qi-qc.bin \
      69dd41d3d285318c95cac0d2dde10f8a3b25f802bf52ac781f29c87adb964cc4 ishmael-qi-qr.bin | sha256sum -c -
    ./run_ishmael "$captured_inputs" replay-want.bin 1 "$4" > replay.log
fi
