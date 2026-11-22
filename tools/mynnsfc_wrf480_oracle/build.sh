#!/usr/bin/env bash
# Build the MYNN-SFC module replay against a strict WRF v4.8.0 referee build.
# Run inside localhost/wrfref2404:1 with the port folder at /w, e.g.
#   podman run --rm -v $HOME/wrf480-port:/w -v $PWD:/t -w /t localhost/wrfref2404:1 \
#     bash /t/build.sh /w/build/WRF-4.8.0-serial-pristine /t/replay_mynnsfc.exe
set -euo pipefail
B=${1:?referee build dir}
OUT=${2:?output executable}
HERE=$(cd "$(dirname "$0")" && pwd)
FLAGS="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz"
gfortran $FLAGS -I"$B/main" -I"$B/phys" -I"$B/frame" -I"$B/share" -I"$B/dyn_em" -I"$B/inc" \
  -c "$HERE/replay_mynnsfc.F90" -o "${OUT%.exe}.o"
gfortran -o "$OUT" $FLAGS "${OUT%.exe}.o" "$B/main/libwrflib.a" \
  "$B/external/fftpack/fftpack5/libfftpack.a" "$B/external/io_grib1/libio_grib1.a" \
  "$B/external/io_grib_share/libio_grib_share.a" "$B/external/io_int/libwrfio_int.a" \
  -L"$B/external/esmf_time_f90" -lesmf_time "$B/external/RSL_LITE/librsl_lite.a" \
  "$B/frame/module_internal_header_util.o" "$B/frame/pack_utils.o" \
  -L"$B/external/io_netcdf" -lwrfio_nf -L/w/build/netcdf-serial/lib -lnetcdff
gfortran --version | head -1
sha256sum "$OUT"
