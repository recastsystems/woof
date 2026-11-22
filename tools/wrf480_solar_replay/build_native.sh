#!/usr/bin/env bash
# Build the replay driver against a strict WRF referee build, inside the same
# container and with the same flags the build used. Run inside
# localhost/wrfref2404:1 with the port root mounted at /w:
#   podman run --rm -v $HOME/wrf480-port:/w -v $PWD:/r -w /r localhost/wrfref2404:1 \
#     bash /r/build_native.sh /w/build/WRF-4.8.0-serial-pristine /r/native-480
set -euo pipefail
B=$1; OUT=$2
FLAGS="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w -ffree-form -ffree-line-length-none"
grep -q -- "-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize" "$B/configure.wrf"
gfortran $FLAGS -I"$B/main" -I"$B/phys" -I"$B/share" -I"$B/frame" -I"$B/inc" -c "$(dirname "$0")/native_solar_replay.f90" -o "$OUT.o"
gfortran $FLAGS -o "$OUT" "$OUT.o" "$B/main/libwrflib.a" \
  "$B/external/fftpack/fftpack5/libfftpack.a" "$B/external/io_grib1/libio_grib1.a" \
  "$B/external/io_grib_share/libio_grib_share.a" "$B/external/io_int/libwrfio_int.a" \
  -L"$B/external/esmf_time_f90" -lesmf_time "$B/external/RSL_LITE/librsl_lite.a" \
  "$B/frame/module_internal_header_util.o" "$B/frame/pack_utils.o" \
  -L"$B/external/io_netcdf" -lwrfio_nf -L"$(dirname "$B")/netcdf-serial/lib" -lnetcdff -lnetcdf
sha256sum "$B/phys/module_radiation_driver.o" "$B/main/wrf.exe" "$OUT"
gfortran --version | head -1
