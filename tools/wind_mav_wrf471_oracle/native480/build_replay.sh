#!/usr/bin/env bash
# inside localhost/wrfref2404:1: build the replay driver against one pinned strict referee build
# usage: build_replay.sh BUILD_DIR OUT_EXE [-DW480]
set -euo pipefail
B=$1; OUT=$2; DEF=${3:-}
gfortran --version | head -1
FL='-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz'
cd "$(dirname "$0")"
gfortran $FL -cpp $DEF -I$B/main -I$B/phys -I$B/frame -I$B/share -I$B/inc -I$B/external/esmf_time_f90 -c fitch_replay.F90 -o $OUT.o
gfortran $FL -o $OUT $OUT.o $B/main/libwrflib.a $B/external/fftpack/fftpack5/libfftpack.a $B/external/io_grib1/libio_grib1.a \
  $B/external/io_grib_share/libio_grib_share.a $B/external/io_int/libwrfio_int.a -L$B/external/esmf_time_f90 -lesmf_time \
  $B/external/RSL_LITE/librsl_lite.a $B/frame/module_internal_header_util.o $B/frame/pack_utils.o \
  -L$B/external/io_netcdf -lwrfio_nf -L/w/build/netcdf-serial/lib -lnetcdff
# record that the archive member is the build's own object
mkdir -p /tmp/ar.$$ && cd /tmp/ar.$$ && ar x $B/main/libwrflib.a module_wind_fitch.o && sha256sum module_wind_fitch.o $B/phys/module_wind_fitch.o && cd / && rm -rf /tmp/ar.$$
