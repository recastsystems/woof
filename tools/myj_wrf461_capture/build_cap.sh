#!/usr/bin/env bash
# build_cap.sh WRF_BUILD OUT_DIR
#
# A built WRF 4.6.1 tree (configure 34, GNU dmpar) plus wrapper capture around
# MYJSFC and MYJPBL.  Compiles module_cap_io and the two patched drivers with
# the tree's own flags (taken from its compile.stdout recipe) and relinks
# wrf.exe into OUT_DIR.  The scheme objects themselves are the tree's own,
# untouched; only the drivers gain the capture blocks.
set -eo pipefail
W=$(realpath "$1"); B=$(realpath -m "$2")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f "$W/main/libwrflib.a" && -f "$W/main/wrf.o" ]] || { echo "not a built WRF tree: $W" >&2; exit 2; }
if [[ -n "${MPI_SETVARS:-}" ]]; then set +u; source "$MPI_SETVARS" >/dev/null 2>&1; set -u; fi
mkdir -p "$B"; cd "$B"
cp "$HERE/module_cap_io.f90" "$HERE/patch_drivers.py" .
FFLAGS="-O2 -ftree-vectorize -funroll-loops -w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz"
INC="-I. -I$W/dyn_em -I$W/external/esmf_time_f90 -I$W/main -I$W/external/io_netcdf -I$W/external/io_int -I$W/frame -I$W/share -I$W/phys -I$W/wrftladj -I$W/chem -I$W/inc -I/usr/include"
CPPDEFS="-P -nostdinc -I$W/inc -DEM_CORE=1 -DNMM_CORE=0 -DNMM_MAX_DIM=2600 -DDA_CORE=0 -DWRFPLUS=0 -DIWORDSIZE=4 -DDWORDSIZE=8 -DRWORDSIZE=4 -DLWORDSIZE=4 -DNONSTANDARD_SYSTEM_SUBR -DWRF_USE_CLM -DUSE_NETCDF4_FEATURES -DWRFIO_NCD_LARGE_FILE_SUPPORT -DDM_PARALLEL -DNETCDF -DLANDREAD_STUB=1 -DUSE_ALLOCATABLES -Dwrfmodel -DGRIB1 -DINTIO -DKEEP_INT_AROUND -DLIMIT_ARGS -DBUILD_RRTMG_FAST=0 -DBUILD_RRTMK=0 -DBUILD_SBM_FAST=1 -DSHOW_ALL_VARS_USED=0 -DCONFIG_BUF_LEN=65536 -DMAX_DOMAINS_F=21 -DMAX_HISTORY=25 -DNMM_NEST=0 -I. -traditional-cpp"
mpif90 -f90=gfortran -o module_cap_io.o -c $FFLAGS $INC module_cap_io.f90
python3 patch_drivers.py "$W/phys/module_pbl_driver.F" "$W/phys/module_surface_driver.F" module_pbl_driver.F module_surface_driver.F
for m in module_pbl_driver module_surface_driver; do
  sed -e 's/^\!.*'"'"'.*//' -e 's/^ *\!.*'"'"'.*//' $m.F > $m.G
  /lib/cpp $CPPDEFS $m.G > $m.bb
  "$W/tools/standard.exe" $m.bb | /lib/cpp -P -nostdinc -traditional-cpp -DUSE_NETCDF4_FEATURES -DWRFIO_NCD_LARGE_FILE_SUPPORT > $m.f90
  mpif90 -f90=gfortran -o $m.o -c $FFLAGS $INC $m.f90
done
cp "$W/main/libwrflib.a" libwrflib.a
ar r libwrflib.a module_cap_io.o module_pbl_driver.o module_surface_driver.o
mpif90 -f90=gfortran -o wrf.exe $FFLAGS "$W/main/wrf.o" "$W/main/module_wrf_top.o" libwrflib.a \
  "$W/external/fftpack/fftpack5/libfftpack.a" "$W/external/io_grib1/libio_grib1.a" \
  "$W/external/io_grib_share/libio_grib_share.a" "$W/external/io_int/libwrfio_int.a" \
  -L"$W/external/esmf_time_f90" -lesmf_time "$W/external/RSL_LITE/librsl_lite.a" \
  "$W/frame/module_internal_header_util.o" "$W/frame/pack_utils.o" -L"$W/external/io_netcdf" -lwrfio_nf \
  -L/usr/lib -L/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lnetcdf \
  -lnetcdff -Wl,-Bsymbolic-functions -flto=auto -ffat-lto-objects -Wl,-z,relro -Wl,-z,now -lnetcdf -lm
sha256sum wrf.exe module_pbl_driver.F module_surface_driver.F module_cap_io.f90 > build.sha256
echo BUILD_OK
