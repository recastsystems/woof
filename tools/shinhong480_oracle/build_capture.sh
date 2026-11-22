#!/usr/bin/env bash
# Build the Shin-Hong capture COPY of the 4.8.0 strict serial tfix build (capture
# only, never a referee) and check its Shin-Hong objects are the referee's.
#   build_capture.sh PORT_ROOT          (on a development machine, outside the container)
set -euo pipefail
P=$1
here=$(cd "$(dirname "$0")" && pwd)
cd "$P"
S=build/WRF-4.8.0-serial-tfix
B=build/WRF-4.8.0-serial-tfix-capL09
rm -rf "$B"; cp -a "$S" "$B"
python3 "$here/instrument_pbl_driver.py" "$B"
podman run --rm --name wrf480-capL09 -v "$P:/w" -w /w/$B -e NETCDF=/w/build/netcdf-serial \
  -e WRFIO_NCD_LARGE_FILE_SUPPORT=1 localhost/wrfref2404:1 \
  bash -c 'timeout 3600 ./compile -j 8 em_real > compile-capL09.log 2>&1; ls -la main/wrf.exe'
rm -f $B/main/real.exe $B/main/ndown.exe $B/main/tc.exe $B/run/real.exe $B/run/ndown.exe $B/run/tc.exe
for o in phys/physics_mmm/bl_shinhong.o phys/module_bl_shinhong.o; do
  a=$(sha256sum "build/WRF-4.8.0-serial-pristine/$o" | cut -d' ' -f1)
  b=$(sha256sum "$B/$o" | cut -d' ' -f1)
  echo "$o referee=$a capture=$b"
  test "$a" = "$b"
done
sha256sum $B/main/wrf.exe
echo "$(date -u +%FT%TZ) capL09 built"
