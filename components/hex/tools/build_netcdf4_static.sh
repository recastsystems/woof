#!/usr/bin/env bash
# Build a static netCDF-C WITH netCDF-4 (HDF5 + zlib) for rw_mpas_stream.
#
#   tools/build_netcdf4_static.sh PREFIX
#
# Writes PREFIX/lib/libnetcdf.a as ONE archive holding netCDF-C, HDF5 (with its
# high-level library, which netCDF-4's dimension scales use) and zlib, so a
# crate that declares #[link(name = "netcdf")] links it with LIBRARY_PATH=PREFIX/lib
# and nothing beyond glibc at run time.
#
# THE BREAKAGE THIS PREVENTS (lane H7, measured 2026-10-14; release readiness
# B5): rw_mpas_stream writes MPAS streams in every io_type MPAS names, and
# woof.hex.output's history default is io_type netcdf4 (HDF5).  Linked against a
# distribution's shared libnetcdf it needs libnetcdf.so.15, which Debian 12 does
# not have, and the classic-only static library the CDF-5 copier uses
# (tools/build_netcdf_classic_static.sh) cannot write netCDF-4 at all.
#
# Sources are the upstream release tarballs, pinned by SHA-256.  DAP, byte-range,
# NCZarr, S3, szip and plugins are off: the bridge reads and writes local files.
# The licence notices ship in the wheel under woof/hex/data/licenses/.
#
# The build is reproducible, so a rebuild gives the bytes
# src/hexcore/data/engine-bridges.json pins.  THE BREAKAGE THIS PREVENTS
# (measured 2026-10-14, two builds of woof 24b6f08c): rw_mpas_stream
# came out different every time, and carried the build container's name.  HDF5
# copies its configure summary into the library (date, user@host, uname, and
# the temporary build directory in its paths), and netCDF-C stamps
# __DATE__/__TIME__ into its version string.  So the build runs in one fixed
# directory, SOURCE_DATE_EPOCH pins the compiler's date, and HDF5's templates
# get fixed text for the three host-describing lines of that summary.
set -euo pipefail
prefix=${1:?PREFIX}
zlib_version=1.3.1
zlib_sha256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
hdf5_version=1.14.6
hdf5_sha256=e4defbac30f50d64e1556374aa49e574417c9e72c6b1de7a4ff88c4b1bea6e9b
netcdf_version=4.9.3
netcdf_sha256=a474149844e6144566673facf097fea253dc843c37bc0a7d3de047dc8adda5dd
work="$prefix/build-work"
rm -rf "$work" && mkdir -p "$work"
trap 'rm -rf "$work"' EXIT
# A fixed date (2025-02-07 00:00:00 UTC); GCC uses it for __DATE__ and __TIME__.
export SOURCE_DATE_EPOCH=1738886400
deps="$work/deps"
jobs=${CARGO_BUILD_JOBS:-${MAX_JOBS:-8}}
cd "$work"
# Each source comes from the first URL whose bytes are the pinned SHA-256; a URL is tried three times.
# THE BREAKAGE THIS PREVENTS (the release runs of 0.6.0 and 0.6.1, 2026-11-17 01:31:35Z and 2026-11-19
# 00:57:42Z): zlib.net answered the release build's request with a success status and bytes that were not
# zlib-1.3.1.tar.gz ("zlib.tar.gz: FAILED"), 15 to 20 minutes after the dry run on the same URL got the
# file, and both release runs needed a rerun.  zlib.net is one server, and the old fetch kept neither the
# bytes nor what they were.  The first URL is upstream's own GitHub release asset (the same bytes, served
# from the network the build runs in); zlib.net's fossils folder, which keeps every released version at a
# fixed path, is the second.  A URL that gives other bytes is reported with its size, sum and first bytes,
# and the build stops only when no URL gives the pinned bytes.
fetch() {
  local out=$1 sum=$2 url try got rc
  shift 2
  for url in "$@"; do
    for try in 1 2 3; do
      rc=0
      curl -fsSL -o "$out" "$url" || rc=$?
      if [ "$rc" -eq 0 ]; then
        got=$(sha256sum "$out" | cut -c1-64)
        if [ "$got" = "$sum" ]; then echo "$out: OK ($url)"; return 0; fi
        echo "$out: $url (try $try) gave $(wc -c <"$out") bytes, sha256 $got, not the pinned $sum;" \
             "first bytes: $(head -c 160 "$out" | tr -c '[:print:]' '.')" >&2
      else
        echo "$out: $url (try $try): curl exit $rc" >&2
      fi
      sleep $((5 * try))
    done
  done
  echo "$out: FAILED: no URL gave the pinned sha256 $sum" >&2
  return 1
}
fetch zlib.tar.gz "$zlib_sha256" ${GPUWM_CI_ZLIB_URL:+"$GPUWM_CI_ZLIB_URL"} \
  "https://github.com/madler/zlib/releases/download/v$zlib_version/zlib-$zlib_version.tar.gz" \
  "https://zlib.net/fossils/zlib-$zlib_version.tar.gz"
fetch hdf5.tar.gz "$hdf5_sha256" "https://github.com/HDFGroup/hdf5/releases/download/hdf5_$hdf5_version/hdf5-$hdf5_version.tar.gz"
fetch netcdf.tar.gz "$netcdf_sha256" "https://downloads.unidata.ucar.edu/netcdf-c/$netcdf_version/netcdf-c-$netcdf_version.tar.gz"
tar -xzf zlib.tar.gz && tar -xzf hdf5.tar.gz && tar -xzf netcdf.tar.gz
log() { "$@" >"$work/step.log" 2>&1 || { tail -40 "$work/step.log"; exit 1; }; }
(cd "zlib-$zlib_version" && CFLAGS="-O2 -fPIC" log ./configure --static --prefix="$deps" && log make -j"$jobs" && log make install)
# HDF5's configure.ac fills these three from date, whoami@hostname and uname -a,
# and H5build_settings.c compiles the summary into the library.
(cd "hdf5-$hdf5_version" \
   && sed -i -e 's/@CONFIG_DATE@/(reproducible build)/' -e 's/@CONFIG_USER@/(reproducible build)/' \
             -e 's/@UNAME_INFO@/(reproducible build)/' \
             src/H5build_settings.autotools.c.in src/libhdf5.settings.autotools.in \
   && log ./configure --prefix="$deps" --disable-shared --enable-static --with-pic \
   --enable-hl --disable-fortran --disable-cxx --disable-tests --disable-tools --without-szlib \
   --with-zlib="$deps" && log make -j"$jobs" && log make install)
(cd "netcdf-c-$netcdf_version" && CPPFLAGS="-I$deps/include" LDFLAGS="-L$deps/lib" LIBS="-lhdf5_hl -lhdf5 -lz -lm -ldl" \
   log ./configure --prefix="$work/netcdf" --disable-shared --enable-static --with-pic \
   --enable-netcdf-4 --disable-dap --disable-byterange --disable-libxml2 --disable-nczarr --disable-s3 \
   --disable-utilities --disable-testsets --disable-examples --disable-doxygen --disable-plugins \
   --disable-filter-testing && log make -j"$jobs" && log make install)
mkdir -p "$prefix/lib" "$prefix/include"
cp -a "$work/netcdf/include/." "$prefix/include/"
# One archive, so #[link(name = "netcdf")] needs no extra link arguments.
cat >"$work/merge.mri" <<EOF
create $prefix/lib/libnetcdf.a
addlib $work/netcdf/lib/libnetcdf.a
addlib $deps/lib/libhdf5_hl.a
addlib $deps/lib/libhdf5.a
addlib $deps/lib/libz.a
save
end
EOF
ar -M <"$work/merge.mri"
ranlib "$prefix/lib/libnetcdf.a"
echo "static netCDF-C $netcdf_version with netCDF-4 (HDF5 $hdf5_version, zlib $zlib_version) in $prefix/lib/libnetcdf.a"
