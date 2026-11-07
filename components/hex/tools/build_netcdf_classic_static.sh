#!/usr/bin/env bash
# Build a static, classic-format-only netCDF-C for gpuwm-hex-netcdf-copy and
# rw_mpas_physics_pool (the MPAS route's Registry pool reader; native MPAS
# reads its streams through PnetCDF, classic CDF-1/2/5 only, so the reader
# needs no netCDF-4/HDF5 either).
#
#   tools/build_netcdf_classic_static.sh PREFIX
#
# Then build the wheel with LIBRARY_PATH=PREFIX/lib so both link
# PREFIX/lib/libnetcdf.a and needs nothing beyond libc at run time.
#
# THE BREAKAGE THIS PREVENTS (lane H7, 2026-10-14): linked against the
# distribution's shared libnetcdf, the copier made auditwheel graft 31 shared
# libraries into the wheel (HDF5, libcurl, OpenSSL, Kerberos, OpenLDAP,
# libssh, libidn2, libunistring ...), each a third-party binary with its own
# licence terms riding in a public wheel unannounced, for a tool that copies
# classic CDF-1/2 templates into CDF-5 and uses none of them.  The vertical
# door only hands the copier classic files (vertical_spec checks the magic).
#
# The source is the official Unidata release tarball, pinned by SHA-256.  Its
# copyright notice ships in the wheel as woof/hex/data/licenses/netcdf-c-COPYRIGHT.
set -euo pipefail
prefix=${1:?PREFIX}
version=4.9.3
sha256=a474149844e6144566673facf097fea253dc843c37bc0a7d3de047dc8adda5dd
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
curl -fsSL -o "netcdf-c-$version.tar.gz" "https://downloads.unidata.ucar.edu/netcdf-c/$version/netcdf-c-$version.tar.gz"
echo "$sha256  netcdf-c-$version.tar.gz" | sha256sum -c -
tar -xzf "netcdf-c-$version.tar.gz"
cd "netcdf-c-$version"
./configure --prefix="$prefix" --disable-shared --enable-static --with-pic \
  --disable-netcdf-4 --disable-hdf5 --disable-dap --disable-byterange --disable-libxml2 \
  --disable-nczarr --disable-utilities --disable-testsets --disable-examples \
  --disable-doxygen --disable-plugins --with-plugin-dir=/usr/local/hdf5/lib/plugin >configure.log 2>&1 || { tail -40 configure.log; exit 1; }
make -j"${CARGO_BUILD_JOBS:-${MAX_JOBS:-8}}" >make.log 2>&1 || { tail -40 make.log; exit 1; }
make install >install.log 2>&1 || { tail -40 install.log; exit 1; }
test -f "$prefix/lib/libnetcdf.a"
echo "static classic netCDF-C $version in $prefix/lib/libnetcdf.a"
