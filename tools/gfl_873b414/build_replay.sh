#!/bin/bash
# Build the native GFL replay against the WRF v4.8.0 strict referee's own objects (inside wrfref2404:1).
set -e
B=/w/build/WRF-4.8.0-serial-pristine/phys
O=/l/native-replay; mkdir -p $O; cd $O
F="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w -ffree-form -ffree-line-length-none"
gfortran $F -I$B -c /l/native-tools/gfl_replay_native.F90 -o gfl_replay_native.o
OBJS="$B/ccpp_kind_types.o $B/module_cu_gfl_common.o $B/module_cu_gfl_deep.o $B/module_cu_gfl_sh.o $B/module_cu_gfl_wrf.o"
gfortran gfl_replay_native.o $OBJS -o gfl_replay_native_glibc
gcc -O2 -c /l/native-tools/tgammaf_cr.c -o tgammaf_cr.o
gfortran gfl_replay_native.o tgammaf_cr.o $OBJS -lquadmath -o gfl_replay_native_crgamma
sha256sum $OBJS gfl_replay_native_glibc gfl_replay_native_crgamma > SHA256SUMS
gfortran --version | head -1 >> SHA256SUMS; ldd --version | head -1 >> SHA256SUMS
nm gfl_replay_native_crgamma | grep -i " T tgammaf"; nm -D gfl_replay_native_glibc | grep -i tgammaf
