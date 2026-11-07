#!/usr/bin/env bash
# Dump the arrays WRF's own p3_init reads from the 3-moment table
# (itab_3mom, itabcoll_3mom) without editing module_mp_p3.F: the array
# dimensions (:48-57) and the TRIPLE_MOMENT_ICE read loop (:446-468) are
# cut out of the pinned source by line number and wrapped in a program
# that opens the table and reads its two header records exactly as
# p3_init does (:416-444).
#   bash table3_dump.sh <WRF_SOURCE_ROOT> <OUT_DIR> [TABLE_FILE]
# Writes OUT_DIR/table3.bin: itab_3mom then itabcoll_3mom, float32,
# Fortran order; TABLE_FILE defaults to WRF's run/ copy.
set -euo pipefail
WRF=$1; OUT=$2
SRC=$WRF/phys/module_mp_p3.F
TAB=${3:-$WRF/run/p3_lookupTable_1.dat-v5.4_3momI}
[ "$(sha256sum "$SRC" | cut -d' ' -f1)" = 716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6 ] || { echo "FATAL: source hash" >&2; exit 1; }
mkdir -p "$OUT"; F=$OUT/table3_dump.f90
{
  echo "program table3_dump"
  echo " implicit none"
  sed -n '48,57p' "$SRC"
  echo " real, dimension(zsize,densize,rimsize,isize,tabsize_3mom) :: itab_3mom"
  echo " real, dimension(zsize,densize,rimsize,isize,rcollsize,colltabsize) :: itabcoll_3mom"
  echo " character(len=1024) :: dumstr, version_header_table_1_3mom"
  echo " integer :: i, j, ii, jj, zz, u"
  echo " real :: dum"
  echo " double precision :: dp_dum1, dp_dum2"
  echo " itabcoll_3mom = 0."
  echo " open(unit=10, file='$TAB', status='old')"
  echo " read(10,*) dumstr,version_header_table_1_3mom"
  echo " read(10,*)"
  sed -n '446,468p' "$SRC"
  echo " close(10)"
  echo " open(newunit=u, file='table3.bin', access='stream', form='unformatted', status='replace')"
  echo " write(u) itab_3mom, itabcoll_3mom"
  echo " close(u)"
  echo "end program table3_dump"
} > "$F"
cd "$OUT"
gfortran -cpp -ffree-form -ffree-line-length-none -O0 -ffp-contract=off "$F" -o table3_dump
./table3_dump
sha256sum table3.bin
