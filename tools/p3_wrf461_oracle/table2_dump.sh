#!/usr/bin/env bash
# Dump the ice-ice collection tables WRF's own p3_init loads for nCat > 1
# (itabcolli1/itabcolli2, module_mp_p3.F:71-72, read at :474-519), so the
# WOOF parser of p3_lookupTable_2.dat-v5.3 is checked against the numbers the
# Fortran actually holds rather than against the text it reads.
#
#   bash table2_dump.sh <WRF_SOURCE_ROOT> <OUT_DIR>
#
# The two arrays are private module variables, so a COPY of the pinned
# module_mp_p3.F gets one read-only subroutine inserted before its
# END MODULE line (the module itself in WRF's tree is never edited), compiled
# with arm A's flags.  Writes OUT_DIR/itabcolli.bin: itabcolli1 then
# itabcolli2, each (25,4,5,25,4,5) float32, Fortran order.
set -euo pipefail
WRF=${1:?WRF_SOURCE_ROOT}; OUT=${2:?OUT_DIR}
SRC=$WRF/phys/module_mp_p3.F
TAB=$WRF/run/p3_lookupTable_2.dat-v5.3
[ "$(sha256sum "$SRC" | cut -d' ' -f1)" = 716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6 ] \
  || { echo "FATAL: source hash" >&2; exit 1; }
[ "$(sha256sum "$TAB" | cut -d' ' -f1)" = c9103041ead719cf9e2e215cf846e0ef81b2004f29386156444a079af6b24d97 ] \
  || { echo "FATAL: table 2 hash" >&2; exit 1; }
mkdir -p "$OUT/tables"
cp "$TAB" "$WRF/run/p3_lookupTable_1.dat-v5.4_2momI" "$OUT/tables/"
cd "$OUT"
n=$(grep -n '^ END MODULE microphy_p3' "$SRC" | cut -d: -f1)
{
  head -n $((n - 1)) "$SRC"
  echo " subroutine p3_dump_table2(path)"
  echo "  character(len=*), intent(in) :: path"
  echo "  integer :: u"
  echo "  open(newunit=u, file=path, access='stream', form='unformatted', status='replace')"
  echo "  write(u) itabcolli1, itabcolli2"
  echo "  close(u)"
  echo " end subroutine p3_dump_table2"
  tail -n +"$n" "$SRC"
} > module_mp_p3_dump.F
sed -i 's/^ public :: p3_main, polysvp1, p3_init$/ public :: p3_main, polysvp1, p3_init, p3_dump_table2/' module_mp_p3_dump.F
cat > t2.f90 <<'X'
program t2
  use microphy_p3, only: p3_init, p3_dump_table2
  integer :: stat
  call p3_init('tables', 2, .false., 'WRF', stat, .true.)
  if (stat /= 0) stop 5
  call p3_dump_table2('itabcolli.bin')
end program t2
X
FREE="-cpp -ffree-form -ffree-line-length-none -O0 -ffp-contract=off -fno-unsafe-math-optimizations"
gfortran $FREE -c module_mp_p3_dump.F -o m.o
gfortran $FREE -c t2.f90 -o t2.o
gfortran m.o t2.o -o t2
./t2 > t2.log 2>&1
sha256sum itabcolli.bin
