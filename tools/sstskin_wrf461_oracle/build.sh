#!/usr/bin/env bash
set -euo pipefail
src=$(realpath "$1")
out=$(realpath -m "$2")
scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pin=214328c30a6f542aed0e6e77eeeea4152174738977e7b6460eff3a7001fb9630
test "$(sha256sum "$src/phys/module_sf_sstskin.F" | cut -d' ' -f1)" = "$pin"
mkdir -p "$out"
cd "$out"
printf 'module module_wrf_error
end module
' > stub.F90
gfortran -c -O0 -ffp-contract=off -fno-tree-vectorize stub.F90
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none -ffp-contract=off -fno-tree-vectorize "$src/phys/module_sf_sstskin.F"
gfortran -O0 -ffp-contract=off -fno-tree-vectorize -ffree-line-length-none "$scripts/run.F90" module_sf_sstskin.o -o oracle
nm -u module_sf_sstskin.o > symbols.txt
! grep -q _ZGV symbols.txt
sha256sum "$src/phys/module_sf_sstskin.F" oracle > hashes.txt
gfortran --version | head -1 > compiler.txt
