#!/usr/bin/env bash
set -euo pipefail
src=$(realpath "$1"); out=$(realpath -m "$2"); here=$(cd "$(dirname "$0")" && pwd)
pin=d66e442fccc04111067e29274c9f9eaccc3cef28
test "$(git -C "$src" rev-parse HEAD)" = "$pin"
git -C "$src" diff --quiet HEAD -- phys/module_ra_hs.F dyn_em/module_damping_em.F
mkdir -p "$out"; cd "$out"
flags=(-O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none)
gfortran "${flags[@]}" -c "$here/stub_wrf_error.F90" "$src/phys/module_ra_hs.F" "$src/dyn_em/module_damping_em.F"
gfortran "${flags[@]}" -o run_columns "$here/run_columns.F90" module_ra_hs.o module_damping_em.o
./run_columns
nm -u module_ra_hs.o > lowering-symbols.txt
objdump -dr module_ra_hs.o > lowering-assembly.txt
gfortran "${flags[@]}" -fdump-tree-original -c "$src/phys/module_ra_hs.F" -o lowering.o
sha256sum "$src/phys/module_ra_hs.F" "$src/dyn_em/module_damping_em.F" columns.hex > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
printf '%s\n' "$pin" > wrf-commit.txt
