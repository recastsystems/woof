#!/usr/bin/env bash
set -euo pipefail
root=$(realpath "$1"); out=$(realpath -m "$2"); here=$(cd "$(dirname "$0")" && pwd)
pin=d66e442fccc04111067e29274c9f9eaccc3cef28
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
git -C "$root" diff --quiet HEAD -- phys/module_irrigation.F
mkdir -p "$out"
cd "$out"
gfortran -O0 -fno-tree-vectorize -ffp-contract=off -ffree-form -ffree-line-length-none "$root/phys/module_irrigation.F" "$here/run_irrigation.F90" -o run_irrigation
./run_irrigation > columns.txt
sha256sum "$root/phys/module_irrigation.F" "$here/run_irrigation.F90" "$here/build.sh" run_irrigation columns.txt > SHA256SUMS
gfortran --version | head -1 > compiler.txt
