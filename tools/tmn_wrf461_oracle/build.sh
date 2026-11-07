#!/usr/bin/env bash
set -euo pipefail
source_root=$(realpath "$1")
out=$(realpath -m "$2")
expected=d66e442fccc04111067e29274c9f9eaccc3cef28
test "$(git -C "$source_root" rev-parse HEAD)" = "$expected"
# Pin the v4.6.1 tag, and refuse local changes to the source we compile.
test "$(git -C "$source_root" describe --tags --exact-match HEAD)" = v4.6.1
git -C "$source_root" diff --exit-code HEAD -- phys/module_sf_tmnupdate.F
mkdir -p "$out"
gfortran -cpp -ffree-form -ffree-line-length-none -O2 -fno-tree-vectorize -ffp-contract=off -fPIC -shared \
  -J "$out" "$source_root/phys/module_sf_tmnupdate.F" \
  "$(dirname "$0")/wrapper.F90" -o "$out/libtmn.so"
sha256sum "$source_root/phys/module_sf_tmnupdate.F" "$out/libtmn.so"
gfortran --version | head -1
