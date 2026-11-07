#!/usr/bin/env bash
set -euo pipefail
wrf=$(realpath "$1"); out=$(realpath -m "$2"); here=$(cd "$(dirname "$0")" && pwd)
source_file=$wrf/phys/module_mp_ntu.F
expected=b0269298d3c8234164f36266508075f2cacd64809ce9ef5acb6a19ef67ff53ac
actual=$(sha256sum "$source_file" | cut -d' ' -f1)
[[ $actual == "$expected" ]] || { echo "NTU source digest mismatch: $actual" >&2; exit 2; }
mkdir -p "$out"
python "$here/apply_fixes.py" "$source_file" "$out/module_mp_ntu.fixed.F"
bash "$here/build_oracle.sh" "$source_file" "$out/pristine" -fno-tree-vectorize
bash "$here/build_oracle.sh" "$out/module_mp_ntu.fixed.F" "$out/fixed" -fno-tree-vectorize
python "$here/f2cu.py" "$out/module_mp_ntu.fixed.F" --params-out "$out/params.f90"
gfortran -O2 -fno-fast-math -ffp-contract=off -ffree-line-length-none "$out/params.f90" -o "$out/params"
"$out/params" > "$out/params.txt"
python "$here/f2cu.py" "$out/module_mp_ntu.fixed.F" --params-in "$out/params.txt" --out "$out/ntu_gen.cuh"
python "$here/assemble.py" "$here/ntu_rt.cuh" "$out/ntu_gen.cuh" "$here/ntu_driver.cuh" "$source_file" "$here/wrf461-ntu-fixes.patch" "$out/ntu.cu"
sha256sum "$source_file" "$out/module_mp_ntu.fixed.F" "$out/ntu.cu"
