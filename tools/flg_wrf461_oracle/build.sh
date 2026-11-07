#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: build.sh WRF_SOURCE_ROOT BUILD_DIR' >&2; exit 2; }
wrf=$(realpath "$1"); out=$(realpath -m "$2"); here=$(cd "$(dirname "$0")" && pwd)
pin=d66e442fccc04111067e29274c9f9eaccc3cef28
[[ $(git -C "$wrf" rev-parse HEAD) == "$pin" ]] || { echo 'wrong WRF revision' >&2; exit 3; }
git -C "$wrf" diff --quiet HEAD -- phys/module_ra_flg.F || { echo 'edited FLG source' >&2; exit 3; }
mkdir -p "$out"; cd "$out"
flags=(-O0 -fPIC -cpp -ffree-form -ffree-line-length-none -fno-tree-vectorize -ffp-contract=off)
printf '%s\n' "gfortran ${flags[*]}" > compiler-flags.txt
gfortran "${flags[@]}" -c "$wrf/phys/module_ra_flg.F" -o module_ra_flg.o
gfortran "${flags[@]}" -c "$here/stub_wrf_error.F90" "$here/run_flg.F90"
gfortran -o run_flg module_ra_flg.o stub_wrf_error.o run_flg.o
gfortran -shared -o libflg_oracle.so module_ra_flg.o stub_wrf_error.o
mkdir -p baseline repeat mutant
./run_flg baseline
./run_flg repeat
./run_flg mutant flip-coefficient
for file in columns.csv driver.csv inputs.csv cases.csv; do cmp "baseline/$file" "repeat/$file"; done
sha256sum "$wrf/phys/module_ra_flg.F" "$here/run_flg.F90" "$here/build.sh" baseline/*.csv mutant/*.csv > sha256sums.txt
gfortran --version | head -1 > compiler.txt
printf '%s\n' "$pin" > wrf-commit.txt
python "$here/check.py" --reference baseline --candidate repeat --out oracle-repeat.json
if python "$here/check.py" --reference baseline --candidate mutant --out coefficient-mutation.json; then
    echo 'one-bit coefficient mutation was not caught' >&2; exit 4
fi
python - <<'CHECK'
import json
result=json.load(open('coefficient-mutation.json'))
assert result['status']=='fail' and any(row['bit_misses'] for row in result['fields'].values())
CHECK
python "$here/extract_tables.py" "$wrf/phys/module_ra_flg.F" extract_flg_tables.F90
gfortran "${flags[@]}" -o extract_flg_tables extract_flg_tables.F90 module_ra_flg.o stub_wrf_error.o
./extract_flg_tables
python "$here/package_tables.py" "$out" "$wrf/phys/module_ra_flg.F" "$out/table-rebuild"
echo 'FLG oracle repeat passed and one-bit coefficient mutation was caught'
