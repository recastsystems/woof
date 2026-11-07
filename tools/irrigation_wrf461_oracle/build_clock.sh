#!/usr/bin/env bash
set -euo pipefail
root=$(realpath "$1"); out=$(realpath -m "$2"); here=$(cd "$(dirname "$0")" && pwd)
test "$(git -C "$root" rev-parse HEAD)" = d66e442fccc04111067e29274c9f9eaccc3cef28
git -C "$root" diff --quiet HEAD -- share/module_date_time.F frame/module_domain.F frame/module_wrf_error.F share/module_model_constants.F
mkdir -p "$out"; cd "$out"
flags=(-O0 -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch)
gfortran "${flags[@]}" -c "$here/../noah_wrf461_oracle/stub_wrf.F90"
gfortran "${flags[@]}" -c "$root/share/module_model_constants.F"
gfortran "${flags[@]}" -c "$root/frame/module_wrf_error.F"
python "$here/extract_clock.py" "$root/share/module_date_time.F" "$out/date_leaves.F90" > extraction-sha256.txt
gfortran "${flags[@]}" -c date_leaves.F90
gfortran "${flags[@]}" -c "$here/run_clock.F90"
gfortran -o run_clock date_leaves.o run_clock.o module_wrf_error.o stub_wrf.o
# Independent processes keep the public getter's saved month table local to
# one date pair. The numerical get_julgmt and split_date_char bodies are intact.
while IFS= read -r line; do printf '%s\n' "$line" | ./run_clock; done < "$here/fixtures/clock-cases.txt" > clock-oracle.txt
sha256sum "$root/share/module_date_time.F" "$root/frame/module_domain.F" "$here/run_clock.F90" clock-oracle.txt > SHA256SUMS
