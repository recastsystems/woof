#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: run_suite.sh WRF_SOURCE_ROOT BUILD_DIR' >&2; exit 2; }
S=$(cd "$(dirname "$0")" && pwd)
bash "$S/build.sh" "$1" "$2"
cd "$(realpath "$2")"
for n in 12 24 42 60; do
 ./run_gfdleta "$n" 1 "n$n-a1" > "n$n-a1.log" 2>&1
 ./run_gfdleta "$n" 0 "n$n-a0" > "n$n-a0.log" 2>&1
 cmp "n$n-a1-tables.csv" "n$n-a0-tables.csv"
 cmp "n$n-a1-fields.csv" "n$n-a0-fields.csv"
done
./run_primitives > primitives.log 2>&1
./run_lw_helpers > lw-helpers.log 2>&1
mkdir -p stock-co2
ln -sfn "$(realpath "$1")/run/co2_trans" stock-co2/co2_trans
(cd stock-co2 && ../run_gfdleta 42 0 stock > stock.log 2>&1)
PYTHONPATH="$S${PYTHONPATH:+:$PYTHONPATH}" python -c 'from validate_gfdleta import check_coverage; import json; from pathlib import Path; Path("coverage.json").write_text(json.dumps(check_coverage("."),indent=2)+"\n")'
sha256sum *-fields.csv *-tables.csv primitives-*.csv lw-*.csv stock-co2/*.csv > oracle-output-sha256.txt
