#!/usr/bin/env bash
# Run compare.py on every fixture directory given, under the strict build
# (GPUWM_WRF_EXACT=1) and default arithmetic, writing RESULTS/<dir>-exact<E>.json
# and RESULTS/<dir>-exact<E>.txt.  Run it inside a GPU grant, from a tree whose
# woof is on PYTHONPATH:
#
#   card-lease LANE --cards 1 --est 1500 --wait 1800 \
#       bash tools/urban_slucm_regime_oracle/run_gpu_compare.sh RESULTS DIR...
#
# CUPY_CACHE_DIR is split per arithmetic (CUPY_CACHE_BASE-e1 / -e0) so a
# default-arithmetic image is never served to a strict run.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
results=$1; shift
mkdir -p "$results"
base=${CUPY_CACHE_BASE:-${CUPY_CACHE_DIR:-$HOME/.cupy/kernel_cache}}
for e in 1 0; do
    for d in "$@"; do
        tag=$(basename "$d")
        echo "== GPUWM_WRF_EXACT=$e $tag"
        GPUWM_WRF_EXACT=$e CUPY_CACHE_DIR=$base-e$e \
            python "$here/compare.py" "$d" "$results/$tag-exact$e.json" \
            > "$results/$tag-exact$e.txt" 2>&1
        echo "rc=$?"
        cat "$results/$tag-exact$e.txt"
    done
done
