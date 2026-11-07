#!/usr/bin/env bash
# One bounded queue hold per lane, with strict and default subprocesses.
set -u
export L=/work/pool-verify-lanes-stack
export T="$L/src"
source /work/env.sh
export PYTHONPATH="$T:$T/tests:$T/recast-woof-data"
export TMPDIR="$L/tmp"
source "$L/column-env.sh"
cd "$T" || exit 1
mkdir -p "$L/gpu-gates"
printf '%s\n' "$L/gpu-gates" "$L/gpu-gates.log" >> "$L/MANIFEST.txt"
while IFS= read -r lane; do
    short=${lane##*/}
    out="$L/gpu-gates/$short"
    printf '%s\n' "$out" >> "$L/MANIFEST.txt"
    echo "START $lane $(date -u +%FT%TZ)"
    ${GPU_MUTEX_DIR:?set GPU_MUTEX_DIR}/run.sh pl-verify-lanes-stack --cards 1 --min-cards 1 \
        --est 1800 --wait 1800 -- \
        python tools/verify_lanes_stack.py --phase gpu --lane "$lane" --out "$out" \
        --gpu-cache "$L/gpu-gates/cache.json"
    status=$?
    echo "END $lane exit=$status $(date -u +%FT%TZ)"
    if ((status != 0)); then
        # Stop at the first concrete gate failure. Repair it before more cards.
        exit "$status"
    fi
done < <(python - <<'PY'
import json
from pathlib import Path
plan = json.loads(Path('tools/verify_lanes_stack.json').read_text(encoding='utf-8-sig'))
for lane in plan['lanes']:
    if lane['take'] and lane.get('gpu_tests'):
        print(lane['branch'])
PY
)
