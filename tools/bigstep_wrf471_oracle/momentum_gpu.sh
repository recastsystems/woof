#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
    echo 'usage: momentum_gpu.sh SOURCE_DIR PYTHON RECEIPT_DIR OWNER OWNER_LOCK' >&2
    exit 2
fi
task_source=$(realpath "$1")
task_python=$2
task_receipts=$(realpath -m "$3")
owner=$(realpath "$4")
mutex=$(realpath -m "$5")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$task_receipts"
cd "$task_source"
flock "$mutex" sh -c 'printf "%s %s start pid %s bounded 3 min (compiled WRF momentum oracle; shared)\n" "oracle-bigstep-momentum" "$(date -u +%FT%TZ)" "'"$$"'" >> "'"$owner"'"'
release() {
    flock "$mutex" sh -c 'printf "%s %s release\n" "oracle-bigstep-momentum" "$(date -u +%FT%TZ)" >> "'"$owner"'"'
    touch "$task_receipts/gpu.done"
}
trap release EXIT
PYTHONPATH=. nice -n 10 "$task_python" "$script_dir/momentum_measure.py" tests/data/wrf471_bigstep "$task_receipts/momentum-measurements.json"
