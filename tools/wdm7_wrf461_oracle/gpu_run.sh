#!/usr/bin/env bash
# One input, explicit output. Caller selects strict numerics and GPU reservation.
set -euo pipefail
[[ $# == 3 ]] || { echo "usage: gpu_run.sh TREE INPUT OUTPUT" >&2; exit 2; }
PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}" python3 "$1/tools/wdm7_wrf461_oracle/run_woof.py" "$2" "$3"
