#!/usr/bin/env bash
# Run one captured or diagnostic input, with explicit output paths.
set -euo pipefail
[[ $# == 3 ]] || { echo "usage: run_oracle.sh BUILD_DIR INPUT OUTPUT" >&2; exit 2; }
"$1/run_wdm7_stock" "$2" "$3" "$3.constants.txt"
