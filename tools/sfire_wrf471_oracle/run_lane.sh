#!/usr/bin/env bash
# One persistent remote command. Run on G1, never on the desktop.
set -euo pipefail
lane=$(realpath "$1")
T="$lane/src"
source /work/env.sh
mkdir -p "$lane/tmp"
export TMPDIR="$lane/tmp" PYTHONDONTWRITEBYTECODE=1
cd "$T"
# Only this task creates files beneath this dedicated root. Record the full
# inventory before and after the run, without following any link.
manifest() { find -P "$lane" -print > "$lane/MANIFEST.txt"; }
trap manifest EXIT
manifest
CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 \
    bash tools/sfire_wrf471_oracle/build_regimes.sh "$lane/WRF" "$lane" > "$lane/build.log" 2>&1
CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 \
    python -m pytest -q tests/test_sfire_regime_receipt.py -m 'not gpu' > "$lane/cpu.log" 2>&1
cat "$lane/cpu.log"
card-lease pl-sfire-oracle-verdict --cards 1 --min-cards 1 --est 180 --wait 7200 -- \
    bash tools/sfire_wrf471_oracle/run_regime_check.sh "$lane" > "$lane/gpu.log" 2>&1
cat "$lane/gpu.log"
