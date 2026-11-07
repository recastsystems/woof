#!/usr/bin/env bash
# One card, both arithmetic arms over every fixture case, plus the planted
# one-bit negative control.  Call it through card-lease.
#   T=<tree> bash run_gpu.sh <FIXDIR> <OUTDIR>
# Writes <OUTDIR>/{strict,default,plant}/<case>.ref and a receipt.json each.
set -euo pipefail
FIX=$(realpath "${1:?FIXDIR}"); OUT=$(realpath -m "${2:?OUTDIR}")
: "${T:?set T to the tree}"
H=$T/tools/gsfcgce_wrf461_oracle
cd "$T"
mkdir -p "$OUT"
nvidia-smi -i "${GPU_MUTEX_CARDS:-0}" --query-gpu=name,driver_version \
    --format=csv,noheader > "$OUT/device.txt" 2>/dev/null || true
env GPUWM_WRF_EXACT=1 CUPY_CACHE_DIR="$OUT/cache-strict" \
    python "$H/run_woof.py" "$FIX" "$OUT/strict" > "$OUT/strict.log" 2>&1
echo "strict done"
env -u GPUWM_WRF_EXACT CUPY_CACHE_DIR="$OUT/cache-default" \
    python "$H/run_woof.py" "$FIX" "$OUT/default" > "$OUT/default.log" 2>&1
echo "default done"
env GPUWM_WRF_EXACT=1 CUPY_CACHE_DIR="$OUT/cache-plant" \
    python "$H/run_woof.py" "$FIX" "$OUT/plant" --plant \
    > "$OUT/plant.log" 2>&1
echo "plant done"
