#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2
export CUDA_PATH="$HOME/work/sol-sfire/venv/lib/python3.14/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$CUDA_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 GPUWM_NO_LOCAL_GPU=0
python="$HOME/work/sol-sfire/venv/bin/python"
receipt="tools/sfire_coupled_ideal/geographic_units/gpu.json"
owner="$HOME/card-lease"
rc=1
release() {
  { flock -x 9; printf 'sol-sfire-geographic %s release rc %s\n' "$(date -u +%FT%TZ)" "$rc" >> "$owner/OWNER"; } 9>"$owner/OWNER.lock"
  printf '%s\n' "$rc" > tools/sfire_coupled_ideal/geographic_units/GPU-DONE
}
trap release EXIT
{ flock -x 9; printf 'sol-sfire-geographic %s start pid %s bounded 3 min (native geographic ignition units and constructor wiring; shared)\n' "$(date -u +%FT%TZ)" "$$" >> "$owner/OWNER"; } 9>"$owner/OWNER.lock"
nice -n 10 "$python" -m pytest -q tests/test_sfire_geographic_wrf471_parity.py
nice -n 10 "$python" tools/sfire_coupled_ideal/geographic_units/grade.py "$receipt"
rc=0
