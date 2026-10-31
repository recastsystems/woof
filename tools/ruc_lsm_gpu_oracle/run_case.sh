#!/usr/bin/env bash
# One oracle case end to end on a box with a card behind a card lease:
#   run_case.sh OUTDIR MODE [run_woof.py options...]
# MODE is "strict" (GPUWM_WRF_EXACT=1) or "default" (unset).  Runs the WOOF
# GPU side (writes inputs.bin), builds and runs the WRF side on the CPU, then
# the free-running and replay comparisons.  Needs T (the tree) and
# WRF_SRC (pristine WRF 4.6.1) in the environment, and env.sh sourced.
set -euo pipefail
out=$(realpath -m "$1"); mode=$2; shift 2
case "$mode" in strict|default) ;; *) echo "unknown arithmetic mode: $mode" >&2; exit 2 ;; esac
tool="$T/tools/ruc_lsm_gpu_oracle"
mkdir -p "$out"
exact=""
[ "$mode" = strict ] && exact="GPUWM_WRF_EXACT=1"
gpu() {
    card-lease "${RUC_ORACLE_LANE:-ruc-lsm}" --cards 1 --min-cards 1 --est 120 --wait 3600 \
        env -u GPUWM_WRF_EXACT $exact bash -c "cd $T && python $tool/run_woof.py $out $* "
}
gpu "$@" > "$out/woof.log" 2>&1
bash "$tool/build_wrf.sh" "$WRF_SRC" "$out" > "$out/wrf-build.log" 2>&1
gpu "$@" --replay "$out/outputs-defined.bin" > "$out/woof-replay.log" 2>&1
cd "$out"
CUDA_VISIBLE_DEVICES= python "$tool/compare.py" . --arms defined,o0 | tee compare.txt
CUDA_VISIBLE_DEVICES= python "$tool/compare.py" . --arms defined --woof woof-replay.npz | tee -a compare.txt
