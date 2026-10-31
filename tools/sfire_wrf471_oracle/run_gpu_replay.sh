#!/usr/bin/env bash
# Short node-only replay. This script claims and releases the card in OWNER.
set -euo pipefail
scratch=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mutex="$HOME/card-lease"
python_path=${SFIRE_PYTHON:-$HOME/v274venv/bin/python}
lane=sol-sfire/oracles
done_file="$scratch/gpu-replay.done"
if [[ -f "$done_file" ]]; then
    echo 'Existing completion marker must be preserved or removed by its creator' >&2
    exit 2
fi
exec 8>"$mutex/OWNER.lock"
flock 8
python3 - "$mutex/OWNER" <<'PY'
import os,sys,datetime
active={}
for line in open(sys.argv[1]):
    p=line.split()
    if len(p)<3: continue
    if p[2]=='start': active[p[0]]=p
    elif p[2]=='release': active.pop(p[0],None)
for p in active.values():
    try:
        pid=int(p[p.index('pid')+1]); os.kill(pid,0)
    except (ValueError,ProcessLookupError): continue
    print('Active OWNER hold: '+' '.join(p),file=sys.stderr)
    raise SystemExit(3)
PY
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo 'An active CUDA process is using the card' >&2
    exit 3
fi
printf '%s %s start pid %s bounded 4 min (SFIRE native routine GPU parity; shared)\n' "$lane" "$(date -u +%FT%TZ)" "$$" >> "$mutex/OWNER"
flock -u 8
release() {
    rc=$?
    flock 8
    printf '%s %s release rc %s\n' "$lane" "$(date -u +%FT%TZ)" "$rc" >> "$mutex/OWNER"
    flock -u 8
    printf '%s\n' "$rc" > "$done_file"
}
trap release EXIT
cd "$scratch/gpu-replay"
export CUDA_VISIBLE_DEVICES=0 GPUWM_NO_LOCAL_GPU=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -n "${SFIRE_CUDA_PATH:-}" ]]; then
    export CUDA_PATH="$SFIRE_CUDA_PATH"
    export LD_LIBRARY_PATH="$SFIRE_CUDA_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
nvidia-smi --query-gpu=name,driver_version --format=csv
"$python_path" -c 'import cupy; from cupy_backends.cuda.libs import nvrtc; print("CuPy",cupy.__version__,"NVRTC",nvrtc.getVersion())'
if [[ "${SFIRE_TG_PROBE:-0}" == 1 ]]; then
    nice -n 10 "$python_path" -m tools.sfire_wrf471_oracle.probe_tg_math tools/sfire_wrf471_oracle/fixtures/math_tg.npz "$scratch/probe-tg-actual.npz"
else
    nice -n 10 "$python_path" -m pytest tests/test_sfire_wrf471_parity.py tests/test_sfire_atm_wrf471_parity.py ${SFIRE_EXTRA_TESTS:-} -v -s
fi
