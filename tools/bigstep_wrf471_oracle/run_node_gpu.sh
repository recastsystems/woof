#!/usr/bin/env bash
# NODE_PYTHON SOURCE_ROOT DONE_LOG COMMAND [ARG...]
# Small shared oracle jobs refuse a live exclusive OWNER claim.
set -euo pipefail
python_bin=$1; source_root=$2; log_prefix=$3; shift 3
owner="$HOME/card-lease/OWNER"
lane=sol-oracle-bigstep-prep
claimed=0
release() {
    rc=$?
    if (( claimed )); then
        (flock -x 9; printf '%s %s release rc %s\n' "$lane" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >> "$owner") 9> "${owner}.lock"
    fi
    printf '%s\n' "$rc" > "${log_prefix}.done"
}
trap release EXIT
export PYTHONPATH="$source_root" PYTHONDONTWRITEBYTECODE=1
export CUPY_CACHE_DIR="${log_prefix}.cupy-cache" CUDA_CACHE_PATH="${log_prefix}.cuda-cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9> "${owner}.lock"
flock -x 9
"$python_bin" - "$owner" <<'PY'
import os,re,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
latest={}
for line in Path(sys.argv[1]).read_text().splitlines():
    words=line.split()
    if len(words)>2 and words[2] in ('start','release'): latest[words[0]]=line
for line in latest.values():
    if ' start pid ' not in line or 'exclusive' not in line.lower(): continue
    m=re.search(r'^\S+ (\S+) start pid (\d+) bounded (\d+) min',line)
    if not m: raise SystemExit('Unparseable exclusive OWNER record')
    alive=True
    try: os.kill(int(m[2]),0)
    except ProcessLookupError: alive=False
    if alive or datetime.now(timezone.utc)<datetime.fromisoformat(m[1].replace('Z','+00:00'))+timedelta(minutes=int(m[3])):
        raise SystemExit('Oracle is queued behind a live exclusive GPU OWNER record')
PY
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
(( free >= 2048 ))
printf '%s %s start pid %s bounded 5 min (compiled WRF oracle; shared; below 1 GiB)\n' "$lane" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" >> "$owner"
claimed=1
flock -u 9
exec 9>&-
timeout 4m nice -n 10 "$python_bin" "$@"
