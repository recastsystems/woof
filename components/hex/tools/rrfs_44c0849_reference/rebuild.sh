#!/usr/bin/env bash
# Rebuild using fresh local clones of the exact published public source pins.
set -euo pipefail
REF=$(realpath "$1")
WORK=$(realpath -m "$2")
CORE=$3
HERE=$(dirname "$(realpath "$0")")
mkdir -p "$WORK/input/src"
python3 - "$REF" "$WORK/input" <<'PY'
import json,shutil,subprocess,sys
from pathlib import Path
ref,dst=map(Path,sys.argv[1:])
pins=json.loads((ref/'SOURCE-PINS.json').read_text())
def clone(source,target,commit):
    subprocess.run(['git','clone','--no-hardlinks','--no-checkout',str(source),str(target)],check=True,timeout=180)
    subprocess.run(['git','-C',str(target),'checkout','--detach',commit],check=True,timeout=60)
clone(ref/'src/MPAS-Model',dst/'src/MPAS-Model',pins['model_commit'])
for row in pins['submodules']:
    clone(ref/'src/MPAS-Model'/row['path'],dst/'src/MPAS-Model'/row['path'],row['commit'])
shutil.copy2(ref/'SOURCE-PINS.json',dst/'SOURCE-PINS.json')
PY
bash "$HERE/build.sh" "$WORK/input" "$WORK/build" "$CORE"
