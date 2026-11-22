#!/usr/bin/env bash
# Make REF, the folder rebuild.sh reads, from the public upstream repositories at the pins in SOURCE-PINS.json
# beside this script. hex ships no RRFS-MPAS or MPAS source: every file in REF is fetched by you from its upstream.
#
# Usage: make_ref.sh REF        (REF new or empty; needs git and network access to github.com)
#
# REF afterwards holds:
#   REF/SOURCE-PINS.json     the pins: RRFSx/MPAS-Model 44c0849, its nine submodules, MPAS-Dev/MPAS-Data
#   REF/src/MPAS-Model       RRFSx/MPAS-Model at 44c0849, each submodule cloned at its pin in its own path
#   REF/MPAS-Data            MPAS-Dev/MPAS-Data at its pin (the WRF-physics data files the build installs)
set -euo pipefail
REF=$(realpath -m "${1:?usage: make_ref.sh REF}")
HERE=$(dirname "$(realpath "$0")")
if [ -e "$REF" ] && [ -n "$(ls -A "$REF")" ]; then
    echo "make_ref.sh: $REF is not empty; give a new folder" >&2
    exit 2
fi
mkdir -p "$REF/src"
python3 - "$HERE/SOURCE-PINS.json" "$REF" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

pins = json.loads(Path(sys.argv[1]).read_text())
ref = Path(sys.argv[2])


def fetch(url, target, commit):
    subprocess.run(['git', 'clone', '--quiet', '--no-checkout', url, str(target)], check=True, timeout=3600)
    subprocess.run(['git', '-C', str(target), 'checkout', '--quiet', '--detach', commit], check=True, timeout=600)
    head = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
    if head != commit:
        raise SystemExit(f'make_ref.sh: {url} checked out {head}, not the pin {commit}')
    print(f'{commit[:9]}  {url}', flush=True)


model = ref / 'src/MPAS-Model'
fetch(pins['model_url'], model, pins['model_commit'])
for row in pins['submodules']:
    fetch(row['url'], model / row['path'], row['commit'])
fetch(pins['mpas_data_url'], ref / 'MPAS-Data', pins['mpas_data_commit'])
(ref / 'SOURCE-PINS.json').write_text(json.dumps(pins, indent=2) + '\n')
print(f'REF ready: {ref}')
PY
