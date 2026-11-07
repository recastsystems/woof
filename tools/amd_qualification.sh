#!/usr/bin/env bash
# Pending physical rerun. Run inside the chosen ROCm 7.x environment.
set -euo pipefail
if [[ $# != 5 ]]; then
    echo 'usage: bash tools/amd_qualification.sh REGIONAL.toml HRRR.toml HEX.grid.nc HEX.init.nc OUT' >&2
    exit 2
fi
case_config=$(realpath "$1")
hrrr_config=$(realpath "$2")
hex_grid=$(realpath "$3")
hex_init=$(realpath "$4")
mkdir -p "$5"
qual_out=$(realpath "$5")
export WOOF_BACKEND=amd WOOF_RUST_BACKEND=hip
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4 WOOF_HEX_HOST_COPY_THREADS=8
rocm-smi --showproductname --showmeminfo vram --json > "$qual_out/card-before.json"
woof doctor --backend amd --json > "$qual_out/engine-doctor.json"
woof hex doctor --backend amd --json > "$qual_out/hex-doctor.json"
for mode in strict fast; do
    export GPUWM_NUMERICS="$mode" WOOF_HEX_NUMERICS="$mode"
    woof go "$case_config" --backend amd --outdir "$qual_out/regional-$mode" --products 2m_temperature > "$qual_out/regional-$mode.log" 2>&1
    woof go "$hrrr_config" --backend amd --outdir "$qual_out/hrrr-$mode" --products 2m_temperature > "$qual_out/hrrr-$mode.log" 2>&1
    woof hex forecast --backend amd --mesh x1.2562 --grid "$hex_grid" --init "$hex_init" --hours .2 --dt 120 --history-every-minutes 2 --checkpoint-hours .2 --no-products --out "$qual_out/hex-$mode" > "$qual_out/hex-$mode.log" 2>&1
done
python - "$qual_out" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]);rows=[]
for p in sorted(root.rglob('*')):
    if p.is_file() and (p.suffix in ('.nc','.pkl') or p.name.startswith('wrfout')
                       or p.name in ('certification-capsule.json','carried-identity.json','boundary-fingerprints.jsonl')):
        rows.append(dict(path=str(p.relative_to(root)),size=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()))
(root/'output-hashes.json').write_text(json.dumps(rows,indent=2)+'\n')
PY
