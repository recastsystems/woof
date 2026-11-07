#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); source_file=$(realpath "$1"); out=$(realpath -m "$2"); mkdir -p "$out"
python - "$source_file" "$out/module_mp_ntu.F" <<'PY'
import sys
from pathlib import Path
s=Path(sys.argv[1]).read_text()
for name in ('IBAER','TBLRC','TBLXF'):
 rows=[row for row in s.splitlines() if 'PRIVATE' in row and ':: '+name in row]
 if len(rows)!=1: raise RuntimeError(f'{name}: expected one private declaration')
 s=s.replace(rows[0],rows[0].replace(', PRIVATE',''))
Path(sys.argv[2]).write_text(s)
PY
cd "$out"
gfortran -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -ffree-form -ffree-line-length-none -cpp -DRWORDSIZE=4 -DDWORDSIZE=8 -c "$here/stub_wrf_error.F90" "$out/module_mp_ntu.F" "$here/probe_tables.F90"
gfortran stub_wrf_error.o module_mp_ntu.o probe_tables.o -o probe_tables
for mode in 1 2 3 4; do ./probe_tables "$mode" "tables-${mode}.bin"; done
