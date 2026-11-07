#!/usr/bin/env bash
# Run on the referee host. Only writes the supplied private source copy.
set -euo pipefail
cd "$1"
python3 - <<'PY'
from pathlib import Path
import re
p=Path('configure.wrf'); s=p.read_text()
changes={
 'DM_FC':'mpif90', 'DM_CC':'mpicc', 'FC':'mpif90',
 'FCOPTIM':'-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize',
 'NETCDF4_DEP_LIB':'-L/usr/lib/x86_64-linux-gnu -lnetcdff -lnetcdf -lm',
 'CFLAGS_LOCAL':'-w -O2 -std=gnu17 -c',
}
for key,value in changes.items():
 s,n=re.subn(r'^'+key+r'\s*=.*$',key+' = '+value,s,flags=re.M)
 assert n==1,(key,n)
p.write_text(s)
PY
export NETCDF=/usr J=-j8
./compile -j 8 em_real
