#!/usr/bin/env bash
# usage: build_reference.sh WORK SOURCE RECIPE CORE MPI_PREFIX PNETCDF_PREFIX TABLES [primary|secondary]
# Extends f0's offline recipe; only call through the host's bounded build prefix; -j4; cpuset 8-23.
set -euo pipefail
W=$(realpath -m "$1"); SRC=$(realpath "$2"); RECIPE=$3; CORE=$4
MPI=$(realpath "$5"); PNETCDF=$(realpath "$6"); TABLES=$(realpath "$7")
REFERENCE=${8:-primary}; DIAGNOSTIC=${9:-}
HERE=$(dirname "$(realpath "$0")")
case "$RECIPE" in gnu-strict|gnu-o3) ;; *) exit 2;; esac
case "$CORE" in atmosphere|init_atmosphere) ;; *) exit 2;; esac
export PATH="$MPI/bin:$PATH" LD_LIBRARY_PATH="$MPI/lib:${LD_LIBRARY_PATH:-}"
export MPICH_FC=gfortran MPICH_CC=gcc MPICH_CXX=g++
case "$REFERENCE" in primary) D="$W/native/rrfs-$RECIPE-$CORE";; secondary) D="$W/native/rrfs-secondary-$RECIPE-$CORE";; *) exit 2;; esac
case "$DIAGNOSTIC" in '') ;; bounds-snan|bounds-snan-integer-poison) D="$D-diagnostic-$DIAGNOSTIC";; *) exit 2;; esac
if [ "${RRFS_RESUME_BUILD:-0}" = 1 ]; then
  python3 - "$D" "$RECIPE" "$REFERENCE" "$DIAGNOSTIC" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); pins=json.loads((root/'SOURCE-PINS.json').read_text())
assert pins['recipe']==sys.argv[2] and pins['precision']=='single' and not pins['musica']
assert pins.get('reference','primary')==sys.argv[3]
assert (pins.get('diagnostic') or '')==sys.argv[4]
PY
else
  diag_args=()
  if [ -n "$DIAGNOSTIC" ]; then diag_args=(--diagnostic "$DIAGNOSTIC"); fi
  python3 "$HERE/prepare_build.py" "$SRC" "$D" --recipe "$RECIPE" --tables "$TABLES" --reference "$REFERENCE" "${diag_args[@]}"
fi
# Compiler intermediates belong to this lane, rather than the shared /tmp.
export TMPDIR=${TMPDIR:-$W/compiler-tmp}
mkdir -p "$TMPDIR"
export RRFS_REAL_COMPILERS
RRFS_REAL_COMPILERS=$(python3 - <<'PY'
import json,shutil
print(json.dumps({n:shutil.which(n) for n in ('mpif90','mpicc','mpicxx','gfortran','gcc','g++')}))
PY
)
mkdir -p "$D/compiler-shim"
for name in mpif90 mpicc mpicxx gfortran gcc g++; do
  [ -e "$D/compiler-shim/$name" ] || ln -s "$HERE/compiler_capture.py" "$D/compiler-shim/$name"
done
export RRFS_COMPILE_RECEIPT="$D/compile-commands.jsonl"
{
  echo "reference=$REFERENCE recipe=$RECIPE core=$CORE diagnostic=$DIAGNOSTIC precision=single MUSICA=off STOCHASTIC_PHYSICS=false host=$(hostname)"
  gfortran --version | head -1
  "$MPI/bin/mpichversion"; getconf GNU_LIBC_VERSION
  taskset -pc $$; grep -m1 'model name' /proc/cpuinfo
  sha256sum "$PNETCDF/lib/libpnetcdf.a"
  grep '#define PNETCDF_VERSION ' "$PNETCDF/include/pnetcdf.h"
  nf-config --all
} > "$D/toolchain.txt"
export PATH="$D/compiler-shim:$PATH"
cd "$D"
ulimit -s unlimited
make -j4 gfortran CORE="$CORE" PRECISION=single MUSICA=false STOCHASTIC_PHYSICS=false PNETCDF="$PNETCDF" NETCDF="${NETCDF_PREFIX:-/usr}" > build.log 2>&1
test -x "${CORE}_model"
python3 "$HERE/audit_flags.py" "$D" --recipe "$RECIPE" --core "$CORE" --out "$D/flag-audit.json"
sha256sum "${CORE}_model" > executable.sha256
echo "BUILD-OK $D"
