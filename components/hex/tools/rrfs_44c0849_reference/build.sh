#!/usr/bin/env bash
# Usage: build.sh REFERENCE_ROOT NEW_BUILD_DIRECTORY atmosphere|init_atmosphere
set -euo pipefail
REF=$(realpath "$1")
BUILD=$(realpath -m "$2")
CORE=$3
HERE=$(dirname "$(realpath "$0")")
LEGACY="$HERE/../native_physics_reference/rrfs"
OPT=${RRFS_FORK_CPU_OPT:?set RRFS_FORK_CPU_OPT to the fork CPU toolchain prefix (holds mpich/)}
export PATH="$OPT/mpich/bin:$PATH"
export LD_LIBRARY_PATH="$OPT/mpich/lib:${LD_LIBRARY_PATH:-}"
export MPICH_FC=gfortran MPICH_CC=gcc MPICH_CXX=g++
python3 "$HERE/prepare.py" "$REF" "$BUILD"
export TMPDIR="$BUILD/compiler-tmp"
mkdir -p "$TMPDIR" "$BUILD/compiler-shim"
export RRFS_REAL_COMPILERS
RRFS_REAL_COMPILERS=$(python3 - <<'PY'
import json,shutil
print(json.dumps({n:shutil.which(n) for n in ('mpif90','mpicc','mpicxx','gfortran','gcc','g++')}))
PY
)
for name in mpif90 mpicc mpicxx gfortran gcc g++; do
    ln -s "$(realpath "$LEGACY/compiler_capture.py")" "$BUILD/compiler-shim/$name"
done
export RRFS_COMPILE_RECEIPT="$BUILD/compile-commands.jsonl"
{
    echo "recipe=gnu-strict core=$CORE precision=single MUSICA=false STOCHASTIC_PHYSICS=false"
    gfortran --version | head -1
    "$OPT/mpich/bin/mpichversion"
    getconf GNU_LIBC_VERSION
    taskset -pc $$
    sha256sum "$OPT/pnetcdf/lib/libpnetcdf.a"
    nf-config --all
} > "$BUILD/toolchain.txt"
export PATH="$BUILD/compiler-shim:$PATH"
cd "$BUILD"
ulimit -s unlimited
timeout 3600 nice -n 10 taskset -c 8-23 make -j 8 gfortran CORE="$CORE" PRECISION=single MUSICA=false STOCHASTIC_PHYSICS=false PNETCDF="$OPT/pnetcdf" NETCDF=/usr > build.log 2>&1
test -x "${CORE}_model"
python3 "$LEGACY/audit_flags.py" "$BUILD" --recipe gnu-strict --core "$CORE" --out "$BUILD/flag-audit.json"
sha256sum "${CORE}_model" > executable.sha256
echo "BUILD-OK $BUILD"
