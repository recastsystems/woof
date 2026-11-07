#!/usr/bin/env bash
# Build and run the SSiB column oracle against the byte-unmodified WRF v4.6.1
# phys/module_sf_ssib.F.
#
#   bash tools/ssib_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# THE REFERENCE is module_sf_ssib.F at -O0 with -finit-local-zero.  The
# second flag is the only departure from a plain compile and it changes no
# defined behaviour: it gives every automatic local the value 0 on entry.
# It exists because SSIB reads automatic locals before writing them on two
# paths (UPDAT1's SNOWW(2) under the multi-layer snow pack, every step; and
# TEMRS2's ICASE family when the top snow layer's ice fraction comes out
# above 1).  What WRF reads there is whatever the stack held, so no WRF
# build has a defined value to match; the oracle pins it to zero and the
# GPU port zero-initialises the same locals.  The two evidence builds below
# show which outputs depend on it.
#
# EVIDENCE BUILDS (never the reference):
#   plain -O0 (no init)                     -> out-O0-noinit.bin
#   WRF's own scalar flags (configure.wrf FCOPTIM of the scalar referee
#   build: -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize) with
#   -finit-local-zero                       -> out-O2wrf.bin
# compare.py --evidence reports where each differs from the reference.
#
# libmvec: the -O0 reference object must reference no _ZGV* vector symbol.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# WRF 4.6.1 and 4.7.1 distribute identical bytes for this module. Pin the
# actual compilation unit, including when using an official source archive.
pinned_source="2b83b39640fd5d53859f1f281be59a9866baf97924e5d03d9964ec5d69bd5c73"
src="phys/module_sf_ssib.F"

test -f "${source_root}/${src}"
source_digest=$(sha256sum "${source_root}/${src}" | cut -d' ' -f1)
if [[ "${source_digest}" != "${pinned_source}" ]]; then
    echo "${src} digest ${source_digest} differs from ${pinned_source}" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"
python3 "${here}/layout.py" --fortran > layout.inc

defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none -fallow-argument-mismatch"

variant() {   # name, flags...
    local name=$1; shift
    mkdir -p "${name}"
    ( cd "${name}"
      gfortran -c "$@" -cpp ${defines} ${free} "${source_root}/${src}"
      gfortran -c -O0 ${free} "${here}/stub_wrf.F90"
      gfortran -c -O0 -cpp ${free} -I .. "${here}/ssib_oracle.F90"
      gfortran -o ssib_oracle module_sf_ssib.o stub_wrf.o ssib_oracle.o )
}

variant ref -O0 -finit-local-zero

# A second, explicitly changed oracle checks the default unit repair. This
# source copy belongs to the oracle build only; the strict WRF source is untouched.
mkdir -p default-units
python3 - "${source_root}/${src}" default-units/module_sf_ssib.F <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
at=s.index('      SUBROUTINE SSIB_SEAICE')
end=s.index('        END SUBROUTINE SSIB_SEAICE', at)
part=s[at:end]
import re
for name, component in [('SNOA',2),('SNOB',1)]:
    pattern=rf'(?im)^(\s*{name}\s*=\s*CAPAC\({component}\))\s*$'
    part,count=re.subn(pattern, r'\1*1000.', part)
    if count != 1: raise SystemExit(f'{name}: expected one return-unit statement, got {count}')
Path(sys.argv[2]).write_text(s[:at]+part+s[end:])
PY
( cd default-units
  gfortran -c -O0 -finit-local-zero -cpp ${defines} ${free} module_sf_ssib.F
  gfortran -c -O0 ${free} "${here}/stub_wrf.F90"
  gfortran -c -O0 -cpp ${free} -I .. "${here}/ssib_oracle.F90"
  gfortran -o ssib_oracle module_sf_ssib.o stub_wrf.o ssib_oracle.o )

variant noinit -O0
variant o2wrf -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -finit-local-zero

( cd ref
  gfortran -c -O0 ${free} -I . "${here}/dump_tables.F90"
  gfortran -o dump_tables module_sf_ssib.o stub_wrf.o dump_tables.o
  ./dump_tables )
python3 "${here}/gen_tables.py" "${source_root}" ssib_veg_tables.json
HERE="${here}" python3 - <<'PY'
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["HERE"])
from gen_tables import MODULE_TABLES, flat
doc = json.load(open("ssib_veg_tables.json"))
n = sum(int(np.prod(s)) for _, s in MODULE_TABLES)
mine = flat(doc)[:n].view("<u4")
theirs = np.fromfile("ref/tables-gfortran.bin", "<u4")
if theirs.shape != mine.shape or (theirs != mine).any():
    bad = np.flatnonzero(theirs != mine) if theirs.shape == mine.shape else "shape"
    raise SystemExit(f"parsed tables differ from gfortran's compiled DATA at {bad}")
print(f"tables: {n} module words equal to gfortran's compiled DATA, bit for bit")
PY
nm -u ref/module_sf_ssib.o | sed 's/^ *//' | sort > undefined-ref.txt
if grep -q '_ZGV' undefined-ref.txt; then
    echo "reference object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-ref.txt >&2
    exit 4
fi

python3 "${here}/gen_columns.py" columns.bin
./ref/ssib_oracle columns.bin out-ref.bin 2> ref-messages.txt
./default-units/ssib_oracle columns.bin out-default-units.bin 2> default-units-messages.txt
./noinit/ssib_oracle columns.bin out-O0-noinit.bin 2> noinit-messages.txt
./o2wrf/ssib_oracle columns.bin out-O2wrf.bin 2> o2wrf-messages.txt

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# reference libm symbols:"
    grep -E 'exp|pow|log|sqrt|_ZGV' undefined-ref.txt || true
} > toolchain.txt
sha256sum "${source_root}/${src}" "${here}"/*.F90 "${here}"/*.py "${here}/build.sh" \
    columns.bin out-ref.bin > oracle-sha256sums.txt
cat toolchain.txt
echo "oracle built in ${build_dir}"
