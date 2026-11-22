#!/bin/bash
# TEMPO column oracle, referee side, for the v3.2.5 generation
# (tempo_version = "v3.2.5", woof/core/kernels/tempo325).  CPU only.
#
#   build_reference_v325.sh <work dir> <TEMPO table dir>
#
# Builds NCAR/TEMPO tempo_v3.2.5 (c62699fb660e) with its own
# Makefile.gfortran flags, LABELLED PATCHED: one line, final_weight = 1 before
# its conditional assignment in sum_tendencies (module_mp_tempo_main.F90
# :1752-1755 reads final_weight where v3.2.5 leaves it unset; the port
# defines it the same way).  Nothing else is changed; the patch is written to
# <work>/PATCH.diff and checked to be exactly that one line.
# Then as build_reference.sh: tempo_init on the pinned tables, the init dump
# (tempo_blob.bin), the column set (make_columns.py) and ref_out.bin.
set -euo pipefail
WORK=${1:?usage: build_reference_v325.sh WORK_DIR TABLE_DIR}
TABLES=${2:?usage: build_reference_v325.sh WORK_DIR TABLE_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
PIN=c62699fb660ebbb14e4388f7b59a8b38decac1ab
mkdir -p "$WORK"
cd "$WORK"
if [ ! -d TEMPO ]; then
  git clone -q https://github.com/NCAR/TEMPO.git TEMPO
fi
git -C TEMPO checkout -q $PIN
test "$(git -C TEMPO rev-parse HEAD)" = $PIN
test -z "$(git -C TEMPO status --porcelain --untracked-files=no)"   # unmodified clone

rm -rf src-patched && cp -r TEMPO/src src-patched
python3 - "$WORK/src-patched/module_mp_tempo_main.F90" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "          dens_weight = rho_g(nrhg)*frozen_terms/massg_sources\n"
assert s.count(anchor) == 1, "final_weight anchor not found once"
s = s.replace(anchor, "          final_weight = 1._wp ! PATCHED (woof referee): v3.2.5 reads it unset\n" + anchor)
open(p, "w").write(s)
PY
diff -u TEMPO/src/module_mp_tempo_main.F90 src-patched/module_mp_tempo_main.F90 > PATCH.diff || true
test "$(grep -c '^+[^+]' PATCH.diff)" = 1
test "$(grep -c '^-[^-]' PATCH.diff)" = 0

FCFLAGS="-c -g -O0 -fbacktrace -fcheck=all -ffree-line-length-0 -fdefault-real-8 -fdefault-double-8"
rm -rf build && mkdir -p build
for f in machine module_mp_tempo_cfgs module_mp_tempo_params module_mp_tempo_ml \
         module_mp_tempo_utils module_mp_tempo_aerosols module_mp_tempo_diags \
         module_mp_tempo_tables module_mp_tempo_main module_mp_tempo_driver; do
  gfortran $FCFLAGS -J build src-patched/$f.F90 -o build/$f.o
done
# v3.2.5 dropped the params scalar r (= rdry = 287.04 at v3.0.3); the port's
# blob keeps the record (unread by any kernel), written here as rdry.
sed "s/call wr(u, 'r', \[r\])/call wr(u, 'r', [rdry])/" "$HERE/../tempo_ref_driver.F90" > build/tempo_ref_driver_v325.F90
grep -q "call wr(u, 'r', \[rdry\])" build/tempo_ref_driver_v325.F90
gfortran $FCFLAGS -J build -I build build/tempo_ref_driver_v325.F90 -o build/tempo_ref_driver.o
gfortran $FCFLAGS -J build -I build "$HERE/tempo_oracle_driver.F90" -o build/tempo_oracle_driver.o
LIBOBJ=$(ls build/*.o | grep -v -e tempo_ref_driver.o -e tempo_oracle_driver.o)
gfortran -o build/tempo_ref_driver build/tempo_ref_driver.o $LIBOBJ
gfortran -o build/tempo_oracle_driver build/tempo_oracle_driver.o $LIBOBJ
gfortran --version | sed -n 1p > build/COMPILER.txt
ldd --version | sed -n 1p >> build/COMPILER.txt

# tempo_init reads the tables from the working directory; the scenarios
# read TEMPO/test/data, so the init dump runs inside the clone (untracked).
mkdir -p TEMPO/run_ref run_oracle
for t in ccn_activate.bin freeze_water_data_tempo_v3 qr_acr_qg_data_tempo_v3 qr_acr_qs_data_tempo_v3; do
  ln -sf "$TABLES/$t" TEMPO/run_ref/$t
  ln -sf "$TABLES/$t" run_oracle/$t
done
ulimit -s unlimited || true
(cd TEMPO/run_ref && "$WORK/build/tempo_ref_driver")
mv TEMPO/run_ref/tempo_blob.bin TEMPO/run_ref/tempo_ref_columns.bin .

"${PYTHON:-python3}" "$HERE/make_columns.py" "$WORK"
(cd run_oracle && "$WORK/build/tempo_oracle_driver" "$WORK/columns_in.bin" "$WORK/ref_out.bin")
sha256sum tempo_blob.bin tempo_ref_columns.bin columns_in.bin columns_catalog.json ref_out.bin \
  PATCH.diff "$TABLES"/ccn_activate.bin "$TABLES"/*_tempo_v3 > SHA256SUMS
cat SHA256SUMS
