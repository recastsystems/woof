#!/usr/bin/env bash
# Run every fixture through the built Fortran arms.
#   bash run_fortran.sh <BUILD_DIR> <FIXDIR> [ARMS...]   (default arms: A B)
# Dumps land in <BUILD_DIR>/out/<arm>/<fixture>.ref, FORMAT.md layout.
# A fixture set whose index.json says "predict_nc": true (make_columns_nc.py)
# runs through the mp_physics=51 driver and lands in out/<arm>-nc/.
# A fixture set whose index.json says "mp_physics": 52 runs through the
# 2-category driver (p3_ref_driver_2cat) and lands in out<SUFFIX>/<arm>
# with SUFFIX taken from the environment (default "-2cat"), so the mp=50
# and mp=52 dumps of one build never overwrite each other.
set -euo pipefail
OUT=${1:?BUILD_DIR}; FIX=${2:?FIXDIR}; shift 2
ARMS=${*:-A B}
FIX=$(cd "$FIX" && pwd)
DRIVER=p3_ref_driver; OUTSET=out; ARMSUF=""
if grep -q '"mp_physics": 52' "$FIX/index.json"; then
  DRIVER=p3_ref_driver_2cat; OUTSET=out${SUFFIX:--2cat}
elif grep -q '"predict_nc": true' "$FIX/index.json"; then
  DRIVER=p3_nc_ref_driver; ARMSUF="-nc"
fi
for ARM in $ARMS; do
  D="$OUT/$OUTSET/$ARM$ARMSUF"; rm -rf "$D"; mkdir -p "$D"
  for CTL in "$FIX"/*.ctl; do
    N=$(basename "$CTL" .ctl)
    sed -e "s#@TABLEDIR@#$OUT/tables#" -e "s#@FIXDIR@#$FIX#" -e "s#@OUTDIR@#$D#" "$CTL" > "$D/$N.ctl"
    if "$OUT/build/$ARM/$DRIVER" "$D/$N.ctl" > "$D/$N.log" 2>&1; then
      echo "arm=$ARM $N ok $(sha256sum "$D/$N.ref" | cut -c1-16)"
    else
      echo "arm=$ARM $N FAILED rc=$?"; tail -5 "$D/$N.log"
    fi
  done
done
