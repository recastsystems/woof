#!/usr/bin/env bash
# Run the Shin-Hong capture copy on the A195 gate's inputs (our own init, HRRR recipe:
# Thompson, RUC, RRTMG) with bl_pbl_physics = 11 and the revised MM5 surface
# layer, one hour, writing Shin-Hong captures at steps 1, 2, 50, 100, 240.
#   run_capture.sh PORT_ROOT
set -euo pipefail
P=$1
B=$P/build/WRF-4.8.0-serial-tfix-capL09
R=$P/capture-runs/L09
until [ -x $B/main/wrf.exe ] || ! podman ps --format '{{.Names}}' | grep -q wrf480-capL09; do sleep 20; done
test -x $B/main/wrf.exe
rm -rf $R; mkdir -p $R/captures
cd $P/captures/runs/A195-tfix
for f in *; do case $f in wrfout*|wrfrst*|rsl.*|wrf.stdout|run-receipt.json|wrf.exe|namelist.output) ;; *) ln -s $PWD/$f $R/$f;; esac; done
rm -f $R/namelist.input
sed -e 's/^ *bl_pbl_physics *=.*/ bl_pbl_physics               = 11,/' \
    -e 's/^ *sf_sfclay_physics *=.*/ sf_sfclay_physics            = 1,/' \
    namelist.input > $R/namelist.input
grep -E "bl_pbl_physics|sf_sfclay_physics|sf_surface|mp_physics|run_hours|time_step " $R/namelist.input
ln -sf $B/main/wrf.exe $R/wrf.exe
start=$(date -u +%FT%TZ); t0=$(date +%s)
podman run --rm --name "capL09-run" -v $P:$P -w $R localhost/wrfref2404:1 bash -c "ulimit -s unlimited; ./wrf.exe > wrf.stdout 2>&1"; rc=$?
echo "{\"start\": \"$start\", \"wall_s\": $(( $(date +%s)-t0 )), \"rc\": $rc}" > $R/run-receipt.json
tail -2 $R/rsl.error.0000 2>/dev/null || tail -2 $R/wrf.stdout
ls $R/captures | wc -l
