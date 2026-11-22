#!/usr/bin/env bash
# run_native.sh RUNDIR NAME : native wrf.exe in the pinned referee container, same-path mount
RD=$1; NAME=$2; P=${WRF480_ROOT:?set WRF480_ROOT to the referee working root}
start=$(date -u +%FT%TZ); t0=$(date +%s)
podman run --rm --name "g0-$NAME" -v $P:$P -w $RD -e GPUWM_CAPTURE_DIR="${GPUWM_CAPTURE_DIR:-}" -e GPUWM_CAPTURE_STEPS="${GPUWM_CAPTURE_STEPS:-}" localhost/wrfref2404:1 bash -c "ulimit -s unlimited; ./wrf.exe > wrf.stdout 2>&1"; rc=$?
t1=$(date +%s)
ok=false; grep -q "SUCCESS COMPLETE WRF" $RD/wrf.stdout && ok=true
printf "{\"run\": \"%s\", \"start\": \"%s\", \"wall_s\": %d, \"rc\": %d, \"success\": %s, \"wrf_exe_sha256\": \"%s\"}\n" "$NAME" "$start" $((t1-t0)) $rc $ok "$(sha256sum $RD/wrf.exe | cut -c1-64)" > $RD/run-receipt.json
