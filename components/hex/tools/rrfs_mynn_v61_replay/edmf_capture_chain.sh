#!/usr/bin/env bash
# The six EDMF captures on the RRFS referee of record (a128a1e8, labelled patched): C1 cold start, C2 restart from
# C1's +36 min restart, C3 the RRFS DA-cycled start (do_DAcycling true, do_restart false, the da_state stream as the
# initial state, written by a jedi_da twin of C1). Each: mynnedmf_driver PRE/POST and MPAS_to_physics (pres2_hyd_p).
# Needs the edmf_capture.py environment (MYNN_LANE, MYNN_SCRATCH, MYNN_OBSERVERS, RRFS_REFERENCE,
# RRFS_REFEREE_BINARY, RRFS_INIT) and PY. Referee tool only.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
S=$MYNN_SCRATCH
cap() { stem=$1; shift; timeout 900 nice -n 10 $PY "$HERE/edmf_capture.py" --patched --stem $stem "$@" > "$MYNN_LANE/evidence/$stem-capture.log" 2>&1; }
cap c1p-mynn & cap c1p-interface --boundary interface & cap c3bg-mynn --set "config_jedi_da=true" & wait
R=$S/c1p-restart.2023-03-31_12.36.00.nc
M=$S/c3bg-mpasout.2023-03-31_12.36.00.nc
cp "$S/c1p-mynn-native-3steps/ordinary/restart.2023-03-31_12.36.00.nc" "$R"
cp "$S/c3bg-mynn-native-3steps/ordinary/mpasout.2023-03-31_12.36.00.nc" "$M"
START="config_start_time='2023-03-31_12:36:00'"
cap c2p-mynn --set "$START" --set "config_do_restart=true" --file restart.2023-03-31_12.36.00.nc=$R &
cap c2p-interface --boundary interface --set "$START" --set "config_do_restart=true" --file restart.2023-03-31_12.36.00.nc=$R &
cap c3p-mynn --set "$START" --set "config_do_DAcycling=true" --set "config_jedi_da=true" --file mpasout.2023-03-31_12.36.00.nc=$M &
cap c3p-interface --boundary interface --set "$START" --set "config_do_DAcycling=true" --set "config_jedi_da=true" --file mpasout.2023-03-31_12.36.00.nc=$M &
wait
echo CHAIN-DONE
