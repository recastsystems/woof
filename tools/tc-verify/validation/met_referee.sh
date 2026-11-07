#!/bin/bash
set -euo pipefail
# Usage: MET_WORKSPACE=/path/to/isolated/build ./met_referee.sh
# Workspace contains met-src (v12.2.2) and met-deps/lib. No GPU is used.
cd "${MET_WORKSPACE:?set MET_WORKSPACE to your isolated MET build}"
export LD_LIBRARY_PATH=$PWD/met-deps/lib
export MET_BASE=$PWD/met-src/data
mkdir -p runs/referee
cat > runs/referee/config <<'CFG'
model = [ "HFSA", "HFSB", "HFAI", "HFBI", "OFCL", "GFSO" ];
interp12 = NONE;
match_points = TRUE;
check_dup = TRUE;
CFG
for id in al142024 al092024; do
 curl -fsSL https://ftp.nhc.noaa.gov/atcf/archive/2024/a${id}.dat.gz -o runs/referee/a${id}.dat.gz
 curl -fsSL https://ftp.nhc.noaa.gov/atcf/archive/2024/b${id}.dat.gz -o runs/referee/b${id}.dat.gz
 gzip -dc runs/referee/a${id}.dat.gz > runs/referee/a${id}.dat
 gzip -dc runs/referee/b${id}.dat.gz > runs/referee/b${id}.dat
 met-src/src/tools/tc_utils/tc_pairs/tc_pairs -adeck runs/referee/a${id}.dat -bdeck runs/referee/b${id}.dat -config runs/referee/config -out runs/referee/${id} > runs/referee/${id}.log 2>&1
done
