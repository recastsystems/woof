#!/usr/bin/env bash
# job3.sh: meeting-perf-a13 layout-identity rerun on box E. CPU staging first (no cards held), then 4 cards through
# card-lease for four arms: tip 2x2, tip 1x4, base (534c6dea3) 2x2, base 1x4. A second instance exits at once.
set -u
W=/work/meeting-perf-a13; L=$W/job3.lock
mkdir $L 2>/dev/null || { echo "job3 already running or ran (lock $L); exiting"; exit 0; }
echo "$$ $(date -u +%FT%TZ)" > $L/pid
exec > >(tee -a $W/job3.out) 2>&1
[ -f $W/case/c3/prepared/proof.json ] || { $W/stageE.sh > $W/stage3.out 2>&1; grep -q "PREP DONE rc=0" $W/stage3.out || { echo "STAGE FAILED"; exit 1; }; }
exec card-lease meeting-perf-a13 --cards 4 --wait 7200 $W/gpu3.sh
