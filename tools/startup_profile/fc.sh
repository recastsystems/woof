#!/usr/bin/env bash
# fc.sh ARM CUPY_CACHE_DIR : one 2-card forecast of a prepared bundle, with the
# in-process stack sampler (sitecustomize.py beside this file) writing every
# thread's stack every 2 s to $WORK/prep/spy-ARM/.  Launch through the card
# mutex, e.g.  card-lease <lane> --cards 2 --wait 7200 fc.sh before /work/<lane>/cupy
# WORK holds env.sh (venv + toolchain) and prep/out/chain/hrrr-root-prep.
ARM=$1; export CUPY_CACHE_DIR=$2
WORK=${WORK:-/work/pi-startup}
HERE=$(cd "$(dirname "$0")" && pwd)
cd $WORK/prep
. $WORK/env.sh
export PYTHONPATH=$HERE${PYTHONPATH:+:$PYTHONPATH} WOOF_PISTARTUP_SAMPLE=$WORK/prep/spy-$ARM
mkdir -p $CUPY_CACHE_DIR spy-$ARM
echo "arm=$ARM cards=$GPU_MUTEX_CARDS cache=$CUPY_CACHE_DIR entries=$(find $CUPY_CACHE_DIR -type f | wc -l) start=$(date -u +%FT%TZ)" > fc-$ARM.meta
timeout -k 60s 100m nice -n 10 python -m woof.cli go out/chain/hrrr-root-prep/experiment.toml --prepared-root out/chain/hrrr-root-prep --wps-namelist out/chain/hrrr-root-prep/namelist.wps --products none --outdir fc-$ARM > fc-$ARM.log 2>&1 &
pid=$!
for i in $(seq 1 80); do
  sleep 10
  kill -0 $pid 2>/dev/null || break
  R=$(ls -td fc-$ARM/run-*/ 2>/dev/null | head -1)
  [ -n "$R" ] && grep -q "\"step\": 30," ${R}chain/run/progress.jsonl 2>/dev/null && break
done
wait $pid; rc=$?
echo "rc=$rc end=$(date -u +%FT%TZ) entries=$(find $CUPY_CACHE_DIR -type f | wc -l)" >> fc-$ARM.meta
exit $rc
