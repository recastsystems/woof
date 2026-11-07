#!/bin/bash
# loop6.sh NPROC STOP_EPOCH TAG : chain 1-card batch6 runs through the mutex (--est 1500, no new job after 1000 s)
# until every job has a record or STOP_EPOCH passes.
NPROC=$1; STOP=$2; TAG=$3; L=/work/tclock/s6
while [ $(date +%s) -lt $STOP ]; do
  total=$(( $(cat $L/jobs-fx6.jsonl $L/jobs-80.jsonl 2>/dev/null | grep -c .) + $(grep -c . $L/jobs-ad6.jsonl) ))
  n=$(( $(ls $L/results/cells/*.json 2>/dev/null | wc -l) + $(ls $L/results/adaptive/*.json 2>/dev/null | wc -l) ))
  echo "$(date -u +%FT%TZ) loop $TAG: $n of $total" >> $L/logs/events.log
  [ $n -ge $total ] && break
  left=$(( STOP - $(date +%s) )); run=$(( left < 1000 ? left : 1000 )); [ $run -lt 60 ] && break
  ${GPU_MUTEX_DIR:?set GPU_MUTEX_DIR}/run.sh tclock --cards 1 --est 1500 --wait 1800 $L/batch6.sh $NPROC $run >> $L/logs/mutex-$TAG.log 2>&1
  [ $? -eq 3 ] && { echo "$(date -u +%FT%TZ) loop $TAG: validate failed, stop" >> $L/logs/events.log; break; }
  # stale claims of workers the batch ended are taken over by pid check in _claim
  sleep 5
done
echo "$(date -u +%FT%TZ) loop $TAG: end" >> $L/logs/events.log
