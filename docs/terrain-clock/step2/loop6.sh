#!/bin/bash
# loop6.sh JOBS NPROC STOP_EPOCH TAG : chain 1-card sweep6 batches through the mutex (each --est 1500, no new job after
# 1200 s) until every job in JOBS has a row or STOP_EPOCH passes (set ~50 min before the box dead-man time).
JOBS=$1; NPROC=$2; STOP=$3; TAG=$4; W=/work/tclock
total=$(grep -c . $JOBS)
while [ $(date +%s) -lt $STOP ]; do
  done_n=$(ls $W/results/rows6/*-blowup.json 2>/dev/null | wc -l)
  echo "$(date -u +%FT%TZ) loop6 $TAG: $done_n of $total rows present" >> $W/logs/sweep6-events.log
  [ $done_n -ge $total ] && break
  card-lease tclock --cards 1 --est 1500 --wait 3000 $W/sweep6.sh $JOBS $NPROC 1200 >> $W/logs/mutex-loop6-$TAG.log 2>&1
  rc=$?; [ $rc -eq 3 ] && { echo "$(date -u +%FT%TZ) loop6 $TAG: validate failed, stop" >> $W/logs/sweep6-events.log; break; }
  sleep 5
done
echo "$(date -u +%FT%TZ) loop6 $TAG: end" >> $W/logs/sweep6-events.log
