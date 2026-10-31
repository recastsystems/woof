#!/bin/bash
# loop4.sh JOBS NPROC STOP_EPOCH HARD_EPOCH TAG : chain 1-card batch4 holds through the mutex (--est 1800, new runs only
# for 1500 s per hold) until every job has a row or STOP_EPOCH passes. HARD_EPOCH: no run starts after it (the box
# lifecycle deadline is managed separately by the box owner).
JOBS=$1; NPROC=$2; STOP=$3; HARD=$4; TAG=$5; W4=/work/tclock/s4
total=$(grep -c . $JOBS)
while [ $(date +%s) -lt $STOP ]; do
  n=$(ls $W4/results/rows2030/*.json 2>/dev/null | wc -l)
  echo "$(date -u +%FT%TZ) loop4b $TAG: $n of $total rows" >> $W4/logs/events.log
  [ $n -ge $total ] && break
  card-lease tclock --cards 1 --est 1800 --wait 1800 $W4/batch4b.sh $JOBS $NPROC 1500 $HARD $TAG >> $W4/logs/mutex-$TAG.log 2>&1
  sleep 5
done
echo "$(date -u +%FT%TZ) loop4b $TAG: end" >> $W4/logs/events.log
