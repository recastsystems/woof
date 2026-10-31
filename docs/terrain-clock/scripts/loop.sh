#!/bin/bash
# loop.sh JOBS NPROC : chain 1-card sweep batches through the mutex (each --est 1500, no new job after 1100 s) until every
# job in JOBS has a row on this host or 03:35Z passes (STOP=1791344100), which leaves the last batch ~40 min before the box's
# own dead-man time to finish and be copied home.  Each batch re-queues behind the DA lane and the other tenants.
JOBS=$1; NPROC=$2; W=/work/tclock
total=$(grep -c . $JOBS)
while [ $(date +%s) -lt 1791344100 ]; do
  done_n=0; while read -r line; do k=$(python3 -c "import json,sys;j=json.loads(sys.argv[1]);print('%s-dx%g-c%g-r%g-x%d'%(j['settings'],j['dx'],j['crest'],j['ridge_slope'],j['sound_steps']))" "$line"); [ -e $W/results/rows/$k.json ] && done_n=$((done_n+1)); done < $JOBS
  echo "$(date -u +%FT%TZ) loop: $done_n of $total rows present" >> $W/logs/sweep-events.log
  [ $done_n -ge $total ] && break
  card-lease tclock --cards 1 --est 1500 --wait 3000 $W/sweep.sh $JOBS $NPROC 1100 >> $W/logs/mutex-loop.log 2>&1
  sleep 5
done
echo "$(date -u +%FT%TZ) loop: end" >> $W/logs/sweep-events.log
