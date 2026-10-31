#!/bin/bash
# batch6.sh NPROC RUN_S : inside card-lease tclock --cards 1. Step 6. Once per card, re-checks the instrument
# (the step-2 validate cell, map criterion, peak w 34.655418395996094 to the bit), then NPROC workers take the fixed
# jobs (cell6.py) and then the adaptive re-walks (adapt6.py) until RUN_S passes.
set -u; source /work/tclock/s6/env6.sh
L=/work/tclock/s6; NPROC=$1; RUN_S=$2; DEADLINE=$(( $(date +%s) + RUN_S )); C=$GPU_MUTEX_CARDS
V=$L/results/validate-c$C.jsonl; mkdir -p $L/results/cells $L/results/adaptive $L/logs
if [ ! -s $V ]; then
  cd $T4 && $PY tools/terrain_clock_probe.py fixed --dx 3000 --crest 3000 --ridge-slopes 0.1 --winds 40 --sound-steps 4 --steps 24 \
     --seconds 10800 --courant --out $V > /dev/null 2>> $L/logs/validate.err
  $PY $L/val6.py $V $C >> $L/logs/events.log || { rm -f $V; exit 3; }
fi
echo "$(date -u +%FT%TZ) start card=$C nproc=$NPROC deadline=$DEADLINE" >> $L/logs/events.log
for i in $(seq 1 $NPROC); do
  ( for J in $L/jobs-fx6.jsonl $L/jobs-80.jsonl; do [ -s $J ] && $PY $L/cell6.py $J $L/results/cells $DEADLINE; done
    $PY $L/adapt6.py $L/jobs-ad6.jsonl $L/results/adaptive $DEADLINE ) >> $L/logs/worker-c$C-$i.log 2>&1 &
  sleep 1
done
wait
echo "$(date -u +%FT%TZ) end card=$C cells=$(ls $L/results/cells/*.json 2>/dev/null | wc -l) rows=$(ls $L/results/adaptive/*.json 2>/dev/null | wc -l)" >> $L/logs/events.log
