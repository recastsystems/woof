#!/bin/bash
# sweep.sh JOBS NPROC RUN_S : run inside card-lease tclock --cards 1 [--est S]. Starts NPROC probe workers on
# the one granted card; they claim jobs from JOBS (per-host claims), and none starts a new job after RUN_S seconds.
# Rows land in $W/results/rows/<key>.json; every run streams to <key>.partial.jsonl until its row is done.
set -u; source /work/tclock/env.sh
JOBS=$1; NPROC=$2; RUN_S=$3
DEADLINE=$(( $(date +%s) + RUN_S )); O=$W/results/rows; mkdir -p $O $W/logs
cat > $W/results/provenance-$(hostname).json <<EOJ
{"host": "$(hostname)", "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU_MUTEX_CARDS | head -1)",
 "card": "$GPU_MUTEX_CARDS", "driver": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)",
 "tree_commit": "$(cat $T/TREE-COMMIT)", "probe": "tools/terrain_clock_probe.py sweep",
 "python": "$($PY -c 'import sys;print(sys.version.split()[0])')", "cupy": "$($PY -c 'import cupy;print(cupy.__version__)')"}
EOJ
echo "$(date -u +%FT%TZ) start card=$GPU_MUTEX_CARDS nproc=$NPROC deadline=$DEADLINE jobs=$JOBS" >> $W/logs/sweep-events.log
cd $T
for i in $(seq 1 $NPROC); do
  $PY tools/terrain_clock_probe.py sweep --jobs $JOBS --out-dir $O --winds 20,30,40,50,60 \
     --rungs-per-km 6.5,6.0,5.5,5.0,4.5,4.0,3.5,3.0 --deadline $DEADLINE \
     --provenance $W/results/provenance-$(hostname).json >> $W/logs/worker-$(hostname)-c$GPU_MUTEX_CARDS-$i.log 2>&1 &
  sleep 2
done
wait
echo "$(date -u +%FT%TZ) end card=$GPU_MUTEX_CARDS rows=$(ls $O/*.json 2>/dev/null | wc -l)" >> $W/logs/sweep-events.log
