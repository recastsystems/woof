#!/bin/bash
# batch4.sh JOBS NPROC RUN_S HARD_EPOCH TAG : inside card-lease tclock --cards 1 --est 1800. NPROC workers of
# `terrain_clock_probe.py adaptive-sweep --criterion blowup` (12 h runs, winds 40,50,60, the shipped adaptive ladder
# 15..5 s/km and below to 3 s/km) on the granted card; no new run starts after min(now + RUN_S, HARD_EPOCH).
set -u; source /work/tclock/s4/env-s4.sh
JOBS=$1; NPROC=$2; RUN_S=$3; HARD=$4; TAG=$5
O=$W4/results/rows; L=$W4/logs; mkdir -p $O $L
DL=$(( $(date +%s) + RUN_S )); [ $DL -gt $HARD ] && DL=$HARD
H=$(hostname); P=$W4/results/provenance-$H-c$GPU_MUTEX_CARDS.json
cat > $P <<EOJ
{"host": "$H", "box": "W1 (AWS us-west-2)", "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU_MUTEX_CARDS | head -1)",
 "card": "$GPU_MUTEX_CARDS", "driver": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)",
 "tree": "$(head -1 $S/TREE-COMMIT) + tools/terrain_clock_probe.py adaptive-sweep (step 4)",
 "probe_sha256": "$(sha256sum $W4/tools/terrain_clock_probe.py | cut -c1-64)",
 "probe": "tools/terrain_clock_probe.py adaptive-sweep --criterion blowup --seconds 43200",
 "python": "$($PY -c "import sys;print(sys.version.split()[0])")", "cupy": "$($PY -c "import cupy;print(cupy.__version__)")"}
EOJ
echo "$(date -u +%FT%TZ) start $TAG card=$GPU_MUTEX_CARDS nproc=$NPROC deadline=$DL" >> $L/events.log
cd $W4
for i in $(seq 1 $NPROC); do
  $PY $W4/tools/terrain_clock_probe.py adaptive-sweep --jobs $JOBS --out-dir $O --winds 40,50,60 --criterion blowup \
     --seconds 43200 --deadline $DL --provenance $P >> $L/worker-$TAG-c$GPU_MUTEX_CARDS-$i.log 2>&1 &
  sleep 2
done
wait
echo "$(date -u +%FT%TZ) end $TAG card=$GPU_MUTEX_CARDS rows=$(ls $O/*.json 2>/dev/null | wc -l)" >> $L/events.log
