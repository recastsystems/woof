#!/bin/bash
# sweep6.sh JOBS NPROC RUN_S : inside card-lease tclock --cards 1 --est S. Terrain-clock local-face lane, STEP 2
# rows: every candidate row measured directly under the lead-decided held test ("blowup": the run stays finite for the
# full 3 h and peak w stays under 10 x the old bound 4 x wind x slope + 20). First (once per host) re-checks the
# instrument: 3 km / 3000 m / ridge 0.1 / 40 m/s / 24 s / 4 substeps / 3 h at the map criterion must give step 1s
# peak w 34.655418395996094 to the bit (box S RTX 5090 and box G RTX PRO 6000 both did). Rows land in $W/results/rows6/.
set -u; source /work/tclock/env6.sh
JOBS=$1; NPROC=$2; RUN_S=$3
DEADLINE=$(( $(date +%s) + RUN_S )); O=$W/results/rows6; mkdir -p $O $W/logs
H=$(hostname)
V=$W/results/validate6-$H-c$GPU_MUTEX_CARDS.jsonl
if [ ! -s $V ]; then
  cd $T && $PY tools/terrain_clock_probe.py fixed --dx 3000 --crest 3000 --ridge-slopes 0.1 --winds 40 --sound-steps 4 --steps 24 \
     --seconds 10800 --courant --out $V > /dev/null 2>> $W/logs/validate6.err
  $PY -c "import json,sys; r=json.loads(open(sys.argv[1]).readline()); ok=r[\"peak_w\"]==34.655418395996094; print(\"VALIDATE card\", sys.argv[2], \"OK\" if ok else \"MISMATCH\", repr(r[\"peak_w\"])); sys.exit(0 if ok else 3)" $V $GPU_MUTEX_CARDS >> $W/logs/sweep6-events.log || { echo "$(date -u +%FT%TZ) validate mismatch card $GPU_MUTEX_CARDS, no sweep" >> $W/logs/sweep6-events.log; exit 3; }
fi
P=$W/results/provenance6-$H-c$GPU_MUTEX_CARDS.json
cat > $P <<EOJ
{"host": "$H", "box": "W1 (AWS us-west-2)", "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU_MUTEX_CARDS | head -1)",
 "card": "$GPU_MUTEX_CARDS", "driver": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)",
 "tree": "$(head -1 $T/TREE-COMMIT) $(sed -n 2p $T/TREE-COMMIT)", "probe": "tools/terrain_clock_probe.py sweep --criterion blowup",
 "python": "$($PY -c "import sys;print(sys.version.split()[0])")", "cupy": "$($PY -c "import cupy;print(cupy.__version__)")"}
EOJ
echo "$(date -u +%FT%TZ) start card=$GPU_MUTEX_CARDS nproc=$NPROC deadline=$DEADLINE" >> $W/logs/sweep6-events.log
cd $T
for i in $(seq 1 $NPROC); do
  $PY tools/terrain_clock_probe.py sweep --jobs $JOBS --out-dir $O --winds 20,30,40,50,60 --criterion blowup \
     --rungs-per-km 6.5,6.0,5.5,5.0,4.5,4.0,3.5,3.0 --deadline $DEADLINE \
     --provenance $P >> $W/logs/worker6-c$GPU_MUTEX_CARDS-$i.log 2>&1 &
  sleep 1
done
wait
echo "$(date -u +%FT%TZ) end card=$GPU_MUTEX_CARDS rows=$(ls $O/*.json 2>/dev/null | wc -l)" >> $W/logs/sweep6-events.log
