#!/bin/bash
# validate.sh : run inside card-lease tclock --cards 1 --est 900.  Re-checks the instrument.
#  A: the SHIPPED probe (tools/terrain_clock_probe.py at integrate/2.8.7 5bd9a0085, unmodified, as orig_probe.py) on cells
#     where the shipped map holds a step the lane's sweep saw stop: 3 km / 3000 m / ridge 0.1 / 40 m/s at 8.0 and 6.0 s/km
#     (map entry 8.0, tried 13.33), and 2 km / 4500 m / ridge 0.1 / 40 m/s at 5.0 s/km (map entry 5.0), each for the map's
#     half hour (1800 s) and for three hours (10800 s), four substeps.
#  B: the lane probe (tree2) on the same cells with and without the Courant recording, to show the recording does not
#     change the run.
set -u; source /work/tclock/env.sh; O=$W/results/validate; mkdir -p $O; cd $W/tree
for s in 1800 10800; do
  for c in "3000 3000 0.1 40 8.0" "3000 3000 0.1 40 6.0" "2000 4500 0.1 40 5.0"; do set -- $c
    echo "{\"tool\": \"shipped 5bd9a0085\", \"dx\": $1, \"crest\": $2, \"ridge_slope\": $3, \"wind\": $4, \"per_km\": $5, \"seconds\": $s, \"result\": $($PY $W/orig_probe.py cell --dx $1 --crest $2 --ridge-slope $3 --wind $4 --per-km $5 --sound-steps 4 --seconds $s 2>/dev/null | tail -1)}" >> $O/validate.jsonl
  done
done
cd $W/tree2; export PYTHONPATH=$W/tree2
for c in "3000 3000 0.1 40 24.0" "2000 4500 0.1 40 10.0"; do set -- $c
  $PY tools/terrain_clock_probe.py fixed --dx $1 --crest $2 --ridge-slopes $3 --winds $4 --sound-steps 4 --steps $5 --seconds 10800 --out $O/lane-nocourant.jsonl > /dev/null 2>&1
  $PY tools/terrain_clock_probe.py fixed --dx $1 --crest $2 --ridge-slopes $3 --winds $4 --sound-steps 4 --steps $5 --seconds 10800 --courant --out $O/lane-courant.jsonl > /dev/null 2>&1
done
echo "$(date -u +%FT%TZ) validate end" >> $W/logs/sweep-events.log
