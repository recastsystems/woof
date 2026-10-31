#!/bin/bash
# recheck.sh CELLS NPROC : run inside card-lease tclock --cards 1 --est S.  Each line of CELLS
# (dx crest ridge_slope wind sound_steps settings step_s) runs once through tools/terrain_clock_probe.py fixed
# --criterion blowup --courant for three hours on the lane tree2 (TREE-COMMIT2); NPROC at a time on the granted card.
set -u; source /work/tclock/env.sh; export PYTHONPATH=/work/tclock/tree2; cd /work/tclock/tree2
CELLS=$1; NPROC=$2; O=$W/results/recheck; mkdir -p $O
echo "$(date -u +%FT%TZ) recheck start card=$GPU_MUTEX_CARDS cells=$(grep -c . $CELLS)" >> $W/logs/sweep-events.log
nl -ba $CELLS | while read -r n dx crest rs wind sub st step; do
  f=$O/$(hostname)-$(basename $CELLS .txt)-$n.jsonl; [ -s $f ] && continue
  echo "$PY tools/terrain_clock_probe.py fixed --dx $dx --crest $crest --ridge-slopes $rs --winds $wind --sound-steps $sub --steps $step --settings $st --criterion blowup --courant --out $f > /dev/null 2>> $W/logs/recheck-err.log"
done | xargs -P $NPROC -I{} bash -c "{}"
echo "$(date -u +%FT%TZ) recheck end card=$GPU_MUTEX_CARDS files=$(ls $O | wc -l)" >> $W/logs/sweep-events.log
