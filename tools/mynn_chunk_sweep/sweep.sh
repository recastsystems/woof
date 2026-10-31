#!/bin/bash
# Sweep the MYNN column chunk on one card and prove the forecast does not move.
#
#   sweep.sh [--dry] [--chunks "4096 8192 16384"]
#
# The widths come from --chunks, or from CHUNKS in the environment, or from
# the default list.  The default list BRACKETS the shipped width, narrower
# and wider, and it is that way because of what the first sweep could not
# find: it started at the width then shipped and only ever widened, so the
# only thing it could return was a direction, and it ranked its own narrowest
# arm first.  Running downward put the optimum two widths below it.  A ladder
# with the shipped width at one end answers "which way", never "where".
#
# For each width in CHUNKS it runs STEPS root steps of a prepared control leg
# with WOOF_MYNN_COLUMN_CHUNK set to that width, records the per-root-cycle
# wall time from the run's own step log, hashes every wrfout frame the run
# wrote, and then deletes the frames.  The widths are run PASSES times, the
# even passes in reverse order, so an arm that won only because it ran while
# the card was cool or alone is visible as a disagreement between passes.
#
# It waits for an idle card and an absent mutex itself, claims OWNER, writes
# DONE (or FAILED with the tail of the log) and releases OWNER, so it can be
# started detached under tmux and left alone.
#
# What the integrate stage reads when it is finished:
#   $W/sweep.tsv      one row per arm: width, quiet median seconds per root
#                     cycle, the radiation populations, wall, peak memory,
#                     the width the RECEIPT says the run used, and how many
#                     card samples saw a process outside this engine's own
#                     install (foreign_seen) and how many saw the arm's own
#                     forecast (own_seen)
#   $W/digests/       one sha256 list per arm, over every frame of that arm
#   the IDENTITY line EQUAL, or the first arm and frame that disagreed
#   $W/summary.txt    the table and the verdict, as text
#
# IDENTITY IS THE POINT.  Every MYNN column kernel gives one CUDA thread one
# whole column, reads no neighbour, and holds no shared memory, no atomic and
# no per-chunk seed, so the width is workspace shape only and the frames must
# be bit-identical across every arm.  A digest that moves is a defect to find,
# not a tolerance to widen: stop the sweep and report the arm.
set -u

# Resolved BEFORE the cd below: summarise.py lives beside this script, and
# $0 is relative to the directory the sweep was started from.
HERE=$(cd "$(dirname "$0")" && pwd)

DRY=0; CHUNKS_CLI=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry)      DRY=1; shift ;;
    --chunks)   [ $# -ge 2 ] || { echo "sweep.sh: --chunks needs a quoted list of column counts" >&2; exit 64; }
                CHUNKS_CLI=$2; shift 2 ;;
    --chunks=*) CHUNKS_CLI=${1#--chunks=}; shift ;;
    -h|--help)  sed -n "2,34p" "$0"; exit 0 ;;
    *) echo "sweep.sh: unknown argument '$1'.  Usage: sweep.sh [--dry] [--chunks \"4096 8192 16384\"]" >&2; exit 64 ;;
  esac
done

# --- what to sweep, and where ---------------------------------------------
ROOT=${ROOT:-$HOME/mynn-chunk-sweep}
PREPARED=${PREPARED:-$ROOT/baseline/pair/ctrl/prepared}
BASECONF=${BASECONF:-$ROOT/baseline/pair/ctrl/experiment.toml}
ENGINE=${ENGINE:-$HOME/gpuwm-venv}
PY=${PY:-$ENGINE/bin/python}
W=${W:-$ROOT/mynn-chunk-sweep}
LANE=${LANE:-mynn-chunk-sweep}
CHUNKS=${CHUNKS:-"4096 8192 12288 16384 24576"}
PASSES=${PASSES:-2}
STEPS=${STEPS:-200}
DT=${DT:-12}                 # root time step, seconds; STEPS x DT = run_seconds
WARM=${WARM:-5}              # root cycles dropped before timing (JIT + first write)
KEEP_FRAMES=${KEEP_FRAMES:-0}
BOUND=${BOUND:-180}          # minutes, declared in the mutex
FREE_GIB=${FREE_GIB:-60}     # refuse rather than fill the disk mid-sweep
MUTEX=${MUTEX:-$HOME/card-lease}
OWNER=$MUTEX/OWNER

if [ $DRY = 1 ]; then
  CHUNKS=${CHUNKS_DRY:-"8192 16384"}; PASSES=1; STEPS=${STEPS_DRY:-10}
  W=$ROOT/mynn-chunk-sweep-dry; LANE=$LANE-dry; BOUND=20; FREE_GIB=20
fi

# An explicit list outranks both the environment and the --dry shorthand:
# it is the whole instrument when the question is a new one.
[ -n "$CHUNKS_CLI" ] && CHUNKS="$CHUNKS_CLI"

# The list sizes every arm's workspace and names every row of the ranking,
# so a width that is not a positive column count is REFUSED here rather
# than carried into an arm whose timing would belong to some other width.
# A repeated width is refused for the same reason: two arms of the same
# pass would overwrite one another's run directory and digest list.
seen=""
for chunk in $CHUNKS; do
  case "$chunk" in
    ""|*[!0-9]*) echo "sweep.sh: '$chunk' is not a positive column count; the width list sizes the MYNN workspace and labels the ranking, so it is refused rather than rounded" >&2; exit 64 ;;
  esac
  [ "$chunk" -ge 1 ] 2>/dev/null || { echo "sweep.sh: width '$chunk' is not at least one column" >&2; exit 64; }
  case " $seen " in *" $chunk "*) echo "sweep.sh: width $chunk is listed twice; each arm of a pass writes its own run directory and digest list, so a repeat would overwrite one" >&2; exit 64 ;; esac
  seen="$seen $chunk"
done
[ -n "$seen" ] || { echo "sweep.sh: no widths to sweep" >&2; exit 64; }

RUNSECS=$((STEPS * DT))
mkdir -p "$W/logs" "$W/digests" "$W/runs"
cd "$W" || exit 1
rm -f "$W/DONE" "$W/FAILED"
exec > >(tee -a "$W/sweep.log") 2>&1

utc () { date -u +%Y-%m-%dT%H:%M:%SZ; }
mark () { printf '%s\t%s\t%s\n' "$1" "$(utc)" "$(date +%s.%N)" >> "$W/stages.tsv"; echo "=== $1 $(utc) ==="; }
SAMPLER=""; OWNED=0
fail () {
  echo "FAIL: $*"
  { echo "FAILED $(utc): $*"; echo "--- last 60 lines of sweep.log ---"; tail -n 60 "$W/sweep.log"; } > "$W/FAILED"
  exit 1
}
cleanup () {
  [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null
  if [ $OWNED = 1 ] && [ -f $OWNER ] && grep -q "^$LANE " $OWNER; then rm -f $OWNER; echo "OWNER released $(utc)"; fi
  [ -f "$W/DONE" ] || [ -f "$W/FAILED" ] || { echo "FAILED $(utc): sweep exited without DONE"; echo "--- last 60 lines ---"; tail -n 60 "$W/sweep.log"; } > "$W/FAILED"
}
trap cleanup EXIT

echo "sweep start $(utc) dry=$DRY lane=$LANE pid=$$ chunks='$CHUNKS' passes=$PASSES steps=$STEPS engine=$ENGINE"
[ -x "$PY" ] || fail "engine venv missing at $ENGINE"
[ -d "$PREPARED" ] || fail "prepared control tree missing at $PREPARED"
[ -s "$BASECONF" ] || fail "experiment config missing at $BASECONF"
free_gib=$(df -BG --output=avail "$W" | tail -1 | tr -dc '0-9')
[ "${free_gib:-0}" -ge "$FREE_GIB" ] || fail "only ${free_gib}G free under $W; each arm writes the leg's frames before hashing them, so the sweep needs at least ${FREE_GIB}G.  Free space, or point W= at a larger filesystem, or set KEEP_FRAMES=0 with a smaller history interval in the config"

# --- the config: same prepared leg, shortened to STEPS root steps ----------
n=$(grep -c "^run_seconds = " "$BASECONF")
[ "$n" = 1 ] || fail "experiment.toml has $n run_seconds lines; the sweep edits exactly one"
sed -e "s/^run_seconds = .*$/run_seconds = ${RUNSECS}.0/" "$BASECONF" > "$W/experiment.toml"
PROOF=$PREPARED/proof.json
[ -s "$PROOF" ] || fail "no preparation proof at $PROOF"
FARGS="--prepared-root $PREPARED \
  --preparation-receipt-sha256 $(sha256sum "$PROOF" | cut -c1-64) \
  --experiment-config $W/experiment.toml \
  --experiment-config-sha256 $(sha256sum "$W/experiment.toml" | cut -c1-64) \
  --io-mode history"

# --- wait for an idle card and an absent mutex (poll 60 s, up to 6 h) ------
mark wait_start
n=0
while :; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " \n")
  if [ -z "$apps" ] && [ ! -f $OWNER ]; then break; fi
  n=$((n+1)); [ $n -ge 360 ] && fail "card busy or OWNER present for 6 h (apps=$apps owner=$(cat $OWNER 2>/dev/null))"
  sleep 60
done
echo "$LANE $(utc) pid $$ bounded $BOUND min" > $OWNER; OWNED=1
echo "OWNER claimed: $(cat $OWNER)"
echo "$(utc) pid $$" > "$W/STARTED"
mark start
nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader

# Foreign jobs (another lane's, which do not take this mutex) are RECORDED,
# never killed and never reniced.  The collect stage judges an arm that ran
# beside one; this file is how it can.
( while :; do
    printf '%s\t%s\t%s\n' "$(utc)" \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' ')" \
      "$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"
    sleep 5
  done >> "$W/cardwatch.tsv" ) &
SAMPLER=$!

unset PYTHONHOME
run_arm () {   # run_arm <chunk> <pass>
  local chunk=$1 pass=$2 tag out rc
  tag="c${chunk}-p${pass}"
  out="$W/runs/$tag"
  rm -rf "$out"
  mark "arm_${tag}_start"
  WOOF_MYNN_COLUMN_CHUNK=$chunk \
  PYTHONPATH="" PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 \
  CUPY_CACHE_DIR=$ROOT/cache/cupy CUDA_CACHE_PATH=$ROOT/cache/cuda \
  GPUWM_CASE_DATA_ROOT=$ROOT/case-data-root CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 \
  timeout 3600 "$PY" -m woof.prepared_domain_tree_forecast $FARGS --outdir "$out" \
    > "$W/logs/$tag.log" 2>&1
  rc=$?
  [ "$rc" = 0 ] || { tail -n 25 "$W/logs/$tag.log"; fail "arm $tag exited $rc"; }
  [ -s "$out/evidence/run-receipt.json" ] || fail "arm $tag wrote no run-receipt.json"
  ( cd "$out" && find wrfout -type f -name "wrfout_d??_*" | sort \
      | xargs -P 4 -n 4 sha256sum | sort -k2 ) > "$W/digests/$tag.txt"
  [ -s "$W/digests/$tag.txt" ] || fail "arm $tag wrote no frames to hash"
  "$PY" "$HERE/summarise.py" --arm --outdir "$out" --chunk "$chunk" \
     --pass-index "$pass" --warm "$WARM" --cardwatch "$W/cardwatch.tsv" \
     --own-prefix "$ENGINE" \
     --digests "$W/digests/$tag.txt" --tsv "$W/sweep.tsv" \
     || fail "arm $tag could not be summarised"
  [ "$KEEP_FRAMES" = 1 ] || rm -rf "$out/wrfout"
  mark "arm_${tag}_end"
  echo "--- $tag done: $(tail -1 "$W/sweep.tsv")"
}

for pass in $(seq 1 "$PASSES"); do
  order="$CHUNKS"
  if [ $((pass % 2)) = 0 ]; then
    order=$(echo "$CHUNKS" | tr ' ' '\n' | tac | tr '\n' ' ')
  fi
  echo "=== pass $pass order: $order"
  for chunk in $order; do run_arm "$chunk" "$pass"; done
done

mark summarise
"$PY" "$HERE/summarise.py" --collect --work "$W" > "$W/summary.txt" 2>&1
rc=$?
cat "$W/summary.txt"
case $rc in
  0) ;;
  2) fail "an arm did not run the width it was given: the override did not reach the run, so its timing belongs to a width nobody chose.  See the mismatch list in summary.txt, and check that WOOF_MYNN_COLUMN_CHUNK survives into the forecast process" ;;
  3) fail "THE FRAMES MOVED ACROSS ARMS.  The column chunk is workspace shape -- one thread per column, no neighbour read, no shared memory, no atomic, no per-chunk seed -- so a digest that moves with it is a defect in the walk, not a tolerance.  IDENTITY in summary.txt names the first frame that disagreed; keep $W/digests and the arm logs" ;;
  *) fail "collect exited $rc: $(tail -n 5 "$W/summary.txt")" ;;
esac
mark done
{ echo "DONE $(utc)"; echo "arms: $(( $(wc -l < "$W/sweep.tsv") - 1 ))"; \
  grep "^IDENTITY:" "$W/summary.txt"; } > "$W/DONE"
du -sh "$W" 2>/dev/null
