#!/usr/bin/env bash
# gpuE.sh: under card-lease with 4 RTX 5090 cards on box E. Case c3: full HRRR grid 1797x1057x50,
# HRRR physics, HRRR static file, 2026-10-03 21Z, 3 h, hourly history.
# Both runs use the MEASUREMENT copy (engine-meas: the lane engine with the card refusal disabled, scratch only), so the
# printed ADMITTED/REFUSED lines are the lane's price and the sampler gives the true per-card peak.
# A: [devices] 2x2.  B: [devices] 1x4.  Histories compared byte for byte (layout identity at full grid on 4 cards).
set -u
exec > >(tee -a /work/meeting-perf-a13/gpuE.out) 2>&1
trap "date -u +%FT%TZ > /work/meeting-perf-a13/GPU-DONE" EXIT
W=/work/meeting-perf-a13; c=$W/case/c3
source $W/envE.sh
echo "cards=$GPU_MUTEX_CARDS start=$(date -u +%FT%TZ)"
SMI=/usr/bin/nvidia-smi
PROOF=$(sha256sum $c/prepared/proof.json | cut -d' ' -f1)
SRC=$(python -c "import json;p=json.load(open('$c/prepared/proof.json'));print(p.get('input_manifest_sha256',''))")
CONT=$(python -c "import json;p=json.load(open('$c/prepared/proof.json'));print(p['prepared_cache']['content_sha256'])")
[ -n "$SRC" ] || SRC=$(sha256sum $c/prepared/source-evidence/input-manifest.json | cut -d' ' -f1)
run() {  # name grid
  name=$1 grid=$2; out=$W/runs/$name; rm -rf $out; mkdir -p $out
  ( while :; do $SMI --query-gpu=index,memory.used --format=csv,noheader,nounits -i $GPU_MUTEX_CARDS | awk -v t=$(date +%s.%N) '{print t", "$0}'; sleep 0.2; done ) > $out/nvml.csv 2>/dev/null &
  smi=$!
  t0=$(date +%s)
  PYTHONPATH=$W/sitex:$W/engine-meas python -m woof.prepared_single_domain_forecast --source rap-native --prepared-root $c/prepared \
     --proof-sha256 $PROOF --source-manifest-sha256 $SRC --prepared-content-sha256 $CONT \
     --experiment-config $c/experiment.toml --wps-namelist $c/namelist.wps --progress-format jsonl --io-mode history \
     --devices-table "{\"count\": 4, \"grid\": \"$grid\", \"ids\": [0, 1, 2, 3], \"transport\": \"auto\"}" \
     --outdir $out/out > $out/run.log 2>&1
  rc=$?
  kill $smi 2>/dev/null
  echo "$name rc=$rc wall=$(( $(date +%s) - t0 ))"
  ( cd $out/out && find . -name "wrfout*" -type f | sort | xargs -r sha256sum ) > $out/frames.sha256
  grep -E "ADMITTED|REFUSED|out of memory|OutOfMemory|Error" $out/run.log | head -12
}
run a-2x2 2x2
run b-1x4 1x4
diff <(cut -c1-64 $W/runs/a-2x2/frames.sha256) <(cut -c1-64 $W/runs/b-1x4/frames.sha256) && echo "LAYOUT HISTORIES IDENTICAL" || echo "LAYOUT HISTORIES DIFFER"
echo "end=$(date -u +%FT%TZ)"
