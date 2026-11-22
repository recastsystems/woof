#!/bin/bash
# The 120 km install check and the 3 km storm example on one 96 GB card of compute capability 12.0 (for example an
# RTX PRO 6000 Blackwell). Needs: Python 3.14 with recast-woof[ad]; 96,600 MiB free on the card; 258,057 MiB of
# host memory for the 3 km sensitivity (81,920 MiB for the 120 km commands). The package fetches its own inputs
# (about 2.4 GB for the 120 km example and 2.7 GB for the 3 km one, once, each checked by SHA-256).
# Expected numbers: expected.json, "96gb". About 30 minutes in all on a first run.
set -u
worst=0   # the script ends with the highest exit status of its commands (0 only when each ended 0)
OUT=${1:-hex-ad-96gb}; mkdir -p "$OUT"
woof hex adjoint-test --out "$OUT/at"; rc=$?; echo "adjoint-test exit $rc"; [ $rc -gt $worst ] && worst=$rc
woof hex tangent      --steps 3 --out "$OUT/tl"; rc=$?; echo "tangent exit $rc"; [ $rc -gt $worst ] && worst=$rc
woof hex sensitivity  --steps 3 --metric theta_m --box 30,48,-105,-80 --levels 1-10 --out "$OUT/se"; rc=$?; echo "sensitivity exit $rc"; [ $rc -gt $worst ] && worst=$rc
woof hex param-grad   --steps 3 --check --out "$OUT/pg"; rc=$?; echo "param-grad exit $rc"; [ $rc -gt $worst ] && worst=$rc
woof hex sensitivity  --row hrrrv5-q3.96.38278.n40.60w91.30 --steps 59 --circle 40.6,-91.3,150 --levels 1-10 --out "$OUT/storm"; rc=$?; echo "storm sensitivity exit $rc"; [ $rc -gt $worst ] && worst=$rc
woof hex sensitivity  --session-stop
exit $worst
