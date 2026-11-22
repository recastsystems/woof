#!/bin/bash
# The 12 km example end to end on one 32 GB card of compute capability 12.0 (for example a GeForce RTX 5090).
# Needs: Python 3.14 with recast-woof[ad] (an environment of its own, not on top of [gpu]), 28,700 MiB free on the card, 47,104 MiB of host memory, 3.4 GB of
# disk in the hex cache. The package fetches its own inputs (1.20 GB, once, each checked by SHA-256).
# Expected numbers: expected.json, "32gb". About 12 minutes in all on a first run with the
# inputs fetched from GitHub (693 s measured on one RTX 5090 with 0.6.0's published wheel).
set -u
worst=0   # the script ends with the highest exit status of its commands (0 only when each ended 0)
OUT=${1:-hex-ad-32gb}; mkdir -p "$OUT"
ROW=hrrrv5-q12.96.2890.n40.60w91.30
woof hex adjoint-test --row $ROW --steps 15 --at 40.6,-91.3 --radius-km 150 --out "$OUT/cut12"; rc=$?; echo "adjoint-test exit $rc"; [ $rc -gt $worst ] && worst=$rc
woof hex sensitivity  --row $ROW --steps 15 --circle 40.6,-91.3,150 --levels 1-10 --out "$OUT/cut12s"; rc=$?; echo "sensitivity exit $rc"; [ $rc -gt $worst ] && worst=$rc
python "$(dirname "$0")/own_measure.py" "$OUT/own"; rc=$?; echo "own_measure exit $rc"; [ $rc -gt $worst ] && worst=$rc
exit $worst
