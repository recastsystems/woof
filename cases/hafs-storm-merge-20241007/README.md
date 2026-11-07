# Initialization comparison, 2024-10-07 00 UTC

These are experiment diagnostics, not generic source adapters. The source rows
and composition profiles carry HAFS acquisition and field mapping. No DA code
or analysis data are included here.

Scientific geometry: 218 x 180 parent at 12 km, 160 x 160 following child at
3 km, 49 levels, 30/7.5 s time steps, 72 hours. Parent projection is Mercator,
reference 24.531214313377404,-88.05, true latitude 22.5, standard longitude
-93.4; initial child parent offsets 44,52. The existing cyclone planner can
author this corridor from the requested cycle's HURDAT2 track. Observed later
positions determine geographic coverage only and never steer the live tracker.

Required arms are GFS, HAFS-A and GFS with HAFS-A storm insertion. A qualified
data-assimilation arm is separate and private. The merge center is the cycle vitals
22.5,-93.4, radius 150 km, transition width 150 km. HAFS supplements its missing
land-surface fields from same-cycle GDAS through the source composition table.

## Saved-history diagnostics

Run on the forecast host with this branch and its Python dependencies:

```
python score_history.py RUN/chain/run/wrfout diagnostics/fixes.json
python render_history_cf.py RUN/chain/run/wrfout diagnostics/maps 'WOOF; GFS baseline'
python plot_history.py --arm 'GFS=diagnostics/fixes.json' --verification SCORED_CASE.json --out diagnostics/charts.png
```

The reduction uses the engine's existing sea-level-pressure diagnostic, finds
centers without future observations and reports instantaneous intensity extrema.
It does not equate a model wind snapshot with the observed one-minute average.
The initial 150 to 300 km ocean ring is reported for spin-up inspection. The
maps use only `woof.rustwx`, common physical scales and explicit engine labels.
The condensate path covers only resolved pressure layers. Track/time-series
charts use matplotlib.

Export the fixes with the shared TC verification harness. Its ATCF interchange
rounds coordinates to 0.1 degree and wind/pressure to integer values. Archive
both the full-precision fixes and scorer output, and distinguish those scores.
Never infer a 72-hour result from a shorter archive.

`compare_off_history.py` requires every configured domain and hourly frame from
initialization through the requested endpoint, compares every array's raw
bytes, and fails for missing frames or variables. It compares numerical arrays,
not netCDF container bytes or creation timestamps. This tool has not yet been
used on a completed whole-case pair.

`preserve_small_results.py` stages only small metadata/log/configuration files,
with a 20 MB package cap. Transfer rendered pictures and these receipts back
promptly; do not transfer GRIB, history, restarts or environments.

## Evidence status

The first baseline has saved output through +3 hours. Direct HAFS 72-hour CPU
preparation passed. Full forecast comparisons, cost and whole-case OFF checks
are pending; no scientific qualification or default-on claim follows from
these diagnostics. See the lane delivery RESULT.md for the current evidence.

For a bounded-disk replay, `hash_history.py --root HISTORY --start
2024-10-07T00:00:00Z --hours 72 --out off-arrays.json` requires every hourly
frame and records SHA-256, shape and dtype for every raw variable. After
reducing and rendering that history, retain this manifest. On the base
replay, add `--reference off-arrays.json`; any missing frame, changed variable
or array digest fails. Report this as a complete numerical-array hash
comparison, distinct from comparing the netCDF container bytes.
