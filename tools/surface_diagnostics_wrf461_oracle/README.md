# Public WRF RASM and solar referee

Sources are pinned to public WRF 4.7.1, with RASM and solar byte-identical to
4.6.1. `sources.json` records each licence, immutable URL and SHA-256; the
public-domain statement is checked before source instrumentation or extraction.
No external tables are needed. Existing Arm/Core-Math support retains its MIT notices.

Run `python verify_sources.py`. On an owned, pristine native tree run:

```
python capture_native.py --wrf-source /scratch/WRF --schema /scratch/schema.json
```

Build the complete strict native WRF with at most eight jobs. Set
`WOOF_SURFACE_CAPTURE_DIR` to an existing owned directory and run a real forecast.
The passive monitor captures three calls per patch. It changes no model fields
or clock. Use the produced schema and the original simulation start:

```
GPUWM_WRF_EXACT=1 python replay_capture.py /scratch/captures/*.bin \
  --schema /scratch/schema.json --start 2024-05-25T18:00:00 --out /scratch/replay.json
```

`build_cloud_oracle.py --out /scratch/cloud` extracts unchanged `cal_cldfra1` and `cal_cldfra2`
from the verified public radiation driver and compiles a C wrapper with strict
GNU Fortran options. Only scheme-number constants are supplied by the stub.
This separately checks the fused CUDA cloud fraction producer on real captured
profiles. Run `GPUWM_WRF_EXACT=1 python replay_cloud.py --capture REAL_CAPTURE
--library /scratch/cloud/cloud_oracle.so --out /scratch/cloud-replay.json`.
Runtime physics and all reductions execute on CUDA, with Rust NetCDF IO.

Lane evidence on 2026-11-01: 160 real RASM/solar calls, 173,352 output words,
zero differences; seven native cloud producer branches, 7,717,500 words, zero
differences. The real three-hour stock-namelist GPU forecast, native restart,
checkpoint continuation, both nest feedback modes and ordinary production
ensemble are recorded in the lane report. The run spans a diurnal-bin finish,
not a calendar month. Monthly averaging/reset follows the public native calendar
and has 62 separately checked native Gregorian calendar alarm cases, including
leap years, month ends, year rollover and fractional steps. Solar fields without native radius producers retain
the incoming native words. Strict numerics is proof-only; fast numerics is default.
