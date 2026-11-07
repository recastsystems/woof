# Observation FDDA, native WRF 4.7.1 authority

The production option is `obs_nudge_opt=1`. Rust decodes native formatted
`OBS_DOMAIN<n>01` files and consecutive volumes, applies native projection,
QC, duplicate, pressure and wind rules, and publishes the immutable device
pack at setup. CUDA implements ERROB and NUDOB, native time windows,
surface and sounding weights, VIF and held tendencies. Fast numerics are
the default; strict arithmetic is a proof option.

## Source and licence

`source-pin.json` records every source URL, size, SHA-256 and licence
review. `WRF-NOTICE.txt` preserves the pinned WRF README's public-domain
statement. The listed observation, projection, surface-driver and constant
sources contain no overriding licence; no coefficient table is required.
No competitor code is used. Native WRF 4.6.1 distributes byte-identical
copies of the three assigned observation modules and can provide the
whole-run authority without changing their mathematics.

## Formatted reader check

```sh
python3 tools/fddaobs_wrf471_oracle/check_format.py /path/to/WRF /path/to/check-output
```

This offline developer command compiles the production decoder, its unit
tests and a Fortran probe using WRF's exact formatted READ descriptors.
It compares 360 float32 words, 36 integer words and 3,096 fixed-text bytes,
with one and two requested workers. Constructed cases cover native
padding, missing/QC sentinels, exponent formats and Gregorian UTC dates.
This check alone is not a real-input physics proof.

## Real native capture and strict replay

Copy an existing native WRF tree into private scratch before instrumenting
it. Verify its assigned source hashes against the pin first. Then:

```sh
python3 tools/fddaobs_wrf471_oracle/instrument_native.py /private/WRF --ktau 4
# Rebuild the entire native em_real executable with <= 8 make jobs,
# -O0 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations.
# Run a real, complete forecast there with obs_nudge_opt=1, OBS_DOMAIN101,
# and obs_ipf_errob=obs_ipf_nudob=1. Capture files land in its run directory.
python3 tools/fddaobs_wrf471_oracle/replay_native.py /private/captures /private/replay.json
```

The instrumenter only adds stream dumps around native ERROB and NUDOB;
it does not replace the routines. Captures include native tile bounds,
controls, reports, surface ratios, fields, VIF heights, input tendencies
and weight history, plus the native outputs. The replay compiles the
production CUDA with `woof.wrf_exact.install()`, preserving disabled
contraction, FTZ and fast-math settings. It compares output words and
counts changed tendencies so zero forcing cannot qualify accidentally.
Its offline prefix calculation preserves NUDOB's first-zero surface VIF
exit. That calculation is CUDA in production.

For the Rust admission comparison, run the production `rw_fddaobs` with a
JSON request containing the native grid attributes, original UTC start,
`rovg`, file path, output pack and window controls (the schema lives in
`rw-obs/src/bin/fddaobs.rs`). The captured native window must be a prefix
of the admitted pack:

```sh
rw_fddaobs request.json
python3 tools/fddaobs_wrf471_oracle/replay_policy.py observations.pack lane-errob-TILE-in.bin policy.json
```

The lane's actual two-tile real ASOS capture compares 455,408 CUDA output
words and 108 Rust admission words with zero differences. It exercises
surface wind, temperature and moisture; it does not claim a native replay
of every sounding branch. No native undefined residue was used.

## Forecast and state contract

A stock WRF real.exe directory is reachable through
`woof go /path/to/real-output --outdir /path/to/forecast`. Its producing
namelist supplies the observation controls and observation files. The
ordinary U/V/T/QVAPOR history fields carry the result; native WRF defines
no additional observation FDDA history fields. Native pressure nudging
(`obs_nudge_pstr` and `obs_coef_pstr`) is obsolete and remains a no-op.
Pressure reports still enter temperature/humidity conversion.

Each domain/member owns independent report QC, innovation, ratio, weight
and CUDA clock arrays. These are in the common restart/checkpoint
manifest; immutable forcing content and CUDA source bind the setup
fingerprint. Nested domains use their own `OBS_DOMAIN<n>01`, projection
and cadence. Both feedback modes retain their ordinary coupling. The
packed ensemble driver routes observation FDDA through the existing
member-local driver, keeping the complete operator and private reports.
Memory admission counts decoded levels with Rust and prices all CUDA
carriers, including working arrays and graph clock storage.

A strict module replay does not establish full-model bitwise parity.
The lane delivery report records the complete native/GPU forecasts,
Rust maps beside observations, measured stage times, continuation checks
and their practical limits.
