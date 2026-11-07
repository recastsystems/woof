# TEMF column oracle, WRF v4.6.1

## Current native-run proof

`source-pin.json` records the public WRF source, source hashes and licence.
Both TEMF modules are distributed under WRF's public-domain statement.
The modules are byte-identical in WRF v4.6.1 and v4.7.1. TEMF has no
external table file. Its 63 constant words are generated from the source.

`capture_native.py WRF_SOURCE OWN_OUTPUT` instruments owned copies at the
two native entry points. It records the first eight real columns before
and after the call, retaining every carried field. It changes no physics
statement. Compile those modules with `-O2 -ffp-contract=off -fno-fast-math`
and relink an owned native library and executable. The case receipt names
the remaining native build flags, namelist and input files.

The checked-in small JSON capture comes from a real WRF v4.6.1 forecast
of 1974-04-03 12 UTC, on a 250 by 200 by 49 grid, with a 60 second step.
No math shim was linked. In strict mode the production PBL matches all
7,136 output words and the production surface layer all 160 words:

```
GPUWM_WRF_EXACT=1 python tools/temf_wrf461_oracle/replay_native.py tests/fixtures/temf_real_native
GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_temf_real_native_replay.py
```

Run these under the machine's shared GPU reservation protocol. Fast
arithmetic remains the forecast default. Native PBL taper truncation,
solver denominator, temperature and Exner indexing, upper spacing and
mass-flux recurrence are retained. Surface F3 remains a declared
zero-moisture limit. This small first-call replay establishes no forecast
skill and does not cover every branch or a complete forecast trajectory.

## Historical synthetic oracle

The following tools and fixtures retain the prior synthetic experiment.
Its corrected PBL arm is not the production algorithm. Its math-shim
results must not be presented as untouched native-run proof. The current
qualification command is the native replay above.

The WRF numbers WOOF's TEMF pair is graded against: the Total Energy - Mass
Flux PBL (`bl_pbl_physics = 10`, `phys/module_bl_temf.F`) and its surface
layer (`sf_sfclay_physics = 10`, `phys/module_sf_temfsfclay.F`), which WRF
runs only together (`share/module_check_a_mundo.F:2287-2304`).

## What it builds

```
bash tools/temf_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
python3 tools/temf_wrf461_oracle/make_columns.py <CASES_DIR>
<BUILD_DIR>/<arm>/run_temf <CASES_DIR>/cases-<grid>.bin out-<arm>-<grid>.bin
```

Both modules are pinned by sha256 to the v4.6.1 files.  `run_temf.F90`
calls WRF's own entry points in WRF's order -- `temfinit` and
`temfsfclayinit` for the cold start, then per step `temfsfclay` and
`temfpbl` -- and keeps every carried field across steps as WRF's grid
does, including the INTENT(OUT) arrays a branch leaves unwritten (the
updraft fields of a stable column keep last step's words; so do PBLH and
KPBL outside the convective branch).  Between steps the column moves by the
PBL tendencies alone (`x = x + dt*rx`, a negative qv or qc set to zero), so
the state the scheme sees evolves and any wrong word compounds.

`make_columns.py` writes 216 columns: twelve regime families (convective
land dry and moist, cloud-topped, stable night, neutral windy, sea
convective and stable, very shallow, deep convective with the 4 km cap,
calm, high terrain, and edge cases: MAVAIL = 0, a dry column, southern
hemisphere, snow) on three vertical grids (35, 45 and 61 levels, dt 120,
60 and 20 s), 30 steps each.

## Arms

| arm | modules | flags | purpose |
| --- | --- | --- | --- |
| `stock` | byte-unmodified | WRF's own on this box: `-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize` | grades the transcription (kernel built with `-DTEMF_VERIFY_STOCK_WRF=1`) |
| `fixed` | with the six prior edits (`apply_fixes.py`) | same | historical comparison only |
| `stock-O0` | byte-unmodified | `-O0` | shows what -O2 folds |
| `stock-raw` | byte-unmodified, no shim | WRF's | glibc 2.39's own atanf/atan2f/log10f, reported |
| `only-F1` .. `only-F6` | one fix each | WRF's | what each fix moves on its own |

Every arm but `stock-raw` links `cr_shim.c`, which substitutes
correctly rounded atanf, atan2f and log10f. This is a declared arithmetic
divergence from the box's ordinary WRF build. The `stock-raw` arm keeps
the ordinary C-library calls and is separately recorded. WOOF uses its
own float32 math routines. These shimmed fixtures cannot be described as
untouched WRF arithmetic.

## The declared fixes

`apply_fixes.py` records the prior edits and their source lines.
The five PBL edits are retired from production. Historically:
F1 the updraft taper weight is REAL (INTEGER truncated the taper band to a
step), F2 the implicit solver's bottom row is conservative, F3 no 0/0 in
EXCH_TEMF and CHS at MAVAIL = 0, and three index slips against the scheme's
own "TEMF level k is WRF level k-1" convention: F4 the liquid-water
potential temperature takes its own level's temperature, F5 the updraft
temperature takes its own level's Exner function, F6 the turbulence-layer
depth DZT is the spacing of its own two mass levels.

## Into the tests

`digest_outputs.py` reduces an output file to per-step SHA-256 digests of
every field (`tests/fixtures/temf_wrf461/digests.json`, arms `fixed` and
`stock`); `tests/test_temf_wrf461_parity.py` replays the cases on the card
and compares digests.  `compare_arms.py` compares any two output files
word by word for diagnosis.


## Field-level GPU receipts

The current command replays the real native capture, under strict numerics:

```
GPUWM_WRF_EXACT=1 python -m tools.temf_wrf461_oracle.run_validation --out native-fields.json
```

Its receipt counts every output word and records native and device hashes.
The historical synthetic tools remain for provenance only. Their modified
source arms and math shim are separate from current qualification.
Raw native history and capture files are disposable after their small
JSON word fixture, receipts, logs and rendered comparisons are recorded.
