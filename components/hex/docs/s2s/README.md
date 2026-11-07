# Global S2S ensembles

`woof hex s2s` plans and runs a multi-week ensemble (7 days or longer, for
example 10, 30 or 90 days) on the 120 km or 60 km whole globe. The seven-day
minimum is the week-1 product window; verification covers the first ten days,
or the whole run when it is shorter. Short experiment windows use the same
surface updates, checkpoint recovery and complete-window products as longer
runs. They do not establish seasonal-skill qualification.
`--members COUNT` is the total output member count (1 to 1000, default 4),
balanced across `--lagged` offsets. Native-member ranges such as `1-4` belong
to the `global` door and are not S2S counts.
The default is fast numerics, daily history, surface updates
and checkpoints every five days. SPPT and SKEB follow the stochastic
policy: on for every ensemble with more than one output member
([stochastic physics](../stochastic-physics.md)). Lagged deterministic
ensembles enable both. Their amplitudes, scales and taper use the shared stochastic defaults.
Each card holds up to four
members in separate CUDA streams, within the existing co-tenancy admission.
Every visible card gets a queue. `--devices 0` restricts the queue to card 0.

```sh
woof hex s2s --row global-120km --source gfs --members 4 --days 30 \
  --cycle 2026-10-17T00 --devices 0 --work-dir s2s-20261017
```

Repeat the identical command after an interruption. The door keeps its state
in `s2s-state.json`, locks the work directory, finds each member's newest
checkpoint, and writes the resumed segment into a fresh attempt directory.
It preserves original initialization times and timestep numbers. Products
merge history segments by valid time, using the newer segment at overlaps.
Changing source, dates, members, cadence or seed needs another work directory.
Changing card placement is allowed. A checkpoint contains a Python pickle;
only use checkpoints produced by your trusted run.

With `--cycle latest`, the first launch saves the resolved cycle in
`s2s-plan.json`. Repeating the command in that work directory keeps the saved
cycle even after newer cycles post. Use a new work directory to start a newer
cycle. A dry-run in an existing directory also previews its saved cycle.

`--dry-run` prints the plan without creating the work directory or downloading
inputs. `--stage mesh|static|init|surface|forecast|products|verify` runs one
stage. `--history-hours 6` requests six-hourly history. `--prepared-init
YYYY-MM-DDTHH[:MEMBER]=PATH` reuses a prepared init and validates its source
receipt. Keep one prepared input copy per host, under `/workspace/shared-prepared`.
Device indices in `--devices` refer to the inherited `CUDA_VISIBLE_DEVICES`
allocation. For example, `--devices 0` selects the first GPU granted by a pool,
including when the pool identifies it by UUID. Forecasts, products and
verification retain that allocation; a selection outside it is refused.

Sources come from the init-source table, including `gefs`, `ifs-ens`, `gfs`,
`era5` (keyless: public ARCO analyses through the engine Rust fetcher, no CDS account or
key; `era5-arco` is its alias and `era5-cds` the keyed CDS route), `cfs`, and compatible
Earth-2 generated forecast rows. ERA5 is under the Licence to Use Copernicus Products,
which permits commercial use with attribution; the attribution is in every receipt. See
[generated initialization](../init-sources.md#generated-earth-2-forecasts)
for model licences and forecast clocks. Research rows require the exact
`--accept-licence` value recorded in their table entry.
`--members` is the total ensemble count. `--lagged 0,24,48` balances that count
across initialization offsets in hours. Native member numbers restart at 1
for each lag. GFS and CFS default to offsets `0,6,12,...`. A lag selects
`source_cycle = base_cycle - lag`, `lead_hours = lag`, and the common
`valid_time = base_cycle`. The source door prepares that forecast state;
every S2S member then starts at the common valid time, without model warm-up.
Source cycle, source member, lead, lag and valid time are recorded in each
member manifest. The source table validates published cycles and forecast leads.
ERA5 publishes hourly analyses. Its default lags are `0,6,12,...`, with each
analysis fetched at source lead zero and its actual valid time. The model
integrates each earlier analysis through the lag before writing history on
the common forecast window. Member `valid_time` and `start_time` describe the
analysis; provenance `window_start` describes the common window, and
`spinup_hours` records the integration between them. Run length and checkpoint
offsets include that integration. Hourly lags such as `--lagged 0,1,2,3` are
also accepted. Surface forcing covers the earliest analysis through the common
end. Source forecast rows retain the common-clock behavior described above.
An explicit `--lagged 0` shares one init with independent stochastic seeds.
Member provenance also records seed, mesh, physics, surface series and every restart.

The short-range global door also accepts generated stochastic identities on a
deterministic source, for example `global --source era5 --members 1-8`. It builds
one input per source cycle and lead, then binds that input, its provenance and
an independent stochastic identity to each member. Combining `--members` and
`--lagged` takes their Cartesian product. Disabling both SPPT and SKEB refuses
multiple unperturbed members of the same input because they would repeat one
forecast and understate spread.

Surface forcing defaults to the CFS forecast cycle 48 hours before the earliest
member init. The Rust producer fetches six-hourly SST and sea-ice records through
the end of the run, using the public NOAA forecast mirror and the NCEI archive.
Each record carries its actual source cycle, lead and valid time. This covers
future dates without waiting for future analyses. `--surface-cycle YYYY-MM-DDTHH`
selects another CFS cycle. `--surface-source gfs|era5` selects dated analyses for
a historical window; every requested analysis must exist. `--surface-series PATH`
supplies an existing native series instead. Interrupted production restarts into
a fresh private output; complete series are reused after matching their plan.

New runs stage the native file at `WORK/surface_series.nc`, matching the surface
consumer's automatic discovery name. It contains padded `xtime(Time,StrLen)`,
`sst(Time,nCells)` in Kelvin and `xice(Time,nCells)` as fractions, with uniform
six-hour records covering both endpoints. Existing `WORK/surface-series.nc`
files remain selected when the canonical filename is absent; an explicit
`--surface-series PATH` takes precedence. Resident members receive the resolved
path through `WOOF_HEX_SURFACE_SERIES`, so their nested attempt directories
use the same file and surface updating turns on automatically.

For separate preparation, use the same S2S arguments with `--stage surface`,
then repeat the full command. An externally built native series can instead
be selected with `--surface-series WORK/surface_series.nc`. The producer's
direct interface is `woof hex surface-series --source cfs --cycle START
--forecast-cycle CFS_CYCLE --hours HOURS --static STATIC.nc --out
WORK/surface_series.nc --threads N --jobs N --interval-hours 6`.

## Products and interpretation

Temporal reductions, pressure interpolation, MSLP diagnostics, remapping and
ensemble reductions run on the GPU. Rust reads and writes NetCDF and renders
all weather maps. Wheel builds include the S2S renderer in `woof/hex/libexec`.
`tools/build_engine_bridges.sh ENGINE_SOURCE OUT_DIR` builds it with the other
bridges, applies the current renderer extension in a private source copy, and records
their hashes and the binary hash. The engine source stays read-only. A source
wheel build can also use `GPUWM_HEX_ENGINE_SOURCE=PATH`; no runtime renderer
environment override is needed for the resulting wheel. The accompanying
[renderer patch](engine-patches/0001-rw-wrfbatch-s2s-window-scales.patch) adds
S2S mean/anomaly scales and an `s2s-window` ABI token. The door refuses an older
renderer that would paint Z500 or MSLP on its generic 0 to 100 scale.
Apply the [global projection patch](engine-patches/0002-rw-wrfbatch-global-ensemble-projection.patch)
after the scale patch. It pairs a global Robinson extent with its matching
inverse raster transform. Without it, a degree extent is sampled as metres
and the map becomes a nearly constant equatorial fill. The door also requires
its `global-ensemble-projection` ABI token.

Products include every complete seven-day window and complete weeks 1–2,
3–4 and 5–6 windows. A 30-day run does not contain weeks 5–6. Window means
use trapezoids of the written daily or six-hourly diagnostics. These are
sampled means, not averages of unwritten model steps. Precipitation is the
difference of cumulative `rainc + rainnc`, reported as mean mm/day. Decreasing
cumulative precipitation, a missing scheduled time or a missing member refuses
the product. Spread is population standard deviation of member window means.
Below-ground Z500 stays masked. Remapping selects the nearest native cell on
the sphere after temporal reduction; it is not a conservative precipitation
remap.

Each window file also retains every member's reduced field as `FIELD_members`,
with the roster order in `s2s_member_ids`. The small JSON receipt records finite
counts and ranges of each mean, spread, anomaly and probability plane.

By default the builder acquires the
required fields from the public WeatherBench2 ERA5 analysis archive and builds
matching calendar window means across 1990-2019, excluding the target year.
Rust decodes Zarr chunks concurrently; CUDA computes window distributions,
linear empirical terciles and nearest-grid remapping. The source grid is
1.5 degrees; remapping does not add resolution. The receipt records source
URLs, hashes, baseline years, calendar, thread count and artifact hash. Raw
chunks stay in memory and are released after each source/year reduction.

Temperature anomalies put the ERA5 normal on the model terrain before subtraction:
`normal_on_model = normal_ERA5 - 0.0065 * (model_height - ERA5_height)`,
with heights in metres and a fixed lapse rate of 6.5 K/km. The model height is
the native lowest `zgrid` interface, remapped with the same native-cell index
as temperature. All ensemble members must have identical surface terrain.
The ERA5 surface geopotential comes from the matching conservative 240 by 121
analysis grid and is divided by 9.80665 to obtain metres. Its latitude and
longitude words must equal the climatology's coordinates. Acquisition and
remapping retain source URLs and hashes. This product adjustment removes the
elevation comparison term; it does not diagnose local lapse rates or inversions
and does not repair forecast temperature drift, including over sea ice.

`--climatology PATH` supplies another ERA5 window climatology. Alternatively set
`WOOF_HEX_S2S_ERA5_CLIMATOLOGY_URL` to fetch a precomputed window artifact. It must declare
`s2s_climatology_source=ERA5`, `s2s_climatology_start_mmdd=MM-DD`,
matching `XLAT`/`XLONG` coordinates and field units, and hold one product-grid plane per window and
field, for example `weeks_1_2_temperature2`, in K, mm/day, m, hPa and m/s.
It must include matching UTC start hour and historical tercile thresholds, as
detailed below.
It must also contain finite `orography(south_north,west_east)` in metres,
representing the terrain of its temperature normal. Missing or invalid terrain
refuses ERA5 anomaly products. If automatic ERA5 acquisition is unavailable,
the door refuses historical anomalies and terciles. Each ERA5 product retains
`model_orography`, `climatology_orography`, the adjusted `temperature2_baseline`
and the correction convention. Temperature anomaly probabilities use that same
adjusted baseline. An uncorrected old temperature-anomaly map is not a shipping
product. Redraw from the retained member means and exact model terrain.

The completed historical galleries predate this correction. Their temperature
anomaly and above-normal probability maps are held diagnostic evidence, pending
redraw with exact terrain. Their forecast timings and history/restart checks
retain their original source scope. A separate suspected sea-ice energy-balance
gap on the pinned Noah-MP route remains unqualified; updating SST and ice
categories alone does not establish complete ice thermodynamics.

A separate card-0 CUDA check on 2026-10-17 passed the terrain adjustment and
area-weighted SST/ice reduction oracles, plus fresh ERA5 normal and matching
orography acquisition. Whole wrapper wall was 3.239 s, peak own VRAM 552 MiB,
under a 4096 MiB pool reservation with graphs off. Acquisition used 16
fetch/decode workers and three terrain workers; reductions and remapping ran
on the GPU. The 7,039,848-byte normal was hashed and deleted after checking.
This qualifies those operations, without a new forecast, full product-builder
or renderer check, historical redraw, or sea-ice energy-balance qualification.
Receipts: `evidence/surface-final-integration/gpu-terrain/` in the lane delivery.

`tools/audit_s2s_surface.py --static STATIC --grid GRID --series SERIES
--source-receipt RECEIPT --history HISTORY --out OUT` prints daily global sea-ice
area, area-weighted SST and ice fraction north of 60N and south of 60S, and model
2 m temperature weighted by ice-covered area. It compares actual written SST/ice
words with the matched producer's remapped CFS series, applying the 0.02 consumer
cutoff separately from raw source ice. It uses the shipping Rust reader, four
history-reader threads and CUDA reductions. It refuses missing daily coverage
or unit-sphere areas. Classification and albedo checks explicitly state when
those optional history fields were not written. This is not an independent
re-decode of the source GRIB, and requires the original registered static/grid
and matching series. It never regenerates weather or substitutes another grid.

Verification uses the existing observation scorecard, bounded to the first
ten days. Missing or future analyses retain missing coverage in the scorecard.
A successfully integrated forecast is not evidence of seasonal forecast skill.

Direct analysis verification also compares T2, Z500 and diagnosed MSLP on the
native mesh at day 0 through day 10. GFS analysis states are prepared in parallel
where available. `--analysis-reference era5:YYYY-MM-DDTHH=INIT.nc` supplies an ERA5
analysis with its source receipt, without credential access. An existing GFS
member init at the required date is reused. GPU area-weighted bias and RMSE are
written per member and for the ensemble mean. These compare the projected
analysis state, including pressure recovered from its dry density, theta and vapor.
Day 0 is an initialization check. Instantaneous analyses cannot verify accumulated
precipitation; that field remains with the observation scorecard. Future or
unavailable analyses and unsupplied ERA5 dates are recorded as missing.

## Cost measurements

Measured on physical card 0, RTX PRO 6000, with surface updating, SPPT and SKEB ON,
public timestep defaults, daily history and fast numerics. Each row is one 24-hour
batch including process startup. Peak VRAM is this lane's process memory sampled
at 1 Hz; other tenants are excluded from this VRAM number. Card 0 was shared
with the speed lane, so wall times include the observed contention. Per-row receipts
record the sampled other-tenant VRAM range. No five-day checkpoint occurs in these
batches. This is a one-day cost extrapolation.
Exact commands: delivered `scripts/run-benchmarks.sh` and `scripts/benchmark-s2s.py`.
Per-row evidence: delivered `evidence/remote/benchmarks/ROW-mN/cost-receipt.json`.

| Row | Resident members | Wall seconds / member-day | Peak VRAM GiB | 20 members × 45 days, card-hours |
| --- | ---: | ---: | ---: | ---: |
| global-120km | 1 | 41.171 | 13.477 | 10.293 |
| global-120km | 2 | 34.122 | 24.451 | 8.531 |
| global-120km | 4 | 31.088 | 40.199 | 7.772 |
| global-60km | 1 | 164.484 | 24.535 | 41.121 |
| global-60km | 2 | 142.042 | 39.680 | 35.510 |
| global-60km | 4 | 146.718 | 70.668 | 36.680 |

For a measured batch wall time `t` with `m` members and `d` days, seconds per
member-day is `t / (m*d)`. The corresponding 900 member-day estimate is
`900*t/(m*d*3600)` card-hours, assuming full batches of the measured size.
It excludes input preparation, products, verification and interruption recovery.

<!-- EXTRA-RUNTIME-MEASUREMENTS -->

## Maximum-duration and source-lead checks

A separate global-120km GFS run completed 90 days with one member, from
2026-10-17 00Z through 2027-01-15 00Z. Surface updates, SPPT and SKEB were ON;
CUDA graphs were explicitly disabled and the final policy records `replay: false`.
It used the public 600-second timestep, 91 daily frames and 18 five-day
checkpoint publications. The 361-record CFS forcing series extends through
source lead 2208 hours, from cycle 2026-10-15 00Z. This is a maximum-duration,
calendar and products check, not an ensemble calibration or seasonal-skill result.

| Measurement | Actual result |
| --- | ---: |
| Forecast stage, including five-day checkpoints | 2585.282 s |
| Forecast seconds / model day | 28.725 s |
| Complete command, including preparation/products/verification | 2645.818 s |
| Peak own process-tree VRAM, sampled at 1 Hz | 13.654 GiB |
| Complete weekly/fortnight windows | 15 |
| Rust-rendered maps | 300 |

This run had different shared-card load from the one-day cost table; it is not
a paired speed comparison. Its 15 windows include complete weeks 5 and 6.
Six additional full-size maps were inspected and captioned. A solo member has
zero ensemble spread and binary above-normal probabilities, without calibration.
The first-ten-day comparison still has only a day-0 GFS analysis available for
this current cycle. Evidence: delivered `evidence/remote/run90-runtime-summary.json`,
`evidence/remote/qualification90/`, `scripts/run-s2s90-eager.sh`, and
`evidence/S2S-90DAY-2026-10-17/`.

The earlier four-member 30-day qualification used older analyses and model warm-up.
After connecting the delivered source-lead contract, a separate four-member
24-hour GPU check passed from GFS leads 0/6/12/18 at the common valid time
2026-10-17 00Z, with surface updates and SPPT/SKEB ON, graphs OFF and two daily
frames per member. Its measured forecast wrapper wall was
121.308 seconds. The new source-lead contract was
not rerun for a full four-member 30 days. Evidence: delivered
`evidence/remote/lag-contract24-proof.json` and `scripts/run-lag-contract24.sh`.

<!-- SIX-HOUR-RUNTIME-MEASUREMENT -->

## Common-valid six-hour history and restart check

A separate solo global-120km GFS run completed 30 days with the corrected
source-lead contract: source cycle 2026-10-16 18Z plus lead 6 hours, common valid
2026-10-17 00Z. New-run discovery selected `surface_series.nc` without an
explicit path override. Surface updates, SPPT and SKEB were ON, graphs OFF.
The public command was interrupted after its day-5 checkpoint and resumed by
repeating the identical command. Forecast integration finished at 2026-11-16 00Z with all 121
six-hour frames, six product windows and 120 Rust maps. Six images were viewed
and captioned. Verification initially rejected the nonzero-lead member init as
an analysis. Commit 98771b44 corrected automatic analysis selection to require
zero lead; one further identical-command invocation fetched a separate GFS
analysis and finished postprocessing without another forecast attempt. This
is a recovered check, rather than a fresh end-to-end run of all corrected bytes.
Complete wall across all three invocations, including interrupted
work, restart, preparation, products and verification, was
920.716 seconds; peak own card-0 VRAM sampled at 1 Hz was
13.643 GiB. Six five-day checkpoint publications were
hashed before raw-output disposal. The native CFS forcing had 121 records,
produced in 3.681 seconds with 48 decode
workers, 48 remap threads and 16 fetch jobs.

This measures solo source-lead, canonical filename, six-hour products and
restart behavior. Population spread is zero and probabilities are binary;
it does not repeat the full four-member 30-day qualification with the corrected
planner or establish seasonal skill. Evidence: delivered
`scripts/run-s2s30-six-hour.sh`, `evidence/remote/run30-six-hour-proof.json`,
`evidence/remote/qualification-six-hour/`, and
`evidence/S2S-SIX-HOUR-2026-10-17/`.


## CFS-initialized 24-hour GPU check, 2026-10-17

The generic CFS init-source row was exercised through public S2S preparation,
then the public resident forecast ran one global-120km member for 24 hours.
The CFS analysis is 2023-03-31 12Z, lead zero, through 2023-04-01 12Z.
All 144 steps completed, with two daily history frames. Surface updating,
SPPT and SKEB were ON, fast numerics were used, and both CUDA graph settings
were OFF. The five native surface records cover both endpoints, supplied by
CFS forecasts initialized 2023-03-29 12Z. The final receipt records all five
surface reads with one packed H2D transfer and zero D2H bytes per read.

| Stage | Measured wrapper wall, seconds | Parallelism |
| --- | ---: | --- |
| CFS initialization | 19.710449 | One distinct source state, 48 Rust threads |
| Surface production | 1.085002 | Five decoder workers with 10/10/10/9/9 threads; 48 remap threads; 16 fetch jobs |
| 24-hour forecast | 38.342863 | One resident GPU stream, 48 host threads |

Peak own VRAM was 13790 MiB, sampled at 1 Hz on hex physical card 0. The
producer's internal wall was 0.549561 seconds. The wrapper's single initial
surface sample missed that subsecond parallel work; its measured one-thread
sample is not the producer's worker count. Initialization peak host usage
was 5.892 cores and 152 threads; forecast peak was 3.802 cores and 128 threads.
The focused harness status label remains `init`; exact forecast argv and driver
receipts identify the resident work. The final receipt has 144 stochastic calls.

Exact commands: delivered `scripts/run-cfs-init-check.sh`, then
`scripts/run-cfs-check24.sh`. Assertions, bounded archive and disposal:
`scripts/cfs-check24-proof-and-dispose.py`. Evidence:
`evidence/remote/cfs-check24/proof.json`, its native source/producer receipts,
and the three `cfs-*-measured` folders. All 13 runtime hashes match the
current source. Six consumed raw files totaling 779380607 bytes were logged
and deleted; the final raw scan is empty. Retained archive hashes match.

This closes the CFS-initialized 24-hour GPU source check. It does not establish
a full 30-day CFS ensemble, ten-day analysis skill or seasonal skill. This CFS check predates the native-GEFS four-member 30-day qualification below.

## Current-runtime native GEFS 30-day qualification, 2026-10-17

Four native GEFS members 1 to 4, lead 0/lag 0 at 2026-10-17 00Z, completed 30 days on global 120 km through the public S2S door. Surface updates, SPPT and SKEB were ON; numerics were fast and both CUDA graph flags were OFF. The harness interrupted after all four day 5 checkpoints and repeated the identical command once. All four resumed and completed 4320 steps, 31 daily frames each, six weekly/fortnight product windows and 120 Rust maps. Four distinct initial meteorological states were verified through the Rust reader. This is a fresh check of runtime 98771b44, separate from the older GFS qualifications.

Whole measured wall across both invocations, including restart and all stages, was 3472.186993 s, or 57m 52s (28.934892 s per member-day). Peak sampled own VRAM was 47826 MiB. These full-window GEFS numbers include observed shared-card load late in the forecast and are separate from the 24-hour controlled GFS cost table above. Initial preparation used four processes with 12 Rust threads each; forecast used four co-resident CUDA streams in one forecast process, GPU products four history readers, and the renderer 48 Rust threads. Exact per-stage timers and sampled peaks are retained in evidence/gefs30/.

Products use ERA5 1990-2019 WeatherBench2 smoothed six-hour climatology; 1.5 degree source; GPU window means and nearest-grid remap. Analysis verification measured 1 available reference, with 21 missing and 5 metric rows. This is not complete ten-day verification or a seasonal-skill claim. Eight original Rust figures and six contact sheets covering all 120 maps were viewed and captioned. Checkpoint hashes, compact health/physical-gate evidence and bounded product archives remain; raw history, checkpoints and prepared inputs were consumed after acceptance.

Exact launch scripts/run-s2s30-gefs.sh; acceptance scripts/gefs30-proof.py; evidence/gefs30/proof.json; figures evidence/S2S-GEFS-2026-10-17/. This new native-GEFS check does not repeat corrected four-lag GFS 30 days, long 60 km or multiple-card qualification. The 88 unchanged landing G0 failures/errors and unrun G1/G2 remain stated in REPORT.md.


### Two-day stochastic cost pair, shared card 0

On 2026-10-17 two serial 48-hour solo-member forecasts used the same borrowed
GEFS member-1 init, 2026-10-06 00Z, seed711, dt600s, GFS surface analyses and
daily history. Both completed288steps and three frames with finite physical
gates. SPPT/SKEB were OFF for control and ON for the second arm. Fast numerics,
graphs OFF, physical card0, 24576MiB pool reservation and16 host threads.

| Arm | Whole wrapper s | s/member-day | Peak own MiB |
| --- | ---: | ---: | ---: |
| control | 180.130723 | 90.065361 | 12748 |
| sppt-skeb | 184.377960 | 92.188980 | 13906 |


The ON arm retained 3 matched warm complete-step CUDA samples.
Aggregate pattern fraction 0.920314%; forcing fraction
0.988908%; every-sample pattern<5% criterion
true. Same-stream event elapsed includes
dispatch gaps and contention. Other card0 workloads varied between arms, so
the observed whole-wall change (+2.358%)
does not establish isolated stochastic overhead or a speedup. The receipt
retains `qualification_complete=false`; this cost check does not establish
physical or seasonal-skill qualification. Preparation,
products, checkpoint/restart and analysis skill are outside these arm timers.
The controlled cost matrix above keeps its original scope.

Receipt: `evidence/stochastic-cost-pair48/proof.json` in the lane delivery.
All six runtime source hashes and archived receipt hashes match. Own histories
and surface forcing were consumed after acceptance; the borrowed shared init
was left untouched. No additional month run was made for this comparison.

### Complete GPU product-builder fixture, 2026-10-17

The current builder, terrain-adjusted ERA5 normal, Rust NetCDF writer and
rw_wrfbatch renderer were exercised together on 128 analytic cells, two
prescribed members and eight daily frames spanning seven days. All 20 maps
were written; seven were viewed and captioned in the delivered
`evidence/S2S-PRODUCT-FIXTURE-2026-10-17/`. These are labeled synthetic
fields, with clipping recorded in the captions.

The initial GPU wrapper took 7.746294 s and peaked at 928 MiB of own VRAM
under a 4096 MiB card-0 pool reservation, graphs OFF. It returned 1 after
rendering because a 1 mm Z500 oracle threshold rejected a 1.465 mm float32
reduction difference. The original failure is retained. CPU-only verification
of the same products passed at four float32 ULP (1.953 mm), taking 0.147145 s;
the builder was not rerun. Temperature mean/anomaly errors were below
0.000032 K; the two-member above-normal fraction and rain mean/spread matched
the independent oracle exactly. MSLP was checked for finite positive values
and zero member spread, without an independent SLP-value oracle.

Receipt: `evidence/full-product-check/retained/evidence/proof.json`.
GPU diagnostics/reductions and remapping used one card, history reads four
workers and rendering a 48-thread Rust budget. All 20 owned raw files,
12663604 bytes, were consumed after verification; the lease was released.
This qualifies the complete builder on an analytic fixture. Historical
temperature anomaly/probability map holds, fresh native-weather correction
qualification, the native-weather daily audit and sea-ice energy balance remain open.

### Daily surface-audit CLI transport check, 2026-10-17

The CLI now restores the stored NetCDF dtype after the Rust bridge's float64
numeric transport. SST/ice word comparisons require stored float32 and disable
CF mask/scale transforms; genuinely float64 SST is rejected. This fixes a
valid-file failure before the reduction stage without changing model physics.

Eight cases passed through the actual CLI on eight analytic cells and three
daily frames spanning two days: reordered cell IDs, absent optional fields,
intentional SST/ice/classification/albedo mismatches, missing day, wrong IDs,
stored float64 SST, unit-sphere areas and incomplete producer coverage.
The independent scalar summaries agree within 2.981e-8; each day uses one
64-byte packed Rust-reader H2D transfer with zero existing-state D2H bytes.
Whole wrapper wall was 7.711501 s, own peak 552 MiB under a 4096 MiB physical
card-0 pool lease, graphs OFF. History reads use four workers and reductions
CUDA; the short test had a 16-thread host budget. All 52 focused S2S CPU checks
passed, including three regressions for stored dtype and float-word recovery.

Receipt: `evidence/surface-audit-cli/retained/evidence/proof.json`. Fixture raw
files were consumed and the lease released. This qualifies the CLI transport,
reordering, summaries and rejection paths on synthetic files. An audit of
original native forecast history and sea-ice energy balance remains open.

## Weekly wind, historical terciles and regional views

The part A product builder includes earth-relative eastward and northward wind
at 925, 850 and 200 hPa. It reads `u_zonal`/`v_meridional` (or native
`uReconstructZonal`/`uReconstructMeridional`) from full histories and interpolates
in log pressure on CUDA. Targets outside the column's mass-level support are
masked. Each component has window mean, ensemble population spread, anomaly
and probability products. Missing wind histories refuse the product.

Automatic ERA5 baselines now use actual matching calendar windows in each year
1990 through 2019, excluding the target year. The baseline is the mean of those
historical window means. Its lower/upper tercile thresholds are linear empirical
quantiles at 1/3 and 2/3 over the same historical window means. They are not
quantiles of instantaneous states or a smoothed hourly normal. Temperature
normals and both thresholds receive the same exact model-terrain adjustment.
Precipitation uses six-hour accumulations ending strictly after the initial
endpoint; other fields use endpoint trapezoids. February 29 initialization uses
February 28 in non-leap baseline years; other dates retain their month/day/hour.
The receipt records the actual baseline years and source grid.

Categories are below (`x < q1`), near (`q1 <= x <= q2`) and above (`x > q2`).
Ties belong to near, including dry climatologies with coincident thresholds.
Every probability uses the fixed complete ensemble roster. Any missing member
or unsupported historical threshold masks all three probabilities at that point.
These are uncalibrated fractions, not a seasonal forecast skill claim.

`--product-region NAME:SOUTH,NORTH,WEST,EAST` adds regional maps alongside the
global maps. Repeat the option for additional views. West greater than east
crosses the date line. Crops preserve the global reduced values and reorder
crossing longitudes onto a continuous axis. All maps run through
`woof.rustwx.run_ensemble_product_renderer` and `rw_wrfbatch`/`rw_ensbatch`.
Build the added renderer capability with `tools/build_s2s_renderer.sh`; the
builder requires its `s2s-wind-terciles` ABI marker before writing maps.

The public archive's own `datasets/era5/LICENSE` names the **Licence to Use
Copernicus Products**, which permits commercial use with attribution, so
automatic acquisition is the default. Acquisition and generated products carry
the licence and its attribution. A supplied `--climatology` must
contain `s2s_climatology_start_hour` in addition to month/day, coordinates,
orography and field units, and `<window>_<field>_tercile_lower` and
`<window>_<field>_tercile_upper` for every field. An unavailable historical
baseline refuses anomalies and terciles. It does not substitute an ensemble
relative normal.

Zarr decompression runs through the small `rust/s2s-zarr` library and the
BSD-3-Clause c-blosc codec. The wheel builds and includes this library on Linux.
For a source checkout, set `WOOF_HEX_S2S_ZARR_LIBRARY` to its release `.so`.
The codec is resolved from system libblosc or the netCDF4 wheel's bundled
library; `WOOF_HEX_S2S_BLOSC_LIBRARY` selects an explicit codec library. NumPy
only wraps its unchanged decoded transport buffer. All climatological window
reductions, empirical quantiles and remapping run on CUDA; NetCDF transport
uses the Rust bridges. Compressed/decoded source chunks remain in bounded
memory. Source URLs, hashes, calendar, grid and parallelism are retained.
