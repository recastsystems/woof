# NextGen hydrology in WOOF-ArWen

WOOF-ArWen's hydrology work uses NOAA-OWP's NextGen framework and t-route, with
numerical processing in CUDA and Rust. Python handles configuration and
launches. Native ngen, t-route and runoff modules serve as development
referees; they are never installed as the product's runtime.

**Development status, 2026-10-18:** strict native comparisons pass for an
initialized 48-hour, three-catchment Noah/PET/CFE run and a separate one-hour
Lower Colorado routing run with seven level pools and twelve RFC reservoirs.
Their strict shared-pool RTX 4090 costs are 0.246 and 2.898 ms per step
respectively, at the separately pinned revisions below. Routing defaults to
fast mode, measured at 1.530 ms per step, whose output differs from native.
The routing run uses captured lateral inflows; these are separate component
compositions. The complete runoff, channel, reservoir and terrain forecast,
its speed target and physical event verification remain incomplete. See the [measured development scopes
and stage timings](../nextgen-development-evidence.md).

The permanent public demo still refuses acquisition pending verified data
rights and complete prepared resources. Initialized standalone Noah/PET/CFE
restart matches uninterrupted execution in the [original strict
test](../nextgen-output.md#initialized-standalone-continuation) and the newer
[PET v2 fast test](../nextgen-output.md#pet-v2-fast-continuation), each at its
separately pinned driver version. Fast restart is exact against the same
model run; its native comparison still differs in 88 of 864 words. Physical
parent-atmosphere coupling and complete forecast continuation remain pending.
The registered startup and gauge-analysis doors below do not establish a
completed hydrological forecast.

## Lower Colorado example

The demo selects t-route's HYFeatures example from
[`test/LowerColorado_TX_v4` at the pinned revision](https://github.com/NOAA-OWP/t-route/tree/12a8eae0cdfed437143c590659fa7077605a5e70/test/LowerColorado_TX_v4).
The similarly named `LowerColorado_TX` directory contains the NHD example.
The selected configuration is `test_AnA_V4_HYFeature_noDA.yaml`: Muskingum-Cunge
routing with waterbodies, with hybrid diffusive routing and assimilation
disabled. It is not a demonstration of RFC or persistence assimilation.

The permanent demo door is:

```sh
woof hydro demo lower-colorado-tx --prepare-only
```

It currently refuses before creating cache or run folders because the supplied
v201 GeoPackage's data licence is unverified. The public demo awaits a
hydrofabric built from USGS/EPA NHDPlus V2 and NOAA NWM parameter files, with
their source terms and preparation recipe verified, or a NextGen fabric under
verified US-government terms. Neither replacement is available in this row yet.

From a source checkout, `python -m woof.hydro_demo` provides the same
preparation options without loading the full command tree.

The held development manifest identifies 53 pinned files, totalling 32,451,588 bytes: the
HYFeatures GeoPackage, supplied hourly lateral-inflow CSV files, original
configuration, licence and NOAA terms. It acquires no native source code or
executable. Each file has an exact byte count and SHA-256 in
`woof/data/hydro/demos.v1.json`. Rust performs HTTP transfer, pin verification
and copying into an isolated input folder. Adding another example is a table
row; the preparer contains no example-specific adapter. Each row requires
explicit licence admission before acquisition. The [demo table and driver
contract](../nextgen-demo.md) lists the fields, generated settings, receipts
and targeted source-checkout build for the Rust asset bridge. A compatible
bridge does not clear the held dataset admission.

The example requests one hour beginning 2023-04-02 00:00 UTC. The upstream
configuration describes a full day with 300-second routing intervals;
preparing that configuration does not advance any routing state. The supplied
lateral inflows exercise routing inputs, not the full set of NextGen runoff
modules. The example's `LowerColorado_NGEN_v201.gpkg` is separate from the
hydrofabric v2.2 event subset used for broader validation.

For an admitted row, preparation creates a unique child of `out/hydro-demo/`, containing:

- `inputs/`: independent copies of the pinned source files;
- `hydro.toml`: the selected input paths and runtime metadata;
- `demo.json`: source pin, per-file digests, cache reuse and measured setup time.

The immutable input cache defaults to `~/.woof/hydro/demos`. Cached bytes
are checked before reuse. Unexpected bytes are preserved under an
`.unverified-…` name and downloaded again. Each run receives independent
copies, so changing one run's inputs cannot alter the cache or another run.

Choose different locations explicitly:

```sh
python -m woof.hydro_demo lower-colorado-tx --prepare-only --outdir hydro-runs --data-dir hydro-data
```

`--seconds` selects a positive duration in seconds; `--device` selects the
CUDA device ordinal (default 0). Fast numerics is the default;
`--strict-numerics` explicitly requests strict proof mode. These simulation
options do not make `--prepare-only` run numerical work.
For simulation, the duration must be a whole number of 300-second intervals;
invalid durations are rejected before input acquisition.

Simulation binds the selected numerics before preparing kernels. A process
uses one mode; switching between fast and strict requires a fresh process.
Strict selection alone does not establish agreement with a native referee.
For prepared `woof hydro run` or coupled `woof go --hydro` runs, select
proof mode with `numerics = "strict"` in the settings file's `[hydro]` table.
The `--strict-numerics` flag belongs to the demo command. See the
[configuration reference](../nextgen-configurations.md#toml-settings-files).

The public `woof hydro demo lower-colorado-tx` door also binds the standalone
runner. Omitting `--prepare-only` requests a simulation after preparation.
Data admission and the full realization's GPU stage composition are
still pending; that request cannot yet complete. The direct `python -m woof.hydro_demo`
entry point supports preparation only and refuses simulation before acquiring
inputs. A `prepared` receipt records input acquisition, not a completed model.

The preparation command accepts these arguments:

| Argument | Meaning |
|---|---|
| `lower-colorado-tx` | Example ID selected from the pinned table. |
| `--prepare-only` | Acquire verified inputs without advancing the model. |
| `--outdir DIR` | Parent for unique run folders; default `out/hydro-demo`. |
| `--data-dir DIR` | Immutable input cache; default `~/.woof/hydro/demos`. |
| `--seconds N` | Positive simulation duration; default 3600 for this row. |
| `--device N` | Nonnegative CUDA device ordinal; default 0. |
| `--strict-numerics` | Select strict numerics explicitly; default fast. |

## Settings and runtime composition

The preparer writes a `[hydro]` table with `enabled`, the selected `realization`,
`interval_seconds = 300`, `start_time`, `numerics`, and absolute `case_directory`,
`routing_config`, `hydrofabric` and `forcing_directory` paths. The runtime
realization must bind the actual runoff, routing, reservoir, forcing and
output components before stepping. A settings file that identifies inputs
alone is not a complete runnable realization.

Configuration rows live in `woof/data/hydro/configurations.v1.json`.
`resolve_configuration()` returns immutable metadata;
`configure()` expands runtime defaults and explicit overrides before device
setup. Baseline rows currently have empty stage lists. The initialized
`ngen-shipped-noah-pet-cfe-48h` row binds five prepared resources for the small
three-catchment Noah/PET/CFE example; it contains no reach or terrain routing.
The Lower Colorado row requires an explicit prepared-network binding and remains separate from
the held demo admission. See the [configuration API, row scopes and path
bindings](../nextgen-configurations.md). Metadata resolution alone does not
run a forecast or establish module agreement.

Settings files use nested TOML tables such as
`[hydro.settings_bindings]`; resource paths are literal strings and should
be absolute. The runtime epoch key is `time_origin`. The held preparer's
raw input paths and descriptive `start_time` do not produce the required
prepared network. See the [TOML file contract](../nextgen-configurations.md#toml-settings-files).

Stage rows select preparation factories and their launch order. Rows with
`phase = "prepare"` create checkpointed resources without a separate launch;
for example, a reservoir advanced by routing hooks must not also advance as
an independent stage. Launched rows use `phase = "before"` or `"after"` around
the land-surface exchange. Output history is allocated before stepping and
written through Rust at output/checkpoint boundaries.

The command reference includes these runtime doors:

```text
woof hydro run CONFIG --seconds SECONDS --outdir DIR [--device N] [--restart FILE]
woof go CONFIG --hydro [SETTINGS.toml]
```

These are templates for complete settings, not completed demo results.
Standalone duration must be an exact positive multiple of the hydro interval.
`hydro run` also accepts `--device N` and `--restart PATH`. For coupled runs,
`go --hydro` without a path uses the forecast configuration's `[hydro]` table;
a `[hydro]` table with `enabled = true` also enables coupling without the flag.
The hydro interval must match its parent land-surface cadence.
All accepted flags appear in [CLI-OPTIONS.md](CLI-OPTIONS.md).

The selected writer records the actual output directory in its receipt.
The newer reach and diagnostic writers preserve earlier output by selecting
fresh paths when needed. See the [output versions, timestamp rules and
measured routing and standalone restart scopes](../nextgen-output.md). Use
`restart.nc` for continuation; `history.nc` contains diagnostics only.
Standalone and routing continuation are separately measured compositions;
combined runoff/routing/terrain and parent-atmosphere restart remain pending.

Coupled mode uses WOOF-ArWen's Noah-MP runoff and ET, aggregated by catchment area
weights. Standalone mode advances selected NextGen runoff modules from
forcing. Both modes need resident state on the forecast member's card and
the prepared launch sequence. The parent atmosphere owns the fixed hydro
domain; moving child nests do not create another routing domain.

Coupling requires the parent to select WOOF-ArWen Noah-MP with
`sf_surface_physics = 4`. A complete coupled realization also needs prepared
catchment-to-parent overlap weights and the physical routing inputs; selecting
`--hydro` supplies neither. The initialized three-catchment standalone row is
not a coupled forecast configuration. The [coupling and checkpoint
contract](../nextgen-coupling.md) describes setup, cadence and continuation.

For already prepared terrain geometry,
`woof hydro prepare --cell-overlap CONFIG.json` exports catchment intersections
as cell-ordered sparse weights in Rust. Its
[request fields and geometric limits](../nextgen-coupling.md#preparing-terrain-overlap)
are separate from the runtime configuration. This setup command does not
provide physical producer inputs or a runnable public demo.

## Forcing inputs

Standalone forcing preparation consumes ordered catchment CSVs or already
decoded NetCDF grids. Source rows declare variable names, quantities, time
alignment and wind orientation; a source name alone does not acquire and
complete raw weather inputs. Prepared packets must use the realization's
catchment order and clock.

Winds in the canonical packet are earth-relative. A newer forcing owner
supports declared source-grid rotation before catchment averaging. Its
angle convention, units, preparation version and remaining input limitations
are described in the [forcing contract](../nextgen-forcing.md).
Coupled Noah-MP runoff and ET use the land-surface exchange directly, without
an additional external weather or PET stage.

## Preparing a hydrofabric

The registered startup command reads a user-supplied GeoPackage and builds
the network, attributes and optional catchment-to-parent-grid area weights:

```sh
woof hydro prepare FABRIC.json
```

`woof hydro fabric FABRIC.json` remains an alias for the same Rust
preparation path. Preparation runs before simulation and does not advance
hydrology or enable the held public demo.

The JSON configuration requires `geopackage`, `output` and `data_license`.
Use a dataset whose actual terms permit your intended use; the licence field
records provenance and does not grant permission. Optional `gauge` selects an
upstream subset. HUC subsetting also requires the actual membership column.
See the configuration and output schema in
[`tools/nextgen_hydrofabric/README.md`](../../tools/nextgen_hydrofabric/README.md).

The Rust tool prepares routing order and array payloads before simulation.
Its `manifest.json` records payload dtypes, shapes and hashes, data provenance,
coverage and per-stage setup timings. Use absolute input and output paths in
the request. An optional projected parent-grid definition supplies overlap
weights; those weights must match the actual forecast parent before coupling.
Gauge IDs and routing reach IDs require an authoritative crosswalk. An
ordinary flowpath roster is insufficient to reproduce t-route's collapsed
waterbody network or its incoming-flow reduction order.

This source-checkout door requires the Rust hydrofabric executable and its
system SQLite/PROJ dependencies. Installed-platform packaging remains pending.
It supplies no DEM, runoff calibration or completed realization automatically.

On a CONUS domain, `--hydro on` can build its fabric from EPA's NHDPlus V2.1
source archives, which are `.7z` files. Their decoder, py7zr, is LGPL-2.1+, so
it is an explicit extra rather than a base dependency:

```sh
pip install "recast-woof[hydro-conus]"
```

Without it the door prepares the worldwide fabric instead and says why.

## Gauges and observation skill

The registered gauge door delegates acquisition, decoding, alignment and
scores to Rust. A gauge table selects stations, source layouts and units;
another source uses another table row.

```sh
woof hydro verify-usgs fetch --table gauges.json --start 2023-01-05T00:00:00Z --end 2023-01-17T00:00:00Z --out observations
woof hydro verify-usgs observations --observations observations/observations.json --out observed-plots
woof hydro verify-usgs score --model model/hydrographs.json --reference native/hydrographs.json --observations observations/observations.json --out comparison
```

These are templates using your gauge table and actual model/reference
histories. The observation-only command draws measured streamflow; it does
not run hydrology. A source checkout needs the `rw_nextgen_verify` Rust
executable. See its [table and history formats](../../tools/nextgen_verify/README.md).

Scores include NSE, KGE, RMSE and peak timing. Model and native observation
scores use the same exact UTC intersection without interpolation. Peak
timing is model minus reference, so positive means late. Undefined metrics,
including constant-reference NSE, remain null with a reason. Agreement of
imported histories is distinct from captured-input module replay.

## Published streamflow retrospectives

The registered NextGen retrospective command is:

```sh
woof hydro retrospective --start START --end END --mapping mapping.json --candidate point-flow.json --cache archive-cache --workers 8 --output archive-comparison
```

This is a template using your actual mapping and point-flow history. It
launches Rust `rw_zarr reach-archive`; a source checkout needs that bridge
built with reach-archive support. `python -m woof hydro retrospective`
and `python -m woof.hydro_retrospective` use the same parser and launcher.
See the [source, mapping, interchange and output contract](../../tools/zarr_bridge/REACH_ARCHIVE.md).

`--start` and `--end` are UTC timestamps on the archive's hourly samples,
both inclusive, and must be supplied together. Event requests need an explicit
`--mapping`: archive reach IDs and NextGen flowpath IDs must not be assumed
equal. The default source is `nwm-v3-conus`; `--source-table` and `--source`
select another compatible source row. Omit both times to inspect actual
coverage. An out-of-coverage request fails rather than substituting another
model's output. `--store` selects a captured archive for offline replay;
`--cache` captures remote compressed objects, so omit it with a local store.
`--request` supplies a complete request JSON instead of assembling one from
the other source/event options.

The captured `nwm-v3-conus` archive spans `1979-02-01T01:00:00Z` through
`2023-02-01T00:00:00Z`, inclusive. For the requested 2023–2024 flood comparison,
the event must fit within January 2023; later floods are outside this archive.
The selected Russian River January 5–17, 2023 window fits. This date selection
does not supply missing NextGen event outputs or establish model skill.

`--candidate` accepts `arwen.point-flow.v1` JSON with discharge in `m3 s-1`,
explicit feature IDs and UTC timestamps. Rust producers should emit the
interchange directly; its `provenance` object records forcing, state and
parameter context.
Only exact identifiers and times join; missing flow remains missing.

Without a candidate the receipt says `reference_only`, with no model skill
claim. Candidate kind `model` compares the supplied model against the archive.
For independent gauge measurements, kind `observations` makes the archive
the prediction: bias is archive minus observation and NSE uses observed
variance. `comparison_complete` with `archive_against_observations` therefore
means an archive/gauge comparison, not a completed NextGen run. NWM output is
a model reference, not an observation. Declared provenance differences alone
do not establish why a bias occurred.

Rust writes `coverage.json`, `report.json`, `reference.json`, `paired.csv` and
one SVG analysis hydrograph per requested archive reach, with UTC and m3/s.
Inspect every plot before interpreting the result. Repeated runs into an
existing output directory preserve prior results in a new `run-*` child;
`report.json` records the actual output directory. These offline CPU timings
are separate from GPU routing-step speed. Actual NextGen event histories,
permitted hydrofabric and authoritative gauge joins remain necessary for a
complete model/retrospective/USGS comparison.

## Hydrographs, inundation and ensemble products

The ordinary `rw_wrfbatch` build includes the original NextGen products
renderer. Coupled forecast history invokes it automatically; supplied saved
history can also use the standalone door:

```sh
python -m woof.nextgen_products --request PRODUCTS.json
```

`woof.nextgen_products.run_nextgen_products` launches `rw_wrfbatch` after
checking its exact products ABI, including `component-series`.
`woof.rustwx` supplies renderer discovery and environment setup. Python reads
no field values. `woof hydro` registers no products subcommand; use the
standalone door above for a saved-history request. Rendering supplied data does
not establish native replay, physical coupled-run or observational-skill acceptance.
See the [request schema and output paths](../nextgen-products.md).

Repeating a products request preserves earlier results. An occupied output
directory, file or symlink selects a fresh sibling automatically; an absent or
empty directory keeps the requested layout. The command prints the actual
receipt path and reports any redirection. Receipts record both
`requested_output_directory` and `output_directory`; use the returned paths
when opening products or chaining commands.

Requests identify an ordered node roster, timestamped buffers and an output
directory/domain. Explicit index maps join source buffers to nodes; a missing
gauge/reach crosswalk must be resolved before comparison. Raw buffers and
NetCDF variables are supported. A NetCDF `record` selects one leading-axis
index, including variables shaped `[time, station]`. Native float64 flow is
preserved in CSV and float64 exports before display conversion. Only supplied
model, native and observed series appear in hydrographs.

For catchment or column diagnostics, use the `gpuwm.component-series.v1`
request schema through the same command. Supply the quantity, units, strictly
increasing sample positions and source time labels explicitly. Runoff in
metres per step, storage in metres and temperature in kelvin stay in their
declared units. The renderer infers neither a timezone nor a conversion to
reach discharge. Each chart produces a full-precision CSV and PNG under
`<out>/<domain>/component_series/<clock_label>/`. Finite model/referee word
comparisons in its receipt describe the supplied diagnostic series; they
do not establish whole-model agreement or observational skill.

Depth fields have different meanings:

| Request field | Meaning |
|---|---|
| `terrain_depth_m` | Dynamic terrain solver water depth in metres. |
| `channel_depth_m` | Channel water depth in metres above its local bed, used for a HAND estimate. |
| `member_terrain_depth_m` | Complete same-time ensemble of dynamic depth fields. |
| `member_channel_depth_m` | Complete same-time ensemble of above-bed depths for HAND estimates. |

HAND is height above the first downstream drainage cell. The estimated depth
is `max(channel_depth_m - HAND, 0)`; it is a drainage-connected estimate,
not the dynamic 2D solution. Absolute water-surface elevation cannot be used
as above-bed channel depth. Discharge alone supplies neither depth field.
Paths without a downstream drainage connection stay missing.

Depth exceedance uses strict `depth > flood_threshold_m`, default 0.1 m.
Supply the complete member roster: dry members remain in the denominator,
negative or missing member values mask a cell, and missing files fail the
request. Dynamic and HAND probabilities have separate product names. The
reach-flow ensemble reducer uses its own discharge threshold and `>=`
comparison; its output must not be relabelled as depth exceedance.

The ensemble owner's optional collector uses `device_budget_bytes` under
`[ensemble.hydro]` and writes `nextgen/reach-exceedance-*.nc`. Its v2 contract
publishes the same float32 threshold used for the comparison and preserves
native discharge precision. See the [ensemble configuration, variables and
continuation contract](../nextgen-ensemble.md), including integration status.
Actual weather-ensemble and mass-launcher acceptance remain pending.

Maps carry metres, stream order or probability fraction and follow
`<out>/<domain>/<product>/<valid-day>/`. Captions must identify engineering
forcing or observation-only input when applicable. Product processing runs
at output boundaries; its CPU timings are separate from GPU hydro-step speed.

## Licences, outputs and evidence

The inspected t-route software pin is
`12a8eae0cdfed437143c590659fa7077605a5e70`. Its
[LICENSE](https://github.com/NOAA-OWP/t-route/blob/12a8eae0cdfed437143c590659fa7077605a5e70/LICENSE)
states “Apache License”, “Version 2.0, January 2004”. The full notice is
retained at `licenses/LICENSE-t-route-Apache-2.0.txt`, with the
[NOAA terms](https://github.com/NOAA-OWP/t-route/blob/12a8eae0cdfed437143c590659fa7077605a5e70/TERMS.md)
at `licenses/NOTICE-t-route-NOAA-terms.md`. These software notices do not
establish a separate hydrofabric dataset licence. Dataset attribution and
licence verification belong in the case receipt; they remain pending for
this example. The product does not imply NOAA endorsement.

Module availability also depends on notices in the actual implementation.
TOPMODEL is excluded: at pin `89e4dc529dfdf7f23b7365530973e9f238fb740d`,
`src/topmodel.c` requires a Lancaster University royalty agreement for
commercial or paid consultancy use. Its repository-level Apache notice does
not clear that restriction. No TOPMODEL factory or default is supplied, and
another runoff module is not silently substituted for it. Any future inclusion
requires an independently permissive source or a coordinator decision.
SoilFreezeThaw and SoilMoistureProfiles continue under their independent
Apache-2.0 notices; their availability does not clear TOPMODEL.
Diffusive routing helpers with unresolved nested provenance are also held. The selected
no-assimilation example does not enable hybrid diffusive routing.

The separate Lynker hydrofabric v2.2 dataset is CC BY-NC-SA 4.0 according to
its source-bucket notice, superseding an earlier README-only ODbL description.
It is restricted to noncommercial development and referee work here: it is
excluded from shipped data, demo kits and default downloads. That notice does
not resolve the licence of this example's v201 GeoPackage.

Report discharge in m³/s, depth in metres, timing in milliseconds per actual
hydro step, and ensemble exceedance probability with its threshold and member
count. Hydro fields use WOOF-ArWen's Rust renderer through `woof.rustwx`.
Reference-model agreement and USGS observation skill are separate results.

The speed target is under 10 ms for the complete Lower Colorado hydro step
on an RTX 4090, with no per-step host data processing. The same-area 1 km
terrain calculation has its own target. Preparation time is not a routing
measurement. Completion requires native captured-input replay, a short run
with a viewed comparison plot, and the speed owner's measured stage receipt.
