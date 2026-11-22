# Resident NextGen forcing and PET

Fresh Apache-2.0 implementation based on NOAA-OWP/ngen
`be11a180565a78ff069d855b0b5dbfc23845e7d8` and evapotranspiration
`6c410f2a8bc843043ad25e13239928031b8c5e64`. PET's complete upstream
LICENSE and TERMS travel in `woof/hydro/`; attribution is in NOTICE.
No WRF-Hydro source, fixture, implementation or ABI is used.

The Python forcing loader searches this crate's `target/release` and
`target/debug`, then the shared runtime, wheel and fetched bridge directories.
`WOOF_NEXTGEN_FORCING_BRIDGE` takes precedence; a missing explicit path is an
error. A source build needs no environment override. Libraries are loaded
without changing their executable permission. ABI1 also exports
`arwen_forcing_source_revision`, the release verifier's
`GPUWM_BRIDGE_SOURCE_REV=<commit>` stamp. An unstamped older build must be
rebuilt for release pinning; numerical and packet contracts are unchanged.

The source table's Apache-2.0 notice covers software definitions and
implementation. Weather, hydrofabric geometry and derived area weights have
separate dataset terms and attribution. A prepared packet's source hashes and
successful replay establish provenance and numerical behavior; they do not
clear those inputs for public defaults, acquisition or redistribution. Keep
the source-specific data permission evidence with the preparation receipt.
An unresolved dataset permission holds the affected case and its derived
weights, without disabling generic forcing preparation on permitted inputs.

Coupled forecasts use the existing WOOF-ArWen Noah-MP atmosphere forcing and LSM
API. They require no external forcing packet, extra surface-forcing stage or
separate PET calculation. The current NextGen coupling factory is
`woof.hydro.noahowp.factory.prepare_exchange`: it consumes `runsrf`, `runsub`
and genuine bottom drainage `soldrain` (float32 mm per LSM step), ET (mm/s)
and surface-head storage (mm). Its output runoff/recharge/ET is float32 m/s;
bottom drainage is already included in the subsurface output. The coupling
owner supplies these conversions and its layers/CSR contract. This lane does
not duplicate that exchange. The parent meteorological CSR helper below is
optional and is not a prerequisite for the existing coupled LSM.

`arwen-nextgen-forcing REQUEST.json` runs setup-only Rust/Rayon CSV or NetCDF
reading, declared units/accumulation conversion and ordered catchment area
remapping. The request names a profile JSON file and optional `profile_row`;
new source names/variable mappings are rows, without an adapter module.
Built-in rows are in `woof/data/hydro/forcing-sources.v1.json`, selected by
`hydro_forcing_profile` in the shared init-source registry.

Request fields: `source_profile`, `profile_row`, ordered `inputs`, `output`,
`time_origin`, `interval_seconds`, ordered `catchment_ids`, and optional
`area_weights`/`rain_baseline`. Gridded inputs contain one or more consecutive
time samples, already on the spatial grid used by the weights. The caller
orders files and declares cadence/origin. Rust checks every CF time coordinate
against the continuous clock before regridding. The profile's
`time_variable` defaults to `time`; `time_offset_seconds` defaults to zero and
can explicitly align interval-end timestamps (for example, -3600 for hourly
totals). Units must be seconds, minutes, hours or days since a UTC epoch, with
a Gregorian calendar. Reversed, repeated or missing hours fail preparation
before a resident packet is written. This prevents silently shifted weather.
The scalar or one-dimensional coordinate names the time dimension; its name
and axis position are not hard-coded. Multi-time fields must contain that
dimension, preventing silent broadcasting of a static grid through a series.
The Rust/Rayon loader normalizes first, middle or last time axes without
reordering spatial axes, and carries cumulative rain baselines across samples
and files. Decoding holds one file's weather fields at setup; no file access or
host data work is added to the resident routing step.
Area weights JSON contains
`catchment_ids`, `grid_shape`, `offsets` (u64), `indices` (u32), and `weights`
(f64 positive fractions summing to one per catchment). Geometry and any
projection/intersection work belong to the Rust hydrofabric tool.

The optional setup command [forcing-zarr-coordinates](ZARR-COORDINATES.md)
exports sealed local Zarr coordinate arrays and explicitly inferred bounds for
hydrofabric's weight preparation. The optional [local weather subset](ZARR-SUBSET.md) command extracts sealed
packed-int16 chunks into the existing NetCDF preparation door. Geographic CSR
remains separate; neither command adds a resident stage.

Canonical fields are `T2D` K, `Q2D` specific humidity kg/kg, `U2D`/`V2D`
earth-relative m/s, `PSFC` Pa, `LWDOWN`/`SWDOWN` W/m2, `RAINRATE` mm/s.
The profile's `humidity` defaults to `specific_humidity`. A row declaring
`"humidity":"mixing_ratio"` supplies dry-air mixing ratio through its Q2D
variable. Rust applies the field's scale/offset first, then computes float64
`q=r/(1+r)` separately for each source cell before CSR averaging. This also
applies to catchment CSVs. For example, a Q2D mapping with `"scale":0.001`
accepts g/kg mixing ratios. Negative mixing ratios and unknown humidity modes
fail before packet publication. The profile hash seals this interpretation;
existing rows retain their specific-humidity behavior and packet bytes.
Winds default to earth-relative. A row can instead declare a source grid basis:

```json
"wind": {
  "frame": "grid_relative",
  "angle": {"variable": "grid_east_angle", "scale": 0.017453292519943295}
}
```

The angle is positive counterclockwise from earth east to grid east. After
the declared scale/offset it must be in radians (the example accepts degrees).
Rust rotates each source cell into earth east/north **before** CSR averaging:
`east=u*cos(angle)-v*sin(angle)`, `north=u*sin(angle)+v*cos(angle)`.
The angle variable may be a static spatial grid or have the weather time
dimension in any supported position; spatial order must match the weights.
Catchment CSVs can declare an angle column with the same convention. Angles
are instantaneous (`temporal="rate"`); missing angles fail preparation.
The profile and input hashes seal the declared orientation. This does not
derive angles from a projection or connect a WOOF-ArWen parent rotation provider.
The default earth-relative rows do no rotation and retain their packet bytes.

No missing humidity, radiation, rotation or meteorological fields are guessed. Relative humidity and
dewpoint conversion remain unconnected. The decoded canonical
GFS/HRRR/ERA5/NWM rows are a binding for prepared NetCDF grids: raw GRIB fetching,
decode and meteorological completion are not connected by this change.
The generic GRIB2 stack export preserves the complete ordered statistical
specification list, encoded missing count, increment descriptors and interval
end without treating totals as rates. Levels with unequal descriptors refuse.
The earlier first-specification-only metadata hold is retired; actual provider
interval semantics, geographic weights and all required meteorological fields
still need qualification before statistical GRIB weather enters this forcing
door. See [the stack contract](../grib1_bridge/README.md).
The hourly AORC row divides precipitation amounts by 3600 seconds and aligns
the source's hour-ending coordinate with the preceding interval
(`time_offset_seconds=-3600`). For a run beginning at 00Z, the first source
sample is stamped 01Z. This follows NOAA's
[AORC dataset description](https://registry.opendata.aws/noaa-nws-aorc/).
Temperature is an instantaneous endpoint sample; this preparation holds the
co-stamped nonprecipitation weather over that preceding hour. It does not
claim those samples are observed hourly means or interpolate endpoints.
Already start-aligned derived inputs need their own declared table row;
do not apply the AORC end-time offset twice. Previously prepared AORC packets
must be prepared again; profile hashes distinguish the corrected row.

`check_aorc_clock.py --executable PATH/TO/arwen-nextgen-forcing --scratch
OWN_SCRATCH/aorc-clock --out OWN_SCRATCH/aorc-clock.json` tests the shipped
row with literal two-hour NetCDF-4 grids. Rust must produce exactly the same
float32 and float64 packets as independently declared start-aligned rates,
and reject a missing first hour or an already start-aligned input. This is
an analytic time/amount transport regression, not native or event evidence.

`woof.hydro.forcing_inputs.load_series` verifies typed packet hashes in Rust
and uploads once; each subsequent `advance(dt_seconds, *, stream=None)`
enqueues one kernel. Windows use duration-weighted interval means. Hour-aligned
hourly windows replay ngen's native CSV provider exactly; historical native
sub-hour window quirks are deliberately not claimed as an equivalent path.
Out-of-range windows set resident status/NaN without an out-of-bounds read.

`PreparedCatchmentForcing` binds same-card float32 parent fields plus hydrofabric
CSR. It writes stable catchment views. Two cumulative precipitation grids need
two explicit baseline grids; previous values are per CSR edge so catchments
sharing a cell cannot race. A decreasing/nonfinite accumulator fails visibly
and still updates its saved baseline; blindly retrying would use changed state.
Restore producer and cross-step arrays together, including the precipitation
baseline, before replaying an interval. Canonical humidity and
wind orientation must be supplied before binding.

Actual terrain runoff/recharge availability and the existing Noah exchange
binding are recorded in TERRAIN-SOURCES.md (an internal lane note, not in the public tree). Weather/PET
and captured reach laterals do not supply a physical terrain source budget.

`PreparedPET` binds the seven canonical meteorological fields and a same-card
float64 `(catchment,12)` parameter table in `PARAMETER_NAMES` order. PET is m/s,
all five native methods are implemented, and native mutable parameter updates
are retained. Fast is the shared default. The saturation-pressure exponential
uses the existing MIT Arm implementation in both modes; the cancellation-sensitive
flux sums preserve native rounded boundaries. Fast otherwise retains contraction
and CUDA libdevice log/pow. Explicit `GPUWM_NUMERICS=strict` keeps
unfused arithmetic and the existing MIT Arm exp/log/pow adaptation; a fresh
implementation evaluates the published Borges differential correction
(https://arxiv.org/abs/1904.09481, section4.2). The balanced-operand branch keeps
Algorithm4's expression; the separated-operand branch evaluates the squared
residual with Algorithm5's explicit fused operations. This repairs cancellation
at cat-52 hour674. Replacing both branches with Algorithm5 moved the mismatch
to hour533 and was rejected; the failed trial is retained in the receipt.
Current actual three-catchment 720-hour sequence replay is bitwise in both modes:
0/17280 forcing words and 0/2160 PET words differ, with identical initial/final
mutable parameters. This supersedes the earlier single 2ULP PET residue. PET
restart identity is now `noaa-owp-pet-cuda-v2` and names the wind-norm expression;
older checkpoints require fresh preparation under the new identity. The full
hydro acceptance and speed requirements remain open.

All components expose `.arrays` (stable same-card restart inventory), `.fields`
and `.restart_identity`. No per-step CPU numerical work, allocation, readback,
file access or synchronization occurs. No CUDA graphs are used.

## Driver binding and interval convention

The complete factory, field, checkpoint and timing inventory is in
[CONTRACT.md](CONTRACT.md), checked against driver implementation
`15b52b6157be07baa3e2db594c9415d4de91997c`.

The current module entry points are
`woof.hydro.forcing_inputs:prepare_series` (standalone factory) and
`woof.hydro.forcing:prepare_parent` (optional coupled meteorological factory).
Direct constructors are `woof.hydro.forcing.PreparedForcingSeries` and
`PreparedCatchmentForcing`. The external setup doors are
`forcing_inputs.prepare_external(request_path, *, library=None)` and
`load_series(manifest_path, *, library=None, device_id=0,
native_precision=False, start_seconds=0.0)`.

Default `.fields` values are contiguous float32 `[ncatchment]` views on the
prepared card, in canonical `FIELDS` order and hydrofabric catchment order.
The mapping and views retain their pointers. Explicit `native_precision=True`
selects float64 standalone fields for native PET/replay; it is not the default
float32 weather ABI. No private WRF-Hydro implementation, import or fixture is
used by these NextGen entry points.

Canonical fields/units are `T2D` K, `Q2D` specific kg/kg, `U2D`/`V2D` earth-relative
m/s, `PSFC` Pa, `LWDOWN`/`SWDOWN` W/m2 and `RAINRATE` mm/s.

Preparation allocates, uploads, validates and compiles; it does not populate
the published output fields. Enqueue the first `advance` before any consumer
uses them, and enqueue dependent consumers on that stream. The caller owns
the current device and any cross-stream producer dependency. Each
`advance(dt_seconds, *, stream=None)` enqueues one kernel; None selects the
current stream. It performs no device allocation, synchronization, readback
or host numerical transformation. Python assembles the launch arguments only.

For a standalone series, `time_origin` anchors frame zero in UTC, each frame
is an interval mean of length `interval_seconds`, and `start_seconds` is the
initial offset from that origin. A launch returns duration-weighted means
over `[clock_seconds, clock_seconds + dt_seconds)`, then advances the resident
clock to the window end. It does not interpolate instantaneous endpoint values.
Plan the complete horizon at setup. An invalid/out-of-range window sets
resident status and NaN fields; the invalid-window branch leaves the clock
unchanged. Invalid values inside a covered window set status/NaN but still
advance the clock. Recover the full upstream/component state before replay;
these error paths do not implement rollback. Clock/status are not polled by
`advance`.

Parent meteorology is the current producer sample when the kernel executes.
Direct rain is already mm/s. Accumulated rain is the new cumulative-mm total
minus its saved per-CSR-edge baseline, divided by the actual elapsed LSM
interval; the kernel saves that total as the next baseline. The first
baseline and its UTC epoch must describe the same producer state.
`rain_baseline="at_prepare"` explicitly samples that state at setup; it does
not infer a baseline at forecast start. Enqueue aggregation after the producer
has completed the interval being differenced. This optional sampling convention
must not be substituted for the existing Noah-MP-to-routing flux exchange.

Checkpoint all `.arrays` in place: standalone frames/output/clock/status, or
parent output/status/CSR/input aliases and `previous_rain_mm` when accumulated
rain is used. Preserve `.restart_identity` with its epoch, source/order/weights,
humidity and numerical mode. These forcing components keep cross-step numerical
state in the registered CUDA arrays; no extra Python clock is restored.
Hydrofabric owns weights/order; the driver owns launch ordering, checkpoint
boundaries and member/device scopes. The existing coupled Noah-MP path requires
no added meteorological or PET stage.

Realization factories, using the coupled driver's generic table loader:

The exact shared selectors, admitted factory options, setup source checks and
software/data provenance are in [CONFIGURATION.md](CONFIGURATION.md).

- `woof.hydro.forcing_inputs:prepare_series`: `manifest`, optional `library`,
  `native_precision` and `start_seconds`; standalone only. Optional `source`
  validates the canonical init-source ID, profile row and table hash against
  the prepared packet; optional `time_origin` validates the realization epoch.
- `woof.hydro.forcing:prepare_parent`: `hydrofabric_manifest`, `parent_fields`
  (canonical name -> dotted attribute or dictionary key), `time_origin`, optional
  one or two `rain_accumulators` and explicit `rain_baseline="at_prepare"`; coupled only.
  Paths default to `state`; explicit `state.`, `physics.` and `components.` roots
  resolve once at setup. Actual WOOF-ArWen surface paths include `physics.fields.t2`,
  `physics.fields.q2`, `physics.fields.psfc`, `physics.fields.glw` and
  `physics.fields.swdown`. Set `humidity="mixing_ratio"` for WOOF-ArWen `q2`:
  CUDA converts each cell with float64 `r/(1+r)` before CSR area weighting.
  The default remains canonical specific humidity. Wind inputs still require
  earth-relative m/s and explicit rotation provenance; names `u10`/`v10` alone
  do not establish their orientation. Cumulative rain is explicitly bound to
  `physics.rainc` and `physics.microphysics.rainnc`. When cumulus is disabled,
  `rainc` is None: name only `physics.microphysics.rainnc`; a resident zero grid
  supplies the absent second accumulator. No reset or baseline is inferred.
  Checkpoint identity v2 records humidity semantics and accumulator count.
  This optional meteorological aggregation is unnecessary for the existing
  Noah-MP LSM-to-routing exchange.
- `woof.hydro.pet:prepare`: prior `forcing_component`, `method`, complete named
  scalar `parameters` table; broadcast at setup on the GPU. A heterogeneous
  population instead supplies `catchment_parameters` rows, each with
  `catchment_id`, `method` and complete `parameters`, in the exact forcing order.
  CUDA selects the resident method per catchment in the same launch. The method
  vector is part of checkpoint arrays; row identity seals the configuration.

`capture_forcing.cpp` and `capture_pet_bmi.c` are observation-only native
referee tools. They are not runtime modules or wheel artifacts. Replay tools
read capture arrays through Rust, compare on CUDA and emit metadata receipts.
`bench_resident.py` measures actual LowerColorado geometry with uniformly
imposed captured weather; it is capacity evidence, not an event simulation.
Physical coupled/native/USGS plot and the speed owner's whole-step gate are
separate required acceptance evidence.

The timestamp regression runs the real Rust executable on tiny NetCDF files
made with `ncgen` (fixture generation only):

```bash
python3 tools/nextgen_forcing/check_grid_clock.py --executable PATH/TO/arwen-nextgen-forcing \
  --scratch OWN_SCRATCH/grid-clock --out OWN_SCRATCH/grid-clock-result.json
```

It checks valid ordering, reversed/repeated hours, a wrong realization epoch,
and an arbitrary profile row with a renamed time coordinate and explicit
interval-end alignment. This is an input-clock regression, not native model
or GPU speed evidence. Add `--netcdf-format netcdf4` and use a separate scratch
directory to exercise HDF5 coordinate dimension scales. Classic and NetCDF-4
files with identical values produce identical typed forcing packets. The shared
reader's coordinate-scale fix is adopted unchanged from verify-usgs commit
`92455072c9da55406f7a0660166c213b7a0a6ed9`: coordinate scales are retained as
variables; NetCDF's dimension-only placeholders remain hidden.
Time-coordinate conventions follow
[CF 1.12 section 4.4](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#time-coordinate).

`check_grid_series.py` takes the same executable/scratch/out arguments. It
compares multi-time NetCDF-4 files against ordered one-sample files, including
square dimensions with first/middle/last time axes and an arbitrary cumulative
rain row with an explicit baseline. It rejects repeated/gapped/reversed times,
overlapping files and missing weather time dimensions. Matching f32/f64 packet
hashes establish input transport equivalence only; the native module replay and
physical-run evidence remain separate.

`check_wind.py` takes the same executable/scratch/out arguments and optional
`--netcdf-format classic|netcdf4`. Literal opposite source vectors and opposite
quarter-turns yield exact earth-north catchment vectors; comparing both typed
packet hashes catches rotation after averaging, wrong handedness, wrong units,
and a misplaced time axis. It also checks static angles, CSV scale/offset,
missing/default-fill values, missing angle columns, invalid frames and shapes.
A focused Rust CSV quarter-turn check bounds the near-zero east residual at
1e-15 and requires exact north=1. These are analytic transport checks, not
native physics replay or physical event evidence. The shared NetCDF reader
now masks a numeric type's default fill even without an explicit `_FillValue`,
before scale/offset; an explicit `_FillValue` overrides that default.
NetCDF's byte convention retains the full signed-byte range without an
explicit fill declaration. This follows the
[NetCDF attribute convention](https://docs.unidata.ucar.edu/netcdf-c/current/attribute_conventions.html).

`native_chain_factory:factory` (put this tool directory on `PYTHONPATH`) binds
the real hourly cat-27 forcing/PET chain for the speed lane's
`tools/nextgen_speed.py`. Set `NEXTGEN_FORCING_MANIFEST`,
`NEXTGEN_PET_CAPTURE` (720x39 native f64 capture) and
`WOOF_NEXTGEN_FORCING_BRIDGE`. It returns the speed owner's
`woof.nextgen_execution.ResidentSequence`, with one launch per stage and no
history copies. Include warmup in the maximum 720 hourly advances. The speed
lane's shared numerics selection and execution modules must be integrated;
neither native executables nor channel/runoff components are part of this tool.

Exact sealed fixture paths, current ABI, fresh-state commands and 720-versus-719
timing semantics are in NATIVE-CHAIN-SPEED.md (an internal lane note, not in the public tree).
