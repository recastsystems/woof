CUDA spectral ocean waves
=========================

The reachable door is `python -m woof.ocean.waves`. A CUDA device and CuPy are
required. There is no CPU solver. Build the small native boundary once with
`bash woof/ocean/waves/native/build.sh woof/ocean/waves/native/bin` (Rust 1.82+
and netCDF-C development libraries). Alternatively set `WOOF_WAVES_IO_DIR` to
a directory containing `rw_waves_io` and `libwaves_io.so`.

Run `python -m woof.ocean.waves run CONFIG.json`. See `examples/storm.json` for
configuration. Paths are relative to the configuration. Existing output folders
are retained; subsequent runs use numbered sibling folders and print the actual
path. Each run writes hourly CF NetCDF diagnostics and `receipt.json` with kernel
times, geometry, source hashes and memory-pool accounting. Use a process VRAM
monitor to include the CUDA context in a hardware receipt.

Inputs are model-independent CF NetCDF containers: one-dimensional regular
`latitude`, `longitude`; increasing `time` in `seconds since 1970-01-01 00:00:00`;
`U10`, `V10` [time,latitude,longitude] in `m s-1` or `m/s`, earth relative;
`LANDSEA` [latitude,longitude] or the full time grid, land fraction in [0,1].
Bathymetry has regular latitude/longitude and `altitude` [latitude,longitude] in
metres, positive upward. It must include the forcing halo. Rust reorders the
axes and bilinearly interpolates elevation, then uses land fraction >= 0.5 as
land and negative ocean elevation as depth (minimum 1 metre). Missing wet-point
bathymetry is an error. This format is a field contract, independent of the
atmospheric model that supplies it.

The existing `rw_zarr extract` bridge can produce these winds from ERA5. ERA5
requires explicit `input_licenses: ["CC-BY-4.0"]` and
`research_data_licenses: ["CC-BY-4.0"]` under this workspace's data policy. The
model does not download ERA5 or silently grant a data-licence exception. Public
NOAA bathymetry and observations are separate inputs.

Render with `python -m woof.ocean.waves render RUN/waves.nc PICTURES --valid
2022-09-28T12:00:00Z`, replacing the timestamp with a time in your run. This calls
`woof.rustwx.run_cf_renderer` and the real `rw_wrfbatch`; set
`WOOF_RW_WRFBATCH` to the renderer if it is not installed in the usual bridge
location. The package's metadata table maps Hs to the existing numeric colour
palette, labelled in metres. Output follows the renderer's domain/product/day
layout. No Python field rendering is used.

Compare against an uncompressed NDBC historical standard-meteorology text file:
`python -m woof.ocean.waves verify RUN/waves.nc OBS.txt STATION LAT LON PAIRS.csv`.
The Rust verifier prints Hs and peak-period bias/RMSE, circular direction error,
and paired maximum heights. It samples the nearest wet cell within 0.5 degree
and nearest output within 30 minutes. The CSV retains observation and model times
and locations. A calm initialization is included in these statistics; a spin-up
window must be stated when interpreting them. NDBC MWD is the direction at the
dominant period, whereas this solver reports the integrated mean direction, so
that direction comparison is indicative rather than an identical observable.

Check the prescribed winds with `python -m woof.ocean.waves verify-wind
FORCING.nc OBS.txt STATION LAT LON WIND-PAIRS.csv`. The native verifier compares
10 m forcing speed with raw buoy anemometer speed, using the same half-hour time
tolerance. It reports bias, RMSE and paired peak speeds. No height correction is
applied, so this comparison diagnoses forcing differences without attributing
all wave error to the wave source terms.

`WaveModel` accepts contiguous CUDA float32 geometry and wind arrays. `step`
advances one CFL/source-limited step and returns its actual seconds; callers must
advance their clock by that value. `output` updates seven diagnostic device
planes; `coupling_fields()` then exposes `arwen.waves.surface.v1`, including
wave-dependent effective roughness and friction velocity. Atmospheric feedback
is applied by the separate coupling lane using this same solver. Wind sea uses
component phase speed < 1.2 times the projected 10 metre wind; the remaining
variance is swell. Partition heights add in quadrature.

See SCIENCE.md for equations, discretization and numerical limits; NOTICE.md for
licence provenance. This door computes spectral waves with static water depth;
water-level surge and inundation require the separate hydrodynamics component.

The shared ocean ABI factory is `create_solver(grid, bathymetry, config=...)`.
It accepts the coupling lane's regular geographic CUDA grid and bathymetry
objects, exposes `set_forcing`, `set_water_level`, `step`, and persistent `outputs`
with `hs`, `tp`, and propagation **towards** degrees clockwise from true north.
Dry cells and cells with variance at or below the diagnostic cutoff of
`1e-15 m²` return zero for all three shared outputs. This common energy predicate
prevents positive Hs paired with undefined period/direction; invalid wet-state
NaNs remain visible. Setters and `step` only enqueue device work on the caller's
current stream. Local source substeps
and growth reductions run inside each cell's CUDA block, so this path has no
host scalar reads or per-step synchronization. The fixed prepared ocean-connected
coastline is retained as water level changes; the surge solver owns inland wetting.
Depth updates recompute dispersion/refraction quasi-statically: unsteady-depth
frequency shifting and wave-current interaction are not represented. Coupled
whole-run qualification belongs to the coupling lane; it uses the same source
and propagation equations as the standalone door.


For source-balance diagnosis, add `"capture_cells": [{"id": "buoy", "i": 4,
"j": 3}]` to a run configuration. Indices refer to the prepared south-to-north,
west-to-east forcing grid. Each output retains only those cells' spectra,
last-step local source increments, pre-limiter DIA rate, unlimited DIA source
at the captured state, cell limiter, moments,
wavenumbers and wind. CUDA gathers the values and Rust writes compact float32
files under `spectra/`; the run receipt describes the exact layout and source
step. Frame zero has no source increment. These captures support diagnosis;
last-step budgets are not time-integrated source totals, and output moments
follow DIA whereas the local increments precede it. No full-grid spectrum archive
is created by this option.

Measured qualification limits
-----------------------------

The standalone door has completed a real 120-hour, 63-by-99-cell storm run with
36 frequencies and 36 directions. Public buoy comparisons showed significant
height bias of approximately -0.81 to -0.91 m and peak-period bias of -2.73 to
-3.22 s, including calm initialization. Those results establish execution and
verification coverage; they do not establish operational forecast skill. Source
balance, calm start, external swell, spatial resolution and boundary effects
require assessment for a user's application. No empirical correction is applied.

The standalone adaptive driver synchronizes scalar growth limits each step; its
measured host CPU usage was close to one core during the shared-card run. The
asynchronous shared ocean ABI avoids those per-step host reads. Neither shared
card utilization nor cumulative process CPU establishes a GPU-idle interval.

Lakes (new in 3.0.1, opt-in with `--waves on`; checked on one case, a Lake
Superior gale on 5 November 2026 at 12 km for 9 hours against three NDBC
buoys; lake ice by a unit test only; not validated against a season of buoys): the forecast door passes the
model's water mask, surface height and ice to the Rust preparation. A connected
body of model water that the ocean wave source does not cover, that touches no
ocean wet cell and that covers at least 5,000 km² becomes wave water with depth
equal to the model's lake surface minus the ETOPO 2022 bed (cells shallower
than 1 m or under ice stay dry). It starts calm, a domain edge across it is open
and calm, and its waves grow from the coupled wind. Each lake found, made water
or left out with its reason, is a `{"lake": ...}` line in the preparation log
and a row of the prepared manifest's `lakes`.
Known limits: lakes need 5,000 km² or more; nests carry no waves (only the
outermost domain does); on wave grids of about 3 km, wave height in very shallow
shore cells is not yet capped by depth-induced breaking.
