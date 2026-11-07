# Stochastic physics (SPPT, SKEB) on any MPAS mesh

hex carries its own stochastic physics, written from the published methods.
No code, registry or documentation of NOAA's `stochastic_physics` package
(LGPL) was read, downloaded or used, and none ships; MPAS namelist option
names (`do_sppt`, `config_sppt_*`) are read only as configuration.

| Piece | Method | Reference |
| --- | --- | --- |
| SPPT | The net physics tendency (theta, qv, edge momentum) is multiplied by `1+r`. `r` is a smooth random pattern, clipped so `1+r` stays in [0,2]. It is tapered to zero near the ground and in the stratosphere. Where the perturbed humidity tendency would supersaturate the air or make the humidity negative, the unperturbed tendencies are used at that point. | Buizza, Miller & Palmer (1999), QJRMS 125, 2887-2908; Palmer et al. (2009), ECMWF Tech. Memo. 598 |
| Multi-scale SPPT | Up to five independent patterns, each with its own amplitude, length, time scale and seed, are summed. | Palmer et al. (2009) |
| SKEB | A streamfunction pattern's rotational wind, scaled so it injects a fraction of the dissipated kinetic energy back into the flow. A thermal pattern does the same for temperature. | Shutts (2005), QJRMS 131, 3079-3102; Berner et al. (2009), JAS 66, 603-626 |
| Pattern | Real spherical-harmonic coefficients follow AR(1) in time, with a Gaussian variance spectrum. | Weaver & Courtier (2001), QJRMS 127, 1815-1846 (Gaussian correlation on the sphere) |
| Saturation | Saturation vapour pressure over water. | Bolton (1980), MWR 108, 1046-1053 |

## The pattern law

- **Space.** The correlation is `exp(-r**2/(2*length_m**2))`, with `r` the
  great-circle distance. `length_m` is therefore the distance at which the
  correlation falls to `exp(-1/2)` = 0.61. The variance spectrum is
  `exp(-l(l+1)*length_m**2/(2*R**2))`, normalised to unit point variance by
  the addition theorem.
- **Time.** Coefficients follow AR(1): the lag correlation is
  `exp(-dt/time_s)`, with a stationary first draw.
- **Truncation.** `degree: 0` (the default) picks the smallest degree whose
  omitted variance is below 1e-4: 55 at 500 km, 61 at 450 km, 182 at 150 km,
  capped at 512.
- **Synthesis.** The harmonics are summed on the GPU at every mesh cell and
  vertex (the `harmonic_sum` kernel, one thread per point). No
  points-by-modes basis is stored, so the same law holds unchanged on a
  global, variable-resolution (zoom globe) or limited-area mesh. The m
  diagonal and the longitude rotation are carried in double precision. The
  l recurrence runs in float on a power-of-two-scaled value, so a high-order
  term near a pole cannot underflow before it matters.
- **Seeds.** Philox4x32-10 (Random123, BSD; `licenses/LICENSE-Random123.txt`)
  draws the innovations from a per-member seed:
  `sha256(schema, member, seed)`. Members, reorderings and restarts therefore
  reproduce the same words with the same build.
- **Restart.** The checkpoint holds the coefficients, the clock and the draw
  counter (schema `gpuwm-hex.stochastic/v3`). A restart with different
  stochastic settings is refused.

## SPPT details

- **Bound.** The default is a hard clip `|r| <= clip` (clip 1). `logit: true`
  (or `config_sppt_logit`) selects a smooth bound of the same range,
  `clip*tanh(r/clip)`, whose slope at zero is one.
- **Vertical taper.** It rises linearly from 0 at the ground to 1 at
  `surface_m` (500 m; the lowest level is always 0). It falls linearly to 0
  between `top_start_m` and `top_end_m` (18 and 25 km by default, 15 and
  27 km for RRFS).
- **Moisture guard (`moist_guard`, on by default).** The guard forms the
  state after one step with the perturbed tendencies, using MPAS's equation
  of state, `p = p0*(zz*Rd*rho_zz*theta_m/p0)**(cp/cv)`. If the perturbed
  qv would be above saturation over water (Bolton 1980) or below zero, both
  the theta and qv tendencies at that point keep their unperturbed values.
  The receipt counts the reverted points on the first calls.
- **Edges.** The edge factor is the mean of the two adjacent cell factors.
  On a limited-area boundary edge it is the one existing cell's factor.
- **Not perturbed.** Density, other species, and the lagged microphysics
  heating carrier are left unchanged.

## SKEB details

- **Dissipation.** The rate is the positive kinetic-energy sink of the
  unperturbed physics momentum sources (boundary-layer drag, gravity-wave
  drag, convective momentum transport), reduced per level over the mesh.
  Numerical (dycore) dissipation is not added.
- **Momentum.** A vertex streamfunction gives a dual-edge curl. The taper is
  applied before the curl, so the forcing is non-divergent.
- **Normalisation.** The streamfunction and thermal patterns are the
  unit-variance pattern channels, used unclipped (Berner et al. 2009). The
  curl is divided by its expected one-component RMS for the pattern's
  spectrum, `sqrt(sum w**2 l(l+1)/(8 pi))/R`. This is a property of the
  spectrum, not a sample over the mesh: a limited area much smaller than the
  length scale is not amplified. The earlier sample norm reached 0 on lam7k,
  where the clipped streamfunction was flat, and the winds went NaN.
- **Temperature.** On a closed mesh the thermal pattern has its area mean
  removed, so it adds no net heat. A limited area keeps its slice of the
  global pattern.
- **Energy budget.** The fractions are 0.1 (kinetic) and 0.01 (thermal) of
  the dissipation, with the integrated AR(1) covariance
  `2*D/h*tanh(h/(2*time_s))`, `h = max(dt, interval_s)`.
- **Limited-area meshes.** An edge without both cells and both dual vertices
  carries no curl and contributes no dissipation; native leaves its edge
  density undefined.

## Limited-area meshes

The same law holds on a limited-area mesh, because the harmonics are summed
at the region's own points. Cells in the lateral-boundary zone
(`bdyMaskCell > 0`) take no forcing: the boundary series drives them.
Missing neighbours (outside the region, or the padded garbage cell) are
ignored. The limited-area MPAS suite route (`--lbc-dir`, including RRFS)
admits `--stochastic`, `--sppt` and `--skeb`.

## Controls

- **Forecast door.** `--sppt/--no-sppt` and `--skeb/--no-skeb` override the
  same keys of `--stochastic JSON`. For example:
  `--sppt --stochastic '{"member":3,"seed":711}'`.
- **Global ensemble door.** `--sppt/--skeb` work the same way, with
  `--stochastic-seed`. The member identity comes from the source roster.
- **Ensemble default.** SPPT and SKEB are on in every ensemble of more than
  one member, GEFS-sourced ones included. `--no-sppt` and `--no-skeb` turn
  them off. This follows the spread-skill run below; before it, they were on
  only below ten source members.
- **MPAS namelist (`&nam_stochy`).** `do_sppt`, `do_skeb`,
  `config_sppt_N`, `config_sppt_lscale_N`, `config_sppt_tau_N` and
  `config_iseed_spptN` set the scales (N = 1..5). `config_sppt_logit` sets
  the smooth bound and `config_sppt_sfclimit` the surface taper.
  `config_sppt_hgt_top1/2` set the stratospheric taper, `config_spptint` the
  update interval (0 = every step) and `config_stochini` whether a restart
  must carry the state. Any other active stochastic option is refused by
  name.
- **RRFS v2 (hrrrv5) ensemble settings.** NOAA's published RRFS v2
  namelist (`verification/source-namelists/rrfs-v2-hrrrv5-namelist.atmosphere`)
  sets one SPPT scale: amplitude 0.8, 450 km, 6 h, smooth bound, surface
  limit, taper 15-27 km, update every step. SKEB is off.
  `do_sppt = .true.` in that deck runs exactly those values.

## Cost

The synthesis is one kernel per pattern update. Its measured time is in the
evidence receipt beside the step time; with `config_spptint = 0` it runs
every step. The memory ledger (`device_admission.stochastic_array_ledger`)
holds coordinates, topology, recurrence tables, tapers, the moisture metric
and patterns. There is no basis term: at x1.40962 with degree 48, the stored
basis that this replaces was 1.18 GB per member.

## Measured (2026-10-29)

Evidence folder: `HEX-STOCHASTIC-PHYSICS-2026-10-29` (receipts and maps).

**Pattern law** (`tools/stochastic_pattern_stats.py`, RTX 4090; 450 km,
6 h; 200 independent draws on x1.10242 and 2 million cell pairs):

- **Space:** the correlation falls to `exp(-1/2)` at 450.5 km, and a
  Gaussian fit gives 451.0 km. The largest gap from the exact truncated
  Legendre series is 0.014.
- **Variance:** the point variance is 1.001 on average. A regional subset
  (the points a limited-area mesh holds) gives the same law; its own
  measurement, 460 km, comes from only 209 points.
- **Time:** over 2,160 steps of 600 s, the lag correlation reaches `1/e` at
  5.996 h. The lag-1 value is 0.97259 (expected 0.97260).
- **Spectrum:** projected back with area-weighted quadrature, each degree is
  within 5.7 % of the requested variance, inside the sampling error of 100
  draws.
- **On-the-fly synthesis:** it equals the stored-basis product to 2.0e-5,
  where field values reach 3.57.
- **Kernel time:** degree 61 at 5.87 million points (conus3km cells and
  vertices) takes 41-73 ms for one channel and 119-221 ms for three. Those
  are two measurements on a 4090 shared with live jobs. SPPT alone is
  synthesised at cells only, a third of those points.

**MPAS-A route, global ensemble door** (10 GEFS members, 2026-10-27 00Z,
x1.40962, convection_permitting, 6 h, scored by
`tools/stochastic_verify_day.py` against the GFS 06Z analysis, area weighted):

| | z500 spread | z500 mean RMSE | ratio | t2m spread | t2m mean RMSE | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| SPPT and SKEB off | 3.21 m | 4.065 m | 0.79 | 0.439 K | 1.698 K | 0.26 |
| SPPT and SKEB on | 3.50 m | 4.078 m | 0.86 | 0.511 K | 1.692 K | 0.30 |

The spread maps (Rust renderer, CONUS) show the same structures in both arms,
with more spread in the "on" arm, most over the western Atlantic and New
England. Pattern synthesis took 2.4 ms per update and the whole forcing
3.1 ms per step (36 steps of 600 s), under 1 % of the step. Forecast stage
medians were 18.8 s on and 19.7 s off.

**Restart.** A member with SPPT and SKEB on was run three ways:

- straight;
- with a checkpoint written at 3 h;
- restarted from that checkpoint.

At 6 h all three are identical: 0 of 95,686,466 history words and 0 of
5,243,123 diagnostics words differ.

**Both off is unchanged.** A member with SPPT and SKEB off is identical in
every word to the release before this change (rel/hex-0.4.10 c8704ed0):

- MPAS-A route: 0 of 95,686,466 history and 0 of 5,243,123 diagnostics
  words differ.
- RRFS hrrrv5 x1.2562, 6 h: 0 of 3,666,223 history and 0 of 568,751
  diagnostics words differ.

**RRFS hrrrv5 x1.2562** (2023-03-31 12Z, 6 h, `--sppt` with NOAA's RRFS v2
settings, 10 members from one initial state): every member completes. The
2 m temperature spread at 18Z is 0.5-1.5 K over land in the afternoon
boundary layer and small over the oceans.

| Against the GFS analysis at 18Z | z500 spread | z500 mean RMSE | t2m spread | t2m mean RMSE |
| --- | --- | --- | --- | --- |
| SPPT off (deterministic) | 0 | 11.42 m | 0 | 2.516 K |
| SPPT on (10 members) | 0.98 m | 11.41 m | 0.37 K | 2.513 K |

A `--restart-from` run on this route is not the straight run, with or
without stochastic physics: 1.79 M of 3.67 M history words differ at 18Z on
rel/hex-0.4.10 itself. The stochastic restart was therefore proved on the
MPAS-A route; the route's own restart defect is filed for its owner.

**RRFS hrrrv5 limited area** (lam7k, a 270 km cut of conus3km, 3 km,
2 h, members 1 and 2, `--sppt --skeb`): both complete. The 2,567 cells of
the lateral-boundary zone take no forcing. Against the run with both off at
14Z:

- 10 m v wind differs by 0.6-1.0 m/s RMS;
- 2 m temperature by 0.12-0.17 K RMS;
- MSLP by 5-11 Pa RMS.

Every word is finite. With SPPT alone, the 10 m v wind differs by 0.23 m/s
RMS. The moisture guard reverts 6-14 % of the level-cells, the saturated
cloudy columns of a convective day.

## Verification tools

`woof.hex.stochastic_verification.forecast_fields_gpu` extracts z500 from
GPU `(cell, level)` pressure (Pa) and geopotential height (metres) arrays,
ordered from bottom to top, and passes through GPU `(cell,)` 2 m temperature
in Kelvin. It interpolates linearly in log pressure without extrapolation,
including exact 500 hPa endpoints. Invalid or nonmonotone columns return
NaN. The reader must supply geopotential height at mass levels; interface
`zgrid` and geometric height need conversion before this call. It uses GPU
array operations and has no CPU field fallback. Analytic extraction and
paired scoring fixtures passed on the GPU in the qualification above;
real forecast extraction and scoring remain unmeasured.

Paired verification uses `woof.hex.stochastic_verification.score_pair_gpu`.
Pass two mappings from member identity to GPU `z500` (metres) and `t2m`
(Kelvin) cell arrays, the identically aligned GFS analysis arrays, and GPU
cell areas. The function aligns identities, computes sample variance with
`ddof=1`, area-weighted spread and ensemble-mean RMSE, and emits one small
daily JSON receipt. Both arms use the same finite cells. Area coverage and
forecast values missing on otherwise valid analysis cells are recorded.
Grading requires resolving those missing values or supplying a documented
shared terrain mask, so a failing forecast cannot improve its score by
silently losing cells. Decoding and remapping remain the Rust caller's job;
this API has no CPU weather-field path. GFS analysis acquisition and
remapping are not yet wired for this lane's run.

The daily file consumer is `tools/stochastic_verify_day.py`. It invokes
the engine's `rw_netcdf dump` once per input file with the requested
variables together, uploads its decoded binary transport through CuPy,
and calls extraction and paired scoring. Rust decodes variables and file
chunks concurrently. Actual worker counts and timings still need a run
with that reader and a granted GPU. It removes and inventories its own
temporary decoded payloads on success or failure. It leaves the caller's
input files in place. It reads no NetCDF through Python.

Its JSON request has schema `hex-stochastic-day-v1`, `lead_hours`,
`valid_time` with explicit UTC, and these entries (paths should be absolute):

```json
{
  "geometry": {
    "path": "/workspace/shared-prepared/init.nc",
    "height_variable": "zgrid",
    "height_kind": "mpas_constant_gravity_interfaces",
    "model_gravity_m_s2": 9.80616
  },
  "control": {
    "member-01": {"path": "/path/to/run/off/member-01/history.nc", "diagnostics": "/path/to/run/off/member-01/diag.nc"},
    "member-02": {"path": "/path/to/run/off/member-02/history.nc", "diagnostics": "/path/to/run/off/member-02/diag.nc"}
  },
  "stochastic": {
    "member-01": {"path": "/path/to/run/on/member-01/history.nc", "diagnostics": "/path/to/run/on/member-01/diag.nc"},
    "member-02": {"path": "/path/to/run/on/member-02/history.nc", "diagnostics": "/path/to/run/on/member-02/diag.nc"}
  },
  "analysis": {
    "source": "GFS",
    "path": "/path/to/run/aligned-gfs-analysis.nc",
    "analysis_id": "GFS analysis object and content hash",
    "remapping_provenance": "Rust/GPU remap receipt and source/mesh hashes",
    "terrain_mask_variable": "z500_support"
  }
}
```

This shows two members for a small check; the qualification request must
name all ten identical identities in both arms. Add the schema, lead and
valid time at the top level. Each history contains `pressure` in Pa,
`indexToCellID` and one embedded `xtime`; t2 in K may be in the same file
or the optional diagnostics file, which also needs the IDs and clock.
Dimensions are reordered on the GPU by name. Geometry supplies `areaCell`
in m2, cell IDs and the declared height variable. A supplied
`geopotential_mass` height uses `(nCells,nVertLevels)` in metres directly.
The explicit MPAS constant-gravity option averages interface `zgrid`
heights and computes geopotential height as `model_gravity*z/9.80665`.
The model gravity must come from the run configuration, not from a fit
to observations. The current preparation constant is
`cuda_physics_prep_v841.GRAVITY_F32`, 9.80616 m/s2.

With the default `native_netcdf` format, the analysis file contains `z500` in metres and `t2m` in K on the exact
same native cells, IDs and embedded valid time. It must already have been
remapped using the stated Rust/GPU provenance. An optional dimensionless
0/1 terrain mask defines shared z500 support before scoring, with 1
meaning included; it never masks t2m. Forecast missing values on that
support remain counted. IDs in every frame must match geometry in order.

Alternatively, replace `analysis` with a raw GFS analysis specification:

```json
{
  "source": "GFS",
  "format": "gfs_grib2",
  "url": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261017/00/atmos/gfs.t00z.pgrb2.0p25.f000"
}
```

The URL illustrates the existing GFS public-object convention.
Supply exactly one HTTPS `url` or local GRIB2 `path`. The URL path reads
the object's `.idx` and fetches just HGT at 500 mb, TMP at 2 m above ground
and PRES at the surface, in three concurrent bounded byte ranges. It
checks the analysis date, `anl` label, HTTP 206, exact Content-Range and
length. A separate 2026-10-17 00Z analysis smoke check fetched these three
fields successfully; each requested range returned HTTP 206.

The existing engine Rust `gpuwm_grib2_stack` ABI decodes these fields in
three concurrent calls through ctypes, which releases the GIL. There is
no new GRIB decoder. This generic exporter is serial within each field;
the wrapper overlaps the three independent fields. The receipt records
the native worker IDs and each call's specification and wall time. Actual
core occupancy must be measured on the granted box. `--grib-bridge` can
name its engine Rust shared library explicitly; otherwise the engine
resolver is used. Only instantaneous deterministic PDT 0, forecast step
0, the exact raw WMO selectors and matching reference/valid times pass.
Decoded payload sizes and Rust-recorded SHA-256 hashes are checked.

Geometry must additionally contain `latCell` and `lonCell` in radians.
GPU bilinear interpolation evaluates point values at these cells, with
periodic longitude and either axis scan direction. It accepts a regular
global lat/lon grid spanning both poles with no duplicate seam column;
alternating-row, column-major and projected grids are refused. Missing
positive-weight corners stay missing, and zero-weight missing neighbors
do not contaminate exact grid points. This remap is for point verification,
not a conservative flux transfer. GPU event time covers terrain support
and remapping, separately from Rust decode and upload.

GFS can extrapolate isobaric height below ground. A z500 source corner
is supported only when its surface pressure is finite and at least
50,000 Pa. Every positive-weight corner must pass, defining the same
analysis support for both arms. This does not mask t2m or forecast errors.
The consumer generates the analysis identity from the three content-bound
Rust manifests and records grid, field hashes, reader identity and remap
method. Every owned raw range and decoded payload is inventoried and
removed on exit; a caller-supplied GRIB file remains in place.

On the granted card, consume each day before plotting the ten receipts:

```sh
PYTHONPATH=src python tools/stochastic_verify_day.py day-01.request.json \
  --reader /path/to/rw_netcdf --work /path/to/run/decoded-day --out day-01.json
```

The decoder's provenance, exact commands or native ABI specifications, elapsed times and deleted
transport paths/sizes are included in the daily receipt. An error receipt
retains that audit and no scores. The real GFS acquisition and existing Rust
stack decoder passed a smoke check in 1.275 seconds, using three concurrent
field workers and an unmodified debug build of the engine library. The
receipt is `receipts/native-gfs-probe.json` in the lane delivery. Raw ranges
and decoded payloads were deleted after their hashes and manifests were
recorded. This transport check performed no weather-field arithmetic.
Separate decoded GPU fixtures now cover remapping and the daily consumer;
real GRIB-to-GPU analysis and forecast verification remain unmeasured.

Native ABI tests can run separately with an existing engine library:

```sh
HEX_STOCHASTIC_RUST_TESTS=1 \
WOOF_CPU_PREPROCESS_BRIDGE=/path/to/libgpuwm_preprocess_cpu.so \
PYTHONPATH=src:/path/to/engine python -m pytest tests/test_stochastic_verification.py -q
```

These tests use generated GRIB2 records to check selectors, clocks, all four
supported scan directions, bitmap missing values and repeated decoded words.
They do not establish stochastic-pattern repeatability on the GPU.

The existing Rust `rw_netcdf` reader also passed real transport tests using
independent generated CDF-1, CDF-2, CDF-5 and compressed NetCDF-4 fixtures.
The tests check scale/offset unpacking, both missing markers, named axes,
units, padded MPAS timestamps, CF times and repeated payload words. The
transport context owns cleanup on success, reader failure and consumer
failure; numeric upload remains in `decode_gpu` through CuPy. Run these
checks with an existing reader and the test fixture writer installed:

```sh
HEX_STOCHASTIC_NETCDF_TESTS=1 \
HEX_STOCHASTIC_NETCDF_READER=/path/to/rw_netcdf \
PYTHONPATH=src python -m pytest tests/test_stochastic_verification.py -q
```

The four-container checks are small ABI tests, not real forecast-history
or GPU-transfer validation. The receipts retain reader/source hashes,
commands and deletion inventories; the fixtures and build products are deleted.

Once the ten real daily receipts exist, run:

```sh
PYTHONPATH=src python tools/stochastic_spread_skill.py \
  --cycle 2026-10-16T00:00:00Z --out verification/paired day-*.json
```

This checks ten identical member identities through days 1-10, verifies UTC
valid times, GFS provenance and units, and writes z500 and t2m analysis charts
plus `spread-skill.json`. Its stated policy requires increased spread and no
larger mean RMSE on every day for both fields. A numerical pass leaves
`qualification_complete=false`: visual inspection, complete-route GPU word
repeatability, step-cost measurements and landing gates still need separate receipts.
