# Native woof to CPU WRF direct export

woof can emit `wrfinput_d01` and `wrfbdy_d01` directly from its sealed
launch-ready preparation cache.  The runtime path invokes neither WPS nor
`real.exe`; stock CPU WRF consumes the resulting NetCDF files normally.

## Supported v1 slice

- one specified projected domain -- Lambert conformal, Mercator, or polar
  stereographic, with `MAP_PROJ`/`MAP_PROJ_CHAR` derived per projection
  (non-Lambert projections carry oracle-verified plus smoke-run-verified
  maturity, not matched-run verification);
- 49 mass levels / 50 full levels, hybrid coordinate option 2;
- WSM6 microphysics, YSU PBL, classic/old-MM5 surface layer (option 91), and Noah LSM;
- consecutive hourly f00..fNN external forcing with a five-cell specified
  boundary (the forecast horizon is not hard-coded);
- warm-soil initialization (the current HRRR proof has all soil layers above
  freezing).

Other domains and physics fail closed.  In particular, the v1 exporter does
not silently approximate frozen-soil liquid-water partitioning.

## Command

For the complete downloaded-HRRR plus WPS_GEOG path, use the public unified
command:

```bash
woof-wrf-init --source hrrr \
  --source-root /path/to/hrrr-files \
  --source-sha256s /path/to/SHA256SUMS \
  --source-sha256s-sha256 MANIFEST_SHA256 \
  --geog-root /path/to/WPS_GEOG \
  --domain-spec /path/to/hrrr-target-domain.json \
  --namelist-input /path/to/namelist.input \
  --valid-time 2026-07-18_00:00:00 \
  --output-root /path/to/output --pipeline-workers 8
```

It builds domain-specific static fields, runs the parallel native GRIB decode
and initialization, then performs the direct WRF export.  Its final products
are under `output/wrf-native-input`.  The decoder may be pinned with
`WOOF_HRRR_DECODER`, and the Python environment with `WOOF_PYTHON`.

The v1 downloaded series is exactly f00 through f12.  Files must be named
`hrrr.tHHz.wrfnatfFF.grib2` and `hrrr.tHHz.soilfFF.grib2`; `HH` comes from
the exact-hour valid time and `FF` is 00..12.  Every file must be covered by
the supplied SHA-256 manifest.  Missing, extra-contract, gapped, wrong-cycle,
or hash-mismatched input is refused.

To export an already-prepared native cache directly:

```bash
python -m woof.wrf_direct \
  --prepared-cache /path/to/prepared-cache \
  --static-cache /path/to/static-domain.npz \
  --geometry-receipt /path/to/static-domain.json \
  --output /path/to/wrf-native-input \
  --valid-time 2026-07-18_00:00:00
```

The output directory is published atomically and contains:

- `wrfinput_d01`;
- `wrfbdy_d01`;
- `manifest.json`, binding both products to the native prepared-state,
  static-cache, geometry, and frozen WRF declaration contract.

The exporter refuses to overwrite an existing directory unless `--overwrite`
is explicit.  Every floating-point field is reopened and checked for finite
values before publication.

## Stock-WRF proof

The 500 x 500 x 49 HRRR proof product was consumed by an unchanged WRF v4.6.1
`wrf.exe`.  Stock WRF accepted `wrfinput_d01`, initialized Noah, accepted the
12-record f00..f12 `wrfbdy_d01`, advanced two 5-second timesteps, and printed
`wrf: SUCCESS COMPLETE WRF`.  Dynamic arbitrary-Lambert proofs also pass for
Oklahoma and Ohio 192x160x49 at 3 km and Oklahoma 1000x1000x49 at 1 km; the
large case read all 12 boundary records and completed a 5-second proof step.

The direct ERA5 path is also stock-WRF gated.  A combined GRIB1 series for
1974-04-03 12Z through 1974-04-04 00Z was decoded with the hash-bound all-Rust
bridge and mapped to a 250x200x49, 12-km Lambert domain.  Native decode,
initialization, prepared-cache write, and direct export took 17.346 seconds.
Unchanged WRF v4.6.1 accepted both emitted files, completed the 5-second step
in 14.523 seconds, printed `wrf: SUCCESS COMPLETE WRF`, and emitted a finite
history file.

The direct GFS path is stock-WRF gated as well.  A real NOAA GFS
`pgrb2.0p25` 2026-07-20 00Z f000/f003 series supplied a complete 21-level
pressure atmosphere and all four exact Noah soil layers.  The public
`woof-wrf-init --source gfs` route produced a 250x200x49, 12-km Lambert
`wrfinput_d01` and `wrfbdy_d01` in 21.559 seconds internally and 22.57
seconds process wall time.  Unchanged WRF v4.6.1 accepted both files,
completed the 5-second main step in 14.680 seconds, printed
`wrf: SUCCESS COMPLETE WRF`, and emitted a finite history file.  The stock
acceptance evidence SHA-256 is
`b8fca75fc263b23699b17ccadaf4a122199d92f460c83fbae3a8def702c05cc3`.
The emitted `wrfinput_d01` and `wrfbdy_d01` SHA-256 values are
`064b216010c0ed43ad06ac197b82cd3dbfe884890251e0ebac30a4405fe387f3`
and `c74c5b9636f0e5a5cb54dbd9366d86a221ecbeb9a203fbbd6a3c0d39adc923f8`.

For masked GFS surface and soil fields, native woof uses land-aware nearest
donors and a globally proven nearest-water search for target lake cells.  That
policy is not numerically identical to WPS METGRID's masked four- and
sixteen-point interpolation.  Stock-WRF acceptance proves interoperability
and stable execution for this exact slice, not WPS numerical parity or
forecast skill.

The woof lightweight configuration intentionally uses no longwave radiation,
but stock WRF rejects `ra_lw_physics=0`.  The interoperability oracle therefore
uses stock RRTM longwave solely to permit a stock-WRF time advance and pins
`ghg_input=0` so WRF does not implicitly select the time-varying CAM gas-table
path.  `use_theta_m` is 0 on both namelists: the native model stores dry
theta and the export writes dry theta (`USE_THETA_M = 0`, `THM` equal to `T`,
`T_B*` dry-coupled), which stock WRF integrates directly.  It was bound as
`use_theta_m=0 -> 1` while the export declared `USE_THETA_M = 1`; that pair
carried the same dry-coupled boundary rows, which a moist-theta WRF read as
a boundary colder than the analysis.
These deltas are explicit in the hierarchy receipt; the emitted initial and
boundary files remain unchanged during the stock run.

Arbitrary-domain acceptance uses a fresh run directory and a pinned stock-WRF
binary rather than reusing an old run tree:

```bash
python tools/run_hrrr_stock_wrf_acceptance.py \
  --export /case/wrf-native-input \
  --domain-spec configs/hrrr_target_oklahoma_192x160_3km.json \
  --gpuwm-namelist /case/frozen-gpuwm-namelist.input \
  --valid-time 2026-07-18_00:00:00 --run-seconds 15 \
  --template-run-dir /wrf/clean-template-run \
  --wrf-exe /wrf/WRF-4.6.1/main/wrf.exe \
  --expected-wrf-sha256 PINNED_64_HEX_SHA256 \
  --run-dir /case/stock-wrf-acceptance/run \
  --evidence /case/stock-wrf-acceptance/evidence.json
```

The helper reopens every direct-export field with dynamic target dimensions
and bounded finite-value checks.  It copies only sealed IC/LBC files and the
template's resolved table symlinks; old namelists, executables, IC/LBC,
history, restart, `met_em`, and log artifacts are excluded.  It refuses a
non-symlink table artifact or broken link, regenerates `namelist.input`, pins
the exact `wrf.exe` SHA-256 before and after execution, and requires both
`Input data is acceptable to use` filename gates plus
`wrf: SUCCESS COMPLETE WRF`.  The output JSON binds those exact log lines,
the completed end time, process status/timing, input and output hashes, and a
finite stock-WRF history readback.

Known v2 limitations are recorded in every output manifest.  Vertical-wind,
terrain-shadow, and perturbation-pressure lateral-boundary arrays remain zero
at every interval.

Hydrometeor arrays (QCLOUD QRAIN QICE QSNOW QGRAUP and the paired number
moments the contract declares) carry what a forecast's ring is fed when the
prepared boundary carries them: the same function the forecast holds them with
(`woof.ingest.lateral_bc.hold_ring_tables`), at the cache's own
`lbc_hydrometeor_envelope_km` (default 25: the smaller of the two bracketing
frames, eroded along the edge, zero tendency; 0: the linear table).  The key
does not bind the prepared cache, so a forecast run against the same cache at
another value is fed differently from this file; its run record
(`lateral_boundary_hydrometeors`) states its own envelope.  The file's
`GPUWM_LBC_HYDROMETEORS` global attribute, the manifest's `lbc_hydrometeors`
block and its limitations say which arrays are carried and how.  Stock WRF
reads them only with `&bdy_control have_bcs_moist = .true.` (and
`have_bcs_scalar = .true.` for the numbers); at WRF's default `.false.` it
ignores them (WRF `run/README.namelist`).  A boundary that carries water vapour
only leaves them zero, and the attribute says they are not carried.

## Arbitrary-domain limits

`--domain-spec` is a strict `gpuwm-hrrr-target-domain-v1` JSON document.  The
current gate accepts only:

- one CONUS Lambert specified domain (`map_proj="lambert"`);
- `nz=49`, positive equal `dx_m`/`dy_m`, finite projection parameters, and a
  minimum horizontal axis of 13 cells;
- `spec_bdy_width = spec_zone + relax_zone`, with the verified direct-export
  slice using 5 = 1 + 4;
- a target whose mass, U, and V interpolation stencils fit completely inside
  the 1799x1059 native HRRR grid, with an explicit
  `surface_fallback_radius_cells` (0 through 64) whose soil donor search box
  stops at HRRR's own edge; and
- WSM6 + YSU + classic/old-MM5 option 91 + Noah state using the frozen 49-level hybrid
  coordinate.

The exact crop is derived and checked before decoding.  Only the soil donor
search is clipped, at HRRR's own edge, where no donor exists; no projection
substitution, missing-level synthesis, or out-of-coverage extrapolation is
allowed.  Other vertical grids, nests, physics suites,
projections, and HRRR-Alaska remain unsupported.

Version-1 domain documents that omit `surface_fallback_radius_cells` retain
the original radius of eight. New arbitrary-domain documents should set it
explicitly; `woof domain` writes 24 wherever the domain sits on HRRR's
grid. The document is also the per-run
override: edit the key in the `.d01-target.json` beside the experiment and
`woof go` prepares with it.

The radius sizes the decoded source window: that many HRRR cells are read
beyond the domain on every side, up to HRRR's own edge. A target land cell with no HRRR land corner
takes the nearest HRRR land cell (ties to the lowest row, then the lowest
column). A land cell HRRR's 3 km land mask has as sea, such as a small
island on a fine nest, can have no HRRR land within the radius; its search
continues through the decoded window and takes the nearest land cell when
that cell is nearer than anything outside the window could be (an edge of
the window that is HRRR's own edge has nothing beyond it). A cell whose
nearest land the window cannot vouch for is refused, with the radius that
would decide it and, when that radius is past the maximum of 64, a trim that
has been checked to leave every remaining land cell a donor.

The effective radius controls source-window sizing for every f00, boundary,
and nested-child land-surface mapping. Mapping receipts bind the maximum
donor distance, ceiling-distance histogram, zero unresolved or cross-surface
donors, and `distant_donors`: every land cell whose donor is more than 8 HRRR
cells (24 km) away, with its position, its HRRR donor cell and the distance.
The preparation also prints one warning per grid that has any.
