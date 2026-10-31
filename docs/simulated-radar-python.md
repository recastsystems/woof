# Simulated radar from Python

`woof.simulated_radar.simulate` drives the same Rust beam simulator and
writers as `woof simulated-radar`. It does not step a forecast.

```python
from woof.simulated_radar import simulate

result = simulate("wrfout_d01_2026-01-01_00:00:00", site="KTLX",
                  formats=("level2", "cfradial1", "odim"), outdir="products")
print(result.volume_time, result.site, result.scan_strategy)
for format, paths in result.paths.items():
    print(format, paths)
```

A path list scans each internal history time. `result.volumes` contains the
paths, UTC `volume_time`, resolved `RadarSite`, strategy and native metadata
for each volume. `result.volume_time` is a convenience for a single volume;
for a series use `result.volumes`. Paths are absolute. Reusing an output
directory returns the products of this call, not unrelated older volumes.
History `Times` is authoritative; `valid_time` cannot relabel saved history.

## Sites, scans and fields

The bridge resolves WSR-88D IDs such as `KTLX` against its shipped site table.
Custom coordinates use antenna height above mean sea level:

```python
from woof.simulated_radar import RadarSite

site = RadarSite("SIM1", lat=35.0, lon=-97.0, height_m=350.0,
                 band="S", beamwidth_deg=0.95)
```

IDs have four letters or digits because Level II stores four bytes. Custom
IDs beginning `T` except `TJUA` retain the existing Level II reader refusal.
The bridge currently exposes an **S-band** bulk polar operator. Asking for
another band raises `radar_band_unsupported`; it cannot substitute S-band
scattering for a C-band TDWR or a different world radar. A geographic location
alone does not establish the instrument band or its scattering tables.

`scan_strategy="vcp212"` is the default and uses the native VCP 212 physical
cuts, including repeated elevations. `"low_tilts"` uses 0.5, 0.9 and 1.3
degrees. A custom scan uses `scan_strategy="custom", elevations_deg=(0.5, 1.5)`.
`timing="history"` scans snapshots. `timing="scan"` uses neighboring history
times; inspect each volume's `metadata["timing_used"]` for any unbracketed
snapshot behavior inherited from the door.

Formats are `level2` (default), `cfradial1`, `cfradial2` and `odim`.
`fields=None` keeps the existing automatic selection: reflectivity, velocity
and available ZDR, RHOHV, PHIDP and KDP. Explicit names are `reflectivity`,
`velocity`, `zdr`, `rhohv`, `phidp`, `kdp` and `spectrum_width`. Request all
seven explicitly when spectrum width is needed; automatic CLI requests keep
their previous field set. The scientific writers name these DBZH, VRADH,
ZDR, RHOHV, PHIDP, KDP and WRADH. Level II names the corresponding moments
REF, VEL, ZDR, RHO, PHI and SW; KDP has no standard Message 31 block.
Velocity is positive away from the antenna. Unavailable explicitly requested
moments fail by name. Scientific provenance, including `reflectivity_source`
and `dual_pol_status`, comes from the native manifest. With dual-pol enabled,
the bridge uses its scheme-aware bulk S-band scattering, as the CLI does.

## In-memory domains

```python
# domain supplies .state, .cfg (RunConfig or an object with .run), and .grid.
result = simulate(domain, site=site, valid_time="2026-01-01T00:00:00Z",
                  outdir="products")
# A bare DomainState uses the same route:
result = simulate(state, cfg=run_config, grid=grid, site=site,
                  valid_time="2026-01-01T00:00:00Z", outdir="products")
```

Call between steps on the state's owning thread, after pressure diagnostics
have been updated. The grid supplies georeferenced mass coordinates and wind
rotation; grid dimensions alone do not locate the storm. The existing
history extraction and microphysics reflectivity operator supply the fields.
An unconsumed microphysics-time reflectivity handoff is read without consuming
it. The forecast is not advanced. Native field extraction may use CUDA and
the same diagnostic scratch as history output.

An object carrying a `native_columns` path uses
`input_contracts()["native_columns"]` and the shared Rust canonical-atmosphere
adapter. Its timestamp is authoritative. That contract has rectilinear
geographic axes, top-to-surface levels, earth-relative mass winds and moist-air
mass fractions. It cannot represent a WOOF projected staggered C-grid
verbatim. Such a state uses one temporary WRF history frame with radar
coordinates, winds, thermodynamics, hydrometeor moments and reflectivity;
the Rust history writer retains its stored precision and staggering. The
frame is removed on success and on error. `result.input_transport` identifies
the route. The manifest records the temporary source's hash for provenance;
its source pathname no longer exists after the call.

## Output naming, admission and cost

The native writer owns `outdir/radar/manifest.json` and volume paths below
`outdir/radar/`. It chooses generation names from input, configuration and
implementation identities. Use the returned paths rather than constructing
filenames. The same request can reuse complete products; changed inputs or
options produce the appropriate generation. PPIs and loops are produced by
the existing Rust path as well.

The forecast door's `require_admitted` binary and host-memory checks apply
before snapshot extraction. Full scene dimensions are estimated before beam
sampling. Missing or stale bridges keep their existing named error and fetch
command: `woof fetch-bridges --partial --artifact rw_simradar`.

`result.resources` is the existing native resource estimate. Before running,
use `woof.rustwx.estimate_simulated_radar(paths, outdir=..., config=...)`, with
`SimulatedRadarOptions.from_mapping(...).to_mapping()` for the config, or the
CLI's `--estimate`. It reports cuts, rays, gates, work, memory and an output
upper bound, not a portable wall-time promise. Work grows approximately with
cuts times rays times gates; wider azimuth spacing, wider gates and shorter
range reduce it. Each volume's `metadata["elapsed_seconds"]` records measured
native time. See [the radar door](simulated-radar.md) for the full scan and
scientific limitations.
