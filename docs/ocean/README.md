Ocean input and device coupling
===============================

`woof ocean` prepares bathymetry, extracts real forcing, runs registered wave
and surge solvers, and renders their products with `rw_wrfbatch` through
`woof.rustwx`. `woof ocean prove-forcing` exercises the same device hand-off
without claiming a solver run. No surrogate solver is selected.

An atmosphere configuration enables coupling with `ocean_config = "ocean.toml"`.
The named TOML file has this section:

```toml
[ocean]
prepared = "prepared-ocean"
solvers = ["waves", "surge"]
interval_seconds = 60.0
```

Solver factories and product/source metadata live in `woof/ocean/rows.json`.
A future compatible solver needs a row and a factory implementing
`woof/ocean/interface.py`. Factories may be overridden by the `[ocean.factories]`
table. Per-solver settings are in `[ocean.options.waves]` or `.surge`.

The common-grid atmosphere hook checks coordinates, passes east/north 10 m wind,
absolute pressure, friction velocity and density-derived stresses every model
step on the current CUDA stream, and partitions the time step at the configured
coupling interval. Waves receive surge water level; surge receives wave moments.
The surface layer uses the wave-age Charnock roughness on connected open water.
The online history writer carries `OCEAN_HS`, `OCEAN_TP`,
`OCEAN_DIRECTION`, `OCEAN_ETA`, `OCEAN_INUNDATION`, `OCEAN_ELEVATION` and
`OCEAN_WET`; the ordinary Rust renderer can draw them with `var:<name>`.
MYJ consumes it inside each sea-point transfer iteration. MYNN consumes it
before its exchange coefficients, including the SPP path;
MM5 consumes the incoming ZNT and retains the wave value afterward. Geographic
preparation orders rows south to north and columns west to east. Projected target
coordinates retain the atmosphere's order and require dx/dy and rotation planes.
A different ocean grid currently requires preparing atmosphere forcing on that
grid before the replay door; online atmosphere attachment requires a shared grid.

Native preparation accepts JSON through `woof ocean prepare REQUEST.json`:

```json
{"operation":"prepare", "grid_file":"forcing.nc", "output":"prepared-ocean",
 "datum":"EGM2008", "sources":[{"path":"etopo.nc", "sha256":"<verified digest>",
 "format":"netcdf", "licence":"CC0-1.0", "datum":"EGM2008"}]}
```

A NetCDF source has `lat`, `lon`, `z` by default; variable names are configurable.
A GeoTIFF source uses `format: "geotiff"`; Rust transforms geographic sample
coordinates to its CRS. Source `elevation_units` is `m` (default), `ft`, or
`US survey foot`; conversion to contract metres happens in Rust. Later sources overlay earlier sources where valid,
allowing local coastal DEMs over the ETOPO base. Datums must agree; convert a
coastal DEM's vertical datum explicitly before combining it. Preparation checks
hashes, source licences, nodata and coverage; ocean connectivity excludes isolated
below-datum depressions. The preparation manifest retains source rows, datum,
checksums, wet/coastal counts, geometry, parallelism and wall time. ETOPO 2022
uses EGM2008; it is not a local tidal datum. Local tidal offsets and astronomical
boundary conditions belong in the selected solver's settings and source receipt.

Forcing extraction uses `operation: "extract"`, `input`, `output`, and `fields`
(mapping contract names to NetCDF variables). It normalizes rectilinear axis order
in Rust and preserves exact epoch times in JSON. A forcing request names
`streams` (native extraction directories), `licence`, and `source_label`. The
replay inputs are `u10`, `v10`, `psfc`, `temperature`, `dewpoint`, all in SI units.
The replay diagnoses neutral-log-layer friction velocity on the GPU from current
roughness; an online atmosphere supplies its own surface-layer friction velocity.
Replay is driven with:

```
woof ocean prove-forcing ocean.toml --forcing forcing.json --outdir proof --render \
  --research-data-licence CC-BY-4.0
woof ocean run ocean.toml --forcing forcing.json --outdir coupled --render \
  --research-data-licence CC-BY-4.0
```

The research flag explicitly names any data licence outside the default allowed
set; ERA5 is CC-BY-4.0 and is used only with this opt-in. Source acquisition is
independent of the coupling protocol. The public NOAA ETOPO 2022 metadata states
"SPDX License: Creative Commons Zero v1.0 Universal (CC0-1.0)". Coastal DEM terms
and datum must be checked per dataset; the generic coastal catalogue is not a
blanket grant of a licence.

Roughness uses Drennan et al. (2003), doi:10.1029/2000JC000715:
`z0/Hs = 3.35 (ustar/cp)^3.4`, with finite-depth dispersion for `cp`, the
smooth-flow viscous term, and Charnock coefficient bounds 0.005 to 0.1. The
relation was fitted to pure wind seas; swell dominance is outside that fit.
The 0.02 m roughness cap bounds the diagnosed drag at strong wind. These choices
are physical parameterization assumptions, not an observational skill claim.

Products use `<case>/<domain>/<product>/<valid-day>/`. Runs preserve prior output
by choosing a fresh sibling on repetition. Receipts identify forcing versus
coupled mode, every output time, extrema, stages, step counts and CUDA allocations.
Use a process-level GPU memory sampler alongside the run for peak VRAM, including
CUDA context and compiler overhead. NetCDF/GeoTIFF processing and map rendering
remain Rust; only device launches, byte transfers and orchestration use Python.


A checked coastal example is NCEI dataset `gov.noaa.ngdc.mgg.dem:5163`
(1/3 arc-second NAVD88). Its official metadata states: "Produced by the NOAA
National Centers for Environmental Information. Not subject to copyright
protection within the United States."
https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ngdc.mgg.dem:5163/html
Its datum differs from ETOPO's EGM2008, so it is a separate NAVD88 preparation
unless a documented vertical transformation has already been applied. The proof
uses ETOPO only; checking the coastal example's terms is not a claim to have
mosaicked or run that coastal dataset.

Shared output semantics: all moment planes remain finite, including calm/dry
cells; calm peak period and propagation direction are zero. The eta carrier
may equal bed elevation at dry points. The native eta product row masks those
points and retains initially connected ocean and positively inundated land.
This prevents high terrain from appearing as surge without altering solver state.

Whole coupled qualification (2026-10-20)
--------------------------------------

The replay door completed all 121 hourly ERA5 records for the September 2022
Gulf landfall, 2022-09-25 through 2022-09-30, on a 63 by 99 ETOPO-prepared grid.
Both real solver factories ran for 7,200 sixty-second steps and 7,321 forcing
handoffs, with wave-age roughness feeding the next friction-velocity/stress
diagnosis. Surge's nonfinite, negative-depth, CFL and invalid-forcing counters
remained zero at every output. The corrected near-calm wave outputs were used.

On a shared RTX 5090 the door took 97.565 seconds (90.822 seconds replay/output,
4.325 seconds rendering); invocation wall was 99.875 seconds and sampled process
peak VRAM was 810 MiB under a 2,048 MiB admission. No CUDA graphs were used.
The 26 normal maps and three additional peak-inundation maps were individually
inspected. Hourly maxima were Hs 7.445 m and inundation 2.060 m. These are model
outputs, not observed skill scores. This is an ERA5-driven coupling proof: winds
are prescribed, and the coarse grid does not resolve local coastal flood risk.
The online atmosphere roughness hooks have separate covering CUDA checks; an
evolving WOOF-ArWen forecast with ocean feedback is not claimed by this replay.

Process telemetry records about one CPU core issuing device work through the
90-second replay. Per-process GPU utilization was not sampled, so idle GPU time
cannot be inferred from that host scheduling observation. The delivery receipt
retains stage timings, telemetry, source pins, validity counters and limitations.
