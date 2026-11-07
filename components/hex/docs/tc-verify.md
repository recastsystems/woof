# Tropical cyclone verification

`woof hex tc-verify` runs the standalone Rust `tc-verify` executable. Build with
`cargo install --locked --path tree/rust/tc-verify` or set `WOOF_TC_VERIFY` to a built
binary. The hex front door accepts the same arguments and executable.

```
woof hex tc-verify fetch --catalog tree/docs/tc-verify/seeds.json --cache /path/to/cache --out manifest.json
woof hex tc-verify score --manifest manifest.json --out verification
woof hex tc-verify export --input fixes.json --storm AL012024 --model ARWN --init 2024060100 --out forecast.dat
```

A score manifest is `{"cases":[{"storm":"AL012024","best":"best.dat.gz",
"best_format":"atcf","adecks":["forecast.dat"],"cycles":["2024060100"],
"models":["ARWN","OFCL"],"notes":"Forecast configuration and limitations"}]}`.
Paths are relative to the manifest. Empty cycles means all cycles. Explicit models
form the homogeneous cohort. Unavailable models give an empty intersection, never
an automatically smaller cohort. Every metric has its own sample intersection.
The JSON also retains individual scores and matched signed errors for audits.
Use `"metric_models":{"mslp_hpa":["ARWN","AVNO"],"rain":["ARWN"],
"stations":["ARWN"]}` to declare a different cohort for a particular metric.
Each numeric metric override must be a subset of `models`. Rain and station
cohorts can include models with no ATCF track. Every homogeneous row carries its
cohort; the index and pooled tables preserve those distinctions. This is useful
because OFCL usually has no pressure forecast and operational tracks do not
supply rainfall or water-level fields. Overrides are explicit, never inferred.

Truth formats: `atcf` (BEST only), `hurdat2`, `ibtracs-usa` (USA agency one-minute
wind; never mix with JMA ten-minute wind). Gzip is detected by magic bytes.
Repeated radius records merge into one fix; inconsistent duplicates fail.
Non-tropical/subtropical initial and valid observations are excluded by default.
`all_statuses: true` is an explicitly labelled diagnostic alternative.

Export input: an array or `{"fixes":[...]}` with `valid` UTC, `lat`, `lon`,
`vmax_kt`, `mslp_hpa`, optional `status`, optional `radii_nmi` mapping threshold to
NE/SE/SW/NW arrays. Missing intensity writes ATCF missing zero; missing radii write
-999. The compact atlas detection evidence `rows[].tracker_match` is accepted
with `latitude_deg`, `longitude_deg`, `mslp_pa`, `wind10_ms`. ATCF rounds position
to 0.1 degree and intensity to integer units; this quantization is explicit.

Outputs: static index and storm HTML, SVG track-coordinate and intensity charts,
per-storm JSON with pairs, per-lead metrics, RI contingency counts, hashes and
stage times, and summary JSON. No server or external JavaScript is required.
This is offline: no integration code, forecast state, GPU synchronization, or
per-step work is changed. Paired rain and gauge/buoy contracts are described below. MET-TC 12.2.2
was used as an independent numerical referee. Published-table agreement and
archive discrepancies are reported in the delivery, not assumed by the CLI.

Distance conventions are explicit. `track_km`/`track_nmi` are great-circle errors
on a 6371.2 km sphere. `met_track_nmi` uses MET-TC's 60 nmi/degree planar
approximation and `met_track_km` multiplies it by 1.852. Along/cross errors use
MET's backward six-hour best-track tangent, historical 111.1 km/degree constants,
and positive cross to the right of motion. With fewer than two matched positions
these components are absent. Stationary tracks retain the preceding heading,
then a following heading, then north. No best-track positions are interpolated
for the actual track/intensity pairs. Wind is one-minute maximum wind in knots.

Raw and interpolated aids remain distinct unless the manifest explicitly defines
an ordered family, for example `"aid_priority":{"HFAI":["HFAI","HFA2"]}`. The
first available complete record at each cycle/lead wins, and `aid_selection`
records the original technique. This accommodates NHC's 6/12-hour interpolated
aids without treating a raw run as an early forecast. The selected priority and
the data vintage are part of the comparison, not an invisible default.

`plots:false` omits charts for numerical audits. `plot_hours` defaults to 126;
short completed runs should set it to their duration. The plotted horizon does
not filter score pairs. All models share axes, and best track is black. The index
has cross-storm per-lead tables. JSON pools only identical model cohorts, with
individual forecast samples as weights. No significance or superiority claim is
computed from this small sample; overlapping RI windows are dependent.

Optional `rain` and `stations` manifest keys name JSON files, relative to the
manifest. Every input is hashed. These are contracts for already collocated
observations, not an implicit interpolation or vertical-datum conversion.

Rain input fields:

```json
{"nx":2,"ny":2,"dx_km":4,"cell_area_km2":16,
 "grid_id":"common-projection-and-cell-centres",
 "start_utc":"2024-06-01T00:00:00Z","end_utc":"2024-06-02T00:00:00Z",
 "observation_source":"Stage IV URL, product time, digest, remapping method",
 "observed":[100,0,0,null],"forecasts":{"ARWN":[80,10,0,null]},
 "thresholds_mm":[50,100,150],"neighborhoods_km":[25,50,100],
 "track_distance_km":[0,50,100,null],"max_track_distance_km":100}
```

Arrays are row-major storm-total or 24-hour precipitation in mm. The source must
use exactly the same projection, cell centres, accumulation bounds and area as
every forecast. Missing cells are null. The mask intersects all requested models
and observations, optionally within the supplied best-track distance swath.
Neighbourhoods are square, rounded to odd cell counts; actual widths are reported.
Windows clip at edges and normalize by common valid cells. All-dry FSS is undefined,
not perfect. FSS, ETS, frequency bias, mean amounts and integrated volume bias are
reported. DOI: 10.1175/2007MWR2123.1. Stage IV fetching/decoding and conservative
remapping remain the data-preparation lane's responsibility; no Python gridded
processing is introduced here.

Station input is an array of records:

```json
[{"station":"station-id","variable":"water_level","units":"m",
 "datum":"NAVD88","observation_source":"NOAA CO-OPS URL and retrieval digest",
 "times":["2024-06-01T00:00:00Z","2024-06-01T01:00:00Z"],
 "observed":[0.2,0.5],"forecasts":{"ARWN":[0.3,0.7]}}]
```

Allowed variables: `water_level` or `surge` in m, `significant_wave_height` in m,
`peak_wave_period` in s. Use `datum:"not-applicable"` for waves. Times must be
strictly increasing and paired exactly; missing samples are null. All models
share the observation mask. Report RMSE, MAE, signed bias, peak magnitude error
and peak timing error on that shared window. The earliest tied peak wins. A total
water level must never be compared with a surge residual without first removing
the same tidal reference. The scorer validates the declared contract; it cannot
infer an undocumented datum or grid mismatch. Tables appear on each storm page.

Data terms: NHC and JTWC government observations and government-model aids are
public domain, with NOAA attribution. IBTrACS USA columns retain JTWC/NHC winds;
cite dataset DOI 10.25921/82ty-9e16 and paper DOI 10.1175/2009BAMS2755.1. JMA has
attribution and modification-notice terms and ten-minute winds; direct JMA data
are not ingested. Pre-2025 ECMWF and UKMET aid terms do not establish unrestricted
redistribution: use their archived aids for internal aggregate comparisons only,
and do not distribute raw fixes or plots of those aids. The supplied seed pages
exclude them. No private DA code or data are part of this tool.

The separate `tree/docs/tc-verify/program-catalog.json` stages the 15-storm, 60-cycle
program design. It is a data catalogue, not evidence of completed engine runs.
The current-season b-deck can be provisional; missing status remains excluded
by the default tropical filter. The eight-cycle West Pacific aid gap and any
pre-genesis cycles are reported in the delivered staging receipt. Cached files
are immutable snapshots by URL; use a separate cache directory to refresh a
changing current-season source, and compare the recorded hashes.
