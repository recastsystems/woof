# Storm cells as objects: `woof cells`

WOOF-ArWen writes fields. A person deciding about a storm -- a forecaster, a
researcher, a verification script -- needs objects: *this* cell, how
old it is, how fast its updraft is, whether it is still growing, where it
will be in twenty minutes. `woof cells` is the door that turns a run's
history into those objects and the per-cell numbers a decision reads.

## What titan is, and what WOOF-ArWen adds

**titan** is a Rust implementation of the TITAN storm-cell
engine: three-dimensional identification of convective cells in a
reflectivity volume, persistent tracking of each cell through time
including splits and merges, a lineage graph, trend estimation, and
forecast footprints at fixed lead times. It reads checksummed Cartesian
volumes and writes an analysis bundle (`frames.jsonl`, `tracks.json`,
`lineage.json`, `objects.geojson`, `forecasts.geojson`, `summary.json`).
It is a separate program that WOOF-ArWen does not ship: `woof` does not
vendor it, depend on it or reimplement any part of it. The `analyze` door
is optional. When no titan binary is installed it is off, and says so in
one plain message; `export` and `catalog` work without it.

titan supplies, per cell per frame: object id, track id, whether this is
the track's first observation, area, volume, geometric centroid, maximum
and mean reflectivity, echo tops at 18/30/40/50 dBZ, base and top height,
VIL, estimated hail size, the tracker's motion vector, the trend of area,
volume, top and peak reflectivity, and the forecast footprint at each
lead. Those numbers pass through the catalog untouched, each with its
provenance beside it. Two motions are carried and named apart:
`trend_*` is titan's robust fit of the track's recent positions (the
motion its forecast footprints are advected with) and `motion_*` is the
tracker's Kalman velocity state at that scan; on a constant-velocity
test storm the trend is exact and the state is not, so read the trend
for how fast a cell is moving.

**WOOF-ArWen** adds what a radar never sees and a model always has, sampled
inside each cell's own footprint (the grid columns under titan's 3-D
voxels):

| column | unit | what it is |
|---|---|---|
| `peak_w_mps`, `peak_w_ft_min` | m/s, ft/min | maximum vertical velocity over the footprint columns and every w level (ft/min = m/s x 196.85), with the height and the column it sits in |
| `min_w_mps` | m/s | the strongest downdraft in the same columns |
| `cloud_top_m_msl`, `cloud_top_temperature_c` | m MSL, C | the highest mass level with cloud water + cloud ice above 1e-6 kg/kg, and the model temperature there |
| `cloud_base_m_msl`, `cloud_base_temperature_c` | m MSL, C | the lowest such level |
| `cloud_depth_m` | m | top minus base |
| `freezing_level_m_msl` | m MSL | footprint mean of the lowest 0 C crossing, linear between mass levels |
| `level_minus5c_m_msl` ... `level_minus20c_m_msl` | m MSL | the same for -5, -10, -15 and -20 C |
| `slwp_max_kg_m2`, `slwp_mean_kg_m2` | kg/m^2 | supercooled liquid water path: cloud water x air density x layer depth summed over levels colder than 0 C, max and mean over the footprint |
| `lifetime_so_far_s` | s | frame time minus the track's first observation (titan's `created_ms`) |

Every column name carries its unit; `catalog.json` embeds the full
column table with units and provenance, and `catalog.csv` has the same
columns in the same order.

## The doors

```
woof cells analyze WRFOUT... --out CASE [--profile severe] [--ladder 250:18000:250] [--titan PATH]
woof cells export  WRFOUT... --out DIR  [--ladder 250:18000:250] [--no-temperature]
woof cells catalog WRFOUT... --bundle DIR --out DIR
```

`analyze` is the whole chain: export the series onto the height ladder,
run `titan analyze` with the chosen profile, then write the catalog.
Everything lands in the render folder layout,
`<CASE>/<domain>/cells/<first-valid-day>/`:

```
d02-2km/cells/2011-04-27/
  input.tfs             the titan input stream (checksummed TFS1)
  export-receipt.json   ladder, interpolation rule, grid, per-frame maxima, digest
  titan/                titan's analysis bundle, untouched
  analyze-receipt.json  the stages and their wall times
  catalog.csv           one row per cell per frame
  catalog.json          the same rows with units, provenance, and footprints in lat/lon
  cells.geojson         cell footprints as WGS84 polygons, catalog row as properties
  overlays/             one renderer overlay file per frame (see below)
```

`WRFOUT...` is any wrfout series: files, a directory of them, or a
glob. One domain per invocation. An initial frame written before the
microphysics has produced `REFL_10CM` is skipped and named in the
receipt.

The titan binary resolves through `--titan PATH`, then `WOOF_TITAN`,
then the bridge directories (`libexec/bridges`, the packaged directory,
`~/.woof/bridges`), then the `PATH`. `--profile` picks titan's own
threshold set; `severe` (30/45 dBZ envelope/core, 3 km^3 minimum) is
the default, as it is titan's, and the measurement behind keeping it is
in the receipt of the first real run: on a 2 km run of a real outbreak
the `research` profile (25/40 dBZ, 0.5 km^3) added about two thousand
three-cell objects with a median peak updraft under 1 m/s and separated
the merged convective line no better. `--profile research` asks for
those smaller objects by name; `--titan-config FILE` overrides any
profile key (raise `low_dbz` to split a model MCS whose 30 dBZ envelope
is one connected object).

titan's profiles are tuned for radar scans a few minutes apart: the
trend that advects every forecast footprint is fitted over
`forecast_history_s` (1,800 s in every profile) and a track ends after
`max_gap_seconds` without an observation. Model history comes at
whatever interval the run wrote, and at an hour that window holds one
point, so every trend is zero and every footprint stands still. The
door measures the series' median interval and raises the two keys to
span three and two intervals respectively, writes them to `titan.cfg`
beside the bundle, prints the change, and records it in the receipt;
nothing is lowered, and a key in `--titan-config` is never overridden.

`export` alone needs no titan and is how a stream is handed to the
engine elsewhere. `catalog` alone re-joins an existing bundle to its
series (a second profile, a re-run over the same frames).

The MCP server exposes the same chain as `arwen_cells` (a job) and the
catalog as `arwen_cells_catalog` (a reader); see
[the MCP page](mcp-server.md).

## The height ladder and the interpolation rule

titan wants one set of level heights shared by every column; a model's
levels follow the terrain and differ column by column. The exporter
resamples reflectivity (and, by default, temperature) onto a fixed
ladder of cell-centre heights above sea level, `--ladder
BOTTOM:TOP:STEP` in metres, default `250:18000:250`: 72 levels from
250 m to 18 km every 250 m.

The rule is linear interpolation in the field's own units (dBZ for
reflectivity, degrees C for temperature) between the two model mass
levels that bracket each ladder height. A ladder level below the lowest
mass level but at or above the terrain takes the lowest level's value;
a level below the terrain, or above the highest mass level, is missing
(NaN), which titan treats as no data rather than as no echo. Linear in
dBZ is the convention of the radar gridders that titan's output is compared against,
and the model's field is already logarithmic. The ladder and the rule
are written into `export-receipt.json` and the ladder into the stream
itself, so a bundle never has to be guessed at.

Horizontally the volume is the model's own grid: cell `(x, y)` is mass
point `(west_east, south_north)`, the spacing is the file's `DX`/`DY`,
and the origin puts the domain centre at `(0, 0)` metres. titan's
products are in those metres; the catalog puts every centroid and
footprint vertex back on the map through the model's `XLAT`/`XLONG`,
and `cells.geojson` and the overlay files carry degrees.

## Drawing the cells on the reflectivity

`overlays/cells_<stamp>.json` is the renderer's own overlay format:
each cell's footprint as a closed line coloured by track, the forecast
footprint at the first lead thinner in the same colour, and a label at
the centroid with the track id, its lifetime so far and its peak updraft.
Rendering one frame with its overlay:

```
woof render WRFOUT --products refl --overlays CASE/d02-2km/cells/2011-04-27/overlays/cells_20110427T120000Z.json --out PLOTS
```

The renderer takes one overlay file per invocation and draws it on every
panel, so a series is one `woof render` per frame with that frame's
file. The overlays are lines and labels in degrees, not GeoJSON; the
GeoJSON beside them is for map tools.

## Resolution: what a model updraft means

A model resolves a feature only when it spans several grid cells. An
updraft core spans roughly four to six cells before the model's
numerics represent it rather than smear it, so the updraft a run can
show is bounded by its grid: at 3 km spacing the narrowest resolvable
core is 12-18 km across, which is wider than most real updrafts, and
the peak `W` a 3 km run reports is a grid-limited lower bound on the
storm's, not a measurement of it. At 1 km the bound is 4-6 km and at
500 m it is 2-3 km, which is where an individual updraft starts to be a
thing the model has rather than a thing it implies. The catalog states
`dx_m` in its receipt and the domain token (`d02-3km`) in its folder for
this reason: read every `peak_w_*` with the spacing beside it, and
compare runs at different spacings only knowing that the coarser one
could not have shown what the finer one did.

The same holds, less sharply, for cloud top and base: they are the
highest and lowest cloudy mass levels, so their precision is the model's
vertical spacing at that height (a few hundred metres aloft), and the
isotherm heights are linear between mass levels. The supercooled liquid
water path is the model's cloud water on those levels -- a microphysics
scheme's number, defensible as such and not as an observation.

## What the catalog is not

titan's forecast footprint is titan's own trend model over the tracker's
motion; the confidence it carries is the engine's, and the catalog
passes it through rather than grading it. The catalog does not decide
anything: it supplies the objects and the numbers, with units and
provenance, and the decision stays with the reader.

## Proof of record

Measured 2026-09-01 at the door, on this tree, with titan 0.1.0:

- A 2 km run of a real outbreak (`d02`, 282 x 216 x 49, 15-minute
  history, 47 frames with echo): 2,618 cells on 312 tracks in 300 s of
  door wall time (export 91 s, `titan analyze` 40 s, catalog 167 s).
  The series' strongest updraft is 42.57 m/s (8,380 ft/min) at 9,107 m
  MSL in a 27,196 km^2 cell at 65 dBZ, 9 h 15 min into the run.
- An hourly 3 km run of a second real outbreak (`d02`, 500 x 400 x 49,
  12 frames with echo): 1,630 cells on 230 tracks. Under titan's own
  1,800 s window every trend was zero; with the window sized to the
  cadence, every one of the 1,400 rows carrying an hour or more of
  history has a non-zero fitted motion, median 27.8 m/s.
- `tools/cells_peak_w_check.py` re-reads `W`, `PH` and `PHB` from the
  wrfout with netCDF4 (not the door's own reader), decodes titan's voxel
  indices from the layout in the export receipt, and takes the maximum
  over the unique columns under them: on all 2,618 + 1,630 cells the
  catalog's `peak_w_mps`, `peak_w_ft_min`, `peak_w_height_m_msl`,
  `min_w_mps`, the peak's column and the column count agree exactly.
  Swapping the decode to y-fastest makes every cell disagree, so the
  check sees what it measures.
- `tools/cells_gallery.py` draws each frame through `woof render
  --overlays`; the same frames rendered by `rw_wrfbatch` on Windows and
  on Linux are byte-identical PNGs.

The gallery, catalogs, check receipts and hashes are delivered beside
the evidence for that date.
