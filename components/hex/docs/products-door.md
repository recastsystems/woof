# The product door: `woof hex products`

A forecast's history on any mesh in, the standard product set out, through
the Rust renderer. It is also the forecast door's last stage: a bare
`woof hex forecast ...` ends by drawing `<out>/products/`
(`--no-products` stops at the history frames and prints this command). By
default the product stage runs as its own CPU-only process, log
`<out>/products.log`, and the forecast returns as soon as its receipt is
written, so its cards are released while the maps draw: on four same-host
cards an in-command product stage held about 66 GB per card for 30 minutes
after the forecast ended (rented box, 2026-10-13). `--no-products-detached` runs
the stage inside the forecast command instead; before it starts, the door
empties its device memory pools and prints how much it released and how much
is still in use.

```
woof hex products RUN/fc
```

`RUN/fc` is the forecast door's `--out` folder. Its `forecast-receipt.json`
names the grid the run was bound to, the init it started from and its start
time, so nothing the forecast already knew is typed again, and the products
are filed in `RUN/fc/products`. Every one of those can still be given
(`--mesh`, `--init`, `--simulation-start`, `--out`), and must be for frames
that have no receipt beside them (`--history FILE... --mesh GRID.nc --out
DIR`). A path the receipt names that is no longer on disk is refused as the
receipt's, with the flag that replaces it.

## What it draws

| View (domain folder) | When | Products |
| --- | --- | --- |
| `global` | always | whole sphere at the mesh's own median spacing; on a variable-resolution mesh (a zoom globe), at its picture's pixels (0.2 degrees across 1800) when the median is finer |
| `refined-core` | the mesh has a refined region | the cells within 2x the finest spacing, at their spacing |
| `refined-context` | the mesh has a refined region | the core and its transition rings, 8x the core's half-width; one picture (never tiles under `--tiles auto`) |
| `north-america`, `europe` | the mesh is uniform | ground extents; dx from the cells inside |
| `--region NAME=region:CLAT,CLON,W_KM,H_KM` | asked for | the same rule |
| `points` | always (unless `--no-points`) | skew-T log-p soundings every 6 h and one meteogram per point |

Maps: 2 m temperature with 10 m wind, 10 m wind, MSLP, 500 hPa height and
absolute vorticity, 250 hPa height and wind (jet), precipitation since init,
precipitable water, simulated infrared brightness temperature, SBCAPE,
composite reflectivity and precipitation type everywhere; regional views add
2 m dewpoint, 500 hPa height and wind, 850 hPa temperature, MUCAPE and cloud
cover by level. Precipitation type is the model's own: where rain, snow and
graupel reach the lowest level (0.01 g/kg together), the frozen mass fraction
and that level's temperature choose rain, freezing rain, ice pellets or snow
(`rw-wrfbatch` `explicit_precipitation_type`).

Interval maps (`WINDOWED_PRODUCTS`): 1 h, 6 h and 24 h precipitation, the
run-maximum 10 m wind and the 0-24 h maximum and minimum 2 m temperature.
Each is the difference or the extreme of several stored frames, so each view
imports its whole series into ONE renderer store (`woof hex render
--series`) and the renderer closes each window from the frames it holds:
6 h precipitation at f012 is the stored total at f012 minus the one at f006,
filed at f012. The door requests a window only when the stored frames can
close it: 1 h precipitation, the run-maximum wind and the daily temperature
extremes need every hour stored, 6 h precipitation every 6 h, and the 0-24 h
windows a day of run. A window the frames cannot close is not requested; the
manifest's `windows_not_requested` names it with the cadence that rules it out
(a 3-hourly history has no 1 h precipitation, and a snapshot every 3 h is not a
daily maximum temperature). The global rows store hourly frames by default
(`global-120km`, `global-60km`, `global-30km`), so every default product draws;
`global-15km` stores every 3 h, where hourly frames would cost about 125 GB a
day, and `--history-every-minutes 60` adds the hourly windows. A frame before
the first window closes carries a skip in its view's manifest. The same store
also means one renderer start per view instead of two per frame.

The set is the renderer's own slugs (`products_door.py`
`GLOBAL_PRODUCTS` / `REGIONAL_PRODUCTS`). A member a history cannot draw is
not dropped silently: the manifest's `products_unavailable` names it with
the renderer catalog's own reason (for example `precipitation_type` on a
renderer built before the explicit-type planes:
`missing-fields: not stored: categorical_rain_surface, ...`).

Views draw side by side (`--jobs`, default 3): each is its own converter
and renderer pass over the frames, with its own scratch folder. Inside a
view, frames are converted and imported several at a time (`--frame-jobs`,
default `auto`; `rw_mpas_convert --jobs`, `rw_wrfbatch --import-jobs`). A
frame's conversion and its imported fields depend on that frame alone, so
the pictures are the same for every setting; what changes is memory, one
history and its fields per frame in flight. `auto` plans on half of the
host's available memory, shared by the views drawing side by side, at eight
history files per frame in flight (measured: 7.9 on 30 km global frames,
6.8 on 48-3 km patch frames), two cores per frame, at most 64 (the
converter's own ceiling), and 1
where `/proc/meminfo` cannot be read. An engine built before these flags is
asked (its own `--help`) and runs one frame at a time; the door says so.
`rw_wrfbatch` takes no `--import-jobs`: it imports a series' frames on
lanes it sizes itself from the host's threads and memory.

While a forecast runs, its doors draw each frame the moment it is written,
several frames at once (as many as the host's memory and cores carry), and
each view's converted frame is kept for the run in
`<products>.frame-cache-scratch/` so the pass after the forecast converts no
frame twice: it draws the whole-run maps (windowed precipitation, the run's
maximum wind, the 0-24 h temperature extremes) from those frames, the
soundings, and anything a live draw missed, then removes the cache.
The point pictures draw beside the views, and what they need
(`rw_hex_points`, `--init`) is checked before any view starts.

Soundings carry a hodograph (surface to 10 km above ground, coloured by
layer, Bunkers right and left movers marked) and, beside the surface-parcel
numbers, mixed-layer and most-unstable CAPE, 0-1 km shear and 0-1 km and
0-3 km storm-relative helicity. Helicity is taken about the cyclonic Bunkers
motion: the right mover north of the equator, the left mover south of it,
and the chart says which.

Points: `--points 'Name@LAT,LON;...'`, or by default the cities from
`DEFAULT_POINTS` inside a refined core (within 0.9 of its half-width of the
core centre), else a six-city world set. A point off a limited-area mesh is
refused by name.

## Layout (the render folder law)

```
<out>/<case>/<domain>/<product>/<valid-day>/<grid>_<domain>_<product>_f<lead>.png
<out>/<case>/points/sounding-<point>/<valid-day>/sounding_<point>_<YYYYMMDD_HHMM>_f<lead>.png
<out>/<case>/points/meteogram-<point>/<init-day>/meteogram_<point>_<YYYYMMDD_HH>.png
<out>/<case>/products-manifest.json
<out>/<case>/index.html
```

`index.html` lists every delivered picture by view and product in lead
order, with relative links, so the case folder can be copied anywhere and
opened in a browser.

`<case>` defaults to `<mesh>-<YYYYMMDDHH>` read from the mesh file name and
the run start (`--case` names it). Each view keeps its render manifest
(engine digests, remap digest, per-frame precipitation budget) in its own
folder; `points/points.json` holds every number a sounding or meteogram
drew.

## How mesh fields reach the render grid

`rw_mpas_convert --remap product` (the door default, `crates/rw-mpas/src/remap.rs`):

* **State** (temperature, pressure, moisture, wind, heights, reflectivity,
  fluxes): linear barycentric interpolation inside the triangle of the three
  cell centres that meet at each MPAS vertex (`cellsOnVertex`), the mesh's
  own Delaunay dual. It reproduces a linear field exactly (unit test on the
  sphere to 1e-5), never overshoots its three cell values and has no
  cell-edge steps. On the 30 km global mesh every one of the 885,780 window
  points fell inside a triangle of its nearest cell (0 needed the second
  ring, 0 clamped).
* **Accumulated precipitation** (`RAINC`, `RAINNC`, `SNOWNC`, `GRAUPELNC`):
  first-order area-conservative. A pixel's value is the area-weighted mean of
  the cells overlapping its footprint on the sphere; overlaps are measured by
  quadrature on an n x n sub-grid of each pixel (n >= 8, and at least eight
  samples across the smallest cell under the pixel), weighting each sample by
  the exact area element of the footprint. MPAS cells are the Voronoi regions
  of their generators, so a sample belongs to its nearest cell centre. For a
  window that holds the whole mesh the converter prints and records the
  window integral of RAINC+RAINNC against the mesh's own
  `sum(value * areaCell)`; a regional window claims no budget. Measured
  2026-10-12: on the 30 km global mesh at its own spacing the relative
  difference stayed between -8.6e-5 and +1.1e-5 over eight 6-hourly frames
  (window area within 2.9e-6 of the mesh's); on a 48-3 km mesh drawn globally
  at 48 km, where one pixel holds about 250 cells, -1.6e-4 and -1.0e-4.
* **Categories and soil** (`LANDMASK`, `LU_INDEX`, `TSLB`, `SMOIS`): the
  nearest cell's value. A category has no average, and soil arrays carry
  placeholder values over water that a blend would smear into the coast.

The method per variable is metadata (`fieldmap.rs` `CONSERVATIVE_FIELDS`,
`NEAREST_FIELDS`, linear otherwise), written into every frame as each
variable's `MPAS_RESAMPLE` attribute. `--remap nearest` keeps the frozen
nearest-cell map (every value a model cell's own, extremes untouched);
`--compose` frames always use it, because a composite switches source point
by point and has no single mesh to interpolate across.

Soundings and meteograms (`rw_hex_points`) use the same linear interpolant
as the maps, so a sounding agrees with the map under it. Sea-level pressure
on a meteogram is the hypsometric reduction of the surface pressure through
a 6.5 K/km column from the 2 m virtual temperature, and is labelled so.
SBCAPE/CIN on a sounding are wx-math's `cape_cin_core` for the surface
parcel from the 2 m state.

## Refusals

* No history frame in `--forecast`: named, with what to pass instead.
* `rw_hex_points` absent: refused before any view is drawn, by the shared engine
  ladder (`engines.POINTS`: `--points-exe`, `$WOOF_HEX_RW_HEX_POINTS`,
  `$WOOF_RW_HEX_POINTS`, woof's bridge directories, `PATH`; a build beside
  the converter in use is taken first), with the build line and `--no-points`.
  `woof hex doctor` lists it with the other engines. As the forecast's
  automatic last stage the door does not refuse for this: it draws the maps,
  prints `PRODUCTS soundings and meteograms skipped: ...` with the rerun
  command, and the forecast's exit code is unchanged.
* Soundings without `--init`: zgrid lives only there.
* `--remap product` with `--compose`: no single mesh to interpolate across.
* A uniform mesh's `refined-core` view is not a failure: the converter's
  "no refined region" refusal switches the door to the default regions.

Exit codes: 0 every view and the points drew; 1 a view or the points failed
(the manifest's `failed_views` and `points.refusal` say which); 2 a named
refusal before anything was drawn.
