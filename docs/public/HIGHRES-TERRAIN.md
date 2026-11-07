# High-resolution terrain and land cover, worldwide

The baseline static geography is `topo_gmted2010_30s`, GMTED2010 at 30
arc-seconds (roughly 900 m), with MODIS land use at the same 30
arc-seconds. That is fine for a 3 km domain and much too coarse for a 100 m
one, where a whole ridge can fall inside a single source cell and a town
arrives as a few 1 km squares.

`[static.highres]` replaces them with real high-resolution sources:
terrain, land use and soil, worldwide.

## The default for domains at 1 km or finer

A configuration that declares no `[static.highres]` block still gets
high-resolution terrain on every domain at 1 km or finer: Copernicus DEM
GLO-30, terrain only. Coarser domains keep the 30-arc-second baseline, so
a 2.25 km parent over a 750 m child keeps its terrain and the child takes
GLO-30. Copernicus is the default source because 3DEP stages no tile over
the sea or outside the United States, and one source serves every sub-km
domain of the tree. Land use and soil stay on the baseline; the default
land-cover file alone is 2.28 GB.

Before it downloads anything the console says what it will fetch:

```
[static.highres] d02: grid spacing 750 m is at or finer than 1000 m, so its terrain comes from copernicus-dem-glo30; 2 one-degree tile(s), 0 already cached, 2 to download (about 80 MB; all-sea tiles are not published and download nothing); cache ~/.cache/woof/highres-cache.  This is the default for domains at 1000 m or finer; a declared [static.highres] block replaces it: enabled = false keeps the 30-arc-second baseline, and its cache_root moves the cache
```

Tiles are about 40 MB each and are cached per user
(`%LOCALAPPDATA%\woof\highres-cache` on Windows,
`$XDG_CACHE_HOME/woof/highres-cache` or `~/.cache/woof/highres-cache`
elsewhere), so the next case over the same ground downloads nothing.

The rule is one row of `HIGHRES_DEFAULT_BY_DX` in
`woof/static/highres_production.py`, keyed on grid spacing. A declared
block replaces it and is taken as written:

```toml
[static.highres]
enabled = false            # keep the 30-arc-second terrain everywhere
cache_root = "highres-cache"
```

```toml
[static.highres]
enabled = true
cache_root = "D:/gpuwm-cache/highres"
max_dx_m = 1000.0          # only domains at or finer than 1 km
```

The same block is consumed by ordinary runs and prepared GFS, ERA5,
caller-mapped, and native HRRR inputs. It applies before initialization to
the root, children, and any sealed moving-domain static corridors. Prepared
receipts bind the requested settings, date, grid placement, and resulting
static bytes. The fetch folder (`cache_root`) is not part of that binding,
so a sealed preparation can be joined, extended and run on another machine.
A reused preparation must retain that binding; changing an
active overlay requires rebuilding its static preparation. Relative cache
paths stay relative to the original case file when a prepared bundle writes
its own configuration. `enabled = false` retains the baseline, and
An all-water footprint is accepted without a terrain replacement. The former
`fallback-30s` option is retired; unavailable required land coverage stops the run.

## What you get, and where

| | Source | Resolution | Where it is published |
|---|---|---|---|
| Terrain (default abroad) | Copernicus DEM GLO-30 | ~30 m | 90 S to 84 N, all longitudes |
| Terrain (on request) | SRTM 1 arc-second v3 | ~30 m | 56 S to 60 N, all longitudes |
| Terrain (default in the US) | USGS 3DEP | ~10 m | conterminous United States |
| Land cover (default) | CGLC-MODIS-LCZ | 100 m, 2018 | 60 S to 78 N, all longitudes |
| Land cover (on request) | Annual NLCD | 30 m, year nearest the case | conterminous United States |
| Soil texture | SoilGrids v2 | 250 m | global |

None of them needs an account, a token or an API key. They are fetched by
plain anonymous HTTPS. This is deliberate: the program
already asks for one set of credentials (ERA5 through CDS) and that single
requirement is its largest source of user friction. A second one would be a
worse product.

## Install

```
pip install recast-woof
```

That is the whole install. What reads and reprojects the DEM tiles is the
Rust `static-fields` library, and it ships in the bridge bundle every
install line the project publishes stages -- `woof`, `recast-woof[all-cu12]`,
`recast-woof[all-cu13]`, `recast-woof[render]`, all of them. There is no extra to
remember and none to forget.

Check it before you run anything:

```
woof doctor
```

The line to look for is `static builder (default static-field engine)`.
It reports the staged library and its ABI when the path can run, and
names the exact command to build or stage it when it cannot.

There is a second line, `geography stack (rasterio + pyproj, the highres
fallback)`. Those two are the pure-Python parity reference the Rust
substrate was proven against, and they are what `WOOF_STATIC_PYTHON=1`
runs on. They are NOT needed to build terrain, which is why that line
says `info` rather than `missing` when they are absent. If you ever want
to bisect a difference against the reference by hand, `docs/install.md`
names the extra that adds them.

> **Through 2.3.2 this was not true.** Those two libraries lived in an
> optional `geog` extra that `[all]` excluded and no quickstart named, and
> the check for them ran *after* the tile download. Following this page on
> a documented install fetched 160.7 MiB of Copernicus tiles and then died
> on a bare `ModuleNotFoundError`. If you are on 2.3.2, `pip install
> --upgrade woof`.

## Build terrain: a worked example

One 40 x 40 km domain at 1 km over the Bernese Alps, terrain from
Copernicus DEM GLO-30. It costs about 80 MB of tiles (two 1-degree tiles)
and runs in well under a minute on a warm cache. The example asks for
terrain alone (`fields = "terrain"`); leave that line out and the same
domain also takes land use and soil (see *Land cover, worldwide* below).

You need the 30-arc-second baseline first -- high-resolution terrain
*replaces a field inside* a baseline static build, it does not stand
alone. Fetch it once:

```
woof fetch-geog
```

Write `alps.wps`, which is where the domain's geometry is declared:

```
&share
 max_dom = 1,
/

&geogrid
 parent_id         = 1,
 parent_grid_ratio = 1,
 i_parent_start    = 1,
 j_parent_start    = 1,
 e_we              = 41,
 e_sn              = 41,
 dx = 1000.0,
 dy = 1000.0,
 map_proj  = 'lambert',
 ref_lat   = 46.55,
 ref_lon   = 7.98,
 truelat1  = 30.0,
 truelat2  = 60.0,
 stand_lon = 7.98,
/
```

Write `alps.toml` beside it. Set `geog_root` to the tree
`woof fetch-geog` wrote:

```toml
[experiment]
name = "alps_terrain_demo"
start_time = 2024-06-01T00:00:00
run_seconds = 3600.0
restart_interval_s = 0.0

[projection]
map_proj = "lambert"
ref_lat = 46.55
ref_lon = 7.98
truelat1 = 30.0
truelat2 = 60.0
stand_lon = 7.98

[shared]
nz = 49
ztop = 20000.0
p_top = 5000.0

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 40
ny = 40
dx = 1000.0
time_step = 5
specified = true
nested = false
history_interval_s = 3600.0

[case_data]
# `woof static` builds geography only: it reads geog_root and the WPS
# namelist, and never opens the forcing GRIB or the Vtable.  Both are
# still declared -- the config describes a whole case -- but neither has
# to be on disk to build terrain.
forcing = ["not-read-by-gpuwm-static.grib"]
vtable = "not-read-by-gpuwm-static.Vtable"
wps_namelist = "alps.wps"
geog_root = "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG"
sfcp_to_sfcp = true
output_title = "alps terrain demo"

[static.highres]
enabled = true
cache_root = "highres-cache"
fields = "terrain"
```

Build it:

```
woof static alps.toml --output alps_static.npz
```

You should see the overlay report itself, name its source, and say what it
left alone:

```
[static.highres] d01: APPLIED (terrain only, copernicus-dem-glo30; cells replaced: 1600 of 1600; receipt .../static_highres_..._d01_auto_lc-auto.json)
[static.highres] d01: land use and soil remain the 30-arc-second baseline (fields = "terrain" was requested)
static alps_terrain_demo: alps_static.npz
```

`HGT_M` in that NPZ is the terrain. Over this footprint it runs from about
527 m in the Lauterbrunnen valleys to about 3676 m on the Jungfrau ridge --
3149 m of relief that the 900 m baseline cannot resolve.

```
python -c "import numpy; h=numpy.load('alps_static.npz')['HGT_M']; print(h.shape, h.min(), h.max())"
```

Every run writes a receipt under `cache_root/receipts/` naming the source,
the vertical datum, the tiles fetched and the cell count replaced.

## Land cover, worldwide

The default land cover is **CGLC-MODIS-LCZ** (Demuzere, He, Martilli and
Zonato 2023), the 100 m global land cover WRF and WPS ship from version
4.5: the Copernicus Global Land Service map of 2018 in WRF's MODIS legend,
with the urban areas drawn as Local Climate Zones from the global LCZ map.
It is published from 60 S to 78 N at every longitude.

It is one 2.28 GB GeoTIFF. The first preparation with a given
`cache_root` downloads it (resumable, and it says so on the console);
every later one reads it from there. The download is checked against the
published size and MD5 and a pinned SHA-256; a file that fails is removed
and fetched again next time, never used. Each domain reads only the
window it covers, one byte per pixel.

What the engine does with it:

- **Categories 1 to 21 are WRF's own** and pass through as they are.
- **The Local Climate Zones (categories 51 to 61) become WRF's urban
  category 13.** They are the built zones LCZ 1 to 10 and LCZ E (bare rock
  or paved), numbered 51 to 61 in WRF since 4.4.2 (31 to 41 before).
  WRF's Noah and Noah-MP treat them as urban when no urban canopy scheme
  runs, and the engine runs none, so the zones are folded into category
  13 before the area fractions are taken. The model
  therefore keeps its 21 land-use categories, and every table and reader
  keyed on them (LANDUSE, VEGPARM and SOILPARM, Noah, Noah-MP, RUC, the
  wrfout attributes) is unchanged. One difference from WRF: WRF picks a
  cell's dominant class over all 61 categories and maps it to urban
  afterwards, so a cell whose built area is split over several zones can
  come out natural there and urban here. Here the zones count together, as
  NLCD's four developed classes always have.
- **Water comes from the map itself.** It separates the sea (17) from
  lakes and rivers (21) at 100 m, so no 30-arc-second split is made. Past
  its coastal zone the open sea is unclassified (0); those cells, like
  everything poleward of 78 N or 60 S, take the 30-arc-second baseline
  (*Where a source stops*).
- **The map represents 2018.** A case in another year says so in the
  receipt, in years, as any modern map used for a past date does.

Annual NLCD stays available inside the United States with
`landcover_source = "annual-nlcd"`: 30 m and matched to the case year
(1985 to 2024). It has one open-water class, so its water is split
against the domain's own 30-arc-second water field, sea to category 17
and inland water to 21, with both counts in the receipt. A domain wholly
outside the United States asking for it is refused by name.

The receipt's `landcover` entry names the source and year, the pixel
count of every raw class in the domain's window, how many pixels were
folded into the urban category and how many were unclassified, and the
`coverage` entry says for every field how many cells took the source,
how many were blended at a coverage edge and how many kept the baseline.

## Configuration

```toml
[static.highres]
enabled    = true
cache_root = "D:/gpuwm-cache/highres"

# Optional. Defaults shown.
terrain_source   = "auto"   # auto | copernicus-dem-glo30 | srtm-gl1 | usgs-3dep-13as
landcover_source = "auto"   # auto | cglc-modis-lcz | annual-nlcd
fields           = "auto"   # auto | all | terrain
on_refuse        = "error"  # required coverage failures stop the run
max_dx_m         = 1000.0   # absent: every domain; else domains at or finer
```

`terrain_source = "auto"` uses 3DEP inside the conterminous United States
and Copernicus DEM GLO-30 everywhere else. Naming a source pins it; a
domain wholly outside that source's published coverage refuses with the
source id, the footprint and how far past the edge it lies, and a domain
that only partly leaves it is built, the cells beyond taking the baseline
terrain.

`landcover_source = "auto"` uses CGLC-MODIS-LCZ everywhere;
`"annual-nlcd"` pins the United States collection.

From a WRF namelist, `woof import-namelist` writes this block when
`geog_data_res` names `cglc_modis_lcz` (the token of WPS's
`GEOGRID.TBL.ARW_LCZ`): `fields = "all"`, `landcover_source =
"cglc-modis-lcz"`, and `cache_root` from `--static-cache-root`, else the
per-user cache the default terrain already uses. Named on only the finer
domains, it adds `max_dx_m`. The block also replaces terrain and soil,
which the WPS token does not, and the import report says so. With the
urban canopy on and `use_wudapt_lcz = 1` the Local Climate Zones stay
categories 51 to 61, so `num_land_cat = 61` imports there and nowhere
else. The other `geog_data_res` tokens the engine builds (`default`, `5m`,
`modis_lai`) are read from the `namelist.wps` that `[case_data]
wps_namelist` names; any other token is refused by name.

`fields = "auto"` selects `"all"` wherever the land-cover source reaches
part of the footprint (60 S to 78 N for the default) and `"terrain"` where
it reaches none of it. `fields = "terrain"` is also valid inside the United
States: that is how the two terrain sources are cross-validated against
each other on the same domain.

## Where a source stops

Every source is published over its own area: CGLC-MODIS-LCZ stops at 78 N
and 60 S and leaves the open sea past its coastal zone unclassified,
Annual NLCD covers the United States and a strip of near-shore water, 3DEP
stages no tile over open sea or wholly outside the country, and the global
DEMs publish no all-water tiles. A cell outside a source's coverage takes the
30-arc-second baseline for that field, exactly what the engine uses
without `[static.highres]`: the sea stays the baseline's sea (its own
land/water mask and land-use index), terrain keeps the baseline's height,
and soil keeps the baseline's texture.

The hand-over is not a cliff. Over the five cells in from a coverage edge
the high-resolution value is blended with the baseline the way WRF blends
a nest's terrain into its parent's: the k-th cell in carries k/6 of the
high-resolution value. Terrain and land-use fractions therefore step no
more at the edge than they do anywhere else.

The console says so once; this is a 1 km parent reaching about 110 km out
to sea, with the default sources:

```
[static.highres] d01: APPLIED (terrain usgs-3dep-13as, land use cglc-modis-lcz-2018, soil soilgrids-v2; cells replaced: 39072 of 51076; receipt .../static_highres_..._d01_auto_lc-auto.json)
[static.highres] d01: WARNING: part of this domain lies outside the high-resolution sources and takes the 30-arc-second baseline there: terrain (usgs-3dep-13as) 10035 of 51076 cells, lat 39.68..40.71 lon -74.07..-72.73; soil 0-30 cm (soilgrids-v2) 1792 of 51076 cells, ...
```

CGLC-MODIS-LCZ classifies the sea over this whole domain, so land use has
no cell outside it; with `landcover_source = "annual-nlcd"` the same
domain has 10031 land-use cells past the collection's offshore edge, and
they keep the baseline's ocean. The receipt's `coverage` entry gives, per
field, the source, the cells outside its coverage, the cells blended and
the latitude/longitude bounds of the cells outside, and its `cell_groups`
say what every cell took. Only a cell that neither the source nor the baseline
covers is refused, naming the field, the count and where the cells are.

## Choosing between Copernicus DEM and SRTM

Prefer Copernicus DEM unless you have a specific reason not to. It is newer
(TanDEM-X, 2010–2015), actively maintained, void-filled, and it reaches
84 N. SRTM stopped acquiring in 2000 and stops at **60 N and 56 S**, which
excludes Canada, Scandinavia, Alaska and most of Russia; ask for it there
and you get a refusal naming the cut-off.

They are not interchangeable in the vertical either. Copernicus DEM heights
are metres above the **EGM2008** geoid; SRTM uses **EGM96**; USGS 3DEP is
orthometric on **NAVD88**. The three differ regionally, so every run records
which source and which datum it used instead of treating the numbers as one
quantity.

Copernicus DEM and SRTM are **surface** models: they see forest canopy and
buildings. USGS 3DEP 1/3 arc-second is **bare earth**. That difference is
larger than the datum difference and it is measured below.

## What changing `terrain_source` does to your terrain

Measured, not asserted: one 50 x 50 km domain at 500 m over the Colorado
Front Range (39.55 N 105.55 W, 2.2 km of relief, spanning treeline) built
three times through the same code path. Full method, pre-registered
predictions and resolution limits are in the evidence gallery under
`2026-08-14-intl-highres-terrain`.

### Copernicus GLO-30 against USGS 3DEP

**Copernicus reads about 3.5 m higher, and that is the forest.**

| | median (Copernicus − 3DEP) |
|---|---|
| below 3000 m (forested) | **+3.34 m** |
| 3000–3500 m (dense subalpine forest) | **+5.43 m** |
| above 3500 m (alpine, bare rock and tundra) | **−0.07 m** |

The offset tracks vegetation and vanishes where vegetation does. Copernicus
is a surface model and 3DEP is bare earth; above treeline, where there is no
canopy for the two to disagree about, they agree to 7 cm. The remaining
datum term (NAVD88 vs EGM2008) is what is left up there: below the sources'
own accuracy in this footprint, though it will not be everywhere.

**Practical consequence:** switching a forested US domain from 3DEP to
Copernicus raises its terrain by a few metres. Switching an alpine or
desert domain barely moves it.

### The shape is the same

Once that single offset is removed:

| | |
|---|---|
| cells within 20 m | **9999 of 10 000** |
| cells past 10 m | 166 of 10 000 |
| median terrain slope, 3DEP vs Copernicus | 0.13703 vs 0.13671 m/m: **0.09 % apart** |
| ridge-vs-valley agreement (Laplacian sign) | **99.06 %** |
| residual spread after the offset | 4.2 m RMS |

That last figure is the two datasets' own vertical accuracy, not this code's
error: 3DEP at ~1–2 m RMSE and GLO-30 at 4 m LE90 combine to ~3.1 m, and
both degrade in steep terrain. It is the same size above treeline as below,
so it is not canopy variation either.

### Against the 900 m baseline, which is what you are replacing

| | 3DEP | Copernicus |
|---|---|---|
| RMS difference from the baseline | 27.5 m | 26.1 m |
| cells past 50 m | 749 | 659 |
| median slope, relative to baseline | **1.18×** | **1.18×** |
| ridge-vs-valley agreement with baseline | 86.5 % | 86.5 % |

The 900 m baseline is 18 % too flat, puts this domain's highest cell 73 m
too low and its lowest cell 56 m too high, and disagrees about where the
ridges are once every seven cells.

**The comparison that matters:** the two high-resolution sources agree with
*each other* 99.1 % of the time on ridge-and-valley placement, and with the
baseline only 86.5 %. On this one Colorado Front Range domain, Copernicus
agrees with 3DEP to the measures above after the canopy offset is removed.
This is evidence for that domain; other terrain and land-cover types have
not been compared here, so it does not establish the same agreement elsewhere.

You can reproduce any of this. The tool ships in the wheel, so run it as a
module -- `python tools/...` only works from a source checkout, which a
`pip install` does not give you:

```
python -m tools.terrain_source_crossvalidation --cache-root <cache> --out report.json
python -m tools.terrain_source_crossvalidation --self-test-only   # offline
```

It costs about 540 MB of cache, most of it one 3DEP tile. The self-test
arm needs no network and no cache.

It reads the 30-arc-second baseline from `$WPS_GEOG`, or from
`$GPUWM_CASE_DATA_ROOT/WPS_GEOG`, or from `--geog-root DIR`.

## What the gates protect

- **Coverage is per source.** Each dataset declares its own envelope and the
  footprint is checked against the source actually selected, so a refusal
  says which dataset does not reach where, not merely that something was
  out of bounds. Only a footprint wholly outside a requested source is
  refused; one partly outside is built on the baseline beyond the edge.
- **Unpublished tiles.** Neither global product publishes all-water tiles,
  and 3DEP stages none over open sea or outside the country. An absent
  tile is not read as sea level: it stays no data in the mosaic, and the
  cells under it keep the baseline terrain, whether the baseline calls
  them sea or land. The absent tile ids are listed in the receipt. A
  footprint where *every* tile is absent runs on the baseline terrain and
  says so.
- **Antimeridian.** A domain straddling 180 degrees is a domain, not an
  error. Its footprint is reported as a CONTINUED longitude range (for
  example 179.25 to 180.75 rather than -180 to 180), the one-degree tile
  enumerators read that frame directly and return the handful of tiles
  either side of the line, and the near-global sources are marked as
  published for every longitude so no coverage check reports a crossing as
  an overshoot. One step is still outstanding: the derived mosaic window is
  written in the cut -180..180 frame, so a continued footprint refuses
  naming `dateline-window-unbuilt` rather than producing a shifted mosaic.
  That refusal is made on the footprint, before the plan is resolved and
  before one tile is enumerated or fetched, in both `fields` modes; the two
  window writers call the same check as a backstop for a caller that
  reaches them directly. Nothing is downloaded for a domain that is then
  told its mosaic cannot be built.
- **A domain wrapped around its projection pole.** A footprint whose
  corners span 180 degrees of longitude or more has no continued range at
  all and occupies every longitude: the polar-stereographic domain that
  encloses the pole, and the conic domain whose corners fan more than half
  a turn about the cone apex, both look like the whole band from a lat/lon
  frame. It reaches the same unbuilt mosaic window, and it is refused
  under the same name, but it is told its own fact and its own way out:
  that footprint is not on 180 degrees, has no line with tiles either side
  of it, and cannot be moved off one, so the refusal names the span and
  asks for a smaller domain, or one further from the projection pole,
  until the corners span less than 180 degrees. Both cases keep the second
  way out, which is to leave `[static.highres]` disabled and run on the
  30-arc-second baseline.
- **Coast safety.** Each land-cover source declares its water rule.
  CGLC-MODIS-LCZ separates the sea (17) from inland water (21) itself and
  its classification stands. Annual NLCD has one open water class and
  cannot tell a lake from the sea, so its split is made against the
  domain's own 30-arc-second baseline water field: open water on a cell the
  baseline calls WRF ocean category 17 stays ocean, and everywhere else it
  becomes lake category 21. The receipt carries the counts and names the
  rule. Terrain-only runs no land-use rule at all
  (`LANDMASK`, `LU_INDEX` and `LANDUSEF` pass through from the baseline
  untouched), so the distinction is never made there and nothing is split.
- **Zero cells replaced is a refusal.** An enabled feature that changed
  nothing must never read afterwards as a feature that ran. A domain no
  terrain or land-cover source reaches at all is the one exception: there
  was nothing to change, and the warning and the receipt state it.

## Attribution

Runs that use these sources must carry their attribution. The exact strings
live on each source in `woof/static/highres_fetch.py` and are copied into
every receipt.

- **Copernicus DEM GLO-30**: produced using Copernicus WorldDEM-30 © DLR
  e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided
  under COPERNICUS by the European Union and ESA; all rights reserved.
- **SRTMGL1 v3**: NASA JPL 2013, doi:10.5067/MEaSUREs/SRTM/SRTMGL1.003;
  distributed by OpenTopography, doi:10.5069/G9445JDF.
- **USGS 3DEP**: public domain.
- **CGLC-MODIS-LCZ**: Demuzere M., He C., Martilli A. and Zonato A.
  (2023), doi:10.5281/zenodo.7670653, CC BY 4.0; built from the Copernicus
  Global Land Service LC100 v3 (Buchhorn et al. 2020) and the global Local
  Climate Zone map (Demuzere et al. 2022, Earth Syst. Sci. Data 14, 3835).
- **Annual NLCD**: public domain (MRLC).
- **SoilGrids v2**: CC-BY-4.0 (ISRIC).
