# RRFS-MPAS initial conditions and boundaries

`woof hex init --physics-suite hrrrv5` (or any RRFS-MPAS suite: `rrfs_mesoscale_reference`,
`rrfs_convection_permitting`, `rrfs_none`) makes the initial condition the way RRFS itself does
from GFS: NOAA GSL's `init_atmosphere` fork (MPAS-Model-rrfs at 1d6e370e), case 7 for the initial
state and case 9 for hourly boundaries. The fork's init is ported to Rust (`rw_mpas_init_v831`,
all cores); no Fortran runs.

## What it makes

| Piece | Recipe (the fork's `parm/hrrrv5/namelist.init_atmosphere`) |
|---|---|
| Soil | RUC's nine nodes (0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.0, 1.6, 3.0 m) remapped from the first guess's four layers by `init_soil_properties_ruc` |
| Vertical grid | v8.3.1 terrain-following grid on the RRFS L60 levels (59 layers, top 25,878.712 m, 30 surface smoothing passes) |
| Aerosols | TEMPO's `nwfa`/`nifa` from the first guess when it carries them, else the monthly `QNWFA_QNIFA_SIGMA_MONTHLY.dat` climatology |
| Hydrometeors | interpolated from the first guess when present |
| Static | MODIS 15 s land use, BNU soils (16 classes), `landusef`, `soilf`, `lai12m`, `nvegopt = 2`; deep-soil temperature from ERA5 (see below) |
| Drag statistics | `ugwp_oro_data.nc` from the GSL generator (`rw_mpas_gsl_oro`), the file RRFS's `bl_ugwp_gwdo` reads; hex's own method stays selectable (`--ugwp-oro hex-ugwp-oro`) |
| Boundaries | case 9 per hour with RRFS hydrometeors (`--lbc-met TIME=MET`, repeatable) |

The recipe is one row of `data/profiles/init-producers.v1.json`. A model with its own init is a new
row there, not a new code path. Suites with no producer row (the MPAS-A v8.4.2 suites and the
WOOF-route suites) keep the door's own case-7 init.

## Running it

```bash
woof hex init --physics-suite hrrrv5 \
  --met GFS:2023-07-08_12 --static mesh.static.nc --grid mesh.grid.nc \
  --out init.nc --start-time 2023-07-08_12:00:00 --nfglevels 34 --nfgsoillevels 4 \
  --geog WPS_GEOG
```

`--met` is the WPS intermediate for the start time (rw-wps writes it). If `--static` already
carries the fork's static fields it is used as is; otherwise `--grid` and `--geog` build the fork
static first (the geography needs `modis_landuse_20class_15s`, `bnu_soiltype_top` and
`lai_modis_30s`). With `topo_ugwp_30s` and `topo_ugwp_2.5m` in the same geography the GSL drag
statistics are written beside `init.nc`. The receipt `init.nc.provenance.json` records the
producer, its recipe, every stage's time and the output hash.

## Deep-soil temperature: ERA5, not NOAA's ISLSCP-I

When the init builds the fork static (any region or mesh whose static lacks the suite's fields), its
deep-soil temperature (`soiltemp`, the temperature RUC holds its lowest soil level toward) comes from
`soiltemp_era5_1deg`: ERA5's 1990-2019 annual-mean 2 m temperature reduced to sea level at 0.0065 K/m,
on the same 1-degree layout. `woof fetch-geog` builds it from ERA5 (Licence to Use Copernicus
Products: commercial use allowed, attribution required).

NOAA's recipe reads `soiltemp_1deg` (ISLSCP-I) instead. Its licence is unclear (CEDA states
"Permitted Use: Unstated"), so it is held out of the default geography. The two differ over land by:

| | ERA5 minus ISLSCP-I |
|---|---|
| all land, mean (area-weighted) | +1.9 K (+1.8 K) |
| 95 % of land cells | -0.6 to +4.8 K |
| CONUS / Europe / East Asia, mean | +1.6 / +2.1 / +2.4 K |
| largest | -6.7 K to +11.1 K (a few mountain and desert cells) |

So a region built here starts its deep soil about 2 K warmer than NOAA's recipe would. NOAA's own
static (the conus3km preset reads NOAA's published file) is not rebuilt, so that preset keeps NOAA's
field exactly. `--deep-soil-temperature islscp-i` selects NOAA's dataset by name (it prints its licence
note); the receipt's `deep_soil_temperature` block records which one a run used.

## Regions of any shape (country and regional doors)

`woof hex country` and `woof hex regional` with `--forecast-arg=--physics-suite=hrrrv5` take the
producer end to end on the region they cut: case 7 on the cut (its fork static built from the
geography) and case 9 boundaries from the source's hourly (GFS) or 3-hourly (IFS) leads. Both doors
take the same shapes: `--point LAT,LON --radius-km R`, `--ellipse LAT,LON,A_KM,B_KM[,AZIMUTH]` and
`--shape FILE` (a polygon, lat-lon box or cap Shape row, or GeoJSON; `country --polygon` as before).
A region larger than one card runs on several with `--samehost-devices 0,1,2,3` (one rank per card;
the region's partition is built from its own grid).

## Source pin selection

`--physics-pin-set rrfs-44c0849` selects the route whose `init_producer` is
`rrfs-mpas-44c0849`. Without an explicit pin, the door follows the route marked
`route_default` (implicit true for existing rows). The new route stays off default
until the whole forecast proof is accepted. The old `rrfs` pin remains selectable.

Both producers use `rw_mpas_init_v831` and the existing bridge ABI. The new row
selects configured hybrid coordinates (default true, transition 30,000 m), omits
retired `defc_a` and `defc_b`, includes `tke` from the new init Registry, and
clears unfrozen ice and snow over open water.
Static and GSL orography sources are unchanged between these pins.

## Proof at 1d6e370e

Every stored value is compared with the fork's own strict native `init_atmosphere` (GNU, single
precision, no FP contraction, one MPI rank) on real GFS inputs; no field is excluded and there is
no tolerance.

| Check | Result |
|---|---|
| Producer module replay, landed code: 22 case-7 inits (x1.2562 and x1.10242, 2023-03-31 and 2022-12-22, RUC9 and Noah4 soil, canonical hrrrv5, field-order fixtures, a CONUS cut, conus12km) and 104 case-9 boundary files (GFS and RRFS hydrometeor options, 25 hourly endpoints per date, conus12km) | 126 files, 1,359,006,343 values, 0 different |
| Fork static (MODIS 15 s, BNU, LAI, nvegopt 2) on x1.2562 | 74 fields, 1,316,394 values, 0 different |
| GSL drag statistics generator (78 native cases, x1.2562 to x1.40962) | 5,839,920 words, 0 different |
| The door's own job builder from the table: canonical hrrrv5 case 7 on both meshes and dates and the CONUS cut | 97,943,436 values, 0 different; every recipe value equal to the replayed jobs |
| The door's boundaries (carrier from its case-7 init) on four hourly case-9 files | 5,924,138 values, 0 different |

One exceptional-terrain fixture (non-finite terrain input) is bitwise equal to the replayed
producer, but a rerun of the native writes different NaN payloads than its original run did, so
that row is equal to the producer of record, not to a fresh native rerun.


## Proof at 44c0849

The referee is the strict single-precision, one-rank init derivative with exactly
two optional aerosol pointer `NULLIFY` statements carried from the earlier
approved init correction. The pristine native binary segfaults on these case-7
inputs. The reference publication labels the carried ruling as confirmation
pending; these results are against the labelled derivative.

| File | Stored values | Different |
|---|---:|---:|
| x1.2562, 2023-03-31 12Z, hrrrv5 RUC9 case 7 | 9,072,521 | 0 |
| lam7k, same time and recipe, case 7 | 34,645,137 | 0 |
| lam7k case 9, 12Z | 14,131,908 | 0 |
| lam7k case 9, 13Z | 14,131,908 | 0 |
| Total | 71,981,474 | 0 |

Both the engine module replay and the door's table-built job replay pass this
matrix with no excluded fields or tolerance. The engine's committed plan is
`tools/rustwx/crates/rw-mpas/tests/rrfs_native/module-file-replay-44c0849-v1.json`.
The door uses `tests/rrfs_init_door_replay.py --physics-pin-set rrfs-44c0849`.
The plan authenticates the published native executable, correction, audit,
rebuild, run manifests, inputs and reference files.

On an RTX 5090 host, the old worker/producer took 1.084 s prepare and 5.142 s case 7 on
x1.2562. The new door/producer took 0.733 s prepare and 2.194 s case 7 on the
same input. These are single-run wall times on a shared machine.
The final worker also preserves the old producer on x1.2562 and the regional
hourly cut: 13,096,075 stored values, zero differences.
