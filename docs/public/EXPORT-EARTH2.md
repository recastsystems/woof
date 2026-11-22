# Export WOOF output for Earth2Studio

`woof export --earth2 PROFILE` converts WOOF's wrfout-shaped histories,
including WOOF-ArWen output and hex's converted frames, into per-channel Zarr
stores for Earth2Studio 0.19.0. It uses the Rust engine behind
[`woof ml-export`](../ml-export.md) for pressure interpolation, wind rotation
and regridding. `--for` selects a consumer's input channels from metadata;
exporting does not download or run model checkpoints.

<!-- Regenerate tables with: python tools/gen_export_earth2_doc.py -->

## Profiles

`e2s-native` keeps the regional grid with two-dimensional `lat` and `lon`.
The fixed grids below come from the packaged
[`profiles.json`](../../woof/data/ml_export/profiles.json).
HEALPix uses DLESyM's `HEALPIX_PAD_XY` ordering; the HRRR profile uses
Lambert geometry with metre-valued `dx` and `dy`.

<!-- BEGIN GENERATED earth2-profiles -->

| Profile | Kind | Spatial dimensions | Shape | Grid parameters |
| --- | --- | --- | --- | --- |
| `e2s-latlon-0.25` | `latlon` | `lat`, `lon` | 721 x 1440 | `lat0`=`90.0`; `lon0`=`0.0`; `dlat`=`-0.25`; `dlon`=`0.25` |
| `e2s-healpix` | `healpix` | `face`, `height`, `width` | 12 x 64 x 64 | `nside`=`64`; `ordering`=`HEALPIX_PAD_XY` |
| `e2s-native` | `native` | `y`, `x` |  |  |
| `e2s-hrrr-3km` | `lambert` | `hrrr_y`, `hrrr_x` | 1059 x 1799 | `lat0`=`21.138123`; `lon0`=`237.280472`; `truelat1`=`38.5`; `truelat2`=`38.5`; `stand_lon`=`262.5`; `dx`=`3000.0`; `dy`=`3000.0`; `radius`=`6371229.0` |

<!-- END GENERATED earth2-profiles -->

## Channels

`--for NAME[,NAME]` exports the union of the ordered channel lists in
[`earth2-models.v1.json`](../../woof/authorities/earth2-models.v1.json).
`--channels NAME[,NAME]` adds channels, such as `tp06`, to that union.

For a history directory `HISTORY`, the four global consumers share one export:

```bash
woof export HISTORY --earth2 e2s-latlon-0.25 \
  --for windgust-afno,precip-afno,climatenet,tc-tracker \
  --channels tp06 --out earth2
```

<!-- BEGIN GENERATED earth2-channels -->

| Consumer | Earth2Studio class | Export profile | Declared input shape | Count | Input channels (in order) |
| --- | --- | --- | --- | --- | --- |
| `windgust-afno` | `WindgustAFNO` | `e2s-latlon-0.25` | 720 x 1440 | 17 | `u100m`, `v100m`, `msl`, `u300`, `u850`, `u925`, `v300`, `v850`, `v925`, `z300`, `z850`, `z925`, `t850`, `t925`, `q500`, `q850`, `q925` |
| `precip-afno` | `PrecipitationAFNO` | `e2s-latlon-0.25` | 720 x 1440 | 20 | `u10m`, `v10m`, `t2m`, `sp`, `msl`, `t850`, `u1000`, `v1000`, `z1000`, `u850`, `v850`, `z850`, `u500`, `v500`, `z500`, `t500`, `z50`, `r500`, `r850`, `tcwv` |
| `climatenet` | `ClimateNet` | `e2s-latlon-0.25` |  | 4 | `tcwv`, `u850`, `v850`, `msl` |
| `tc-tracker` | `TCTrackerWuDuan` | `e2s-latlon-0.25` |  | 5 | `u10m`, `v10m`, `msl`, `u850`, `v850` |
| `stormcast-conus` | `StormCastCONUS` | `e2s-hrrr-3km` |  | 26 | `u10m`, `v10m`, `t2m`, `tcwv`, `sp`, `msl`, `u1000`, `u850`, `u500`, `u250`, `v1000`, `v850`, `v500`, `v250`, `z1000`, `z850`, `z500`, `z250`, `t1000`, `t850`, `t500`, `t250`, `q1000`, `q850`, `q500`, `q250` |

<!-- END GENERATED earth2-channels -->

Pressure suffixes are hPa. `z` is geopotential in m2 s-2, `q` is kg kg-1,
`r` is percent and `w` is omega in Pa s-1. `tp06` is the preceding six-hour
precipitation depth in metres. The
[`variable rows`](../../woof/data/ml_export/variables.json) define these units.

## GRIB2 rows

Export both pressure and surface files with the [GRIB2 export door](../grib2-export.md):

```bash
woof export-grib2 HISTORY --out grib2 --fields all --winds earth
```

SPFH, VVEL and DZDT are written in `wrfprs` at the requested pressure levels.
UGRD/VGRD at 100 m are written in `wrfsfc` as earth-relative winds.
The identities and formulas below come from
[`catalog.rs`](../../tools/rustwx/crates/rw-grib2export/src/catalog.rs).

<!-- BEGIN GENERATED earth2-grib2 -->

| Row | Discipline.category.parameter | Fixed surface | Units | Formula / method |
| --- | --- | --- | --- | --- |
| `specific_humidity` | `0.1.0` | 100/pressure in Pa | kg kg-1 | SPFH: rw-mlexport specific_humidity, QVAPOR/(1+QVAPOR+sum condensates), linear in ln(P+PB); lowest mass level held below ground, missing above top mass level |
| `omega` | `0.2.8` | 100/pressure in Pa | Pa s-1 | VVEL: rw-mlexport omega operator, -rho*g*w with rho=(P+PB)/(Rd*T*(1+0.61*max(QVAPOR,0))), W unstaggered to mass levels; linear in ln(P+PB); lowest mass level held below ground, missing above top mass level |
| `vertical_velocity` | `0.2.9` | 100/pressure in Pa | m s-1 | DZDT: w=(W[k]+W[k+1])/2 on mass levels, linear in ln(P+PB); lowest mass level held below ground, missing above top mass level |
| `u100` | `0.2.2` | 103/100 m AGL | m s-1 | UGRD: unstagger U/V to mass levels; linear in z = mean(PH+PHB)/g - HGT to 100 m AGL; rotate to east/north with SINALPHA/COSALPHA |
| `v100` | `0.2.3` | 103/100 m AGL | m s-1 | VGRD: unstagger U/V to mass levels; linear in z = mean(PH+PHB)/g - HGT to 100 m AGL; rotate to east/north with SINALPHA/COSALPHA |

<!-- END GENERATED earth2-grib2 -->

The 24-hour WOOF-ArWen proof exported 25 frames, decoded all 50 GRIB2 files and
passed 30 point comparisons against source arithmetic. Source omega at
500 hPa reached -42.12 Pa s-1. At +12 and +24 hours, aligned land-cell
100 m / 10 m median speed ratios were 1.44 and 1.71, using 10 m speeds
of at least 1 m s-1. These checks establish agreement with source carriers;
they do not validate physical magnitudes against observations.

## Examples

The four diagnostics below were run together on the five-frame hex sample,
from 2024-05-21 00Z through 2024-05-22 00Z. Set `WOOF_EARTH2_PYTHON` to
an interpreter with Earth2Studio 0.19.0 and the consumer extras named in
`earth2-models.v1.json`. With the store at `earth2/d01.zarr`, the command is:

```bash
woof earth2-diagnose earth2/d01.zarr \
  --for windgust-afno,precip-afno,climatenet,tc-tracker \
  --accept-licence 'NVIDIA Community Model License' \
  --accept-licence 'Not declared in the NGC asset metadata; check checkpoint licence before use' \
  --out diag
```

Only `tc-tracker` (an Apache-2.0 algorithm with no weights) runs bare. The
gust model's weights are under the NVIDIA Community Model License and the
precipitation and ClimateNet checkpoints declare no licence in their NGC
metadata, so the door refuses each of them by name until the exact licence
string from its `consumers` row is given with `--accept-licence`.

`diag/earth2-diagnose.json` records inputs, frames, timings and output files.
Paths here are relative to your working directory.

### Wind Gust AFNO

`windgust-afno.nc` contains preceding-hour maximum gust `fg10m`:
five frames, 0.3072 to 44.3899 m s-1 over this run, in 12.57 s.
Earth2Studio's coordinate mapping selects the
consumer's 720 latitude rows from the 721-row export.

### Precipitation AFNO

`precip-afno.nc` contains six-hour precipitation `tp` at each valid time:
five frames, 0 to 0.0491813 m over this run, in 9.58 s.
This is the diagnostic model's output; the export's `tp06`
channel comes from the simulation's accumulated precipitation.

### ClimateNet

`climatenet.nc` contains the three class probabilities `climnet_bg`,
`climnet_tc` and `climnet_ar`: five frames in 1.42 s. The tropical-cyclone
class reached 0.96046 in this run.

### TC tracker

`tc-tracker-tracks.csv` contains nine paths across the five input times;
the run took 5.75 s. Each CSV row carries path, step, valid time and the
tracker's output variables.

The [CF rendering guide](../cf-diagnostic-rendering.md) covers maps,
observation panels, probability contours and track overlays for these files.

### StormCast-CONUS conditioning

Export the WOOF-ArWen parent history, then run the packaged
[StormCast example](../../examples/earth2/stormcast_from_woof.py) in an
environment with `earth2studio[stormcast-conus]==0.19.0` and matching NATTEN:

```bash
woof export conus12/wrfout --earth2 e2s-hrrr-3km \
  --for stormcast-conus --out arwen12-hrrr
python examples/earth2/stormcast_from_woof.py \
  --init 2024-05-21T18 --hours 6 --cond arwen12-hrrr/d01.zarr \
  --window 401,913,643,1411 --out refc.nc
```

The example starts from HRRR analysis and reads the 26 conditioning channels
from the exported WOOF-ArWen forecast through `InferenceOutputSource`. It uses
`pregridded_conditioning()` for the already matching HRRR coordinates.
The measured 512 x 768 window produced seven hourly frames through
2024-05-22 00Z: 73.6 s forecast time and 7.53 GiB peak GPU memory on a
5070 Ti. `refc.nc` also contains `t2m`, `u10m`, `v10m` and `msl` by valid time.

## Quickstart

With Earth2Studio 0.19.0 installed, place an `e2s-latlon-0.25` export at
`earth2/d01.zarr`. This reads five channels through Earth2Studio's own
`InferenceOutputSource`, which combines initialization time and lead time into
valid times. The measured hex 60 km sample, initialized at 2024-05-21 00Z,
has five frames, six hours apart.

```bash
python - <<'PY'
from earth2studio.data import InferenceOutputSource

channels = ["t2m", "tcwv", "z500", "u100m", "tp06"]
source = InferenceOutputSource("earth2/d01.zarr", engine="zarr")
data = source(source.da.time.values, channels).load()
assert data.dims == ("time", "variable", "lat", "lon")
assert data.shape[-2:] == (721, 1440)
print("dimensions:", dict(data.sizes))
print("channels:", data["variable"].values.tolist())
print("valid times:", data.time.values.astype("datetime64[h]").astype(str).tolist())
print("dtype:", str(data.dtype))
PY
```
