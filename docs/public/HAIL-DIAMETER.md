# Maximum hail diameter

Convection-allowing moist runs at grid spacings of 4 km or finer enable hail
diagnostics automatically. `[shared] hailcast_opt = 0` opts out; `1` enables
them explicitly. `haildt` is the HAILCAST sampling interval in seconds, from
0 (every timestep) to 60 (the default). The PSD diagnostic runs every timestep.
These diagnostics read the forecast state and do not change its evolution.

The PSD diagnostic finds the diameter above which the integrated particle
number concentration is 0.0005 particles per cubic metre. It uses transported
mass, number and volume moments where the microphysics carries them, and the
scheme's specified distribution otherwise. It solves the continuous gamma
PSD tail, rather than WRF diag_nwp's 50 diameter-bin interpolation. It does not
apply WRF diag_nwp's 75 mm bin ceiling. There is no reflectivity, updraft-helicity
or graupel-mass regression in this calculation.

The spherical PSD table supports Lin, WSM6, WSM6R, WDM6, WSM7, Thompson,
aerosol-aware Thompson, Milbrandt, Morrison and NSSL two-moment. Unsupported
PSDs do not emit PSD diameter fields. HAILCAST remains available with other
moist schemes and reads their available hydrometeor categories. A missing
hydrometeor category supplies zero concentration.

HAILCAST has two stone models, chosen per domain with `hailcast_version`:

- `wrf_39_noaa` (the default) is the HAILCAST in NOAA's operational HRRR v4
  (NOAA-EMC/HRRR v4.1.21, WRF 3.9 based), the field HRRR posts as surface hail.
- `wrf_461` is WRF v4.6.1's. Its later ventilation, heat-budget and melting
  changes melt most small and medium stones below cloud base. On 5,367 real
  updraft columns from 3 km HRRR-configuration forecasts it lands a stone in
  8% of columns and 1 inch or more in 0.8%; `wrf_39_noaa` lands a stone in
  99% and 1 inch or more in 4.9%.

Each GPU stone model matches its own Fortran on real storm columns
(tests/data/oracles/hailcast). Both contain WRF's five-embryo growth,
fall-speed, wet/dry growth, melting and shedding integrations, and share the
rules below. A column becomes eligible after an updraft
of at least 10 m/s has persisted in its three-by-three neighborhood for more
than 900 seconds. The WRF 254 mm rejection remains. Two undefined upstream
cases, an embryo outside the interpolation column and an insertion below
cloud base, explicitly return no surviving stone. The empirical droplet impact-speed
fit is bounded at zero before calculating rime density: near its low Stokes-number
threshold the upstream fit can return a negative speed and make its fractional
power undefined. A captured real forecast trajectory reproduces that upstream
failure; the corrected independent Fortran integration retains a 23.935 mm stone.
Other nonfinite numerical
results remain nonfinite and verification refuses them. Columns currently
support up to 256 mass levels; larger columns are refused before allocation.

History windows retain each embryo's largest stone, with maximum, mean and
sample standard deviation across the five embryos. History submission resets
the window accumulators. Restart files preserve the window, run maximum,
updraft mask and duration. Run maxima never reset at history output.

| History field | Units | Meaning |
| --- | --- | --- |
| HAIL_MAXK1 | m | Window maximum PSD diameter at the lowest model level |
| HAIL_MAX2D | m | Window maximum PSD diameter across the column |
| HAIL_MAXK1_RUN | m | Entire-run maximum at the lowest level |
| HAIL_MAX2D_RUN | m | Entire-run maximum across the column |
| HAILCAST_DIAM_MAX | mm | Largest of five embryo window maxima |
| HAILCAST_DIAM_MEAN | mm | Mean of five embryo window maxima |
| HAILCAST_DIAM_STD | mm | Sample standard deviation of those five values |
| HAILCAST_RUNMAX | mm | Entire-run HAILCAST maximum |
| HAILCAST_DHAIL1 through HAILCAST_DHAIL5 | mm | Individual embryo window maxima |

`woof render --list-products` exposes `hail_max_surface`, `hail_max_column`,
`hail_max_surface_run`, `hail_max_column_run`, `hail_max_hailcast`,
`hail_max_hailcast_mean`, `hail_max_hailcast_std` and `hail_max_hailcast_run`.
Every plot displays millimetres, through the Rust renderer. Diameter bins
include 25.4, 38.1, 50.8 and 76.2 mm.

## Observations

`woof obs mrms` decodes MESH and its 30, 60, 120, 240, 360 and 1440 minute
maximum products in millimetres. Missing and no-coverage values remain masked;
zero means observed no hail. The hail verification door uses MESH_Max_30min,
with contiguous 30 minute model history windows. It rejects reflectivity,
stub observations, missing units and mismatched forecast timestamps.

```
woof obs hail spc --day YYYYMMDD --out reports.json
woof obs hail verify --request request.json --out verification.json
```

The SPC date is the report day from 12 UTC through 11:59 UTC the next day.
The Rust reader converts hundredths of inches to millimetres. Preliminary
SPC reports are sparse positive samples: missing reports never count as
observed no hail. MESH is a radar size estimate, not a measured stone.

A request binds actual model output, decoded observation packs and geometry:

```json
{
  "schema": "gpuwm.hail-verification-request.v1",
  "spc": "/absolute/path/reports.json",
  "max_distance_m": 5000,
  "frames": [{
    "wrfout": "/absolute/path/wrfout_d01_YYYY-MM-DD_HH_MM_SS",
    "valid_time": "YYYY-MM-DDTHH:MM:SSZ",
    "window_seconds": 1800,
    "mesh": "/absolute/path/mesh.pack",
    "geometry": "/absolute/path/mesh.geometry.pack"
  }]
}
```

The receipt carries input hashes, matched-cell counts, diameter-bin confusion
matrices, MAE/RMSE and threshold POD/FAR/CSI. SPC comparisons retain each
report's observed size and model diameter at the nearest interior cell and
within 40 km. The boundary forcing strip is excluded; unmatched reports are
not negative observations. Column PSD, surface PSD and HAILCAST are scored
separately because their physical interpretations differ.

The pinned source, generator and independent Fortran comparison tools live
in `tools/hail_wrf461`. WRF's public-domain notice ships in
`licenses/WRF-HAILCAST.txt`. NOAA observation terms are at
https://www.weather.gov/disclaimer and SPC report conventions at
https://www.spc.noaa.gov/climo/reports/.
