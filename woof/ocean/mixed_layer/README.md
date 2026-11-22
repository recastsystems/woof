# Prognostic mixed-layer ocean

`sf_ocean_physics = 1` selects the existing Pollard mixed layer. Option 2
selects the CUDA translation of WRF v4.6.1's three-dimensional PWP ocean,
including horizontal and vertical advection, perturbation pressure gradients,
Coriolis rotation, penetrating radiation, evaporation, convective adjustment,
and bulk/gradient Richardson mixing. Its atmospheric call slot is after the
land surface and before lake physics. Inland lakes are excluded.

An omitted `sf_ocean_physics` is 0 (no ocean model, WRF's default) at every
run length; options 1 and 2 are chosen by setting them. Through 3.0.1 a run of
48 h or more with surface fluxes took option 2 by itself, so a 47 h and a 48 h
run of one case differed from the first hour. The plan (`woof check`, the
`woof go` banner and its `--dry-run`, the `woof run --met-em` plan review),
the run log (`note: ocean:`) and the run record (`report.json` key `ocean`)
state the ocean every run, and for a surface-coupled run of 48 h or more with
no ocean model they add a one-line offer of option 2. The words and the offer's
threshold are metadata in `woof/data/surface_ocean.json`. Ocean physics acts
on water cells, excluding land and inland lakes. The 72-hour coastal storm
qualification is recorded in [QUALIFICATION.md](QUALIFICATION.md); it scored
one run with option 2 on and none with it off, and one event is not a general
skill claim.

An experiment's `[shared]` table accepts:

```toml
sf_ocean_physics = 2
ocean_levels = 30
omdt = 1.0
ocean_analysis = "/absolute/path/ocean-analysis.nc"
ocean_analysis_source = "hycom"
```

`omdt` is the maximum ocean substep in minutes. Every surface interval is
advanced completely, splitting the interval equally when necessary. This
preserves the atmospheric coupling clock; it does not reproduce WRF's
separate scheduler. Public RTOFS and HYCOM analysis variable bindings, plus a
CF binding, live in `analysis.py`. Additional source bindings are table rows.
The caller must retain the source's data terms. NetCDF decoding and CF time
decoding use `woof.netcdf_bridge` and the Rust `rw_netcdf` executable.

The analysis must contain one time within the preceding 24 hours of the
forecast start, with temperature, salinity, and both earth-relative currents.
GPU mapping selects the nearest complete profile within one great-circle
degree. Currents rotate into the forecast grid using its sine/cosine planes.
Profiles interpolate to 5 m layer bottoms. This follows WRF's flat-depth ocean,
without bathymetry: coastal points may borrow a complete offshore profile
within the search radius. Uncovered ocean is refused. This limitation needs
assessment in the full coastal storm qualification.

Explicit `ocean_z`, `ocean_t`, and `ocean_s` arrays can instead supply layer
bottoms in metres, temperature in Kelvin, and salinity in PSU. They must each
contain `ocean_levels` values. WRF input OM_TMP/OM_S/OM_DEPTH and optional
OM_U/OM_V/OM_TINI/OM_SINI fields are also accepted. They are already in model
grid coordinates. All ocean profiles use a separate `ocean_levels` history
axis and persist through restart along with circulation scratch.

Without an analysis or explicit profile, initialization fetches the calendar
month's NOAA WOA18 (2005–2017) quarter-degree mixed-layer-depth climatology.
This is the potential-density increase of 0.125 kg/m³ from 10 m, from
Locarnini et al. (2018), NOAA Atlas NESDIS 81, and Zweng et al. (2018),
NOAA Atlas NESDIS 82. Rust reads `M_an`; CUDA maps the nearest valid MLD
within one degree. Atmospheric analysis SST anchors a profile using the
configured thermocline lapse rate, 35 PSU, and resting currents. The
fallback is named in the initialization warning and acquisition receipt;
it does not supply the current ocean's anomalies. NOAA's source metadata
states “These data are openly available to the public.” The NCEI open-data
policy identifies NOAA-produced data as public domain in the United States:
https://www.ncei.noaa.gov/sites/default/files/2023-12/NCEI%20PD-10-2-02%20-%20Open%20Data%20Policy%20Signed.pdf .

`WOOF_OCEAN_CACHE` relocates the managed download cache (otherwise the
normal woof case-data root's `ocean/` subdirectory). Alternatively,
`ocean_mld_climatology = "configured_uniform_mld"` explicitly selects the
spatially uniform `oml_hml0` fallback and logs that choice. Public analysis files are supplied through `ocean_analysis`; automatic
contemporary-analysis acquisition is not provided. Without a file the named
WOA18 MLD fallback runs automatically, including its stated missing anomalies.
The qualification used that fallback after the public HYCOM subset service
returned HTTP 500; it did not test contemporary-analysis skill.

The adjacent WRF-NOTICE.txt carries the upstream public-domain notice.
The port intentionally repairs zero-radiation absorption and the upstream
integer truncation in unequal-layer stirring. It masks land/lake stencil
neighbors and uses symmetric radiative conditions at domain corners rather
than WRF's temperature-only corner assignment. No bitwise WRF parity or
observational skill is claimed. The atmosphere supplies its own Coriolis
plane rather than recomputing WRF's approximate latitude formula.

Both ocean options publish `OCEAN_SST` (K) and `OCEAN_SST_CHANGE` (K from
the initial coupled SST). Land and lakes are missing values. Render these
stored planes through `woof render --engine rust --products
var:wrf_ocean_sst,var:wrf_ocean_sst_change`; this uses the normal
`woof.rustwx`/`rw_wrfbatch` front door without a separate plotter.

Native observation verification is reached with
`python -m woof.ocean.mixed_layer.verify /absolute/request.json`. Its
on-demand artifact `rw_ocean_verify` delegates NetCDF decoding to the installed
`rw_netcdf`, parses NDBC standard-met WTMP in Rust, and writes SST bias,
RMSE, MAE and paired cooling-change scores. The request uses absolute paths:

```json
{
  "forecast": ["/run/wrfout_d01_2022-09-26_00:00:00"],
  "oisst": ["/observations/oisst-daily.nc"],
  "buoys": [{"id": "station", "path": "/observations/standard-met.txt",
             "latitude": 25.0, "longitude": -85.0}],
  "output": "/run/ocean-verification.json",
  "jobs": 8,
  "daily_max_gap_hours": 3,
  "buoy_max_distance_km": 25.0,
  "buoy_match_minutes": 30
}
```

Forecast files must contain OCEAN_SST, XLAT, XLONG and Times. OISST files
must be chronological, with a single CF time and sst/lat/lon. Complete UTC
days are integrated with a time-weighted left-interval mean; missing temporal
support stays missing. Incomplete days are labelled and omitted from scoring.
Nearest OISST cells are matched to the model grid within half a degree;
cosine-latitude weights apply on model support. This is not area-conservative.
Buoys match the nearest persistently wet model cell within the stated radius,
and one report nearest each forecast time within the stated window; each
report is used once. JSON carries matched pairs, source hashes, wall time,
thread count, unmatched stations and limitations. OISST includes buoy SST, so
these two comparisons are not independent. Neither comparison corrects for
foundation SST versus model layer temperature or buoy sensor depth. The qualification reports measured errors for one event; it makes no general
observation-skill claim.
