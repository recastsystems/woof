# Smoke, dust and air quality

WOOF-ArWen can carry wildfire smoke, dust, sea salt, black and organic carbon and
the CAMS air-quality gases through a forecast, at the forecast's own
resolution. Everything here is off by default: a configuration without
`chem_sets` runs exactly as it always has, and its wrfout files are byte for
byte the same.

There is no gas-phase chemistry. Ozone, NO2, CO and SO2 are carried,
mixed, deposited and taken in at the edges; nothing produces or destroys
them. That is a deliberate scope, not a gap waiting to be filled.

## Turning it on

Each option is a set of species, named in `[shared]`, plus the sources
that feed them.

### Wildfire smoke (HRRR-Smoke style)

HRRR-Smoke is off by default in 2.9.4 (opt in with --smoke; it currently costs ~11x per step); history is hourly (the 15-min history subset stream is next release). The packaged `hrrr_v4_gsd41` recipe ships smoke off; `--smoke` on `woof go`, `woof domain --recipe`, `woof standby` and `woof race` turns on its full HRRR-Smoke set (measured on full CONUS, 8x RTX 5090). A card that cannot hold the smoke still drops it for that run and says so.

```toml
[shared]
chem_sets = "smoke"
chem_sources = "rave-3km"
```

One transported smoke tracer (fire PM2.5). Fire emissions come from NOAA's
RAVE satellite fire product (public, no key), placed on the grid with a
mass-exact remap; the Freitas plume model lifts them (recomputed every hour),
a diurnal cycle spreads them over the day, vertical mixing carries them
through the boundary layer, and large-scale rain washes them out
(`wetscav_onoff = -1`, the default with `smoke`; set 0 to turn it off).

Fire timing, `fire_emission_mode`:

| value | what it uses | when |
|---|---|---|
| `"trailing_24h_dcycle"` | each fire's mean emission over the hours it burned in the 24 hours before the start (the newest posted hour included, so a fire a few hours old already emits), on a diurnal cycle peaking in the late afternoon | the default; HRRR-Smoke's method, what a real-time forecast has |
| `"observed_hourly"` | the hour itself | hindcasts only; refused past the newest posted file |
| `"daily_mean_dcycle"` | NOAA RRFS's daily mean scaled by fire-weather potential | needs the previous day's model fire-weather fields, so it is refused at a cold start |

The previous default, the same hour of the previous day
(`"persistence_hourly"`), is refused: a fire younger than a day emitted
nothing under it.

Smoke already in the air: a run that starts from HRRR (`[fetch] source =
"hrrr"` or `"hrrr-prs"`) takes HRRR-Smoke's own 3-D smoke as its start
state and as its edge values every hour (the `hrrr-native-smoke` source,
added to `chem_sources` for you), so it begins with the smoke HRRR has in
the air and smoke from fires outside the domain blows in. `woof go`
fetches it with the forecast's other data; by hand:

```
python -m woof.data_store_fetch fetch --source hrrr-native-smoke --area S,W,N,E --start START --end END
```

A run that starts from anything else starts with no smoke and takes none in
at the edges, so it spins its smoke up from the fires inside the domain.

### Dust, sea salt and carbon (GOCART-lite)

```toml
[shared]
chem_sets = "gocart_primary"
dust_opt = 1
seas_opt = 1
```

`gocart_primary` carries five dust bins, four sea-salt bins and the
hydrophobic and hydrophilic black and organic carbon, with WRF-Chem's
GOCART dust (`dust_opt = 1`) or AFWA dust (`dust_opt = 3`), GOCART sea salt,
gravitational settling, dry deposition and carbon aging, and computes the
550 nm extinction and aerosol optical depth. `dust` carries the dust bins
alone. The dust source needs three static datasets the default geography
download leaves out:

```
woof fetch-geog --datasets chem-dust
```

`gocart_lite` and `gocart_simple` add the sulfur rows (SO2 to sulfate,
WRF-Chem's simple GOCART chemistry). They need background OH, H2O2 and NO3
from the CAMS forecast (`chem_sources = "cams-oxidants"`, which needs the
Atmosphere Data Store key below); without them the configuration is
refused.

### Air quality from the CAMS global forecast

```toml
[shared]
chem_sets = "cams_aq"
chem_sources = "cams-global"
```

Ozone, NO2, CO and SO2 from the Copernicus Atmosphere Monitoring Service
forecast, interpolated to the domain at its start and fed in at its edges as
the forecast runs, then carried with WRF-Chem's Wesely dry deposition. With
`cams-global` enabled, the GOCART dust, sea-salt, carbon and sulfate rows
take their start and edge values from CAMS as well.

CAMS data comes from the Atmosphere Data Store and needs your own
personal access token in `~/.adsapirc`:

```
url: https://ads.atmosphere.copernicus.eu/api
key: <your token>
```

and, once, the store's terms of use, data protection statement and the
CAMS product licence accepted on its website with the same account. Until
then every request is refused and `woof go` stops at its chem fetch stage
naming the remedy.

### Sets together

Sets combine: `chem_sets = "gocart_primary,smoke"` runs both, and the
PM2.5 and PM10 fields then include the smoke.

### Aerosol-aware microphysics

With the aerosol-aware Thompson scheme (`mp_physics = 28`),

```toml
[shared]
aerosol_mp_coupling = "diagnose"
```

sets Thompson's water-friendly and ice-friendly aerosol numbers from the
dust, sea-salt, sulfate and organic-carbon rows after every chem step (NOAA
GSL's `get_niwfa`), so the clouds form on the forecast's own aerosol. Those
rows must start from real values: enable `cams-global` in `chem_sources` so
CAMS fills them at the start and at the edges. Without a source that fills
them they start at zero, the clouds would form in almost clean air, and the
configuration is refused.

## What it writes

Every species goes to wrfout under its WRF-Chem name (`smoke`, `DUST_1` to
`DUST_5`, `SEAS_1` to `SEAS_4`, `BC1`, `BC2`, `OC1`, `OC2`, `so2`, `sulf`,
`o3`, `no2`, `co`), with:

| field | meaning | units |
|---|---|---|
| `PM2_5_DRY`, `PM10` | particulate matter (WRF-Chem's GOCART sums, plus smoke) | ug m-3 |
| `DUST_SFC` | near-surface dust | ug m-3 |
| `SMOKE_SFC` | near-surface smoke | ug m-3 |
| `SMOKE_COLUMN` | smoke in the whole column | mg m-2 |
| `EXTCOF55`, `AOD5502D` | 550 nm extinction and aerosol optical depth | km-1, 1 |
| `EDUST1` to `EDUST5`, `ESEAS1` to `ESEAS4` | emitted dust and sea salt | kg m-2 |
| `FRP_MEAN`, `PLUME_TOP`, `FIRE_EMITTED` | fire power, plume top, emitted smoke | MW, m, kg |

The run receipt carries a mass ledger for every species: what was emitted,
moved through the edges, deposited, settled, washed out and left in the
air, closing to float64 rounding.

The Rust renderer draws `smoke_near_surface`, `smoke_column`,
`pm25_near_surface` (coloured by the EPA AQI breakpoints), `aod_550`,
`dust_near_surface` and `ozone_near_surface` from any wrfout that carries
their fields:

```
woof render --products smoke_near_surface,pm25_near_surface,aod_550 OUT/wrfout_d01_*
```

## What it is not

- No gas-phase chemistry, by design (see the top of this page).
- `aer_ra_feedback = 1` in `[shared]` connects active `optics.gocart`
  mass to option 4 RRTMG or RRTMGP radiation, including clear-sky fluxes.
  See [chemistry radiation feedback](../chem-radiation-feedback.md) for
  spectral assumptions, bulk-smoke composition and event qualification.
- With a cumulus scheme on (grids coarser than about 4 km), set
  `chem_conv_tr = 0`: convective transport of the species is not
  transcribed, and leaving it at WRF's default 1 is refused.
- Anthropogenic emission inventories are not read: the EDGAR rows exist,
  but enabling `edgar-v81` is refused because no download or remap brings
  its files onto the grid yet.
- MYNN's own chem mixing (`mynn_chem_vertmx`) is refused; species mix
  through WRF-Chem's `vertmx` with any boundary-layer scheme.
- Tile streaming is refused for chem runs.

Every refusal names what would go wrong, and the configuration is checked
before anything is downloaded.
