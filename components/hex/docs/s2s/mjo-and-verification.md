# S2S diagnostic and verification inputs

The S2S door now includes `export` and `diagnostics` stages. Export runs the
existing full-field export door for every completed member, using a persistent
common-time history binding over restart attempts. Its input list includes the
initial frame and every scheduled 6-hourly or daily frame through the requested
end. The newest completed file wins at restart overlaps. Each export uses that
member's own initialization and simulation start.

`diagnostics` reads native OLR and zonal winds through `rw_s2s`, interpolates
winds in log pressure to 850 and 200 hPa, integrates daily endpoint trapezoids,
and projects anomalies on fixed observed EOFs. `rw_wrfbatch` renders the RMM
and BSISO phase trajectories and longitude-time Hovmoller diagrams. BSISO also
writes latitude-time diagrams. Python schedules these programs and reads their
receipts. It does not process weather arrays or draw fields.

These are **RMM-style fixed-basis indices**. Equivalence to a particular
operational index requires its reference EOFs, normalization and preprocessing.
The code does not fit EOFs to the forecast or distribute an invented reference.
The [RMM method](https://www.bom.gov.au/climate/mjo/about/WH04.pdf) uses
equatorial OLR, u850 and u200. The
[BSISO definition](https://iprc.soest.hawaii.edu/users/jylee/bsiso/definition.htm)
uses spatial OLR and u850 over the Asian monsoon region and four observed EOFs.

Build the native reducer with:

```sh
cargo build --locked --release --manifest-path tools/rustwx/Cargo.toml -p rw-s2s
```

Build the diagnostic renderer in a private copy of the installed engine source:

```sh
CARGO_TARGET_DIR=/absolute/private/target \
  bash tools/build_s2s_diagnostics.sh /read-only/engine /absolute/private/renderer
export WOOF_HEX_RW_S2S=/absolute/private/renderer/rw_s2s
export WOOF_RW_WRFBATCH=/absolute/private/renderer/rw_wrfbatch
```

The build keeps the existing ensemble-renderer capabilities. Its additional
probe is `rw_wrfbatch --s2s-diagnostics-abi`. The runtime resolves the renderer
and its environment through `woof.rustwx`. A clean engine installation needs
these built binaries; no binary is downloaded implicitly by this path.

For a qualified reference bundle and prepared observation windows:

```sh
woof hex s2s --row global-120km --source era5 --cycle 2019-11-01T00 \
  --days 90 --members 4 --work-dir /absolute/case \
  --surface-source era5 --mjo-reference /absolute/reference/mjo.json \
  --s2s-observations /absolute/reference/observations.json \
  --s2s-reference-license CC-BY-4.0
```

This is a command template, not a measured run. Use the actual licence declared
by the supplied data. No non-permissive data is fetched or enabled by default.
The explicit research opt-in must match every non-permissive reference licence.

The independent reference preparation driver acquires public NOAA interpolated
daily OLR and keyless ARCO ERA5 zonal winds. All NetCDF reads and CF unpacking
use `rw_s2s extract`; compressed Zarr decoding uses the Rust bridge. Binning,
seasonal fitting, filtering, covariance and eigenvectors run on CUDA. For example:

```sh
python -m woof.hex.s2s_reference --start 2019-11-01_00:00:00 --days 30 \
  --first-year 2000 --last-year 2009 --output /absolute/reference
```

This fits three annual harmonics on the independent training years, subtracts
the preceding 120-day anomaly mean, applies one RMS scale per field, and
diagonalizes the uncentred training covariance. PC scaling is the training RMS.
Signs use the largest EOF coefficient. These independent modes do not reproduce
the operational RMM phase orientation. The reference receipt records the exact
sources, chunk hashes, training days, retained features and explained variance.
An observed orography mask excludes u850 features with less than 95 percent of
their source-bin area below 1100 m. The same fixed mask applies to forecasts,
seasonal fields and prior observations; missing unmasked features still fail.
No forecast enters the training sample.

After the product stage has written its window grid and independent historical
climatology, create the verification manifest on that same grid:

```sh
python -m woof.hex.s2s_reference --mode weekly \
  --start 2019-11-01_00:00:00 --days 30 --first-year 2000 --last-year 2009 \
  --output /absolute/reference --product-dir /absolute/case/s2s-products \
  --climatology /absolute/case/era5-window-climatology.nc
```

The first/last years describe the independent index training period. Weekly
anomalies use the separate product climatology and retain its identity. Weekly
ERA5 references use six-hour endpoint trapezoids for instantaneous fields and
following-interval precipitation accumulations, with the documented terrain
adjustment for T2. Complete seven-day windows are scored; trailing partial
weeks are excluded. NOAA data are public domain with source attribution, and
the current ERA5 catalogue identifies its CC-BY licence. Source and attribution
links are retained in the reference receipts. Raw downloaded OLR and native
array transports are disposable after receipts and references are retained.

Without a reference bundle an all-stage run records `diagnostics` as
`unavailable`. Asking for `--stage diagnostics` without it raises an error. No
weekly ERA5 or RMM skill is claimed when `--s2s-observations` is absent. The
existing first-ten-day analysis verification remains the fallback in that case.

## MJO reference JSON

Top-level required keys:

| Key | Contract |
| --- | --- |
| `schema` | `gpuwm-hex.s2s-mjo-reference/v1` |
| `source`, `training_period`, `basis_id` | Nonempty source, training period and stable basis identity |
| `license` | SPDX identifier or explicit named data licence |
| `daily_interval_start_hour` | UTC hour anchoring the trained daily intervals; must match the common forecast clock |
| `method` | `fixed-observed-eofs` |
| `raw_fields` | Mapping from `olr`, `pressure`, `u` to native variable names |
| `definitions` | List containing `rmm`, optionally `bsiso1` and `bsiso2` |

Each definition has these required keys:

| Key | Contract |
| --- | --- |
| `id` | `rmm`, `bsiso1` or `bsiso2`, no duplicates |
| `fields` | RMM: `["olr","u850","u200"]`; BSISO: `["olr","u850"]` |
| `domain` | RMM: `[-15,15,0,360]`; BSISO: `[-10,40,40,160]` in degrees |
| `nx`, `ny` | Spatial bins, west to east then south to north; RMM requires `ny=1` |
| `standard_deviation` | Positive training field scales in field order |
| `eofs` | Two unit-norm orthogonal EOF vectors, each of length `fields*ny*nx` |
| `eof_modes` | `[1,2]` for RMM/BSISO1; `[3,4]` for BSISO2 |
| `pc_standard_deviation` | Two positive training PC scales |
| `phase_order` | Eight phase numbers indexed by angle sectors starting at positive PC1; conventional RMM is `[5,6,7,8,1,2,3,4]` |
| `seasonal` | `YYYY-MM-DD` to raw-unit background vector for every forecast daily interval |
| `prior_anomalies` | `YYYY-MM-DD` to raw-unit preprocessed anomaly vector for all 120 dates before the common start |

`minimum_bin_area_fraction` defaults to 0.9 and must be in 0.8..1. Winds below
ground are missing, not zero. Spatial bins renormalize finite native cell areas
only when this support threshold is met. All EOF features must then be finite.
An optional fixed boolean `feature_mask` has the full EOF feature length. A
false entry excludes that feature from the EOF, seasonal/prior vectors and
forecast projection; its EOF coefficients must be zero. Observed preparation
uses this for fixed terrain support of the 850 hPa surface. This differs from
turning an unavailable contributing cell into a measured zero.

The feature order is field, latitude bin, longitude bin. RMM latitude averaging
uses native cell areas across the equatorial band. BSISO retains spatial bins.
Bins use the same native area-weighted definition as the supplied EOF training
grid. No implicit spatial remapping of EOFs occurs. Every bin must be populated.
Native mesh coordinate order and `xtime` must match the bound history.

The input pressure is in Pa and zonal wind is in m/s; OLR is positive outgoing
in W/m2. No sign inference or accumulated-radiation conversion is attempted.
Those transformations belong to preparation of the qualified reference.

Daily intervals are labelled by their starting date. The daily seasonal
background is removed first. The mean of the preceding 120 daily anomalies is
then removed causally, followed by the supplied training field scale and EOF
projection. Prior anomalies are unfiltered seasonal anomalies, before that
120-day subtraction. Completed forecast days replace the oldest prior days.
The supplied seasonal backgrounds must incorporate any additional trained
annual-cycle/interannual removal needed by the chosen basis. No acausal future
observation is used to remove the background of a forecast day.

The receipt records every member's PC pair, amplitude, phase (zero below unit
amplitude), raw-unit Hovmoller anomalies, binding identity, source, licence,
training period and method qualification. It records Rust wall time, Rayon
thread count and zero GPU memory for this CPU reducer. It does not substitute
for forecast-stage GPU memory telemetry.

`s2s-mjo/stage-receipt.json` tracks the complete reduction and rendering
attempt separately from the numerical `receipt.json`. It records failed
rendering attempts even when the numerical reduction succeeded. Export and
native reduction retries invalidate earlier success before validating inputs.
The S2S state also replaces the old export, diagnostics or verification stage
entry when its retry starts or fails. Rendered products must be acknowledged
by that renderer invocation and must cover exactly the expected member,
diagnostic and field set. Old PNGs in the folder do not count. The stage
receipt retains figure SHA-256 hashes, render wall time and the numerical
receipt path; it does not duplicate the full daily diagnostic series.

## Observation manifest and reference NetCDF

`--s2s-observations` is JSON with:

| Key | Contract |
| --- | --- |
| `schema` | `gpuwm-hex.s2s-observations/v1` |
| `source` | `ERA5` |
| `license`, `provenance` | Explicit licence and nonempty preparation/source provenance |
| `minimum_area_fraction` | Required paired global area, at least 0.95 |
| `weeks` | Exactly one row for every complete requested week |
| `rmm` | Observed PC pairs keyed by daily interval start date, covering the full run |
| `rmm_basis_id` | Must match the forecast MJO reference basis identity |
| `rmm_sampling` | `daily-interval-start` |
| `rmm_interval_start` | Exact UTC start of the first observed 24-hour index interval, matching the forecast start |

Each `weeks` row has `week`, `start`, `end`, `observations` and `climatology`.
Times use `YYYY-MM-DD_HH:MM:SS` and describe the exact common verification
window. Paths are absolute or relative to the observation manifest.

Both reference NetCDF files contain two-dimensional `XLAT`, `XLONG`,
`model_orography`, `temperature2`, `precipitation_rate`, `z500` and `mslp` on
`(south_north,west_east)`. Units are `degrees_north`, `degrees_east`, `m`, `K`,
`mm day-1`, `m`, `hPa`, respectively. The coordinates and model terrain must
match the existing weekly products. Prepare reference temperature on that
declared comparison terrain. The grid must be a regular complete globe with
both poles and no duplicate longitude seam. Fill values and CF scale/offset
are decoded by Rust.

Global attributes are `s2s_reference_source="ERA5"`,
`s2s_reference_kind="observations"` or `"climatology"`, and
`s2s_window_start` / `s2s_window_end` matching the row. Observations are weekly
means and mean daily precipitation over those dates. Climatology is the
independently prepared ERA5 normal for the same seasonal window, never the
forecast ensemble mean. Preserve its training period and preparation provenance
in the manifest's `provenance`.

For each member and ensemble mean, scores are cosine-latitude-weighted RMSE,
bias and uncentred spatial anomaly correlation about that ERA5 normal. Missing
pairs reduce the recorded area coverage; insufficient coverage fails. RMM
correlation is the uncentred two-component inner product normalized by the two
vector norms, with bivariate RMSE. Scores cover each complete week and the
whole daily RMM series. Undefined zero-norm correlations are JSON null.
Trailing days outside a complete week are explicitly counted as unscored for
the weekly atmospheric scores.

The reducer has no credentials or acquisition code. Preparing a qualified EOF
bundle, seasonal backgrounds, observation windows and the real model hindcast
is required before these products can establish a scientific capability claim.
