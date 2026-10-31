# Solar energy products

Surface irradiance is available directly in forecast history. All four fields
below are fluxes in W/m2, with the native radiation solver's values:

| Field | Quantity |
| --- | --- |
| `SWDOWN` | Global horizontal irradiance (GHI), direct plus diffuse |
| `SWDDNI` | Direct normal irradiance (DNI), normal to the solar beam |
| `SWDDIF` | Diffuse horizontal irradiance (DHI) |
| `SWDDIR` | Direct horizontal irradiance |

## Transported smoke and irradiance

For a configuration with transported smoke or aerosol optical mass rows:

```sh
woof go examples/solar-smoke/northeast.toml --smoke-radiation on --outdir smoke-on --products none
woof go examples/solar-smoke/northeast.toml --smoke-radiation off --outdir smoke-off --products none
```

`on` sets `aer_ra_feedback=1`, `aer_op_opt=1`, `aer_opt=0`, and
`simple_dir_fdb=0` in a per-run copy. This selects live aerosol optics in place
of climatology and the separate shortwave-only smoke addition. It preserves
transport, source selection, cloud activation and radiation solver selection.
The radiation solvers must consume those optics; incompatible settings and a
prescribed smoke manifest refuse with the conflicting physics named. A request
without optical mass rows refuses as `smoke-radiation-no-optics`. A capacity
fallback cannot discard smoke after an explicit ON request.

`off` disables `aer_ra_feedback` and `simple_dir_fdb`, retaining transport and
the chosen background aerosol setting. Omit the option to preserve the
configuration exactly, including an operational recipe's own radiation choice.
The existing `--smoke` transport shortcut keeps its original behavior unless
`--smoke-radiation` is also supplied. Apply the option before preparation;
prepared inputs and restarts retain their bound physics configuration.

The same engine setting is WRF `&chem aer_ra_feedback = 1` or Python
`RunConfig(..., chem_sets="smoke", aer_ra_feedback=1, aer_op_opt=1,
aer_opt=0, simple_dir_fdb=0, ra_lw_physics=4, ra_sw_physics=4,
ra_rrtmg_variant="rrtmg_legacy")`. Populate initial, boundary or emission
sources separately. For a parsed experiment document, the Python API
`woof.smoke_radiation.apply_smoke_radiation(document, "on")` applies the same
choice; normal experiment validation checks the complete physics selection.

Legacy RRTMG evaluates the full-domain Mie optics once per radiation call and
gathers each solver chunk from those bands. The cache refreshes at the next
radiation time and is absent when feedback is off. The retained band storage
and overlapping refresh are included in memory admission. Modern RRTMGP keeps
its separate spectral implementation. See [chemistry radiation feedback](chem-radiation-feedback.md)
for the optical assumptions: bulk smoke represents organic particulate mass
and does not resolve soot or brown carbon.

The [example](../examples/solar-smoke/northeast.toml) uses public NOAA initial
and boundary smoke for a 3 km, 24-hour Northeast smoke event. Neither a source
declaration nor an empty diagnostic allocation proves smoke is present. No
automatic smoke threshold or default feedback change is selected.

Native forcing preparation also corrects a source-role defect introduced with
separate initial and boundary smoke sources in 2.9.4: lateral slabs could read
the initial-only source at later forcing times. The initial state now reads the
initial source, and later slabs read the configured boundary source. This affects
new preparation through `woof go`, `woof prep`, and the native preparation API;
it changes smoke forcing where those sources differ. Prepare again to apply the
correction to an older prepared bundle. The optics comparison uses this correction
in both arms so a source change cannot be mistaken for an optics effect.

Acquisition uses table-declared extended forecast cycles when the newest cycle
cannot cover the requested interval. Initial-only sources are fetched for the
initial bracket; boundary and emission sources cover their consumed forecast
window. A request beyond every available cycle still refuses with its horizon.

## Measured example

The Northeast example completed 24 hours in each arm, with 7,200 steps and
25 hourly histories. At Penn State SURFRAD, daylight GHI RMSE decreased from
124.06 W/m2 with feedback OFF to 105.49 W/m2 with feedback ON across the same
14 matched hours. Both runs remained too bright: daylight bias was +106.44 and
+86.87 W/m2, respectively. This is one case and one radiation station.

The maps show a broad reduction in surface sunlight over northern New York
moving east and south through the afternoon, plus localized changes in cloud
shadows. At the 19 UTC maximum of domain-mean ON temperature, the northern
smoke region is cooler; the largest absolute ON/OFF T2 difference is 1.78 K.
Quality-screened GOES comparisons at 15, 18 and 21 UTC retain the same 18,487
model-cell/scan pairs per arm. AOD RMSE is 2.01294 OFF and 2.01149 ON, with a
large negative bias in both. Smoke-only model AOD and total retrieved AOD
have different composition, and missing retrievals are excluded. The solar
improvement does not establish aerosol forecast skill or justify a new default.

On one shared RTX 4090, engine wall time was 379.88 s OFF and 395.06 s ON
(+4.0%), with compiled kernels already cached. Radiation consumed 11.95 and
19.93 s, or 1.66 and 2.77 ms per model step. Peak process VRAM was 11,060 and
11,144 MiB. These are sequential runs with changing co-resident workloads;
stage spans overlap and are not additive kernel-only measurements.

A separate three-hour comparison of the old and cached optics, with identical
corrected boundary inputs and feedback ON, matched every history word:
233,746,644 typed values in four files. The feedback-OFF comparison also matched
all 233,746,644 values. After excluding each stage's first call, radiation took
453.16 ms per call before caching and 146.47 ms after it; whole model steps
averaged 46.38 and 45.98 ms. Raw engine walls were 183.17 and 103.20 s, but cold
first-step compilation differed substantially, so that wall-time reduction is
not an isolated measure of the optics saving. Container attributes are excluded
from the field identity comparison.

## Solar products and observations

`woof solar catalog` prints the product rows, default PV system rows, and
NOAA observation acquisition/verification rows embedded in `rw_wrfbatch`.
`woof solar products --wrfout FRAME ... --out DIR` computes and renders a
native, chronological, fixed-grid sequence. The Python door calls
`woof.rustwx.run_solar_product_renderer`; Rust reads, computes and renders.
Multi-record history files expand automatically; advanced frame rows can select
`time_index`. Optional `bounds` are `[west,east,south,north]` and affect only the
picture, preserving computation on the complete native grid.
Use `--products solar_ghi,solar_pv_ac_fixed` to select pictures. All products
still receive per-frame statistics in `receipt.json`.

The default rows describe 1 MWdc fixed 30-degree south-facing and horizontal
north/south single-axis tracking systems, with 60-degree limits, backtracking
and GCR 0.35 on level cross-axis terrain. Each row carries DC/AC ratio, inverter
efficiency, temperature coefficient, aggregate loss fraction, ground albedo,
Faiman thermal coefficients, module height and ramp threshold. Supply a JSON
list with `--systems FILE` for other systems; no new model code is needed.
Power units are MW for the row's actual nameplate, not a capacity factor.

GHI uses SWDOWN. Paired SWDDNI/SWDDIF are used when present; otherwise the receipt
names the Erbs hourly decomposition. A half-pair or unreadable present native
field fails instead of silently mixing methods. Radiation history preserves
native direct and diffuse output by default when the producer supplies it.

POA uses isotropic sky and ground reflection. PV power uses PVWatts v5 DC and
inverter equations with a Faiman steady-state module/cell temperature estimate,
module-height wind adjustment and the row's losses. This is not a claim of
full PVWatts simulator equivalence: Fuentes thermal inertia, Perez sky diffuse,
external shading, snow and a separate angle-dependent glass model are not used.
The clear-sky index uses empirical Haurwitz GHI; clear-sky GHI below 20 W/m2 is
missing in the ratio rather than producing an unstable night-time index.

Ramp maps are signed MW/min between native endpoints. Event maps apply the row's
fraction of AC nameplate per minute. The first frame and gaps beyond the
configured maximum are missing. A coarse history cannot resolve intra-interval
ramps. Daily GHI is kWh/m2 and daily AC energy is MWh, with UTC day boundaries.
Trapezoids are split at midnight, gaps poison affected cells, and a day without
all 86400 seconds of coverage receives no daily map and missing daily statistics.

For observations, `woof solar fetch-observations --network surfrad --prefix bon
--directory Bondville_IL --date 2023-03-31 --out OBS` acquires the public daily file
and writes its acquisition receipt. SOLRAD uses `--network solrad` and its archive
station prefix. These rows describe the post-2015 one-minute archive. Both
formats use paired value/QC columns, but their direct/diffuse indices differ.
Only finite QC=0 values above the missing sentinel enter verification. Valid
negative nighttime instrument offsets are preserved. Missing data stay missing. NOAA terms and
acknowledgement are in the acquisition receipt and `licenses/solar-sources.md`.

Advanced requests use `woof solar products --request REQUEST.json`:

```json
{
  "out": "solar-output",
  "frames": [{"label": "model", "kind": "wrfout", "path": "FRAME",
    "fields": {"solar_ghi": "SWDOWN", "solar_dni": "SWDDNI",
      "solar_dhi": "SWDDIF", "temperature_2m": "T2", "wind_speed_10m": "WSPD10"}}],
  "stations": [{"id": "bon", "network": "surfrad", "latitude": 40.05,
    "longitude": -88.37, "files": ["OBS/bon23090.dat"]}],
  "products": ["solar_ghi", "solar_pv_ac_fixed", "solar_daily_energy_fixed"],
  "max_gap_seconds": 3600,
  "domain": "d01"
}
```

Frames can also use the existing native GRIB or WxStore reader and explicit
valid-time metadata. GRIB field selection is canonical and does not branch on
model names. Station matching uses the nearest native model cell after checking
station containment, and an observation mean within the table's bounded time
window. The receipt distinguishes instantaneous model flux from the short
observation mean. `verification.csv` retains matched GHI/DNI/DHI and endpoint GHI
ramps; the receipt gives all-time and daylight error statistics. These compare
irradiance, not measured PV generation.

Pictures follow `DIR/domain/product/UTC-day/time.png`. Receipts name source
hashes, system parameters, methods, dimensions, stage wall times and parallelism.
The CPU/Rayon product route uses no GPU and reports zero GPU VRAM.

The command group is implemented by `woof.solar`. Python callers can use
`woof.cli.main(["solar", "fetch-observations", ...])` for the same acquisition
and receipts, and `woof.rustwx.run_solar_product_renderer(renderer, request_path)`
for a native product request. Decoding, matching, calculations and weather-field
rendering remain in Rust.

## Sources and licences

Engine code is Apache-2.0. Legacy WRF-Chem optical tables and arithmetic are
public domain under the WRF distribution's licence; the existing RRTMGP tables
use the shipped BSD-3-Clause notice. NOAA HRRR/RAP meteorology and smoke, GOES
ABI aerosol optical depth, and SURFRAD/SOLRAD observations are public NOAA data.
Acquisition receipts retain URLs, dates, hashes, quality rules and NOAA
attribution. GOES AOD is total retrieved aerosol; smoke-only model AOD is not
the same composition. Missing or rejected observations remain missing.

Solar equations and their source notices, including the BSD-3-Clause pvlib
reference notice, are listed in [solar sources](../licenses/solar-sources.md).
No new optical equation, third-party code or model weight is introduced by
the smoke-radiation option.
