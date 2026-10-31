# Radar-band demonstration

A two-hour forecast produced five TOKC C-band volumes without changing any
forecast storage words. S-band API products and CLI files matched the prior
simulator byte for byte. This is an operator and integration demonstration;
storm placement and radar-intensity errors remain in the forecast.

The example uses public NOAA HRRR initialization and boundaries for
2024-05-07 00 to 02 UTC, a 128 x 128 x 40 grid at 3 km, and 600 steps of
12 seconds. History is saved every 30 minutes. The radar scan has one 0.5-degree
sweep, 360 azimuths, 500 m gates and 150 km maximum range. TOKC is an observed
C-band TDWR, and the comparison uses its public TZ0 low-sweep reflectivity.

## Run the example

From a source checkout with the native bridges installed, supply an existing
geography directory and a fresh work directory:

```sh
python examples/prepare_simulated_radar_bands.py fetch --work radar-demo
python examples/prepare_simulated_radar_bands.py prepare --work radar-demo --geography /path/to/geography
python examples/prepare_simulated_radar_bands.py off --work radar-demo
python examples/prepare_simulated_radar_bands.py on --work radar-demo
python examples/simulated_radar_bands.py radar-demo/off/wrfout --outdir radar-demo/bands --site TOKC --latitude 35.276 --longitude -97.51 --height-m 398.9832 --elevation-deg 0.5
```

Use the exact catalog coordinates and antenna MSL height for an observational
comparison; `--list-sites` is the authority for the installed catalog. The last
command is also a standalone S/C/X replay example for any saved history and
coordinate-defined site. Preparation uses native decoding, regridding and
static-field generation. Python orchestrates the commands. Forecast stages
need a CUDA device; replay, scattering, propagation, encoding and the evidence
maps run on CPU in Rust. Set `RUSTWX_RASTER_DEVICE=cpu` for CPU-only rendering.

The ordinary CLI and `simulate()` API share the same native bridge. There is
no new WRF physics namelist selector: this is a read-only history product,
configured in `[simulated_radar]` or by the replay/API options.

## Measured cost and identity

On one RTX 5090 with a 24-core host, the corrective OFF and C-band ON runs took
38.278 and 48.221 seconds inside the engine, respectively: +9.943 seconds,
or +16.571 ms per forecast step (+25.975%). Process wall times, including
startup and final column output, were 45.986 and 50.848 seconds. The host was
shared, and most of that difference was not radar. The per-step log of the ON
run has three single-step stalls of 1.4, 3.3 and 4.0 seconds (OFF had two, 1.9
seconds in all), and none of them falls while a radar volume is being computed.
Without stalls longer than 0.5 seconds, the forecast steps took 35.09 seconds
ON and 33.51 seconds OFF (+4.7%). The 69 steps that overlapped the roughly
one-second C-band scans averaged 72.7 ms against 57.1 ms for the other ON steps,
about 1.1 seconds in all, because the scan uses the host's cores while the step
loop runs. The final writer drain includes computing the last volume. Radar runs
at history boundaries in its own process through the existing asynchronous
product stream; no radar work was added to the numerical step loop.

| Engine stage | OFF seconds | ON seconds |
| --- | ---: | ---: |
| Restore prepared cache | 0.329 | 0.207 |
| Initialize physics | 0.123 | 0.088 |
| Forecast execution with asynchronous output | 36.723 | 44.886 |
| Final writer drain | 0.529 | 2.478 |
| Canonical final digest | 0.074 | 0.066 |
| Total | 38.278 | 48.221 |

Stage timings overlap and do not sum to the total. The preflight measurement
was 3.160 / 0.539 seconds and is reported separately by the engine.
Five complete history files contain 877 fields, 522,019,567 stored bytes and
130,504,963 words. The OFF/ON comparison found zero differing words, including
dtype, shape and missing-value payloads. The final 226-array state digest was
also identical. An initial pair and the corrective pair had identical history;
the correction restored the native column-output executable, not model physics.

The final five-volume CPU replay, including native images and file publication,
measured 0.371, 1.139 and 1.163 seconds per volume for S, C and X respectively.
The earlier artifact check measured 0.817, 3.126 and 2.699 seconds on the same
host, demonstrating the effect of shared load. These sequential samples are
not a controlled ranking of C versus X. S uses the earlier bulk operator; C/X use
the finite-particle operator, so an S/C comparison includes both wavelength
and operator differences.

## Table and input provenance

`tools/build_radar_scattering_tables.py` generates the tables using
`rustmatrix==2.2.0` (MIT). The solver performs scattering in Rust; the Python
builder orchestrates the solver and writes build artifacts outside forecasting.
The generator checks the electrostatic spheroid limit, polarization convention
and positive forward extinction. The runtime ports the PSD and melting
integrations to Rust and was compared with an independent GPU evaluation at
the unchanged 0.2% tolerance across 30 scheme/band cases.

Each shipped C/X table is 7,258,480 bytes, includes both horizontal and vertical
extinction, and has a pinned SHA-256 checked before use. With 24 native workers,
C took 297.107 seconds and X took 250.922 seconds to build under shared CPU
load. The unused research S table took 88.648 seconds. Tables are shipped;
users do not install the solver or regenerate them to simulate a volume.

| Source | Use and license |
| --- | --- |
| [rustmatrix](https://pypi.org/project/rustmatrix/2.2.0/) | T-matrix solver, MIT; notice shipped beside the tables |
| [NOAA HRRR archive](https://registry.opendata.aws/noaa-hrrr-pds/) | Initialization and boundaries, public US-government data |
| [NOAA NEXRAD archive](https://registry.opendata.aws/noaa-nexrad/) | TDWR Level III reflectivity, public US-government data |
| [NOAA station API](https://api.weather.gov/radar/stations) | Station coordinates and types, public US-government factual data |
| [NOAA TDWR specifications](https://www.weather.gov/gsp/tdwr_specs) | C-band and beam-width metadata, public US-government information |
| [NOAA TDWR list](https://www.weather.gov/media/tg/tdwr-radar-list.pdf) | Antenna elevations, public US-government factual data |

World-reference C-band labels use the published network descriptions from
[FMI](https://en.ilmatieteenlaitos.fi/fmi-radar-network),
[DMI](https://www.dmi.dk/dmis-vejrprodukter/radar/),
[SMHI's 2025 frequency-plan response](https://www.smhi.se/download/18.53cdce23194f389da054cc7/1740738658954/Yttrande%20%C3%B6ver%20remiss%20%E2%80%93%20f%C3%B6rslag%20till%20allm%C3%A4nna%20r%C3%A5d%20om%20den%20svenska%20frekvensplanen.pdf),
the [DWD-network study](https://doi.org/10.5194/amt-18-5177-2025) (CC-BY 4.0),
and [SHMU's network poster](https://www.dwd.de/EN/specialusers/research_education/met_applications_specials/wxrcalmon2025_presentations/poster_downloads/e5.pdf?__blob=publicationFile&v=2).
Only factual band labels are included; no agency imagery, document text or
software is copied. Mixed or unverified networks retain an unknown band.

Equation references: DOI `10.1175/2007MWR2083.1`, `10.1175/2010JAMC2363.1`,
and `10.5194/amt-11-3883-2018`. These publications describe the scientific
assumptions; no publication text or restricted code is redistributed. New
implementation code follows the package's Apache-2.0 license; existing native
renderer, radar writer and BowEcho notices remain in their vendored trees.

Observed TZ0 provides reflectivity, not an independent ZDR or Kdp validation.
At 00:00, 00:30, 01:00, 01:30 and 02:00 UTC, jointly finite reflectivity
biases were +0.407, -4.263, +1.056, +1.955 and -10.061 dBZ. RMSE was
19.37, 18.12, 17.75, 18.88 and 20.80 dBZ. There were no jointly >=20 dBZ
gates in the first two frames, so the small initial mean bias does not show
storm agreement. The matching observed scans were 141, 140, 140, 9 and 9
seconds after the model times; no time interpolation was applied.
The comparison uses nearest azimuth/range gates beyond 10 km, jointly finite
values and the same elevation. It reports conditional intensity errors,
not a detection score; clear-air and coverage misses are not counted as hits.
The Rust evidence renderer uses equal map extents for model/observation and
shared S/C/X color limits within each frame. Expanded dual-pol limits include
all displayed finite values; read each frame's legend when comparing times.

Across all five frames, the modeled northwest line moves northeast, with
scattered cells to its east and later to the southwest. C retains intense
cores with radial ZDR deficits behind them. X loses much of the rearward echo,
leaving a narrow bright leading arc and larger negative-ZDR shadows. Kdp is
larger in the C/X cores than in legacy S. By +2 hours, the observed line is
broader and farther southwest than the simulation; many simulated eastern
cells have little observed counterpart. The maps show these placement and
coverage failures directly. Band-dependent appearance is demonstrated;
observational skill, especially dual-pol skill, is not established.
