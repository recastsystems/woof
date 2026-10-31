# Grid and spectral nudging

Nudging is optional. It relaxes the forecast toward the atmospheric fields
from the run's initial-condition source throughout the requested period.
Spectral nudging retains selected large horizontal scales; grid nudging uses
the full horizontal difference. Both run on the GPU and use native prepared
analyses. No external WRF preprocessing executable is needed.

In the five-day, 9 km central-US example, spectral nudging reduced Stage IV
rain RMSE from 20.58 to 14.95 mm and 500 hPa sounding-height RMSE from 7.59
to 5.01 m. Daily temperature and dewpoint extrema became worse. These are
results for one case, not a recommendation to enable nudging for every run.

## Run it

```sh
woof go examples/nudging/regional.toml --nudging spectral --outdir nudged
woof go examples/nudging/regional.toml --nudging off --outdir control
```

The example is 166 by 134 cells, 49 levels, 9 km spacing, a 45-second step,
and 120 hours beginning 2023-07-01 00 UTC. It acquires six-hour ERA5 fields
through the ARCO provider. The normal geography and source-data requirements
still apply. Separate preparation from GPU execution with `--prepare-only`.

For an existing source configuration, add:

```toml
[nudging]
mode = "spectral"
cutoff_x = 1
cutoff_y = 1
relaxation_seconds = 5000.0
minimum_height_m = 1500.0
ramp_depth_m = 1500.0
boundary_cells = 5
```

This regional convention relaxes physical temperature, horizontal wind and
geopotential above the height ramp. Moisture is unchanged unless `gq` is
nonzero. `guv`, `gt`, and `gph`, when supplied, are separate inverse-second
coefficients; otherwise each uses `1 / relaxation_seconds`.

The root is selected by the top-level table. Nests require explicit
`[nudging.domain.2]` tables, using their actual grid IDs. Unselected nests
remain unnudged. Use `mode = "grid"` for grid nudging; it selects the WRF
potential-temperature convention. Grid nudging does not relax geopotential.

## WRF namelists

Import the ordinary domain and physics namelists, including the FDDA section:

```sh
woof import-namelist namelist.wps namelist.input --output imported.toml
```

Bind the desired source using the normal `[fetch]` configuration, then run
`woof go imported.toml`. Source acquisition is not described by WRF namelist
syntax. `grid_fdda = 0, 1, 2` means off, grid, or spectral for each domain.
The importer preserves domain-array selection and WRF Registry defaults.

| Controls | Meaning |
| --- | --- |
| `xwavenum`, `ywavenum` | Largest retained wavenumbers in each horizontal direction |
| `guv`, `gt`, `gph`, `gq` | Independent wind, potential-temperature, geopotential and water-vapor coefficients |
| `if_no_pbl_nudging_uv/t/ph/q` | Suppress the selected tendency in the live model boundary layer |
| `if_zfac_*`, `k_zfac_*`, `dk_zfac_*` | WRF method-specific vertical masks and transition levels |
| `fgdt` | Tendency refresh in minutes, rounded to model steps; tendencies are held between refreshes |
| `fgdtzero` | WRF tendency clearing between regular FDDA calls |
| `gfdda_interval_m` | Required analysis cadence in minutes, available from the source frames |
| `gfdda_end_h` | Last nudging-analysis time in hours after initialization |
| `if_ramping`, `dtramp_min` | End ramp; a positive duration continues after the last analysis, a negative duration ramps down before it |

The namelist convention nudges potential temperature and follows WRF's
grid/spectral masking and end-ramp methods. It does not promise bitwise WRF
identity. `gfdda_inname` and `io_form_gfdda` are recorded as replaced by the
engine's own prepared-analysis store.

Surface and point-observation nudging (`grid_sfdda`, `obs_nudge_opt`) remain
refused because gridded source fields do not supply those observations.
Nonzero alternate begin/end date components are refused because they would
change the requested window; use `gfdda_interval_m` and `gfdda_end_h`.

## Python configuration

The same option is `ExperimentConfig.nudging`:

```python
from dataclasses import replace
from woof.experiment import load_experiment, experiment_config_document
from woof.spectral_nudging import NudgingConfig, NudgingOptions

experiment = load_experiment("regional.toml")
experiment = replace(experiment, nudging=NudgingOptions((
    (experiment.root.grid_id, NudgingConfig(
        enabled=True, cutoff_x=1, cutoff_y=1, boundary_cells=5)),
)))
document = experiment_config_document(experiment)
```

Pass the selected configuration through source preparation before running
`run_prepared_forecast` or `run_prepared_tree`. Enabling the option on an
already prepared OFF bundle cannot create its missing analyses. Reprepare
from the source configuration. The low-level `build_nudger` API is also
available for callers that own GPU field bindings and analysis timing.

## Inputs, restart and limits

Analysis supply uses the source preparation interfaces and source metadata,
including ERA5, GFS and the native/mapped forecast routes. It writes initialized
fields on each selected domain's grid. New source rows use those interfaces;
the nudging operator contains no source-specific decoder.

Analyses live next to the prepared root in `<prepared-root>-nudging/dNN/`.
`series.json`, array manifests and field hashes bind the source identity,
geometry and times. Keep this directory with its prepared bundle. Endpoints
are interpolated on the GPU and refreshed at analysis boundaries. Restarts
retain held tendencies and rebuild the graph against the same sealed source
series and restored clock. Missing or altered analyses are errors.

Selected domains must use a fixed time step and a resident regional grid.
Moving, delayed or spawned domains are refused because their grid changes
invalidate fixed prepared analyses. Tile-local or distributed transforms are
refused because they would change the requested horizontal wavelengths.
Analysis times must coincide with model steps and bracket the nudging window.
An end time inside the run must coincide with a source analysis. A live
as-posted input plan must finish sealing before nudging, because its final
source digest is not yet stable. These restrictions prevent a different
method from silently running under the requested settings.

## Measured result and cost

The example completed 9,600 steps, 121 hourly captures and 19 analysis
refreshes with nudging enabled. The OFF control matched the base forecast's
5,643,267,272 native field and metadata values with zero differing words.
A separate 12-hour, two-domain control matched another 70,067,800 values.
The base required three input-admission repairs to accept this ERA5 bundle;
its numerical forecast implementation was unchanged.

| Independent observation metric | OFF RMSE | ON RMSE |
| --- | ---: | ---: |
| Stage IV daily rain, mm | 20.576 | 14.948 |
| CMORPH daily rain, mm | 19.560 | 14.483 |
| 500 hPa sounding height, m | 7.588 | 5.010 |
| 500 hPa sounding temperature, C | 0.915 | 0.750 |
| Station 10 m wind speed, m/s | 2.324 | 2.238 |
| Station daily maximum temperature, C | 2.762 | 2.960 |
| Station daily minimum temperature, C | 2.023 | 2.100 |
| Station daily maximum dewpoint, C | 2.036 | 2.214 |
| Station daily minimum dewpoint, C | 3.003 | 3.134 |

Stage IV scores cover four complete 12Z-ending daily windows; CMORPH covers
five UTC days. Five additional rain comparison sheets use 00-00 UTC sums of
six-hour Stage IV analyses. Station samples are matched from ISD and IGRA.
The rain sheets show reduced excessive rain in later days, but both arms
retain misplaced rain bands and excessive southeastern rainfall. Better
large-scale agreement does not resolve all convective placement errors.

On one RTX 5090, forecast plus output took 530.072 seconds ON and 417.549
seconds OFF, or 55.216 and 43.495 ms per model step. Other processes shared
the card for 529.3 and 159.6 seconds respectively, so the apparent 27.0%
difference is not an isolated overhead measurement. ON field-stage event
times were u 0.889, v 0.215, temperature 0.194 and geopotential 0.212 ms/step;
the first event also includes scheduling wait. Analysis refresh added
0.078 ms/step amortized; setup took 0.234 seconds.

A separate 300-launch profile of the complete production nudging graph,
including packing and unpacking, measured 0.677 ms median, 0.766 ms mean
and 0.926 ms p95 after discarding 20 warmup launches. Host graph launch
averaged 0.015 ms. This measures the operator, not a repeat whole forecast.
Per-process peak VRAM was 12,388 MiB ON and 12,256 MiB OFF. Preparation and
verification cost are separate from these forecast timings.

Both 24-hour WRF-namelist modes completed. With the last analysis at +18 h
and a one-hour end ramp, retained-band 500 hPa height error at +18 h was
1.418 m for grid and 1.302 m for spectral, against 6.826 m OFF. Errors rose
after nudging stopped. Those curves measure agreement with the driving
analysis, not independent observational skill.

## Sources and licences

- The GPU implementation is distributed under this package's Apache-2.0
  licence. Method references are WRF v4.7.1 `phys/module_fdda_psufddagd.F`,
  `phys/module_fdda_spnudging.F`, `phys/module_fddagd_driver.F` and
  `Registry/Registry.EM_COMMON`. WRF distributes these under its public-domain
  statement; the unchanged notice is included in `spectral_nudging/WRF-NOTICE.txt`.
- ERA5 is Copernicus Climate Change Service information. The ARCO distribution
  retains the upstream Copernicus attribution and data terms. See the
  [source guide](SOURCES.md) for acquisition and licence links.
- NOAA ISD, IGRA, Stage IV and CMORPH observations are US-government data.
  Geography uses the existing source-preparation assets and their notices.
  No observation files or ERA5 data are embedded in the package.
