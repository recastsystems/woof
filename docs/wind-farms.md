# Wind farms

Opt-in Fitch turbine forcing removes momentum and adds turbulence inside the
GPU MYNN step. A 24-hour, 3 km Oklahoma forecast starting 24 August 2023 at
00 UTC reduced matched farm-power RMSE from 68.45 to 28.62 MW and hub-wind
RMSE from 1.589 to 1.345 m/s against AWAKEN observations. Afternoon power
remained too low, and the upper rotor-layer shear was still wrong. This is
one case, not a general forecast-skill claim.

## Run

Use an existing forecast configuration with MYNN (`bl_pbl_physics = 5`) and
an every-step PBL (`bldt = 0`):

```sh
woof go forecast.toml --wind-farms uswtdb --outdir farm-forecast
woof go forecast.toml --wind-farms /data/turbines/windturbines.txt --outdir custom-farm
```

The first command fetches and caches the US Wind Turbine Database and decodes
it with `rw-wind-table`. Each domain, including nests, selects its own
interior turbines. The forecast year filters commissioning years; a current
inventory does not reconstruct retirements or within-year commissioning.
Rows without usable capacity are excluded and counted in the input receipt;
missing hub or rotor geometry uses the declared generic class and is marked
as imputed in the turbine table.
`WOOF_WIND_CACHE` chooses the input cache. Custom geographic tables work
worldwide. Empty domains add no forcing.

For TOML or Python, set `wind_farms` on the shared/domain run configuration
or `RunConfig`, respectively. `windfarm_tke_factor` defaults to 0.25.
The rotor must lie inside the vertical grid and its hub must be above the
lowest mass level. Other PBL schemes are refused because their TKE carriers
are not coupled. A held PBL tendency is refused because it would repeat drag
without the matching TKE source. Runs without the option retain their fields.

## WRF input

Set `windfarm_opt = 1` in `&physics`, with `bl_pbl_physics = 5`, and place
`windturbines.txt` beside `namelist.input`. Import the pair normally:

```sh
woof import-namelist namelist.wps namelist.input --output forecast.toml
```

Each turbine row is `latitude longitude type_number`, optionally followed by
a farm identifier. Put `wind-turbine-N.tbl` beside it for each type. The type
file contains the curve row count, then hub height (m), rotor diameter (m),
standing thrust coefficient and rated power (kW), followed by rows of wind
speed (m/s), thrust coefficient and power (kW). The native decoder checks the
table and includes every referenced curve in its identity. `windfarm_opt=2`
remains refused: the MAV momentum and TKE coupling is not implemented.

## History and interpretation

`POWER` is grid-cell power in W. `TURBINE_POWER`, `TURBINE_HUB_WIND`,
`FARM_POWER` and `FARM_HUB_WIND` carry individual and farm diagnostics;
`WIND_TKE_SOURCE` and `WIND_TKE_SOURCE_MAX` carry the TKE source in m2/s3.
The run receipt records turbine and farm order, input identity and
options. Power is an ideal aerodynamic diagnostic, not metered generation:
availability, curtailment and electrical losses are not modeled. Generic
inventory curves are not certified manufacturer curves.

The demo used 6,944 turbines in 1,120 occupied columns and 25 hourly frames.
Eighty-five turbines matched the SCADA inventory, with 80 to 85 reporting
at each time. The same available subset was scored in both arms; three
documented shutdown turbines were excluded independently of the forecast.
Power scores use the same reference-density curve calculation for ON and
OFF. Lidar 90 m RMSE improved from 1.455 to 1.309 m/s at A1 and 1.622 to
1.342 m/s at H. Surface-station changes were small. Overnight wake deficits
were distinct, weakened after daytime mixing, and rotated with the wind.
Late southeast convective differences are not interpreted as turbine wakes.

The final 5,760-step run, with an 8 GiB allocator limit, took 241.75 s ON
and 329.81 s OFF including asynchronous history I/O on an RTX 5090.
The card was shared for 96.9 s ON and 274.4 s OFF, so that difference is
not an isolated kernel-overhead measurement or evidence of a speedup.
OFF matched the pre-feature engine in all 3,007,269,469 checked history
words across all 25 frames. ON matched the run used for the observation
scores and maps in all 3,089,611,319 checked history words.

A separate first-hour stage profile on an RTX 4090 used the same 151 x 151
x 69 case, with an 8 GiB allocator limit. The first ten step samples and
first radiation call were excluded from the warm averages. These CUDA-event
spans are instrumented measurements; nested rows must not be added together.

| Stage | OFF, ms/call | ON, ms/call |
| --- | ---: | ---: |
| Whole step | 37.646 | 37.495 |
| Non-microphysics physics | 12.747 | 12.659 |
| MYNN, including turbine forcing | 8.367 | 8.385 |
| Microphysics | 1.888 | 1.885 |
| Surface layer | 0.555 | 0.536 |
| Noah land surface | 0.100 | 0.099 |
| Radiation, once per 48 steps | 173.806 | 170.787 |
| Fitch kernel call | absent | 0.157 |

The turbine call is about 0.42% of the measured ON step. The small difference
between overall step averages is not evidence that turbine forcing makes
the model faster. Legacy RRTMG workspace sizing honors the allocator limit;
this changes chunk width, not the per-column equations.

## Sources and licenses

- Fitch equations and WRF table layout: WRF `phys/module_wind_fitch.F`,
  public domain as distributed by NCAR/UCAR. The complete notice is retained
  in `tools/wind_fitch_oracle/NOTICE.txt`.
- USWTDB: USGS, American Clean Power Association and Lawrence Berkeley
  National Laboratory, CC0-1.0, DOI `10.5066/F7TX3DN0`;
  <https://energy.usgs.gov/uswtdb/>. Attribution and source digest travel
  with the decoded input.
- Generic turbine curves and GPU implementation: original Apache-2.0 code.
- Demo forcing: NOAA HRRR open government data. Verification: AWAKEN SCADA
  and profiling lidar from the public model-comparison benchmark
  (<https://awaken-benchmark.readthedocs.io/en/latest/phase3.html>, benchmark
  DOI `10.1088/1742-6596/2767/9/092034`). The benchmark does not state a blanket
  SCADA redistribution license; raw observations are never bundled. NOAA
  surface and sounding observations are distributed by IEM. No observation-
  derived curve fitting was used.

See `examples/wind-farms/README.md` for the demo configuration and commands.
