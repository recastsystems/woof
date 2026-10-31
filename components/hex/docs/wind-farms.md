# Wind farms on Voronoi cells

`--wind-farms` adds opt-in Fitch momentum drag and turbine-generated TKE
inside MYNN on the GPU. Turbines are assigned to Voronoi cells, and forcing
uses each cell's actual spherical area. Inputs and turbine curves are shared
with woof 3.0.

For an initialized global or zoom globe, add the option to the usual forecast:

```sh
woof hex forecast --mesh MESH_ROW --grid mesh.grid.nc --static mesh.static.nc \
  --init init.nc --init-source "GFS 2023-08-24 00 UTC" \
  --start-time 2023-08-24_00:00:00 --hours 24 --history-every-minutes 60 \
  --wind-farms uswtdb --out forecast
```

`woof hex global` accepts the same option. A custom path replaces `uswtdb`:
`--wind-farms /data/turbines/windturbines.txt`. The file and adjacent
`wind-turbine-N.tbl` curves use WRF's geographic format, in degrees, metres,
m/s and kW. A JSON turbine inventory using `gpuwm-wind-farm.v1` is also
accepted. Inputs outside the United States use the same table and mapping
code. Python callers set `ForecastRequest.wind_farms` to the same value.

USWTDB is fetched and cached on first use. `WOOF_WIND_CACHE` chooses the
cache location. The forecast's commissioning-year filter is not a historical
inventory reconstruction. Fetch inputs before reserving a GPU for a long run.
Rows without usable capacity are excluded and counted; generic class geometry
fills missing hub/rotor dimensions and is marked as imputed in the table.

The MPAS physics route requires `bl_mynn`, with its default every-step PBL
cadence. The WOOF route uses the same forcing with `bl_pbl_physics=5` and
`bldt=0`. Other PBL schemes have no coupled turbine TKE carrier and are
refused. A held PBL tendency would repeat momentum drag without matching
TKE and is also refused. The initial implementation uses resident,
single-rank, complete-globe cell ownership. Culled boundaries and partitioned
cells are refused because the turbine inventory does not yet carry their
outside-domain and rank ownership. The ordinary 96-to-3-km zoom globe has
complete ownership and is supported.

History includes cell power `POWER`, `TURBINE_POWER`, `TURBINE_HUB_WIND`,
`FARM_POWER`, `FARM_HUB_WIND`, the full `WIND_TKE_SOURCE`, and its column
maximum `WIND_TKE_SOURCE_MAX`. Power is W, wind is m/s, and TKE source is
m2/s3. Compact `WIND_FARMS` metadata records table identity and references
`wind-farms.json` by SHA-256; that sidecar records turbine/farm order and
source attribution. Keep the sidecar with the history files.
Farm wind is the arithmetic mean of turbine hub winds. The power diagnostic
does not include curtailment, availability or electrical losses.

## Demonstration

The evaluation configuration uses a globe with 96 km background spacing,
a 3 km core centered at 36.33 N, 97.50 W, GFS initialization at
2023-08-24 00 UTC, and 24 hours of MPAS physics with hourly history.
The vertical grid must place the lowest mass level below every turbine hub.
The accompanying 60-level demonstration starts with a 20.734 m first
interface; a grid that does not resolve a requested hub fails explicitly.
Apply the command above to that prepared mesh and start, and repeat without
`--wind-farms` for a control. Scoring uses the identical contemporaneously
available AWAKEN turbine subset for both runs.

Results for that case, sampled on native cells at the 85 matched AWAKEN
turbines of the King Plains farm, the A1 and H profiling lidars and three
ASOS stations (END, WDG, PNC), over the 25 hourly frames:

| Measure (RMSE) | Control | `--wind-farms` |
| --- | ---: | ---: |
| Observed-turbine farm power, MW | 80.2 | 55.1 |
| Turbine power, MW per turbine | 1.03 | 0.79 |
| SCADA nacelle wind, m/s | 2.68 | 2.44 |
| A1 lidar 90 m wind, m/s | 2.53 | 2.47 |
| H lidar 90 m wind, m/s | 2.66 | 2.34 |
| ASOS 10 m wind END / WDG / PNC, m/s | 1.06 / 1.73 / 1.68 | 1.04 / 1.68 / 1.67 |

The gain is largest overnight under the low-level jet. From 7 to 12 hours
the control holds the observed turbines near rated power (about 230 MW)
while SCADA reports 105 to 145 MW; with turbines the farm-power error over
those hours is 48 MW against 103 MW. Hub winds at the turbines are 1.2 m/s
low with turbines and 0.2 m/s low without. Both runs are too calm for the
first four hours while the GFS start spins up. After 19 hours, evening
storms over central Oklahoma form in different places in the two runs, and
the farm-power error for those hours is 36.6 MW with turbines and 31.6 MW
without. Model power is a reference-density value from the model hub wind
and the turbine curves; it is compared with the SCADA power of the turbines
reporting in each hour.

With `--wind-farms` off, every history and diagnostic field of this case is
bit-identical to woof hex 0.4.10 (10,919,686,650 words, 0 different). On
one RTX 5090 shared with other jobs, the 24-hour forecast took 1,457 s with
turbines and 1,572 s without (median step 310 ms and 326 ms). The card was
shared, so this shows no measurable cost, not a speed-up.

## Sources and licenses

- Shared CUDA implementation and generic turbine curves: Apache-2.0.
- Fitch equations and table format: WRF `phys/module_wind_fitch.F`, public
  domain as distributed by NCAR/UCAR. woof retains the full WRF notice.
- US Wind Turbine Database: USGS, American Clean Power Association and
  Lawrence Berkeley National Laboratory, CC0-1.0, DOI `10.5066/F7TX3DN0`,
  <https://energy.usgs.gov/uswtdb/>. Attribution and source hashes are kept
  with the input. Generic curves are assumptions, not manufacturer data.
- Demonstration initialization: NOAA GFS open government data.
- AWAKEN SCADA and lidar are evaluation-only benchmark measurements:
  <https://awaken-benchmark.readthedocs.io/en/latest/phase3.html>, benchmark
  DOI `10.1088/1742-6596/2767/9/092034`. No blanket SCADA redistribution
  license is stated; raw observations are not bundled or used to fit curves.
