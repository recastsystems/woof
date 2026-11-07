# Soil-texture downscaling of the initial soil state

*A deliberate divergence from WRF. Default-on since 2.3.4.*

## The symptom

Over land, a gpuwm 2 m dewpoint plot used to show a faint rectangular quilt.
The rectangles are 0.25 degrees on a side: 16.8 km by 27.6 km at 53 N,
which is 5.5 by 9.1 cells on a 3 km domain and 16.5 by 27.6 on a 1 km one.
Their edges sit on whole multiples of 0.25 degrees of latitude and
longitude, not on the model grid: rotate the projection and the rectangles
do not rotate with it. They are strongest in the first few forecast hours,
peak in the afternoon, and fade after sunset. Water is clean.

That is the forcing model's own mesh, printed into the forecast.

## Where it comes from

A forcing model delivers its soil state on its own grid: 0.25 degrees for
both GFS and ERA5. The horizontal stage carries it to the model grid with
WPS's `sixteen_pt` overlapping-parabolic interpolant, a faithful
transcription of `geogrid/src/interp_module.F`. That interpolation is not
the bug. It is exactly what WPS metgrid does, byte for byte, and it was
verified as such: the measured interpolant on a real run is `sixteen_pt`,
and `woof/ingest/horiz.py` is unchanged across releases.

What an interpolant cannot do is invent information below the source
spacing. Fit a 16-term bicubic *inside each 0.25 degree source cell* on a
real 3 km European run and it explains

| field | R² inside one source cell |
|---|---|
| `T2` | 1.00000 |
| `TSLB` layer 4 | 0.99978 |
| `SMOIS` layer 4 | 0.99972 |
| `Q2` | 0.99988 |
| 2 m dewpoint | 0.99650 |
| `HGT` (control) | 0.90691 |
| `LU_INDEX` (control) | 0.51153 |

The first five hold *no* sub-source-cell information at all. The two
controls do, which is what makes the measurement a measurement rather than
a statement that smooth fields are smooth.

`T2`, `Q2`, `TSK`, `U10`, `V10` and the 2 m dewpoint carry it only at
initialisation and relax within an hour. `SMOIS` (all four layers), `SH2O`
and deep `TSLB` carry it **permanently**, because soil moisture is a
prognostic reservoir with no mechanism that erases a scale it was
initialised with.

And it reaches the screen. Soil moisture controls the latent heat flux;
the latent heat flux controls 2 m humidity. On the reference run the
block-scale correlation between 2 m dewpoint and soil moisture is +0.41
through the first five forecast hours, tracking the latent-heat-flux
correlation (+0.415) and collapsing after sunset: the fingerprint of an
evaporative pathway rather than a plotting artifact. About 18 % of the
daytime block-scale dewpoint variance over land is soil-moisture driven.

## What WRF does

`real.exe` consumes metgrid's interpolated `SM`/`ST` layers unchanged.

* It adjusts soil **temperature** for terrain: `adjust_soil_temp_new`,
  `share/module_soil_pre.F:993-1073`.
* It floors pathological soil **moisture** at a constant 0.005:
  `account_for_zero_soil_moisture`, `dyn_em/module_initialize_real.F:3363-3395`.
* At no point does it consult the target grid's own soil texture, and it
  never rescales moisture.

So the soil texture WRF hands Noah (`ISLTYP`, from geogrid's 30 arc-second
STATSGO/FAO dataset) and the soil moisture WRF hands Noah are inconsistent
with each other by construction. A cell whose 30 arc-second texture is sand
receives the volumetric water content of a 0.25 degree cell that was mostly
clay, and for the same physical wetness those are different numbers, because
sand saturates at 0.339 m³/m³ and clay at 0.468.

**Stock WRF quilts soil moisture exactly like this.** This is not a woof
regression; it was reproduced on 2.0.0, 2.2.1 and 2.3.3, and
`woof/ingest/horiz.py` is byte-identical between them.

## What woof does instead

Volumetric water content is not the quantity the two grids share. The
dimensionless wetness of the soil is. So moisture crosses the resolution
change as Noah's own degree of saturation and is reconstituted against the
target grid's texture:

```
SRATIO  = (SMC - SMCDRY_cell) / (SMCMAX_cell - SMCDRY_cell)
SMC_new = SMCDRY_target + SRATIO * (SMCMAX_target - SMCDRY_target)
```

* `SMCDRY` is `SOILPARM.TBL` `DRYSMC` and `SMCMAX` is `MAXSMC`: the same
  air-dry and saturation constants Noah itself uses, read from the same
  table the physics driver reads.
* `SRATIO` is literally Noah's direct-evaporation variable
  (`phys/module_sf_noahlsm.F:1214`). Nothing here is invented.
* `SMCDRY_target` / `SMCMAX_target` come from the target grid's own
  `ISLTYP`, i.e. the 30 arc-second soil database at model resolution.
* `SMCDRY_cell` / `SMCMAX_cell` are the **area mean of those same target
  parameters over one source cell**: what a 0.25 degree cell can
  represent of the same database. GFS does not transmit its soil type
  (there is no `SOTYP` in the decoded inventory), so the coarse-grained
  target texture is the best available description of the source cell, and
  it has the decisive property below.

`SRATIO` is clipped to `[0, 1]`, so the result is bounded by
`[SMCDRY_target, SMCMAX_target]` cell by cell, by construction. Water,
sea ice, and land cells whose soil category is water are untouched.

### Why this is better and not merely different

1. **It is the only operation that adds real sub-source-cell structure.**
   Smoothing the interpolated field (WPS `smooth_option`) removes the
   curvature break and the visible facet edges but leaves the block-scale
   amplitude intact, because it adds no information. That is cosmetic, and
   woof deliberately does not ship it as the remedy.
2. **It makes moisture and texture mutually consistent**, which is what
   Noah's own hydrology assumes. Handing Noah a wetness its texture cannot
   support is how a sandy cell starts the forecast at an impossible
   saturation.
3. **It conserves water.** The reconstitution is affine in `SRATIO` and the
   coarse texture is the cell *mean* of the fine texture, so the
   source-cell mean of the result equals the source value it came from,
   exactly, wherever `SRATIO` is uniform in the cell. Measured on the
   reference European run, domain soil water moves by +0.034 %.

## The deep-soil-temperature analogue

`TSLB` carries the same imprint, and for a layer-form source it is worse
than for a level-form one. GFS's four slabs coincide with Noah's four
layers, so the layer-form mapping copies them straight through and the
target grid's own deep-soil climatology (`TMN`, geogrid `SOILTEMP` adjusted
to the model terrain) never enters the column at all. WRF's own level-form
path does not have that hole: `init_soil_2_real`
(`share/module_soil_pre.F:1591-1608`) brackets the profile with `TSK` at
0 m and `TMN` at 3 m, so every layer is anchored on the target's deep
temperature in proportion to its depth.

woof applies exactly that anchoring, and only to the part of `TMN` the
source mesh cannot represent:

```
TSLB_k += (midpoint_k / 3 m) * (TMN - cellmean(TMN))
```

Subtracting the source-cell mean is what makes it a downscaling rather than
a nudge: the source keeps every scale it resolves and contributes nothing at
the scales it does not, so **each layer's source-cell mean is unchanged,
exactly**. The weights are 0.017, 0.083, 0.233 and 0.500 for Noah's four
layer midpoints, so this is a deep-layer correction by construction: layer
1 barely moves and layer 4 takes half the anomaly.

## What this does NOT change, measured

Two arms of the same build on the reference European run, one declaration
apart, measured with the instruments above:

* **The soil state is decisively fixed and stays fixed.** `SMOIS` layer 1
  reads 0.98394 / 0.89040 (before / after) at hour 0 and 0.97814 / 0.88913
  at hour 12. Soil moisture is prognostic: what it starts with is what it
  keeps.
* **The 2 m dewpoint's block-scale AMPLITUDE barely moves**: 2.097 K to
  2.078 K at forecast hour 5, a 0.9 % change, and its block-scale
  correlation with soil moisture is +0.424 before and +0.416 after. That
  is the expected result and not a disappointment. The fix does not remove
  soil-driven variance from the dewpoint -- soil moisture really does drive
  2 m humidity -- it changes what that variance is organised BY: the target
  grid's own soil texture instead of the forcing mesh.
* **The dewpoint's own source-mesh signature was already at the null from
  hour 1, in both arms.** Its curvature concentration reads 0.002-0.008
  against terrain's 0.004 from F001 onward. The dewpoint carries the mesh
  strongly only at F000.
* **At F000 the two arms' 2 m dewpoint is byte-identical**, including the
  rendered image. The initial 2 m dewpoint is the interpolated forcing's
  own `Q2`/`PSFC`, produced before any soil coupling, so a soil fix cannot
  reach it. If a quilt is visible in an analysis-time dewpoint plot, that
  is the interpolated near-surface fields themselves and it is a separate
  problem with a separate remedy -- there is no fine-scale "true" 2 m
  humidity field to reconstitute against the way there is a soil map.

Anyone claiming this fix removes visible boxes from a dewpoint plot should
show which hour they mean and measure it, because on this run the accurate
answer is "hour 0 is unchanged and later hours were already at the null".
What it unambiguously removes is the forcing mesh from the soil state,
where it was large, permanent, and physically inconsistent with the soil
texture the same model integrates.

## Reproducing stock WRF

Declare, in the experiment TOML:

```toml
[ingest]
soil_texture_downscale = false
```

That restores the byte-exact previous behaviour: both halves return their
input arrays untouched, and the run receipt records that the switch was
declared. It is the **only** value that turns anything off: silence means
on, because a bare default run must stop showing the defect.

`[ingest]` is a table of its own rather than a `[case_data]` key because
`[case_data]` selects the ERA5 config-driven route and declares that
route's inputs. A `woof go` config carries no `[case_data]` at all, and
that is the door most users come through. The table is validated by
`woof.ingest.soil_downscale.parse_ingest_table`: an unknown key or a
non-boolean value refuses at config load rather than silently running the
default under the name of your setting.

## The run receipt

Every preparation records the soil-state source resolution, whether or not
anything was downscaled, under `soil_texture_downscale` in `proof.json`:

```json
"soil_texture_downscale": {
  "source_spacing_deg":       {"lat": 0.25, "lon": 0.25},
  "target_spacing_deg":       {"lat": 0.0274, "lon": 0.0455},
  "source_cell_in_target_cells": {"x": 5.4952, "y": 9.1234},
  "resolution_ratio": 5.4952,
  "advisory": true,
  "downscale_enabled": true,
  "applied": true,
  "land_cells": 106183,
  "land_without_texture": 2984,
  "downscaled_cells": 103199,
  "soil_water_kg_m2": {"before": 363.556, "after": 363.681},
  "soil_water_change_pct": 0.0342,
  "sratio_clipped_dry": 10639,
  "sratio_clipped_saturated": 0,
  "sratio_samples": 412796,
  "fields": {"SMOIS_L1": {...}, ...},
  "deep_soil_temperature": {...},
  "wrf_reference": {...}
}
```

Those are the real values from the reference European run.
`land_without_texture` counts land cells whose dominant soil category is
water -- geogrid's landmask and its soil category are independent fields
and disagree along every coastline -- which have no air-dry value to build
a ratio from and are left exactly as they arrived.  `sratio_clipped_dry`
counts layer-cells whose source moisture sat below the source cell's own
air-dry value; they land at the target texture's `SMCDRY`, which is as dry
as that soil can physically be.

"How coarse was the soil I started from" is a question a reader of the
output must be able to answer without the config, so this receipt is
unconditional, unlike `soil_moisture_floor` and `deep_soil_repair`, which
appear only when they fire.

When the model resolves more than five grid cells across the narrow side of
one source cell, preparation also prints an advisory naming the forcing
mesh and its width in model cells. Five is where the quilt first becomes
visible in a rendered 2 m field.

## Scope

| path | covered |
|---|---|
| `woof go` / GFS direct | yes |
| ERA5 direct | yes |
| mapped (rw-wps) sources | yes |
| `prepare_real_case` / `woof run` | yes |
| nested children (`woof.ingest.nest_init`) | yes, the plan rides `PreparedChildInput`, and the declaration rides `InputCatalog`, so a child cannot seam against its parent |
| tiles preparers (`tilestream.realcase`) | yes |
| RUC land surface (`sf_surface_physics = 3`) | yes, applied to the source profiles before the vertical remap, against `SOILPARM.TBL`'s `STAS-RUC` block |
| native HRRR | no source mesh is declared: HRRR's 3 km grid is not coarser than a typical target, and the route announces that it declared none |
| `woof downscale` offline child | inherits whatever the `--child-surface-from` file was built with; woof does not build that file |

## Measuring it yourself

Two instruments, both in `tests/test_soil_downscale.py`:

* `bicubic_r2_within_source_cell`: fit 16 polynomial terms inside each
  source cell. R² at 1.0 means the field holds nothing below the source
  spacing.
* `phase_concentration`: bin |curvature| by phase within a source cell.
  A field whose curvature lives on the cell boundaries reads far above the
  smooth-field null (terrain 0.108, land use 0.109).

`test_instrument_detects_the_defect_it_is_meant_to_detect` runs both
directions on the same fixture: the instrument must trip on the unfixed
field and stay quiet on one that genuinely carries sub-cell structure.

## Soil temperature a source analysed wrong

Two rules rebuild a land column's soil temperature linear in depth from
the skin temperature at 0 m to the deep soil temperature at 3 m, keeping
its soil moisture (`woof/ingest/soil.py`, on Noah's layers and on RUC's
levels alike, on every route):

* **real.exe's band.** Any source sample outside 170 to 400 K
  (`unreasonable_land_soil_columns`, following
  `dyn_em/module_initialize_real.F:3521-3596` with its deep-temperature
  sign corrected).
* **Under snow.** A snow-covered land column (at least 10 kg m-2 of snow
  water, the threshold real.exe itself sets `SNOWC` at) whose top source
  sample is more than 30 K below its skin temperature
  (`snow_soil_below_skin_columns`, `SNOW_SOIL_SKIN_DEFICIT_K`). A snowpack
  insulates the ground: in the cold season heat flows up out of the soil
  through the snow, so the soil top is warmer than the snow surface and
  falls below it only while a warming surface has not yet reached it.
  The band lets through what HRRRv2 carried under western snowpack in
  2017, top soils of 170 to 243 K under a skin near 268 K. This rule
  goes beyond real.exe. It is the default because on the one case
  tested it reduced 2 m temperature error against observations:
  on 2017-01-19 15Z over Idaho (native `--source hrrr`,
  150 x 120 cells at 3 km, 3 h) the 2 m temperature at 13 ASOS stations
  on snow-covered land had a mean absolute error of 3.76 K and a bias of
  -2.51 K over hours 1 to 3 without it, and 1.65 K and -0.15 K with it.
  That is a single-case validation result, not a general skill measurement.
  A healthy HRRRv4 analysis over the same ground (2024-01-19 15Z) never
  puts its top soil more than 10 K below the skin under snow, so the rule
  does not touch it.

The preparation prints one line per rule with the count, the range and
where, and the proof's `soil_temperature_repair` receipt records every
rebuilt column (`repaired_land_columns`, `bounding_box`) and each rule's
own count and box (`outside_band`, `snow_top_soil_below_skin`). A
preparation that rebuilt nothing carries no receipt.
