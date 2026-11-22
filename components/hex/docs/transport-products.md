# Hex moisture transport and wind products

`woof hex products RUN_DIR` now includes `ivt_column`, `ivt_vectors`,
`isotherm_height_0c`, and both `250mb_height_winds` and `300mb_height_winds`.
Regional views also include `925mb_height_winds` and `850mb_height_winds`.
Each history time is rendered, so the jet workflow is the two pressure-level
sequences, with height contours, wind-speed fill in knots and earth-relative
wind barbs. Use `--region NAME=region:LAT,LON,WIDTH_KM,HEIGHT_KM` for any region.
The 925 hPa view diagnoses winds only; it does not establish dust concentration.

All fields pass through `rw_mpas_convert` and `rw_wrfbatch`. The companion engine
change is required for IVT and 925 hPa product admission. An older renderer names
unavailable products in the standard-set manifest instead of inventing a field.
For a strict request for just these maps, use the render door:

```sh
woof hex render --history RUN/history.*.nc --mesh GRID.nc --init INIT.nc \
  --simulation-start START \
  --window region:35,-130,6000,4000 --products ivt_column,ivt_vectors,isotherm_height_0c,925mb_height_winds,850mb_height_winds,250mb_height_winds,300mb_height_winds \
  --series --out PICTURES
```

START is the run's start, `YYYY-MM-DD_HH:MM:SS`. Without it every frame is
filed as lead 0, so the frames' pictures replace one another and one frame's
set is kept. A forecast on the default MPAS route writes
`history.*.nc`; on the WOOF-ArWen route (`--physics-suite arwen_mpas_column`) the
frames are `RUN/cuda-history.*.nc`.

IVT is the pressure integral of specific humidity times the earth-relative
wind vector, divided by standard gravity (9.80665 m s-2). The renderer converts
the stored dry-air mixing ratio `r` to specific humidity `r/(1+r)`. Integration
uses native model levels after horizontal conversion, with pressure midpoint
interfaces. The bottom reaches the actual surface pressure; the top reaches
an extrapolated half-level, clipped at zero pressure. End values are held
constant to those interfaces. The title explicitly says model column: no
transport above that model top is inferred, and no fixed 1000-300 hPa
truncation is implied. Components are integrated before their vector norm.
Invalid, missing, negative-humidity or nonmonotone columns remain missing.
The scale is fixed in kg m-1 s-1, with an over-range marker above 2000.
IVT arrows point toward transport and have a fixed length for direction;
the filled magnitude carries strength. They have no knot pennants.

`isotherm_height_0c` is the existing Rust lowest 273.15 K crossing in metres
above mean sea level, interpolated between model levels. It is missing when
no crossing is bracketed. It is not a wet-bulb-zero or snow-level diagnostic.
Pressure-level wind maps remain missing below terrain and outside the model
pressure range. Supply the actual init containing `zgrid` for height products.

For AR verification, inspect the complete sequence of IVT magnitude/direction,
freezing level and rainfall beside a time-matched independent reference on the
same scale. A PW map alone does not verify transport. Product implementation
and a rendered hindcast do not by themselves qualify forecast skill.
