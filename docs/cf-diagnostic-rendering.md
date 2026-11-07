# CF diagnostic fields

`woof render --cf FILE.nc --variable NAME` reads diagnostic fields directly.
It uses the production `rw_wrfbatch` panel renderer and reports the NetCDF
`source` attribute as provenance. There is no WRF conversion or model-state import.
Classic NetCDF and NetCDF-4, CF UTC time, 1-D latitude/longitude axes and matching
2-D coordinates are supported. Latitude may run in either direction. Longitude
rows and their values are normalized and rotated together to the -180..180 seam.

```sh
woof render --cf gust.nc --variable fg10m --valid 2024-05-22T00:00:00Z \
  --bounds=-125,-66,24,50 --overlays stations.json --out renders \
  --receipt renders/gust-receipt.json

woof render --cf precipitation.nc --variable tp --valid 2024-05-22T00:00:00Z \
  --bounds=-125,-66,24,50 --observation MRMS_QPE06H.grib2.gz --out renders

woof render --cf water-vapor.nc --variable tcwv --valid 2024-05-22T00:00:00Z \
  --bounds=-180,-60,0,65 --contour-file probabilities.nc \
  --contours climnet_ar --overlays tracks.json --out renders
```

The equivalent native door is `rw_wrfbatch --cf ... --out-dir renders`.
`--size 1200x900` controls each panel in the Python CLI; the native door accepts
`--width` and `--height`. `--bounds W,E,S,N` requires ordered longitudes in
-180..180. An omitted extent uses the diagnostic grid's extent, including for
both sides of an observation comparison. Each comparison draws native grids on
the same map extent and ladder and composes the finished PNGs in Rust. It does
not regrid observations to the coarser model or compute verification scores.
Outputs follow `cf-grid/product/valid-day/` under the output directory.

Product identity is data in
`tools/rustwx/crates/rw-wrfbatch/data/cf-products.json`. `--cf-table FILE.json`
loads a replacement table. A row matches an exact variable name first, then
`standard_name`; ambiguous matches fail. A new variable is a new row. Rows carry
the product/title, production recipe, display units, explicit unit factors,
default input units and optional contour threshold and observation product.
A row may carry its own `scale` (`levels`, one RGB `colors` entry per interval,
and optional `mask_below` in display units: values under it are left blank
instead of drawn in the first colour, so a dry precipitation window is a blank map).
Defaults apply only when the file omits units. Explicit unknown units fail rather
than painting numbers against the wrong ladder. Gust uses the production wind
ladder in knots; QPF and precipitable water use their production inch ladders.
Probability rows use the fraction ladder. Packed CF values are masked before
scale/offset and display conversion. Contours can share the input file or use a
companion at the selected UTC time on exactly the same grid.

An observation may be another CF file (`--observation-variable NAME` overrides
the diagnostic variable name). GRIB2 and gzipped GRIB2 use the existing
`comparison-observations.json` rows and decoder, including the MRMS six-hour
MultiSensor QPE Pass2 row. The decoded message identity and valid time must match.

`--overlays` retains the standard geographic overlay JSON fields. It also accepts
CSV sources in the CF door. Paths are relative to the overlay JSON:

```json
{
  "csv_points": [{
    "paths": ["stations-a.csv", "stations-b.csv"],
    "units": "kt",
    "value_columns": ["gust", "peak_wind_gust"],
    "time_column": "valid",
    "start": "2024-05-21T18:00:00Z",
    "end": "2024-05-22T00:00:00Z"
  }],
  "csv_tracks": ["tracks.csv"]
}
```

A point CSV has `station,lat,lon,value` by default. `path` accepts one file;
`paths` merges several before reducing the window to a maximum per station.
Configured value columns are combined by maximum; missing and negative gust
values are omitted. Times are UTC, both window endpoints are inclusive, and no
sustained wind is substituted for a missing gust. Marker fills use the panel's
own ladder and black outlines. Track CSV columns are
`path,step,valid_time,tclat,tclon`; later points are excluded, and missing fixes,
step gaps and date-line jumps split polylines. Other columns are ignored.
