# Terrain grid preparation

Original WOOF-ArWen code, Apache-2.0. Raster decoding and projection reuse the
existing MIT `static-fields` crate; its notice remains in `tools/rustwx/LICENSE`.
No UCAR code, source, fixtures or ABI are used.

The terrain unit owns this DEM/static-grid preparation and its source table.
The hydrofabric owner supplies case bounds, CRS, divides, network and catchment
mapping. Products consumes the same sealed elevation/grid for HAND; it does
not need a second DEM download or provider. Keep the source raster digest,
table and request digests, grid identity, attribution and data licence with
each prepared case. The service URL is mutable; the retained raster SHA256
is the reproducible source pin. Confirm the vertical datum before binding
receiving stages or comparing water elevations.

`rw_terrain REQUEST.json` prepares startup-only terrain fields with Rayon. A
source is a JSON table row, including its URL, query parameters, units and data
licence. New GeoTIFF providers require rows rather than provider adapters.
The initial supported target projection is EPSG:5070, with square cells.

```json
{
  "source_table": "/checkout/woof/data/nextgen/terrain-sources.v1.json",
  "source": "usgs-3dep",
  "output": "/data/scratch/lowercolorado-dem",
  "grid": {
    "nx": 737, "ny": 582,
    "x0": -730254.9996999999, "y0": 612815.0003,
    "dx": 1000.0, "epsg": 5070
  }
}
```

Build with `CARGO_HOME`, `CARGO_TARGET_DIR` and `TMPDIR` directed to the lane's
data-disk scratch directory. Download uses the system `curl` executable with a
90-second request limit. Network transfer is startup work; field decoding,
coordinates, topology, encoding and hashes run in Rust. No Python field work.

The tool refuses an existing output directory. The downloaded GeoTIFF must
declare the requested shape, CRS and north-first affine transform; a missing
or different georeference is an error. The manifest seals source bytes, table,
request and each little-endian C-order output. Dynamic-service downloads are
identified by their receipt SHA256, not claimed to be immutable upstream URLs.
For exact offline reuse, set `input` to the retained GeoTIFF and
`expected_sha256` to its receipt hash. A hash mismatch is an error.

Output fields are `elevation_m`, `latitude`, `longitude` (f32), `active` and
`downstream_cell` (i32). Rows run north to south; x increases east. `x0,y0`
are the southwest outer corner. Finite DEM pixels have active=1; this is a
coverage mask and must be intersected with hydrofabric catchments before an
actual basin run. Elevation nodata is retained as NaN. No terrain is invented.

Topographic D8 takes the largest strictly positive downslope gradient, using
cardinal/diagonal distance. Ties use W,E,N,S,NW,NE,SW,SE order. Indices are
zero-based flat cell indices, with -1 for pits, inactive cells or no lower
neighbor. This is an unconditioned diagnostic drainage map: pits are retained,
there is no fill/breach, and it does not assign channel nodes. The terrain
solver independently evaluates dynamic water-head gradients on the GPU.
Catchment/channel mapping and source hydraulic parameters are separate inputs;
the tool does not invent calibrated Manning, soil or aquifer fields.

Every manifest reports fetch/hash, decode, preparation and write/hash wall times
and the actual Rayon thread count. The tool has no per-routing-step role.
