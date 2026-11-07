# woof with woof hex

woof hex is the global model of this family: a GPU port of the
MPAS-Atmosphere v8.4.2 dynamical core on unstructured, variable-resolution
meshes. It owns no physics of its own. Every physics column runs through
woof, and its initial conditions, product maps, soundings and scores are
drawn by woof's Rust tools. So a woof hex release is always paired with a
woof release.

The woof hex release on PyPI is 0.3.3. Its wheel declares `gpuwm>=2.8.0,<2.8.1`, so
`pip install woof hex` in this environment replaces this woof with 2.8.0 and it ships no
runner for `woof preset`. Keep woof hex 0.3.3 in its own environment.

`woof preset list` and `woof preset check` count the installed woof hex: a runner that will
not import, or a release outside the preset's declared range, is a blocker with the install
command beside it, and `woof preset prepare` and `woof preset run` refuse with the same words
before any card is touched.

## What woof provides to woof hex

- **Physics seam.** `woof.core.mpas_column_suite.run_mpas_column_physics`
  runs any registered scheme selection on MPAS columns.
- **Variable-resolution convection.** New Tiedtke uses each cell's own grid
  spacing, so it can run across a mesh whose cells range from 60 to 3 km.
- **Physics fixes hex relied on.** Kain-Fritsch accumulated convective rain no
  longer steps backwards; RRTMGP keeps positive layers above the model top
  (hex meshes produced non-finite longwave heating there).
- **Initial conditions.** `met_intermediate` reads DWD ICON global through its
  table row, and is in the bridge bundle (`woof fetch-bridges` installs it).
  `rw_mpas_init` computes its momentum columns in parallel, with unchanged
  bytes.
- **Products.** `rw_mpas_convert --remap product` (interpolated state,
  area-conservative accumulated precipitation, nearest categories, with a
  precipitation budget line per frame) and `--jobs N`, and the `global-mesh` and `region:NAME` render
  windows with the `GPUWM_GRID_LABEL` source label that hex's product views
  ask for.
- **Soundings and meteograms.** `rw_hex_points` draws them at named points.
  It ships in the platform wheels and the bridge bundle (`woof fetch-bridges`
  installs it), and `woof doctor` checks it. To use another build, point the
  hex door at it with `$WOOF_HEX_RW_HEX_POINTS`.

## What does not work together yet

- **WOOF point runs on hex** (a regional point forecast handing off to a hex
  fine core) are refused by name: the projected-intermediate producer they need
  is not qualified.
- A suite (`woof suite`) row that runs hex plans and validates; running it
  unattended needs the same `--bindings` document as any other row.

## Global composed sources through rw-wps

`woof source-intermediate` composes a packaged source into the canonical
rw-wps state, then exports one global frame to WPS intermediate v5 in Rust.
It reads the same packaged profile and composition used by `woof prep`.
For an `earth2-<key>` source the model table supplies the profile; name
`--profile` when the fetch receipt selects a different donor composition.
No Earth-2 adapter is needed in hex.

```bash
woof source-intermediate --source earth2-fcn3 \
  --input fetched/earth2-fcn3.nc \
  --supplement physical_analysis_surface_data=fetched/analysis/era5-combined.nc \
  --valid-time 2024-09-24T12:00:00 --work-dir intermediate-work --out composed.met
```

The input must cover one complete longitude period without a duplicate seam
and the global latitude grid, with earth-relative winds and pressure levels.
A cropped forecast is refused. The composition closes the donor fields and
soil layer depths. `wps-intermediate-fields.v1.json` defines the source-neutral
canonical-to-WPS names and units. Land fraction is classified at 0.5 for the
binary WPS interpolation mask; permitted missing masks become WPS's -1e30.
The export receipt binds the frame stream, composition, field contract and
WPS bytes by SHA-256, and reports `nfglevels` and `nfgsoillevels`.

Pass the resulting file to the existing `woof hex init --met composed.met`
with those two counts, `--use-spechumd yes`, a global grid and static, and the
usual explicit vertical/physics switches. Also pass `--sfc-met composed.met`
to consume the composed SST and sea ice through the existing surface door.
The init door and forecast door remain the standard hex paths.
