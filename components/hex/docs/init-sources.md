# Init sources: one table, one path

`woof hex init-source` builds an initial condition from any row of
`src/hexcore/data/init-sources.v1.json` or `src/hexcore/earth2_init_rows.json`.
Every row travels the same four
stages; no line of `src/hexcore/init_sources.py` names a source, and
`tests/test_init_sources.py::test_the_path_names_no_source` holds that.

```
acquire  ->  [remap]  ->  decode (met_intermediate)  ->  init door  ->  [order + case JSON]
```

```sh
python -m woof.hex.cli init-source --list
python -m woof.hex.cli init-source --source ifs-ens --member 1 --cycle 2026-10-03T12 \
  --out-dir OUT --grid GRID.nc --static STATIC.nc --vertical-spec SPEC.json \
  [--order-map ordering.nc --hex-order hex_order --case-template case.json] \
  [--staged DIR] [--writer met_intermediate] [--remapper gdt101_remap]
```

`--writer` and `--remapper` are optional. Without them the door finds both
binaries the way it finds every engine: its own variables
(`WOOF_HEX_MET_INTERMEDIATE`, `WOOF_HEX_GDT101_REMAP`), woof's
(`WOOF_MET_INTERMEDIATE`, `WOOF_GDT101_REMAP`), the `libexec/bridges`
directory the woof wheel installs them in, `~/.woof/bridges`, then `PATH`.
`woof hex doctor` reports `met_intermediate`. The writer, the remapper and the
Vtable are all resolved before anything is downloaded.

The receipt `OUT/init-source-receipt.json` records the table digest, the row,
every product's address, byte count and SHA-256, the writer argv and digests,
and the init argv. The init door's own provenance receipt sits beside the init.

## What a row declares

| Key | Meaning |
|---|---|
| `products` | Files to acquire. `url` is a template or a list of mirror templates (`{yyyymmdd}`, `{hh}`, `{member}`, `{member_tag}`, set keys). `index` selects records of a large file by its JSON-lines index (`select` is a key/value map, or `"member"`). `transport: cds` is a Copernicus request. `codec: bz2` unwraps on arrival. `expand` repeats the product over one of the row's `sets`. |
| `members` | Control and perturbed member addressing: a filename tag, or index keys. |
| `sets` | Named lists of template variables (pressure levels, soil nodes, soil layers). |
| `remap` | For a source on an unstructured grid: mesh size, centre, the regular target, and which products carry latitude, longitude and land fraction. Each product then names its record selector and remap method. |
| `decode` | The Vtable (under the package's `data/vtables/`, so every wheel and sdist carries it) and the `met_intermediate` map-source label. |
| `first_guess` | `nfglevels`, `nfgsoillevels`, `use_spechumd`: facts of the source's ladder. |
| `vertical`, `surface`, `grid`, `limits`, `license` | What the source publishes, in words a reader can check. |
| `status: refused` + `refusal` | A source whose bytes cannot travel the path, with the breakage named. |
| `surface_input` | A companion analysis the init reads as MPAS's surface input (`--sfc-met`): the products of `product_role`, decoded with their own `vtable` and `map_source` into a second WPS intermediate whose SEAICE is interpolated over water with that file's own LANDSEA. Never part of the primary decode (`ifs`, `ifs-0p1`, `ifs-ens`, `aifs`, `aifs-ens`: the same cycle's GFS ICEC, because ECMWF open data publishes sea-ice thickness only, and AIFS no sea-ice field). |
| `cycles` | The publication's own cycle window: `first` (the earliest cycle posted) and `refused` windows `[from, until)` of posted cycles that cannot make an init, each with its sentence. Checked from the table alone, before the mesh, the static or any download (`ifs-0p1`: the first cycle, 2026-10-20 06Z, has no soil). |

`run_switches` at the top of the table holds the init switches that belong to
the run and the mesh. They are the same for every row, so two sources differ
only in their meteorology.

## Adding a source

1. Inventory real bytes (`grib2_inventory`) and write the rows a Vtable needs.
   Reuse an existing Vtable when the centre's codes match.
2. Add the row. If the centre is new to `met_intermediate`, its closed
   map-source vocabulary needs a row there (the rules come from WPS `rrpr.F`).
3. Run `init-source`, then `initial-source-skill audit` on the init.

No Python is added. If a source needs a transform the path does not have, the
transform is added once as a declared stage (as `remap` was), never as a
branch on the source's name.

## Rows today

| Row | Products | Vtable / map source | First guess |
|---|---|---|---|
| `gfs` | one pgrb2 0.25-degree file | `Vtable.GFS` / `ncep-gfs` | 34 levels, 4 soil |
| `ifs` | one open-data oper file, plus the same cycle's GFS ICEC + LAND (surface input) | `Vtable.ECMWF-OD.rw` / `ecmwf` | 15 levels, 4 soil |
| `aifs` | one open-data AIFS file, plus the same cycle's GFS ICEC + LAND (surface input) | `Vtable.ECMWF-OD.rw` / `ecmwf` | 15 levels, 2 soil, specific humidity |
| `ifs-ens` | one member's records by index, plus the oper terrain record and the same cycle's GFS ICEC + LAND (surface input) | `Vtable.ECMWF-OD.rw` / `ecmwf` | 15 levels, 4 soil |
| `gefs` | one member's pgrb2a + pgrb2b pair | `Vtable.GFS` / `ncep-gefs` | 32 levels, 4 soil |
| `era5` | CDS pressure-level + single-level GRIB1 | `Vtable.ERA5.rw` / `ecmwf` | 38 levels, 4 soil |
| `icon` | 120 bz2 records on the native icosahedral mesh, remapped | `Vtable.ICON.rw` / `dwd` | 19 levels, 8 soil |

The `icon` row needs the engine at or after the commit that gives
`gdt101_remap` a whole-sphere target and `met_intermediate` the `dwd` map
source and the soil-water-mass rule. Every other row runs on the unchanged
writer.

## Generated Earth-2 forecasts

The Earth-2 extension lists each model's output channels, pressure levels,
units, transforms, explicit exclusions, source licences and pinned weights
revision. There are no model adapter modules. Generation uses the engine's
shared Earth2Studio runner. Its existing Rust composition profiles read the
forecast and physical surface/soil donor, then export WPS intermediate.
The export's actual atmospheric and soil level counts drive hex initialization.
The installed model contract and Rust mapping are checked before inference.

```sh
woof hex global --row global-120km --source earth2-fcn3 \
  --cycle 2024-10-08T06 --hours 24 --work-dir earth2-global
woof hex s2s --row global-120km --source earth2-fcn3 \
  --cycle 2024-10-08T06 --days 30 --members 2 --lagged 0,6 \
  --work-dir earth2-s2s
```

`--cycle` names the hex initialization time. The model starts one forecast
step earlier, so even source lead zero consumes a positive-lead forecast.
For a lagged S2S member, the requested source lead is added to that step;
all members retain the common hex valid time. The inference receipt records
both clocks. Surface and soil fields come from the physical donor at the
inference cycle, broadcast in time by the shared engine composition.

FCN3 and SFNO weights are Apache-2.0; Aurora weights are MIT. These are the
default rows, with licence links in the table. Other verified weight grants
require `--accept-licence 'EXACT NAME'` on the global, S2S or init-source door.
Rows without a weight grant or a compatible global pressure-column contract
refuse with the concrete missing requirement. A research opt-in does not
override those refusals.

Install the engine's Earth2Studio environment and set `WOOF_EARTH2_PYTHON`
to its interpreter. The shared runner checks its pinned Earth2Studio version;
model-specific runtime dependencies remain the engine's responsibility.
Inference and Rust composition use 14 workers by default. `RAYON_NUM_THREADS`
or `init-source --jobs` sets another budget. On Linux the inference child
inherits a bounded CPU affinity, which the engine uses to size Torch and BLAS
pools; the caller's affinity is restored even when inference fails.
The initialization receipt includes the extension digest, inference receipt,
channel bindings, Rust export command and effective first-guess counts.

For ARCO surface donors, `generation.global_donor` declares an explicit
360-degree native request. It bypasses the corner-string dateline heuristic;
Rust acquires only the surface and soil fields needed by composition.
Rerunning the same work directory reuses the generated forecast only when
its model contract, clocks, member, checkpoint revision and every output's
size and SHA256 match the saved runner receipt. A mismatch refuses before
inference instead of overwriting a forecast of different provenance.
