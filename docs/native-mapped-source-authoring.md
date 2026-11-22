# General mapped-source authoring

RW-WPS can now compile an explicit source descriptor and author the exact
input manifest used by the native GRIB1, GRIB2, and NetCDF path. This removes
the need to edit a hash manifest by hand and avoids adding a Python adapter for
every named forecast model.

This is a general *configuration* path, not automatic scientific inference.
The emitted status is `VALIDATED_NOT_STOCK_WRF_CERTIFIED`. A new descriptor
does not inherit the retained ERA5/GFS stock-WRF gates merely because it uses
the same decoder.

## Native preparation and WRF files

Mapped preparation uses the experiment's configured physics. The companion
WRF file export is `optional` by default, for both a single domain and a tree.
A configuration the native runtime accepts can therefore produce its prepared
forecast even when the WRF exporter cannot represent it. The proof records
`REFUSED` with the exporter's reason, and the command prints that outcome.

Use `--stock-wrf-export required` when `wrfinput`/`wrfbdy` are required products.
The same exporter-owned configuration checks run before source decoding or
initialization; prepared arrays still receive their full validation at export.
The hierarchy export retains its existing supported physics slice. Use
`--stock-wrf-export off` (or `--no-stock-wrf-export`) to skip WRF files entirely.
These controls apply to the mapped runner, including packaged source profiles.

Only an explicit unsupported-export result is optional. Failed writes,
corrupt caches, and mismatched physics remain preparation failures. Export
mode does not change the native meteorological arrays or requested physics.
The low-level WRF export API keeps its existing required behavior.

## Authority boundary

`rw-wps.descriptor.v1` has the same coordinate, field, derivation, and target
semantics as executable `rw-wps.mapping.v1`. For GRIB input, each direct
field replaces `selectors` with `vtable_selectors`. Each reference must resolve
exactly one row in the supplied 11-column WPS Vtable. RW-WPS imports only the
numeric GRIB selector from that row. The descriptor remains the authority for:

- canonical field meaning and source/target units;
- axes, rank, staggering, and grid location;
- missing-data and mask policy;
- vertical inventory, ordering, derivations, and cadence;
- soil meaning and target WRF/domain requirements.

A Vtable's mnemonic, unit label, and description are retained in the
authoring receipt but are not treated as those scientific decisions. This is
important because real Vtables reuse names such as `TT` for both 3-D model
levels and 2-m temperature.

## Declarative soil geometry and remapping

The composition contract owns soil semantics independently of a source name.
`soil_layers` now declares:

- canonical temperature and volumetric-moisture fields;
- metre-valued, contiguous, shallow-to-deep source and target layer bounds,
  with the exact temperature and moisture selector bound to every source
  depth;
- either layer-bottom point interpolation to Noah midpoints with explicit
  surface/deep anchors, or conservative layer-mean overlap remapping with
  complete source coverage; and
- a fail-closed missing-data rule: source-land gaps are rejected before
  horizontal interpolation, while ocean soil is repaired after interpolation
  from skin temperature and unit moisture.

The checked ERA5 contract declares its historical WRF `module_soil_pre`
layer-bottom interpolation. The GFS contract declares conservative layer
means; because its four bounds exactly equal Noah's, that route remains a
bit-identical copy. No runtime branch refers to ERA5, GFS, or legacy soil slab
field names. Every GRIB1, GRIB2, and NetCDF selector is checked
position-by-position against the selector stored beside its declared source
depth; GRIB2 type-106 numeric bounds receive an additional semantic check.
The Rust frontend independently repeats the engine's selector, depth, order,
remap, unit, and ocean-policy checks. This incompatible replacement of the
old named-packing document is explicitly versioned as
`gpuwm-mapped-composition-v2`; v1 evidence remains replayable at its sealed
implementation commit but cannot be mistaken for this contract.
Changing the composition bytes intentionally invalidates the retained exact
mapping/composition stock-WRF certificate. The new contracts remain runnable
but are not rebound to that certificate until a fresh unchanged-stock-WRF gate
is recorded, even though focused tests prove byte-identical ERA5/GFS soil
state relative to the retired packing branches.

One GRIB direct field looks like this:

```json
{
  "air_temperature_2m": {
    "vtable_selectors": [{
      "metgrid_name": "TT",
      "grib2_level_type": 103,
      "level1": 2,
      "selector": {"level_value": 2.0}
    }],
    "units": {"source": "K", "target": "K"},
    "source_axes": ["y", "x"],
    "target_axes": ["y", "x"],
    "location": "surface",
    "staggering": "none",
    "missing": {"kind": "reject"}
  }
}
```

The extra explicit `selector.level_value` is required for numeric GRIB2
`Level1` rows because the Vtable does not provide a general physical-unit
contract for GRIB2 scaled fixed surfaces. Bounded layers require the atomic
`second_level_type` and `second_level_value` pair. Zero matches, multiple
matches, reused rows, overlapping broad/narrow direct selectors, hand-authored
GRIB selectors, incomplete surfaces, and unknown keys all fail before a
mapping is written. NetCDF descriptors keep
the existing explicit `name`/`standard_name` selectors and reject Vtables.

GRIB1 references must additionally declare both `selector.center` and
`selector.table_version`; a Vtable does not carry enough authority to infer
them. A numeric GRIB1 `Level1` cannot be changed by an override (a `*` row may
be narrowed explicitly). GRIB2 local-use codes 192--254 require an explicit
four-part Section 1 authority binding: `center`, `subcenter`,
`master_table_version`, and `local_table_version`. Both Rust inventory and
dump tools emit those values, and the Python runtime rejects any disagreement
between them. `local_table_version=255` cannot authorize a local-use code.
Missing/undefined identifier code 255 is also refused for required GRIB1
identifiers and for GRIB2 parameters, categories, disciplines, and second
surfaces.
GRIB2 `level_type=255` is retained only as the explicit no-fixed-surface case
and cannot carry fixed-surface values. Duplicate JSON object keys are rejected
at every nesting level rather than silently taking the last value.

## Create-only authoring

The public command can compile the mapping and create the input manifest in
one author-only call:

```bash
rw-wps --source mapped --source-format grib2 \
  --descriptor /case/source.descriptor.json \
  --vtable /case/Vtable.PRODUCT \
  --author-mapping /case/source.mapping.json \
  --composition /case/source.composition.json \
  --input /data/source-f000.grib2 \
  --input /data/source-f003.grib2 \
  --supplement terrain=/data/terrain-f000.grib2 \
  --supplement terrain=/data/terrain-f003.grib2 \
  --provenance terrain_provenance=/case/terrain-source.md \
  --author-input-manifest /case/input-manifest.json \
  --author-only
```

`--input-list FILE` says the same thing as the repeated `--input` flag as a
file -- one path per line, UTF-8, in the same deterministic time/file order,
blank lines skipped, nothing interpreted. It exists for command-line length:
Windows caps a whole command line at 32 KB, and a field-per-file source is
hundreds of `--input` flags per prepared state (a real ICON-EU cycle is 251).
Exactly one of the two spellings per invocation; both `rw-wps`/`woof prep`
and the internal mapped runner accept it, so a `--dry-run` preview of a large
run stays pasteable. When the per-file spelling is used and the platform
refuses the internal relaunch's length, the front door retries once through a
temporary list file by itself -- a command line the platform accepts is never
rewritten.

Use the exact role names declared by the composition. For an installed
distribution the launcher discovers its bundled GRIB bridge paths relative to
the installation, not through `PATH`; explicit decoder flags that resolve to
any other file are rejected. Manifest authoring verifies the ELF identity,
usage probe, and the generic GRIB2 tabular ABI before hashing a decoder.
Author-only and dry-run modes do not require a CUDA device; end-to-end native
WRF file production still does.

The same flags may be supplied with the normal WPS geometry, geography,
experiment, and output arguments. In that form RW-WPS authors the create-only
contract, inserts the resulting manifest path and digest into the internal
argv, and immediately runs the native initializer. `--dry-run` is strictly
side-effect free and refuses authoring flags. Use `--author-only` to write the
contract without starting initialization, then use those emitted paths in a
separate `--dry-run` if an argv preview is wanted.

Authoring never overwrites a mapping, receipt, or manifest. Source order and
role partitions are preserved; every resolved file is bound by byte count and
lowercase SHA-256. Files are hashed through stable open handles, the candidate
manifest is checked by the runtime verifier, all authorities are rechecked,
and only then is it published with an atomic no-clobber link. A failed rewrite
therefore cannot replace a prior valid manifest.

The stable-handle checks detect persistent path, identity, metadata, or byte
drift visible during the before/after snapshots. They are not a filesystem
lock and do not claim to defeat a hostile same-user process that can swap an
authority to different bytes only while an external decoder reopens it and
restore the original perfectly before the final snapshot. Production inputs
and bundled decoders should therefore be staged read-only or protected from
concurrent writers while authoring and decoding run.

When mapping and manifest authoring are requested in one call, a manifest
failure removes only the exact mapping and receipt proven to have been created
by that invocation. A corrected retry therefore does not hit a stale partial
transaction.

The generic mapped runner is reported as runnable but not globally
stock-WRF-certified. Machine-readable stock-WRF evidence is keyed to exact
retained mapping/composition SHA-256 pairs; a newly authored pair never
inherits those gates by sharing a decoder or format.

## Which engine reads the bytes

Nothing above depends on it -- a mapping document is the same document
either way -- but it changes what a mapped command needs on the machine.

By default the mapped routes decode through `gpuwm_mapped_engine`, one
executable resolved through the usual bridge ladder
(`WOOF_MAPPED_ENGINE_BIN`, this checkout's `tools/rw_wps/target`,
`libexec/bridges`, the packaged copy, then `~/.woof/bridges`) and held
to the contract marker `gpuwm-mapped-frameset-v1`. It reads GRIB2 in
process, so a GRIB2 mapped command needs no `--grib2-inventory` /
`--grib2-dump` paths at all, and an input manifest seals the ENGINE as
its decoder rather than the subprocess pair.

Three paths are still decoded by the Python engine on a bare run: a
cross-source composition, a mapping whose `format` is `grib1`, and a
mapping whose `format` is `netcdf`. All three are unported work rather
than a decision, all three are printed by `woof doctor`, and
`gpuwm_mapped_engine capabilities` is the binary's own statement of what
it can do -- so if you are authoring a GRIB1 or NetCDF mapping today,
nothing about the document changes, only which engine reads it.

`--mapped-engine python` (or `GPUWM_MAPPED_ENGINE=python`) runs the
Python decode path instead. It is a WORKAROUND for a decode the Rust
engine gets wrong, not a supported mode; manifests sealed against the
subprocess tools replay on it. Naming a decoder tool explicitly selects
it for that run, because those flags pin which binary reads the bytes.

`woof doctor` reports the engine and which one a bare run uses. The
contract, the refusal-class table and the parity gate are in
`docs/dev/decode-vendor-design.md`.

## Current boundary

This slice removes source-family dispatch, manual manifest JSON, and the two
named four-layer soil packings from the mapped path. It does not yet generalize
the terrain-only supplement engine. Curvilinear GRIB grids, vertical nest
refinement, moving nests, and two-way feedback also remain unsupported. A
descriptor must materialize the existing complete canonical source frame and
pass the existing donor-grid, hierarchy, initialization, and export gates.


## Height-based model levels

These keys are read by the mapped engine from a mapping JSON document.
The descriptor authoring path (`rw-wps --author-mapping`) does not yet
accept `interface_levels`, `half_level`, `height_from_interfaces`,
`mass_fraction_rebase`, or `dependency_only`. Author these mappings directly
as JSON and validate them with the mapped input-manifest authoring path.

A GRIB source that publishes pressure on mass levels and heights on their
interfaces uses `vertical.kind = "model_level"`, dimensionless `levels`, and
an ordered `interface_levels` list with one more entry than `levels`.
Direct interface heights use `source_axes` and `target_axes` of
`["half_level", "y", "x"]` and target units `m`. The
`height_from_interfaces` derivation takes that field as `source` and averages
adjacent interfaces into `geopotential_height` on `["vertical", "y", "x"]`.
Every column retains its own heights. Missing interfaces, nonfinite heights,
and crossing layers refuse because they would misplace the mass levels.

A NetCDF source whose files carry the column on their own coordinates (WRF's
`bottom_top` and `bottom_top_stag`) may instead declare no `levels` and
`"interface_levels": "from_file"`: every level and every interface the file
offers is the column, and a file that offers N levels with other than N+1
interfaces refuses. The packaged `hex-global` profiles do this, so a
woof hex parent decodes on whatever level count its suite used (55 on the
default suites, 59 on hrrrv5). A declared `levels` list beside it refuses,
because it would cut the mass column the interfaces still bound.

Interface heights may declare `time_binding = "cycle_invariant"`; assembly
broadcasts the complete coordinate to each forecast time. Prognostic mass
and soil fields still require valid-time data. Pressure remains a direct
field. No pressure-from-level-number inference is made.

The packaged ICON-D2 route uses these declarations by default: 65 mass levels,
66 invariant height interfaces, pressure, specific humidity, and five
condensate species. The emitted model top is 6000 Pa. These are source-table
entries; the mapped engine does not dispatch on the source name.

`mass_fraction_rebase` converts a declared `source` fraction to a reference
mass with the fractions listed in `exclude` removed: source divided by
(1 minus those fractions). Dependencies must share axes and units `kg kg-1`.
Nonfinite or negative fractions, or a nonpositive reference mass, refuse.
The source table excludes all water species for dry-air condensate mixing
ratios, and only condensate for gas-specific humidity. This preserves the
same dry-air basis when initialization converts humidity to vapor mixing ratio.
DWD notes that the condensate GRIB names are mixing ratios but the values are
specific quantities in its [field documentation](https://www.dwd.de/SharedDocs/downloads/DE/modelldokumentationen/nwv/icon/icon_dbbeschr_aktuell.pdf).

A field that is decoded only so a derivation can read it declares
`"dependency_only": true`. Both engines decode and validate it, pass it to
every derivation that reads it, and leave it out of the frame, so it takes
no space in the compose stream. The packaged ICON-D2 mapping marks its six
raw mass fractions this way, and each frame carries only the humidity and
mixing ratios derived from them. A field the target requires cannot be
`dependency_only`; that mapping is refused because its frames would lack
the field. A field a composition borrows from a donor is published even
when the donor's own mapping marks it `dependency_only`.

A NetCDF source names its interface coordinate the way it names its mass-level
coordinate: `vertical.interface_selector` (for a WRF-shaped file,
`{"format": "netcdf", "name": ["bottom_top_stag"]}`) beside `vertical.selector`.
Fields on `["time", "half_level", "y", "x"]` then read on that dimension, and
`height_from_interfaces` averages them to mass levels exactly as on GRIB.

Two closed-catalog derivations serve height-coordinate producers that publish
potential temperature and mixing ratios:

- `air_temperature_from_potential_temperature` (`potential_temperature`,
  `pressure`; optional `kappa`, default 2/7, and `reference_pressure_pa`,
  default 1e5): T = theta (p / p0)^kappa, with the producer's own constants.
- `specific_humidity_from_mixing_ratio` (`mixing_ratio`): q = r / (1 + r),
  which initialization's r = q / (1 - q) inverts.

The packaged `hex-global` row (profile `hex-history-netcdf-v1`) uses all three:
a `woof hex export --field-set full --window region-latlon:CLAT,CLON,W_KM,H_KM`
frame set is a WOOF-ArWen domain's initial and boundary source. Lat-lon export
frames carry CF 1-D `lat`, `lon`, `Time`, `bottom_top`, `bottom_top_stag` and
`soil_layers_stag` coordinates (rw_mpas_convert writes them on every lat-lon
window). The row declares no boundary cadence of its own -- the spacing is
whatever the export carries -- so `woof domain` takes it from `--cadence`,
and the frames handed to `woof prep` are the ones covering the run's window
(f00/f03/f06 of a 3-hourly export for a 6 h run from the export's first
time), each one also named as its own in-band surface supplement. The whole
route, as run on 2026-10-17 (a 3 km 300x240 domain prepared in 23 s from
three 60 km hex frames):

```bash
# 1. plan the domain; --cadence 3 is the export's 3 h frame spacing
woof domain --source hex-global --cadence 3 --point 36,-91 --root-dx 3 \
  --point-extent-km 900 --hours 6 --cycle 2026-10-16T00 --name hexarwen \
  --out hexarwen.toml
# 2. stage the geography this config reads (woof check names any dataset
#    still missing, with this same remedy)
woof fetch-geog --config hexarwen.toml
# 3. prepare from the frames covering the window; the manifest path is
#    written by prep (authored), not supplied
woof prep --source hex-global \
  --input frames/wrfout_d01_2026-10-16_00_00_00 \
  --input frames/wrfout_d01_2026-10-16_03_00_00 \
  --input frames/wrfout_d01_2026-10-16_06_00_00 \
  --supplement hex_in_band_surface=frames/wrfout_d01_2026-10-16_00_00_00 \
  --supplement hex_in_band_surface=frames/wrfout_d01_2026-10-16_03_00_00 \
  --supplement hex_in_band_surface=frames/wrfout_d01_2026-10-16_06_00_00 \
  --author-input-manifest inputs.json \
  --experiment-config hexarwen.toml --wps-namelist hexarwen.namelist.wps \
  --geog-root WPS_GEOG --output-root prepared
# 4. run and render (the line prep prints); the config has no [fetch] table,
#    so it is driven by `woof sim prepared`, not `woof go`
woof sim prepared --experiment-config hexarwen.toml \
  --wps-namelist hexarwen.namelist.wps --outdir prepared-forecast \
  --render-products all
```
