# NextGen hydrofabric startup tool

Source/notice: `NOTICE`, pinned ngen Apache-2.0 LICENSE in `notices/`.
This is a fresh Rust implementation; no withdrawn UCAR implementation or fixture
is used. Hydrofabric database licences remain separate from code licences.

The concrete static exporter fields, driver resource row, read APIs and missing
physical initialization inputs are listed in [STARTUP.md](STARTUP.md).

Build **from this directory** with `cargo build --release --offline -j 8`.
Put CARGO_HOME, CARGO_TARGET_DIR and TMPDIR under your data-disk scratch root.
The CLI and cdylib use pinned static PROJ 9.7.1 and SQLite 3.46.1 on Linux and
Windows, with an embedded projection database and network disabled. See
[`hydro_proj`](../hydro_proj/README.md) for sources and build requirements.
The `--proj-probe` command and library projection probe exercise that dependency.
Wheel staging remains the release tooling's separate responsibility.

`rw_hydrofabric prepare CONFIG.json` reads the configured GeoPackage, resolves
reach->nexus->reach edges, produces upstream CSR and both routing level orders,
preserves attribute types, and optionally clips divide polygons to a parent grid.
`woof.hydro_preprocess.register_cli(hydro_subparsers)` supplies
`woof hydro prepare CONFIG.json` when the shared CLI optional-loads it, using
`set_defaults(func=callable)`. Registration and help need neither a Rust artifact
nor CUDA; execution delegates to the same Rust preparation tool. The existing
`woof.hydrofabric.register_cli` remains available for the older `fabric` command.
The shared CLI owner controls registration in the common command door.

Config keys: `geopackage`, `output`, `data_license`; optional `gauge`, or `huc`
with the actual `huc_column`; `tables` maps roles to actual layer names.
Gauge selection prefers the native hydrolocation/network nexus over possibly
duplicated rl_gages attributes. gauges.json retains authoritative network station
links, removing exact duplicate records and preserving any remaining ambiguity.
HUC selection requires a real catchment/flowpath membership column; HUC outlet
hydrolocations alone do not establish polygon membership.
Roles include divides, flowpaths, nexus, network, flowpath_attributes,
model_attributes, hydrolocations, lakes. Missing optional attribute tables stay
absent: no fabricated CFE parameters or hydraulic reach meshes.

Optional `grid`: `nx`, `ny`, `x0`, `y0`, `dx`, `dy`, `crs`. Origin is southwest
**cell edge**, positive spacing is metres, flattening x-fast, with `row_order` set to `south_to_north` (default) or
`north_to_south` to match the actual parent/terrain grid.
The CRS must be projected in metres. PROJ transforms divide vertices into that
CRS, network vertices are piecewise linear, polygon/rectangle areas are planar
projected m2. Rings/holes/multipolygons are supported. Weights normalize covered
area; coverage_fraction explicitly reports truncation at the grid boundary.
channel_node i32[ny,nx] rasterizes actual flowpaths into the same indexed reach roster;
stream_order preserves native integer order. At junction cells the downstream-most
topological rank wins deterministically. These are channel cells, not filled
catchment labels.
No DEM, terrain parameters or synthetic parent meteorology is generated here.

For a prepared WOOF-ArWen forecast, coupled-driver's `ParentGeometry` retains the
**executed** `ProjectedGrid`. During setup use:

```python
from woof.hydrofabric import parent_grid_contract
contract = parent_grid_contract(physics.hydro_parent_geometry)
config["grid"] = contract["grid"]
config["parent_grid_contract"] = contract
```

The helper uses the original grid's cached native handle and the additive
`gpuwm_static_grid_hydrofabric_json(handle, buf, capacity) -> i64` C ABI. Return
value is full UTF-8 length, or -1 with the static bridge's last error; sizing
uses null/zero. `static-fields::highres::raster_geometry` supplies the projected
footprint. Rust converts its north-first affine origin to a southwest cell edge;
resident mass arrays retain `south_to_north` indexing. Translated native specs
retain composed integer offsets and extent. CRS metadata explicitly describes
the existing 6370000 m model sphere, including Mercator's `ref_lon` anchor.
Latitude-row grids cannot supply this rectangular projected-metre contract.
No corners, projection transforms, areas or weights are computed in Python.

Preparation checks that the contract's grid equals the configured grid and
retains the whole contract in the manifest and `load_cuda` restart identity.
The driver's definition SHA and verified static/geometry authority hashes are
preserved; changed forcing provenance does not change geometry identity. This
seam supplies no forecast input itself. Actual parent/catchment overlap weights
require a prepared forecast plus the selected catchment geometry; an existing
channel-cell incidence map cannot stand in for that overlap.

Preparation uses Rayon across read-only per-table SQLite connections, catchments
and payload writes. Dependency traversal is startup-only. Network rows preserve
SQLite scan order as ngen's `SELECT *` does. Default upstream reduction order is
**not claimed native**; `upstream_order` accepts an actual native-captured ID
roster. Never call it proven without comparing the referee connection order.

Outputs use a config+input-digest directory under output, so changed requests
coexist and repeats recompute only their own payloads. manifest.json declares
relative filenames, dtype, shape, length, SHA256, CRS, data provenance and stage
timings. Network i32 indices use -1 at an exterior boundary; IDs preserve full
strings, plus native numeric suffixes in i64 when all IDs have them. MC parameter
columns f32[N,8]: length_m,BtmWdth,TopWdth,TopWdthCC,n,nCC,ChSlp,So. Native f64
attribute values remain in attributes.json. GPU weight CSR: offsets u64,
indices u32, weights/overlap_m2 f64. catchment_to_node i32 maps the independently
preserved divides order into flowpath order. `exchange_offsets` i64,
`exchange_indices` i32 and `exchange_weights` f32 are Rust-prepared aliases for
the Noah-OWP exchange ABI; forcing retains the original f64 geometry weights.

`woof.hydrofabric.load_cuda(manifest, device_id=...)` reads/verifies payloads
through the Rust ABI into pinned memory and transfers them once. Python reads
metadata and orchestrates CUDA copies; no Python numeric decode/regrid. Prepared
arrays and restart identity match the fresh driver setup contract. No step
callback, host-per-step work, CUDA graphs or GPU ownership is introduced here.
The declared `auxiliary["network.json"].sha256` is also required and verified in
Rust before parsing the node/catchment rosters or entering CUDA setup. A changed
roster fails instead of silently relabelling the resident arrays. The verified
digest is retained as `restart_identity.network_sha256`; older identities need
a fresh preparation rather than resuming with an unbound roster.

Other production input factories can reuse this transport without the offline
replay tool. Resolve `reader = woof.hydrofabric.RawPayloadReader()` once during
setup, allocate caller-owned pinned memory, then call
`reader.read_into(path, expected_sha256, destination_address, nbytes)` for each
manifest field. The synchronous ABI1 `arwen_hydrofabric_read_into` reads and
validates the entire file in Rust, then performs a native byte copy. It leaves
the destination unchanged on read/length/hash failure. The caller owns the
destination lifetime, declared dtype/shape, and H2D copy; no Python numeric file
read, conversion or state initialization occurs. `read_raw_into` is a convenience
wrapper accepting an optional reusable `reader=`.
For JSON metadata, `reader.read_json(path, expected_sha256)` uses the same Rust
read/hash transport and lets Python parse only the verified bytes. This does not
authenticate a changed manifest: the driver still pins the selected request's
manifest authority.

`nextgen_hydrofabric` is registered in the existing production bridge artifact
roster, ABI table, static contract marker and wheel-stager workspace map. Its
library resolves through the existing package/staged/explicit-override ladder;
there is no scratch-directory default. The cdylib and GIS CLI carry their pinned
static SQLite and PROJ libraries, with embedded projection resources; neither
requires a system installation of those libraries. Actual platform binaries still must be built
and staged by the release owner; registration alone does not qualify a wheel.

The existing manifest is specified in `schema/manifest-v1.schema.json`; its
array descriptors are relative path, little-endian dtype, shape, bytes and
SHA256. Grid, routing, reservoir, parent identity and replay provenance metadata
remain intact. `output` and original source paths are provenance, never a new
base directory for reading a relocated artifact. The manifest directory is the
payload root. Data terms are retained independently from the software licence.

Driver-owned Rust input binding can use `manifest::PreparedManifest::open(path)`
once, inspect `.document.files`, then call
`read_field_into(name, &mut driver_owned_transport)`. Rust checks shape products
and dtype widths without overflow, resolves confined relative paths, verifies
the complete byte count/SHA, and copies only after validation. Use
`read_auxiliary("network.json")` before parsing that metadata. It verifies the
declared digest. The caller pins the manifest/identity to its selected request;
the digest is an integrity check, not independent authority for edited input.

For a C/ctypes binding, `include/hydrofabric.h` declares the additive ABI1 entry
`arwen_hydrofabric_manifest_read_into(manifest_path, field_name, dst, length)`.
The caller allocates transport; status0 succeeds,1 rejects pointer/UTF-8,2 means
manifest/payload validation or I/O failure,3 means panic. On failure the target
is unchanged. The old raw read/hash helper remains available. No factory,
allocation policy, numeric conversion, routing state initialization or stage
sequence is selected here: driver owns binding/init and speed owns bound
numerical capture/sequence. The executable example `examples/read_manifest.rs`
compares all real fields through this Rust API against the existing raw reader.

This validates prepared **transport**, not original GIS reconstruction. Captured
network/parameter payloads do not prove derivation from an original DEM. The
channel-only inverse raster CSR is incidence, not complete D8 catchment
drainage, off-channel routing or a parent-area partition. Missing DEM/physical
providers stay missing. Original federal topology/polygon/flowline replays are
separately scoped to their actual source records and native captures.

Replay: `rw_hydrofabric replay --strict MANIFEST NATIVE.tsv` byte-compares
parsed_network.tsv with actual ngen-captured bytes, header `layer\tid\ttoid`,
roles divides/flowpaths/nexus, then native feature scan order. No source-derived
or copied expected values count as a native replay. Native fixture and short
coupled run/picture are acceptance requirements, pending owner evidence.


`prepare-routing-replay CONFIG.json` prepares an explicitly captured native
routing network, not an independent reconstruction from GPKG. The config names
`capture_manifest`, `fabric_manifest`, `reservoir_manifest`, `output`,
`time_origin`, and `data_license`. Rust checks native forward/reverse/actual
compute graphs, retains incoming SUM order, derives MC-only levels and reservoir
selections, packs native typed parameters/state/lateral input, remaps catchments
and channel raster, and incorporates the reservoir owner's sealed payloads.
All input identities are part of the request identity. This replay route does
not download data and is not a public demo path.
The native capture must explicitly include `assume_short_ts` from its actual
compute call. Rust preserves it as `routing.assume_short_ts`; a missing value
cannot silently select a different upstream-discharge update convention.

The prepare-only realization resource is `woof.hydrofabric:prepare_resource`
with `options.manifest`. MC uses `network_component` and `lateral_component`
naming that resource, plus `reservoir_component`. Its prepared
`routing.initial_state_array` supplies initial state by default;
`initial_state_component` remains available as an explicit override.
Reservoirs are a prepare-only resource; MC owns their level-ordered launches.
`mc_ordered_nodes` is the channel-only roster; reservoir depths are elevations
and must be labelled separately. Numerical packing stays in Rust.


`replay-collapse --strict FABRIC_MANIFEST CAPTURE_MANIFEST OUTPUT` reconstructs
collapsed connections and native incoming lists independently from the sealed
GPKG tables and compares their canonical JSON bytes. It also compares every MC
parameter word after the native f64-to-f32 conversion. The native HYFeatures
lake-to-lake replacement is included. Seven case(b) lakes have multiple outlet
flowpaths; the arbitrary upstream Python set choice is reported as unproven,
while the running replay bundle uses the actual captured crosswalk. A passing
verdict covers connections, incoming order, membership and parameters only.

`output_reach_ids` maps each internal common node to t-route's final output
ID. Native t-route renames lake IDs to selected outlet flowpath IDs after routing.
Use this ID table at the Rust output boundary when comparing final histories;
internal `node_ids` retain lake IDs for routing. Neither vector changes per step.

For a prepared channel raster, `routing.cell_mapping` names inverse CSR
`routing_cell_offsets` i64[N+1] and `routing_cell_indices` i32[M] on the current
collapsed `node_ids` axis. Cells ascend within each row; grid, roster and source
raster hashes are explicit. Preparation checks every source pixel by an inverse
roundtrip. This is channel-cell incidence only, with nonchannel cells excluded;
it does not assign their drainage or choose reservoir lateral transfer policy.

Original federal source acquisition is separate from the restricted HYFeatures
replay. `data/public-sources.json` records exact EPA object URLs, content SHA256,
sizes, actual provider metadata and table column mappings. These rows are not a
complete demo: NOAA routing/reservoir parameters and complete catchment coverage
remain required. Gauge identifiers are taken from the original `GageLoc.dbf` table,
never inferred from HYFeatures feature numbers or coordinates.

`inspect-source-dbf TABLE.dbf` reads the actual fixed-width schema and samples.
`query-source-dbf CONFIG.json` selects configured columns where `where_column`
matches `equals`; config also requires `path` and `sha256`. Character IDs retain
leading zeroes. `subset-source-network CONFIG.json` reads independent `edges`
and `attributes` sources, each with `path` and `sha256`; column mapping keys are
`from_column`, `to_column`, `id_column`, and `attribute_columns`. Other keys:
exactly one of `outlet` (original integer ID), `outlets` (nonempty list of
distinct positive original IDs), or `selection` (sealed spatial seed manifest), `check_reaches` (optional), `output` (new
directory), `provenance`. A list selects the union of upstream networks in one
source decode, sharing common upstream nodes. Every listed outlet is an explicit
downstream cut: use the domain's terminal outlets, not its internal gauge IDs.
Outgoing arcs from every cut remain in `boundary_connections`, including an arc
to another selected node. List order does not affect the output. The scalar
form retains its original payload format; the list form records sorted `outlets`
instead of `outlet`. This is source topology selection only: it does not choose
outlets from a drawn polygon, resolve cross-region source coverage, initialize
water state, or produce an admitted runtime fabric. It exports all upstream
nodes and arcs plus original boundary
arcs, including diversions, and reports missing attributes/cycles explicitly.
Scientific-notation decimal IDs are parsed with integer arithmetic. Attribute
values remain source strings; no missing hydraulic parameters are synthesized.
Reads and fixed-width decoding use Rayon; dependency traversal is startup only.

`replay-source-network --strict NETWORK.json NATIVE.csv` compares the complete
node/edge sets with a real independent native GDAL SQLite capture. Its numeric
CSV header is `kind,from_id,to_id`, with kind1 for node IDs and kind2 for arcs.
The reference query and executable/input hashes accompany the actual receipt.
GDAL is a referee tool only, never a production or wheel dependency here.

Build the optional source acquisition binary with
`cargo build --release --offline --features source-archive --bin rw_hydrofabric_source_archive`.
It consumes a JSON array of jobs: `archive`, required `sha256`, exact `members`,
`output`, `max_extracted_bytes`. Empty members means inventory only. Archives
are independent Rayon jobs; each compressed stream is sequential. The separate
binary links system libarchive (notice retained), without adding it to the normal
CLI or raw bridge. Bounded prefix inventory supports `inventory_entry_limit`
and `inventory_skip_budget`; `complete_archive_inventory=false` explicitly
prevents treating an HTTP fragment digest as a complete archive inventory.

For pinned original-source regions, the same binary can plan those jobs from
the catalogue without acquiring or extracting data:

```text
rw_hydrofabric_source_archive --plan data/public-sources.json RECIPE ARCHIVE_ROOT EXTRACTION_ROOT MAX_COMPRESSED_BYTES
```

The catalogue's `recipes` map selects exact object names, member paths, distinct
output subdirectories and extraction budgets. Its existing `objects` rows supply
the original URLs, sizes and hashes. A new region is metadata work. The plan
prints `downloads` and `jobs`, the catalogue hash, original metadata hash and
agency credit. Pass its `jobs` array to the existing extractor after acquiring
the selected archives. Roots must be absolute and disjoint; the planner writes
no files. The compressed-byte budget bounds the selected original objects even
when some are already cached. The extractor still hashes actual archives before
decoding and refuses existing output files.

Use `--check` with the same arguments to inspect the selected local archives
before acquisition. Its `cache` section reports verified reusable bytes, missing
objects and their exact remaining download budget. Missing files are a valid
`ready=false` result; wrong size, SHA256, symlinks and non-file entries are errors
that name the archive and leave all files untouched. Checks run across independent
archives with Rayon and stream SHA256 using 64 KiB per job. No cache or output
directory is created, nothing is fetched or deleted, and extractor hashing still
runs before decoding. Readiness is an integrity observation at check time, not
domain coverage or physical fabric admission.

Use `--fetch` with the same arguments to acquire only the missing pinned
archives. This optional preparation command requires system `curl` for HTTPS
transport; Rust streams and hashes the response with a 64 KiB buffer per job,
never writes beyond the catalogue's compressed-size bound, and checks exact
size and SHA256 before publication. Independent missing objects use Rayon.
TLS verification stays enabled, redirects stay HTTPS, and curl configuration
files are ignored. Transport failures retry up to three times from a fresh
partial file; a checksum or size failure is returned immediately. Verified
existing archives require no network or curl invocation.

Each download uses a newly created temporary directory beside its final cache
path. Publication creates a hard link without replacing any destination;
a concurrent winner must pass the same size/hash checks. Temporary bytes are
removed on success or error; verified independent downloads remain reusable
if another job fails. A killed process can leave its named temporary directory
for the owning preparation workflow to clean. No extraction directory is
created and no archive is decoded. This command acquires original source
objects only; domain selection, physical parameters and the shared fabric
builder remain separate preparation stages.

`nhdplus-ca18-topology` selects the CA18 attributes and national gauge-location
archive (15,430,270 compressed bytes); `nhdplus-ca18-geometry` selects source
catchments and flowlines (248,696,911 bytes). These are explicitly selected
regional source plans, not a complete CONUS tile index or user-domain coverage
selector. They retain the previously measured source gaps and invalid polygons;
neither plan supplies model parameters or initializes water state. Source
archives may be absent after bounded extraction and cleanup. `--plan` does not
infer their presence; `--check` inspects bytes and `--fetch` explicitly acquires
only verified missing objects.

These plans feed original-source preparation for the `conus-nhdplus-v21` row
of the HYDRO-WORLD contract. Final construction still uses the shared world
builder's writer; no alternative fabric writer or physical admission is added.
HydroSHEDS v1 remains research/cross-check only under the coordinator's
2026-10-18 11:50Z ruling. Worldwide DEM hydrography and calibration stay with
their respective owners; original NHDPlus geometry is the CONUS reference.

Original polygon and flowline geometry can be bounded without a GDAL runtime:
`rw_hydrofabric subset-source-geometry CONFIG.json`. The configured `shp`,
`shx`, `dbf`, `prj`, and `network` each require `path` and `sha256`; `network`
is the actual `arwen.source-network-subset.v1` node roster. `id_column`,
`data_license`, `provenance` and a fresh `output` directory are required.
Optional `stem` names the output shapefile (default `Catchment`).
This generic shapefile reader supports Polygon, PolygonZ, PolyLine and PolyLineZ.
PolyLineZ preserves every source Z value and optional M value, including finite
no-data sentinels; no elevation datum or measure interpretation is inferred.
It validates physical SHP/SHX records and joins DBF by physical record number,
including deletion markers. Selected geometry and DBF records remain verbatim;
output headers/indexes describe the selected bounds and record count. A source
ID with several records retains them all. `missing_node_ids` explicitly
records requested reaches without source geometry. The source CRS declaration
is retained; degree coordinates are never interpreted as metres or grid cells.

`rw_hydrofabric replay-source-geometry --strict MANIFEST.json NATIVE.csv`
compares every selected XY bit and closed-ring edge sequence with an independent
GDAL decode of the original full source. Native CSV columns are `source_id,wkb`,
with hexadecimal WKB from `hex(ST_AsBinary(geometry))`. Canonicalization removes
ring start/orientation/group-order differences while preserving coordinate bits,
edges and multiplicity. Polyline replay compares all source XY/Z/M bits,
dimensional modifiers and directed point sequences. Collection grouping and
part order alone are normalized; lines are never reversed. The output schema
is `arwen.source-polyline-subset.v1`, separate from resident GPU manifests.
It does not prove polygon containment, projected areas, elevation semantics,
parent-grid weights or hydraulic parameters. GDAL is a referee only, never a
runtime dependency. `reference/replay_source_geometry.py` rebuilds from the
metadata-pinned original archive in fresh scratch and executes this comparison;
Python writes metadata and launches tools only. Its `--download` explicitly
acquires the source when no local `--archive` is supplied.

### Source-domain seeds and upstream closure

`rw_hydrofabric select-source-domain CONFIG.json` intersects original active
flowpaths with a supplied simple, closed polygon ring. Required keys:
`shp`, `shx`, `dbf`, `prj` (each `{"path": ..., "sha256": ...}`),
`id_column`, `domain`, `output` (new directory), `data_license`, `provenance`.
`domain` contains `ring` (an array of `[x,y]` pairs including the repeated closing
vertex) and either `crs_sha256` equal to the verified source PRJ digest, or an
input `crs` plus explicit `edge_segments`. The latter form transforms the
domain in Rust through the existing PROJ binding, with network access disabled
and visualization axis order (longitude/latitude for geographic input). Each
straight input edge is uniformly split into the requested number of segments
before transformation. `edge_segments` is 1..256 and the sampled ring has at
most 4096 edges, bounding native simple-ring validation and allocation.
The resulting source-CRS straight segments define the selection boundary;
sampling does not promise a projection error bound. The prep caller chooses
sampling in its request metadata. Original input CRS, input coordinate words,
sampling count and transformed coordinate words are retained in the sealed
`domain_projection` record. Both input and transformed rings must be simple.
The source-digest form retains its existing output bytes and needs no PROJ
transform. Mixing the two CRS forms refuses rather than guessing units.
There is no implicit geodesic edge, longitude wrapping, hole or multipart-domain handling.
Self-crossing, overlapping or zero-area rings refuse because their interior is
ambiguous. Intersections include boundary contacts and segments crossing the
polygon with both endpoints outside. A bounding-box overlap alone is insufficient.
Polyline and PolyLineZ XY use finite f64 straight-segment predicates; no tolerance
or snapping is applied. This is not an exact-predicate computational geometry
implementation for arbitrary near-degenerate floating-point inputs.

The output `selection.json` (`arwen.source-domain-selection.v1`) retains sorted
source IDs, matched physical record indices, source/CRS digests, the domain,
licence and provenance, plus deleted/NULL counts. No intersection refuses with
an explicit coverage gap. Intersections within a supplied source subset do not
establish that tiles cover the entire domain; the source catalogue/prep owner
must establish that separately. The tool reads records and intersects independent
polylines on Rayon, without duplicating the source SHP buffer.

Pass `selection: {"path": ".../selection.json", "sha256": "..."}` to
`subset-source-network`, omitting both outlet keys. It verifies the manifest seal
and schema, requires distinct positive IDs and actual source attribute membership,
then selects the union of their upstream closures. Spatial seeds are **not outlet
cuts**: every original arc within the closure remains, even between seeds. Original
arcs leaving the selected set remain explicit boundary evidence. Scalar/list
outlet behavior is unchanged. This stage supplies topology only; incomplete tile
coverage, divide geometry, parameters, initial stores, the shared world writer
and runtime-fabric admission remain separate requirements.

### Projected original-source XY and planar area

`rw_hydrofabric project-source-geometry CONFIG.json` consumes the original
`shp`, `shx`, `dbf`, `prj` and a `network` subset (each `{ "path": ..., "sha256": ... }`),
plus `id_column`, `target_crs`, `output` (new directory), `data_license` and
`provenance`. The network schema is `arwen.source-network-subset.v1` with distinct
positive `node_ids`. All five input seals and the physical DBF/SHP record join
are checked. Original physical records, parts and vertex direction are retained;
deleted/NULL geometries are excluded and unmatched IDs remain explicit gaps.
No coordinates are simplified or densified. Source Z/M is explicitly excluded.

The output `projected.json` is `arwen.source-projected-geometry.v1`, an
intermediate for the shared HYDRO-WORLD builder. It is not a second GeoPackage
writer or an admitted runtime fabric. `records[].parts_xy_f64_bits` contains
arrays of `[x_bits,y_bits]`, exact unsigned-u64 IEEE-754 words. Decode them with
Rust `f64::from_bits`; do not convert those integers through a floating JSON
number type. `physical_record` is zero-based. Numerical diagnostic area fields
have authoritative `signed_ring_area_m2_bits` and `source_winding_area_m2_bits`
companions for the same reason: a generic decimal JSON float parser can change
the original words.

Offline PROJ transforms use visualization XY order and isolated contexts in
bounded Rayon tasks. The target must have two Cartesian metre axes; angular
and foot axes refuse. EPSG:5070 is the adopted CONUS equal-area convention.
Other accepted projected metre CRSs imply planar area only. Polygon areas use
translated-origin signed shoelaces, then the negative ring sum under the SHP
clockwise-exterior/counter-clockwise-interior convention. Open, zero-area,
nonfinite rings and nonpositive total areas refuse. Ring containment, general
geometry validity, geodesic flowpath length, vertical datum, native CFE area
conventions, tile coverage, physical parameters and water stores still require
separate assembly/admission. Missing polygons are never assigned zero area.

`rw_hydrofabric replay-source-projection --strict PROJECTED.json NATIVE.csv`
compares projected XY words and sequences against independent GDAL WKB
(`source_id,wkb` header; hexadecimal WKB from `ST_AsBinary`). Polygon ring
rotation/winding and group order are normalized; directed line order is retained.
A discrepancy exits 2. This comparison proves coordinates, not area reduction
identity or containment. One-worker and multi-worker payloads are deterministic;
stage timings, context counts and source revision are outside the payload.
GDAL remains a qualification referee, not a production dependency.

### Offline primitive NetCDF ID-row subset

`cargo build --release --features source-netcdf --bin rw_hydrofabric_netcdf_subset`
from this crate builds a separate system-NetCDF-C acquisition tool. It does not
add NetCDF/HDF5 dependencies to the production raw-buffer bridge or the normal
GIS CLI. Run `rw_hydrofabric_netcdf_subset JOB.json` with this metadata:

```json
{
  "source": "source.nc",
  "sha256": "PINNED_FULL_FILE_SHA256",
  "source_url": "SOURCE_URL",
  "source_version": "ACTUAL_SOURCE_VERSION",
  "source_terms": "REVIEWED_TERMS_AND_ADMISSION_SCOPE",
  "selection_manifest": "source-network.json",
  "selection_sha256": "PINNED_SELECTION_SHA256",
  "ids_key": "node_ids",
  "id_variable": "link",
  "output": "NEW_SUBSET_DIRECTORY"
}
```

The rank-one ID variable must be signed integer. Selection IDs are unique i64
JSON metadata. The tool independently scans source IDs, rejects duplicates for
selected IDs, and records all missing IDs without filling or replicating rows.
Primitive row-bearing variables retain native little-endian types, C order,
all trailing/preceding dimensions, fixed-width characters and exact attribute
bytes. Requested-ID order is retained for matched rows. Variables without exactly
one ID dimension are explicitly listed as skipped; nothing is broadcast.
Packed values, fill words and units are preserved without interpretation.
Unsupported/string/compound types refuse rather than silently convert. ID scan
and row gather use all Rayon workers; NetCDF-C/HDF5 I/O is serialized for safety.
The manifest reports each stage and the source library/version/hash authorities.

Output schema `arwen.source-netcdf-subset.v1` is a **source audit**, not the
resident hydrofabric ABI, a native-capture importer, an initializer, a HYFeatures
crosswalk or a GWBUCK-to-CFE mapping. Callers must separately establish source
terms, model parameter compatibility, network closure and initialization policy.
The tool creates a distinct output directory and refuses an existing one.
# Original-source divide coverage evidence

`rw_hydrofabric audit-source-divides CONFIG.json` joins a pinned source network,
physical Polygon SHP/SHX/DBF records and a pinned attribute DBF. It reports every
selected routing ID in sorted order, original area text, exact parsed area words,
physical row numbers and explicitly selected evidence fields. Configuration:

```json
{
  "network": {"path": "network.json", "sha256": "..."},
  "polygons": {
    "shp": {"path": "Catchment.shp", "sha256": "..."},
    "shx": {"path": "Catchment.shx", "sha256": "..."},
    "dbf": {"path": "Catchment.dbf", "sha256": "..."},
    "id_column": "FEATUREID"
  },
  "attributes": {
    "dbf": {"path": "PlusFlowlineVAA.dbf", "sha256": "..."},
    "id_column": "ComID",
    "local_area_column": "AreaSqKM",
    "local_area_unit": "km2",
    "evidence_columns": ["TotDASqKM", "Divergence", "Fcode", "WBAreaType"]
  },
  "data_license": "Actual original-source terms and attribution",
  "provenance": {"source": "Pinned federal source archives"}
}
```

The four classifications distinguish present/absent polygons and positive/zero
local source areas. Source digest mismatches, missing/duplicate selected attribute
IDs, duplicate selected polygon IDs, invalid geometry and blank/negative/nonfinite
areas fail. Deleted DBF rows retain their physical positions; NULL geometry is
reported as absent. Positive-area gaps and polygons paired with zero source area
remain visible findings; this diagnostic does not impose a runtime gate.

Use the source **local incremental area** column. `TotDASqKM` is a separate
upstream drainage quantity and cannot substitute for it. A reach without its own
local catchment can still carry upstream flow. Keep its routing ID and arcs; do
not invent a polygon, drop the reach, or turn its total drainage into a runoff
source. The shared `arwen.hydro-world.fabric.v1` contract can represent a flowpath
with `has_divide=false` and a NULL `divide_id`; final assembly must reconcile this
evidence with its own source/geometry admission. This audit does not prove full
tile/domain coverage, polygon containment/union, disjoint runoff budgets, native
CFE area conventions, physical parameters/state or default-on runtime readiness.

## Original-source polygon associations

`rw_hydrofabric group-source-polygons CONFIG.json` consumes the pinned output
of `project-source-geometry` and appends exterior/interior ring associations:

```json
{
  "projected": {"path": "projected.json", "sha256": "..."},
  "output": "NEW_GROUPED_DIRECTORY"
}
```

The `arwen.source-polygon-groups.v1` intermediate retains source IDs, physical
records, missing routing IDs, provenance and all original XY/area words. Each
record's `polygons` lists `exterior_ring_index`, `interior_ring_indices` and
`planar_area_m2_bits`; indices address its original `parts_xy_f64_bits`. No ring
is simplified, reversed, rotated, duplicated or dropped in production. Decimal
area fields remain diagnostic; their `_bits` companions are authoritative.

Under the SHP specification's clockwise-exterior/counter-clockwise-interior
convention, holes belong to the smallest exterior containing all their
non-boundary vertices, with at least one interior vertex. Independent records
run on Rayon. Open/nonfinite/zero-area rings, inconsistent authoritative areas,
orphan holes, ambiguous equal-area containers and inconsistent ID rosters fail
before creating output. Source rings must be simple, noncrossing and correctly
wound: these f64 predicates do not constitute a geometry validity test, exact
predicate library, intersection repair or inter-feature overlap audit.

`rw_hydrofabric replay-source-polygons --strict GROUPED.json NATIVE.csv` compares
the independent GDAL `source_id,wkb` capture while retaining exterior/hole and
polygon associations. It normalizes only ring start/direction, hole order and
polygon order. The earlier projection comparator flattened all rings; its
coordinate proof alone did not establish these associations. The new referee
rejects repeated/dropped/out-of-range ring indices and duplicate IDs.

This is input for the existing `arwen.hydro-world.fabric.v1` shared builder, not
a competing GeoPackage writer or default-on provider. Shared writer/checker
admission, full source coverage, geometry validity/disjoint runoff budgets,
geodesic flowpath lengths, native model area/parameters and physical stores
remain separate. Missing divides never remove routing reaches or their arcs.

### Original-source geodesic flowpath lengths

`rw_hydrofabric measure-source-lengths CONFIG.json` reads pinned original SHP,
SHX, DBF, PRJ and source-network inputs. The config uses the same source objects,
`id_column`, `output`, `data_license` and `provenance` as source projection, with
no planar target CRS. The fixed length contract is WGS84 (EPSG:4326): source
coordinates are transformed through existing PROJ with network disabled and
normalized longitude/latitude axes. Each consecutive segment is measured on the
WGS84 ellipsoid through PROJ's public GeographicLib C API. Z/M are excluded;
there is no sphere, planar substitute, densification or artificial connection
between disconnected source parts. Sequential segment and part sums are stable
across Rayon worker counts. Positive source IDs are unique; null/missing shapes
stay in the missing-ID roster. Zero-total paths, invalid coordinates, inconsistent
DBF/SHP counts and duplicate matched IDs refuse before output creation.

`arwen.source-geodesic-lengths.v1` retains source hashes, source CRS, physical
record indices, IDs, per-part vertex counts and exact `part_lengths_m_bits` /
`length_m_bits` u64 IEEE754 words. The source intermediate is joined by source ID
through the existing shared writer. It is not a collapsed-network routing length,
native routing parameter, full coverage or runtime provider admission. The
bundled PROJ/embedded database and source pins belong in the receipt; offline NAD83 to
WGS84 transformation is not a claim of survey-grade datum accuracy.

`rw_hydrofabric replay-source-lengths --strict LENGTHS.json NATIVE.csv` compares
all ID/part/vertex counts and length words. Native CSV header:
`source_id,length_m_bits,part_lengths_m_bits,part_point_counts`; part lists are
semicolon-separated decimal integers. Native GDAL must read the original source,
transform to WGS84 and call `GeodesicLength`, rather than deriving reference
lengths from the candidate. The two pipelines share PROJ's geodesic math library;
their exact agreement verifies the source pipeline, not an independent geodesic
algorithm. No tolerance or float rounding is used by this comparator.

### Source polygonal-water-feature association

`rw_hydrofabric join-source-waterbodies CONFIG.json` associates a sealed
`arwen.source-network-subset.v1` roster with original source DBFs. Configuration
names SHA-pinned `network`, `flowpaths` and `waterbodies` (`path`, `sha256`),
`flowpath_id_column`, `reference_column`, `waterbody_id_column`,
`attribute_columns`, exact-text `no_reference_values` and
`unresolved_reference_values`, plus `data_license`, `provenance` and a new
`output` directory. Source names, columns and status values are table data;
there is no source-specific code path. Independent table reads and fixed-width
row decoding use Rayon. The deterministic output is `waterbodies.json`.

The original NHDPlus mapping is COMID/WBAREACOMI -> NHDWaterbody.COMID.
The public-source catalog's `nhdplus-ca18-waterbodies` recipe acquires only the
regional waterbody DBF from the already pinned original NHDSnapshot archive.
See the [EPA NHDPlusV2 User Guide](https://www.epa.gov/system/files/documents/2023-04/NHDPlusV2_User_Guide.pdf),
printed pages 160–161. WBAREACOMI describes a polygonal water-feature association;
NHDWaterbody includes several feature types. Preserve FTYPE/FCODE instead of
assuming every associated feature is a lake or reservoir. In this mapping zero
is explicitly configured as no reference; -9998 means missing/undetermined and
is configured separately as unresolved. Neither is converted to a waterbody ID.

Positive source references remain in `flowpath_waterbody_ids` even when their
target is absent; `missing_waterbody_ids` lists those targets, while
`unresolved_reference_node_ids` lists unknown source references. Either gap
makes `complete_reference_join` false. All selected source reference text and
all requested target attribute text remain verbatim. Repeated/nonpositive IDs,
missing selected flowpaths, bad source hashes, ambiguous column/status maps and
unclassified reference values fail before output creation. No sorting operation
reassigns a source ID. Output groups are sorted by ID and identical across Rayon
worker counts. This is an input crosswalk for the existing shared fabric
builder: no outlet selection, lake geometry, hydraulic parameters, storage,
source repair, CONUS coverage or runtime admission is inferred.

The optional `additional_waterbodies` array names further SHA-pinned DBFs using
the same ID and attribute column map. This joins the union of polygon tables,
not a first-match fallback: any ID repeated within or across tables is refused
before output, even when its attributes agree or no selected flowpath uses it.
With additional tables, each joined row carries `source_index`: 0 names
`sources.waterbodies`, and 1..N name `sources.additional_waterbodies[0..N-1]`.
Single-table configurations (including an empty array) retain the v1 byte
contract without these extra fields. Reordering table inputs changes only the
source provenance/index, not the ID-to-attribute association.

For original NHDPlus sources, use the `nhdplus-ca18-waterfeatures` extraction
recipe and configure NHDWaterbody as the primary table, NHDArea as an additional
table. Both are in the same pinned NHDSnapshot archive; only those two DBFs
are extracted. The guide's printed page 162 documents NHDArea, which also
includes polygonal water features such as StreamRiver. A successful union join
does not turn these features into lakes. Unknown source references remain
unresolved even when all positive IDs join; no reference is dropped or repaired.

To retain the original geometry for those associations, the
`nhdplus-ca18-watergeometry` recipe extracts only SHP/SHX/DBF/PRJ for both
tables. In `subset-source-geometry`, set `network` to the sealed
`arwen.source-waterbody-crosswalk.v1` and `waterbody_source_index` to its
source index. The command requires Polygon or PolygonZ geometry and verifies that the
configured DBF digest and ID column match that exact crosswalk source. It
selects the crosswalk's referenced polygon IDs directly; it does not treat
flowpath IDs as polygon IDs or fabricate a network roster. Ambiguous IDs or
source indices refuse before output. The manifest's `selection` records this
scope; its legacy matched/missing node lists contain polygon IDs in this mode.
Without `waterbody_source_index`, the existing network-roster behavior stays
unchanged. Geometry preserves source records and coordinates; it establishes
no outlet, storage, lake classification or hydraulic parameters.

PolygonZ selection preserves raw Z/optional-M arrays and recomputes selected
header extrema. Polygon replay independently compares XY only; raw record
identity checks must separately cover Z/M bytes. No Z datum interpretation or
vertical transform is implied. `project-source-geometry` accepts the same
`waterbody_source_index` and sealed crosswalk, using the shared selector and
source binding. Polygon and PolygonZ both use their XY rings for projection and
planar area; validated source Z/M arrays are excluded, never interpreted as
elevation or storage. `group-source-polygons` retains the original shape type
while associating the projected XY exterior/hole rings. Its independent GDAL
replay checks those associations and coordinate words, not vertical data or
physical lake area/storage. Legacy network-roster projection is unchanged.

`source-waterbody-boundaries CONFIG.json` derives graph evidence for the shared
fabric preparation path. The configuration contains SHA-pinned `network` and
`crosswalk` sources (`path`, `sha256`) and a new `output` directory. The crosswalk
must bind that exact source-network digest and roster; its association/status
partition, feature membership and completeness must agree. All validation
precedes output creation. The command writes `waterbody-boundaries.json` with
verbatim joined source features, original source gaps, sorted internal/inlet/exit
connections, domain-boundary connections and members without a recorded
successor. Missing positive-reference targets remain rows with a null source
feature. Independent source reads and per-feature classification use Rayon;
adjacency indices avoid scanning the entire network for each feature.

Every divergence is retained. Explicit domain cuts stay cuts even if both IDs
are selected or associated with the same feature. Inlets describe edges within
the selected network; the source subset does not supply incoming edges from
outside it. A member without a recorded successor is an evidence gap, not an
inferred terminal reservoir. Graph exits do not select a physical lake outlet,
classify a reservoir, supply stage/storage parameters or admit a runtime fabric.

### Source divergence and morphology evidence

The catalog's `nhdplus-ca18-water-evidence` recipe selects the original
PlusFlowlineVAA, DivFracMP, PlusWaterbodyLakeMorphology and
PlusFlowlineLakeMorphology DBFs from the pinned CA18 attributes archive.
Use the existing archive planner/acquirer and `inspect-source-dbf` /
`query-source-dbf` commands; no separate fabric writer is introduced.
Query values match source lexical text (VAA numeric IDs use scientific notation).

[The EPA NHDPlusV2 User Guide](https://www.epa.gov/system/files/documents/2023-04/NHDPlusV2_User_Guide.pdf),
printed pages 48, 51–53, describes these attributes. Divergence labels identify
network paths, not physical reservoir outlets. A DivFrac value of -9998 means
the fractional split is unknown; default main/minor accumulation is not an
observed split. Lake morphology supports time-of-travel calculations and can
use estimated depth. LakeFract allocates waterbody area among flowlines, not
discharge among outlets. Missing rows remain explicit table coverage gaps.
These source values do not initialize water storage or supply stage-storage
curves, dam outlets or operating rules.
