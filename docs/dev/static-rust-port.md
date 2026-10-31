# Porting `woof/static/*` to Rust: design (lane/static-rust-port)

Design lane output, 2026-08-17.  Base: `integration/release-2.5.0`
@ `1fca7ca56`.  Skeleton crate, seam stub and parity harness are
committed on `lane/static-rust-port`; three port lanes branch from it
with the disjoint file ownership in section 5.

Governing laws: the Python boundary (data-path processing is Rust;
Python keeps orchestration/CLI), fixed-means-default (the Rust path
ships default-on; any Python fallback is a reported workaround),
never-bit-exact-to-a-bug (defined behaviour where the reference is an
accident, documented divergence with evidence), verify against the
artifact (the parity harness drives the real cdylib through the real
ctypes ladder), no case names in generic code.

## 1. What `woof/static` is today (6,435 lines, measured)

| file | LOC | role |
|---|---|---|
| `build.py` | 1,459 | The geogrid-equivalent builder: float32 WPS sampling twins (`_WpsLambert32/_WpsMerc32/_WpsPs32/_TranslatedWps32`), WPS interpolators (`four_pt`, `average_4pt`, `average_16pt`, `sixteen_pt`, `search`), smoothers (`smth_desmth_special`), landmask/dominant rules, `_DomainSampler` (halo-extended grid, pixel→cell `nint` binning, gcell means, categorical fractions, mandatory-coverage receipts), `monthly_interp_to_date`, `GeogSelection` (WPS `geog_data_res` token resolution), `build_static` / `build_static_for_domain` (+ process cache). |
| `highres_fetch.py` | 1,027 | Footprint-parametric fetch/cache for the high-res sources: staged 1×1° elevation tiles, near-global DEM COGs (two named sources), the annual CONUS land-cover bundle, soil WCS windows; sha256 sidecars, resumable `.partial` staging, tile-id enumeration, `CoverageError` refusals; derived windows (mosaic/clip/fill → GeoTIFF) cached by input digest. |
| `highres_production.py` | 856 | `[static.highres]` TOML surface, mode resolution (all/terrain/auto), refusal gates (projected-grid type, MODIS-21 inventory, absent-tile-over-land, zero-cells-replaced), receipts, `on_refuse` policy, `refuse_inert_highres`. |
| `highres.py` | 824 | The overlay science: `BoundRaster` (hash-verified sources), CRS construction for the model grid, `resample_continuous` / `resample_mapped_categories` / `_resample_category_array` (rasterio warp), NLCD→MODIS-21 crosswalk table, `usda_texture_category` triangle, `_soilgrids_categories` depth means, `_nearest_donors` BFS, `merge_terrain_override` / `merge_highres_overrides` (donor fill, TMN lapse, deep-soil gates). |
| `projection.py` | 715 | `ProjectedGrid` base (registration, staggered arrays, MAPFAC/Coriolis/rotation assembly, `nest`, `translated` with exact integer-offset delegation), `MercatorGrid`, `PolarStereoGrid`, namelist/config dispatch (`grids_from_wps_namelist`, `grids_from_projection_config`), `_parse_wps_namelist`. |
| `corridor.py` | 709 | Sealed child-resolution statics corridor: geometry, cost, corridor grid, identity probes, deterministic NPZ seal, receipt sealing/verification, `crop`, `corridor_footprint_statics_builder` seam. |
| `geog.py` | 543 | WPS_GEOG readers: `index` parse (`GeogIndex`), tile inventory with staged/sparse/global extent rules, tile IO (wordsize/sign/endian/border/z-padding/row-order), window mosaic, coverage masks, `missing_tiles` / `required_tile_origins`. |
| `lambert.py` | 176 | `LambertGrid` (set_lc/ijll_lc/llij_lc transcription, map factor, rotation) + compat re-exports. |
| `geog_stack.py` | 103 | rasterio/pyproj presence probe + the one remedy string (doctor surface). |
| `highres_refusal.py` | 21 | Import-free `HighresRefusal` leaf for the CLI except-clause. |
| `__init__.py` | 2 | docstring only. |

### Entry points the rest of woof consumes (signatures frozen)

56 modules import `woof.static.*`.  The surface that must not move:

- `build.build_static(grid, geog_root, halo=3, *, selection=None, source_coverage_report=None, timing_report=None) -> dict[str, np.ndarray]`: `mapped_direct`, `corridor`, tests.
- `build.build_static_for_domain(grid, catalog, domain_id, *, timing_report=None)`: `era5_direct`, `ingest/nest_init`, `ingest/nest_spawn_init`, `ingest/relocation_init` (+ the read-only process cache semantics).
- `build.GeogSelection` (`fallback` / `from_tokens` / `from_case_data`, `.path()`, `.landuse_global_attrs()`), `build.geog_selection_from_catalog`: `gfs_direct`, `hrrr_native_static`, `mapped_direct`.
- `build.monthly_interp_to_date(monthly, valid_time)`: `da/nested_forecast`, `ingest/hrrr_physics`.
- `geog.GeogDataset` (`.tile_coverage_mask`, `.read_window`, `.latlon_to_xy`, index attributes), `geog.tile_coverage_mask`: `ingest/preflight`, `hrrr_native_static`.
- `projection.ProjectedGrid` + `LambertGrid`/`MercatorGrid`/`PolarStereoGrid` (`ij_to_latlon`, `latlon_to_ij`, `latlon_mass/u/v/c`, `mapfac_*`, `coriolis_m`, `rotation_*`, `nest`, `translated`), `grids_from_wps_namelist`, `grids_from_projection_config`, `projection_class`, `WRF_MAP_PROJ_CODES`: wizard, runtime, wrf_direct, obs geometry, verify cases.
- `corridor.*` (the full `__all__`): `source_hierarchy`, `hrrr_hierarchy_direct`, `prepared_domain_tree_forecast`, `runplan`, `go_cli`.
- `highres_production.apply_highres_statics`, `parse_static_table`, `refuse_inert_highres`, `HighresRefusal`: `runtime`, `case_data`, CLI, adapters.

### Output contract (`build_static` fields; all float64, mass grid `(e_sn-1, e_we-1)`)

| field | shape | build rule (arbitrated against pinned geogrid output) |
|---|---|---|
| `HGT_M` | (ny,nx) | gcell(4.0)→four_pt→average_4pt, fill 0, one `smth_desmth_special` pass |
| `LANDUSEF` | (21,ny,nx) | fractional gcell + four_pt, f32-reciprocal normalized |
| `LANDMASK` | (ny,nx) | water fraction (iswater+islake) ≥ 0.5 → 0 else 1 |
| `LU_INDEX` | (ny,nx) | dominant water type over water (lake beats ocean only strictly), dominant land type over land |
| `SOILCTOP`/`SOILCBOT` | (16,ny,nx) | four_pt-only categorical |
| `SCT_DOM`/`SCB_DOM` | (ny,nx) | argmax+1, lowest wins ties |
| `GREENFRAC`, `LAI12M` | (12,ny,nx) | four_pt→average_4pt→average_16pt→search, water-masked 0 |
| `ALBEDO12M` | (12,ny,nx) | same sequence, fill 8, water-masked 8 |
| `SNOALB` | (ny,nx) | month-1 plane, water-masked 0 |
| `SOILTEMP` | (ny,nx) | sixteen_pt-led sequence, water-masked 0 |
| `TMN` | (ny,nx) | `SOILTEMP - 0.0065*HGT_M` on land only |

Plus per-field `gpuwm-geog-source-coverage-v1` receipts and the WRF
land-use global attributes from the selected index
(`MMINLU/ISWATER/ISLAKE/ISICE/ISURBAN`).  The corridor validates this
inventory against `NATIVE_STATIC_REQUIRED` in `native_wrf_contract.py`.

### Source datasets read

- **WPS_GEOG trees** (binary tiles + ASCII `index`): the default
  30-arc-second inventory (terrain, 21-class land use, top/bottom soil
  type, greenfrac, LAI 10m/30s, albedo, max snow albedo, 1° soil
  temperature) and the `5m` global inventory selected by WPS tokens.
  Reference tree verified present at `$GPUWM_STATIC_PARITY_GEOG`
  (`<case data root>/WRF_1974_MP55_reference_bundle/static/WPS_GEOG` by
  default; 30s inventory complete, the 5m directories are NOT staged
  there).
  Tile format: `XSTART-XEND.YSTART-YEND`, 1/2/4-byte integers,
  big-endian default, optional border halo, optional z planes,
  scale/missing on raw values, `top_bottom` row-order flip, staged
  (footprint-minimized) trees with global geometry preserved.
- **High-res endpoints** (`highres_fetch.py`): staged 1×1° elevation
  GeoTIFF tiles and the two near-global DEM COG mirrors (f32,
  deflate); the annual CONUS land-cover zip→GeoTIFF (u8, Albers/NAD83);
  soil WCS GeoTIFF windows (i16, Interrupted Goode Homolosine via
  recorded `crs_override`, scale 0.1, nodata 0).

## 2. Crate + seam design

**Crate**: `tools/rustwx/crates/static-fields` (workspace member;
edition 2024, rust-version 1.85; deps: rayon, serde, serde_json from
the workspace table).  `crate-type = ["cdylib", "rlib"]`, library name
`static_fields`.  Lane 3 declares its GeoTIFF decode dependencies when
it lands them (candidates: `tiff` + an inflate crate, or a vendored
minimal reader, record the choice and provenance in `VENDOR.md`).

**Seam**: ctypes cdylib, the `netcdf-writer` pattern exactly, NOT
pyo3 (one loading discipline across every woof bridge, no
per-interpreter builds).  C ABI prefix `gpuwm_static_*`:

- plumbing (real today): `gpuwm_static_abi_version` (=1),
  `gpuwm_static_source_rev` (release-cut stamp),
  `gpuwm_static_last_error`;
- lane 1: `grid_new` (GridSpec JSON), `grid_nest`, `grid_translated`,
  `grid_free`, `grid_array` (stagger × {lat, lon, mapfac, F, E,
  sinalpha, cosalpha} into caller-allocated f64), `grid_transform`
  (bulk point transforms), `grid_identity_probes` (corridor JSON);
- lane 2: `build_fields` (grid handle + nine resolved GEOG paths JSON →
  fieldset handle), `fieldset_len/name`, `field_dims/read`,
  `field_coverage_json`, `fieldset_free`.  **`gpuwm_static_build_fields`
  is the ABI marker** (registered in `woof/bridges.py`);
- lane 3: `highres_terrain`, `highres_overrides`, `highres_merge`,
  `highres_derive_window` (request JSON; fieldset handles in/out).

Array data crosses as raw native-endian f64 (`numpy.tobytes()`
convention); config crosses as UTF-8 JSON.  Handles are opaque u64 into
process-global registries.

**Python bridge**: `woof/static/rust_bridge.py` (committed), the
nc-writer ladder verbatim: `WOOF_STATIC_BRIDGE` override → crate
release/debug targets → `libexec/bridges` → wheel-bundled → user
default; ABI probe before first call; helpers `grid_new`,
`build_fields`, `fieldset_to_dict`.

### Default-on wiring (per lane, at landing)

Public Python signatures do not move.  Each entry's body becomes: Rust
by default; the numpy implementation runs only under
`WOOF_STATIC_PYTHON=1` or when the library is unloadable, and every
such run is a reported workaround (receipt field + console line), never
silent.  `woof doctor` gains a `static builder` row via the existing
bridge-marker machinery.

### What stays Python, and why (boundary-law accounting)

- **`GeogSelection` + namelist/experiment dispatch** (`geog_data_res`
  tokens, `_parse_wps_namelist`, `grids_from_*`): config resolution.
  The nine RESOLVED paths cross the seam; no tile byte is touched in
  Python.
- **`highres_fetch` network driver** (URL loops, resumable staging,
  sha256 sidecars, cache admission): orchestration.  The bytes it moves
  are opaque payloads written to disk verbatim; every byte is
  subsequently decoded, mosaicked, warped and merged in Rust
  (`highres_derive_window` + the lane-3 calls).  Tile-id enumeration
  and footprint bbox math move to Rust so the driver computes no
  geography.
- **`highres_production` policy shell** (TOML validation, refusal
  routing, `on_refuse`, receipts, console lines) and
  **`corridor` receipt/digest verification**: orchestration and audit
  documents.  Corridor FIELD bytes, geometry, crop and the
  deterministic NPZ seal are Rust.
- **`geog_stack`** doctor probe text (done: it names the bridge as the
  engine and rasterio/pyproj as the fallback's libraries, and
  `missing_highres_engine` is the pre-fetch gate) and
  **`highres_refusal`** (import-free leaf).
- **`build_static_for_domain`'s process cache** (read-only memo keyed
  by geometry+selection): lifetime management around the seam.

Everything else in `woof/static` (every array, every interpolation,
every warp, every merge) is Rust after the port.

## 3. Parity contract

**WPS path (lanes 1–2): byte-identical.**  The Python is the oracle
(itself pinned to geogrid v4.6.0 output and the committed llxy oracle
fixtures by the existing suites, so WRF arbitration carries over
transitively).  Equality = `np.testing.assert_array_equal` on every
field of section 1's table, on the harness domains (mid-latitude
Lambert parent + sub-km nest, Mercator and polar smoke domains) over
the reference WPS_GEOG tree.  All the float32 stencil-selection
behaviour is in scope: the twins' f32 arithmetic, the GNU-scalar ULP
nudges, the compiler-band reconciliations, the half-cell 5e-5 snap,
the f32 reciprocal in categorical normalization, `search`'s BFS
frontier order.  These are DEFINED behaviour for this port (they are
pinned by shipped geo_em bytes), not accidents, never-bit-exact-to-a-
bug does not license changing them.

**The libm risk (named, with the mitigation).**  numpy evaluates f64
transcendentals through the platform libm/SLEEF; Rust `std` routes to
its own.  A one-ULP `tan` difference can move a pole solution.  Rule
for lanes 1–2: reproduce numpy's bit results on the harness domains;
where the platform disagrees, evaluate through an explicit
deterministic implementation (the `libm` crate, or the exact operation
sequence numpy uses) until the harness is byte-green; any residual
mismatch is escalated with evidence (per-value ULP map, the
`test_projection_oracle.py` precedent), never averaged away.  The
harness compares arrays, so any drift is caught at the first field.

**Platform qualification**: the committed lane-1 fixtures are the
Windows/NumPy 2.2.6 reference. Linux has remaining exact arithmetic
failures against independently generated native Python output; the
Windows pass is not a cross-platform qualification. See
[the measured platform results and reproduction commands](static-platform-qualification.md).

**Corridor cross-implementation note**: a corridor SEALED by a
pre-port preparation embeds `grid_identity_probes` computed by Python
floats; the port-era runner recomputes them through the default (Rust)
path.  The runner compares the probes within
`woof.static.grid_identity.GRID_POSITION_TOLERANCE_CELLS` of a cell
(`corridor.grid_probe_drift`), not to the bit, because two machines or
two implementations of the same projection may round a position
differently in the last digit.  A probe mismatch therefore means the
corridor was prepared on a grid that lies somewhere else; the remedy
is re-preparation, and the refusal text says so.

**Highres path (lane 3): split contract.**
- Byte-identical: `usda_texture_category`, the crosswalk mapping,
  `_nearest_donors` (BFS seed/push order is the defined tie-break),
  both merges (fills, sentinel discipline, TMN lapse, audit counts),
  every refusal DECISION (which gate fires on which input).
- Defined-behaviour tolerance: the mosaic and the warped planes.
  GDAL's warper is a black box, not a spec; `raster::warp` documents
  its own rule (every valid source pixel centre mapping into a
  destination cell contributes equally; bilinear fallback for
  unreached cells) and the harness measures against rasterio output on
  a pinned footprint with recorded thresholds (terrain: max |Δ| and
  mean |Δ| bounds; fractions: per-cell L1 bound + exact normalization).
  The measured divergence ships in the receipt (`method` string): a
  documented divergence with evidence, per the law.
- The CRS transforms (`raster::transform_points`) are validated
  against the Python geodesy stack on fixture point sets to ≤ 1e-6 m
  BEFORE any warp uses them (validate-the-instrument).

**Speed** (ruling: "make it speedy"): rayon over row chunks with
deterministic reduction order everywhere (bit-stable run to run and
equal to the serial result).  The Prove lane measures Python-vs-Rust
wall time on real domains; the standing measurement to beat is
`build_static` = 14.1 s of a 36.0 s prepare (2026-08-16, documented in
`build.py`).  The parity harness pins only that both paths are
runnable in one process for the measurement.

## 4. Skeleton committed on this branch

- `tools/rustwx/Cargo.toml`: member `crates/static-fields`.
- Crate: `Cargo.toml`, `build.rs` (source-rev stamp, netcdf-writer
  pattern), `src/lib.rs` (constants), `src/error.rs`, `src/types.rs`
  (`Grid2`, `Stack3`, `Field`, `FieldSet`, `Stagger`), the shared
  floor; `src/projection/{mod,lambert,mercator,polar,wps32}.rs`,
  `src/corridor.rs`, `src/npz.rs` (lane 1 stubs);
  `src/geog/mod.rs`, `src/interp.rs`, `src/smooth.rs`,
  `src/sampler.rs`, `src/fields.rs` (lane 2 stubs);
  `src/raster/{mod,geotiff,warp}.rs`, `src/highres/mod.rs` (lane 3
  stubs); `src/capi/{mod,grid,build,highres}.rs` (seam: plumbing real,
  lane entry points refusing by name); `tests/seam_probe.rs` (green:
  version, error discipline, named refusals).
- `woof/static/rust_bridge.py` (loader + bindings + helpers),
  `woof/bridges.py` marker row.
- `tests/test_static_rust_parity.py`: RED by design (marker
  `static_rust_parity`); each lane greens its class.
- This document.

`cargo test -p static-fields` passes (3 seam tests); the cdylib builds
and loads through the ladder; the parity harness fails with the
skeleton's named refusal: verified against the artifact.

## 5. The three port lanes (disjoint file ownership)

Branch from `lane/static-rust-port`.  No lane touches another lane's
files; shared types are frozen by this skeleton (additive changes only,
coordinated through the integration captain).

**Lane 1, grid math** (~1,600 Python LOC displaced)
- Rust: `src/projection/**`, `src/corridor.rs`, `src/npz.rs`,
  `src/capi/grid.rs`.
- Python: `woof/static/projection.py`, `lambert.py`, `corridor.py`
  (route array methods/geometry through the bridge, default-on;
  namelist dispatch and receipt policy unchanged).
- Green: `TestLane1GridParity`, plus the existing `test_lambert.py`,
  `test_projection_oracle.py`, `test_statics_corridor.py` with the
  default (Rust) path, including the sealed-NPZ digest equality.

**Lane 2, tile ingest + field building core** (~2,000 LOC displaced)
- Rust: `src/geog/**`, `src/interp.rs`, `src/smooth.rs`,
  `src/sampler.rs`, `src/fields.rs`, `src/capi/build.rs`.
- Python: `woof/static/build.py`, `geog.py` (public classes stay;
  compute routes to the bridge; `GeogSelection` untouched).
- Green: `TestLane2BuildParity`, `test_static_build.py`,
  `test_hrrr_native_static.py`, `ingest/preflight` coverage suites on
  the default path.  Owns the Prove-lane timing hook.

**Lane 3, the highres family** (~2,700 LOC displaced)
- Rust: `src/raster/**`, `src/highres/**`, `src/capi/highres.rs`,
  decode-dependency vendoring + `VENDOR.md` entry.
- Python: `woof/static/highres.py`, `highres_fetch.py`,
  `highres_production.py`, `geog_stack.py` (rasterio/pyproj leave the
  runtime dependency set when the substrate lands; the doctor row and
  remedy string follow).
- Green: `TestLane3HighresParity` (written out against the landed
  substrate per the scaffold's instructions),
  `test_static_highres_production.py`,
  `test_static_highres_international.py` on the default path, plus one
  end-to-end fetch+apply against the real endpoints on a small
  footprint (verify against the artifact).

Ordering: lanes are parallel.  Lane 2 consumes lane 1's
`ProjectedGrid`/`Wps32Twin` implementations at integration but codes
against the frozen skeleton signatures meanwhile (its own tests can
drive `sampler` with fixture twins).  Lane 3 consumes lane 2's
`smooth` at integration (one function; the signature is frozen here).

## 6. Acceptance (what "done" means for the port)

1. `pytest -m static_rust_parity` green with the real cdylib, no
   fallback env set.
2. Every existing static suite green on the bare default (Rust) path.
3. A real prepare (`woof go` prepare stage on a current case config)
   runs the Rust builder by default and its receipts say so; wall time
   for `static.build_static` measured against the 14.1 s baseline and
   reported by the Prove lane.
4. Fallback runs print the workaround line and stamp the receipt.
5. `woof doctor` shows the `static_fields` bridge row with the marker
   check.

## 7. Integration status (assembled 2026-08-17)

The three lanes merged onto this branch and the default-on wiring
landed.  What routes to the crate on a bare default call today:

- **`build_static` / `build_static_for_domain`**: the whole build via
  `gpuwm_static_build_fields` (grid handle + nine resolved paths +
  halo); coverage receipts come back per field as the same JSON
  document the numpy body assembled.  A crate refusal maps onto the
  exception type the numpy body raises for the same defect (coverage
  -> `FileNotFoundError`, everything else -> `ValueError`); message
  bytes are pinned equal by the crate's golden tests.
- **`ProjectedGrid` array methods** (`latlon_{mass,u,v,c}`,
  `mapfac_{m,u,v}`, `coriolis_m`, `rotation_{m,u,v}`), via
  `gpuwm_static_grid_array` on a per-instance cached handle
  (`weakref.finalize` frees it).  Translated grids cross as reference
  handle + integer offset so the crate delegates exactly as
  `translated` documents.  Scalar `ij_to_latlon`/`latlon_to_ij` stay
  Python: orchestration-scale, and bit-equal to the crate's f64 path
  by the lane-1 numerics ledger.
- **`corridor.grid_identity_probes`**, via
  `gpuwm_static_grid_identity_probes` (shortest-round-trip floats,
  parsing back to the receipt's exact bits).
- **highres appliers**, as lane 3 landed them (USDA triangle, both
  merges, terrain/overrides seam calls).

### 7.1 The warp substrate flip (2026-08-18)

The `resample_*` entry points were the one deferral left in section 7,
and the deferral was recorded nowhere a user could see it: a source
comment in `highres.py` and a sentence in `pyproject.toml`, with no
registry entry, no doctor row, no receipt field and no env flag.  A
run with `[static.highres] enabled = true` was decoding GeoTIFFs and
warping in Python and saying nothing.  Closed:

- `resample_continuous`, `resample_mapped_categories` and the new
  `soilgrids_category_fractions` (one seam crossing for read +
  depth-weighted mean + USDA triangle + warp) route to
  `gpuwm_static_highres_resample`;
- `highres_fetch`'s `derive_terrain_window`,
  `derive_global_terrain_window` and `derive_landcover_window` route to
  `gpuwm_static_highres_derive_window`, and the SoilGrids IGH window
  snap to `gpuwm_static_highres_transform_points`.  The network driver
  itself stays Python;
- both modules report through `rust_bridge.route`, the one decision
  point, so the console WORKAROUND line and the receipt's
  `static_compute` field cannot disagree;
- the land-cover clip's two SOURCE-COVERAGE refusals cross as their own
  return code (-2) rather than as a sentence to pattern-match, because
  `on_refuse = "fallback-30s"` may answer a coverage refusal with the
  30-arc-second baseline and must never answer a decode fault that way;
- `require_geography_stack` now refuses when the engine that WOULD run
  cannot, instead of when rasterio is absent: a wheel carrying only the
  shipped default used to be refused by that gate;
- rasterio/pyproj left `[project.dependencies]` for the `[geog]` extra.
  They are the parity reference and the `WOOF_STATIC_PYTHON=1` body,
  nothing else.  `tests/test_static_highres_warp_routing.py` makes
  rasterio, pyproj and affine unimportable and requires every default
  high-resolution call to still answer.

### 7.2 What the flip's own proof found: a PixelIsPoint defect

Proving the flip on real bytes -- `woof static` on the staged
Copernicus GLO-30 tiles over a 500 m Alpine domain, once on each engine
-- surfaced a decode defect in the crate that the committed goldens
structurally could not have caught.

`src/raster/geotiff.rs` ignored `GTRasterTypeGeoKey` (1025).  Copernicus
DEM GLO-30 ships **RasterPixelIsPoint**: its tiepoint names a pixel
CENTRE, so the raster origin is half a pixel north-west of it (the
N46/E008 tile's tiepoint is exactly `(8.0, 47.0)`).  Read as a corner,
every tile sat half a pixel (~15 m) south-east, and the mosaic sampled
the wrong 30 m source pixel wherever that half-pixel crossed a boundary.
MEASURED end to end before the fix: terrain max |delta| 62.2 m, mean
|delta| 8.05 m against the rasterio path.

The goldens could not see it because they were extracted BY RASTERIO out
of the source tiles, and rasterio writes PixelIsArea with the shift
already folded into the tiepoint -- the fixture generation normalised
away the exact convention under test.  The gate is now
`test_the_mosaic_samples_the_pixel_that_contains_the_cell`, which builds
both registrations with rasterio and asserts the crate takes the source
pixel CONTAINING each destination centre.  Validated both directions:
red on the `point` arm without the fix, and the `area` arm green before
and after, so it is measuring the convention and not the weather.

**Residual, recorded rather than explained.**  After the fix the two
engines still differ on the real 4-tile Alpine window: terrain max
|delta| 49.3 m, mean 6.4 m; the derived mosaic itself mean 8.3 m
excluding the pixels rasterio leaves uncovered.  On the containing-pixel
rule the Rust side matches at every cell probed and `rasterio.merge`
does not, but `rasterio.merge` reproduces the correct answer on every
synthetic case that could be built for it (one tile and two, both
registrations, aligned and fractionally offset bounds), so the cause of
the real-window disagreement is not yet named.  Separately and
independently: the rasterio path leaves the derived window's last row
(2848 pixels) uncovered and fills it with 0 m sea level in the middle of
the Alps -- visible as `sea_level_filled_pixels: 2848` in the shipped
2026-08-14 receipt -- where the Rust mosaic covers it.

**The terrain lattice (2026-09-28).**  Both engines now cut every
terrain window on one fixed pixel lattice per source instead of at the
footprint's own west/north edge: the first tile's own grid when the
resolution is inherited, pixel centres on the whole multiples of a
declared resolution (`warp::mosaic`, and `_terrain_lattice` with
`_lattice_bounds` in `highres_fetch.py`).  Cut at the footprint's edge,
a moving nest's statics corridor and the nest itself sampled the same
ground at different sub-pixel offsets, and the nest's first move was
refused with `footprint-rebuilt statics differ`.  On the pinned seam
golden, now cut on that lattice, the Rust mosaic equals `rasterio.merge`
on the lattice-snapped bounds at every cell (45,793 of 45,793
bit-identical, no holes on either side), for the declared and the
inherited lattice alike, so the mosaic gates are exact.

The consequence for the fallback's status is stated where a reader will
meet it (`woof/static/highres.py`): `WOOF_STATIC_PYTHON=1` is NOT a
byte-parity twin on this path.  It is the reference implementation for
bisecting a difference, and a production run on it is a workaround in
the full sense.

The fallback is one shared decision point,
`woof.static.rust_bridge.route`: env flag read at call time, one
WORKAROUND console line per operation per process.  The parity
harness's oracle sides now run under the explicit fallback so the
comparison stays Rust-vs-numpy after routing.

Known duplication for a later cleanup, not a defect: lane 2's
`SamplerMesh::from_twin_outputs` (sampler.rs) and lane 1's
`wps32::sampling_surface` both assemble the compiler-band/ULP mesh.
The production build path uses the sampler.rs implementation
(`DomainSampler::new`); the wps32 one stays as the lane-1 reference,
and each is pinned bit-equal to the same Python-extracted goldens
(`sampling_surfaces_bit_equal` in lane1_goldens.rs for wps32, the
lane-2 build/mesh tests for sampler.rs), so they cannot diverge
without a golden going red.

## 8. Prove-and-finish record (2026-08-17, lane/static-rust-port-prove)

**Dual-run parity: PASS, byte-for-byte, no divergence.**  The full
`build_static` ran twice per domain in separate processes -- once under
the explicit `WOOF_STATIC_PYTHON=1` fallback, once on the bare default
(Rust) -- over the reference 30-arc-second WPS_GEOG tree, on five real
domains reusing existing test-domain configs (the reference bundle's
4-nest WPS namelist; the projection-oracle Mercator/polar smoke
domains):

| domain | cells | fields byte-equal | receipts | NPZ digest |
|---|---|---|---|---|
| Lambert parent 251x201 @ 12 km | 50,000 | 14/14 | equal | equal |
| Sub-km nest 601x601 @ 333 m | 360,000 | 14/14 | equal | equal |
| Mercator 111x89 @ 12 km | 9,680 | 14/14 | equal | equal |
| Polar stereo 111x89 @ 12 km | 9,680 | 14/14 | equal | equal |
| Statics corridor 1503x1503 @ 333 m | 2,259,009 | 14/14 | equal | equal |

Every field compared at `tobytes()` equality; coverage receipts at
canonical-JSON equality; both sides' output sealed through the
corridor's deterministic NPZ writer and compared at the file sha256.
The corridor case (the 4th domain's parent-extent corridor at child
resolution, built through `corridor_grid`'s translated-grid seam
crossing) additionally matched `grid_identity_probes` exactly.  Zero
defined-behaviour divergences were needed on the WPS path: the crate
reproduces the numpy bits everywhere the dual run looked.

**Benchmark (cold = fresh process, warm = second in-process call;
single-run accurate numbers, this box):**

| domain | Python cold | Rust cold | speedup | Python warm | Rust warm |
|---|---|---|---|---|---|
| corridor 2.26M cells | 57.10 s | 5.35 s | 10.7x | 62.46 s | 5.52 s |
| nest 360k cells | 10.08 s | 1.93 s | 5.2x | 9.72 s | 1.75 s |
| parent 50k cells | 4.97 s | 5.31 s | 0.94x | 5.06 s | 4.91 s |

Where the remaining Rust time goes: source-window ingest, not model
cells.  Rust wall time tracks the summed source-window pixel count
(parent: 78.0M source px / 114 tiles -> 5.3 s at only 50k cells;
polar: 28.4M px -> 1.7 s; Mercator: 9.4M px -> 0.9 s), because a
coarse (dx >= 1 km) domain averages every source pixel under each grid
cell.  The corridor is the opposite regime -- 2.2M source px but 2.26M
cells x 97 planes of interpolation -- and lands at the same 5.3 s.
The 12-km parent is therefore the one shape where Rust only ties the
numpy body (both are bound by the same tile decode + gcell binning
throughput); every finer-resolution build is 5-11x.  The standing
14.1 s prepare-stage measurement is a nest-class build and is now
~2 s.

**Estate closure (acceptance items 3-5).**  `static_fields` joined
`BUNDLED_ARTIFACTS` (kind library, `WOOF_STATIC_BRIDGE`,
`gpuwm_static_abi_version` in `LIBRARY_ABI`) so `woof fetch-bridges`
stages it and a wheel user reaches the default; `woof doctor` gained
the `static builder (default static-field engine)` row (verified with
the ABI handshake here; MISSING with the build remedy and the
workaround named as a workaround when absent); the bundled-artifact
coverage sweep counts it.  Bindings pinned by
`tests/test_static_rust_parity.py::TestEstateAndDoctor`.

**Suites at the proved tip**: `cargo test -p static-fields` 47 green;
`pytest tests/test_static_rust_parity.py` 13 green;
`test_static_build.py`, `test_lambert.py`, `test_projection_oracle.py`
(109), corridor/highres/native suites (95), and the estate suites
(`test_bridge_fetch.py`, `test_doctor_route_accuracy.py`,
`test_verify_release_artifacts.py`, `test_nc_writer_bridge.py`,
`test_doctor.py`, 227) all green on the bare default path with the
reference bundle present.

Evidence gallery:
the static-rust-port evidence gallery (2026-08-17)
(terrain imagery of the Rust-built nest + corridor statics through the
real `rw_wrfbatch` built from this tree; land-use imagery deliberately
absent -- the renderer's catalog has no land-use product and the render
law forbids a matplotlib substitute; matplotlib for the timing/parity
analysis charts only).

## Portable WPS float32 sampling (cross-machine moving nests)

The float32 sampling projection now uses fixed vendored libm operations
for tan, atan, atan2, asin, acos, log10 and pow. The platform NumPy
authority described above remains the reference for the float64 projection
and the other unchanged kernels, but no longer defines these float32
operations: platform libm rounding moved interpolated terrain between
Windows preparation and Linux relocation even on an unchanged grid.

The replacement WPS goldens are in
`tools/rustwx/crates/static-fields/tests/goldens/wps32/`, with their re-pin
command and reason recorded there. The Python fallback is still a reported
diagnostic path; its platform-dependent WPS values do not define the default
Rust sampling result. Windows-prepared witnesses at 2 km, 1 km and 500 m
over 30 arc-second terrain must match Linux-rebuilt overlap statics exactly.
The one-ulp overlap check and bitwise donor-alignment check remain in force;
there is no field-gradient or grid-position allowance on the statics.
