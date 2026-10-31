# Vendored GRIB decoder provenance

`grib-core/` is THE GRIB decoder for this repository -- one crate, five
consumer declarations, no second copy anywhere.  Four of those five live in the
renderer workspace and reach it by path:

| consumer | manifest | default features |
|---|---|---|
| `rustwx-io` | `tools/rustwx/crates/rustwx-io/Cargo.toml` | on |
| `rustwx-products` | `tools/rustwx/crates/rustwx-products/Cargo.toml` | on |
| `rw-obs` | `tools/rustwx/crates/rw-obs/Cargo.toml` | on |
| `rw-wrfbatch` | `tools/rustwx/crates/rw-wrfbatch/Cargo.toml` | off |
| `grib1_bridge` | `tools/grib1_bridge/Cargo.toml` | off |

**Why one copy, stated as the breakage it prevents.** Until 2026-08-16 the tree
carried two vendored `grib-core` directories, both declaring
`name = "grib-core", version = "0.1.0"`, and the copy the renderer and ingest
path used -- the one four of the five consumers pointed at -- was the copy
WITHOUT `missing_value_management`.  A complex-packed GRIB2 message (Template
5.2 or 5.3) that declares in-band missing values reserves bit patterns as
markers instead of shipping a Section-6 bitmap; a decoder that never reads
octet 23 of Section 5 hands those markers back as physical values, with no
error and no NaN, and with first-order spatial differencing every value AFTER
the first marker inherits it.  Landing a decoder fix in one copy while the
shipped path used the other is how that survived two release lines.  Do not
re-vendor a second copy; a decoder fix that has to be applied twice is a
decoder fix that will be applied once.

`grib-core/` is an export of BowEcho's ratified hardened decoder:

- source: <https://github.com/FahrenheitResearch/rusty-weather>
- upstream revision: `fe9797f86c8958b5a625a4a7682c6b6aeff6b309`
- authoritative upstream `vendor/grib-core` Git tree pin:
  `7c036c2b18c0a9bb014eddd55cdacf84a244fed0`
- crate/version: `grib-core` 0.1.0
- license: MIT, as declared by the upstream workspace `Cargo.toml` at that
  revision.  The upstream `grib-core` subtree contains no separate license
  file; this provenance note preserves the exact declaration rather than
  inventing different package metadata.

The revision is the one BowEcho pinned after July 2026 GRIB1 hardening.  The
Git tree object above is the base content identity.  gpuwm carries narrow,
auditable local deltas required by the native GFS bridge:

- `src/grib2/parser.rs` exposes Section-1 origin/table/production identity,
  the analysis/forecast generating-process ids, and the second fixed-surface
  type/value.  It also corrects negative sign-magnitude fixed-surface scale
  decoding.  These fields bind the operational NCEP GFS product and validate
  the exact 0-10, 10-40, 40-100, and 100-200 cm GFS soil slabs.  It also
  enforces exact GRIB envelope coverage, the Section 1/3/4/5/6/7 state
  machine, Section-3/5/6 cardinality, bitmap padding and grid-bound bitmap
  reuse, complete PDT 4.0 fixed-surface descriptors, and the required DRT 5.0
  original-field-type octet.  It also parses Code Table 5.5 -- octet 23 of
  Templates 5.2 and 5.3, `missing_value_management`, plus the primary and
  secondary substitutes at octets 24-31.  From the 2026-08-17 convergence it
  also reads WMO Code Table 4.4 as a duration table rather than only
  recognising hours (`statistical_time_range_seconds`, with
  `statistical_time_range_hours` derived from it and refusing to round), and
  decodes the Code Table 4.5 probability-threshold scale factor as
  sign-magnitude like the two fixed-surface scales beside it -- read as
  two's complement, `0x82` turned a 300 K threshold into 3e126.
  `second_fixed_surface()` states the 255 "no second surface" code as an
  `Option` pair for callers that model absence that way.  From the L137
  hybrid closure (2026-08-19) it also parses Section 4's optional
  coordinate list -- NV at octets 6-7 and the trailing IEEE-f32 values
  (`ProductDefinition.coordinate_values`), the channel hybrid model
  levels publish their half-level A/B coefficients through; the tail is
  located by the fixed lengths of templates 4.0/4.1/4.2, an unmodeled
  template with NV > 0 refuses by name, and a tail short of its
  declared count refuses rather than rereading template octets as
  coefficients.  A conformant Section 5 template 4 (IEEE packing) is
  read from its own precision code (1 is 32 bits, 2 is 64, 3 is the
  128-bit width the unpacker names as unimplemented, anything else
  refuses by number) instead of being routed to the simple-packing
  reader, which asked for octets template 4 does not have.
  `DECODE_TEMPLATES` lists the Section-5 templates `parse_section5` has
  a reader for, in the spelling a binary compiles into its contract
  marker, with a test that each listed template parses.
  The generic constituent delta (2026-09-30) adds PDT 4.40, 4.41, 4.42,
  4.44 and 4.48 with their own process, forecast-time and fixed-surface
  octets. Optional fields expose constituent and aerosol identifiers,
  size intervals and optical wavelength intervals. Missing interval
  bounds remain absent; scales use sign-magnitude. PDT 4.44 retains its
  two-octet forecast time. Coordinate-tail lengths cover every modeled
  layout (4.0/1/2/5/6/8/9/10/11/12/40/41/42/44/48), including repeated
  statistical specifications. Unmodeled NV tails still refuse. Existing
  no-tail templates retain their original parsing path. Tests reparse
  existing synthetic Section 4 fixtures, assert every chemical layout,
  refuse every truncation and read NV=276. WMO Manual on Codes, Volume
  I.2 tables:
  <https://github.com/wmo-im/GRIB2/blob/master/GRIB2_Template_4_40_ProductDefinitionTemplate_en.csv>,
  <https://github.com/wmo-im/GRIB2/blob/master/GRIB2_Template_4_41_ProductDefinitionTemplate_en.csv>,
  <https://github.com/wmo-im/GRIB2/blob/master/GRIB2_Template_4_42_ProductDefinitionTemplate_en.csv>,
  <https://github.com/wmo-im/GRIB2/blob/master/GRIB2_Template_4_44_ProductDefinitionTemplate_en.csv>,
  <https://github.com/wmo-im/GRIB2/blob/master/GRIB2_Template_4_48_ProductDefinitionTemplate_en.csv>.
  The 2026-10-18 local statistical-descriptor fix retains the count,
  missing-value count and every ordered 12-octet specification for all modeled
  statistical templates (4.8/9/10/11/12/42). The first-specification convenience
  fields remain. A missing length has no fixed duration. Truncated headers,
  zero ranges and incomplete declared lists refuse; NV coordinate values
  cannot stand in for descriptor bytes. This is an original local change,
  with no new upstream import or change to the MIT declaration above.
  Patched-file SHA-256:
  `b261ac24e406acc2c77f26134bcff43b00d29bd097c70073f565bf0ceadd1e59`.
- `src/grib2/mod.rs` re-exports that identification value object and
  `DECODE_TEMPLATES`, `StatisticalTimeRange`, and `MissingValueMode` / `missing_value_mode`
  beside the unpackers, and (A135, 2026-09-30) `StoredRect` and
  `unpack_message_stored_rect`.
  Patched-file SHA-256:
  `1f751bdf0f8b82f39658b8f2536f554dfc2d7796d53a36d9de4383e122bc86de`.
- `src/grib2/search.rs` initializes the new identity in its pre-existing
  synthetic test messages. `src/grib2/unpack.rs` additionally enforces
  Section-5/7 simple- and complex-packing cardinality, checked bit reads, a
  signed 63-bit simple-packing limit, and exact bitmap consumption, and it
  DECODES Code Table 5.5 modes 1 and 2: a cell whose stored pattern is the
  all-ones marker of its group's width (or, in mode 2, one below it) comes out
  as NaN, which is the same representation this decoder already uses for a
  cell a Section-6 bitmap excludes, and spatial differencing integrates over
  the present cells only so a marker cannot contaminate the values after it.
  Reserved Code Table 5.5 values are refused rather than guessed, and mode
  1/2 with `bits_per_value == 0` is refused for the zero-width groups where
  the marker and the only representable reference are the same number (see
  convergence note C4 below).  The 2026-08-17 convergence also made every
  spatial-differencing step checked (`i64` overflow is an error naming the
  order, not a debug panic or a release wrap-around published as physics),
  expanded a zero-group complex field to the Section-5 count as the constant
  field GRIB2 spells that way, and made the row-window filler refuse when
  the packed data ends early instead of leaving the window tail at its NaN
  fill.  Local patch `4450837cff8d1fd1382082568eed6126bd85cdd6`, committed
  2026-09-04, also applies Section-5/7 preconditions to the row-window entry
  point, checks simple and complex-group bit reads, and requires complex-group
  lengths to match the declared point count.  Truncated payloads now refuse
  instead of synthesizing reference-value fill.  Three regression tests cover
  a short simple payload, a bitmap that outlasts its payload, and a truncated
  complex-group payload.  This is a local delta to the upstream tree pinned
  above; the patched-file pins were rechecked on 2026-09-06.  A
  simple-packed field whose bitmap marks every point missing and whose
  Sections 5/7 carry no coded value decodes as an all-missing field
  (NaN) rather than inventing its reference value; a missing or
  malformed bitmap, or a nonempty payload, is still a decode error.
  A spatially differenced complex field (Template 5.3) is read straight
  into its dense sequence of present cells plus a mask of missing cells,
  instead of one tagged cell per grid point that was then filtered,
  re-tagged and scaled; the arithmetic is the same cell for cell and the
  refusals are raised in the same order.
  A135 (2026-09-30) adds `unpack_message_stored_rect`: a stored-order
  rectangle of a regular grid, decoded through OpenJPEG's decode area for
  a bitmap-free JPEG2000 message whose main header declares the reversible
  5/3 wavelet (the only case whose area integers are the whole decode's),
  and unpacked whole and cropped for every other message.
  3405946bb (2026-10-04) extends the all-missing case to complex packing
  (Templates 5.2 and 5.3): a field whose Section 5 declares zero coded
  values, zero groups and canonical zero descriptors, under a bitmap that
  marks every point missing (exactly the grid, or padded with zero bits
  to the final octet), decodes as all-missing; data bytes, a malformed
  bitmap or a nonempty group table are still decode errors
  (`tests/all_missing_complex.rs`).
  Patched-file SHA-256 values are respectively
  `22254047da83fafcc06816170b7687127cb9f64bc7e82e4de2d239861c149547`
  and `be1485f980301df6c99a91b8b7d1440fc6c90da526627e36494d92662bdc6297`.
- `src/grib2/grid.rs` fails closed on a grid-definition template it has no
  point placement for: `grid_latlon` returns `crate::Result` and refuses
  naming the template number (and that supporting it needs the template's
  point placement implemented there) instead of returning EMPTY coordinate
  vectors -- an unsupported input decoding "successfully" to nothing and
  reaching sha-bound artifacts as a good decode, the same class as the
  unread missing-value octet.  A reduced (quasi-regular) grid without a
  `pl` array is refused the same way.  From the 2026-08-17 convergence,
  Template 3.0/3.40 longitudes fall back to the DECLARED i-direction
  increment where the endpoints do not describe a span -- a cyclic global
  grid may encode identical first and last longitudes (180 to 180 over 721
  half-degree points), and interpolating between equal endpoints collapses
  every longitude onto one meridian -- and take their sign from it on a -i
  scan.  The other descendant preferred the declared increment
  UNCONDITIONALLY; that is a regression and the measurement is recorded here
  because it is the only reason the rule is narrow.  Run against the staged
  `gdas.t06z.sfluxgrbf003` 3072-point Gaussian rows, unconditional-increment
  placement walked the last column from 359.882813 to 359.884348 (0.0015 deg,
  ~170 m, growing linearly across the row) because the declared Di is the
  1e-6 rounding 0.117188 of a true 0.1171875 spacing, while the endpoints
  reproduce the spacing exactly.  Both halves are pinned by tests.
  Patched-file SHA-256:
  `6e2f872fb9cc157572c6397d6dc801cddc67573f28ccbf82abe23815e47c1bd2`.
- `src/grib2/tables.rs` extends the raw GRIB2 Code Table 4.2 rows with
  the 53 NCEP parameter triplets that real RRFS/REFS products carry and
  stock tables cannot name -- every severe-weather field a convective
  workflow reads (composite reflectivity 0.16.5, echo top 0.16.3,
  hourly-max reflectivity 0.16.198, updraft helicity 0.7.15 and its
  hourly max/min 0.7.199/200, downdraft CAPE 0.7.203, effective SRH
  0.7.204, critical angle 0.7.206), the hourly-max wind/vertical-velocity
  family 0.2.220-223 and effective-shear/Bunkers vectors 0.2.234-237,
  the GOES-18 ABI simulated brightness temperatures 3.192.77-85, and the
  precipitation/land/smoke residue of that measured set.  Names and
  units are transcribed from NCEP's published GRIB2 documentation
  (nco.ncep.noaa.gov table 4.2 pages) -- catalog DATA rows, no new code
  path; the specific 0.5.7 kelvin arm is placed ahead of the long-wave
  W/m2 wildcard.  Verified against real bytes: across the eight staged
  RRFS/REFS sample files all 53 triplets now resolve to names.
  The WMO geometric vertical velocity row 0.2.9 is also named so DZDT
  inventories do not fall through to Unknown. Verified against the
  pressure-row export proof on all 25 hourly frames.
  Patched-file SHA-256:
  `354748f0ecd5259a9be0bee6d862579a22a7d2b7a47120ceadebd982319d53a4`.
- `src/grib1/tables.rs` adds the version-aware `parameter_entry` lookup,
  which consults the parameter table a message CITES (PDS table version +
  originating center): WMO table 2 versions 1-3 resolve indicators 1-127
  for any center, the 128-254 center-defined region resolves only for
  NCEP (center 7, whose extension rows the raw table carries), ECMWF
  table 128 (center 98) resolves against vendored ECMWF rows, and any
  other cited table is refused naming version, center, and parameter.
  Before this, `table_version` was parsed and never consulted: a May-1999
  ERA5 message citing ECMWF table 128 had parameter 134 ("Surface
  pressure") named as NCEP's "Sweat index", silently.  Patched-file
  SHA-256:
  `a5dd90dae118c32615feb56d40367c1c17b75e56a6418cb8e7c5a198b0114ecd`.
- `src/grib1/mod.rs` re-exports `parameter_entry` and
  `ParameterTableEntry` beside the raw table functions.  Patched-file
  SHA-256:
  `d8ec99296bed1d7d16f72a16445fac90b2d5358c8225ca5fe679f09c2c71452d`.
- `src/grib1/parser.rs` spells six `Option::is_none_or` checks as the
  semantically identical `Option::map_or(true, ...)`, retaining compatibility
  with the rental runtime's Rust 1.75 compiler.  Its message-level
  parameter lookups (`ProductDefinitionSection::parameter_name` /
  `parameter_units` / `parameter_abbrev` and the `Grib1Message`
  forwarders) now return `crate::Result` and route through
  `parameter_entry`, so they answer from the table the message cites or
  refuse, never from WMO/NCEP table 2 unconditionally.  Patched-file
  SHA-256:
  `200fd76c23402b12233df71ccd5458a6eae099e9662e54ce083b06f14469a356`.

- `src/lib.rs` carries the crate-level convergence notes C1-C4 (the four
  places the two descendants disagreed and how each was resolved
  fail-closed).  It is otherwise the upstream file.  Patched-file SHA-256:
  `98f034828fafe97849776a67b84cd78d1c084a1b53661491cb3fb499315b2369`.

## The 2026-08-17 convergence (one crate, both descendants)

A second descendant of this crate existed in `rusty-weather-consolidated`
and arrived vendored at `tools/rw_wps/vendor/grib-core`.  After CRLF
normalisation the two diverged in 9 files and 2664 lines.  This directory is
now the SUPERSET: gpuwm's hardening plus every behaviour the other side had
alone, with its 14 unique tests carried across.  The other copy is deleted in
the integration commit that repoints `crates/rw-wps`'s path dependency here;
nothing may re-vendor a third.

Grafted from the other descendant, each with its test:

| behaviour | file | test |
|---|---|---|
| cyclic/equal-endpoint longitudes use the declared increment | `grib2/grid.rs` | `test_latlon_grid_uses_increment_for_equal_endpoint_cyclic_grid` |
| everything else keeps endpoint placement (narrowed, measured) | `grib2/grid.rs` | `test_latlon_grid_prefers_endpoints_over_a_rounded_declared_increment` |
| WMO 4.4 fixed-duration statistical intervals, exact, no rounding | `grib2/parser.rs` | `minute_statistical_ranges_preserve_exact_duration_without_rounding` |
| probability-threshold scale factor is sign-magnitude | `grib2/parser.rs` | `probability_threshold_scale_factor_uses_grib_sign_magnitude` |
| fixed-surface scale factor is sign-magnitude | `grib2/parser.rs` | `fixed_surface_scale_factor_uses_grib_sign_magnitude` |
| bounded second fixed surface captured | `grib2/parser.rs` | `parse_section4_captures_bounded_second_fixed_surface` |
| primary missing is not a difference | `grib2/unpack.rs` | `test_complex_spatial_primary_missing_is_not_a_difference` |
| primary and secondary missing skip the recurrence | `grib2/unpack.rs` | `test_complex_spatial_primary_and_secondary_missing_skip_recurrence` |
| width-zero groups classify both missing codes | `grib2/unpack.rs` | `test_complex_width_zero_groups_classify_both_missing_codes` |
| group lengths must total the Section-5 point count | `grib2/unpack.rs` | `test_complex_group_count_must_match_section5_points` |
| spatial recurrence overflow is an error | `grib2/unpack.rs` | `test_complex_spatial_recurrence_overflow_is_an_error` |
| row-window complex missing matches the full decode | `grib2/unpack.rs` | `test_row_window_complex_missing_matches_full_decode` |
| zero-group complex constant matches the row window | `grib2/unpack.rs` | `test_zero_group_complex_constant_matches_row_window` |
| IEEE trailing padding trimmed to the declared count | `grib2/unpack.rs` | `test_unpack_message_trims_ieee_padding_to_grid_point_count` |

`rotated_to_geographic` and the NCEP/WMO table rows the other side listed as
unique turned out to be shared or already supersets here; the diff is the
authority, not the changelog either side wrote.

The four disagreements and their fail-closed resolutions are recorded as
notes C1-C4 in `src/lib.rs`, each cross-referenced from the code site.  One
conflict test could not be carried as written: the other side's
`test_unknown_template_returns_empty` asserts the behaviour C1 refuses, so it
lives here re-expressed as
`grib2::grid::tests::test_unknown_template_fails_closed_naming_the_template`.

### Running the crate's own suite offline

The manifest declares optional `png` and `libaec-sys`, and cargo resolves a
ROOT package's optional dependencies whether or not their features are on.
Neither is in `crates-io/` -- `png` is vendored in the renderer workspace
(the shared registry grib1_bridge resolves from, so the bridge's own build
turns `png_codec` on for MRMS's DRT 5.41), `libaec-sys` nowhere -- so `cargo test` with this directory as the workspace
root fails resolution offline.  Run it from a consumer workspace instead,
which is also the posture that ships:

```
cd tools/grib1_bridge && cargo test --offline --locked -p grib-core   # jpeg2000 only
cd tools/rustwx      && cargo test --offline --locked -p grib-core   # + png_codec
```

Both are green at 182/182 (167 gpuwm + 13 carried across + 2 written here for
the C2 short-payload refusal and the narrowed increment rule).  Dropping the two
optional dependencies to make the standalone invocation work is not an
option: `tests/test_grib_core_convergence.py` pins the feature table at
`{default, jpeg2000, png_codec, ccsds}` because three renderer consumers
build with defaults on, and `png_codec` is how the renderer reads Template
5.41 products.

Every SHA-256 above is over the file's bytes as committed (LF).
`tests/test_ingest_preflight.py::test_bridge_vendor_sha_pins_match_the_vendored_files`
is the gate on them, and it exists because the `grib2/parser.rs` and
`grib2/unpack.rs` pins recorded here had ALREADY gone stale by `ed4c0ef69`
without anything noticing: a provenance record that does not match the files
cannot be used to audit or reproduce the delta it claims to describe, which is
the only job it has.

The committed Cargo lock format is version 3 for the same Rust 1.75
compatibility.  The GRIB1 bridge retains its version-1 JSON metadata plus
little-endian-f64 value-stream protocol; the new GFS bridge has its own
fail-closed FP32 series protocol documented in the parent README.

`crates-io/` is generated from the committed `Cargo.lock` by `cargo vendor`.
Every package retains Cargo's `.cargo-checksum.json` plus any upstream license
files.  `.cargo/config.toml` replaces crates.io with that directory, enabling
`cargo build --locked --offline` from a clean checkout.

## WOOF 1.0.0 re-pins

The patched-file SHA-256 values above were re-derived when WOOF 1.0.0's
text stage replaced the em dashes in these files' comments and messages
with plain punctuation. No decoding logic changed.
