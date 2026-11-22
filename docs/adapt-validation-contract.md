# `woof adapt`: validated for you, and trusted from your declaration

`woof adapt` takes a WPS Vtable, a descriptor you write, and your real
GRIB2 files, and emits a runnable mapped adapter. It runs a large
battery of checks before it publishes anything
([arbitrary-verified-adapters.md](arbitrary-verified-adapters.md)).

This page draws the line those checks stop at.

> A successful adaptation establishes that **the emitted files implement
> your descriptor exactly, and your GRIB files satisfy it.** It does not
> establish that **your descriptor is a correct physical interpretation
> of those files.**

Both halves matter. A descriptor that is internally self-consistent and
physically wrong passes. The failure mode to fear is not a crash: it is
a run that completes, looks plausible, and is scaled by 1000 or shifted
half a cell.

The emitted `adapter.provenance.json` is a byte- and
declaration-provenance record. It is not a physical-validation
certificate.

## How to read the tables

| Verdict | Meaning |
|---|---|
| **Validated** | `woof adapt` refuses before it publishes any adapter file. |
| **Validated downstream** | `adapt` accepts; the engine refuses later, before any `wrfinput`/`wrfbdy` is published. You find out at run time, not at authoring time. |
| **Trusted** | Neither layer checks it. A self-consistent mistake reaches the model. |

"Validated downstream" is a real defence, but it is a worse place to
learn. Section 9's self-checks move most of it to authoring time.

---

## 1. Field identity, units, and conversions

| Validated for you | Trusted from your declaration |
|---|---|
| The selector you named matches exactly one record at each valid time (or exactly one at each declared level). | That the record you selected is the physical quantity you meant. A similarly named parameter, level surface, or statistical product with the same array shape selects cleanly. |
| Source and target unit strings are present and non-empty; `scale` and `offset` are finite numbers. | That the source unit string is dimensionally what the file actually contains, and that `scale`/`offset` convert it correctly. Unit strings are labels here, not enforced dimensions. |
| The declared arithmetic really runs: the engine computes `value * scale + offset` and refuses a non-finite result. | That the arithmetic is the right arithmetic. |
| Every canonical target field has exactly the required axes, location, and target unit string, for example `specific_humidity` must be `kg kg-1` on `(vertical, y, x)` at `mass`. | That the *values* are in those units. The contract checks the label. |
| Derivation structure: `specific_humidity_from_rh` must be given relative humidity, temperature, and pressure; `geopotential_height` must be given geopotential; each derivation's arguments are checked for presence and type. | That you chose the right derivation. Mapping geopotential straight through as `geopotential_height` with an identity transform is structurally valid and wrong by a factor of `g`. |

The Vtable's unit column is authoring evidence. Descriptor compilation
does not read it to validate or synthesize a conversion, and a
successful adaptation must not be reported as having confirmed it.

Some gross unit errors do fail downstream, by accident rather than by
design:

- Specific humidity at or above 1 kg kg-1 (g/kg values passed through as
  kg/kg) is refused: the conversion to mixing ratio requires
  `0 <= q < 1`. A g/kg value *below* 1 (0.01 g/kg read as 0.01 kg/kg)
  passes that gate and is wrong by 1000.
- Surface pressure near 1000 hPa passed through as 1000 Pa often
  produces non-positive dry mass against a typical model top, and is
  refused. This is a consequence of the mass calculation, not a unit
  check; a pressure field scaled the other way can stay finite and
  positive and pass.

Pressure is required only to be finite and positive in the generic
translation. There is no upper physical bound and no hydrostatic or
geopotential/height cross-field consistency test.

**Highest-risk items to re-check by hand:** hPa/Pa factors on the
pressure coordinate and surface pressure; g/kg versus kg/kg; geopotential
versus geopotential height; relative humidity as fraction versus percent
where a direct numeric transform is used; and any selector chosen from a
family of similarly named parameters.

---

## 2. Geolocation and cell registration

| Validated for you | Trusted from your declaration |
|---|---|
| Every selected record shares one grid: GDT, `nx`, `ny`, `lat1`, `lon1`, `dx`, `dy`, scan mode, shape of earth, and resolution flags are identical across records. | That this shared grid is in the right place. A consistently shifted `lat1` or `lon1` passes: all records agree with each other, and there is no independent expected-grid anchor to disagree with. |
| The grid is regular latitude/longitude GDT 0 with scan mode `0x40`. Anything else is refused, at authoring and again at run time. | That the payload really is in the declared scan order. The decoder trusts the scan bits and does not independently infer row orientation from the data. |
| Longitudes are wrapped to `[-180, 180)` and the data columns are reordered by the same permutation, so a 0–360 grid crossing the seam is handled correctly. | Cell registration: whether the declared `lat1`/`lon1` are cell centres or cell edges. A coherent half-cell shift interpolates cleanly and silently moves terrain-related structure. |
| Decoded coordinates must be finite, one-dimensional, strictly increasing, and regular; a duplicate coordinate after seam normalization is refused. | Consistency of the grid *endpoints* with origin, increment, and dimensions. `lat2`/`lon2` are parseable from GRIB but are not part of the inventory contract or the grid fingerprint, so that relation is not checked. |

Longitude convention on its own is low risk: the seam handling is
sound. Origin and registration on either side of the seam are
declarations.

---

## 3. Vertical levels

| Validated for you | Trusted from your declaration |
|---|---|
| The declared level list is a non-empty list of unique finite numbers. | That the numbers are in the unit the mapping implies. `vertical.units` must be a non-empty string; its content is not validated, and 500 declared as Pa when the file means hPa passes. |
| At every accepted valid time, the set of realized levels equals the declared set: no missing level, no extra selected level, no duplicate record. Checked at authoring and again during runtime assembly. | That the declared set is *scientifically sufficient*. There is no canonical expected level list, no minimum count, and no maximum interpolation-gap rule. Deliberately omitting an available intermediate level passes, and the model interpolates across the gap. |
| The lowest-pressure declared level is at or above `adapt.model_top_pa`. Equality is accepted deliberately: it is exact coverage, not an error. | - |
| List order does **not** need manual correction. The WRF initialization path sorts the source column by pressure and applies that order to temperature, humidity, winds, and hydrometeors, so a reversed descriptor list is normalized, not mismapped. | - |

---

## 4. Time

| Validated for you | Trusted from your declaration |
|---|---|
| With two or more accepted valid times: spacing is uniform, and equals the descriptor's declared boundary interval. Rechecked during full materialization. | The intended time semantics. Valid time is derived solely from GRIB reference time plus forecast time; there is no second independent authority to compare it against. A producer that wrote the wrong reference or forecast time, internally self-consistently, is not detectable here. |
| Malformed or unsupported forecast-time units are refused. | Whether **rolling source cycles** are scientifically acceptable. Two valid times three hours apart drawn from two different reference cycles pass both the cadence check and assembly. There is no rule requiring one common source cycle across the forcing sequence. |
| Within one valid time: a single ensemble member, consistent generating-process metadata, and one source cycle per field group. | - |
| Once the adapter is bound to an experiment: the first forcing time must equal the experiment start, and the series must cover the run. | - |
| A single valid time passes `adapt` (cadence is not measurable from one sample) and is refused at materialization for a lateral-boundary product. **Validated downstream.** | - |

---

## 5. Land mask and missing values

| Validated for you | Trusted from your declaration |
|---|---|
| `preserve_mask` is permitted only on soil mappings. Everything else must be `reject`, so a missing value in a non-soil required field fails. | That `LANDSEA` has the polarity and spatial meaning you intend. A wrong or inverted mask that labels genuinely-omitted land as ocean passes, and the declared ocean-repair path then fills it. |
| `land_fraction` must be finite, share the soil grid, and lie within `[0, 1]`: a bounded admission, so a mis-scaled unit transform delivering 2.0 is caught rather than silently read as land. | That the fraction means *land* rather than *water*. A cleanly inverted mask satisfies every check above. |
| Soil values missing on any cell the mask calls land (`land_fraction >= 0.5`) are refused during regular-snapshot translation. **Validated downstream**: the authoring battery decodes but does not run the land-aware materialization. | - |

This is the one place where the two failure directions are asymmetric:
omissions over declared *land* are caught; omissions over declared
*ocean* are repaired by design. An inverted mask converts the first case
into the second.

---

## 6. Soil layers

Soil is the most thoroughly validated dimension in the battery.

| Validated for you | Trusted from your declaration |
|---|---|
| Temperature and moisture bind the same number of layers, with the same layer surfaces and the same numeric bounds. | That the GRIB producer's type-106 depth bounds are physically correct. The bytes contain no independent depth truth. |
| Source layers are ordered and contiguous: a gap between layers is named and refused. | That the values really are layer *means* over those bounds, rather than point values or means over a different support. |
| GRIB2 selectors must be depth-below-land-layer (type 106) with numeric bounds equal to the declared layers. | - |
| The target is exactly the four Noah layers `[0, 0.1]`, `[0.1, 0.4]`, `[0.4, 1.0]`, `[1.0, 2.0]` m. `identity_complete_layers` requires the source to already be that geometry. | - |
| `conservative_layer_means` **deliberately accepts** a different four-layer source geometry when it is contiguous and completely covers every target layer, and requires an explicit missing-value policy. Alternate source depths are expected and valid here, not a mistake. | - |

---

## 7. Derivation constants and scientific choices

| Validated for you | Trusted from your declaration |
|---|---|
| A declared gravity constant must be finite and positive; the geopotential-height derivation visibly divides by it. | The value. `9.80665` versus a producer's own constant is your choice, and either passes. |
| Derivation graphs are acyclic and every dependency resolves. | Whether a *direct* mapping or a *derived* one implements the physical definition you intend. |

---

## 8. Provenance and bytes

This dimension has no trusted column, and it is worth saying so
explicitly.

| Validated for you |
|---|
| The emitted authority triple (`adapter.mapping.json`, `adapter.composition.json`, `adapter.provenance.json`) plus `adapter.inputs.json` bind the descriptor, Vtable, every input and supplement file, the provenance file, and the exact decoder binaries by path, size, and SHA-256. |
| Every composed load re-verifies the manifest digest, the mapping and composition digests, and each referenced file's path, size, and SHA-256: snapshotting and re-checking around the load to catch drift *during* it. |
| A post-authoring edit to any bound byte is refused at load. |

Authoring a *new* manifest creates a new authority; that is not an
undetected edit to the old one. Keep all four emitted files together
with the input GRIB files.

---

## 9. Self-checks you can run

None of these are run for you. Each one closes a specific trusted row
above, at authoring time rather than at run time.

### 9a. Value ranges after your unit transforms: closes §1

`woof-mapped-inspect` goes strictly deeper than the acceptance battery:
`woof adapt` decodes and assembles, while inspect additionally runs
full frame materialization. It reports, per valid time and per field,
the minimum, maximum, missing count, and SHA-256 of the array **after**
your declared unit transforms: that is, in canonical target units.

```bash
woof-mapped-inspect \
  --mapping /case/product-adapter/adapter.mapping.json \
  --input /data/product-f000.grib2 \
  --input /data/product-f003.grib2 \
  --input-manifest /case/product-adapter/adapter.inputs.json \
  --input-manifest-sha256 <input_manifest.sha256 printed by woof adapt>
```

Read `frames[*].fields[*].minimum` and `.maximum` and compare them
against physical ranges. WOOF-ArWen's named-source preflight applies the
bounds below; **the generic mapped path does not apply them**, which is
exactly why you should apply them by eye:

| Canonical field | Target unit | Plausible range |
|---|---|---|
| `air_temperature`, `air_temperature_2m`, `skin_temperature`, `soil_temperature` | K | 150 to 350 |
| `eastward_wind`, `northward_wind` | m s-1 | -200 to 200 |
| `eastward_wind_10m`, `northward_wind_10m` | m s-1 | -150 to 150 |
| `geopotential_height`, `terrain_height` | m | -2,000 to 60,000 |
| `surface_pressure` | Pa | 1.0e4 to 1.2e5 |
| `land_fraction` | 1 | 0 to 1 |
| `volumetric_soil_moisture` | m3 m-3 | -0.01 to 1.1 (small negatives are GRIB simple-packing roundoff) |
| `relative_humidity` (as a source field) | % | -10 to 150 (supersaturated and extrapolated columns are legitimate) |

`specific_humidity` is not in that table. The enforced gate is
`[0, 1)` kg kg-1, which is far wider than physical: saturation specific
humidity in the warmest near-surface air is of order 0.02–0.03 kg kg-1.
A global maximum near 1, or near 0.00003, is a factor-of-1000 unit
error that no check will catch.

`air_pressure` is required only to be finite and positive. Compare its
per-level values against your declared level list: if the levels are
hPa and the target is Pa, the maximum should be about 100× the largest
declared level.

Two more things worth reading in the same output: `frames[*].fields[*].missing`
(a non-zero count on a non-soil field means the field is not what you
think it is), and `materialization.verdict`, which must be `PASS`.

### 9b. Absolute geolocation: closes §2

Run the vendored inventory tool directly on one input file. Its path is
printed by `woof adapt` under
`runtime_bindings.decoders.grib2_inventory`:

```bash
<grib2_inventory> /data/product-f000.grib2
```

From the `lat1`, `lon1`, `dx`, `dy`, `nx`, `ny` columns, check three
things by hand:

1. **The origin**: `lat1`/`lon1` against the product's published grid
   definition, in the same longitude convention.
2. **The far corner**: the axes are built as `lat1 + k*dy` and
   `lon1 + k*dx`, so the last cell is `lat1 + (ny-1)*dy` and
   `lon1 + (nx-1)*dx`. Confirm those against the published grid.
   Nothing in WOOF-ArWen cross-checks the endpoints, because `lat2`/`lon2`
   are not carried in the inventory contract.
3. **Registration**, whether the product documents these as cell
   centres or cell edges, and whether a half-cell offset is implied.

A useful independent check: pick a coastline cell you can identify, read
`land_fraction` there from the inspect output, and confirm land is where
you expect it. That catches an origin shift and an inverted mask at the
same time.

### 9c. Vertical sufficiency: closes §3

Read your descriptor's level list and confirm (a) the unit matches what
the file actually contains, and (b) the spacing is dense enough for the
model you are building, in particular near the surface, near the
boundary-layer top, and near your model top. `adapt` checks membership
and top coverage. Nothing checks the gaps.

### 9d. Time semantics and source cycles: closes §4

From the same inventory output, read `reference_time`, `forecast_unit`,
and `forecast_time` for each input. Confirm that the implied valid times
are the analysis sequence you intend, and decide explicitly whether all
inputs should share one reference cycle. If they should, verify that
they do: nothing enforces it.

### 9e. Soil depth labels: closes §6

Read the `level_value` / `second_level_value` columns for your type-106
soil selectors and confirm the bounds against the producer's
documentation, and that the values are documented as layer means over
those bounds. Under `conservative_layer_means`, alternate source depths
are supposed to differ from Noah's; the check is that they are correctly
labelled, not that they match.

---

## Summary: validated for you

1. Closed, typed descriptor structure: unknown or malformed structures
   never become silent runtime defaults.
2. Canonical target contract: every required field's rank, axes,
   location/staggering, and exact target unit string; complete required
   field membership.
3. Exact record selection: one scalar/surface record per valid time,
   one record per declared level, no extras, no duplicates.
4. Supported GRIB structure and one shared grid across all selected
   records.
5. Decoded array structure: finite, regular, strictly monotonic
   coordinates; expected dimensions; finite values on required non-soil
   fields.
6. Declared transforms execute and produce finite results.
7. Multi-time cadence, uniform and equal to the declared boundary
   interval.
8. Vertical membership and numeric model-top coverage.
9. Soil geometry: counts, matching temperature/moisture bounds, source
   contiguity, GRIB selector bounds, exact Noah targets, coverage, and
   missing-value policy.
10. Byte-level reproducibility of the whole authority bundle.
11. At run time: target domain shape, vertical count, boundary cadence,
    forcing start, and forcing coverage against the experiment.

## Summary: you must declare correctly

Ranked by the damage a wrong declaration does: that is, by how likely
it is to produce a plausible, stable, physically wrong integration
rather than an immediate refusal.

1. **Source field identity, units, and conversions.** Every selector,
   source unit, scale, offset, and derivation. Self-check 9a.
2. **Absolute grid geolocation and cell registration.** Self-check 9b.
3. **Pressure-level units and scientific sufficiency.** Self-checks 9a
   and 9c.
4. **Land/sea classification used to interpret missing soil.**
   Self-checks 9a and 9b.
5. **Time semantics and source-cycle policy.** Self-check 9d.
6. **Soil depth labels and layer-mean semantics.** Self-check 9e.
7. **Derivation constants and direct-versus-derived choices.**
8. **Permitted missing-value interpretation**: every `preserve_mask`
   declaration, and the assumption that missing values are confined to
   source ocean.

## What "runnable" means

The emitted status is `runnable_mapping_not_stock_wrf_certified`.
Runnable means the declared mapping and composition are executable by
WOOF-ArWen's mapped engine and that this exact GRIB inventory passed the
ingest battery. Internally consistent with your descriptor, not
physically validated, and not stock-WRF certified
([arbitrary-verified-adapters.md § Runnable is not certified](arbitrary-verified-adapters.md#runnable-is-not-certified)).

## Recorded, not shipped

Known ways to move rows out of the trusted column. None is implemented
in this release; they are listed so their absence is not mistaken for
their presence.

- Dimension-aware source-unit validation: canonical identifiers for
  supported source units, with allowed conversions checked per target
  field and per vertical coordinate, and an explicit
  geopotential-to-height derivation required instead of an ambiguous
  identity mapping.
- Physical plausibility and cross-field checks inside the acceptance
  battery, reusing the named-source preflight bounds rather than
  maintaining a second set of numbers.
- Running the same full materialization at acceptance that the engine
  runs at run time, which would promote the single-time lateral-boundary
  case and the source-land soil-omission case from "validated
  downstream" to "validated".
- `lat2`/`lon2` in the inventory contract, with endpoint consistency
  checked against origin, increment, dimensions, and scan; and an
  optional declared cell registration and expected grid anchor.
- An explicit vertical contract: a pressure-compatible unit
  requirement, and optional expected level set, minimum count, and
  maximum pressure gap.
- Explicit time policy: at least two times required for
  lateral-boundary products at authoring time, and an optional expected
  first valid time and one-common-cycle flag.
- Land-mask diagnostics in the acceptance receipt: land fraction and
  missing-soil fractions over source land and source ocean, and
  optional overlap against trusted static geography.
