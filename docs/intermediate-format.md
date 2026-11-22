# The WOOF-ArWen intermediate format

**Status: stable.** This page is the specification. A third party can target
it without reading WOOF-ArWen's source.

WOOF-ArWen does not have a list of blessed data sources that you must wait to be
added to. It has an **intermediate contract**: you describe your own files in
a JSON document, WOOF-ArWen verifies the description against the actual bytes, and
the result runs through the same engine every other source runs through. This
is WPS's intermediate-file idea, with the description made explicit and
checked instead of implied by a hard-coded reader.

Two input families are supported today: **GRIB (1 and 2)** and **NetCDF (3
classic and 4)**. Nothing about the contract is specific to a named product.

If you only want the command, it is:

```
woof adapt --descriptor my-source.descriptor.json \
            --input /data/my_file_0000.nc \
            --input /data/my_file_0006.nc \
            --output-dir ./my-adapter
```

with `--vtable Vtable.MYMODEL` added for GRIB input, and omitted for NetCDF.

---

## 1. The three documents

The contract is three versioned JSON schemas. They already existed before this
page did; this page documents and stabilises them rather than inventing a
parallel format.

| Schema string | Written by | Role |
|---|---|---|
| `rw-wps.descriptor.v1` | **you** | What your files contain and what each variable means. The authoring input. |
| `rw-wps.mapping.v1` | WOOF-ArWen | The compiled, executable form of your descriptor. Byte-sealed. |
| `gpuwm-mapped-composition-v2` | WOOF-ArWen | How products on different donor grids join, and the soil-layer geometry. |

You write the first. `woof adapt` compiles it into the second and third,
plus `gpuwm-mapped-composition-inputs-v1` (an input manifest binding every
source file and decoder by SHA-256) and `gpuwm-adapt-provenance-v1` (the
battery receipt).

A descriptor and a mapping have the **same** coordinate, field, derivation and
target semantics. They differ in exactly one way: for GRIB input, a descriptor
field carries `vtable_selectors` (references into an 11-column WPS Vtable)
where a mapping carries numeric `selectors`. For NetCDF input there is no
Vtable, so descriptor and mapping fields are identical and both use
`selectors`.

**Versioning.** The schema string is exact-match. An unrecognised value is
refused, never coerced. A change that would alter the meaning of an existing
document gets a new version string (this is why composition is at v2: the v1
soil contract was replaced rather than extended). Documents are sealed by
SHA-256 at author time and re-checked at every load, so editing a published
mapping invalidates the receipt that vouches for it, by design.

---

## 2. Descriptor structure

```jsonc
{
  "schema": "rw-wps.descriptor.v1",
  "name": "my-model-pressure-level-netcdf",
  "format": "netcdf",              // "grib1" | "grib2" | "netcdf"
  "coordinates": { "horizontal": …, "vertical": …, "time": …, "member": … },
  "fields":      { "<canonical_name>": { … }, … },
  "derivations": [ … ],            // optional
  "target":      { … },
  "adapt":       { … }             // woof adapt policy (soil geometry, model top)
}
```

Unknown keys are refused at every level, with a spelling suggestion. No key is
ignored, because a dropped key runs a default under the name of your value.

### 2.1 `coordinates.horizontal`

```jsonc
"horizontal": {
  "kind": "variables",             // or "embedded_grid" (GRIB only)
  "latitude":  { "format": "netcdf", "name": "lat", "standard_name": "latitude" },
  "longitude": { "format": "netcdf", "name": "lon", "standard_name": "longitude" }
}
```

- `kind: "embedded_grid"` takes the grid from the GRIB grid-definition
  section. It is **refused for NetCDF**, which has no equivalent.
- `kind: "variables"` names 1-D coordinate variables. NetCDF requires this.

**Grid family.** Only a **regular latitude/longitude grid** is accepted. See
§6 for exactly how a projected grid is detected and refused.

### 2.2 `coordinates.vertical`

```jsonc
"vertical": {
  "kind": "pressure",              // see table
  "selector": { "format": "netcdf", "name": "pressure_level" },
  "units": "hPa",
  "positive": "down",              // "up" | "down"
  "levels": [1000, 850, 700, 500, 250]
}
```

`kind` is one of `pressure`, `hybrid_sigma_pressure`, `model_level`,
`height`, `soil_depth`, `embedded_levels`.

- NetCDF **requires** `selector`; the coordinate variable's `units` attribute
  must equal `units` exactly.
- If `levels` is non-empty it must equal the file's coordinate values exactly.
  A mismatch is refused, not interpolated.
- `era_ladders` (GRIB only, optional) lists the whole ladders other
  publications of the same product carry, each drawn from `levels`:
  `"era_ladders": [[5000, 10000, ..., 100000]]` beside a `levels` that also
  has 1000 Pa.  A GRIB decode stacks the largest declared ladder every
  vertical field's records carry in full, so a file from the other side of
  a publisher's level change prepares on the levels it has; a level that no
  declared ladder omits still refuses by name.  The receipt's
  `source_vertical_ladder` entry names the absent levels and one warning
  says so.  The NetCDF decoder does not read it, and a NetCDF mapping that
  declares it is refused.
- `hybrid_sigma_pressure` additionally requires `surface_pressure_field`,
  naming the declared field (in Pa, on `y`/`x`) that closes p = A + B·ps.
  The A/B coefficients arrive through one of two channels:
  the records' **GRIB pv coordinate octets** (primary, ERA5 model levels
  publish 2·(N+1) values, half-level A in Pa then B), or inline
  **`hybrid_a`/`hybrid_b` literal arrays** (the declared-data fallback for
  providers whose bytes carry no pv, NetCDF included). Counts are held to
  N+1 (half-level interfaces) or N (full levels) against `levels`; when
  both channels are present they must agree, and the bytes win. The
  retired `hybrid_a_field`/`hybrid_b_field` spellings are refused by
  name: a coefficient vector cannot be a mapping field.
  `air_pressure` is then materialized with a
  `pressure_from_vertical_coordinate` derivation (full levels as the mean
  of their bounding interfaces), and 3-D `geopotential_height` (which
  hybrid model-level archives usually do not carry) with a
  `geopotential_height_hydrostatic` derivation (surface geopotential +
  virtual temperature integrated up the half-level ladder, ECMWF's own
  build-up; provenance `@derived.hydrostatic`).

### 2.3 `coordinates.time`

```jsonc
"time": {
  "kind": "dimension",             // NetCDF must use this
  "selector": { "name": "valid_time", "standard_name": "time" },
  "units": "seconds since 1970-01-01",
  "calendar": "gregorian"
}
```

- `kind: "embedded_metadata"` reads the time from GRIB sections. **GRIB only.**
- `kind: "dimension"` is **NetCDF only and mandatory there.** The dimension
  must have a coordinate variable of the same name.
- `units` must equal the variable's `units` attribute exactly. CF decoding is
  `netCDF4.num2date` against the **declared** units, not a guessed one.
- **Calendar:** only `standard`, `gregorian` and `proleptic_gregorian` are
  accepted. `noleap`, `360_day`, `all_leap`, `julian` and the rest are refused
  by name: a climate calendar cannot be initialised into a real forecast
  clock.
- All times are converted to naive **UTC**. Duplicate valid times within one
  file are refused.

### 2.4 `coordinates.member` (optional)

Ensemble dimension. NetCDF only, `kind: "dimension"`. WRF initialization
requires exactly one selected member; more is refused.

### 2.5 `fields`

Keyed by **canonical name** (§3). Each entry:

```jsonc
"air_temperature": {
  "selectors":   [ { "format": "netcdf", "name": "t" } ],
  "units":       { "source": "K", "target": "K" },
  "source_axes": ["time", "vertical", "y", "x"],
  "target_axes": ["vertical", "y", "x"],
  "location":    "mass",
  "staggering":  "none",
  "missing":     { "kind": "reject" }
}
```

| Key | Meaning |
|---|---|
| `selectors` | Ordered list. NetCDF selectors take `name` and/or `standard_name`; at least one is required. `name` may be a single spelling or an **ordered list of accepted spellings of the same variable**. Each selector must resolve to **exactly one** variable; see §2.5.1. On a GRIB pressure-level or single-plane field the order is a preference: a record answers the first selector it matches, so a later selector stands in only at a level no earlier one published. A GRIB selector may carry `scale`, the factor that takes its records to `units.source` (geopotential listed beside geopotential height, `"scale": 0.10197162129779283`). |
| `selector_stack_axis` | NetCDF only. Stacks several single-layer variables into one axis (`"soil"` for `stl1..stl4`). |
| `units.source` | Must equal the variable's `units` attribute **exactly**, as a string. No unit parsing, no equivalence: `"m s**-1"` does not match `"m s-1"`. |
| `units.scale` / `units.offset` | Applied as `value * scale + offset` to reach `units.target`. |
| `source_axes` | The file's axis order, from `time`, `member`, `vertical`, `y`, `x`, `soil`. |
| `target_axes` | The canonical order WOOF-ArWen wants. A transpose, never a regrid. |
| `location` | `mass`, `u_face`, `v_face`, `surface`, `soil`. |
| `staggering` | `none`, `x`, `y`, `z`. |
| `missing` | `{"kind":"reject"}` refuses any gap; `{"kind":"value","value":0.0}` fills; `{"kind":"attribute","name":"_FillValue"}` reads the marker from a named attribute (NetCDF only). |
| `published_levels` | GRIB only, optional, hydrometeor mass mixing ratios only (`cloud_water_mixing_ratio`, `rain_water_mixing_ratio`, `cloud_ice_mixing_ratio`, `snow_mixing_ratio`, `graupel_or_hail_mixing_ratio`). `{"levels": [...], "absent": "zero"}` names the declared `vertical.levels` the publisher writes the field on; a file missing one of those still refuses by name, and on every other declared level the field is zero (a record there is read when a file carries one). One ladder is chosen for every stacked field, so without it a mass published on part of the ladder would refuse the file or take temperature, humidity and wind off the levels above it. GFS/GDAS pgrb2.0p25 writes its five masses on 22 of its 33 isobaric levels (50 to 1000 hPa without 70 hPa) and its packaged mapping declares exactly those. The field keeps `{"kind":"reject"}` and a zero unit offset, and `levels` must be a strict subset of `vertical.levels`. |
| `derivation` | Names a `derivations` entry instead of `selectors`. |
| `when_absent` | GRIB2 `terrain_height` only, beside its `selectors`: how terrain is derived when the composition's terrain files carry no terrain record at all. See §2.6. |

`scale_factor` / `add_offset` **packing is honoured automatically**: the
NetCDF library unpacks before WOOF-ArWen sees the values, and `_FillValue` /
`missing_value` masking applies first. `units.scale` is a separate,
descriptor-declared conversion applied after unpacking. Non-finite values
surviving the missing policy are refused.

#### 2.5.1 How a NetCDF selector resolves

A CF `standard_name` is an **identity**, not a second hurdle. The standard
name is what stays true when a producer renames a variable; the variable name
is a label the producer is free to change. So identity may come from
**either**, with this precedence:

1. **A configured name that matches is authoritative.** If any spelling in
   `name` matches a variable in the file, that variable resolves and the
   standard name is not consulted.
2. **Otherwise `standard_name` resolves it.** A variable whose
   `standard_name` equals the configured one satisfies the selector even
   though no configured spelling matched.
3. **A rescue by standard name is reported, never silent.** The battery
   receipt records it under `descriptor_name_drift` and a notice names the
   selector, the variable that satisfied it, and the evidence. The file was
   read correctly; the *descriptor* has drifted from the producer, and you
   should add the current spelling.
4. **Ambiguity refuses.** Two variables satisfying one selector is an error
   naming **both**. Nothing is ever picked silently.
5. **No match refuses**, printing both vocabularies (§5).

Requiring the name *and* the standard name to both match (which is what this
contract did before) defeats the point of a standard name: a correctly
self-describing file breaks on a pure rename. That is not hypothetical.
ECMWF documents `level` → `pressure_level` and `time` → `valid_time` as
legacy-to-new renames, and a current Copernicus delivery still declares
`standard_name = air_pressure` on `pressure_level`. Under the conjunction
that correct standard name could not rescue it and the selector matched
nothing.

Because files of both vintages are still in the wild, prefer **accepting both
spellings** over swapping one for the other:

```jsonc
"selectors": [ { "format": "netcdf", "name": ["SWVL1", "swvl1"] } ]
```

Every spelling in one list must denote the **same** variable. Duplicates and
empty lists are refused. `configs/rw-wps-era5-netcdf.mapping.json` uses this
form throughout.

The same principle governs georeferencing (§6): identity comes from the CF
attributes the file declares, not from a name hardcoded here.

#### 2.5.2 Records an earlier publication spells differently (`record_aliases`)

A GRIB2 publisher can change how it encodes a record between releases of
one product while the record itself stays the same.  A top-level
`record_aliases` list names each earlier spelling and the spelling the
selectors read:

```jsonc
"record_aliases": [
  { "record":   { "discipline": 192, "category": 128, "parameter": 170,
                  "level_type": 106, "level_value": 7 },
    "reads_as": { "discipline": 2, "category": 3, "parameter": 18,
                  "level_type": 151, "level_value": 1,
                  "second_level_type": 151, "second_level_value": 2 } }
]
```

`record` compares only the keys it declares (`discipline`, `category`,
`parameter` required; `level_type`, `level_value`, `second_level_type`,
`second_level_value`, `pdt` optional), so a key the publisher wrote as a
missing value can be left out.  `reads_as` rewrites the keys it declares;
without `second_level_type` and `second_level_value` the record reads as
one with no second surface.  Every later check (the selectors, the soil
layer contract, the duplicate refusals) then sees one spelling, and a
file carrying both spellings of one record refuses as a duplicate.

Refused at load: an alias on a GRIB1 or NetCDF mapping (nothing would read
it), a `record` some selector already reads as spelled (the alias would
change what a current file decodes to), two aliases that can name one
record, and a `reads_as` that no field or more than one field reads.  The
preparation receipt counts the records read this way per field under
`source_composition.record_aliases`.

### 2.6 `derivations`

Computed fields, declared not inferred:

| `operation` | Arguments |
|---|---|
| `copy` | `source` |
| `wind_speed` | `u`, `v` |
| `specific_humidity_from_rh` | `relative_humidity`, `temperature`, `pressure` |
| `specific_humidity_from_dewpoint` | `dewpoint`, `temperature`, `pressure` |
| `relative_humidity_from_dewpoint` | `dewpoint`, `temperature` |
| `geopotential_height` | `geopotential`, optional `gravity_m_s2` |
| `pressure_from_vertical_coordinate` |: (pressure verticals broadcast the ladder; hybrid verticals compute p = A + B·ps through the declared `surface_pressure_field`) |
| `geopotential_height_hydrostatic` | `temperature`, `specific_humidity`, `surface_geopotential_height`, optional `gravity_m_s2` (hybrid verticals only; needs N+1 interface coefficients) |
| `volumetric_soil_moisture_from_layer_mass` | `layer_mass`, `layer_bounds_m`, optional `water_density_kg_m3` |
| `soil_surface_node_from_shallowest` | `source` |
| `surface_pressure_from_sea_level` | `sea_level_pressure`, `level_height`, `pressure`, `surface_height` (WRF real's sfcprs3 relation, per valid time: a surface under 50 m takes sea-level pressure plus the lowest layer's gradient, a higher one is interpolated in log pressure between the levels whose heights bracket it) |

Dependency cycles and missing dependencies are refused at load.

On a `pressure` vertical, `geopotential_height` needs no derivation to be
complete. Where the source publishes none, or leaves levels out, each missing
value is integrated with the hypsometric equation from the frame's own
temperature and specific humidity, anchored on the nearest level of the same
column that carries the source's own height, or on its surface pressure and
terrain height when no level does. Every value the source publishes is kept.
The preparation receipt counts the derived values under
`source_composition.completed_fields`, and preparation prints one warning.

`fields.terrain_height.when_absent` declares how terrain is made when a
composition's terrain files carry no terrain record at all (a publisher
that did not include surface geopotential in some releases of a
product).  One operation exists:

```jsonc
"when_absent": { "operation": "height_at_surface_pressure",
                 "geopotential_height": "geopotential_height",
                 "temperature": "air_temperature",
                 "specific_humidity": "specific_humidity",
                 "surface_pressure": "surface_pressure",
                 "surface_temperature": "air_temperature_2m",
                 "surface_dewpoint": "dewpoint_2m" }
```

Each column's terrain is the height of its surface pressure on its own
pressure-level geopotential height: the hypsometric equation from the
first level above the ground down to the surface pressure, with the mean
virtual temperature of that layer.  The surface temperature and humidity
are interpolated in log pressure between the two levels that bracket the
surface; below the deepest level they come from the 2 m temperature and
dewpoint, as WPS and real extrapolate.  It is derived once, from the
primary's first valid time, and broadcast like a published analysis
terrain.  Every named field must be decoded directly by the mapping, the
vertical must be `pressure` in Pa or hPa, and a surface above the highest
level refuses by count.  The receipt records the derivation under
`source_composition.alignment.derived`, and preparation prints one warning.

### 2.7 `target`

Declares what the initialization needs: `max_dom`, `target_vertical_levels`,
`soil_layer_count`, `boundary_interval_seconds`,
`require_lateral_boundaries`, `pressure_requirement`, and the
`required_fields` list. Every entry in `required_fields` must agree with the
mapped field's `target_axes`, `location` and `units.target`, and the full
canonical set of §3 must be present with exactly the axes, location and units
given there.

`boundary_interval_seconds` is the one spacing a forcing series must have.
A target that also declares `"accept_boundary_interval_multiples": true`
takes a uniform series at any whole multiple of it instead, for a product
whose every valid time on that spacing is published: a 3-hourly series of
an hourly publisher is then accepted, and a spacing that is not a whole
multiple is still refused. The key needs `boundary_interval_seconds`.

---

## 3. Canonical fields, axes and units

These fifteen are **required**. Names, axes, grid location and target units are
fixed by the contract: this table *is* the interface.

| Canonical name | Axes | Location | Target units |
|---|---|---|---|
| `air_temperature` | `vertical, y, x` | `mass` | `K` |
| `specific_humidity` | `vertical, y, x` | `mass` | `kg kg-1` |
| `eastward_wind` | `vertical, y, x` | `mass` | `m s-1` |
| `northward_wind` | `vertical, y, x` | `mass` | `m s-1` |
| `geopotential_height` | `vertical, y, x` | `mass` | `m` |
| `surface_pressure` | `y, x` | `surface` | `Pa` |
| `terrain_height` | `y, x` | `surface` | `m` |
| `skin_temperature` | `y, x` | `surface` | `K` |
| `air_temperature_2m` | `y, x` | `surface` | `K` |
| `specific_humidity_2m` | `y, x` | `surface` | `kg kg-1` |
| `eastward_wind_10m` | `y, x` | `surface` | `m s-1` |
| `northward_wind_10m` | `y, x` | `surface` | `m s-1` |
| `land_fraction` | `y, x` | `surface` | `1` |
| `soil_temperature` | `soil, y, x` | `soil` | `K` |
| `volumetric_soil_moisture` | `soil, y, x` | `soil` | `m3 m-3` |

Plus `air_pressure` (`vertical, y, x`, `mass`, `Pa`) unless the vertical
coordinate is `hybrid_sigma_pressure`, per `target.pressure_requirement`.

**Optional, policy-controlled.** These may be absent, but if absent the
`target.initialization_policies` map must say what happens instead. Silence is
refused:

`cloud_water_mixing_ratio`, `rain_water_mixing_ratio`,
`cloud_ice_mixing_ratio`, `snow_mixing_ratio`,
`graupel_or_hail_mixing_ratio`, `vertical_velocity`,
`snow_water_equivalent`, `snow_depth`, `sea_ice_fraction`.

**Conventions.**

- Winds are **earth-relative** (`eastward` / `northward`), not grid-relative.
- All canonical fields are **unstaggered mass points**. `staggering` describes
  your source; the target is always mass.
- `land_fraction` is a fraction in `[0, 1]` where **1 is land**.
- Soil layers run **shallow to deep**, with metre-valued contiguous bounds
  declared in the composition's `soil_layers`.
- Vertical order is whatever `levels` says; `positive` states the direction.

---

## 4. What WOOF-ArWen verifies, and what it trusts you for

`woof adapt` emits status `runnable_mapping_not_stock_wrf_certified`. That is
a precise claim, not hedging.

**Verified against your actual bytes:** every selector resolves to exactly one
record or variable; declared source units match the file's `units` attribute
exactly; axis rank and order; the vertical inventory equals the file's;
cadence and valid-time uniqueness; the grid family is regular lat/lon;
soil-selector order matches declared depths position by position; the decoder
actually executes end to end on your files; and every document, input file and
decoder binary is SHA-256 sealed and re-checked.

**Trusted from your declaration:** that your unit *labels* are true; absolute
geolocation and cell registration; vertical sufficiency for your application;
intended time semantics (instantaneous versus accumulated); land-mask
polarity; and soil depth labels. `docs/adapt-validation-contract.md` lists
these in two columns with a self-check for every trusted row. Run those before
you run a forecast.

---

## 5. Refusals

Everything below fails **loudly**, with a non-zero exit and a named cause. A
run that ingests nothing while exiting 0 is treated as a defect, not a result.

- **A selector matching nothing** fails the battery and prints **both
  vocabularies**: what the descriptor asked for, and what the files actually
  contain, with `standard_name` shown. Note that a renamed variable which
  still declares the right `standard_name` now resolves (§2.5.1); this fires
  when neither the name nor the standard name identifies anything.
- A selector matching **more than one** variable, naming every candidate.
  Ambiguity is never resolved by picking one.
- Declared units differing from the file's `units` attribute, by exact string.
- A vertical inventory differing from the file's coordinate values.
- Duplicate valid times; a grid or vertical coordinate that changes between
  files in one series.
- An unsupported calendar, by name.
- Missing or non-finite values under `{"kind":"reject"}`.
- Any canonical field absent, or present with the wrong axes, location or
  units.
- An unfilled `--skeleton` scaffold: every `REPLACE_WITH_*` value is listed at
  once.
- Any output path that already exists: authoring is create-only.

---

## 6. Georeferencing: what is supported and what is refused by name

**Supported: regular latitude/longitude grids only.** 1-D geographic
coordinate variables, degrees, WGS84-style spherical earth.

A NetCDF source is refused when:

1. **The file declares a CF grid mapping other than `latitude_longitude`.**
   Any variable's `grid_mapping` attribute is followed to its container, and
   any variable carrying `grid_mapping_name` is read directly. The refusal
   names the projection *and* the variables claiming it, for example
   `'lambert_conformal_conic' (used by T, Q, U, V)`.

2. **A horizontal coordinate carries a projection `standard_name`**:
   `projection_x_coordinate`, `projection_y_coordinate`, `grid_latitude`,
   `grid_longitude` and their angular variants. This is the trap the check
   exists for: a projected file's only 1-D horizontal coordinates are `x` and
   `y` in **metres**, and read as degrees they mis-georeference the entire
   forecast without raising anything.

3. **Identity cannot be confirmed.** A coordinate must carry CF degree units
   (`degrees_north` / `degrees_east` and their accepted spellings) **or**
   `standard_name` of `latitude` / `longitude`. Neither means refusal: the
   descriptor is the authority for *which* variable is latitude, but the file
   is the evidence for what that variable holds.

4. **Values leave the valid range** (|lat| > 90, |lon| > 360).

Rotated poles, Lambert conformal, polar stereographic, Mercator, Gaussian
grids, unstructured meshes and curvilinear (2-D lat/lon) grids are **not
supported**. Regrid to a regular lat/lon grid first. GRIB input is held to the
same family: grid-definition template 0, scan mode `0x40`.

---

## 7. Worked example: authoring a NetCDF source

```bash
# 1. Look at what your file actually contains.
woof-mapped-inspect --mapping my.mapping.json --input /data/f000.nc

# 2. Author the adapter from your descriptor and your real files.
woof adapt \
  --descriptor my-source.descriptor.json \
  --input /data/f000.nc --input /data/f006.nc \
  --output-dir ./my-adapter

# 3. Run preparation through the mapped route.
rw-wps --source mapped --source-format netcdf \
  --composition ./my-adapter/adapter.composition.json \
  --mapping     ./my-adapter/adapter.mapping.json \
  --input /data/f000.nc --input /data/f006.nc \
  --wps-namelist namelist.wps --geog-root /data/geog \
  --experiment-config my-case.toml --output-root ./prepared
```

Step 2 writes `adapter.mapping.json`, `adapter.composition.json`,
`adapter.provenance.json` and `adapter.inputs.json` into a new directory, and
prints a `runtime_bindings` object giving the exact arguments step 3 needs.

`configs/rw-wps-era5-netcdf.mapping.json` in the repository is a complete
worked mapping for a pressure-level NetCDF source; `docs/` also carries
`arbitrary-verified-adapters.md` (the narrative walkthrough) and
`adapt-validation-contract.md` (the trust boundary, with self-checks).

> **Note on producer drift.** `configs/rw-wps-era5-netcdf.mapping.json` now
> accepts both the legacy and the current Copernicus spellings, `level` and
> `pressure_level`, `time` and `valid_time`, `SWVL1` and `swvl1`, because
> files of both vintages are still in the wild. A variable that is renamed
> again but keeps its `standard_name` still resolves, and the rescue is
> reported so you can add the new spelling (§2.5.1).

---

## 8. Reaching it

- `woof adapt --help`: the front door.
- `woof doctor`, reports the arbitrary-input surface: which formats are
  authorable, and whether the GRIB2 decoders needed for GRIB input are
  resolvable. NetCDF needs no external decoder.
- `woof-mapped-inspect`: read a mapping against real files without
  authoring anything.
