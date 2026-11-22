# RW-WPS CLI reference

`rw-wps` and `woof-wrf-init` call the same entry point. Exit status `0`
means the requested validation or run completed. Status `64` is invalid CLI
usage, `70` is an adapter launch failure, and `78` is an unsupported or
invalid scientific configuration.

This page covers the native preprocessor CLI. The `woof domain` wizard is
documented in docs/public/FIRST-LIGHT.md: it works worldwide, auto-selecting
Lambert conformal, Mercator, or polar stereographic from the point latitude
(`--projection` overrides), with presets `12`, `12-3`, `12-3-1`,
`12-3-1-0.5`, or `auto`, and a custom form `--root-dx KM --chain
R1,R2,...` for any other root spacing and integer refinement chain.

## Inventory and validation

| Command | Result |
|---|---|
| `rw-wps --version` | Installed RW-WPS version |
| `rw-wps --list-sources` | Complete source registry and canonical-field contract as JSON |
| `rw-wps --show-source MODEL` | One named source declaration as JSON |
| `rw-wps --show-support-matrix` | Fail-closed release matrix as JSON |
| `rw-wps --namelist-support-report ...` | WPS/WRF geometry, vertical, boundary, and physics-state report |
| `rw-wps ... --dry-run` | Route validation and exact internal argv; no contract authoring or data processing |

`--namelist-support-report` requires `--wps-namelist` and
`--namelist-input`. Add `--source-top-pressure-pa` when the source top is
known; requesting a higher model top then fails before preprocessing.

## `woof adapt`

`woof adapt` is the create-only front door for an arbitrary **GRIB2 or
NetCDF** source whose capabilities can be verified without a named adapter.
No source name is blessed in advance and no code change is needed for a new
dataset. `intermediate-format.md` is the format specification.

```text
woof adapt --vtable VTABLE --skeleton DESCRIPTOR

# GRIB2: --vtable is the selector authority and is required.
woof adapt --vtable VTABLE --descriptor DESCRIPTOR
  --input FILE [--input FILE ...] --output-dir DIR
  [--grib2-inventory EXE --grib2-dump EXE]

# NetCDF: selectors name CF variables directly, so --vtable is refused
# and the GRIB2 decoder overrides do not apply.
woof adapt --descriptor DESCRIPTOR
  --input FILE [--input FILE ...] --output-dir DIR
```

`--skeleton` scaffolds a GRIB descriptor from a Vtable and requires
`--vtable`; a NetCDF descriptor starts from the worked mapping named in
`intermediate-format.md`.

Skeleton mode writes a review-required `rw-wps.descriptor.v1` scaffold
and prints every value it left for you to replace; authoring mode
refuses that scaffold as a scaffold until they are all replaced,
rather than as a type error about whichever one it reached first.
Authoring mode compiles its Vtable selectors, inventories the actual files,
checks exact fields/levels, target unit/axis/staggering bindings, complete
declared soil policy, one regular-lat/lon grid, uniform cadence, and
source-top coverage of the descriptor's model top. For NetCDF it resolves
every declared selector and refuses a selector that matches nothing, printing
both vocabularies -- what the descriptor asked for and what the files contain.
Any failure names the missing capability and writes no adapter.

Success writes an SHA-256-bound mapping/composition/provenance authority
triple plus `adapter.inputs.json`. Its exact status is
`runnable_mapping_not_stock_wrf_certified`; unchanged-stock-WRF evidence is a
separate exact-authority gate. See
`arbitrary-verified-adapters.md` for the schema, battery, refusal examples,
output contract, and runnable-versus-certified boundary.

The battery proves the emitted files implement your descriptor and that your
GRIB files satisfy it. Units, absolute geolocation, cell registration, level
sufficiency, intended time semantics, land-mask polarity, and soil depth
labels are trusted from your declaration. `adapt-validation-contract.md`
gives the two-column contract for every input dimension and a self-check you
can run for each trusted row.

## Named source routes

| `--source` | Required source control | Notes |
|---|---|---|
| `hrrr` | f00..f12 native/surface pairs, SHA manifest, valid time | Named certified CONUS slice; hierarchy reuses a sealed root preparation |
| `gfs` | ordered `pgrb2.0p25` series, cycle, SHA manifest | One- or three-hour uniform series; 1000..100 hPa and four Noah slabs |
| `era5` | combined GRIB1, Vtable, orography, SHA manifest | Uniform series beginning at experiment start |
| `era5-l137` | per valid time, the model-level GRIB2 file (all 137 levels) AND the same hour's ERA5 pressure-level/single-level analysis; SHA manifest | ERA5 on its native 137 hybrid sigma-pressure levels through the generic mapped route.  Packaged profile supplies mapping/composition/provenance -- the A/B interface coefficients ride IN BAND in the GRIB2 coordinate octets, so no per-model coefficient table is involved, and the land surface (surface pressure, orography, land fraction, skin temperature, the 2 m/10 m diagnostics, the four soil layers) is borrowed from the donor hour, which is why two files go in per valid time.  Hourly analyses, no forecast leads; not stock-WRF certified |
| `20crv3` | exact filename-member manifest; paired pressure/surface GRIB2 | Packaged immutable authorities; Lambert `max_dom=4`; not stock-WRF certified |
| `20crv3-cf` | ordered NOAA PSL NetCDF files plus the recovered invariant supplement | The publicly downloadable 20CRv3.  Packaged profile supplies mapping/composition/provenance; ENSEMBLE MEAN, not a member; not stock-WRF certified |
| `hrrr-prs` | ordered hourly `wrfprs` GRIB2 files | HRRR's public pressure-level product through the generic mapped route.  Packaged profile supplies mapping/composition/provenance -- the Lambert grid, grid-relative wind rotation, nine-node RUC soil and the five hydrometeor masses (cloud water, cloud ice, rain, snow, graupel, mapped with the bilinear owner) are table data; in-band terrain; vertical velocity starts at zero; not stock-WRF certified |
| `rap` | ordered hourly `awip32` GRIB2 files | RAP's 32 km Lambert North-America product through the same packaged-profile route as `hrrr-prs`: complete state in one file per valid time, in-band terrain, nine-node RUC soil, RH-derived humidity; JPEG2000-packed and decoded by the bridge's own codec; not stock-WRF certified |
| `rrfs` | hourly `prslev` + `2dfld` GRIB2 file PAIRS | RRFS, HRRR's operational successor, through the same packaged-profile route: the 3 km CONUS grid is bit-for-bit HRRR's Lambert; `prslev` carries the 45-level upper air and `2dfld` carries surface/soil/terrain, so both files are passed per valid time; direct SPFH, nine-node RUC soil; not stock-WRF certified |
| `gem-gdps` | per-valid-time GDPS GRIB2 (Datamart per-variable files, concatenable) plus the analysis terrain record | ECCC's global GDPS through the generic mapped route.  Packaged profile supplies mapping/composition/provenance -- 33 pressure levels, the single 0-10 cm ISBA soil layer, and the once-per-cycle analysis invariants (orography, land mask, ice) broadcast by the generic invariant grammar; not stock-WRF certified |
| `ecmwf-open-data` | ordered three-hourly `oper` GRIB2 step files | ECMWF's open-data IFS at 0.25 degrees through the generic mapped route.  Packaged profile supplies mapping/composition/provenance -- 14 pressure levels (earlier publications: 13 levels, depth-below-land soil and no surface geopotential, read through the same mapping), dewpoint-derived 2 m humidity, index-addressed IFS soil and cycle-analysis in-band terrain are table data; 0.25-degree open data (CC-BY-4.0), the 9 km HRES is access-restricted; not stock-WRF certified |
| `gefs` | one verified member's `pgrb2a`+`pgrb2b` pair per valid time (concatenated), staged by `woof-member-prep` | One MEMBER of NCEP's 31-forecast global ensemble through the generic mapped route.  Packaged profile supplies mapping/composition/provenance -- the 31-level ladder is the measured exact union of the two products' disjoint isobaric sets, every state selector pins PDT 1 (mean/spread and accumulation twins refuse at the byte level), four Noah soil layers split across the pair; in-band terrain; not stock-WRF certified |
| `hex-global` | woof hex export frames (`--field-set full`, a lat-lon window or global-mesh), each also named as its `hex_in_band_surface` supplement, plus `--author-input-manifest` | A woof hex forecast as a WOOF-ArWen domain's initial and boundary source through the packaged `hex-history-netcdf-v1` profile (the parent's own MPAS model levels read directly, as many as the frames carry); the boundary cadence is the export's, named to `woof domain --cadence`; not stock-WRF certified.  The whole route is the fenced block in `native-mapped-source-authoring.md` |
| `mapped` | mapping, composition, ordered inputs, role bindings, SHA manifest | Generic GRIB1, GRIB2, or NetCDF; new mappings are validated, not certified |

All executing routes require an output path that does not already exist.
Static fields come from `--geog-root` or a route-specific sealed cache and
receipt. WPS and experiment/WRF namelists remain explicit authorities.

### Which engine decodes a mapped source

Every mapped route decodes through one engine, selected in this order:
the `--mapped-engine` flag, then the `GPUWM_MAPPED_ENGINE` environment
variable, then the release default. An unrecognised name refuses rather
than falling back, so a run can never report one engine and use another.

```text
--mapped-engine rust     # the Rust mapped engine, in process
--mapped-engine python   # WORKAROUND -- see below
```

`python` is a **workaround, not a mode**. It runs the slower Python
decode path and exists so that a decode the Rust engine gets wrong has a
way around it while the defect is fixed. Reach for it when a source that
used to prepare stops preparing, report what differed, and drop it again
once the fix lands.

Naming a decoder tool explicitly -- `--grib2-inventory`, `--grib2-dump`,
`--grib1-bridge` -- also selects the Python engine for that run,
whatever the default is. Those flags pin WHICH BINARY reads the bytes,
and the Rust engine decodes in process, so honouring the default there
would silently ignore the pin. Omit them and `woof prep` resolves the
tools the route needs through the staged-tool ladder, with no flags.

`woof doctor` reports the engine under `mapped decode engine`,
including which engine a bare run uses on this release.

**On this release the default is the Rust engine**, and it DECODES
GRIB2, **GRIB1 and NetCDF** records inside `gpuwm_mapped_engine`, with no
subprocess per file. That is every source FORMAT woof reads.
`woof doctor` reports `missing` if the engine is not staged, because a
bare mapped run now needs it.

Decode is not the whole route. `woof prep` COMPOSES on every call --
terrain subsetting, cross-source borrow, bound fields -- and `compose`
is declared for those same three formats, so the composition runs in the
engine too. **A bare prep of a mapped source therefore resolves no
decoder tools at all**: no `grib2_inventory`, no `grib2_dump`, no
`grib1_bridge`. The flags below are overrides for pinning a specific
binary, not requirements, and naming one routes that run to the Python
engine on purpose.

Every subcommand the mapped route uses -- `decode`, `inspect`,
`compose` -- is declared for every source format woof reads, so no
mapped source falls back to Python on a bare run.

**The exact-member door rides the same route.** `woof prep --source
20crv3` -- the exact-member GRIB2 adapter, not the `--source 20crv3-cf`
NetCDF one -- verifies its own sealed member manifest, holds the
archive's product-identity contract against the engine's raw
record-inventory surface (`gpuwm_mapped_engine inventory`), and then
composes through `decode_composed_source` like every other packaged
source, with the verified filename member sealed as an explicit binding
the composition stamps onto every frame. A bare prep of it resolves no
subprocess tool; `--mapped-engine python` (or naming a tool) selects
the documented Python-engine workaround, which still reads the archive
through `grib2_inventory`/`grib2_dump`. The other non-mapped routes
(`--source era5`, `--source gfs`) require you to name their bridge with
`--bridge`, so nothing about them is silent either.

That is a route, not a fallback, and it stays checkable: the engine
states what it implements (`gpuwm_mapped_engine capabilities`), woof
reads that list, and a test compares the two against the built binary,
so the split cannot drift from the artifact. If a future format arrives
unported, asking for `--mapped-engine rust` on it still goes to the
engine and returns its own `not_implemented` refusal -- a run that asked
for Rust never silently gets Python.

### Where the engine's scratch goes

The engine stages a frame stream on disk before woof reads it back:
`compose` writes the whole composed stream (tens of GB for the largest
registered sources) and `decode` writes the decoded frameset (a few GB
for a 0.25-degree analysis). `woof prep` places the compose scratch
beside its output directory, on the same disk-backed filesystem the
output lands on. `WOOF_COMPOSE_SCRATCH=<directory>` overrides that
placement, and it also covers `decode` and `inspect`, which have no
output directory of their own and otherwise stage in the system temp.
Set it on a box whose `/tmp` is a quota-limited tmpfs, where a bare
decode of a 0.25-degree set dies with `cannot write the frame stream:
Disk quota exceeded (os error 122)`. The directory must exist: falling
back silently would put the stream on exactly the filesystem the
variable was set to avoid, so a missing one refuses by name. The engine
itself reads no `TMPDIR`; every temporary on the route, the input list
included, lands under that one directory.

The engine sizes the stream once it has decoded the first valid time
(its fields, grid and levels times the number of valid times) and
refuses before writing a byte when the scratch disk has less free space
than that, naming the folder, the bytes needed and the bytes free. A
disk that fills anyway partway through, or a quota, gets the same
refusal. Both are `ScratchDiskRefusal`, printed as `prep: REFUSED:` with
`WOOF_COMPOSE_SCRATCH` as the remedy.

`woof go` and `woof run-plan` price the same stream before they
download anything, from the source's row in
`woof/data/download-bytes.v1.json` (its grid points, the layers each
valid time publishes and the bytes per value, times the valid times the
fetch requests), and measure the disk that holds the scratch folder:
the folder `WOOF_COMPOSE_SCRATCH` names, or `chain/` inside the run
folder. A stream that cannot fit is refused before the download, naming
the folder, the size and the space free; a regional source whose
estimate may not fit, where only the layers its atmospheric window
cannot crop are certain, is a warning. `run-plan --estimate` reports the
stream as `disk.compose_scratch_bytes`.

## Source spellings

`--source` takes the id in the tables above or any other spelling in
this one; the names are matched case-insensitively and `_` reads as `-`.
A model's own name is a spelling, not only the row's id, so the name an
upstream publisher uses selects its route. This table is the registry's
own alias data (`--list-sources` prints the same rows as JSON), and a
test refuses any drift between the two.

| `--source` id | also spelled |
| --- | --- |
| `aorc` | -- |
| `era5-land` | -- |
| `ngen-csv` | `aorc-csv` |
| `nwm-retrospective-forcing` | `nwm-forcing` |
| `hrrr` | -- |
| `hrrr-prs` | `hrrr-pressure`, `hrrr-wrfprs` |
| `gem-gdps` | `gem`, `gdps`, `gem-global` |
| `icon-global` | `icon`, `icon-13km`, `dwd-icon`, `dwd-icon-global` |
| `icon-eu` | `dwd-icon-eu`, `icon-eu-regular` |
| `icon-d2` | `icon-2km`, `dwd-icon-d2` |
| `hrrr-ak` | `hrrrak`, `hrrr-alaska` |
| `gfs` | `gfs-0p25`, `gfs-0.25` |
| `gdas` | `gdas-0p25`, `gdas-0.25` |
| `gefs` | `gefs-ensemble` |
| `aigfs` | `ai-gfs` |
| `aigefs` | `ai-gefs` |
| `hgefs` | `hybrid-gefs` |
| `ecmwf-open-data` | `ecmwf`, `ifs` |
| `ecmwf-ens` | `ifs-ens`, `ecmwf-ensemble` |
| `aifs` | `aifs-v2`, `aifs-single` |
| `rap-native` | `rap-awp130bgrb` |
| `hrrr-native` | `hrrr-mapped-native` |
| `rap` | `rap-awip32` |
| `nam` | -- |
| `hiresw` | `hires` |
| `href` | `href-conus` |
| `sref` | -- |
| `rtma` | -- |
| `urma` | -- |
| `nbm` | `blend` |
| `rrfs` | `rrfs-ops` |
| `rrfs-ens` | `rrfsens` |
| `rrfs-a` | `rrfsa` |
| `rrfs-public` | -- |
| `refs` | `rrfs-ensemble` |
| `rrfs-firewx` | `firewx` |
| `wrf` | `wrf-gdex`, `wrf-arw` |
| `earth2` | `earth2studio` |
| `era5` | -- |
| `era5-l137` | `era5-model-level`, `era5-ml` |
| `20crv3` | `20cr`, `twentycrv3`, `20crv3-member` |
| `20crv3-cf` | `20crv3-netcdf`, `20cr-netcdf`, `20cr-cf` |
| `hex-global` | `hex`, `woof hex`, `hex-regional` |
| `mapped` | `generic-mapped`, `mapping-v1` |

Not every row is runnable on this release; the tables above name the
routes that execute, and `--list-sources` reports each row's status.

HRRR, ERA5, GFS, both 20CRv3 routes, and mapped routes accept:

```text
--preprocess-backend cuda|cpu|auto
--preprocess-workers N
--cpu-preprocess-bridge PATH
```

Without `--preprocess-workers` the CPU backend starts one thread per CPU
this process may use, at most eight: its peak host RAM grows with its
thread count, and the estimate `woof check` and `woof domain` size RAM
against was measured at eight. A larger `N` is honoured and peaks above
that estimate. Under the CUDA backend, `--preprocess-workers N` sets the CPU
threads of the host steps it runs in the Rust preprocessing library (soil,
snow, skin temperature and sea ice), which otherwise use every CPU the
process may use.

Under `cuda`, the horizontal and WRF-real vertical steps reproduce the CPU
arithmetic on the card. For specific-humidity inputs with surface-to-surface
pressure adjustment, `initialize_real` also keeps its column thermodynamics
on the card when the state is on CUDA, no initial perturbation is configured,
and `mp_physics` is not 28. Other column configurations use the host helpers.
The Thompson cold-start closure follows the state onto the card; cloud-number
entry retains the host double-precision power calculation. Prepared-array
identity is checked byte for byte. Cache identity retains the backend and
implementation provenance, so CPU and CUDA cache content digests differ.

An explicit CPU bridge is valid only with the CPU backend. More than one
hierarchy worker also requires the CPU backend; current CUDA hierarchy setup
is deterministic with one worker. During HRRR root preparation,
`--preprocess-workers N` is one total native CPU transform-thread budget, not
`N` threads for each active hour. `--prepare-workers` controls the independent
f01+ mapping/initialization job slots; the controller deterministically
partitions the native budget across those slots and records every effective
job allocation plus the peak allocation. `--pipeline-workers` independently
controls 1 through 64 decoder/hour jobs and is never charged to or multiplied
into the native transform budget.

## When the source does not reach the domain

A source grid that does not cover the requested domain is refused before any
output root is created, by the front door itself: two lines on stderr,
nothing on stdout, exit status `78`, and no traceback.

```text
prep: REFUSED: target points fall outside the source grid: target point
(0, 0) at lat/lon (34.2972, -102.4465) maps to source index x=-1263.144
y=76.756, and the source covers x=0..1376 (lon -23.5..62.5) y=0..656
(lat 29.5..70.5)
remedy: if the window above is a CROP of a wider grid, re-fetch the source
with a margin that contains the whole domain -- the interpolation stencil
reaches one cell beyond every corner.  If the window IS the source's whole
extent, no crop reaches this domain: move the domain inside the window, or
prepare it from a source whose grid covers it (--list-sources names every
source this install runs).
```

The first uncovered corner, the source index it maps to, and the window the
source actually covers are all named, because a crop that is too small and a
regional model that cannot reach the domain have opposite remedies. This is
the same refusal for every source -- the mapped route, a packaged profile
and the native route all raise it -- so a regional model added as table data
gets it with no new code.
A whole-globe source never raises it for a domain that only crosses the
antimeridian: on every route its longitude ring is cut opposite the domain
before the two are paired, and a regional crop across 180 degrees keeps a
continuous longitude axis.

A source whose grid declaration states a projection -- its coordinate
arrays are then that projection's own axes rather than degrees, which is
what a declared `lambert_conformal` grid such as HRRR, RAP or NAM carries
-- is paired by projecting the domain into that plane, on the nested
route exactly as on the single-domain one. A refusal from such a source
names the plane its numbers are in, and prints the domain's own lat/lon
beside them, because the plane coordinates appear in no namelist:

```text
prep: REFUSED: target points fall outside the source grid: target point
(0, 0) at lambert_conformal plane in 100 km units (y, x) = (61.8114,
102.4885), lat/lon (35.9984, 7.9755) maps to source index x=3416.282
y=2060.379, and the source covers x=0..1798 (lambert_conformal plane in
100 km units x 0..53.94) y=0..1058 (lambert_conformal plane in 100 km
units y 0..31.74)
remedy: if the window above is a CROP of a wider grid, re-fetch the
source with a margin that contains the whole domain -- the interpolation
stencil reaches one cell beyond every corner.  If the window IS the
source's whole extent, no crop reaches this domain: move the domain
inside the window, or prepare it from a source whose grid covers it
(--list-sources names every source this install runs).
```

A declared family this install does not pair against is refused by name
at the first look rather than read as degrees by whichever route looked
second.

## Native HRRR subset download

The sealed runtime includes `download_hrrr_native_subset.py` for retrieving
only the atmosphere and soil messages required by the HRRR bridge. For a full
12-hour forcing authority, bounded range and product concurrency can overlap:

```bash
python tools/download_hrrr_native_subset.py \
  --cycle 2026-07-18_00:00:00 \
  --forecast-hours 0,1,2,3,4,5,6,7,8,9,10,11,12 \
  --output-root /case/hrrr-f00-f12 \
  --file-workers 8 \
  --workers 4
```

`--workers` is the range-request limit within one GRIB product;
`--file-workers` is the number of independent atmosphere/soil products in
flight. Both preserve deterministic receipt order. The product of the two
limits may not exceed 64, preventing an apparently small setting from
creating an unbounded connection fan-out. Publication remains create-only,
and each assembled subset is checked for its exact byte count and complete
GRIB framing before it enters the final SHA-256 manifest.  Files from
before HRRRv3 (July 2018) publish hybrid cloud ice as CICE (0/6/0) rather
than CIMIXR (0/1/82); the subset selects it under either name, and the
bridge reads CICE only from a file that publishes no CIMIXR.

## Named 20CRv3 route

`rw-wps --source 20crv3` authors or consumes the custom exact-member manifest
used by the NOAA-CIRES-DOE every-member GRIB2 profile. Mapping, composition,
and provenance are immutable wheel payloads and cannot be replaced through
named-route arguments. Author a manifest with `--source-root`,
`--author-input-manifest`, and `--author-only`; then run with that manifest
and its SHA-256. Member identity remains filename-plus-manifest authoritative
because the accepted archive profile does not encode it in the GRIB2 PDT.
The mapping still declares `max_dom=4`, and this runnable route is not yet a
live unchanged-stock-WRF certificate.

## The 20CRv3 NetCDF route

`woof prep --source 20crv3-cf` reads NOAA PSL's downloadable 20CRv3 --
one variable per NetCDF file, three-hourly, global 1 degree -- through the
Rust `rw_netcdf` bridge.  It takes `--input` once per file, one
`--supplement` (the recovered orography/land-mask file), and the usual
namelist/geography/experiment/output arguments -- or `--source-root DIR`
on the folder `tools/download_20crv3_native_subset.py` writes, which
binds every variable file and `invariant.nc` itself.  It does NOT take
`--mapping`, `--composition`, `--provenance` or `--source-format`: the
packaged profile decides those and byte-checks them, and passing one is
refused rather than honoured, because a packaged source's name has to mean
one thing.

## The ERA5 model-level route

`woof prep --source era5-l137` (also spelled `era5-model-level` or
`era5-ml`) reads ERA5 on its NATIVE vertical coordinate -- all 137
hybrid sigma-pressure model levels -- instead of the interpolated
pressure ladder `--source era5` takes.  Every valid time on this row is
an analysis and the cadence is hourly; there are no forecast leads to
ask for.

Two files go in per valid time, and that is the route's defining fact.
ERA5's model-level product publishes the prognostic atmosphere only, so
the land-surface and near-surface state -- surface pressure, orography,
land fraction, skin temperature, the 2 m and 10 m diagnostics and the
four-layer soil column -- is borrowed from the SAME HOUR's ERA5
pressure-level/single-level analysis.  A donor hour that disagrees with
the model-level hour refuses by name rather than composing across times.

The two files need not share their grid points.  A MARS request with
`grid=0.25/0.25` anchors its lattice at the requested north-west corner,
while the surface analysis is stored on the 0.25 degree lattice, so a
box whose corners are not multiples of 0.25 (say `52.1/-100.3/45.1/-92.3`)
gives two lattices a part of a cell apart.  When the donor publishes
every model-level point the borrow is the exact subset, unchanged; when
it does not, the native engine regrids the donor onto the model-level
grid (bilinear, the land mask nearest, soil, skin and water state over
same-class neighbours only), lets the model-level file's outer row or
column lie up to one donor step outside the surface file, and the
preparation receipt records the regrid, how many rows and columns took
the edge cell, and how far apart the lattices were.  A model-level file
reaching further than one step beyond the surface file still refuses.

The vertical coordinate is table data read from the producer, not a
coefficient table shipped beside the code: the 137 A/B interface
coefficients ride IN BAND in the GRIB2 Section-4 coordinate values (276
numbers, out of the same record).  Pressure is materialized as
`p = A + B*ps` against the borrowed surface pressure, and geopotential
height is integrated hydrostatically from the borrowed terrain, because
the model-level product publishes `z` at level 1 only and a 3-D height
borrow would cross ladders.  `lnsp` is not consumed anywhere: surface
pressure is borrowed exactly rather than approximated.

Like every packaged-profile source it does NOT take `--mapping`,
`--composition`, `--provenance`, `--descriptor` or
`--contributing-mapping`.  The packaged profile decides all five, and
passing one is refused rather than honoured, because a packaged
source's name has to mean one thing.

`woof fetch` has no download route for the model-level half: it is a
queued Copernicus CDS MARS request (dataset `reanalysis-era5-complete`,
`levtype=ml`) run under your own account, not files at a predictable
URL, and `docs/public/SOURCES.md` states that refusal with its remedy.
The surface donor half IS a front-door fetch (`woof fetch --source
era5 --retrieve --out DIR`, which writes `era5-combined.grib`).  Put the model-level
file(s) in the same folder and hand it to the door with `--source-root
DIR`: the source's row binds the GRIB2 model-level files as the ordered
inputs and `era5-combined.grib` as the surface donor, the input manifest
is authored as `DIR/inputs.json`, and the output root defaults to
`CONFIG-prepared` beside the experiment config.  It ends by printing
the `woof sim` line that runs the forecast with `--render-products
all`, drawing every product from each output as it lands.  Run again,
it prepares into the next free `CONFIG-prepared-N` (a folder that
exists is never written over), keeps `DIR/inputs.json` while the
folder's files are unchanged, and after they or this WOOF-ArWen's decoders
change writes a new one and says which one it replaced; each
preparation keeps its own copy of the manifest it was made from.  A
named `--output-root` that exists is refused before anything is
written, and a manifest path named with `--author-input-manifest` is
still never replaced.  For a config `woof domain --source
era5-l137` wrote, `woof go CONFIG.toml --data-dir DIR` binds the same
folder the same way; the two shipped configurations below carry no
`[fetch]` table, so they run through the prep line.  Two shipped configurations carry
the whole procedure in their headers -- the exact MARS request, the
donor fetch, the prep line, and the measured VRAM and wall-clock
envelope: `configs/era5_l137_demo.toml` (one 12 km Lambert domain, two
forecast hours) and `configs/era5_l137_regional_demo.toml` (178x144 at
12 km, three hours), each beside its `.namelist.wps` companion.  One
MARS request for this dataset at a time; the datastore rejects extra
queued requests for it rather than queueing them.

Measured on real Copernicus CDS bytes, the Rust engine and the Python
reference produce byte-identical `air_pressure` and
`geopotential_height` (max ULP 0), and the preparation reaches rc 0 with
`wrfinput`/`wrfbdy` written.  The route is not yet accepted by unchanged
stock WRF, and this row does not cover ERA5 model-level requests on
other grids or level subsets: the mapping's ladder is the full 1..137
set.

## The GDPS route

`woof prep --source gem-gdps` reads ECCC's GDPS 15 km global regular
lat-lon GRIB2 product from MSC Datamart.  Datamart publishes one
JPEG2000-packed single-message file per variable and level; concatenate
each valid time's files into one input (GRIB2 messages are
self-delimiting) and pass `--input` once per valid time.  The terrain
supplement is the `GeopotentialHeight_Sfc` ANALYSIS record, passed once
with `--supplement`: GDPS publishes its invariant surface identities --
orography, land mask, sea-ice analysis -- only at PT000H, so the
packaged composition declares the `cycle_invariant_broadcast`
alignment and the mapping declares the land mask and ice analysis
`cycle_invariant`; the engine proves the record invariant wherever it
appears and binds it to every valid time.  Model state is three-hourly
(P001/P002 are surface-only products).  The soil column is the single
0-10 cm ISBA layer GDPS publishes, anchored by skin temperature at the
surface and the deep-soil temperature at 3 m, which is the WRF-real
treatment of a one-layer source.  Not yet accepted by unchanged stock
WRF.

## The HRRR pressure-level route

`woof prep --source hrrr-prs` reads HRRR's public `wrfprs` product --
one GRIB2 file per hourly valid time, each carrying the 39-level pressure
state, the surface/2 m/10 m fields, in-band terrain and the nine-depth
RUC soil column -- through the converged GRIB2 bridge.  Pass `--input`
once per wrfprs file and `--supplement` once per file again (the terrain
supplement is IN BAND: the same files carry the surface-height record the
composition binds).  The packaged profile decides the mapping,
composition, provenance and source format exactly as `20crv3-cf` does;
the projected Lambert grid, the grid-relative wind rotation and the node
soil geometry are rows in those documents, not code.  This route is
distinct from the certified native `--source hrrr` route: it initialises
from the pressure-level analysis, which is smoother near sharp terrain
than the native hybrid levels, and it is not yet accepted by unchanged
stock WRF.  Files from before HRRRv3 (July 2018) publish cloud ice
under its older GRIB2 code, CICE (0/6/0); the mapping reads it as the
same field it reads from CIMIXR (0/1/82) in later files.

## The RAP route

`woof prep --source rap` reads RAP's `awip32` product -- one GRIB2 file
per hourly valid time on the 32 km Lambert North-America grid (AWIPS
grid 221), each carrying the 39-level pressure state (the same ladder as
HRRR's wrfprs), the surface/2 m/10 m fields, in-band terrain and the
nine-depth RUC soil column.  Pass `--input` once per awip32 file and
`--supplement` once per file again (in-band terrain, exactly as
`hrrr-prs`).  Pressure-level humidity arrives as RH and is converted by
the mapping's declared derivation -- the same one the GFS profile uses.
Two accurate limits: RAP's 13 km CONUS products are not reachable as
tables (`awp130pgrb` carries no soil state, and its `awp130bgrb`
companion duplicates the surface records octet-for-octet so no selector
separates the pair), and the native `wrfprs` grid is rotated lat-lon
(GDT 32769), outside the declared grid families.  The route is not yet
accepted by unchanged stock WRF.

## The GDAS route

`woof prep --source gdas` reads NCEP's analysis-cycle pgrb2 product --
one 0.25-degree global GRIB2 file per hourly valid time, `f000..f009`
and nothing after (f010 is a measured 404) -- through the converged
GRIB2 bridge.  Pass `--input` once per file and `--supplement` once per
file again (terrain is IN BAND, exactly as `hrrr-prs` and `rap`).  Each
file carries the 33-level isobaric state with direct specific humidity
(no RH derivation anywhere on this route), the surface/2 m/10 m fields,
and the four Noah soil layers, bound by their scaled type-106 depth
pairs -- the integer `level` key collides between the 0-0.1 m and
0.1-0.4 m layers, so the depth-pair rows are the layer identity.  Soil
moisture is an NCEP local-table record (2.0.192 under
localTablesVersion 1) selected by octets; nothing asks a table to name
it.  The eight sub-hPa surfaces outside the declared ladder are
admitted and ignored.

Three accuracy facts belong in the same breath as the command.  The
files this route reads are stamped FORECASTS in the bytes even at hour
0 (`typeOfProcessedData` fc, generating process 81 at the analysis
hour, 96 after) -- while the one product stamped an ANALYSIS
(`pgrb2.1p00.anl`) is a strict subset with no soil, no land mask and no
2 m/10 m fields, and is not an initialization route.  Publication lags
the cycle by about seven hours (GDAS is the delayed-cutoff cycle; a
scheduler assuming GFS timing chases a 404).  And the catalogue is
record-for-record byte-identical to GFS `pgrb2.0p25` -- only the
acquisition path says which model a file came from.  The route is not
yet accepted by unchanged stock WRF.

## Source-pinned model specifications

`woof preset list|show|check|export|verify-sources` reads the packaged model
preset registry. The `rrfs-v2-hex-tempo` entry records the native MPAS settings
and source pins, including TEMPO, and runs on the hex dycore through
`woof preset run`. `check` reports any unmet runtime requirement and every
declared divergence from the NOAA pins, and `export` retains the requested
physics switches. See [model-presets.md](model-presets.md)
for the exact commands and limits.

## The RRFS route

`woof prep --source rrfs` reads RRFS -- HRRR's operational successor,
flowing today on the `noaa-rrfs-ops-pds` bucket and NOMADS `rrfs/v1.0`,
with operational implementation dated 2026-10-06.  The HRRR `wrfprs`
analogue is the `prslev` + `2dfld` product PAIR: `prslev` is pure upper
air (the 45-level pressure state, 2 hPa top, 70 hPa where HRRR has 75)
and `2dfld` carries every surface, 2 m/10 m, soil and terrain record.
Pass `--input` once per file for BOTH products at every valid time, and
`--supplement` once per `2dfld` file (terrain is in band, but it lives in
the `2dfld` files).  The 3 km CONUS grid is bit-for-bit HRRR's Lambert --
measured from real bytes, every geolocating octet identical -- so the
grid, the grid-relative wind rotation and the nine-node RUC soil column
ride the packaged documents exactly as `hrrr-prs` does; SPFH is direct,
so no humidity derivation is involved.

Accurate limits, in the same breath: the `natlev` native-level product, the
3 km North-America rotated grid, the thinned `subset` files and every
per-member ensemble GRIB2 exist only in the frozen prototype bucket
`noaa-rrfs-pds` (halted 2026-08-12 by design) and have no live front
door, so this route does not claim them; the relocatable `firewx` nest
changes dimensions and corner day to day; REFS publishes derived
mean/spread/probability products, which cannot initialize a model state;
and the ops bucket currently lists six days, where rolling retention
versus feed age is not yet distinguishable.  The route is not yet
accepted by unchanged stock WRF.

## The AIFS route

`woof prep --source aifs` reads ECMWF's AIFS single deterministic
forecast from open data -- one 0.25-degree global GRIB2 file per
six-hourly step -- through the same converged GRIB2 bridge.  Pass
`--input` once per step file and `--supplement` once with the 0-hour
file: AIFS publishes its invariants (surface geopotential, land mask) at
the analysis step ONLY, and the packaged composition declares that with
`cycle_invariant_broadcast` terrain alignment and a `cycle_invariant`
land-mask binding rather than any code.  The 13-level pressure ladder,
the geopotential-to-metres terrain scale, the dewpoint-derived 2 m
humidity and the two-layer ordinal (WMO type 151) soil column are all
rows in the packaged documents.

Three limits belong in the same breath as the command.  The published
soil column stops at 0.28 m, so the deeper Noah layers are WRF's own
shallow-column interpolation anchored by the skin and static deep-soil
temperatures.  No snow state and no sea-ice fraction are published, so
runs initialise bare-ground and open-water everywhere -- a winter or
polar case should not start from this product.  And an AI emulator
publishes no hydrometeors, so precipitation fields begin from zero and
carry the documented spin-up cost.  The route is not yet accepted by
unchanged stock WRF.
## The ECMWF open-data route

`woof prep --source ecmwf-open-data` (aliases `ecmwf`, `ifs`) reads
ECMWF's public 0.25-degree `oper` product -- one GRIB2 file per
three-hourly step, each carrying the 14-level pressure state, the
surface/2 m/10 m fields and the four IFS soil layers -- through the
converged GRIB2 bridge.  Pass `--input` once per step file and
`--supplement` once per file again: terrain is IN BAND, but ECMWF writes
the surface geopotential only into the cycle's analysis frame, so the
packaged composition declares `time_alignment: "cycle_invariant_broadcast"` --
the one invariance-checked terrain field is carried to every primary
valid time and the carried times are named in the receipt.  The IFS soil
column is addressed by ordinal on fixed-surface type 151 rather than by
metre depths, declared through the composition's
`selector_depth_binding`; 2 m humidity derives from the published
dewpoint; winds are earth-relative; the CCSDS-packed records and the
north-to-south row order are handled by the generic decode.  Earlier
publications of the product prepare through the same mapping: files with
13 levels (50 to 1000 hPa, no 10 hPa) are read on that ladder
(`vertical.era_ladders`), files that spell the soil layers on
depth-below-land surfaces are read through `record_aliases`, and files
with no surface geopotential have terrain derived from each column's
height at its surface pressure (`fields.terrain_height.when_absent`).
The receipt names each and preparation prints one warning for each that
changes what the column is built from.  LICENSING:
this profile is authored against the 0.25-degree open-data distribution
(CC-BY-4.0, attribution required).  ECMWF's native 9 km HRES is
access-restricted -- a data-licensing fact about the feed, not a
capability limit.  Snow and sea-ice fields remain policy-controlled, and
the route is not yet accepted by unchanged stock WRF.

The distribution is the ENSEMBLE MEAN analysis, not one of the 80 members. For a member state use `--source 20crv3` over the every-member GRIB2 archive. `woof fetch --source 20crv3-cf` downloads the public annual archive, splits year boundaries and binds published invariant surface height and land fraction in Rust. Soil uses the four published Noah layers. The published skin temperature supplies an explicit SST proxy over water; there is no separate sub-daily SST variable in this collection.

`tools/download_20crv3_native_subset.py` is a compatibility spelling for that fetch door; `tools/demo_20crv3_netcdf.sh` runs fetch, prep, sim and render end to end. See `docs/native-cf-fetch.md` for coverage, field policies and native go handoff.

## The GEFS member route

`woof prep --source gefs` initialises from ONE verified member of
NCEP's half-degree global ensemble -- the `pgrb2a` + `pgrb2b` pair of
the SAME member at each valid time.  The pairing is not advice: the two
products' isobaric level sets are exactly disjoint (measured zero
overlap on every state variable; specific humidity has zero messages in
`pgrb2a`), so the packaged mapping's 31-level ladder is only
satisfiable by both files together, and the frame assembler's
one-GRIB-member rule refuses a cross-member mix.  Stage and verify the
member first with `woof-member-prep` (below), concatenate each valid
time's verified `pgrb2a`+`pgrb2b` (GRIB2 messages are self-delimiting)
and pass `--input` once per valid time, then the SAME pair again with
`--supplement`: terrain is in band but migrates products -- measured,
the analysis publishes orography in `pgrb2a` and every forecast step
publishes it in `pgrb2b` -- so only the pair carries it at every valid
time, where it is proven invariant across the window.  Every state selector
pins PDT 1, which is what refuses the `geavg`/`gespr` statistic files
(PDT 2) sharing the member directories and the PDT-11 accumulation
twins that appear from f003 -- at the byte level, not by filename.
The four Noah soil layers ride the pair (0-0.1 m in `pgrb2a`, the rest
in `pgrb2b`) under the measured 66-percent ocean bitmap, keyed by the
scaled fixed-surface depth bounds because the integer level octet
collapses the top two layers.  Not yet accepted by unchanged stock WRF.

## Ensemble member preparation (`woof-member-prep`)

Ensemble members are fetched by filename but verified at decode on the
GRIB ensemble identity -- (productDefinitionTemplate,
typeOfEnsembleForecast, perturbationNumber) plus the declared encoded
ensemble size -- and every mismatch refuses naming both what the
filename claims and what the bytes carry.  Which members exist, how
their files are named, and what their bytes must say is a packaged
`rw-wps.members.v1` document (SHA-256-pinned table data); the engine
knows no ensemble's name.

```text
woof-member-prep --list-member-sets
woof-member-prep --describe SET
woof-member-prep --member-set SET --member ID --verify-only FILE
woof-member-prep --member-set SET --member ID --cycle YYYY-MM-DDTHH \
    --steps 0,3 [--products P,P] --inputs ROOT --output ROOT
```

`--inputs` holds fetched files at their declared upstream-relative
paths -- the only layout that preserves member identity on feeds whose
leaf filenames are identical across members.  A verified member is
staged into `OUTPUT/<set>/<cycle>/<member>/<product>/` with a
`member-receipt.json` pinning the grammar document, the member
identity, per-file SHA-256, and the observed ensemble octets.  Ensemble
means/spreads share the member namespace and decode cleanly as ordinary
fields, so a statistic claimed as a member refuses by name -- at the
member-id layer, at the filename layer, and again on the bytes.  The
declared member set is the sizing authority: the encoded
`numberOfForecastsInEnsemble` is verified as identity evidence only,
because real ensembles count their own members incompatibly.

## Ensemble products (`rw_ensbatch`)

The first ensemble renderer on the Rust path.  N member wrfout sets go
in under caller-chosen numbers; ensemble MEAN, SPREAD, exceedance
PROBABILITY, probability-matched mean and PAINTBALL panels come out
through the same store import, projected basemap, styling and PNG
writer the single-run `rw_wrfbatch` uses.  The engine knows no
ensemble's name: member identity is the caller's number, stamped on the
panels and reported on stdout.

A member is a TIME SERIES, not a file.  `frames_per_outfile = 1` is
WRF's default, so a member normally arrives as N single-frame wrfouts;
every frame of the one domain enters that member's store together, in
model-time order, and `--frames` indexes real valid times.  `--member`
therefore takes either one wrfout or the member directory holding its
frames, and the manifest route reads the member directory named by each
`gpuwm-ensemble-manifest.v1` record.

```text
rw_ensbatch --store-root DIR --out-dir DIR
    (--manifest ensemble-manifest.json | --member NUMBER=WRFOUT_OR_DIR ...)
    [--field refl|uh|precip|t2|wspd10]
    [--products mean,spread,prob,pmm,paintball]
    [--threshold V] [--neighborhood-km V] [--frames all|N]
    [--nan-policy mask|refuse] [--pmm-tie-rule flat-index|average]
    [--accept-status LIST] [--width N] [--height N] [--source-label TEXT]
    [--overlays FILE.json] [--annotate FILE.json]
rw_ensbatch --list-fields | --help | --abi
```

The paintball panel overlays each member's threshold contour in a named
color, spelled member=color on stdout.  Refusals name their breakage: a
member directory holding two nests refuses and names both domains,
because averaging d01 for one member against d02 for another is not an
ensemble; a member whose manifest status is not accepted is skipped WITH
its status, so the operator learns the right `--accept-status` from the
refusal; disagreeing grids refuse; and a member whose store cannot serve
the requested frame index refuses saying how many frames it has.  Build:
`cargo build --release --offline -p rw-wrfbatch --bin rw_ensbatch` in
`tools/rustwx`.

## Declarative authoring

The mapped route can compile a `rw-wps.descriptor.v1` plus a WPS Vtable into
an executable `rw-wps.mapping.v1`, then author the exact input manifest:

```text
--descriptor DESCRIPTOR --vtable VTABLE --author-mapping MAPPING
--author-input-manifest MANIFEST --author-only
```

Outputs are create-only. GRIB selectors, decoder bytes, inputs, supplements,
provenance files, sizes, and SHA-256 values are bound. NetCDF descriptors do
not use a Vtable. See `native-mapped-source-authoring.md` for the complete
schema and soil/remapping contract.

## Output transaction

A successful real run publishes an atomic output directory containing
`wrfinput_d01..dNN`, root-only `wrfbdy_d01`, and machine-readable receipts.
Children do not receive external boundary files. A failed run must not leave
a partially published final directory, and existing outputs are never
silently overwritten.

## `woof report`

```
woof report [RUNDIR] [-o PATH] [--dry-run|--list] [--exit-code N] [--log FILE]
```

Collects one anonymous diagnostic bundle for a run: its receipts, the
typed failure, the stage logs, the resolved config, this install's
provenance, the card, and the free space on every volume involved.
`RUNDIR` defaults to the current directory, so the command works with no
arguments from inside a run; when the current directory holds no receipt
but `out/run` below it does, that one is read and the manifest says so.

The bundle is a plain zip of UTF-8 text with `MANIFEST.txt` at its root.
The manifest is printed to stdout as well, so the reporter sees what
they are sending: what was included, what was absent and which route
would have written it, and how many strings of each identity class were
removed. `--dry-run` (alias `--list`) prints that and writes nothing.

It reads only what this product writes. Collection is by allowlist, not
by sweep: model output, input data and any unrecognised file are
inventoried by name and size and never opened. A deny-set refuses
dot-files, private-configuration directories, credential names and
key-shaped suffixes *before* anything opens, reads, hashes or lists
them, and refusals are reported counted by class rather than named,
because a file name can itself be the secret. Paths are resolved before
they are tested, so a symlink or `..` cannot carry a target past the
check. And the command refuses outright when the directory holds nothing
this product wrote, naming what it looked for -- that single rule is
what keeps `woof report` in a home directory from being a sweep.

Redaction covers usernames, home-directory prefixes, hostnames, IP and
MAC addresses, e-mail addresses, credential-shaped strings, and every
environment variable outside an explicit allowlist. Coordinates, dates,
grid shapes, physics choices and SHA-256 digests are kept: they are
scientific and provenance content, not identity.

The archive is assembled in memory and written once, so a full volume
costs a relocation rather than the bundle: `--output`, then the working
directory, then the system temporary directory, then the home
directory. Exit 0 when a bundle was written or listed; a refusal only
when no location accepts the write.

See `public/REPORTING-A-PROBLEM.md` for the user-facing page.

## Native tools and partial fetch

Platform wheels carry every declared native tool, including simulated radar,
ML export and globe packing. `woof fetch-bridges` also serves fallback installs
and repairs a user's bridge directory. `woof fetch-bridges --partial` fetches
only missing or stale binaries through HTTP byte ranges. Add repeatable
`--artifact NAME` arguments to select tools. A renderer launcher also selects
its shared `rw_wrfbatch` executable. Map assets require the full fetch.
