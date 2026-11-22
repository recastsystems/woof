# Getting data

WOOF-ArWen downloads from **every registered source that has public bytes**
through one download front door, `woof fetch`, plus the static
geography tree `woof fetch-geog` downloads and stages. `woof fetch
--source` accepts:

| | sources |
|---|---|
| hand-written transports (this page, in detail) | `gfs`, `gdas`, `hrrr`, `era5` |
| [packaged route table](#every-other-source-the-packaged-route-table) | `hrrr-prs`, `hrrr-native`, `rap`, `rap-native`, `rrfs`, `rrfs-ens`, `gefs`, `aigfs`, `aigefs`, `ecmwf-open-data`, `ecmwf-ens`, `aifs`, `icon-global`, `icon-eu`, `icon-d2`, `gem-gdps` |
| native CF subset (NOAA PSL THREDDS, annual per-variable files, 1836 to 2015) | `20crv3-cf` |
| refused by name, with a remedy | `era5-l137`, `20crv3`, `mapped` -- no bytes at a path a fetch can resolve; the refusal prints the `woof prep` line that runs on a folder you fill (`--source-root DIR`), or for `mapped` on your own mapping (see [SOURCES.md](SOURCES.md#sources-with-no-fetch-door)) |

Registry aliases work everywhere a source id does: `gdps`, `ifs`,
`hrrr-wrfprs`, `20cr`, `ai-gfs`. A registered source that is not
runnable (`nam`, `href`, `rtma`, …) refuses naming its registry status,
because nothing in this WOOF-ArWen could read the bytes a download produced.

This page covers each route end to end, the manifest handoff into the
preprocessor, and accurate disk numbers (all measured).

Three routes to keep straight from the start:

- **ERA5** feeds the config-driven GPU forecast loop
  (`woof check` / `run` / `resume`) through the native GRIB1 decode
  path.
- **GFS and HRRR** are real GRIB2 downloads that feed the `rw-wps`
  native initialization front door, which emits
  `wrfinput_d0N`/`wrfbdy_d01` for unchanged stock WRF
  ([WRF-INTEROP.md](WRF-INTEROP.md)) and archives for
  [downscaling](DOWNSCALE.md). They do not feed the `[case_data]` GPU
  run path today; the domain wizard prints exactly which route your
  source gets.
- **GDAS runs f000..f009 through the staged mapped chain.** `woof
  fetch --source gdas` downloads verified, digest-bound GRIB2, the
  `gfs_grib2_bridge` decoder reads it, and the fetch publishes the
  bound `prep-arguments.json` handoff its packaged composition
  declares, so `woof go` on a GDAS configuration runs fetch, prepare,
  forecast and render like any other staged source. f009 is where NCEP
  stops publishing the cycle, not where this decoder stops. Details are
  in the *GDAS* section below.

Everything downstream of `fetch` is fail-closed: the Rust GRIB bridges
validate envelopes, inventories, grids, and hashes before decoding, and
refuse rather than approximate.

Multi-file downloads are **pooled by default**: up to 6 files move at
once (`--fetch-workers N` to change it; `1` is the serial transport, a
knob, not a workaround).  Cold fetch cost is dominated by per-file
service latency, not bytes, so overlapping requests is where the wall
clock goes.  Politeness is engine policy, not your problem: NOMADS is
capped at 2 in-flight requests on top of the node-wide 2.5 s spacing
governor shared with the Rust backbone, `Retry-After` answers are
honored, and other hosts take the pool as asked.  Concurrency changes
no integrity property -- every file passes the same envelope walk,
record bar and SHA-256 as the serial loop, one failed file still
refuses by name, an interrupted fetch still records a contiguous
verified prefix, and `fetch-manifest.json` receipts the run under
`concurrency`: files, bytes, workers, host caps, wall seconds, the
serial model and the effective speedup.  The progress line counts the
whole request from its first line (`0 of 38 files done`) and counts a
failed file as failed.  The first file to fail, after its own retries,
ends the download at once: files not yet started never start, files in
flight are stopped, and the log names the file that failed and how many
were stopped.  Files that landed stay on disk, and the next run checks
and reuses them.

Measured cold at the default 6 workers
([receipt](receipts/fetch-pool-cold-measured.json)): a 4-file GFS
NOMADS-subset starter fetch takes 10.6 s, and a real 252-file ICON-EU
state from DWD open data comes down in 71.2 s. Many-file states are
where the pool earns its keep; on NOMADS the politeness pair is the
ceiling by design, so a 4-file subset is bounded by the host rather
than by the pool.

## GFS (0.25-degree, NOMADS or the S3 archive)

```bash
woof fetch --source gfs --cycle latest --hours 24 \
  --area 25,-110,45,-85 --p-top-pa 5000 --out data/gfs-latest

# or the whole objects from the S3 archive -- no CGI rate governor,
# no spatial crop, and the archive keeps years, not 10 days
woof fetch --source gfs --cycle 2026-07-29T18 --hours 24 \
  --mode full-file --p-top-pa 5000 --out data/gfs-full
```

- **The model top sets the ladder.** Without `--p-top-pa` the fetch
  takes the certified 21 levels, 1000 to 100 hPa, which serve a model
  top of 10000 Pa or more. Every config `woof domain` writes has a
  5000 Pa top, so the lines above carry `--p-top-pa 5000`, which adds
  the 70 and 50 hPa levels (134 records per file instead of 124; about
  8% more bytes on a measured crop). The fetch line `woof domain
  --explain` prints carries the config's own top, and `woof go` and `run-plan`
  ask for it on their own. Preparation refuses a folder that stops
  short of the config's top and names this flag.

- **Two byte transports, both first-class.** The default downloads
  NOMADS `filter_gfs_0p25.pl` subsets: your spatial window
  plus exactly the records per forecast hour the `gfs_grib2_bridge`
  requires: 124 on the certified ladder (21 pressure levels x 5
  variables, 11 surface, 8 soil), and 5 more for each level a higher
  model top adds. Every file passes an envelope walk at its own record
  count before it counts.
- **`--mode full-file` takes the whole `pgrb2.0p25` objects along the
  endpoint ladder** instead -- the AWS S3 archive (`noaa-gfs-bdp-pds`)
  for every object it has already mirrored, NOMADS for one it has not
  caught up with, the other rung behind either as fall-through --
  through the Rust backbone's parallel range GETs when it is built
  (`--engine auto` finds it; `woof doctor --source gfs` says which
  engine you get) or the stdlib transport otherwise. About 500 MB per
  forecast hour
  against single-digit MB for a regional crop, so the crop stays the
  default; what the whole file buys is *no NOMADS rate governor, no
  re-encode, and the archive's reach* (years, against NOMADS' ~10
  days). Each object is verified three ways before the manifest admits
  it: the GRIB2 envelope walk, the live `.idx` message census, and --
  on resume -- the previously recorded SHA-256. `--area` becomes
  optional request identity (nothing is cropped), and the decode still
  selects only the ladder your request declared, so a whole-globe
  object never drags its mesospheric levels into a run.
- `--cycle latest` resolves the newest cycle whose *final* requested
  hour exists (anonymous S3 HEAD probes, newest first) -- a
  mid-publication cycle can never win.
- `--forecast-start-hour K` begins the window at the cycle's f{K}
  forecast lead instead of its f000 analysis. `--hours` stays the
  window *length*, so `--forecast-start-hour 174 --hours 66` fetches
  f174..f240 and nothing before it. A run whose `start_time` is
  `cycle + K` is then initialized from f{K}, with lateral boundaries
  from f{K+i} -- reaching a window deep in a forecast without
  integrating to it. `woof domain --forecast-start-hour K` emits such
  a config (it sets `start_time` and records the lead in `[fetch]`),
  and `woof go` carries the lead into its own fetch.
  **The initial condition is then itself a K-hour forecast, not an
  analysis**, and every receipt says so: the preparation proof carries a
  `gpuwm-gfs-initial-condition-provenance-v1` block naming the cycle,
  the lead, the model start time, and forecast-generating process 96.
  **Every `wrfout` says so too** -- see
  [the initial-condition provenance a wrfout carries](#the-initial-condition-provenance-a-wrfout-carries)
  below.  Receipts live in a run directory; pictures get separated from
  it, so the durable artifact carries the statement as well.
  Adding the flag to an already-fetched `--out` with
  `--author-front-door-manifest` cuts the front-door manifest to that
  tail of the existing series instead of re-downloading it.
- **`--cadence 1` reaches f120, and no further.** NCEP publishes the
  0.25-degree `pgrb2` product every hour through f120 and every three
  hours from f120 to f384: f121, f122 and f124 do not exist and never
  will, while f120 and f123 do. An hourly window that would cross f120
  is refused before any probe runs, naming the break and pointing at
  `--cadence 3`. It used to be probed for and reported as a cycle that
  had not finished publishing, which sent readers off to wait for data
  no cycle will ever carry.
- **NOMADS keeps about 10 days; the completeness probe reads an archive
  that keeps years.** A cycle in between passed the probe and then died
  inside `urllib` with a raw traceback. The transport's own answer is
  now translated into one refusal naming the age, the HTTP status and
  the rolling window. For a cycle past NOMADS retention, `--mode
  full-file` reads the S3 archive directly -- see *Reach* below.
- Emits `gfs-series.tsv` (relative paths; the directory is relocatable)
  with each row's certified forecast-process ID, and the same four-file
  front door every table route leaves: `inputs.txt`, `prep-command.txt`
  (the bound half -- `--wps-namelist`, `--experiment-config`,
  `--geog-root` and `--output-root` are yours), `SHA256SUMS`, and
  `fetch-manifest.json` with per-file SHA-256.  The first three are
  written before the manifest and digest-bound INTO it, so the manifest
  lands last and claims every file in the directory.
- **Disk:** about 3.3 KB per square degree per forecast hour. Measured:
  ~83 KB/h at 5x5 degrees, ~3 MB/h at 30x30, ~16 MB/h at 60x80.
- **Reach:** NOMADS retention is about 10 days; the S3 archive keeps
  years, and `--mode full-file` reads it directly. Both differences
  between the two published forms are certified in
  `gfs_grib2_bridge` against committed matched pairs
  (`tests/fixtures/gfs-scan-order/`): the row order -- the grib-filter
  crop's south-to-north `0x40` against the raw archive's
  north-to-south `0x00`, normalized to one on decode and proved
  bit-identical on a TMP pair -- and the packing, GRIB2 template 5.3
  (complex + spatial differencing) against the crop's re-encoded 5.0,
  proved on a bitmap-carrying SOILW pair whose missing cells land in
  identical positions and whose present cells decode bit-identically.
  Everything outside that proven envelope (DRT 5.2, embedded
  missing-value management, any third scan mode) still refuses by
  name. Also worth knowing: an *uncropped* grib-filter request is a
  pure byte-range extractor and returns the raw 5.3 north-to-south
  records -- which the bridge now reads, but through NOMADS' rate
  governor; the S3 archive serves the same bytes without one.

## GDAS (0.25-degree analysis cycle, NOMADS)

```bash
# the f000 analysis alone -- acquisition also accepts a single analysis on other sources
woof fetch --source gdas --cycle 2026-07-29T12 --hours 0 \
  --area 25,-110,45,-85 --out data/gdas-analysis

# or the whole certified span, f000..f009
woof fetch --source gdas --cycle 2026-07-29T12 --hours 9 --cadence 3 \
  --area 25,-110,45,-85 --out data/gdas-cycle
```

Both of those land verified GRIB2 on disk and name the preparation
route that consumes them: the container fetch writes the ordered input
list, the prep command and `prep-arguments.json`, the last bound to the
in-band surface role the packaged composition declares.

GDAS is the GFS assimilation cycle's own output, published in the
*same* `pgrb2.0p25` container: same 0.25-degree grid, same variable and
level codes, same 124-record census under woof's selector, same centre
and table versions. Fetch and the `gfs_grib2_bridge` decoder serve it
with a source tag and nothing else, so everything in the GFS section
above about the transport, the record bar and the packing applies
unchanged. The front door does not carry over from GFS: GDAS reaches a
run through the staged mapped composition instead, described below.

**Scope: f000..f009, which is the published span of the cycle rather
than a bound on what this decoder will read.**

*Fetch and decode are certified through f009.* f000 carries analysis
generating process ID 81; real NOMADS f003, f006 and f009 samples carry
forecast process ID 96 -- so the endpoint of the claimed span is itself
one of the committed files, not an extrapolation from the last one that
is. Every one of those four files was verified
whole -- 124 records, scan `0x40`, DRT 5.0, shape 6, centre 7, master
table 2, local table 1, PDT 4.0 -- and the byte/signature ledger for the
fixed samples lives under `tests/fixtures/gdas-process-id/`. The series
*declares* the expected process ID on each row and the fail-closed
`gfs_grib2_bridge` verifies that declaration against its certified
`{81, 96}` capability set; it never infers process 96 from a nonzero
hour or from a source name. Each forecast sample is also required to
fail under the undeclared analysis-only policy. Past f009 refuses up
front and says why.

*The ingest route is the staged mapped composition.* The registry row
declares a packaged profile, and `woof prep --show-source gdas` states
it in machine form: `"runnable": true`, `"runner":
"mapped_composition_v1"`, field mapping
`packaged-rw-wps-gdas-pgrb2-0p25-grib2-v1`, level mapping
`33-pressure-level-to-explicit-wrf-eta-v2`, cadence mapping
`uniform-hourly-forecast-series-v1`. The profile takes all 33 pressure
levels and binds each input's own in-band terrain field
(`--supplement gdas_pgrb2_in_band_surface=FILE`), which is why the
fetch publishes `prep-arguments.json` rather than leaving the binding
to be retyped. `woof go` on a GDAS configuration therefore plans
fetch, prepare, forecast and render, and `woof domain --source gdas`
emits a `[fetch]` table like any other acquired source. For a longer
window use `--source gfs`, which is certified through f384; GDAS stops
where NCEP stops publishing the cycle.

Otherwise the only differences are naming: files, the series TSV and the
manifest role all carry the `gdas` stem, and a manifest authored over a
GDAS series records `"model": "GDAS"` -- which is a label on the bytes,
not a route through them.

**Why bother, given GFS reaches f384:** f000 is an *analysis* -- the
assimilation system's best estimate of the atmosphere at that valid
time -- rather than a forecast field carried forward by the model. For
hindcast and case work, where the initial state is the whole point,
that is the analysis-quality source at identical cost and in a
container WOOF-ArWen's decoder already reads. WOOF-ArWen has not measured the
forecast impact of analysis-versus-forecast initialization, so it makes
no claim about one.

**Antimeridian.** A `--area` longitude pair spanning more than 180
degrees is read as the complementary box crossing 180E:
`--area 45,170,60,-170` is the 20-degree Pacific box over the
dateline, never the 340-degree box that excludes it. `--point` /
`--radius-km` boxes wrap across the seam the same way, and GFS
antimeridian crops decode onto a continuous longitude axis, as does a
regional crop from any mapped source (ICON's remapped window included).
A whole-globe source has no edge at 180 degrees on any route: its
longitude ring is cut opposite the domain before the domain is paired
with it. Boxes genuinely wider than 180 degrees must be requested as the
full band or split.

**Prime meridian.** The NOMADS CGI accepts one longitude interval in
the `[0,360]` convention, so it cannot express a narrow box that crosses
0 degrees. woof currently keeps the correctness-preserving full-band
fallback, but no longer does so silently: before downloading it prints
the requested latitude/longitude box, the actual `0..360` fetched band,
and `360 / requested_width` as the longitude-span amplification. It
also records the fetched NOMADS area and amplification in
`fetch-manifest.json`. Actual compressed-byte amplification is
data-dependent and is labelled that way.

Two independent narrow CGI responses are not a verified stitched GRIB
product: simple concatenation would yield 248 records instead of the
certified 124 and two incompatible grid geometries. Until a stitcher can
re-encode one grid and pass the same envelope, exact-record-count,
geometry, and SHA-256 contracts, the disclosed full-band result is the
fail-closed route.

**Area margin.** The front door takes every model lake's nearest
source-water donor from inside the crop, and interior-continental lakes
can sit far from GFS-resolved water. `woof domain` suggests a fetch
area with that margin already built in (interpolation-stencil halo plus
a 15-degree lake-donor allowance); if you author your own area, allow 15
degrees beyond the outer domain. A lake whose nearer donor could lie
past a narrower crop's edge takes the crop's nearest water, and a lake
in a crop with no GFS water takes the skin temperature GFS has there;
both prepare, are counted under `source_coverage` in the preparation
proof, and are said in one line.

### The manifest handoff (do not hand-author)

The GFS front door requires an input manifest binding every file's
SHA-256 -- including the bridge executable's own hash. **You normally
never author it yourself:** with the `--source-manifest` pair omitted,
`woof prep --source gfs` authors and digest-binds it from the fetched
directory the series lives in, says so on stderr, and proceeds -- so
bare prep follows the fetch directly. It writes that manifest beside
`--output-root`, as `<output-root name>.gfs-input-manifest.json`, and
leaves the download as fetched, so several preparations can start from
one download at the same time. The explicit pair pins an
existing manifest instead. To author one without running prep (a
different namelist/config pairing, or a tail series), one command
writes it from the fetch:

```bash
woof fetch --source gfs --author-front-door-manifest \
  --out data/gfs-latest \
  --wps-namelist configs/myarea.namelist.wps \
  --experiment-config configs/myarea.toml
```

`--bridge` is optional: omitted, the built `gfs_grib2_bridge` is
resolved through the same ladder every other consumer uses (the
`WOOF_GFS_GRIB2_BRIDGE` override, a checkout's own `cargo build`,
`libexec/bridges`, then the `~/.woof/bridges` that `woof setup`
stages into), and the resolved executable is the one the manifest
binds. Pass `--bridge PATH` to name a different one.

This writes `gfs-input-manifest.json` (schema
`gpuwm-gfs-direct-input-manifest-v1`), prints its SHA-256, and prints
the complete ready-to-paste `rw-wps --source gfs ...` command with
`--bridge`, `--source-manifest`, and `--source-manifest-sha256` filled
in. Measured front-door cost after the handoff: 44.7 s for a
two-domain (d01 218x176 @ 12 km + d02 432x352 @ 3 km) static build,
decode, and initialization on the Rust CPU backend.

## HRRR (3 km CONUS, NOMADS or AWS Open Data)

```bash
woof fetch --source hrrr --cycle 2026-07-28T00 --hours 18 \
  --point 35.2,-97.4 --radius-km 400 --out data/hrrr-2026072800
```

- Byte-range downloads via the published `.idx` indexes: the native
  hybrid-level `wrfnat` atmosphere subset (561 records/hour) plus the
  soil records of `wrfprs` (18 records/hour) -- the exact
  `hrrr_grib2_bridge` inventory. Existing files must pass the same
  record-count and manifest-digest bar as fresh downloads; failures
  are quarantined aside, never deleted.
- Two hosts serve byte-identical GRIB **files**: the NOMADS operational
  server (roughly the newest 48 h, where each hour publishes first) and
  the full `noaa-hrrr-bdp-pds` S3 archive. They are an ordered
  **endpoint ladder**, not a set, and the ladder answers two different
  questions with two different orders.
  **Which hosts are asked** is retention: a cycle older than the
  operational window goes straight to the archive and never pays for a
  doomed attempt.
  **Which host moves the bytes** is throughput. `--transport auto`
  (default) asks the S3 archive whether it already serves the requested
  window, and takes it when it does; the operational server's whole
  advantage is having a cycle *first*, and once the archive has the
  same object that advantage is spent. When the archive has **not**
  caught up -- publication lag, the one thing the operational server is
  for -- the fetch comes from NOMADS and says so. Each run prints which
  of the two happened, in one line.
  What that is worth: the operational server paces bulk transfers, and
  at peak hours it served whole files at about 3 MB/s each, so a 3.4 GB
  request took ~20 min where the archive had served the same volume in
  ~3. Earlier measurements of the same contrast: 348/209/418/255 s from
  NOMADS against 69/34/45/44 s from S3 for four objects. Off peak the
  gap narrows or reverses (2026-08-24, one 412 MB `wrfprs`, both
  orderings: NOMADS 26.9 and 16.6 MB/s against S3's 10.5 and 14.3),
  which is why the archive is preferred only for objects it provably
  already has, never as a blanket pin.
  Probing **reorders** the ladder and never shortens it: every rung
  stays behind the chosen one, so fall-through is unchanged and a probe
  that 404s or throttles costs the transfer nothing. `nomads`/`s3` pin
  a rung, disable fall-through and skip the probe. The manifest records
  each file's serving endpoint, and a directory fetched over one host
  resumes over the other. (NOMADS' grib-filter scripts cover only the
  2-D `wrfsfc` file, so subsetting stays `.idx` byte ranges on both
  hosts.)
- **Their `.idx` indexes are not byte-identical, and the difference is
  in the field names.** NOMADS spells hybrid cloud water `CLWMR`; S3
  spells the same record `CLMR`, and has since at least 2025. woof
  treats that as an alias -- one role, either spelling, same record
  count -- so both hosts satisfy the same 561/18 contract. A change in
  the record *count* is a different matter and still stops the fetch
  until you pass `--accept-inventory-change`. If a host publishes an
  inventory woof genuinely does not recognise, `--transport auto` says
  so and moves to the other host instead of failing the run; a pinned
  `--transport` reports the mismatch and stops. Full-file transfers
  (`--engine rust --mode full-file`) never read index field names at
  all, so they are immune to this whole class of divergence.
- `--wait-for` is the live-cycle mode: hours are downloaded in order
  as they publish (polling at most every 30 s), so `rw-wps` can start
  on the early hours before the cycle finishes publishing. On timeout
  (default 90 min, `--wait-timeout-minutes`) the manifest still
  records the complete fetched prefix and re-running the same command
  resumes.
- HRRR files are CONUS-wide; `.idx` subsetting selects records, not
  areas, so `--area`/`--point` act as a coverage check. `--cycle
  latest` requires both the final `wrfnat` and `wrfprs` objects to
  exist (with `--wait-for` it instead picks the newest cycle whose f00
  pair has begun publishing).
- `--forecast-start-hour K` begins the window at f{K} here too, and
  `--hours` stays the window *length*: `--forecast-start-hour 6 --hours
  6` fetches f06..f12 and nothing before it, and the model clock starts
  at cycle + 6 h. `--cycle latest` then resolves a cycle complete
  through the window's **end** (f{K+hours}). The cycle horizon is what
  bounds K: 00/06/12/18Z cycles publish to f48, every other cycle to
  f18, and a window past that is refused by name before anything is
  downloaded. `woof domain --source hrrr --forecast-start-hour K`
  emits the matching config, namelists and printed chain.

  **On the HRRR route the typed time is always the CYCLE.** Every stage
  takes `--cycle` plus `--forecast-start-hour` and derives model time
  zero itself; nothing asks you to compute `cycle + K`. Through v1.4.0
  both preparation stages spelled their time argument `--valid-time`,
  and they read it differently -- `tools/prepare_hrrr_wrf.py` as the
  cycle, `woof.hrrr_hierarchy_direct` as model time zero. Those are the
  same instant only at lead 0, which is the only lead the front doors
  used to allow. `--valid-time` is still accepted on both, with exactly
  the meaning it had there, so existing scripts keep working; passing
  both spellings at once is refused rather than ranked. On the front
  door itself -- `woof prep --source hrrr`, which is what `woof
  domain` prints and what you should type -- the flag is
  `--valid-time`, validated there as an exact hourly HRRR cycle and
  handed to each stage under the name that stage takes.
- The fetch prints the front door's `woof prep --source hrrr` line with
  the source already bound (`--source-root`, `--source-manifest
  SHA256SUMS --source-manifest-sha256 <digest>`, `--valid-time`) and
  names the six flags left to you: `--domain-spec`, `--namelist-input`,
  `--wps-namelist` and `--experiment-config`, which `woof domain
  --source hrrr` writes beside each other, plus `--geog-root` and
  `--output-root`.  With those six added the line runs as printed.
- **Disk:** the default is whole files, so budget **~1.1 GB per
  forecast hour** -- roughly 21 GB for f00..f18. That is one `wrfnat`
  (measured 703,971,338 B on a live 2026-07-29 23Z object) plus one
  `wrfprs` (measured 426,845,586 B, the f004-f006 average of HRRR
  20260608 00Z); the fetch pulls **both** every hour, which is what the
  earlier "~0.4 GB/h" understated -- that figure priced one object in
  `--mode idx-subset`. Opt into `--mode idx-subset` and a measured
  four-object live fetch landed 882 MB for two forecast hours, i.e.
  ~0.44 GB/h, ~8.4 GB for f00..f18 -- less bandwidth, far more wall
  clock (see the mode discussion below).

## Every other source: the packaged route table

Ten more sources have a runnable packaged decode profile, and until
2.5.0 none of them had a way to get its bytes -- the 6 h model battery
brought RRFS's 6.3 GiB down with hand-written `curl`. A capability with
no front door is engine-proven, not shipped, so acquisition for all ten
is now **table data**: the packaged document
`woof/authorities/rw-wps-fetch-routes.v1.json` carries each source's
hosts, key templates, cycle grammar, lead ladder and file-set
composition, and `woof/fetch_routes.py` is the one engine that reads
every row. Adding a model's front door is a row in that file.

```bash
# one command per source; --cycle is explicit (these producers' lags
# differ by hours, so there is no accurate 'latest' to resolve)
woof fetch --source rap      --cycle 2026-08-16T00 --hours 6 --out data/rap
woof fetch --source rap-native --cycle 2026-10-02T18 --hours 3 --cadence 3 --out data/rap-native
woof fetch --source hrrr-native --cycle 2026-10-02T18 --hours 0 --out data/hrrr-native
woof fetch --source icon-eu  --cycle 2026-08-17T12 --hours 6 --out data/icon
woof fetch --source icon-d2  --cycle 2026-09-27T12 --hours 3 --out data/icon-d2
woof fetch --source gefs     --cycle 2026-08-17T00 --hours 6 --out data/gefs --member c00
woof fetch --source aigfs    --cycle 2026-08-17T06 --hours 6 --out data/aigfs
```

Each run leaves the output directory as a front door, not a pile:

| file | what it is |
|---|---|
| `inputs.txt` | the ordered `--input-list` `woof prep` consumes -- the spelling that keeps a field-per-file source's hundreds of inputs inside the 32 KB Windows command line |
| `prep-command.txt` | the bound half of the prep command, supplement role and all, runnable as written |
| `SHA256SUMS` | every downloaded and every composed file |
| `fetch-manifest.json` | request identity, per-file digests, the compose steps, the declared donors, and the pool's concurrency receipt |

**Whole files, in parallel, by default.** Every table route takes the
whole published object through the shared pool (`--fetch-workers`,
default 6; NOMADS stays capped at 2 behind the node-wide governor).
Measured on this box, cold: ICON-EU's 127-object analysis state comes
down in 35.8 s, and GDPS's 177-object state in 45.4 s. `--mode idx-subset`
is accepted where a route supports it and otherwise refuses **in the
row's own words** -- for RRFS, that five of its nine soil layers
collide on the index's level key, so an index-selected subset silently
drops them.

**File-set composition is declared, not assumed.** Three shapes appear:

- **A pair that must travel together.** GEFS's `pgrb2a` and `pgrb2b`
  isobaric level sets are exactly disjoint -- specific humidity, soil
  layers 2/3/4 and the land-sea mask are entirely in `b` -- so the
  route concatenates them per valid time into the multi-record form the
  profile decodes.
- **One message per file.** ICON-EU (125 bz2 objects per lead plus two
  time-invariant ones) and GDPS (174 per lead plus the analysis-only
  invariants) publish a file per variable-level; GDPS's are concatenated
  per valid time, ICON-EU's are passed as they are.
- **A surface supplement, under the composition's own role.** RAP,
  HRRR-prs and IFS bind their in-band surface from the same files; AIFS
  from its first file; ICON-EU from `HSURF`; GDPS from the analysis
  orography, which is deliberately held out of the composed primary.

**Cross-source donors are fetched for you.** AIGFS and AIGEFS publish no
soil, no land mask, no orography, no skin temperature and no 2 m
humidity; their packaged profiles bind those canonicals to the
**same-cycle GDAS analysis**. The fetch says so, pulls the donor into
`<out>/donor-gdas/`, and binds it in `prep-command.txt` -- which is the
difference between a source that runs and a source that refuses at init
naming six missing surfaces. Surface pressure is derived at every lead
from the source's own mean-sea-level pressure at the analysis terrain
height, so it follows the forecast and each member, and AIGEFS files
prepare the same way whether NOMADS or the AWS mirror served them. The
mirror's January to April 2026 cycles, re-encoded by another GRIB writer
(one form without the ensemble octets, one without the generating
process), prepare too: both forms are declared, and where the ensemble
octets are gone the `memNNN` folder is the member identity that is
checked.

**Ensembles keep member identity in the path.** Every AIGEFS member's
leaf filename is byte-identical, so the download preserves the
upstream-relative tree under `<out>/upstream/` and the handoff names the
`woof-member-prep` command that verifies the member before prep sees
it.

**Host choice can be a correctness question, not a speed one.** AIGFS is
NOMADS-only on purpose: `noaa-nws-graphcastgfs-pds` publishes objects
under byte-identical key names that are a *different* product (the
experimental EAGLE stream, `subCentre` 2 against the operational 0, 65
pressure messages against 78, 500 hPa geopotential height differing by
8.7 gpm). Every packaged selector pins `subcenter=0`, so the S3 copies
refuse at decode -- and nothing in the filename would have told you.
`--transport aws` on that source refuses and says this.

## 20CRv3 (ensemble members, you supply the files)

Point this **experimental, runnable** route at NOAA-CIRES-DOE Twentieth
Century Reanalysis version 3 member GRIB2 files you already hold, and it
initializes a run from them. Getting hold of the files is the one part
WOOF-ArWen does not do for you: `woof fetch --source 20crv3` refuses by
name -- the every-member archive is not published on an anonymously
readable endpoint, and only the ensemble-MEAN NetCDF distribution is --
and points at `--source-root`, which takes a directory you filled
yourself.

It is not certified: the route is not yet accepted by unchanged stock
WRF (that gate is pending), and it certifies neither other 20CR
products, nor arbitrary members mixed in one run, nor a larger domain
count than the mapping's `max_dom = 4`.

Obtaining every-member 20CRv3 GRIB2 typically
requires access to the NOAA-CIRES-DOE 20CRv3 archive holdings -- for
instance through the project's own collaboration channels -- rather than
an ordinary download; the publicly downloadable 20CRv3 products are
ensemble **mean/spread NetCDF**, which are *not* inputs to this route.
(A separate NetCDF-CF adapter exists for that family and is
synthetic-gated with no public WRF runner; it does not inherit this
route's evidence.)

What the route does accept:

- **One exact member**, bound by a filename-plus-hash manifest --
  `woof ... --source 20crv3 --author-input-manifest FILE --author-only`
  writes that manifest from the files you point `--source-root` at, and
  the run refuses anything whose name or SHA-256 differs. Mixed member
  labels in one run are rejected. Authoring ends by printing the
  `woof prep --source 20crv3` line with the manifest pair bound; it
  runs as printed once you add `--wps-namelist`, `--geog-root`,
  `--experiment-config` and `--output-root`. The GRIB2 decoders resolve
  themselves, and naming either tool pins the Python decoder.
- **Paired pressure-level and surface analyses** at a uniform
  three-hour cadence. 20CRv3 publishes analyses at successive valid
  times rather than forecast lead hours, so there is no forecast
  horizon here.
- **One-way Lambert nests** through the packaged mapping's
  `max_dom = 4`.

**The config route.** `--experiment-config` here reads
`[experiment]`, `[shared]`, `[projection]` and `[[domain]]` only. It
does *not* read `[case_data]`, which declares the ERA5 config-driven run
path, so the wizard's `--source era5` default is refused (in a sentence
naming this paragraph's remedy, not a traceback). `woof domain` has no
`--source 20crv3` option yet; emit a compatible config with
**`woof domain --source gfs`**, which writes no `[case_data]` table.
Only the geometry and physics tables it writes are consumed here -- the
`[fetch]` hints it also writes are inert on this route, because 20CRv3
has no fetch route at all and you supply the files yourself.

When preparation finishes, the front door prints the complete
hash-bound `prepared_single_domain_forecast.py --source 20crv3` command
on stderr, with every digest filled in, exactly as the GFS door does.

Mapping, composition and provenance authorities are packaged, so the
route's identity is fixed rather than assembled per run. Design and
implemented slice:
[docs/native-20crv3-source-adapter-spec.md](../native-20crv3-source-adapter-spec.md).

## ERA5 (reanalysis)

Deterministic ERA5 reanalysis is available from the public, keyless ARCO store:

```bash
woof fetch --source era5 --era5-provider arco --cycle 1999-05-03T00 --hours 24 \
  --area 30,-105,42,-90 --out data/era5-arco
```

This downloads, validates and publishes `era5-combined.nc` plus
`era5-arco-acquisition.json` in the output directory, with no account.

The Copernicus CDS route and every ERA5 ensemble-member retrieval require a
personal (free) CDS account and API key. For the CDS route, use these three steps:

```bash
# 1. Emit the exact request + instructions
woof fetch --source era5 --cycle 1999-05-03T00 --hours 24 \
  --area 30,-105,42,-90 --out data/era5-request

# 2. Retrieve with your own cdsapi key -- the command is printed, and
#    the script it names retrieves both parts and concatenates them
#    into --out, whatever directory you run it from
python data/era5-request/era5-cds-retrieve.py

# 3. Validate before anything expensive -- pass the SAME --area, and
#    the delivered grid is checked against it, not just reported
woof fetch --source era5 \
  --validate data/era5-request/era5-combined.grib --area 30,-105,42,-90
```

The emitted `era5-cds-request.json` is the precise two-part `cdsapi`
request: pressure-level z/t/u/v/r at all 37 levels, and the sixteen
single-level fields (sp, msl, 10u/10v, 2t/2d, lsm, skt, sst, ci, sd,
stl1-4, swvl1-4) plus invariant geopotential, at your times and area
(CDS order [N,W,S,E]).

`--validate` checks a retrieved set in seconds without decoding data
values: strict GRIB1 envelopes, the five pressure-level parameters at
every valid time with identical level ladders, the required surface
set at every time, soil accepted at native CDS or CDO-normalized level
types, and invariant orography presence. A wrong retrieval fails here,
not at initialization.

- **Disk:** tens of MB per valid time for a regional box at
  0.25 degrees.
- `--cycle latest` is rejected for ERA5 (reanalysis latency is weeks);
  ERA5 is the route for historical cases.

## Transports: who moves the bytes, and how

Three independent choices, and it is worth keeping them apart.

**Which downloader** -- `--engine auto|rust|python` (HRRR).
`python` is the stdlib byte-range transport; it always works and needs
nothing built. `rust` is `rw_fetch`, a binary built once from the
vendored `tools/rustwx` workspace, which brings machinery worth having
on a 700 MB object:

- **16 MiB parallel range GETs** for whole-file transfers -- one serial
  TCP stream becomes several. One fetch keeps at most 48 chunk streams
  open across all the files it moves at once (eight per file with the
  default six files in flight). A stream that delivers under 64 KiB in
  30 s is dropped and its chunk resumed from the byte it reached, and
  nothing limits how long a whole request may take, so a slow link
  finishes and a stalled stream costs 30 s instead of the run. While a
  file moves, the progress line and the run's `fetch_progress` events
  count its bytes as they arrive.
- **`.idx` range coalescing** -- a 561-record selection collapses from
  561 requests to a handful.
- **A cross-process NOMADS rate governor.** A lock file and shared
  state enforce a 2.5-second minimum gap between NOMADS requests and a
  node-wide cooldown when NOMADS' over-rate-limit response is
  recognised -- shared across *every* process on the machine that uses
  it, so two concurrent fetches cooperate instead of racing each other
  into an IP block.
- **A disk cache** keyed by URL and byte range, so a re-run or an
  overlapping window re-reads bytes instead of re-downloading them
  (`--cache-dir`).

`auto` (the default) uses it when it is built and falls back to Python
when it is not. Build it with
`cd tools/rustwx && cargo build --release --locked --offline`; `woof
doctor` reports whether it is there and usable.

**That fall-back is not free, and it now says so.** The Python
transport has no whole-file branch at all, so an install without the
backbone does not merely lose parallel range GETs -- a `--mode
full-file` request becomes `.idx` subsetting, hundreds of small serial
range requests per object. A field measurement of the same 419 MB HRRR
file: **560 s degraded, against 27-35 s taken whole** -- minutes where
the whole-file route spends seconds, and worse on NOMADS, where the
rate governor allows one worker with a 2.5 s minimum gap. Every degrade
therefore prints one `warning:`
line naming the tax and the fix before any bytes move, and the fetch
manifest records `engine_selection` -- `rust`, `python-requested`, or
`python-fallback` -- beside `engine`, so a slow run can be recognised
from its receipt afterwards. Nothing is refused: the Python transport
is correct, and a run that wants it can still ask for it with `--engine
python`, which is recorded as the request it is.

**Which host** -- `--transport` (every NCEP source). See the HRRR
section: the hosts serve byte-identical objects under identical keys,
so the default walks the source's endpoint ladder. The one exception is
AI-GEFS, whose two hosts both stay on its ladder but do not serve the
same bytes: NOMADS marks each member with ensemble type 6 where the AWS
copy of the same member says 3, the AWS surface files carry an extra
surface pressure record, and the AWS pressure-level files are repacked
copies whose heights sit within 0.08 gpm of the NOMADS ones. Preparation
reads AI-GEFS from either host the same way and derives surface pressure
itself on both. Retention decides which rungs are asked at all; throughput decides which one serves. Each
requested object gets one HEAD against the archive first -- milliseconds
against a multi-hundred-megabyte transfer -- and the archive takes any
object it has already mirrored, while an object it has not caught up
with comes from the operational server that published it. A refused
connection, a 403/503 or a Retry-After moves to the next rung, in either
order. `--wait-for` keeps polling the operational server, because seeing
an hour appear first is the point of polling it, but the transfer itself
takes the archive once the archive has the file. Naming a host pins it,
disables fall-through, and skips the probe -- a typed `--transport` is a
decision, and a decision does not get second-guessed.

The same host can live in the experiment: `transport = "s3"` in its
`[fetch]` table takes the values `--transport` takes and is refused in the
same words, and `woof go` and `woof run-plan` hand it to their fetch
stage. `woof go --transport HOST` (a run plan's `run_options.transport`)
wins over the table, and the plan names which one it used: `woof go`
prints `the fetch pins host HOST, from --transport` or `from [fetch]
transport`, and `woof run-plan` records it as a `fetch` / `transport`
entry of `automatic_resolutions`. `auto`, in the table or the flag, pins
no host: it is the ladder written out and keys the same download as
saying nothing, and `woof go --transport auto` over a table that names
a host prints `the fetch walks the host ladder, from --transport auto,
over [fetch] transport = 'HOST'`. A GFS or GDAS table fetches the
grib-filter crop, which has one transport, so `transport` there is refused.

**How much of the object** -- `--mode auto|full-file|idx-subset`
(HRRR, `--engine rust`). This is the byte transport, not the host.

`auto` is a **probe**, and deliberately has no time constants in it.
Before transferring anything, woof reads the last message named by the
`.idx`, learns its declared length from its own GRIB2 header, and asks
the origin for exactly one byte more than that message occupies. If the
origin returns the message and nothing else, the index provably ends
where the object ends and a range subset is safe. If it returns one
extra byte, the object carries records the index never mentions -- an
index published mid-write -- and the **whole file** is taken instead.
Absent index, malformed index, or a coverage question that cannot be
answered: whole file. The whole file is always correct; a subset built
on a short index would be silently, invisibly incomplete.

`full-file` and `idx-subset` force either transport. `idx-subset`
*refuses* rather than quietly degrading when the index cannot carry it
-- an operator who asked for a subset should hear that it was not
possible.

A full-file HRRR hour is the complete `wrfnat` **and** `wrfprs` pair --
measured 704 MB + 427 MB, so ~1.1 GB an hour against ~0.44 GB for the
same hour subset -- and `hrrr_grib2_bridge` consumes them unchanged:
that bridge selects by exact field identity, not by file size. Full files also never read index field *names*, which makes them
immune to the provider-spelling divergence described under HRRR above.

**Record counts.** Every download is checked against a record count
derived from the live provider inventory, with the count this WOOF-ArWen was
certified against (124 GFS/GDAS, 561 + 18 HRRR) retained as a
tripwire. Agreement is silent. Disagreement names both numbers and
stops -- an upstream inventory change is a re-certification event, not
a transient error -- until you pass `--accept-inventory-change`, which
makes the live count the bar and records the acceptance in the fetch
manifest. Field *renames* that leave the count intact are handled as
aliases and do not trip this.

Where the bar can only be read *after* a transfer -- the Rust HRRR
backbone reports a census from the index it kept beside the object it
just wrote -- the refusal says so and quarantines what landed: the GRIB
and its `.idx` are renamed `*.inventory-change-<ns>`, no manifest is
written, and the message names the files and the directory. Nothing is
deleted, so if you then accept the change the evidence of what arrived
is still there. The acceptance itself survives resuming: re-running the
command on a complete directory downloads nothing and republishes the
same `record_bars`, including `inventory_change_accepted`.

## Fetch semantics common to all sources (worth knowing)

- **Resumable by design:** re-running the same command verifies and
  skips complete files; extending `--hours` fetches only the new ones
  (per-hour files are byte-identical for the same source/cycle/area).
  GFS/GDAS atomically refresh the series and manifest after every
  verified hour. Ctrl-C therefore exits without a Python traceback,
  names the exact digest-bound prefix and any unverified `.part` on
  disk, and prints the exact command that resumes those good bytes.
- **Refuses a changed request:** a different area, cycle, or source
  into the same `--out` refuses with the exact per-field difference.
  `--force-refetch` moves the old files aside (nothing is deleted) and
  re-downloads.
- **Refuses an unaccounted directory:** a nonempty `--out` without a
  readable `fetch-manifest.json` (files copied in by another tool, or a
  corrupted manifest) also refuses -- the existing files cannot be tied
  to any recorded request, so they are never resumed. Fetch elsewhere
  or pass `--force-refetch` (quarantines, re-downloads). An interrupted
  current-version GFS/GDAS fetch is not in this category: even a
  first-hour interrupt publishes an empty request-identity manifest and
  records no unverified payload.
- Every fetch writes `fetch-manifest.json`
  (`gpuwm-fetch-manifest-v1`): source, cycle, area, and per-file
  name/role/bytes/SHA-256. Resume re-verifies existing files against
  the manifest's recorded SHA-256 as well as the per-file record bars.

## The initial-condition provenance a wrfout carries

A `run/report.json` lives in a run directory. A `wrfout` outlives it:
it is what gets archived, what `woof downscale` reads back, and what
the pictures are made from. So the statement of what the initial state
**was** travels in the file, not only in the receipt beside it.

WRF itself has no convention for this. Stock v4.6.1 writes exactly two
date globals -- `START_DATE` (this file's own start) and
`SIMULATION_START_DATE` (the simulation's, held across restarts), both
from `share/output_wrf.F:352-376` -- and both are model-clock times.
The only provenance-shaped global WRF writes at all is `FLAG_RESTART`
on a restart file (`output_wrf.F:379-381`). Nothing upstream carries it
either: metgrid's `met_em` globals are geometry, land-use identity and
`FLAG_*` presence bits, with no statement of which cycle or which lead
the fields came from.

So WRF's convention is followed where it exists -- `START_DATE` and
`SIMULATION_START_DATE` keep their WRF meaning exactly, and the new
names are `SCREAMING_SNAKE` `NC_CHAR`/`NC_INT` globals like WRF's own,
with WRF's `%Y-%m-%d_%H:%M:%S` date spelling -- and the gap is filled
in the `GPUWM_` namespace this writer already uses for `GPUWM_VERSION`
and `GPUWM_FEEDBACK`. **This is a documented divergence from WRF, not a
WRF feature.**

| global attribute | type | value |
|---|---|---|
| `GPUWM_INITIAL_CONDITION_SCHEMA` | char | `gpuwm-wrfout-initial-condition-v1` |
| `GPUWM_INITIAL_CONDITION_KIND` | char | `analysis` or `forecast` |
| `GPUWM_INITIAL_CONDITION_SOURCE` | char | driving model, e.g. `GFS` |
| `GPUWM_INITIAL_CONDITION_CYCLE` | char | the source cycle, WRF date spelling |
| `GPUWM_INITIAL_FORECAST_LEAD_HOURS` | int | K (0 for an analysis) |
| `GPUWM_INITIAL_CONDITION_GENERATING_PROCESS_ID` | int | 81 analysis / 96 forecast |
| `GPUWM_INITIAL_CONDITION_MODEL_START_DATE` | char | cycle + K, WRF date spelling |
| `GPUWM_INITIAL_CONDITION_STATEMENT` | char | the receipt's own sentence |

Three rules a reader can rely on:

1. **A forecast lead is never relabelled as an analysis.** The kind,
   the generating-process id and the lead must agree, and cycle + lead
   must compose to the model start; a block that fails any of those is
   refused at the writer rather than transcribed into a file.
2. **An analysis says `analysis`.** Silence is not a label.
3. **Absence means unknown, never analysis.** Idealized cases, files
   written before 1.4.1, and any preparation route that publishes no
   initial-condition receipt carry none of these attributes at all.

`woof downscale` copies the whole block forward onto the child: the
child's initial and boundary conditions are the parent's history, so
the child is no closer to an analysis than its parent was. It also
prints the parent's statement before the run when the lead is nonzero,
and records the block in its plan JSON.

The exact attribute set is pinned by
`tests/data/wrfout_global_attribute_set_v1.json`, regenerated only
through `tools/regenerate_wrfout_global_attribute_set.py`.

## Which physics each route can prepare

Every route prepares the normal profile family (WSM6, Thompson,
Morrison, NSSL2, MYNN), and Noah-MP behind an expert acknowledgement.
RUC has six- and nine-level soil initialization on ERA5, GFS and HRRR.
The selected soil geometry, required fields, and land/soil category
reconciliation remain binding; source/profile evidence membership alone
does not determine whether a configuration can initialize.
The v1.0.1 restriction to YSU + MM5 surface layer + Noah is gone.

The shipped registry is the authority, and answers for your exact
configuration: `rw-wps --show-physics-registry`
(`runner_routes.*.source_template_ids` and `expert_template_ids`).
Details: [PHYSICS.md](PHYSICS.md#which-suites-each-data-route-can-actually-prepare).

## Static geography (WPS_GEOG)

Terrain, land use, soil texture, vegetation fraction, LAI, albedo,
snow albedo, and deep-soil temperature come from the NCAR WPS
geographical dataset -- nine dataset directories the WRF static builder
opens (global coverage, so any domain works), plus the land-use
datasets `woof mesh` needs for the static half of its pair. The
default set carries only permissively licensed sources: soil categories
are the SoilGrids250m tiles (CC BY 4.0) WPS selects with `soilgrids`,
and deep-soil temperature is built on your machine from the ERA5
climatology (Licence to Use Copernicus Products). WPS's own defaults for
those two fields are held out for their licences (see *Provenance and
attribution* below). One command downloads and stages all of it:

```bash
woof fetch-geog
```

- Stages into `$GPUWM_CASE_DATA_ROOT/WPS_GEOG` (exactly the tree
  `woof doctor` checks and wizard configs reference); `--root DIR`
  overrides. With the variable unset, the root defaults to
  `~/Downloads` on Windows and to the XDG data directory
  (`$XDG_DATA_HOME/gpuwm`, usually `~/.local/share/gpuwm`) everywhere
  else.
- **Size:** ~2.7 GB download, ~34 GB unpacked for the default set.
  `--datasets wrf` stages what the WRF static builder opens (~2.6 GB
  download, ~28 GB unpacked) and skips the land-use datasets only
  `woof mesh` reads; `--datasets mesh` narrows the other way. Both
  include the `soilgrids` archive (864 MB, ~13 GB unpacked), which
  holds the default soil categories. `--list` previews the per-dataset table (and what is already
  staged) without touching the network; `--datasets NAME,NAME` stages
  an explicit subset.
- **Resumable and idempotent:** an interrupted download resumes from
  its byte offset; a staged, valid dataset is never re-downloaded;
  re-running the command is always safe.
- **Verified:** every archive must match its packaged size + SHA-256
  pin before extraction, every extracted dataset must carry a parsing
  WPS `index` file (the same bar `woof doctor` applies), and a local
  `geog-fetch-manifest.json` records the observed hashes for future
  re-verification.
- **Built, not downloaded: deep-soil temperature.** `soiltemp_era5_1deg`
  is produced by the native reader (`rw_zarr soiltemp-climatology`) from
  WeatherBench 2's public copy of the ERA5 1990-2019 hourly climatology
  on Google Cloud (keyless; it reads 459 MB and took 8 s on a rented
  box): the annual mean of 2 m temperature over every hour and day,
  reduced to sea level with the 0.0065 K/m lapse rate that
  TMN = SOILTEMP - 0.0065 x HGT undoes, averaged onto 1-degree cells
  area-conservatively over land only. It has `soiltemp_1deg`'s exact
  layout (same index keys, tiles, units, 0 = ocean), so every reader of
  that dataset reads it unchanged. Its `PROVENANCE.json` carries the
  request, the receipt, the tile hashes and the Copernicus attribution.

Two hosts serve byte-identical archives:

- `--source ncar` (default): upstream NCAR (`www2.mmm.ucar.edu`, a
  single server). `--bundle` fetches NCAR's whole
  `geog_high_res_mandatory.tar.gz` (**2.77 GB** compressed --
  2,772,782,816 B, measured 2026-07-30 -- ~29 GB unpacked) instead of the per-dataset tarballs and extracts only what
  you asked for; it is the fallback when per-dataset tarballs are
  unavailable.
- `--source hf`: a verbatim, attribution-carrying copy of the NCAR
  per-dataset tarballs on Hugging Face, republished for CDN bandwidth.

**Integrity, accurately stated.** NCAR publishes no checksums for these
archives. The pins woof enforces were computed from a TLS download
from UCAR on 2026-07-29 (sizes independently confirmed against the
server's own Content-Length headers, contents byte-compared against an
independently staged reference tree); the mirror republishes exactly
those pinned bytes. So a mirror download is verified end-to-end
against the pins, while *first-download* trust in the NCAR route
ultimately rests on TLS to UCAR -- there is nothing upstream to pin
against. If NCAR ever republishes an archive, the pin mismatch refuses
loudly; `--allow-upstream-drift` accepts the new bytes within a size
sanity band and records them as unpinned in the local manifest.

### Manual alternative (exact URLs)

Everything `fetch-geog` does can be done by hand. Download from
NCAR's WPS geographical data page
(<https://www2.mmm.ucar.edu/wrf/users/download/get_sources_wps_geog.html>),
tarballs under `https://www2.mmm.ucar.edu/wrf/src/wps_files/`:

| dataset directory | tarball | download size |
|---|---|---|
| `topo_gmted2010_30s` | `topo_gmted2010_30s.tar.bz2` | 174 MiB |
| `modis_landuse_20class_30s_with_lakes` | `modis_landuse_20class_30s_with_lakes.tar.bz2` | 31 MiB |
| `greenfrac_fpar_modis` | `greenfrac_fpar_modis.tar.bz2` | 945 MiB |
| `lai_modis_10m` | `lai_modis_10m.tar.bz2` | 2.5 MiB |
| `albedo_modis` | `albedo_modis.tar.bz2` | 74 MiB |
| `maxsnowalb_modis` | `maxsnowalb_modis.tar.bz2` | 4.2 MiB |
| `soilgrids` | `soilgrids.tar.bz2` | 824 MiB |
| `soiltemp_era5_1deg` | none: built by `woof fetch-geog --datasets soiltemp_era5_1deg` | 459 MB read |

`soilgrids` holds seven dataset directories under one parent --
`soilcomp`, `texture_top`, `texture_bot` and `texture_layer1` through
`texture_layer4` -- each with its own WPS `index`. The WRF static
builder's default soil categories are `texture_top` and `texture_bot`
(the same 16 categories, water 14, and tile layout as WPS's
`soiltype_*_30s`), and the MPAS static builder reads `texture_top`,
`soilcomp` and the four texture layers, so it is **required by both
doors** and part of the default `fetch-geog` set (~13 GB unpacked). Its
pin was computed the same way as the others, from a TLS download from
UCAR on 2026-08-23. NCAR's mandatory-fields bundle does not carry it,
so `--bundle` refuses it and the per-dataset route is the only one.

**Held out of the default set.** WPS's own soil-category tiles
(`soiltype_top_30s`, `soiltype_bot_30s`, 1.6 and 1.8 MiB) and
deep-soil temperature (`soiltemp_1deg`, 30 KiB) do not clear the
licence bar (next section). They stay selectable by name and are
fetched from NCAR, never the mirror: `woof fetch-geog --datasets
soiltype_top_30s,soiltype_bot_30s,soiltemp_1deg`, then
`geog_data_res = '30s'` (soil) or `'islscp_soiltemp'` (deep-soil
temperature) for the WRF builder, and `--soilcat-data STATSGO` /
`--soiltemp-data ISLSCP` for `rw_mpas_static`. Whenever one is staged
or read, the line naming its terms is printed.

Untar each into one root so the directories sit side by side
(each tarball already contains its directory), then point
`--geog-root` at that root or set `GPUWM_CASE_DATA_ROOT` so it
resolves as `${GPUWM_CASE_DATA_ROOT}/WPS_GEOG`. Each dataset
directory must contain its WPS `index` file.

### Provenance and attribution

The WPS geographical datasets are assembled and distributed by
NCAR/UCAR for the WRF ecosystem (no license or citation text is
attached to the download page). Every dataset in the default set has a
permissive source; a source that is not permissive, or whose licence no
one states, is said so here and held out of the default set. Underlying
sources and licences:

| dataset | contents | ultimate source and licence | default set |
|---|---|---|---|
| `topo_gmted2010_30s` | 30-arc-sec terrain elevation | USGS/NGA GMTED2010 (Danielson & Gesch 2011, USGS OFR 2011-1073); U.S. public domain | yes |
| `modis_landuse_20class_30s_with_lakes` | Noah-modified 20-category IGBP land use + inland lakes | NASA MODIS (MCD12Q1-derived); NASA data are free and open | yes |
| `greenfrac_fpar_modis` | monthly green-vegetation-fraction climatology | NASA MODIS FPAR | yes |
| `lai_modis_10m` | monthly leaf-area-index climatology (10 arc-min) | NASA MODIS | yes |
| `albedo_modis` | monthly surface albedo climatology | NASA MODIS | yes |
| `maxsnowalb_modis` | maximum snow albedo | MODIS-derived (Barlage et al. 2005) | yes |
| `soilgrids` (seven directories; `texture_top` and `texture_bot` are the default soil categories) | 16-category top-/bottom-layer soil texture, Noah-MP soil composition and four-layer texture | ISRIC -- World Soil Information, SoilGrids250m (Hengl et al. 2017, PLOS ONE 12(2): e0169748), from whose 2017 texture classes NCAR generated the WPS tiles; CC BY 4.0 (ISRIC's terms for SoilGrids products since 2019), attribution required | yes |
| `soiltemp_era5_1deg` | 1-degree annual-mean deep-soil temperature (2 m temperature reduced to sea level) | ERA5 (Hersbach et al. 2020, doi:10.1002/qj.3803), 1990-2019 hourly climatology as published by WeatherBench 2 (Rasp et al. 2024, doi:10.1029/2023MS004019), built locally by `woof fetch-geog`; Licence to Use Copernicus Products (<https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf>): free, worldwide, commercial use allowed, attribution required | yes |
| `soiltype_top_30s`, `soiltype_bot_30s` | 16-category top-/bottom-layer soil texture (WPS's default) | STATSGO inside CONUS (USDA NRCS, public domain) + the FAO-UNESCO Digital Soil Map of the World elsewhere. The DSMW's own notes (Version 3.6, Jan 2003, <https://swat.tamu.edu/media/116411/notes.txt>) say "(C) FAO/UNESCO, 1995 All rights reserved worldwide", "All proprietary rights ... shall remain the property of FAO" and "The user shall not reproduce for the benefit of third parties the data", with permission required from FAO's Publications Division. **Not permissive.** | no: by name only (`30s`, `--soilcat-data STATSGO`) |
| `soiltemp_1deg` | 1-degree annual-mean deep-soil temperature (WPS's default) | The Noah-MP TBOT READ_ME (<https://ral.ucar.edu/sites/default/files/public/product-tool/noah-multiparameterization-land-surface-model-noah-mp-lsm/tbot/READ_ME>): 1-degree annual-mean 2 m air temperature from ISLSCP-I (ECMWF 1987-88 analyses, adjusted to 1000 mb). The ISLSCP-I Vol. 4 record at CEDA (<https://catalogue.ceda.ac.uk/uuid/b9c3effaab9fe9a2206b58af500f157d>) states Permitted Use "Unstated" and links a missing licence file; NCAR states none. **Licence unclear, so not cleared.** | no: by name only (`islscp_soiltemp`, `--soiltemp-data ISLSCP`) |

If you publish work built on these fields, credit NCAR/UCAR's WPS
geographical data distribution and the underlying providers: USGS for
GMTED2010; NASA for the MODIS-derived fields; ISRIC for SoilGrids250m,
whose CC BY 4.0 licence makes that credit a condition of reuse; and, for
the deep-soil temperature, "Contains modified Copernicus Climate Change
Service information (ERA5); neither the European Commission nor ECMWF is
responsible for any use of this information", which the Copernicus
licence requires. A run that opts into a held-out dataset should also
credit FAO/UNESCO or ISLSCP-I, and is bound by their terms.

Against the held-out WPS field, the ERA5 deep-soil temperature is 1.9 K
warmer on average over common land cells (median 1.8 K, 5th to 95th
percentile -0.6 to +4.8 K). The largest differences are over high
terrain, where ISLSCP-I's 1000 mb adjustment and ERA5's sea-level
reduction part ways (+8.9 K at 30.5 N 85.5 E on the Tibetan Plateau,
+0.7 K at 38.5 N 98.5 W in Kansas). It also marks 27,542 land cells to
ISLSCP-I's 21,656: every cell ISLSCP-I has, plus small islands and
coasts.

`woof check` verifies the exact tiles your footprint intersects
(presence and hashes) before anything expensive runs. Static fields
arrive at 30 arcsec regardless of nest spacing unless `[static.highres]`
is enabled (below), except terrain on a domain at 1 km or finer, which
takes Copernicus GLO-30 by default when no `[static.highres]` block is
declared; no VAR_SSO/orographic-drag, urban-fraction, or
lake-depth datasets are produced in this release.

**Islands at 0 m.** GMTED2010 holds 0 m on atolls and cays that lie at
sea level, and on some islands it does not carry at all. A domain whose
land is all such shoreline prepares at 0 m and says so once. Land held
at exactly 0 m that has land on all four sides (an island three or more
cells across, or a landmass) is refused by name: it is either terrain
that never arrived or an island GMTED2010 lacks, and `[static.highres]`
with `fields = "terrain"` gives it its Copernicus GLO-30 height.

**Islands a source does not resolve.** A land cell whose source area
holds no land at all (an atoll in an open-sea crop of a coarse source)
takes the source's skin temperature of the sea there, and a soil column
at that temperature with its soil category's field capacity as its
water, where WPS writes a 285 K column saturated at 1.0. The preparation
log and the prepared cache header count those cells.

### High-resolution static sources (`[static.highres]`)

With `[static.highres]` enabled, terrain, land use and soil are replaced
from these sources, fetched on demand into the block's `cache_root`
(docs/public/HIGHRES-TERRAIN.md). None needs an account. Cells a source
does not cover keep the 30-arc-second baseline.

| field | source | resolution | published over | terms |
|---|---|---|---|---|
| land use (default) | CGLC-MODIS-LCZ (Demuzere, He, Martilli, Zonato 2023, doi:10.5281/zenodo.7670653), one 2.28 GB GeoTIFF fetched once and checked against its published size and MD5 | 100 m, 2018 | 60 S to 78 N, all longitudes | CC BY 4.0, attribution required |
| land use (`landcover_source = "annual-nlcd"`) | Annual NLCD Collection 1 (MRLC), the year nearest the case | 30 m | conterminous United States | public domain |
| terrain (default abroad) | Copernicus DEM GLO-30 | ~30 m | 90 S to 84 N | Copernicus DEM licence, attribution required |
| terrain (default in the US) | USGS 3DEP 1/3 arc-second | ~10 m | conterminous United States | public domain |
| terrain (on request) | SRTM 1 arc-second v3 (OpenTopography mirror) | ~30 m | 56 S to 60 N | public domain |
| soil texture | SoilGrids v2 (ISRIC WCS) | 250 m | global | CC BY 4.0, attribution required |

CGLC-MODIS-LCZ is built from the Copernicus Global Land Service LC100 v3
(Buchhorn et al. 2020) and the global Local Climate Zone map (Demuzere et
al. 2022, Earth Syst. Sci. Data 14, 3835).

### `GPUWM_CASE_DATA_ROOT` layout

The root is the directory that *contains* your data, not a dataset
itself:

```
$GPUWM_CASE_DATA_ROOT/
  WPS_GEOG/            <- the geog dataset directories
  my-case-data/        <- whatever your config's [case_data] names
```

`woof doctor` checks this layout and says exactly what is missing.

## Disk budget summary

| item | size | when |
|---|---|---|
| WPS_GEOG static tree (`woof fetch-geog`) | ~2.7 GB download, ~34 GB unpacked; `--datasets wrf` skips the mesh-only land use for ~2.6 GB download, ~28 GB unpacked | once |
| CGLC-MODIS-LCZ land cover (`[static.highres]`, default) | 2.28 GB | once per `cache_root` |
| GFS subsets | ~3.3 KB/deg2/h (e.g. ~3 MB/h at 30x30 deg) | per case |
| HRRR, default whole files | ~1.1 GB/h (~21 GB for f00..f18) -- `wrfnat` 704 MB + `wrfprs` 427 MB, measured | per case |
| HRRR, `--mode idx-subset` | ~0.44 GB/h (~8.4 GB for f00..f18), measured; saves bandwidth, costs wall clock | per case |
| ERA5 retrieval | tens of MB per valid time (regional box) | per case |
| Output frames (`wrfout`) | grid-dependent; ~198 MB/frame on a 250x200x49 domain; 20.1 GB for the four-domain 6 h reference run | per run |
| Restart checkpoints | ~5.7 GB per four-domain checkpoint on the reference case | per `restart_interval_s` |

Output fields and cadence are config-controlled; size a run's disk
before launching it, the way `woof check` sizes its VRAM.
