# 5. Initialization: the arbitrary-source engine

The 2.5.0 initialization system is table-driven end to end. A source is a registry
row (cadence, horizon, coverage, transport, mapping authority), not a code path; the
domain wizard, the fetch door, and the preparation stage all read the row. The
design law behind it: adding a future model must be metadata and table work, and a
per-model adapter file fails the test (section 1.6).

## 5.1 The source registry: ids, coverage, horizons

The registry (`woof/source_adapters.py`) holds 31 rows: 17 runnable (16
weather-model ids plus the generic `mapped` adapter) and 14 registered but not yet
runnable. Counting has to be stated precisely, because three defensible countings
exist: **16 runnable model source ids; folding product variants (HRRR native and
pressure-level, 20CRv3 member-GRIB2 and mean-NetCDF) they cover 14 distinct
upstream systems; 14 ids currently have a `woof fetch --source` door.** This
manual uses the id count and this table [woof/source_adapters.py, read live
2026-08-18; docs/public/SOURCES.md:20-37 publishes the same sixteen rows and gives
RAP's native spacing rounded to 32 km where the registry value is 32.463 km].

| id | aliases | boundary cadence | horizon | native grid |
|---|---|---|---|---|
| `hrrr` | | 1 h | f048 | Lambert 1799x1059 @ 3 km (CONUS) |
| `hrrr-prs` | `hrrr-pressure`, `hrrr-wrfprs` | 1 h | f048 | same Lambert grid |
| `gem-gdps` | `gem`, `gdps`, `gem-global` | 3 h | f240 | global |
| `icon-eu` | `dwd-icon-eu`, `icon-eu-regular` | 1 h | f120 | regular lat-lon 1377x657, 29.5-70.5 N, 23.5 W-62.5 E |
| `gfs` | `gfs-0p25`, `gfs-0.25` | 3 h | f384 | global |
| `gdas` | `gdas-0p25`, `gdas-0.25` | 1 h | f009 | global |
| `gefs` | `gefs-ensemble` | 3 h | f384 | global |
| `aigfs` | `ai-gfs` | 6 h | f384 | global |
| `aigefs` | `ai-gefs` | 6 h | f384 | global |
| `ecmwf-open-data` | `ecmwf`, `ifs` | 3 h | f360 | global |
| `aifs` | `aifs-v2`, `aifs-single` | 6 h | f360 | global |
| `rap` | `rap-awip32` | 1 h | f051 | Lambert 349x277 @ 32.463 km (North America) |
| `rrfs` | `rrfs-ops` | 1 h | f084 | Lambert 1799x1059 @ 3 km, bit-for-bit HRRR's grid, measured identical |
| `era5` | | 6 h | analysis only | global |
| `20crv3` | `20cr`, `twentycrv3`, `20crv3-member` | 3 h | analysis only | global |
| `20crv3-cf` | `20crv3-netcdf`, `20cr-netcdf`, `20cr-cf` | 3 h | analysis only | global |

The 17th runnable row is `mapped` (aliases `generic-mapped`, `mapping-v1`): no
cadence, no horizon, no coverage; it reads all three from the mapping document a
caller supplies.

**Certification is not uniform, and the registry says so per row**: `hrrr`, `gfs`,
and `era5` are `CERTIFIED`; every other runnable row is `RUNNABLE_NOT_CERTIFIED`
[woof/source_adapters.py, read live]. The 14 registered-but-not-runnable rows
(`hrrr-ak`, `nam`, `rrfs-a`, `rrfs-public`, `href`, `sref`, `rtma`, `urma`, `nbm`,
`refs`, `rrfs-firewx`, `hgefs`, `hiresw`, `wrf`) are refused by `woof domain` and
`woof fetch` with the registry status named, not "invalid choice"
[docs/public/SOURCES.md]. Every spelling a user can read is a spelling they can
type, held in both directions [tests/test_documented_source_spellings.py].

## 5.2 Fetch: which sources download, which refuse by name

`woof fetch --source` accepts 14 ids. `era5` downloads keyless from the ARCO
store with `--era5-provider arco`. With the CDS provider, it emits the exact
two-part `cdsapi` request (pressure-level z/t/u/v/r at all 37 levels plus 16
single-level fields), or retrieves it with `--retrieve` and configured
credentials. Validate the CDS retrieval with `--validate`
[docs/public/SOURCES.md; docs/public/DATA.md]. Ten sources ride a packaged route
table (`woof/authorities/rw-wps-fetch-routes.v1.json`, one engine); `gfs`, `gdas`,
`hrrr`, and `era5` keep hand-written transports predating the table.

Sources with no fetch door refuse by naming the concrete breakage: `20crv3`'s
every-member GRIB2 archive is not on an anonymously readable endpoint (only the
ensemble-mean NetCDF is, and a member state is not a mean); `20crv3-cf`'s per-year
per-variable NetCDF has no cycle and no lead for `--cycle`/`--hours` to describe;
`mapped` names no publisher. Each refusal points at the `--source-root` staging
door, and the wizard's next-steps block prints the acquisition note with the
window, hours, and spacing the config needs [docs/public/SOURCES.md; verified live
against the CLI 2026-08-18]. Behavior is pinned by tests: every runnable source
either resolves a route or refuses by name [tests/test_fetch_routes.py], one small
real object per route [tests/test_fetch_routes_live.py].

**Parallel fetch is the default** (`--fetch-workers`, default 6 in flight).
Politeness is engine policy: NOMADS capped at 2 in-flight over a node-wide 2.5 s
spacing governor, `Retry-After` honored. Concurrency changes no integrity
property: every file passes the same envelope walk, record bar, and SHA-256 as the
serial loop; one failed file refuses by name; an interrupted fetch records a
contiguous verified prefix; the manifest carries a `concurrency` receipt block
[docs/public/DATA.md; tests/test_fetch_pool.py]. Measured cold, identical byte
sets both ways: ICON-EU's 252 objects (233.7 MB) complete in 71.22 s under the
pool, and the GFS NOMADS subset (4 files, 1.14 MB) in 10.60 s, a wall the
politeness cap sets by design, not bandwidth. The serial arms live in the receipt
[docs/public/receipts/fetch-pool-cold-measured.json, measured 2026-08-17].

The receipt's own note: the sum of per-file service seconds rose under the pool
while wall fell, so the gain is overlap, not a warmed server cache. Those two
cases are the only measured serial-versus-pooled A/B pairs. Two further published
many-file walls, 35.8 s on ICON-EU's 127-object analysis state and 45.4 s on
GDPS's 177 objects, have no measured serial partner: their comparator is a
modelled serial arm, and the receipt schema's own field name for it is
`modeled_serial_seconds` [docs/public/DATA.md:409-411].

A table route's output directory is a front door, not a pile: `inputs.txt` (the
ordered input list), `prep-command.txt` (the bound half of the prep command,
runnable as written), `SHA256SUMS`, `fetch-manifest.json` [docs/public/DATA.md].
The other hand-written transports stop short of the full set: HRRR writes
`SHA256SUMS` and `fetch-manifest.json` and prints a bound prep fragment, without
`inputs.txt` or `prep-command.txt` [woof/fetch.py].
The GFS route, one of the four hand-written transports, joined the table routes
at that door in 2.5.0: `woof prep --source gfs` authors and digest-binds its
input manifest from the fetched directory, so the printed prep command runs as
printed, and the `--source-manifest` pair pins an existing manifest with half a
pair refusing [commit bdb8590c6]. Pasting a printed prep command a second time
re-authors nothing: identical bytes answer MATCHED with the same digest, and
differing bytes refuse naming both digests and the remedy [commit 95c838369].
Cross-source donors are fetched for you: AIGFS and AIGEFS publish no soil, land
mask, orography, skin temperature, surface pressure, or 2 m humidity, so the
packaged profiles bind those canonicals to the same-cycle GDAS analysis and the
fetch pulls the donor into `<out>/donor-gdas/` [docs/public/DATA.md].

Host choice can be a correctness question, carried as table data: AIGFS is
NOMADS-only on purpose, because the public S3 bucket publishes objects under
byte-identical key names that are a different product (experimental stream,
`subCentre` 2 against operational 0, 65 pressure messages against 78, 500 hPa
geopotential height differing by 8.7 gpm). Every packaged selector pins
`subcenter=0`, so the S3 copies refuse at decode, and `--transport aws` on that
source refuses and says this [docs/public/SOURCES.md].

## 5.3 The domain wizard plans from the registry row

`woof domain --source` takes any registered id or alias; nothing in
`woof/domain_wizard.py` names a model. A new registry row reaches the door with no
wizard code, proven rather than asserted: the test installs a synthetic registry
row and drives the real CLI, checking the emitted `namelist.wps` carries the
cadence the row declared and that a regional variant refuses outside the declared
window, with no code added anywhere [tests/test_wizard_sources.py].

The VRAM budget the fit runs against is declared or measured, never assumed:
`--card` and `--vram-gib` declare it, and with neither the wizard measures the
local card in a short-lived subprocess (no CuPy import; `GPUWM_NO_LOCAL_GPU=1`
has one definition and suppresses the probe before any card is touched) and
refuses naming both flags when there is nothing to measure. The silent 24 GiB
assumption is gone, an assumption not being a budget, and the wizard says when
the source rather than the card stopped the domain search, so a saturated fit
cannot look comfortable [woof/domain_wizard.py; CHANGELOG.md, engine 2.5.0;
receipt:RELEASE-CANDIDATE-2P5-2026-08-18.md]. The envelope every sizing gate
prices is the one measured formula of section 8.5.

Coverage is refused at plan time, in the same coordinates the preparation stage
would use on real bytes. A 3 km ladder centered on Kansas under `--source icon-eu`
exits 2 with nothing written, naming the point, the source index it maps to, the
covered window, and the three ways out (verified live 2026-08-18; the 2026-08-17
model battery paid a full ICON-EU acquisition and 73 s of preprocessing to learn
the same fact as a traceback [receipt:MODEL-BATTERY-6H-2026-08-17.md]). Where the
domain fits but the margined fetch box overruns, the box is clamped into coverage
with an advisory. The two doors deliver refusals differently and both are
pinned: the wizard's plan-time refusal exits 2 with nothing written, and the
preparation front door owns the staged-bytes classes (out-of-coverage grids,
too-short staged series): sentence plus remedy on stderr, exit 78, no traceback
[tests/test_source_coverage_refusal.py].

Other wizard refusals worth knowing: `--hours` past the row's horizon refuses with
the horizon named (GDAS stops at f009); a fitted domain reaching a pole refuses
naming the fitted size, but only where a smaller domain cannot help -- a drawn
`--polygon` that reaches the pole, and a `--point` so close to one that even the
minimum layout contains it.  A point carries no extent, so the fit chooses one
and bounds its own choice: at most `--point-extent-km` per axis (default
6,000 km), clear of the projection pole, and less than one trip around the
globe in longitude, all of which SHRINK the domain rather than refuse it
[docs/public/CLI-OPTIONS.md]. The last one matters on Mercator, which has no
pole to stop a tropical root from wrapping onto its own ground, and on
Lambert, whose root crosses it before it reaches the pole. An extent
below the smallest root the ladder hosts gets that smallest root, and the plan
summary says so rather than reporting a cap. None of this is a warning -- an
ordinary mid-latitude point on an ordinary card is sized by these bounds, so a
`warning:` would flag the normal case; the plan summary states which one bound
in one line beside the sizing line (`domain: point request: extent capped at
...`) and stderr stays clean. That domain is still continental, so its fetch
box is large and says so once: the advisory names `--point-extent-km` and
`--polygon`, the flags that ask a point for less or different ground, because
on these bounds a bigger or smaller card buys no different grid. The box widens with latitude as a conformal domain's corners
fan out (measured under the default 12 km ladder: 108 degrees of longitude at
30 N, 147 at 48.5, 179 at 58), and once it passes 180 the fetch takes the
source's full band with the forecast grid and latitude bounds unchanged -- a
separate warning, also at plan review. The
profiles whose actual radiation pairing has shortwave ON and longwave OFF
are refused for a window including local night unless explicitly acknowledged;
profile/route pairings that cannot be prepared refuse naming the missing
component, while a profile cadence landing on a fractional root step is snapped
to the nearest whole-step value and announced rather than refused (a
hand-written pair keeps the loader's refusal) [commit e8d818f43]; `--area` is only for the
four hand-written transports, since a table route takes whole published objects and
the crop happens at prep, where the namelist geometry is the crop
[docs/public/CLI-OPTIONS.md; docs/public/SOURCES.md].

## 5.4 The mapped route: run any dataset from a mapping you author

`woof prep --source mapped` takes an explicit declarative contract instead of a
built-in adapter: `--mapping` (strict `rw-wps.mapping.v1`, the
field/coordinate/target contract), `--composition` (the product-join contract),
`--descriptor` with `--author-mapping` and `--vtable` to compile a mapping from a
science contract, `--input`/`--input-list` in deterministic order, a SHA-256
manifest binding the files, provenance and supplement role bindings, and
`--mapped-engine {rust,python}` (default rust; python is a documented workaround)
[docs/public/CLI-OPTIONS.md]. `woof adapt` is the authoring front door
(`--skeleton` scaffold; `--descriptor` + `--vtable` + `--input` emits a runnable
adapter with provenance); `--vtable` is never defaulted, because quietly reaching
for a GFS Vtable would mis-map every other product [docs/public/CLI-OPTIONS.md].

The limit `woof adapt` stops at, quoted from its validation contract: "A
successful adaptation establishes that the emitted files implement your descriptor
exactly, and your GRIB files satisfy it. It does not establish that your descriptor
is a correct physical interpretation of those files." The contract carries a
per-item verdict table and names the highest-risk hand-check items: hPa vs Pa,
g/kg vs kg/kg, geopotential vs geopotential height, RH as fraction vs percent
[docs/adapt-validation-contract.md].

**The headline limit, as prominent as the headline: an authored mapping prepares
end to end but does not yet run through `woof sim`.** The forecast stage
certifies only the packaged mapping authorities, pinned by digest; a
caller-supplied mapping fails that pin, correctly, and the refusal was moved to
the door so the reader gets the limit rather than an internal hash mismatch
[docs/public/PIPELINE-STAGES.md; woof/prepared_single_domain_forecast.py:200-235;
tests/test_stage_seams.py::test_a_users_own_mapping_is_refused_at_the_door_naming_the_limit].
Closing this needs a second, narrower certificate for caller-supplied authorities;
until then, the arbitrary-source claim is "prepare arbitrary sources; run the
packaged ones."

## 5.5 Cross-model composition: borrowing fields between sources

All tables, no per-model code [docs/native-mapped-source-status.md]. The primary
mapping declares each borrowed field `"provider": "composition_bound"`, making the
gap a declaration (a mapping carrying one refuses to materialize alone, by name).
The composition's `field_sources` binds each gap to one contributing source: the
contributor's own sealed mapping pinned by SHA-256 inside the composition, the
field list, `grid_alignment` (only `exact_coordinate_subset` lands), and a time
alignment from a closed set (`valid_time_exact`, `cycle_invariant_broadcast`,
`source_cycle_analysis_broadcast`, the last being the hybrid clock: exactly one
donor analysis record whose valid time must be the primary's source cycle, carried
to every lead, every carried time named in the receipt). Refusals each name their
breakage: cross-grid contributions name the missing regrid capability;
member-bearing donors name member alignment; vertical borrows on a different
ladder name vertical interpolation; double provision names its two providers;
wrong-cycle donors, hash mismatches, and unbound gaps all refuse. Provenance
receipts gain a `contributing_sources` block, and single-source receipts are
byte-identical to before.

Measured on real staged bytes 2026-08-17: an AI-atmosphere primary (0.25 degree,
13 levels, no soil, no land mask, no orography in the product) composed with the
same cycle's physical 0.25 degree analysis materialized the complete 16-field
WRF-real canonical set with 4-layer soil in 9.4 s; the 1.0 degree file of the same
cycle refused naming the regrid capability, and the 06Z analysis under the 00Z
primary refused naming the source-cycle rule
[tests/test_cross_source_real_bytes.py; tests/test_cross_source_composition.py].

Shipped consumers of the grammar (the "AI models run with GDAS soil out of the
box" claim, concretely):

- `aigfs`: the hybrid profile ships each donor mapping as a pinned authority,
  forwarded automatically and re-verified by the prepared runner's certificate.
  Measured end to end on real 2026-08-17 00Z bytes: prep composed and initialized
  the 30 km demo domain in 11.9 s (decode+compose 7.5 s); `woof sim` ran the
  six-hour window to PASS in 14.8 s wall on an RTX 5090, with
  `prepared_content_sha256` identical across two independent preparations
  [gallery:aigfs-hybrid-sim-20260817/].
- `aigefs`: a 12-parameter member atmosphere with six `composition_bound`
  land-surface gaps bound to the packaged analysis-donor mapping. Measured on the
  real 2026-08-17 00Z cycle: control plus two perturbed members each prepared and
  ran six GPU forecast hours to PASS (34.5-45.2 s wall at 30 km, 50x50x49), with
  `rw_ensbatch` panels drawn from the three PASS runs
  [tests/test_aigefs_member_hybrid_real_bytes.py].

One donor-side fact the grammar forced into the open: a borrowed field must be
directly selected in the donor, and the checked-in GFS table derives 2 m specific
humidity from RH, so the profile's donor mapping is that table with one table-data
change (a different GRIB selector at 2 m), not an engine change
[docs/native-mapped-source-status.md].

## 5.6 Ensemble members as source data

An ensemble enters as one SHA-256-pinned `rw-wps.members.v1` document plus a
registry row: layout, declared member count, member classes (control/perturbed,
each with ordinals, id and token templates, and a verification block declaring the
GRIB ensemble identity octets), and a statistics section for means and spreads
that share the namespace but are never members [tests/test_member_grammar.py].
Members are fetched by filename but verified at decode on the GRIB ensemble
identity octets, failing closed on a mismatch; a mean or spread claimed as a
member refuses by name [tests/test_member_prep.py;
tests/test_member_identity_real_bytes.py]. `woof fetch --member ID` selects the
member on the ensemble routes (`gefs`, `aigefs`); `woof-member-prep` stages and
verifies (`--list-member-sets`, `--describe`, `--member-set/--member/--cycle`,
`--verify-only`) [woof/member_prep.py:530-575; note this front door is not yet in
the generated public CLI reference, a known documentation gap].

Measured: the GEFS control member prepared and ran 6 h to PASS in the model
battery (247 s wall for the 7-frame run) [receipt:MODEL-BATTERY-6H-2026-08-17.md];
three real GEFS members (control plus two perturbed) prepared through the generic
mapped route ran 3 h each on an RTX 5090 with `rw_ensbatch` panels; the 3-member
spread cross-checked against NCEP's own published spread file for the cycle (NCEP
global mean 0.34 K / max 3.3 K; this domain 0.37 K / 2.8 K). This is a
sanity check that three members differ by amounts of the same order as
NCEP's spread, not an ensemble-calibration or spread-skill test. Those
would require more members, cases and observations
[gallery:gefs-member-ensemble-20260817/]. Chapter 6 covers ensemble products.

## 5.7 The model battery: one domain, eleven initializations

Receipt: [receipt:MODEL-BATTERY-6H-2026-08-17.md];
renders [gallery:model-battery-6h/, 120 PNGs, sha256-verified 0 mismatched 0
missing]. Conditions: one shared 300x300 domain at 3 km, 49 levels, over the
central United States, one physics slice, 6 h forecast, serial on an RTX 5070 Ti;
only the initialization source differs. Each run was checked on its
output: frames, field sanity checks and `rw_wrfbatch` renders. PASS means
completion and those output checks, not accuracy against observations
or agreement with another model. The same meaning applies to "ran to PASS"
in sections 5.5 and 5.6; no forecast in this battery was scored against
observations. The
report does not name the physics slice beyond "one physics slice", so this manual
does not either.

| model | verdict | wall (7 frames) | detail |
|---|---|---|---|
| 20CRv3 (one reanalysis member) | PASS | 233 s | a 1932 analysis; the only pre-satellite-era run |
| AIFS | PASS | 246 s | T2 287-313 K; max composite refl 55 dBZ at h3 |
| ECMWF IFS open data | PASS | 257 s | max abs w 13.4 m/s |
| RAP | PASS | 273 s | every record JPEG2000; maxdbz 59.8; updrafts to 32 m/s |
| GEM (GDPS) | PASS | 366 s | 60 dBZ cores |
| GEFS (c00) | PASS | 247 s | pgrb2a/b pair; updrafts 14.5 m/s |
| AIGFS (+GDAS donor) | PASS | 249 s | CFL 0.39; updrafts 17.2 m/s |
| GDAS | PASS | 322 s | quietest arm: max abs w 3.3 m/s, maxdbz 25.7 |
| AIGEFS (mem000 + donor) | PASS | 282 s | CFL 0.56; updrafts 18.3 m/s |
| RRFS | PASS | 459 s (slowest, all in prep; 6.3 GiB input) | most convective arm: abs w 26.1 m/s, ~75 mm rain by h6 |
| ICON-EU | named refusal | 73 s, rc 1 | European-only grid cannot reach a central-US domain |

The refusal is a result, not a failure: the message names the offending corner,
the source index it mapped to, and the window the source covers, enough to tell
"source does not reach the target" from "crop too small"; deterministic on re-run;
not a decode failure (all 1,752 ICON-EU objects decoded before the geographic
check). A representative ICON-EU forecast needs a European domain, a run nobody
has made yet.

The host-memory scale bound this section used to carry is RETIRED. It read: a
3 km CONUS RRFS preparation carrying 7 valid times peaks near 107 GiB of host
RSS, passes on the 123 GiB nodes and would exhaust smaller boxes, with the
registered fix direction being streaming per-valid-time composition
[receipt:RELEASE-CANDIDATE-2P5-2026-08-18.md]. That fix has landed: the mapped
engine now inventories the messages first and then reads, decodes, assembles,
composes, materializes, writes and DROPS one valid time before asking for the
next, so the peak follows one valid time rather than the number of them. The
same RRFS preparation re-measured end to end peaks at 26.9 GiB at 7 valid
times, rising 0.04 GiB per forcing hour [work/plot_rss_curve.py]. Measured
again on ICON-EU public DWD bytes, decoded and composed through the same
engine, a 3 h window (4 valid times) peaks at 2.32 GiB and a 6 h window (7
valid times) at 2.32 GiB, against 7.21 and 12.68 GiB before the change: the
per-lead-hour slope falls from 1.82 GiB to 0.0003 GiB, and both windows write
byte-identical `frames.json` and `frames.f64`.

What still scales with the series, and is a smaller bound rather than none:
only GRIB2 is sliced per valid time -- every other declared format is decoded
whole and then carved, so a NetCDF or GRIB1 series still holds its whole decode
once; a composition's contributing donors and terrain supplements stay resident
for the whole compose, because the frames borrow from them one valid time at a
time; and `inspect` decodes the whole series by design.

One timestamp caveat: the battery's limitation list is dated: its "no
fetch door for six sources", "wizard accepts only gfs/hrrr/era5", and "`--source
gem` does not resolve" findings are all closed on the release line
[tests/test_wizard_sources.py; tests/test_fetch_routes.py], and its ICON-EU
traceback delivery defect is fixed (exit 78, no traceback)
[tests/test_source_coverage_refusal.py]; quote the battery for its runs and
verdicts, not its gap list. The front doors themselves have since been proven
directly: the battery re-fired bare-default at the assembly tip b63c9f9c3 on
2026-08-18 through the real fetch, prep, 6 h simulation, and render chain on
both GPU nodes, with no decoder-tool flags anywhere; test host 2 (RTX 5090) ran rap,
gem, gdas, aigefs, and rrfs to PASS, and test host 1 (RTX 5070 Ti) ran aifs,
ecmwf-ifs, gefs, and aigfs to PASS with icon-eu the designed named refusal, a
clean sentence naming the exact out-of-grid coordinates
[receipt:battery-refire-2p5-20260818/CAPTIONS.md, render sets sha256-verified
per node; FINAL.summary.json held on each node]. The 20CRv3 member arm is
separately proven on the engine route by the member dual-run; its numbers are
in section 6.6.

## 5.8 Reachability qualifier

The single-command chain (`woof go`, fetch through render) orchestrates GFS only
(`ORCHESTRATED_SOURCES = ("gfs",)` [woof/go_cli.py:115]). Every other runnable
source runs stage by stage: `woof fetch` (or staged bytes), `woof prep`,
`woof sim`, `woof render`, with the fetch output directory binding the prep
command for you. This is the largest reachability qualifier on the
arbitrary-inputs headline and belongs in the same paragraph as the headline.
