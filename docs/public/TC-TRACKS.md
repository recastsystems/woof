# Tropical cyclone tracks from completed output

`woof tc-tracks` consumes completed fields. It writes `tracks.json`, `tracks.csv`,
`manifest.json` and a timing/provenance `receipt.json`. It does not run a forecast.

```bash
export WOOF_EARTH2_PYTHON=/path/to/earth2studio-environment/bin/python
woof tc-tracks --input earth2-fcn3-20240924T12.nc \
  --best-track best-track-AL092024.txt --out tracks/ --render
woof tc-tracks --manifest members.json --backend vitart --workers 4 --out tracks/
WOOF_DETECTNODES=/path/to/DetectNodes WOOF_STITCHNODES=/path/to/StitchNodes \
  woof tc-tracks --manifest members.json --backend tempestextremes --workers 24 --out tracks-te/
```

The default backend is `wuduan`. Earth2Studio is imported only by a subprocess of
`WOOF_EARTH2_PYTHON`. Both GPU backends use one completed valid time per call and
reset the path buffer between members. Their 300 km search distance and two-step
history are recorded with each member's actual cadence. `--workers` bounds native
field-extraction processes and TempestExtremes workers. Detection divides the budget
between members and disjoint valid-time groups, then stitches each full time series once. Extraction divides
available CPU threads among its processes. GPU members run sequentially on the
assigned GPU; use the site's resource allocator before invoking the command.

The interpreter needs Earth2Studio's cyclone dependencies, including CuPy,
cuCIM, SciPy and scikit-image. Importing the tracker class alone does not check
these optional dependencies. For a read-only Earth2Studio environment, install
compatible dependency wheels into a separate directory and name it with
`WOOF_EARTH2_EXTRA_SITE`. Only the tracker subprocess prepends it to `PYTHONPATH`.
Keep the supplied interpreter and environment
unchanged. Match the CuPy/cuCIM CUDA major version to the installed runtime.

The native helper is `rw_tc_fields`; the renderer is `rw_wrfbatch`. Bundled bridges
are discovered normally. A source build can name them with `WOOF_RW_TC_FIELDS`
and `WOOF_RW_WRFBATCH`. No Python atmospheric-array decoder is used by this door.

## Manifest

```json
{
  "schema": "gpuwm.tc-tracks-manifest.v1",
  "members": [{
    "member_id": "ai-000", "family": "ai", "source": "earth2-fcn3",
    "init": "2024-09-24T12:00:00Z", "weight": 1.0,
    "ancestry": {"cycle": "2024-09-24T12:00:00Z"},
    "vocabulary": "earth2studio", "file": "forecast.nc"
  }]
}
```

Families are `ai`, `arwen`, `hex` or `analysis`. Paths are relative to the manifest.
Instead of `file`, supply `frames: [{"path":"frame.nc","valid":"2024-09-24T18:00:00Z"}]`
and an explicit `init`. Each frame can override `vocabulary` and supply a `select`
object of dimension names to integer indices. A file's independent initialization,
lead-time and ensemble axes expand into separate members. Ensemble identity is
retained in the member suffix and each frame's selection. Explicit weights are
split across expanded members. Ancestry is carried through as arbitrary JSON.

Omitted weights default to `1/N` **within each family**. All weights in a family
must be explicit or all omitted. Family totals are never combined into a single
probability distribution. Tracks retain their member's weight, including tracks
with missing fixes. Multiple candidates from one member are not independent members.

`--input` is a one-path shortcut for an Earth2Studio NetCDF, an ERA5-name Zarr, one WOOF-ArWen GRIB2 valid time, or
a WOOF-ArWen run directory. A run without an exported Zarr invokes `woof ml-export`
with the required pressure-level fields and earth-relative winds. Use a manifest
for explicit source/family/ancestry, GRIB2 frame series, or converted hex output.

## Fields and regional coverage

`woof/authorities/tc-track-fields.v1.json` is the field authority. New model names
that use a vocabulary need no reader change. A new export vocabulary is a new row
with variable names or GRIB selectors, pressure-coordinate selections, input units,
scale and offset. The reader branches only on storage format.

| Vocabulary | Wind | Vertical fields |
| --- | --- | --- |
| `earth2studio` | `u10m/v10m/u850/v850`, m/s earth-relative | `z200/z300/z500/z850`, geopotential m2/s2; `t200/t250/t300/t400/t500`, K |
| `era5-zarr` | `u10/v10` and `u/v(level)`, earth-relative | `z(level)` in m2/s2 and `t(level)` in K |
| `arwen-grib2` | UGRD/VGRD, export with `--winds earth` | HGT in gpm multiplied by 9.80665; TMP in K |

All use mean sea-level pressure in Pa. Grid-relative GRIB winds are rejected.
WuDuan needs surface pressure/wind and 850 hPa wind. Vitart additionally needs its
warm-core temperature and height levels. TempestExtremes uses the pressure, surface
wind and 300/500 hPa height contour tests declared in the same authority.

The native reader interpolates onto the covered 0.25-degree lattice. GPU trackers
require global 721x1440 arrays. Complete global inputs use the upstream detector on its native lattice.
For a crop the driver embeds only real cells and
carries an explicit mask. Filled cells have zero vorticity and wind and infinite
pressure during detection. A one-cell derivative halo is excluded. Returned
centres outside real support are rejected. A fix within 300 km of a coverage edge
gets `edge`. Receipts count real, filled and rejected cells. This route cannot
recover a storm outside the crop.

TempestExtremes uses the actual crop with `--regional`, never a global fill. It
flags fixes within the configured maximum contour/search footprint of an edge.
Its executables are optional and external. The lane build used upstream commit
`901cea1e4d6d061d505298ee67d6427a68b583cf`, BSD-2-Clause (CMake support files use
Boost Software License 1.0). DetectNodes runs one process per member, with many
members assigned to the bounded worker pool.

## Output contract

The contract shared with `cyclone-setup --storm auto` is exactly
`gpuwm.tc-tracks.v1`:

```json
{
  "schema": "gpuwm.tc-tracks.v1",
  "tracker": {"backend":"wuduan","config":{},"config_sha256":"...","code_revision":"..."},
  "members": [{"member_id":"ai-000","family":"ai","source":"earth2-fcn3",
    "init":"2024-09-24T12:00:00Z","weight":1.0,"ancestry":{},
    "inputs":[{"path":"forecast.nc","sha256":"..."}]}],
  "tracks": [{"track_id":"ai-000-000","member_id":"ai-000","weight":1.0,
    "fixes":[{"valid":"2024-09-24T12:00:00Z","lead_hours":0,
      "lat":20.0,"lon":-85.0,"mslp_pa":99000.0,"max_wind10_ms":30.0,"flags":[]}]}]
}
```

Optional track identity fields `atcf_id`, `storm_id`, `name`, `storm_name` and
`aliases` let `cyclone-setup` select a labelled system explicitly. Trackers do
not invent observed identities for unlabelled detections.

Longitude is in -180..180. Every sampled valid time remains present. Missing fixes
keep valid/lead time and have null latitude, longitude, pressure and wind, with
`flags:["missing"]`. Finite fixes can have `flags:["edge"]`. Tracks are never
silently shortened to successful detections. CSV has exactly these columns:

```text
member_id,track_id,weight,valid,lead_hours,lat,lon,mslp_pa,max_wind10_ms,flags
```

CSV null quantities are empty cells and flags are JSON arrays. JSON also carries
optional `best_track` and `scores`; provenance metadata has no CSV equivalent.

## Verification and pictures

`--best-track` accepts NHC ATCF BEST b-decks or HURDAT2. Format columns and
coordinate scales are declared in the authority. ATCF special-fix minutes and
HURDAT2 landfall markers are retained. Use `--storm-id AL092024` for a full
multi-storm HURDAT2 file. Public sources are the [NHC archive](https://ftp.nhc.noaa.gov/atcf/archive/2024/)
and [HURDAT2](https://www.nhc.noaa.gov/data/#hurdat).

Each member is scored at common valid times. The nearest candidate is explicitly
associated for centre distance in km and central-pressure error in hPa, within
the authority's explicit `verification.association_max_distance_km` (500 km).
This verification choice is recorded in scores; it is not a detector threshold.
An unrelated distant cyclone cannot stand in for a missing target. Misses
remain counted, and all raw candidates remain in the output. This association
can select different track IDs and is recorded in scores. Model instantaneous
maximum 10 m wind and NHC maximum 1-minute sustained wind are different quantities.

With a best track the Rust renderer also finds ocean-to-land coastline crossings
between adjacent finite fixes. The receipt reports distance and time error against
HURDAT2 landfall fixes, along with the input time bracket. Linear interpolation
cannot establish timing finer than the sampling cadence. A missing crossing is
null. ATCF files without landfall markers cannot supply that comparison.

`--render` draws coloured member tracks, weights in the legend and NHC in black
through `rw_wrfbatch --tc-track-plume tracks.json OUT.png`. Missing fixes break the
lines. Python dispatch uses `woof.rustwx.renderer_env()` for basemap assets. No
matplotlib or cartopy map is involved. The same option writes an intensity chart using
matplotlib, with the two wind definitions labelled separately. Matplotlib is
needed only for that optional intensity product.

Native float staging and temporary NetCDF/Zarr are deleted on success or failure;
`deleted-files.json` records their paths and sizes. Small requests, logs, output
tracks and receipts remain for review.

## Track-stratified source member selection

`woof tc-strata` selects source trajectories from a single parent family's saved
`gpuwm.tc-tracks.v1` document. It never runs an AI model or drops a parent with no
associated target. Use an initial fix with an explicit UTC time, or `--storm auto`
for cyclone-setup's existing lowest-window-pressure selection independently within
each member. A target fix associates the nearest candidate at that exact common
valid time within the authority's radius, with lexical track-id ties. A missing
initial fix is no storm; it is never extrapolated from a later system.

```bash
export GPUWM_NO_LOCAL_GPU=1 CUDA_VISIBLE_DEVICES=-1
export WOOF_RW_WRFBATCH=/path/to/rw_wrfbatch
python -m woof tc-strata --tracks tc-tracks-32.json \
  --target 19.25,-84.0,2024-09-24T12:00:00Z --hours 48 \
  --select 6 --seed 377778 --workers 4 --author-nests --render --out selected/
python -m woof ensemble selected/member-CONFIG/experiment.toml \
  --recipe member-roster --members 6 --trajectories selected/roster.json --dry-run
```

Use an actual selected member configuration path printed in `roster.json` in place
of `member-CONFIG`. If tracking labels differ from source-member tokens, supply
`--member-map source-member-map.json`, a JSON object mapping each tracking label
to its original source member token. Otherwise the token must be explicit in a
member's `member` or `ancestry.member`; labels are never parsed as fetch tokens.
Deterministic sources without a member grammar retain a null source member.

`woof/authorities/tc-track-strata.v1.json` declares the 500 km association radius,
fixed longitude sector edges, hours-since-window-start landfall bins, minimum-MSLP
bins, default six-stratum cap, merge/allocation rules and 12/3 km nest settings.
Use `--rules FILE` for another declared authority, `--max-strata N` for a recorded
cap override, and `--start UTC` to declare another window. Native renderer
coastline intersections supply the first ocean-to-land crossing in that window.
No-landfall and no-storm begin as explicit strata. The cap also respects K:
merge the smallest pair sharing an outcome first, with lexical ties; then merge
storm outcomes if necessary. No-storm remains separate. K=1 with both storm and
no-storm mass is refused because it cannot retain both populations.

Every nonempty final stratum receives one representative. Remaining slots use
capacity-bounded largest remainder proportional to population count, with lexical
ties. SHA256 ordering of `[seed,stratum,member_id]` samples without replacement,
independent of input order and Python RNG versions. Inclusion probability is
`k_s/N_s`. Each selected member carries `stratum_parent_mass/k_s`, giving
`N_s/(N*k_s)` for equal parent weights. Input weights are normalized once; all
must be finite and positive, from one family. This is a stratified representative
estimator with equal contributions within each stratum, not calibrated event probabilities.

`selection.json` (`gpuwm.tc-selection.v1`) saves input and rule hashes, effective
rules, code revision and selector digest, association decisions, raw stratum
components, every merge, parent membership and mass, allocations, inclusion
probabilities and selected weights. `landfalls.json` preserves the native crossing
receipt. `selected-tracks.json` retains only associated selected tracks and all
selected parents, including no-storm parents, with reweighted masses. The native
plume legend shows member weight and stratum; the existing ATCF/HURDAT2 readers
supply black best track through the input document or `--best-track FILE`.

`--author-nests` uses cyclone-setup's existing `select_track` and `plan_cyclone`
functions to author each member's pinned experiment and WPS companions. Each
associated member uses its own track corridor and seed. A no-storm member uses
the explicitly supplied target fix, recorded as such; auto-only selection with a
selected no-storm member needs a declared placement and is refused. Individual
ordinary-tree memory envelopes and their conservative sum are in `nest-plans.json`.
These are CPU estimates, not measured GPU allocation or a packing qualification.

The roster preserves source, cycle, source member, weight, stratum, track identity
and pinned configuration. The existing ensemble door reviews each member's own
configuration and source preparation chain. Its `--dry-run` prints both domains,
weight and stratum, and stops before hardware admission.

To execute on the forecast host, use the same command without `--dry-run` and
without the CPU-only environment variables above:

```bash
unset GPUWM_NO_LOCAL_GPU CUDA_VISIBLE_DEVICES
python -m woof ensemble selected/member-CONFIG/experiment.toml \
  --recipe member-roster --members 6 --trajectories selected/roster.json \
  --outdir ensemble-selected/
```

A roster carrying strata and saved tracks automatically requests cyclone
products through `ensemble/cyclone_products.py`. Optional `--cyclone-products
cyclone.json` overrides that request, for example:

```json
{"workers":4,"backend":"tempestextremes","best_track":"best-track.dat"}
```

Relative best-track paths resolve beside that JSON. ATCF, HURDAT2 and a saved
`gpuwm.tc-tracks.v1` document containing `best_track` are accepted. Without an
explicit reference the roster's saved best track is reused, when present. The
post-run `woof tc-tracks --manifest ... --backend tempestextremes --render` call
uses WOOF-ArWen parent histories, roster weights and strata. Its exact command and
stages are saved under `forecast/cyclone/`. All members, including members with
no detected storm, remain in the track manifest. The plume labels member weights
and strata and draws supplied best track in black.

Gridded means, threshold probabilities, compound-event probabilities, spread and
probability-matched means use roster weights normalized over the frame's member
inventory. Nonfinite cells remain masked under the existing complete-roster
policy. Spread uses reliability-weighted sample variance,
`sum(w*(x-mean)^2)/(W-sum(w*w)/W)`, with zero spread for one member. Equal masses
use the existing count arithmetic. PMM draws from the pooled distribution with
member masses; extrema, finite counts and member identity layers keep their
existing meanings. Reach-exceedance probabilities also use member masses.
Weights, normalization and spread policy appear in each frame/product receipt
and in the aggregate NetCDF metadata.

Members with different parent or moving-nest coordinates retain separate geometry
hashes. Their gridded aggregate products are marked unavailable per product with
the coordinate mismatch reason. Each member's reflectivity, 10 m wind and MSLP
maps still render from its own parent and moving-nest histories. After durable
maps and tracks exist, the writer ledger verifies and retires raw histories;
consumer receipts and per-file deletion sizes remain. A failed product stage
keeps the histories for diagnosis. Forecast hosts need a rebuilt native ensemble
reducer advertising `member-weights-v1`; the native capability check names a stale
binary before it can publish equal-count probabilities for weighted input.

A short moving-nest acceptance forecast can reuse a selection based on the full
saved parent-track window. Author each selected configuration through the same
cyclone planner for three hours, retaining the saved member identities and
weights. The forecast duration comes from those configurations; a full-window
parent plume is guidance for selection, not verification of the short nested
forecast. The CPU roster dry-run reviews all plans before hardware admission.

Each trajectory may declare `"fetch_root": "/path/to/completed-fetch"`. The
member config binds that candidate as `[fetch].reuse_fetch`, equivalent to
`woof fetch --reuse-fetch DIR`. A covering authored forcing window is retained
when the run starts at its cycle, so a three-hour forecast can use six-hourly
bracketing fields. Earth2 reuse checks the model row and table hash, UTC cycle,
initial-condition source, member, time and crop coverage, recorded frame and
donor digests, and native member/seed attributes before publishing the ordinary
preparation handoff. The model runtime is not loaded on a verified match.
The shared payloads remain in place; only receipts and input lists are published
in the managed member download. A missing or nonmatching receipt follows the
ordinary acquisition path; corrupt recorded payloads name the mismatching file.
`--force-refetch` bypasses reuse.

To bind an existing local population without changing configuration hashes:

```bash
python3 - roster.json /path/to/completed-fetch roster-reuse.json <<'PYTHON'
import json, pathlib, sys
roster = json.loads(pathlib.Path(sys.argv[1]).read_text())
for trajectory in roster["trajectories"]:
    trajectory["fetch_root"] = str(pathlib.Path(sys.argv[2]).resolve())
pathlib.Path(sys.argv[3]).write_text(json.dumps(roster, indent=2) + "\n")
PYTHON
woof ensemble /path/to/first-member/experiment.toml --recipe member-roster \
  --members 6 --trajectories roster-reuse.json --dry-run
# Use the same command without --dry-run for execution.
```
