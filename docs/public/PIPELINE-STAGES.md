# The pipeline, unbundled

WOOF-ArWen's pipeline is three stages, and each one runs on its own terms:

| stage | command | takes | leaves |
|---|---|---|---|
| preprocessing | `woof prep` | your source files, your `namelist.wps`, your experiment TOML | a **prepared tree** |
| simulation | `woof sim` | a prepared tree | `wrfout` frames + `report.json` |
| rendering | `woof render` | `wrfout` frames | product PNGs |

`woof go` still exists and is still the recommended first command for
a new user: it runs the documented GFS chain end to end and carries
each stage's digests to the next so you never copy a hash by hand. What
this page documents is that `go` is a **composition** of the three
stages above, not the only way to reach them. Every stage is invocable
alone, with inputs you supply, from a script you own.

If you are integrating WOOF-ArWen into an existing pipeline -- you pull your
own data, you author your own namelist, you have your own scheduler and
your own plotting -- these three commands are your interface, and this
page is the contract. You should not have to read our source to write
to it.

> WOOF-ArWen is a research and educational tool, never a substitute for
> official warnings from your national meteorological service.

---

## The boundary object: a prepared tree

The prepared tree is what preprocessing writes and what the simulation
reads. It is a directory, and the only thing a caller has to know about
it is this:

**A finished preparation leaves a top-level document -- `proof.json` for
a single domain, `receipt.json` for a domain tree -- whose `schema`
field names the preparation that produced it.** That document is how
`woof sim` learns which source prepared the tree and whether it is one
domain or a domain tree; it is why `sim` needs no `--source`, no cycle,
no area and no config to identify what it is looking at.

Every route publishes one, `--source hrrr` included: the native HRRR
preparation writes its own completion receipt
(`public-wrapper-result.json`) beside a `proof.json`, an
`experiment.toml` and a `namelist.wps`, so the two commands below are
the whole route.

A directory with neither a bindable document nor a route's completion
receipt is a partial or interrupted preparation, and `woof sim` refuses
it rather than running it. That refusal is deliberate: a half-written
tree is the one input that can produce a forecast which looks finished
and is not. A tree whose completion receipt says the preparation
FINISHED but that carries no portable authorities is a different thing
and gets a different answer -- the route's own recorded reason, and the
command that publishes them -- because telling a reader their complete
tree was interrupted sends them hunting a crash that never happened.

The digests inside that document -- the input-manifest hash, the
prepared-cache content hash, and for a domain tree the preparation
receipt -- are what the forecast stage binds against. `woof sim` reads
them off the file and hands them to the runner. **The runner still
recomputes every one of them and still refuses on any difference.**
Nothing is weakened by the relay; what you no longer have to be is a
checksum courier.

### A preparation the forecast can start before it finishes

Boundary interval k needs only forcing times k and k+1, so a
single-domain preparation (GFS, ERA5 and ARCO, and every mapped source,
HRRR-on-mapped and 20CRv3 included) is written start first and
published in three parts:

- **the head**: everything the start time makes -- the static fields,
  the receipts, the start state in `prepared-cache/` -- published with
  the route's usual single rename, plus `boundary-stream/head.json`,
  which carries the full interval schedule and the proof without its
  seal keys. Its `head_sha256` is what a forecast binds with
  `--prepared-head-sha256`;
- **one segment per interval**: the interval's arrays written into
  `prepared-cache/` under the file numbers a finished cache gives them,
  then `boundary-stream/segments/NNNNN.json` last. The marker is the only
  ready signal, so the rule holds across processes and machines: copy
  the arrays first and the marker last;
- **the seal**: the same `prepared-cache/header.json` a one-shot
  preparation writes (so `content_sha256` is unchanged), the companion
  WRF files, then `proof.json` last. `proof.json` gains one field,
  `boundary_stream.head_sha256`.

`woof go` (the GFS chain) and `woof run-plan` (the staged mapped chain)
start the forecast as soon as the head exists, and `woof sim` pointed at
a preparation that is still being produced binds its head the same way.
A chain binds only a head its own preparation wrote, so a retry never
starts on the head a failed attempt left in the same output root. The forecast waits at an
interval seam only when that interval is not prepared yet, and says so in
`progress.json` every 5 s (`waiting.reason`, "boundary interval k (...)
is not prepared yet"). At the end of the run it waits for the seal,
checks the sealed cache against the head and every segment it read, runs
the complete preflight on the sealed tree, and only then writes its
report; `report.json` gains `input.boundary_stream`. A producer that
fails ends the forecast with the producer's reason (`failed.json`), and a
producer on a thread of the same process that ends without writing it
(a full disk) ends it too; a producer in another process that stops
refreshing `producer.json` ends it by name. A forecast that fails leaves
the producer running to its seal, so a retry reuses the complete
preparation, and under `woof run-plan` the run's `run-progress.json`
says `waiting:preparation` until the producer ends; only an interrupt of
the chain writes `stop.json`, and the producer then exits unsealed. The next preparation into the same output
root removes such an unfinished tree and builds it again.

The head is published early only when the forecast and the producer fit
on one machine. Host RAM is checked for every preparation: the forecast
process holds the head's arrays, every boundary interval it loads (the
whole series, by the end of the run) and its own working set of about
2 GiB (the interpreter, the CUDA context and the physics tables) while
the producer keeps building, so those three plus what the producer's
builds took above what it keeps must fit the available RAM with 10%
headroom. A GPU preparation must also fit the card: the forecast's
estimate plus the memory the start time's build took, with 10%
headroom. Otherwise the preparation prints the numbers and publishes at
the seal, and the run starts after it, as before. A single native HRRR
domain and a native HRRR domain tree chain. Run as posted (`woof go` and `woof run-plan`, unless
`--whole-cycle`), its decoder reads each hour once the fetch has verified
it, and its head is published on the window's first two hours; the
decoded bridge's and the fetch's SHA256SUMS are written at the seal, each
row held to the hour it was read from. With the window already fetched,
its head waits for the decoder to seal every source hour. A native HRRR
run with a PMSL donor fetches the whole window
before it prepares and says so: the donor is bound into the source
manifest before any hour is decoded. Met_em, the `woof run`
experiment route, the downscale route and a
native HRRR or ERA5 preparation with a water-temperature overlay (its
receipt covers every forcing time and is part of the cache identity)
prepare sealed. Each of them but the
downscale route, whose forcing is its parent run's history, says so in
one `prepare:` line on stderr as it starts building its forcing.

A domain tree from a mapped source or the GFS series chains on the CPU or the card: its
nests are prepared into the head (`hierarchy-head/`), the root's later
intervals stream as above, and the tree runner starts the forecast on
the head. A native HRRR domain tree chains on its root preparation's
head: the hierarchy stage prepares the nests into its own head once the
root's head exists, passes each root interval on as the root writes it,
and seals after the root seals. A tree whose nests follow a storm (a `woof cyclone-setup`
configuration included) has its statics corridor built into the head, so
it starts there too, and a moving nest waits only at the root's intervals.
A tree whose root streams from a host store under `[tiles]` also starts
at the head: the store loads the start state there and takes each later
boundary interval at its seam. This holds for `mode = "on"` and for
`auto` when the planner streams the root. A
time step derived from the terrain reads the root's boundary winds over
the whole run, so a later interval can move it; the forecast so far is
then kept in `.quarantine/streamed-attempt/`, out of the run's history,
and the forecast runs again on the
sealed tree in the same process, so its output is the sealed tree's.

Chained preparation is on by default. A chained forecast writes the same
history files as one started after a sealed preparation. For diagnosis,
`WOOF_CHAINED_PREP=0` turns it off: the same writer then publishes the
whole tree at the seal and the forecast starts after preparation.

To see exactly what would be run, without running it:

```
woof sim PREPARED_ROOT --experiment-config e.toml --wps-namelist n.wps \
          --outdir out/run --print-command
```

That prints one line. It is the real runner invocation with every
digest filled in. Copy it into your own script and you never have to
call `woof sim` again -- that is the point of publishing it.

---

## Stage 1 -- `woof prep`: preprocessing, on your data

`woof prep` is the same program as the standalone `rw-wps` /
`woof-wrf-init` console script: one parser, one implementation, two
spellings. Every flag one accepts, the other accepts.

**It downloads nothing.** You supply the files.

### Input contract

| you supply | flag |
|---|---|
| the source adapter to decode with | `--source MODEL` (`woof prep --list-sources` prints every one, with its status and its evidence) |
| your GRIB/NetCDF files, in deterministic time order | `--input FILE` (repeatable) |
| your WPS namelist | `--wps-namelist namelist.wps` |
| your WRF namelist, when the route reads one | `--namelist-input namelist.input` |
| the resolved experiment TOML | `--experiment-config experiment.toml` |
| staged WPS_GEOG | `--geog-root DIR` |
| where the prepared tree goes | `--output-root DIR` |

For a source with no named adapter, the declarative mapped route takes
an explicit field/level/cadence contract instead of a built-in one:
`--descriptor`, `--mapping`, `--composition`, and
`--author-input-manifest` to write the hash manifest that binds your
files. `woof prep --show-source mapped` prints what that route
declares and, accurately, what it does not certify.

> **Known limit, stated plainly.** Preparing an arbitrary source with
> your own mapping works. **Running the result does not yet.** The
> forecast stage certifies only the *packaged* mapping -- the one the
> `20crv3` route uses -- by pinning its mapping, composition and
> provenance authorities to digests shipped with this distribution. A
> mapping you authored fails that pin, and there is no second
> certificate for a caller-supplied one yet. `woof sim` detects this
> case and says so at the door rather than deep in a hash comparison.
> It is not a flag you are missing. Today the route that prepares *and*
> runs end to end is `woof prep --source 20crv3`.

If you already have a WRF namelist pair and no woof config,
`woof import-namelist namelist.wps namelist.input --output experiment.toml`
translates them and prints a substitution report naming everything it
had to change.

### Output contract

A prepared tree under `--output-root`, containing the top-level
`proof.json` described above. Nothing else about the layout is part of
this contract -- read the document, not the directory listing.

### Worked example

```
woof prep --source gfs \
  --gfs-series data/gfs-series.tsv \
  --cycle 2026-07-29_18:00:00 \
  --bridge tools/grib1_bridge/target/release/gpuwm_gfs_grib2_bridge \
  --wps-namelist authority/namelist.wps \
  --experiment-config authority/experiment.toml \
  --source-manifest data/gfs-input-manifest.json \
  --source-manifest-sha256 <sha256 of that manifest> \
  --geog-root ~/WPS_GEOG \
  --output-root prepared/
```

`--dry-run` validates the arguments and prints the internal command
without running anything.

---

## Stage 2 -- `woof sim`: the forecast, alone

```
woof sim PREPARED_ROOT --experiment-config TOML --wps-namelist WPS --outdir DIR
```

**No fetching. No rendering. No network.** The stage does not import
the download machinery at all, and the whole command completes on a
machine whose sockets refuse to connect -- both are asserted by
`tests/test_stage_seams.py`, not merely promised here.

### Input contract

| you supply | flag |
|---|---|
| the prepared tree | positional `PREPARED_ROOT` |
| the experiment TOML the preparation was bound to | `--experiment-config` |
| the `namelist.wps` the preparation consumed | `--wps-namelist` (single domain; unused by the tree runner) |
| where output goes | `--outdir` |
| optionally, an assertion that the config IS a shipped suite | `--physics-profile ID` |

`--runner {auto,single,tree}` selects the runner arm. `auto` -- the
default -- reads it off the bundle's own schema and domain count. The
explicit values exist so a caller who believes they know better gets
refused precisely when they do not.

`--print-command` prints the exact runner line and exits, running
nothing and requiring no GPU. The line is quoted for the shell it is
printed in: PowerShell on Windows (it starts with the call operator `&`
when the interpreter path needs quotes) and a POSIX shell elsewhere.

### Continuing a prepared forecast

Both prepared runners write canonical checkpoints and restore them. Pass a
checkpoint the earlier run wrote and a fresh output folder. A single-domain
bundle also takes its exact `--wps-namelist`, as a fresh run of it does:

```sh
woof sim PREPARED_ROOT --experiment-config experiment.toml \
  --wps-namelist namelist.wps \
  --restart previous/run/gpuwmrst_d01_TIME.npz --outdir continued
```

For a hierarchy, pass any member of the earlier tree's checkpoint set and
leave out `--wps-namelist`:

```sh
woof sim PREPARED_ROOT --experiment-config experiment.toml \
  --restart previous/run/gpuwmrst_d01_TIME.npz --outdir continued
```

Each runner validates the preparation, the scientific configuration and the
checkpoint's identity before it restores anything. A single-domain bundle binds
its complete configuration and stop time, so its continuation finishes the same
forecast. The tree runner's existing permitted changes to forecast length,
output/restart cadence, history window and adaptive-controller targets still
apply to a hierarchy, and `--sealed-forcing-extension` writes or restores under its
append-only forcing-prefix contract; a single bundle refuses that flag.
Selecting `--runner tree` does not turn a single-domain bundle into a
hierarchy. `--print-command` includes every operand. The original run
directory remains protected from output mixing.

`woof go CONFIG.toml --prepared-root PREPARED_ROOT --restart CHECKPOINT`
reaches the same runners without fetching or preparing again; add
`--wps-namelist` for a single-domain bundle and `--dry-run` to review the
plan first. Ordinary `woof run --restart` and `woof resume` keep their own
direct-run checkpoint path.

### Output contract

Under `--outdir`:

- `wrfout/wrfout_d<NN>_<time>` -- history frames, WRF-shaped NetCDF.
- `report.json` -- the run's own validity verdict: health, stability
  and input-identity gates, plus `input.boundary_interval_seconds`,
  the real spacing of the lateral boundary times it integrated
  against. **This is the file to read to decide whether a run is
  usable.** `status` is the verdict.
- `progress.json` -- republished as the run advances.

### Watching it run

The stage runs the model **in this process**, so everything the model
prints reaches your terminal as it happens rather than through a pipe.
A script that wants typed events rather than prose should drive
`woof run-plan PLAN.json` instead, which emits one append-only JSONL
event stream in which every fact the human output prints is a typed
field on a typed event.

---

## Stage 3 -- `woof render`: pictures, from frames that already exist

```
woof render out/run/wrfout/wrfout_d01_* --out out/render
```

Takes `wrfout` files -- ours, or another model's, as long as the
variables are there -- and writes PNGs. It runs nothing else and
fetches no forcing data.

- `--products LIST` -- comma-separated product names, or `all`.
- `--list-products` -- the engine's catalog with per-file availability,
  i.e. *why* each product is or is not renderable from your file,
  instead of rendering.
- `--engine {auto,rust,matplotlib}` -- the production Rust renderer, or
  the matplotlib workaround by name. `auto` refuses when the renderer is
  unusable rather than degrading to matplotlib weather fields.
- `--source-label TEXT` -- set this when rendering frames this model did
  not produce, so the sheet does not claim them.
- `--streamlines` / `--barbs` -- how the wind is drawn on every product
  that carries a wind layer. Without either flag the engine keeps its
  automatic choice (streamlines on curvilinear and projected grids,
  barbs on plain lat/lon); `RUSTWX_WIND_STREAMLINES` still works and
  these two outrank it.

### Your own variables

`--products var:<stored name>` draws any 2-D field the wrfout carries,
including one you added to your own WRF Registry: a `(Time,
south_north, west_east)` variable named `MSLP_ANOM` is imported as
`wrf_mslp_anom` and rendered by

```
woof render wrfout_d01_* --products var:wrf_mslp_anom --out out/render
```

with no product definition to write. `--list-products` names every
`var:` row a file can serve. The panel gets a neutral full-range fill
and the variable's own units; no curated colortable is claimed for a
field this tree has never seen.

---

## `woof go`, as a composition

`woof go CONFIG` runs six stages in order: authority, fetch, front-door
manifest, preprocessing, forecast, render. `woof go CONFIG --dry-run`
prints all six commands, filled in, and runs none of them -- which is
the fastest way to see how the stages above compose for your config.

Stages 4, 5 and 6 of that chain are exactly `woof prep`, `woof sim`
and `woof render`. Not "equivalent to": the same commands.
`tests/test_stage_seams.py` asserts that the forecast command `go`
composes is byte-identical to the one `woof sim` composes for the same
prepared tree, on the single-domain arm and the domain-tree arm both.
If that ever stops being true, `go` has become a second implementation
and the test fails.

So the two routes are not a choice between a supported path and an
unsupported one. They are the same path, entered at different points.

### What the chain leaves behind: `<outdir>/events.jsonl`

A bare `woof go` writes an append-only event stream beside the run.
You do not ask for it and there is no flag to turn it on.

It is the same grammar `woof run-plan` writes -- schema
`gpuwm.run-plan.event.v1`, one JSON object per line, a dense `sequence`
-- so one reader replays either. `woof.runplan.read_events` is that
reader; `woof.chain_events.stage_walls` and
`woof.chain_events.summarize` are two convenience views over it.

```json
{"schema_version":"gpuwm.run-plan.event.v1","sequence":6,"emitted_unix_ms":1786988412431,
 "event":"stage_finished","stage":"fetch","wall_seconds":11.271,"ok":true,"exit_code":0,
 "bytes":41284919,"downloaded_files":3,"bytes_per_second":4712880.4,
 "verified_files":0,"verified_bytes":0}
```

What is in it:

- one `stage_started`/`stage_finished` pair per stage, named `boot`,
  `authority`, `fetch`, `manifest`, `prepare`, `forecast`, `render`.
  `boot` is the CLI's own start-up plus the memory and geography gates.
  On a chained preparation `prepare` and `forecast` overlap: a
  `prepare_head_ready` event (`head_sha256`, `ready_unix_ms`) marks the
  moment the forecast started, and
  `prepare` finishes at the seal. A chain hosted by `woof run-plan`
  (the staged route `woof go` takes for a mapped source) keeps one
  stage open at a time, so there `prepare` finishes at the head and
  `prepare_sealed` (`prepared_root`, `prepared`) is recorded when the
  preparation seals, while the forecast runs. That chain also records,
  for each seam where the forecast waited, `boundary_wait_started`
  (`interval`, `reason`) and `boundary_wait_finished` (`interval`,
  `seconds`).
- on `fetch`: `bytes`, and `bytes_per_second` **when this run actually
  downloaded something**. A re-run against an existing `--data-dir`
  verifies rather than transfers, and reports `verified_bytes` with a
  null bandwidth instead of dividing hash time into bytes.
- `first_products_ready` with `seconds_from_launch` -- time to first
  plot, measured by the process that published the pictures.
- `live_products_ready` for each later frame the forecast draws as it
  lands, while the `forecast` stage runs: `domain`, `valid_time`,
  `frame`, `pictures`, `render_seconds`, `complete`,
  `published_unix_ms` and `seconds_from_launch`, read from the
  runner's own `live-products.json` within about a second of the
  pictures being published.
- one terminal `completed` (or `failed`), carrying every stage's wall,
  the process wall, how much of it the stages account for, and the
  forecast's own internals read back out of its `progress.jsonl`:
  `preflight_verify`, `restore_prepared_cache`, `initialize_physics`,
  step 1's wall, and `first_step_excess_seconds` when step 1 paid
  something the steady state does not (see PROGRESS.md).

The point of it is a before/after number. Every pre-sim stage in this
tree is being rewritten in Rust, and a rewrite without a measured
baseline is an opinion. `tests/data/pre-sim-stage-baseline.json` is the
pinned reading -- with the box, the card, the date, the version and the
case stated, because a number without those is not comparable to
anything.

---

## Driving it from your own script

The shape that works:

1. Fetch your own data, however you like.
2. Author `namelist.wps` and `namelist.input`, or take them from your
   existing run. `woof import-namelist` turns that pair into an
   experiment TOML and tells you what it substituted.
3. `woof prep ...` on your files. Check the exit code.
4. `woof sim PREPARED_ROOT ... --outdir run/`. Check the exit code,
   then read `run/report.json` and check `status`.
5. Render when and how you like -- `woof render`, or your own plotting
   against the `wrfout` files.

Every stage exits nonzero on refusal and prints one sentence saying
why; add `--explain` to any command for the mechanism behind the
sentence.


### Reuse a prepared bundle or continue its checkpoint

Run an existing prepared bundle directly, keeping the original preparation intact:

```sh
woof go CONFIG.toml --prepared-root PREPARED --outdir NEW_OUTPUT --products none
```

To continue a prepared checkpoint, hierarchy or single-domain, add
`--restart CHECKPOINT`. Use a fresh output folder beside the earlier run. The command verifies the existing bundle and restores
through the same runner as `woof sim`; it does not fetch inputs or repeat preparation.
If your usual command also supplies `--data-dir` or `--geog-root`, those paths
remain in the plan and are reported unused; the existing bundle is the input.
All checkpoint siblings must remain together. The configuration and prepared inputs
must satisfy the runner's existing identity checks.

A single-domain portable bundle additionally needs `--wps-namelist ORIGINAL.wps`,
its exact prepared WPS authority, for a fresh run and a continuation alike; its runner
writes checkpoints at the configuration's restart interval. Configs with `[case_data]`
can use `woof go CONFIG.toml --restart CHECKPOINT` through the ordinary experiment
runtime without a prepared-root operand.

The same operation is available in a `route: "prepared"` run plan using
`run_options.prepared_root`, optional `run_options.restart`, and
`run_options.wps_namelist` for a portable single-domain bundle. `output_root` names
this new run's own directory. Inspect `go` with `--dry-run`, or inspect a plan
with `woof run-plan PLAN.json --resolve`, before running.
