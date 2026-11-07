# 6. The pipeline: fetch, prep, sim, render

## 6.1 Three stages, each invocable alone

The pipeline contract is three stages with a durable boundary object between them
[docs/public/PIPELINE-STAGES.md]:

| stage | command | takes | leaves |
|---|---|---|---|
| preprocessing | `woof prep` | source files, `namelist.wps`, experiment TOML | a prepared tree |
| simulation | `woof sim` | a prepared tree | `wrfout` frames + `report.json` |
| rendering | `woof render` | `wrfout` frames | product PNGs |

The boundary object is the prepared tree: a finished preparation leaves a
top-level `proof.json` (single domain) or `receipt.json` (domain tree) whose
schema field names the preparation that produced it, which is why `woof sim`
needs no `--source`, cycle, area, or config to identify what it is looking at. A
directory with neither a bindable document nor a completion receipt is refused
rather than run. `woof sim` does not import the download machinery at all, and
the whole command completes on a machine whose sockets refuse to connect;
asserted, not promised [tests/test_stage_seams.py]. The same test file asserts
that the forecast command `woof go` composes is byte-identical to the one
`woof sim` composes for the same prepared tree apart from one
`--progress-format` pair, asserted as the only difference, so the chain cannot
become a second implementation.

The preparation stage selects its compute backend as `--preprocess-backend
auto` by default: a box whose CuPy is missing or outside the certified family
falls to the deterministic parallel CPU backend and announces the fall in one
line, and an explicit `cuda` with no usable CuPy refuses by name [commit
54c492a32]. The announced line is in the 10 GiB walk's own transcript
[receipt:ux-walks-replay/gpu-walk-3080.html].

A CUDA preparation is also priced before it allocates anything
[woof/ingest/preparation_price.py]. Every door that can prepare on the card
(GFS, ERA5 and its analysis-ready store, the mapped sources and 20CRv3,
met_em, native HRRR with its spawned boundary workers, `woof run` case data
with or without a host store, and the downscale child's parent
interpolation) prices the model state, the forcing analysis, the vertical
plans and outputs, the boundary tables, any child states held for the
hierarchy export and the CUDA context, reads the card's free memory once in a
subprocess, and decides there. Under `auto` a preparation that does not fit
prepares on the CPU with one line, for example "the CUDA preparation needs
30.2 GiB (...) and the card has 22.3 GiB free of 24.0 GiB", and the receipt's
`selection.device_fit` records the terms, the free memory and the answer.
An explicit `--preprocess-backend cuda` that does not fit is refused by name
before anything is allocated, naming the terms, the stage it would have
stopped in and `--preprocess-backend cpu`. Every one of those doors defaults
to `auto`, and `woof go` prices the CPU road for a preparation the card
cannot hold rather than refusing the run. The price was fitted to four
measured CUDA preparations (1792 x 1024 x 55, 896 x 512 x 59, a 3:1 nest and
a tiled nest) and sits 5% to 11% above each measured card peak
[tests/test_preparation_price.py].

Under either backend, the masked surface fields (soil moisture and
temperature, snow, skin temperature, sea ice) take WPS metgrid's masked chain
in float64 in the Rust preprocessing library, parallel across target cells
(on the CPU backend's worker count, at most eight threads when none is given,
and on every CPU under the CUDA backend), with a result that does not depend
on the worker count. The mapped values and their repair counts are byte-identical to
the single-core NumPy chain it replaced, which is kept only as a test oracle
[tests/test_wps_masked_chain_native.py]. The native HRRR route maps its soil
temperature and soil moisture the same way, under either backend: the
land-only bilinear stencil, the search for the nearest HRRR land cell past the
fallback radius and the stencil's report are built in float64 in the same
library, and on the CPU the stencil is applied there too, byte-identical to
the NumPy builder kept as its test oracle
[tests/test_masked_stencil_native.py]. A library that predates either entry is
refused by name with the rebuild remedy; there is no NumPy route for these
fields. The water surfaces take the same library under either backend: the
search for the nearest source water cell of each model lake, the bilinear
blend of source water temperature over the donors each water body owns, the
sweep that closes a body's remaining holes from its own cells, and the corner
blend of a water-temperature overlay, the labelling of water bodies, the
source cells each body owns, the repair of water cells whose provider left
no admissible temperature, and the per-body assembly of every lake and sea,
byte-identical to the NumPy code kept as their test oracle
[tests/test_water_blend_native.py]. The CPU backend's bounded surface-nearest
search runs there too. These host steps take `--preprocess-workers` when it
is given; otherwise the CPU backend's automatic count (at most eight), and
under the CUDA backend every CPU the process may use. The water-temperature
overlay of the mapped sources and of a case configuration takes the automatic
count when `--preprocess-workers` is not given.

`woof go CONFIG` runs six stages in order (authority, fetch, front-door
manifest, preprocessing, forecast, render); `--dry-run` prints all six filled in.
The chain writes an append-only `events.jsonl` with one started/finished pair per
stage, including `first_products_ready` with seconds from launch: time to first
plot, measured by the process that published the pictures; a pinned baseline
reading with box, card, date, and case exists in the tree
[docs/public/PIPELINE-STAGES.md; tests/data/pre-sim-stage-baseline.json]. The
`go` chain currently orchestrates GFS only (section 5.8).

A 2.5.0 fix worth knowing because it was a composition defect: `woof sim`
refused the run folder it had just created (the stage door allocated the stamped
folder create-exclusively, then dispatched to a runner whose first act was an
exclusive mkdir on the same path). Both runners now accept an empty directory and
still refuse the moment it holds anything [CHANGELOG.md, engine 2.5.0].

## 6.2 Run-stamped outputs

Default-on for `woof go`, `woof sim`, and `woof render`; `--run-stamp off` is
documented as a compatibility workaround, not an alternative
[docs/run-output-folders.md]:

```
out/myarea/                                   <- the case, from `woof go --outdir`
  data/                                       <- the download, cached across runs, never stamped
  latest-run.txt                              <- one line: the newest run folder's name
  run-20260817-041233Z_i202607291800Z/
    authority/  prepared/  run/  png/  events.jsonl
```

`latest-run.txt` sits beside the run folders, in the directory named on the
command line, so a render-only workflow with `--out out/myarea/png` leaves it at
`out/myarea/png/latest-run.txt`, not at the case root
[docs/run-output-folders.md].

Stamp grammar: `run-<launch UTC to the second>_i<init UTC to the minute>[-<NN>]`;
the launch instant leads so a listing sorts into run order, and the `_i...` part
is absent when the door cannot read the init time, because a wrong init time in a
folder name is worse than a missing one. Init comes from the config's fetch cycle
(`go`), the bundle's proof/receipt (`sim`), or `SIMULATION_START_DATE` in the
first wrfout that carries it (`render`). Render layout sits underneath the stamp,
unchanged: `<out>/<run-...>/<domain>/<product>/<valid-day>/<file>.png` with domain
tokens like `d02-3km`; `--run-stamp off --layout flat` together reproduce the
v2.4.1 directory byte for byte [docs/run-output-folders.md;
tests/test_run_stamp.py; tests/test_render_layout.py]. Pointing a door at an
existing run folder: `go` and `sim` refuse a folder that already holds their
outputs; `render` resumes, because rendering is idempotent per filename and
adding products later is a real workflow. For `woof render` the run folder and
the `latest-run.txt` pointer are published only once the first picture lands: a
render that fails before drawing anything removes its empty folder and leaves
the pointer on the last run that produced output, so a script following the
pointer never lands in a failed render's empty tree [docs/run-output-folders.md;
commit c77488cd5]. Evidence:
[gallery:run-stamped-outputs-2026-08-17/].

## 6.3 Rendering and ensemble products

Weather-field products come from the Rust renderer `rw_wrfbatch` driven through
`woof.rustwx`; ensemble products from `rw_ensbatch`, which shares the same store
import, projection, basemap, styling, and PNG writer. `woof enprod` reads a
`member_NNN/` ensemble root and produces mean, spread, probability, paintball,
and probability-matched-mean panels for reflectivity, updraft helicity,
precipitation, T2, and 10 m wind speed, with neighborhood radii and thresholds
(field-native defaults: refl 40 dBZ, UH 75 m2/s2) [docs/public/CLI-OPTIONS.md].
`--engine auto` uses the Rust path when built and falls back to matplotlib with
the reason named; `--engine rust` refuses rather than substituting.

The failure modes are designed to fail closed: bitwise-identical members refuse
outright, so an all-zero spread is never published as if it were measured; a
mean or spread file claimed as a member refuses on the GRIB ensemble identity
octets; grid disagreement refuses; a member is a time series, not a file, and a
2.5.0 fix made `rw_ensbatch` ingest a member's whole forecast (the roster had
kept only the newest outfile per member, drawing the final valid time under the
name of the first) [tests/test_enprod.py, with every probability assertion an
exact equality against a hand-computed number; tests/test_enprod_hardening.py;
gallery:gefs-member-ensemble-20260817/REFUSALS-MEASURED.txt].

A third ensemble-shaped thing exists and must not be conflated with the two that
ship: `tools/ensemble_forecast.py`, a perturbation ensemble engine, is
experimental, is not a `woof` subcommand, and prints its scientific limits on
every run (no mass or wind balance, one shared unperturbed lateral boundary file
per member so spread decays toward the rim by construction, lateral taper only,
no surface/soil/parameter perturbation, determinism only per software-and-device
stack). Its own docstring: nothing this tool produces is on a certified forecast
path [tools/ensemble_forecast.py; tests/test_ensemble_engine.py]. See section
9.2.

## 6.4 Downscaling

`woof downscale` derives and runs a one-way child from an archived parent run
(WOOF or stock WRF via `--parent-namelist`). The documented walkthrough was
walked end to end on a wheel install (Windows 11, RTX 3080, 2026-08-17)
[docs/public/DOWNSCALE.md]. Two facts shape the path, stated up front because
they are the two dead ends: the parent must have written a woof restart
(`--parent-restart` binds it as the physics evidence; every `woof domain`
configuration, single domain or nest ladder, checkpoints hourly or at the end
of a shorter run; a hand-written one needs a positive `restart_interval_s`),
and a full-physics
child needs a child-grid surface file, which the preparation already built
(`--child-surface-from <prepared>/wrf-native-input/wrfinput_d0N`).

The contract, in order: prove the parent (frame inventory, frozen geometry,
regular cadence, one producer); prove the physics (microphysics identity from
companion evidence, a WOOF restart or the WRF namelist, never inferred from the
variable inventory); boundary cadence defaults to the archive's own with one
warning line; full-physics children refuse at the front door without a surface
source; every run writes `report.json` with SHA-256 receipts of parent frames,
physics evidence, surface source, boundary-clock identity, and outputs
[docs/public/DOWNSCALE.md; tests/test_downscale_cli.py].

Measured on the walkthrough: a 120x96x49 child at 3 km, 2 h, full physics, 480
child steps in 27 s wall on an RTX 3080 [docs/public/DOWNSCALE.md]. The measured
cost of driving a child from an hourly archive instead of a live nest
(2026-07-29, RTX 5090; hourly-archived 1 km parent, 501x501 Thompson, downscaled
to a 400x400 500 m child, ratio 2, full physics, 3 h across convective
initiation, scored against the live nest of the same run on the interior grid):

| lead | T2 MAE (K) | T2 corr | PSFC MAE (Pa) | wind10 corr | refl corr | refl MAE (dBZ) | CSI@20 |
|---|---|---|---|---|---|---|---|
| F+0.0 | 0.000 | 1.000 | 0.1 | 1.000 | | | |
| F+0.5 | 0.133 | 0.992 | 37.5 | 0.978 | 0.922 | 0.31 | |
| F+1.0 | 0.127 | 0.988 | 23.0 | 0.957 | 0.906 | 0.80 | |
| F+1.5 | 0.156 | 0.989 | 26.6 | 0.943 | 0.845 | 1.20 | |
| F+2.0 | 0.253 | 0.977 | 23.9 | 0.916 | 0.262 | 3.05 | |
| F+2.5 | 0.289 | 0.975 | 29.8 | 0.871 | 0.248 | 13.06 | 0.000 |
| F+3.0 | 0.430 | 0.964 | 28.4 | 0.802 | 0.148 | 25.38 | 0.020 |

The scope statement the source attaches to the F+0.0 row travels with it: that row
is four comparator metrics on one pair of runs, and no full-state digest was taken
of the offline/live pair. Later rows combine the forcing-path difference
(hourly interval-linear boundaries versus every-parent-step forcing),
start-state differences those four metrics miss, and growth of small
perturbations. No perturbed live-nest control was run, so the table cannot
separate them or show that denser boundary cadence recovers convective-scale
agreement. These are model-to-model differences, not errors against observations
[docs/public/DOWNSCALE.md:143-149].

Read it as two regimes, which is the doc's own framing: the mesoscale envelope
stays close (T2 correlation never below 0.96, PSFC MAE under 40 Pa, wind
correlation 0.80 at F+3), and the convective scale decorrelates at initiation
(reflectivity correlation collapses exactly where the reference run's interior
reflectivity max crosses 0 dBZ; CSI near zero; both runs make storms, in
different places). Write parent history at 15-minute or denser cadence when you
plan to downscale; the CLI warns whenever the archive is coarser
[docs/public/DOWNSCALE.md]. The child can take its OWN eta ladder:
`--child-levels N,STRETCH` remaps the initial state and every lateral-boundary
frame onto it once, on the host, in FP64 (conservative in dry mass and water
substance; a matched ladder is bitwise unchanged). Known limits: terrain is
still SINT-inherited, one child per invocation, and a point-mode child of a WRF
parent needs `--child-config` (which carries its ladder as `eta_levels`, not
`--child-levels`).

## 6.5 Tile streaming (`[tiles]`)

Tile streaming runs one domain out of core on one card. Configuration is a TOML
table, not a flag: `mode = "off" | "auto" | "on"`; `auto` sizes the resident
domain against the card and streams only when it does not fit; optional tiling
keys are for benchmarks; `mode = "off"` with a tile size set is a refusal, not a
hint. The halo is computed (`10 + 3*time_step_sound//2`) and no forecast may set
it, because a smaller one is silently wrong and faster, which is how that defect
class hides [docs/public/TILES.md].

**Bit-exactness is the defining property**: a streamed domain and a resident one
produce identical bytes, held by a 51-case, 14-rung physics gate with negative
controls, and by a real forecast configuration with 229 carriers
[tilestream/test_gate.py; tilestream/test_join.py]. `[tiles]` deliberately
contributes nothing to restart identity: a checkpoint written resident resumes
streamed and vice versa (section 4.5).

Measured tiling tax against the identical resident run (dry, RTX 4090, 1024^2 x
49, 150 steps): tile 128 1.359x, tile 256 1.217x, tile 512 1.346x. Per-cell cost:
a single store is 32.3 B/cell dry and 279.5 B/cell at the full carrier set;
against a measured 44.14 GiB pinned-host ceiling that is 5476^2 x 49 dry.
Resident cost is a fixed 2.0-3.2 GiB per process plus 541-612 B/cell at full
physics, so a 32 GB card holds about 924^2 x 49 resident and a 12 GB card about
483^2 [docs/public/TILES.md]. The limits page carries its own governor, which
this manual keeps: none of these numbers is a capacity multiplier against a
vanilla resident run, and there is no measured full-physics streaming multiplier
to quote; the dry per-cell cost is 8.5x more generous than the full-physics
cost, and a prediction that dry scaling would merely be pessimistic was measured
and refuted (predicted 91% and 7.3x, measured 52.6% and 4.21x)
[docs/public/TILES.md; tilestream/NO-DRY-NUMBERS.md].

One end of a coupling edge streams, never both: streamed-parent/resident-child
and resident-parent/streamed-child are both gated bit-identical to the
all-resident tree; both ends streamed is refused when the config is read, naming
the edges and the three ways out. `mode = "auto"` on a tree is a joint decision:
before a streamed domain chooses its tile, the walk reserves what every
undecided domain below it needs, and the run receipt's `tiles` block records
every grid's decision and reservation [tilestream/test_nest_executor.py;
tilestream/test_streamed_child.py; docs/public/TILES.md].

A safety-gate defect found and fixed here is worth citing as the silent-green
class: under a host store the resident state was never written again, so the
per-substep health gate read the condition the store was filled from. Measured at
672^2 x 49 with poison injected at step 50: 200 of 200 substeps reported
`nan_free=True` while the store ended with 22,579,196 of 22,579,200 `w` cells
non-finite. The health fold is now taken per tile inside the sweep,
bit-identical to the whole-domain report, measured equal on 250 substeps x 8
fields; the full-state validator cannot be folded this cheaply and is now
explicitly unarmed (skipped, counted, warned) rather than silently passing
[docs/public/TILES.md].

Separate from `[tiles]`: `woof stream PLAN.toml` follows an uploading HRRR
cycle with sealed hourly forecast legs (Rust full-file fetch, immutable forcing
prefix, resume from the preceding tree checkpoint, health PASS required, every
receipt and checkpoint member hashed into a durable chain); it never changes
forcing inside a live model process [docs/public/STREAMING.md].

## 6.6 The Rust data path

The 2.5.0 line moves data-path processing to Rust, with Python remaining
orchestration, CLI, and CUDA driver code.

**GRIB decode.** The vendored Rust mapped engine is the default
(`--mapped-engine rust`); parity against the Python engine was proven on 17
real-byte sources: 15 PASS plus 2 gaps then declared routed-to-Python (ERA5
GRIB1, ERA5 NetCDF), zero FAIL, with the refusal battery matching class and
sentence 4/4 [receipt:DECODE-VENDOR-2026-08-17.md; driver
tools/mapped_engine_parity_sweep.py]. Both gaps have since closed and the
engine's gap declaration is empty (`ENGINE_GAPS = ()`): every registered prep
door decodes on the engine by default, and the 20CRv3 exact-member route
composes in the engine, dual-run on the real member bytes with 643 of 647
output files byte-identical (the four diffs are the decoder record and digest
cascade plus timings, by design), member identity sealed on both arms, and a
120-step forecast leg PASS [woof/mapped_engine_bridge.py:230, commit
f46af5f80]. At the wave tip b858ad824 the parity battery stands 157 passed /
11 skipped / 0 failed on Windows and 122/46/0 on Linux with live dual-engine
comparison, and the compose sweep is 32 of 32 with engine_default=rust and
zero ROUTED-PYTHON rows. Two refusal-parity defects were found and
fixed to a defined behavior on both routes (a diagnostic means the bytes are the
problem, `ValueError`; silence means the installation is, `RuntimeError`), and
one message divergence is kept by design: the Python route quotes its subprocess
tool and the engine speaks in-process, because faking a subprocess sentence
would misreport which binary read the bytes
[docs/dev/decode-vendor-design.md]. The engine is the default because it matched the in-house Python decoder
on the real bytes in the stated parity batteries and carries the registered
format surface under the Python boundary. Agreement between two in-house
implementations is a verification check, not an independent proof of correctness. Decode wall at the tip, same frozen entry,
real bytes, warm cache: after the parallel-assembly work the two re-measured
frame-heavy cases land at 18.6 s for the hrrr pressure pair (largest frames,
1002 Mcells) and 7.8 s for the gdas 0.25 pair (444 Mcells), measured on the same
box through the real engine. The remaining cases (icon-eu, rap, rrfs) were not
re-benched after that work; their pre-concurrency walls, the Python-engine
column, and the vintage bookkeeping that keeps the two benches apart live in the
wall-clock receipt [docs/dev/decode-vendor-design.md, section 10.1]. Every
decode figure there compares the Rust default against the Python engine at the
2.5 tip; no decode measurement against the 2.4 line exists.

The pre-fix slowdown on frame-heavy sources was the seam, not the decode: the
engine wrote the full f64 stream to disk, hashed every byte twice, and Python
re-read and re-hashed it (about 7.5 GB for the HRRR pair); closed by removing
the redundant whole-stream hash, verifying per-field digests on a pool, and
running the data path concurrently. The default stayed `rust` throughout because
correctness parity was proven and the governing rule is the Python boundary, not
a stopwatch. Nothing on the registered data path stays Python: cross-source
compose, GRIB1, and NetCDF all decode on the engine, a bare member prep
resolves no subprocess decoder, and the empty gap declaration is mirrored in
code, compared by test, and printed by `woof doctor` [compose sweep 32/32
with zero ROUTED-PYTHON at b858ad824]. The Python engine remains selectable as
the named workaround (`--mapped-engine python`).

**Static fields / geogrid.** The full static build ran twice, numpy fallback vs
the bare-default Rust crate, in separate processes on five real domains over the
reference 30-arc-second WPS_GEOG tree: every field on every domain byte-identical
(14/14 fields on each of a 251x201 Lambert parent, a 601x601 sub-km nest, Mercator
and polar-stereo 111x89 domains, and a 1503x1503 corridor at 333 m), coverage
receipts equal as canonical JSON, sealed NPZ digests equal, zero divergences to
document [receipt:STATIC-RUST-PORT-2026-08-17.md]. Static build wall on the
Rust default, cold (fresh process): 5.3 s for the 333 m corridor, 1.9 s for the
sub-km nest, 5.3 s for the 12 km parent; warm (second in-process call) 5.5 s and
1.8 s. Quote cold with cold and warm with warm; the two must not be mixed in one
sentence. The numpy-fallback arms and the per-domain ratios live in the receipt.
The limit beside the numbers: the coarse parent gains nothing from the port,
because Rust wall time tracks summed source pixels read (the 12 km parent reads
78 M source pixels for 50 k cells, so that build is tile decode plus
per-source-pixel binning, which both languages do at similar speed)
[receipt:STATIC-RUST-PORT-2026-08-17.md;
gallery:static-rust-port-20260817/where-the-rust-time-goes.png]. Workaround
named: `WOOF_STATIC_PYTHON=1` falls back to numpy and says so.

**NetCDF wrfout writing.** The Rust writer was dual-written on three real cases
with every render pair byte-identical (classic CDF-2 container);
`WOOF_WRFOUT_WRITER=python` is the named workaround [CHANGELOG.md;
gallery:2026-08-16-rust-netcdf-writer/].

## 6.7 `woof doctor`

Doctor verifies the runtime estate for real: a subprocess import of every
declared package, probe executions of every bridge, front door, and render
engine, a ctypes load of the CPU library, table hash validation, sealed-manifest
re-hashing, decoder and transport checks per data route, and WPS_GEOG index
files, printing one line per item with the command that closes each gap
[docs/public/CLI-OPTIONS.md; woof doctor help]. The rule that makes it
trustworthy: where a check cannot exercise something it says `untested` and its
detail opens with "not tested", never `ok`, never silence; the test pinning this
names the two 2026-08-14 findings that produced the rule (tools printed `ok` on
a box where the route using them died, and three render bridges were in no
bundle at all while doctor printed a green estate)
[tests/test_doctor_route_accuracy.py].

`doctor --source` accepts every registered source id, the 31 rows of section
5.1: the choice list is the source registry itself, and a registry route
answers in one line naming its packaged profile, the engine that decodes it,
and the transport that fetches it, with the engine's own verdict kept on the
engine's own line so an unbuilt binary reads as one gap with one remedy
[woof/doctor.py; tests/test_ux_cli_polish.py]. After an upgrade, doctor
opens with what changed since the version that last ran on this box (state
in `~/.woof/doctor-state.json`, `WOOF_DOCTOR_STATE` to move it); the lines
are composed from the same describe functions the features export rather
than transcribed beside them, so a surface that moves again cannot leave a
true-sounding sentence behind [commit 6e23bb3e4].

## 6.8 Launch-to-first-step cost

The prep scaling study (test host 1, gpuwm 2.5.0 at `ed4c0ef69`, GFS, 6 h
forcing, single domain, wizard defaults, warm caches, cold download) gives the
launch-to-model-step-1 law by size: 0.14 M cells 15 s; 4.39 M 37 s; 12.6 M 79 s;
largest card-resident 16.8 M 100 s; massive tile-streamed 41.6 M
(1030x824x49, 4 forcing files) 373 s; a second tile-streamed arm at 41.2 M
(1026x820x49) with 24 h forcing and 10 files 339 s, a different grid and a
different forcing set rather than a re-run of the same case. 24 h forcing at mid size costs 98 s (+68%
over 58 s); a 3-domain tree (12/4/1.3 km) 61 s for 13.3 M cells; first run ever
on a box adds a flat +31 s. One known instability is stated in the report
itself: the 6 h tile-streamed arm being slower than the 24 h one is a store-fill
difference (95.1 s vs 1.2 s for the same 8.8 GiB), a speed-up target, not
measurement error [receipt:ARWEN-PRESIM-SCALING-2026-08-16.md].
