# Hardware and VRAM sizing

WOOF-ArWen runs on one NVIDIA GPU with CUDA 12.x or 13.x, field-verified
through 13.2 driver stacks on sm_89 by two independent nodes. This
page explains how
the sizing model works, where its safety factor comes from, and what
we measured on real hardware -- including the run where the estimator
was wrong and what changed because of it.  Host RAM is sized
separately and by hand:
[Host memory (RAM), which nothing above prices](#host-memory-ram-which-nothing-above-prices).

## The short version

Tell the wizard your card; it sizes the grids:

```bash
# --card, --cycle and --out are the required trio; --vram-gib N
# replaces --card for a capacity between the named tiers.
woof domain --point 35.3,-97.5 --card 24gb \
  --cycle 1999-05-03T12 --hours 6 --out configs/myarea.toml
```

Choose the vertical resolution and streaming mode before sizing:

```bash
woof domain --point 35.3,-97.5 --card 12gb --source gfs \
  --cycle 2026-07-29T18 --hours 6 --root-dx 3 --nz 76 --tiles \
  --out configs/myarea.toml
```

`--nz N` sets the number of mass levels; the file contains `N + 1` eta
interfaces resampled from the default stretched ladder. Omitting it keeps
the original 49 levels exactly. The requested levels are included in both
forecast and preparation memory estimates.

The wizard keeps the requested spacing and output intervals. If its default
time step cannot land those events exactly, it chooses the nearest compatible
exact rational step at or below the spacing-based recommendation and names the
adjustment in the console and generated config. Named physics suites keep
their original radiation, cumulus and PBL periods. Explicit timesteps in authored
configs retain their exact values and existing validation.

Bare `--tiles` means `--tiles auto`: the forecast planner chooses resident
or streamed execution using the declared card and this host's RAM.
`--tiles on` forces streaming; `--tiles off` selects resident execution.
The emitted `[tiles]` mode applies to the whole tree. A tree that would
require unsupported adjacent streamed domains is refused with the
planner's reason. Streaming still has to fit its host store, and preparation
still has to fit its own memory budget. Source coverage and polygon bounds
continue to apply. Run the wizard on the forecast host when sizing its RAM.

For existing ERA5 files, add `--forcing path/to/era5-combined.grib` (or
multiple files) with `--source era5`. The native header inventory measures
the cadence before fitting any grid. It requires the requested start,
coverage through the forecast end, and a continuous uniform sequence;
a missing time is diagnosed before the config is written. The same cadence
is written to the case, WPS namelist, and fetch hints. With no supplied files,
the source defaults are unchanged.

A wider supplied window also costs memory: preparation and a resident
forecast retain every boundary interval after the start, including intervals beyond
the requested forecast end. The wizard, `woof check`, and launch admission
use that actual retained count. Shortening `--hours` alone does not remove
those intervals; supply a narrower input window to reduce them.

| card tier | flat reserve | working budget | what fits (measured examples) |
|---|---|---|---|
| 12 GiB | 4 GiB | 8 GiB | Four domains: 156x126 / 312x256 / 336x276 / 268x220, or 398x318 single-domain at 12 km. **Windows: experimental** (see below) |
| 16 GiB | 4 GiB | 12 GiB | full 12-3-1-0.5 km four-domain ladder at ~3.3 GiB alloc estimate |
| 24 GiB | 4 GiB | 20 GiB | four domains: 170x136 (12 km), 336x272 (3 km), 360x294 (1 km), 288x236 (500 m) |
| 32 GiB | 6 GiB | 26 GiB | the reference-class case: 4 domains to 500 m at 400x400+ |

## Drivers, CUDA majors and GPU architectures

The GPU extra comes in two CUDA majors. Install one per environment.

| | `all-cu12` (also `gpu-cu12`, `gpu`, `all` and the desktop runtime) | `all-cu13` |
|---|---|---|
| Minimum NVIDIA driver | R525: 525.60.13 on Linux, 527.41 on Windows (CUDA 12.0) | R580 (CUDA 13.0) |
| NVRTC that compiles the forecast kernels on your machine | 12.9 (12.9.86 on 2026-10-15) | 13.4 (13.4.92 on 2026-10-15) |
| Architectures that NVRTC compiles for | sm_50 to sm_121 (Maxwell to Blackwell) | sm_75 to sm_121 (Turing to Blackwell); not V100 (sm_70) or P100 (sm_60) |
| GRIB2 post kernels (`--grib2`, `export-grib2`) | built-in machine code on sm_10x and sm_12x; a PTX ISA 8.0 image, compiled by the driver once, on every other card from sm_60 on | the same |

The driver minimums are NVIDIA's for each CUDA major, not runs of ours:
the oldest driver this release ran on is 595.91 (CUDA 13.2). On that
driver, with an RTX PRO 6000 Blackwell (sm_120), both extras ran
`woof go forecast.toml --grib2` on a 12 km GFS case. The architecture
rows are each NVRTC's own list of supported targets. The GRIB2 post PTX
is compiled by NVRTC 12.0, so it declares PTX ISA 8.0, the ISA of CUDA
12.0, which NVIDIA's drivers accept from R525 on; a test fails the build if
it declares a newer one. CUDA 12.0's `ptxas` compiles it for sm_60, sm_61,
sm_70, sm_75, sm_80, sm_86, sm_89 and sm_90. Run from the PTX on sm_89
(RTX 4090, driver 610.57), its GPU output equals the CPU's bit for bit, on
the post-processing tests and on a real 12 km frame. On sm_120 (RTX 5070 Ti,
driver 595.99, CUDA 13.2), forced to the PTX, it also equals the CPU bit for
bit on the same tests and frame. Driver 595.91 (CUDA 13.2, RTX PRO 6000,
sm_120) loads it. A card also needs a driver that supports its
architecture (Blackwell, for example, needs R570 or newer).
Before 2.9.1 that image declared ISA 9.4, which drivers older than CUDA 13.4
refuse (595.91 refused it). On every card without built-in machine code the
GRIB2 products were then computed on the CPU, with the same bytes, and
nothing was printed. When a card cannot load the post kernels now, the run
prints one warning line naming the driver's error, and `woof doctor` shows
the same result on its "GPU post kernels" line.

**The two majors give different forecasts.** Each extra compiles the
forecast kernels with its own NVRTC, and different compilers produce
different machine code. On the same card and driver, the same 12 km GFS case
after 1 h differed in 1,700,303 of 3,438,677 float words between `all-cu12`
and `all-cu13`: 2 m temperature by up to 0.34 K, accumulated rain by up to
0.55 mm, and reflectivity by up to 68 dBZ at single points. Neither major is
the reference. Compare runs, and quote identity results, within one major
and one NVRTC build; [DETERMINISM.md](DETERMINISM.md) lists the NVRTC build
among the pins.
The release identity gates run on the `cu13` extra: for 2.9.0, NVRTC
13.4.92 on sm_120.

## Independent runs on multiple GPUs

A prepared single-domain forecast can use several cards through
[`[devices]`](MULTIGPU.md). On a host with several cards,
`woof multi-run` starts any number of independent production runners at
once, with one unique physical GPU per process. The convenient form invokes
`woof run`:

```toml
schema = "gpuwm.multi-run-plan/v1"
summary = "runs/production-summary.json"
preflight = "estimate"  # estimate (default), alloc, or off

[[run]]
name = "coastal"
device = 0              # physical nvidia-smi index or full GPU UUID
config = "configs/coastal.toml"
outdir = "runs/coastal"
scratch = "scratch/coastal"
cache = "cache/coastal"

[[run]]
name = "inland"
device = 1
config = "configs/inland.toml"
outdir = "runs/inland"
scratch = "scratch/inland"
cache = "cache/inland"
```

```bash
woof multi-run plan.toml
```

Independent entries may carry different domains, forcing, and physics. They
need not be variants of one configuration.

Prepared single-domain and domain-tree forecast routes use the shell-free
module form (`woof.prepared_single_domain_forecast` and
`woof.prepared_domain_tree_forecast`). `args` is an argv array, never a
shell command; `{outdir}`, `{scratch}`, `{cache}`, and
`{input0}`, `{input1}`, ... expand to the plan's validated absolute paths:

```toml
[[run]]
name = "prepared-tree"
device = "GPU-..."
module = "woof.prepared_domain_tree_forecast"
inputs = ["prepared/tree", "configs/tree.toml"]
args = [
  "--prepared-root", "{input0}",
  "--preparation-receipt-sha256", "<sha256>",
  "--experiment-config", "{input1}",
  "--experiment-config-sha256", "<sha256>",
  "--io-mode", "history",
  "--outdir", "{outdir}",
]
outdir = "runs/prepared-tree"
scratch = "scratch/prepared-tree"
cache = "cache/prepared-tree"
```

Those two prepared forecast modules are the complete module allowlist for
this schema version. The argv must contain exactly one separate
`"--outdir", "{outdir}"` pair. Abbreviated, `--outdir=...`, duplicate,
fixed, or embedded output forms are refused before launch, so a target cannot
ignore the validated root and write elsewhere. Each runner's path-valued
options are also schema-known: required options appear exactly once, optional
ones at most once, and every value must be one exact declared `{inputN}`.
Literal, abbreviated, duplicate, or undeclared input paths are refused,
including a second run trying to name the first run's output as an input.
The same path cannot be declared twice within one run.

All plan paths are relative to the plan file. Configs, prepared trees, and
other declared `inputs` are read-only and may be shared, including the same
config on two GPUs for a cross-device numerical comparison. The device UUID
is intentionally part of each production capsule, so cross-device capsules
differ in that identity field even when frame bytes and numerical results
match. Every mutable outdir,
scratch, and cache root must be distinct and non-overlapping across the whole
plan, and must be new. Config-run output directories are claimed before
launch; prepared runners retain their own atomic absent-outdir claim. This
preserves earlier outputs and prevents two GPUs from writing the same fixed
filenames. Each run captures top-level output in
`scratch/gpuwm-run.log`; supervised worker logs and progress remain in its
outdir as usual. `CUPY_CACHE_DIR` and NVIDIA's `CUDA_CACHE_PATH` point to
separate `cupy/` and `cuda/` children of each run's new cache root.

Before launching forecasts, `preflight = "estimate"` runs `woof check` once
per selected GPU. `"alloc"` adds the allocation measurement, and `"off"`
skips the checks with a warning. A nonzero check is reported and retained in
the summary but does not block `woof run`: sizing remains advice, consistent
with the single-run command. Missing inputs or an unsafe configuration still
fail in the run's normal validation path and contribute a nonzero aggregate
status. Module-form runners use their production entry point's own input and
memory preflight instead of pretending a config-driven check applies.
Each config-form check runs in a fresh UUID-masked child while holding the
same machine-wide physical-GPU lock used by its forecast, before importing
the check command and any transitively loaded CUDA runtime.

The parent resolves each selector with `nvidia-smi`, rejects aliases to the
same physical UUID, and sets `CUDA_VISIBLE_DEVICES` before starting the child.
It also passes the physical UUID to the existing supervisor lock; module-form
runners acquire that same lock before importing their target module.
The parent resolves one exact machine-wide lock root before replacing each
child's temporary-directory variables and proves that root is outside the
plan, summary, every input, and every output/scratch/cache tree. Set
`WOOF_GPU_LOCK_ROOT` to an explicit stable host directory when the platform's
ordinary temporary root would overlap a run path.
CUDA-facing code that names `Device(0)` or `getDeviceProperties(0)` is
therefore using the only *logical* device visible inside that process, not
hard-coding physical card zero. This process boundary is intentional: there
is no in-process `--device` switch that could run after a CUDA context already
exists.

Before launching children, the orchestrator creates the summary parent and
probes the atomic create-only publication used for the final JSON (a hard link,
or on exFAT and FAT32 a rename that refuses a file). A summary that appears during a later
race is preserved and the orchestrator refuses to overwrite it. The receipt
contains the raw plan SHA-256, SHA-256 for every declared file input and
config, and an explicit delegated-to-runner-receipt marker for directory
authorities rather than implying their contents were hashed. It also contains
UTC start/completion times,
durations, exact child exit codes, resolved device identity, all isolation
paths, logs, and PIDs. It records the sum of child durations, the concurrent
execution window, and their overlap ratio only when every forecast succeeds.
That ratio measures process overlap; it is not a performance speedup, which
would require a separately measured serial baseline. Failed or interrupted
runs retain observed timing but do not report the overlap ratio. The
command exits zero only when every forecast exits zero; otherwise it exits one
after all children finish. Ctrl-C writes an
`interrupted` summary and returns 130. It does not kill children or unrelated
processes; unobserved child PIDs are printed and recorded so a supervised
worker is never orphaned by terminating only its parent supervisor. Once the
create-only publication probe succeeds, interruptions during directory claims,
authority monitoring, checks, forecasts, or final publication also attempt an
accurate create-only receipt naming the stage and every claimed directory and
known child PID. Unexpected failures stop the non-daemon monitor, attempt a
`failed` receipt, and then propagate a group error with the original exception
preserved as its cause.

A sticky periodic monitor starts before the first child and remains active
continuously across config checks and forecasts. It watches the raw plan and
every declared file authority, records the capture SHA-256 plus timestamped
signature observations and errors, and performs a final forced signature
observation before publication. Original config bytes are not reopened after
capture.
A pre-launch mismatch prevents any child from starting; a mutation observed
during checks prevents forecast launch; and a mutation during forecasts makes
the aggregate receipt fail. Restoring the old bytes does not clear a prior
observation. Prepared directories remain delegated because those production
runners bind their directory authorities in their own receipts.

For config-form runs, multi-run reads each experiment config once while loading
the plan, hashes those exact bytes, and durably creates a per-run payload.
CLI route detection, both `woof check` layers, supervisor input discovery,
and every fresh worker consume a payload validated against that SHA-256; the
original path remains only the source identity and relative-path base. The
supervisor also makes its usual create-only worker payload in the run outdir.
Changing the original path after plan capture cannot change any parsed phase.

After input discovery, each multi-run supervisor snapshots forcing, Vtable,
WPS namelist, and declared source-orography files into one plan-wide
content-addressed store beside the summary. Publication is create-only and
locked per SHA-256, so identical inputs shared by any number of runs occupy one
full copy. Workers validate every snapshot against the capsule hash and remap
all file opens to those read-only bytes; provenance and certification retain
the original declared paths. Before remapping or runtime, the worker's parsed
role, absolute original path, and slot-detail multiset must exactly equal the
parent SHA inventory, including duplicate identities. A forcing glob that
temporarily loses, gains, or renames a match therefore publishes a failed
heartbeat and failure capsule even if the original directory is restored
before validation. The store path is collision/isolation checked and its SHA
entries are listed in the summary. Ordinary single `woof run` invocations
keep their historical direct-input behavior and do not create this store.

### Small Windows cards are MEASURED now (2026-08-19, RTX 3080 10 GiB)

Windows/WDDM accounting used to come from exactly ONE machine -- a
32 GiB RTX 5090 running campaign-scale multi-domain forecasts -- whose
two fixed pool constants (4.12 GiB) plus a 1.75x envelope refused every
ladder on a small card for accounting measured somewhere else. An
EXPERIMENTAL "small-card tier" papered over that with a guessed
1.5 GiB reserve, and its advisory asked pioneers to send back one
measured peak.

That measurement exists now: six whole bare-default `woof go`
forecasts on an RTX 3080 10 GiB Windows 11 desktop (WDDM, desktop
resident on the same card), 60x48 through 240x192 at 12 km, rte-rrtmgp
and legacy-RRTMG suites, machine-wide `nvidia-smi` at 0.25 s beside
the runtime's own peak-watcher receipts. The measured peaks track
`itemized estimate + itemized non-pool residency` within
-0.20..+0.95 GiB, so every Windows card now takes ONE measured model
(the affine form below plus a WDDM pool-slack term), the experimental
tier is retired, and the 1.75 multiplier is gone from every gate.

`woof check` does not stop a later `woof run` -- nothing prevents
you from starting a forecast it warned about -- but since v1.1.0 an
observed peak above the WDDM budget is an exit code 4, not a green
exit with a warning in it. A script that reads the exit status is
blocked; a person reading the output is advised.

A first single-domain forecast is far below any of these: the
acceptance run (250x200x49 at 12 km, full physics) used ~6.3 GiB of
device memory and completed 6 simulated hours in 3.6 min on an
RTX 5090 ([FIRST-LIGHT.md](FIRST-LIGHT.md)).

## How the estimate is built

`woof check` (and the wizard, which calls the same estimator
in-process) prices a run in three layers:

1. **Itemized alloc estimate** -- every persistent field, scratch
   arena, and kernel workspace, summed per domain. This is the number
   the hard pass/fail gate compares against your measured free VRAM
   minus `--reserve-gib`.
2. **Footprint projection** -- the alloc estimate plus transient
   call-peak envelopes (radiation chunk workspaces are the largest).
3. **Projected machine peak = the alloc estimate PLUS the non-pool
   residency PLUS measured terms.** It is a sum, not a multiple:

   ```
   peak envelope = alloc estimate
                 + CUDA context + local-memory backing store
                 + 0.50 GiB unmodelled
                 + 5% of the estimate per nest beyond the root
                 + 20% of the estimate WDDM pool slack (Windows only)
   ```

   The middle term scales with the DEVICE (its SM count) and the kernel
   set your physics selects -- not with the grid. That is why the model
   has to have an intercept, and why the version that did not
   [got the sign of its own error wrong](#where-the-envelope-comes-from-the-accurate-part).

   The wizard bisects grid sizes until this fits the budget with
   headroom to spare; a wizard-emitted config passes `woof check` on a
   real card of the tier it was sized for.

Example (24 GiB tier, printed by the wizard on Windows, where the measured WDDM pool-slack term rides on the same affine sum):

```
  domain    dx        mass grid      dt         resident
  d01     12.000 km   220 x 176        60 s     1.09 GiB
  d02      3.000 km   440 x 352        15 s     3.91 GiB
  d03      1.000 km   474 x 378         5 s     4.54 GiB
  d04      0.500 km   380 x 304       5/2 s     2.95 GiB
  peak envelope: estimate 11.31 + non-pool 2.42 (CUDA context + local-memory backing store) + 0.50 unmodelled + 5% of the estimate x 3 nest(s) + 20% of the estimate WDDM pool slack = 18.19 GiB
    envelope basis: windows; measured, RTX 3080 10 GiB / Windows 11 WDDM, six whole bare-default forecasts machine-wide at 0.25 s over a 2.5x span of itemized estimate, rte-rrtmgp + legacy-RRTMG suites
  ingest (preprocessing): root 3 forcing times x 0.42 GiB each, 1 resident at a time + 3 nest initial state(s) 4.72 GiB, all resident for the single export transaction = 5.19 GiB resident; peak envelope 9.25 GiB
    ingest envelope basis: itemized analysis, model state and vertical setup, x1.10 setup residual and x1.20 pool headroom, measured on four CUDA preparations (1792x1024x55 to a 3:1 nest, H100, 2026-09-28), + CUDA context
  BINDING PHASE: the forecast is the memory-binding phase at 18.19 GiB peak envelope (forecast 18.19 GiB, ingest 9.25 GiB); it fits the 19.30 GiB budget with 1.12 GiB to spare
  budget 19.30 GiB (24 GiB card presents about 22.56 GiB free, minus this suite's 3.26 GiB reserve); headroom 1.12 GiB
```

Both the wizard and `woof check` show every term, so you can always
see what priced your grid. `woof check --json` carries them as
`alloc_estimate_bytes`, `non_pool_device_bytes`,
`envelope_unmodelled_bytes`, `envelope_per_nest_fraction` and
`envelope_basis`.

**A card never hands over its nameplate capacity.** A real RTX 4080
(16,376 MiB physical) presents 15.33 GiB free to a fresh CUDA context;
a 32 GiB card with a desktop on it presented 30.27 of 31.84. The
`--card` tiers therefore size against nominal capacity minus the
larger of 0.75 GiB and 6%, so the tier is conservative against the
cards of its class rather than equal to the best one. Before
2026-08-01 they assumed the nameplate, and every ladder the 16 GiB
tier emitted failed `woof check` on a real 16 GB card minutes after
the wizard printed PASS.

## Where the envelope comes from (the accurate part)

The envelope is not a safety margin picked to look prudent; it is a
measurement of the estimator being wrong, kept visible.

### Why it is a sum and not a multiplier (2026-08-01)

Until 2026-08-01 the envelope was `factor x projection` with no
intercept. A model with no intercept cannot describe a cost that has a
large fixed term, and this one does: a CUDA context plus the
launch-time local-memory backing store is 1.5-2.9 GiB before a single
grid cell exists, and it does not move when the grid does.

A 16 GiB fleet node (RTX 4080, Linux, driver 595.58.03) instrumented
whole forecasts machine-wide with `nvidia-smi` at 250 ms across a 6.6x
span of grid size, on an otherwise idle card:

| grid | domains | itemized estimate | old x1.45 envelope | measured peak |
|---|---|---|---|---|
| 170x136 | 1 | 2.07 GiB | 3.00 GiB | 3.65 GiB |
| 224x180 | 1 | 2.75 GiB | 3.99 GiB | **4.38 GiB** |
| 340x272 | 1 | 4.82 GiB | 7.00 GiB | 5.95 GiB |
| 448x360 | 1 | 7.56 GiB | 10.96 GiB | 8.75 GiB |
| 474x378 | 1 | 8.27 GiB | 11.99 GiB | 9.25 GiB |
| 594x476 | 1 | 12.38 GiB | 17.95 GiB | 12.59 GiB |
| 630x504 | 1 | 13.76 GiB | 19.95 GiB | 13.88 GiB |
| 242x194 + 480x384 | 2 | 8.22 GiB | 11.91 GiB | 10.09 GiB |

Read the bold row: a 224x180 domain -- the *cautious first run* the
wizard's own advisory tells you to make -- was declared 3.99 GiB and
peaked at 4.38. **Below about 3.5 GiB of estimate the old envelope was
optimistic**, and above it, increasingly pessimistic, reaching +44% at
the top of the table. That also reconciles two fleet reports that
looked contradictory: a 5090 measuring ~19% *under*-prediction and this
4080 measuring 25-30% *over* are the same model read at different grid
sizes.

Fitting `peak = a x subtotal + b` over the single-domain rows returns
`a = 0.98`. The itemization predicts the pool essentially 1:1; the
residue is a constant. So the model is a sum of the three things this
estimator already knows how to compute, and only one small term is
fitted:

* the **itemized alloc estimate** -- the pool side;
* the **non-pool residency** -- CUDA context plus the local-memory
  backing store of the widest kernel your physics launches, scaled by
  the device's resident-thread capacity;
* **0.50 GiB unmodelled**, plus **5% of the estimate per nest**. The
  worst residual measured over the whole table is +0.10 GiB for a
  single domain and 4.3% of the estimate per nest for a tree, both
  rounded up.

Against every measured run above and the three 2026-07-30 Linux pilots
below, the new envelope is conservative by 5% to 26% and never lands
under a measured peak. The old one was optimistic by 10% at the bottom
of its range.

**The device matters.** The backing store is
`(frame - default stack) x SMs x threads per SM`, so the 170-SM RTX
5090 this module was calibrated on carries 2.2x what a 76-SM 4080 does.
`woof check` reads the SM count off your card whenever it is measuring
your card. When you size for a card that is not in the machine
(`--card`, `--vram-gib`, `--budget-gib`), it uses the largest SM count
sold at that capacity, which over-prices every other card in the class
rather than under-pricing any.

That device is what EVERY device-scaled term of a declared card's
estimate is priced against, in `woof domain` and in `woof check`
alike: the CUDA context, the local-memory backing store, the per-scheme
column workspaces and the shared radiation chunk workspace. The
radiation workspace is device-scaled because the batched chains size
their chunk to saturate the card, and an unstated device is not the
conservative reading of that: the shortwave chain has no width ceiling,
so with no device named it takes a fixed width (2,048 columns) that the
reference device saturates past (2,560).

The device is ONE of the two terms the two doors have to read the same
way before the envelope the wizard prints for the file it writes is the
envelope `woof check` reads back out of that file. The other is the
retained forcing-interval count, which follows from the boundary
cadence: a configuration with no staged case and no cadence hint now
answers that from the recorded producer's own registry row at both
doors, where the check used to fall back to a six-hourly default while
the wizard had already sized on the producer's hourly one.

MEASURED, `woof domain --point=39.7,-96.6 --card 16gb --ladder 12-3
--source hrrr --cycle 2026-09-17T18 --explain`: the wizard prints
`peak envelope ... = 13.74 GiB` inside a 14.54 GiB budget with 0.80 GiB
to spare, and the `woof check` it then runs on the file it has just
written prints `BINDING PHASE: forecast needs 13.74 GiB` and passes,
rc 0. One file, one number, two doors.

### Windows / WDDM: the 1.75 multiplier is retired (measured, 2026-08-19)

The multiplier came from ONE run: the four-domain reference forecast
(2026-07-28, RTX 5090 32 GiB, Windows) peaked machine-wide at 1.746x
its 16.22 GiB footprint projection. It was kept as a floor because
nobody had instrumented a small Windows run to show where the
multiplicative form and the affine form cross.

The 3080 calibration instrumented six of them, and the answer is that
the multiplier never described small configurations at all: the walk's
110x88 forecast was floored to a 9.91 GiB envelope and measured 2.6 GiB
of own contribution -- 3.8x reality -- while the affine terms tracked
every measured peak from above. On Windows the envelope is now the
affine form plus **20% of the estimate as WDDM pool slack**: the worst
measured residual beyond `estimate + non-pool` was +0.30x of the
estimate (legacy-RRTMG pool retention, whose call-peak the itemization
under-counts), and 0.50 GiB unmodelled + 0.20x covers it with 0.33 GiB
to spare. The historical 5090 observation stays recorded in
`PEAK_ENVELOPE_FACTORS` and in this section; it no longer gates
anything. The legacy-RRTMG pool residual is probably a CuPy pool
behaviour rather than WDDM's -- re-measure on Linux before assuming
that lane needs the term too.

### Linux: the three 2026-07-30 pilots, re-read

Three independent first-run pilots (2026-07-30) instrumented the
machine-wide peak with `nvidia-smi` sampling across whole forecasts:

| node | card | grid | alloc estimate | footprint projection | machine peak | peak / alloc |
|---|---|---|---|---|---|---|
| 1 | 4090 | 224x178 (12 km) + 448x352 (3 km) | 7.20 GiB | 11.31 GiB | 9.54 GiB | **1.32** |
| 2 | 4090 | 438x352 (12 km) | 7.29 GiB | 11.39 GiB | 8.99 GiB | **1.23** |
| 3 | 4070 | 342x272 (12 km) | 3.51 GiB | 4.90 GiB | 4.04 GiB | **1.15** |

All three were 6 h GFS-initialised forecasts. Under the affine model
their non-pool terms are 2.30 GiB (4090, 128 SMs) and 1.10 GiB (4070,
46 SMs), which puts the envelope at 10.00, 10.09 and 4.86 GiB against
measured peaks of 9.54, 8.99 and 4.04 -- conservative by 5%, 12% and
20%, on three cards none of which is the one the model was fitted on.

**The peak lands at 0.79-0.82x the footprint projection, not 1.75x.**
Applying the Windows envelope predicted 19.80 and 19.94 GiB against a
20.00 GiB budget on the 4090s -- so the wizard stopped growing the grid
on cards that finished 37-42% used.

**The footprint projection itself is wrong on Linux.** It adds two
grid-independent constants to the alloc estimate --
`pool_retention_residual_bytes` (2.73 GiB) and
`PROBE_DEVICE_OVERHEAD_BYTES` (1.39 GiB) -- both calibrated on one
Windows/5090 fixture, and neither visible in any of the three
measurements. At the wizard's smallest possible layout those constants
are **4.12 GiB of a 5.38 GiB projection: 77% of the floor**. That is
why a 12 GiB card could not be sized at *any* ladder depth while its
GPU sat 66% idle -- shrinking the grid could not touch the part that
did not fit.  (The 3080 calibration confirmed the same on Windows:
those constants stay in the TIER 2/3 projection display and are no
envelope terms anywhere.)

So on Linux the projection **is** the itemized alloc estimate, and the
envelope over it is the affine form described above.

**The reserve is not flat either.** It carries the same local-memory
backing store, which is a property of the SELECTED KERNEL SET: 1.93 GiB
for WSM6 + MYNN, 2.91 for the Thompson default, 3.94 for NSSL2
double-moment, all on the reference 5090 profile. The wizard used to
size against a flat 4.0 GiB and then verify against the real figure, so
both NSSL2 physics profiles emitted a config that failed their own
`woof check` at every card size. The fit loop prices the reserve from
the candidate experiment now -- the same call `woof check` makes -- so
the two cannot disagree about the same file.

None of this moves a gate: the enforced numbers remain the itemized
estimate and the measured `--alloc` legs.

If you hand-build a config, run `woof check CONFIG --alloc` before
the first long run: `--alloc` actually allocates the estimate on the
device and verifies the three-way inequality (measured pool peak <=
estimate <= budget) instead of trusting arithmetic.

## Host memory (RAM), which nothing above prices

Every number on this page so far is **device** memory. Host RAM is a
separate budget on a much larger multiplier, and on a big forcing file
it is the one that runs out first. The two fail nothing alike:

- **VRAM** runs out as a Python exception, with a traceback naming the
  allocation.
- **Host RAM** runs out as a signal. The kernel OOM killer sends
  SIGKILL, which no process can catch: your shell prints `Killed` and
  that is the whole message. systemd stopping the session scope after
  an OOM (`OOMPolicy=stop`), or a userspace watchdog such as `earlyoom`
  or `nohang`, sends SIGTERM first -- your shell prints `Terminated`,
  and woof catches that one: it prints the phase the run had reached,
  its own resident set size, its CuPy pool, the senders worth checking,
  and the paths of the worker logs your terminal never saw.

### What the forcing decode costs

The forcing is decoded on the CPU, to **float64**, on the **source**
grid, before anything touches the GPU:

```
host bytes  ~  8 B  x  2-D fields per valid time
                    x  source grid points
                    x  valid times present in the forcing files
                    x  copies retained
```

Every factor is larger than it first looks:

- **2-D fields per valid time** counts horizontal slices, not
  variables. `woof fetch --source era5` asks CDS for 5 pressure-level
  variables on the 37 standard levels plus 23 single-level variables --
  208 full 2-D arrays at every valid time. The certified GFS ladder is
  5 x 21 + 19 = 124. These are the retrievals this tool writes; a
  narrower request decodes proportionally less, and `woof check`
  prices the count your forcing files actually carry rather than this
  one.
- **Source grid points**, not target grid points. The decode happens
  before horizontal interpolation, so a global field costs what a
  global field costs however small your domains are.
- **Valid times present in the files**, not the times the forecast
  integrates. The input catalog is built from the forcing FILES alone
  (`build_input_catalog`, `woof/ingest/preflight.py`) and takes the
  longest contiguous run of valid times at the declared cadence.
- **Copies retained** is 2 while the root is being prepared: the
  decoder's own arrays and the frozen snapshots copied from them are
  reached through separate process-lifetime caches in
  `woof/ingest/grib.py`. Both are released the moment preparation
  returns, so this is a **peak**, not a lifetime cost -- but it is the
  peak, and it coincides with the root's initial state and every
  lateral-boundary frame being built on the card. The input catalog
  keeps one frozen set after that, because a nest re-ingests the same
  forcing onto its own grid.

Worked, at ERA5's 208 fields per valid time over 8 valid times:

| forcing extent | source grid | per valid time | 8 times, held twice |
|---|---|---|---|
| global, 0.25 deg | 721 x 1440 | 1.61 GiB | **25.74 GiB** |
| 40 x 50 deg box, 0.25 deg | 161 x 201 | 51.4 MiB | **0.80 GiB** |

A factor of 32 for the same forecast on the same card. It is why the
first question to ask about a run that died without saying anything is
how big the GRIB is.

**Rule of thumb: budget 8-16x the forcing GRIB's size on disk.** The
band is the file's packing width: float64 held twice is 16 B per point
per field against 2 B on disk at the usual 16-bit packing (8x), rising
as the file packs tighter. Treat it as a floor. Two further copies are
real and deliberately not claimed above -- the flat bridge buffer a
combined pressure-plus-surface file keeps pinned, and the second merged
set a run holds when the catalog and the runtime decode under different
keys -- and CuPy, the two Python processes of a supervised run, and the
OS all sit on top.

### What the run tells you

`woof run` prints one line per forcing decode, before it builds the
tree, naming what was decoded and how much of it this forecast reads:

```
forcing decode: 8 valid times, 12.69 GiB of host memory (float64, on the source grid); this config's 86400 s run integrates to 2014-11-05T00:00:00 and needs 5 of them.
  3 lie beyond that end and hold 4.76 GiB.  They are still PREPARED -- every decoded time at or after start_time is interpolated, initialized, kept as a boundary frame, and the last one is this case's final analysis -- so refetching a narrower window drops them from the prepared case as well as from memory.
  The decoded window comes from the forcing FILES, not from run_seconds: build_input_catalog never sees it, so a shorter run does not shorten the decode.
```

(a global 0.25 deg ERA5 window, 42 h of it, behind a 24 h forecast --
the row above, one copy, which is what the mapping this line sums is
holding.)

Those are summed `nbytes` over the arrays the run is actually holding,
not an estimate from a grid shape. A "lie beyond that end" line is the
signal to re-fetch a shorter window -- and it is not a free edit: those
times are prepared as boundary frames and the last of them is the
prepared case's final analysis, so dropping them changes the prepared
case as well as its size.

### Three levers, in the order they pay

1. **Subset the area.** By far the largest. Ask for the outer domain's
   footprint plus a margin rather than a global field:

   ```bash
   woof fetch --source era5 --cycle 2014-11-04T00 --hours 24 \
     --area 25,-110,50,-70 --out data/era5
   ```

   For ERA5 that writes the two-part CDS request and the script that
   retrieves it, with `--area` as the box CDS crops to; for the
   downloading sources it crops the fetch itself. `woof domain`
   suggests areas with the required margin already built in.

2. **Ask for only the window the forecast consumes.** The host cost is
   linear in the number of valid times in the files, and the catalog
   charges for all of them: 42 h of 6-hourly ERA5 behind a 24 h
   forecast is 8 times decoded where 5 are needed, a 37.5% overcharge.
   Re-fetch with `--hours` at the forecast length. Unlike lever 1 this
   one is not free: every decoded time at or after `start_time` is
   prepared as a lateral-boundary frame and the last is the prepared
   case's final analysis, so a narrower window changes what is verified
   against as well as what is held.

3. **Nothing else.** Smaller domains, fewer nests, fewer vertical
   levels and `column_chunk` are real VRAM levers and do nothing here:
   this cost is priced on the source grid and the source file.

### Shortening the forecast does NOT shorten the decode

`run_seconds` is not an input to the catalog, so a shorter forecast
decodes exactly the same forcing. That makes it useless as a remedy and
useful as a **diagnostic**. Set

```toml
[experiment]
run_seconds = 3600.0
```

and re-run. Dying at 3600 s as well, with `--outdir/run-progress.json`
reading `preparing:build-domain-tree`, is the decode, and only a
smaller forcing file fixes it. Surviving at 3600 s and dying at full
length is not the decode; that is a forecast problem, and
`--outdir/worker-01.stderr.log` has the traceback.

### A tiled GFS preparation holds its grid in RAM as well

A config with `[tiles]` prepares GFS forcing on the CPU, so that phase
holds no card memory and everything it builds is host RAM: the start
time's analysis and state, every nest's initial state, the
lateral-boundary tables, series and frames for every forcing time, and
the setup temporaries. Unlike the decode above, this term grows with
the target grid, the vertical levels and the forecast length, so those
are its levers, with a machine that has more RAM.

Two figures describe it, and `woof check --json` reports both. The
estimated peak (`ingest_host_preparation_bytes`) is what `woof domain`
sizes a tiled domain against, keeping the same 5% headroom off the RAM
that it keeps off the card. The floor
(`ingest_host_preparation_floor_bytes`) counts only the arrays the
preparation holds at once, with no temporaries; `woof go` and `woof
check` refuse a configuration whose floor is more than all of this
machine's RAM, before the download (`woof check` exits 5;
`--no-host-memory-gate` skips it there), and warn when only the
estimated peak is. Against real preparations from 6 h to 72 h forecasts
and 49 to 96 levels, the floor came to 0.74 to 0.89 of the measured peak
and the estimate to 1.01 to 1.11 of it. For nested trees of 2 and 3 domains
the floor came to 0.68 to 0.94 of the measured peak and the estimate to
1.01 to 1.12 of it. Every one of those preparations ran eight worker
threads, and a CPU preparation starts no more than eight on its own,
however many CPUs the machine has, because its peak grows with its
thread count: on a 64-vCPU machine the 744x594x49 6 h preparation peaked
at 7.10 GiB with eight threads and at 7.79 GiB with 64, past its
7.50 GiB estimate, and on a shared 64-vCPU machine the eight-thread preparation was no slower.
`--preprocess-workers` starts more when you name them; the estimate does
not price that.

## Windows / WDDM notes

- On Windows the display driver (WDDM) owns device memory; the
  usable budget is what the driver grants, not the sticker capacity.
  The preflight reads the real budget and prices against it (measured
  30,472,743,936 B granted on a 32 GiB card).
- Desktop compositing holds VRAM (~3.2 GiB on the acceptance
  machine's desktop). `woof check` measures *free* VRAM at check
  time; close what you can before a big run.
- Consumer GeForce cards have no ECC. We treat sustained operation
  near the WDDM budget as a reliability risk, not an achievement:
  the reserve, and the headroom the fit loop leaves on top of it,
  exist so routine runs never operate there. The reference run that peaked 57.1 MiB under budget
  completed cleanly and bit-deterministically -- and is exactly the
  margin the sizing model now prevents. What running the forecast
  twice and comparing bytes does and does not detect in place of ECC
  is stated precisely in [DETERMINISM.md](DETERMINISM.md).
- Redirected stdout is block-buffered on Windows; watch the run's
  progress file, not the log tail: `run-progress.json` in `--outdir`
  for the config-driven `woof run` route, `evidence/progress.json` for
  the domain-tree tool route, and `progress.json` for the
  single-domain tools
  ([FIRST-LIGHT.md](FIRST-LIGHT.md#5-run-measured-6-h-forecast-in-36-min)).

## Linux notes

- No WDDM: the budget is the CUDA-reported free memory minus your
  `--reserve-gib`. The same affine estimator applies without the WDDM
  pool-slack term (see above), so the same card sizes a somewhat larger
  grid than its Windows twin -- both lanes are measured now.
- Throughput is better than the Windows numbers below suggest.
  Node 2's 438x352x49 single domain at dt 60 s, Morrison + RTE-RRTMGP +
  YSU + Noah + KF, ran 6 simulated hours in 400 s on a 4090 --
  **0.147 wall-s per simulated minute per Mcell**. The same physics and
  time step on the Windows/WDDM 5090 costs 0.229 by the same measure, on
  the stronger card: the gap is the platform, not the hardware, so size a
  Linux run against the Linux figure rather than the Windows tables
  below.
- Output volume, not VRAM, is the binding constraint on a long Linux
  run: node 1's 6 h two-domain 12/3 km forecast wrote 24 GB of
  `wrfout` (32 frames at 15-minute cadence), and node 2's 438x352x49
  frames were 651 MB each.
- CUDA preprocessing and the deterministic Rust CPU preprocessing
  backend are both exercised on Linux; the sealed Linux runtime
  archive bundles the bridges and CPU library
  ([docs/install.md](../install.md)).
- The stock-WRF interoperability receipts (serial and 12/24-rank MPI)
  were produced on Linux nodes ([WRF-INTEROP.md](WRF-INTEROP.md)).

## Throughput reference points (all measured, RTX 5090)

| workload | rate |
|---|---|
| 250x200x49 single domain, Morrison + RTE+RRTMGP + YSU + Noah + Kain-Fritsch, dt 60 s (the [FIRST-LIGHT acceptance run](FIRST-LIGHT.md#5-run-measured-6-h-forecast-in-36-min)) | ~0.55 s/step incl. output; 6 h in 3.6 min |
| four domains 12/3/1/0.5 km to 400x400, matched-run configuration | 67.2 wall-s per simulated minute whole-tree (61.4 pre-convective, 72.9 convective) |
| 500 m offline downscaled child, 400x400x49, dt 2.5 s | 0.91 s/step warm; 3 h in 66 min |
| legacy RRTMG vs RTE+RRTMGP (same 3-domain stack, radt 12/3/1) | 34.8 vs 18.7 wall-s per simulated minute |

Absolute numbers are properties of that machine (they vary up to ~30%
between sessions on the same box); ratios travel better than absolutes.

## FP32, subnormals, and GPU-model caveats

The model state is FP32 throughout, matching WRF's default REAL. Two
machine-level facts worth knowing:

<!-- BEGIN GENERATED ftz-statement: hardware-fp32-subnormals (tools/ftz_receipt/render_statement.py) -->
- FP32 subnormal handling was measured on this machine's GPU
  (NVIDIA GeForce RTX 5090, compute capability 12.0, driver
  13.3 (13030)) across the 6 compile routes the model
  uses, crossed with 6 arithmetic mechanisms.  The answer
  depends on the route:
  - `R1` loader RawModule (`woof/core/kernels/__init__.py:84`,
    woof.core.kernels.load_module), effective NVRTC options `-std=c++17`
    `-ftz=true`: `flush-to-zero` on 6 of 6 mechanisms [disassembly
    `tools/ftz_receipt/receipt/sass/r1.sass`]
  - `R1-ftztrue` loader RawModule + explicit --ftz=true (control)
    (`woof/core/kernels/__init__.py:84 + control flag`, cupy.RawModule),
    effective NVRTC options `-std=c++17` `--ftz=true` `-ftz=true`:
    `flush-to-zero` on 6 of 6 mechanisms [disassembly
    `tools/ftz_receipt/receipt/sass/r1_ftztrue.sass`]
  - `R2` direct NVRTC with the shortwave option tuple
    (`woof/core/rrtmg_sw.py:2906`, cupy.cuda.compiler.compile_using_nvrtc),
    effective NVRTC options `-std=c++17` `--ftz=false`: `ieee-agreement` on
    6 of 6 mechanisms [disassembly `tools/ftz_receipt/receipt/sass/r2.sass`]
  - `R3` direct NVRTC + cuda.function.Module (`woof/core/rrtmg_lw.py:3733`,
    cupy.cuda.compiler.compile_using_nvrtc), effective NVRTC options
    `-std=c++17` `--ftz=false`: `ieee-agreement` on 6 of 6 mechanisms
    [disassembly `tools/ftz_receipt/receipt/sass/r3.sass`]
  - `R4` CuPy-generated ReductionKernel (`woof/core/mynn_pbl_gpu.py:296`,
    cupy.ReductionKernel), effective NVRTC options `--std=c++17`
    `-ftz=true`: `flush-to-zero` on 6 of 6 mechanisms [disassembly of 6
    objects, `tools/ftz_receipt/receipt/sass/r4_0.sass` and siblings]
  - `R5` inline PTX without .ftz (`woof/core/kernels/__init__.py:84`,
    woof.core.kernels.load_module), effective NVRTC options `-std=c++17`
    `-ftz=true` (this route rides `R1`'s compile): `ieee-agreement` on 4 of
    6 mechanisms; `not-applicable` on 2 of 6 mechanisms [disassembly
    `tools/ftz_receipt/receipt/sass/r1.sass`, the object `R1` compiled]
  `R5` and `R1` are kernels inside ONE compiled object -- same device, same
  flags, one compile -- and they did not measure alike, so on this device
  the outcome follows the instruction the compiler emitted rather than the
  hardware alone.
  The control arm is essential: the 3 distinct bit tables among the 6 arms are
  what shows the pipeline responds to the flag at all.
  The consequences reach physics: each known instance is recorded in the
  physics registry ([PHYSICS.md](PHYSICS.md)), and the radiation preparation
  path routes one subnormal-sensitive block through the host by design.
  Evidence: the device bit table `tools/ftz_receipt/receipt/bitpatterns.csv` (both
  passes byte-identical: true), the objects the routes themselves compiled,
  `tools/ftz_receipt/receipt/cubin/`, and their disassembly,
  `tools/ftz_receipt/receipt/sass/` (Cuda compilation tools, release 13.0,
  V13.0.39), all recorded in `tools/ftz_receipt/receipt/receipt.json`.
<!-- END GENERATED ftz-statement: hardware-fp32-subnormals -->
- The direct consequence is a branch flip on physically negligible inputs.
  Such a perturbation can grow into differences in resolved fields during a
  convective forecast. Its effect on a forecast has not been measured here.
- Determinism holds per build and hardware: the reference run
  reproduced output frames SHA256-identically across a mid-run kill
  and relaunch. No cross-GPU or cross-driver bit-identity is claimed.
  The pin set that "per build and hardware" actually names, and the
  three mechanisms that make it necessary, are in
  [DETERMINISM.md](DETERMINISM.md).
