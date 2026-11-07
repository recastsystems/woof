# 6. Running a forecast

## 6.1 The door

```sh
woof hex forecast \
  --mesh    x1.40962 \
  --grid    assets/x1.40962.grid.nc \
  --static  assets/x1.40962.static.nc \
  --init    work/x1.40962.init.nc \
  --init-source "GFS 2026-08-12 06Z" \
  --start-time 2026-08-12_06:00:00 \
  --hours 1.0 --history-every-minutes 30 \
  --out work/fc-01 \
  --case-label quickstart
```

The door resolves the mesh through the registry, cross-examines the
supplied files against that row's pinned bytes, admits the request against
device memory measured at that moment, runs the integration, and writes
history frames plus one run receipt into `--out`. On success its last line
is the render command for what it just produced.

**It runs from the wheel.** The drivers ship inside the package
(`woof/hex/drivers/`) and verify their own executing modules by SHA-256
before CUDA is touched; because they load from the package that executes,
the copy verified is the copy launched. Through 0.3.1 they lived in the
repository's `tools/`, outside the wheel, and the door refused every install
by name; `--repo` is still accepted for older scripts, reported, and not
used.
Measured 2026-09-26 from a wheel built from this tree and installed with pip
into a clean virtualenv, run from a directory outside any checkout: the
contract deck on a 937.5 m point cull (13,227 cells) passed in 126 s, and a
10-minute forecast on it ran 120 steps of 5 s in 50.0 s of integration, 73 s
at the door, on a shared RTX 5090, its receipt naming the installed engine.

**The physics seam is the installed `woof`.** The port pins the engine's
physics seam by the SHA-256 of sixteen individual woof source files; the
run verifies all sixteen at launch and refuses any mismatch by name. With
no `--gpuwm-checkout` it verifies the installed distribution (every engine
from 2.5.8 carries all sixteen in `site-packages`) and writes its version,
the SHA-256 of pip's `RECORD` and its `direct_url` into every receipt and
every history file, so the seam source it executed is named. Pass a woof
**git clone** at the pinned tag as `--gpuwm-checkout` and the receipt names
it by HEAD, tree and dirty paths instead; a dirty pinned file refuses. A
directory that is neither (an unpacked tarball, a copied `site-packages`) is
refused by name, because it names neither a commit nor a record. Through
0.3.1 the clone was mandatory, only for that commit, which kept the forecast
out of reach of a plain install.

Beyond the install: a **registered mesh pair** (chapter 4), an **init** for
it (chapter 5), and **device memory for the mesh**, which the door now
decides for you rather than leaving you to discover mid-run (6.2).

Timing, for the shape of it: the 40,962-cell global mesh ran 1 h in 30
steps at dt=120 s, 97.8 s of integration and 146 s wall, on a 16 GiB
RTX 5070 Ti. Full physics throughout: WSM6 + Grell-Freitas + YSU +
YSU-GWDO + revised-MO + NoahMP + cloud fraction + legacy RRTMG. At the
other end, the 163,842-cell mesh has run 24 h in 720/720 steps at about
3.07 s/step on an RTX 5090 (README, *What is proven*).

Notable flags:

- `--init-source`: a provenance sentence recorded in the receipt. The
  init itself is the authority for the start time, and `--start-time` is
  an assertion against it.
- `--horiz-mixing`: default `2d_smagorinsky`, the native Registry
  default, ported and A/B-proven; `off` is a control lane, reported as the
  configuration native itself cannot integrate on convective cases.
- `--stop-on-refusal`, when the model refuses to publish a step, stop and
  write the receipt for the frames already committed instead of aborting
  with no receipt. No validation is relaxed.
- `--preflight`: 6.2.
- `--scratch`: the kernel cache. It defaults to a **sibling** of `--out`,
  never inside it, and an explicit `--scratch` inside `--out` is refused:
  a cache tree inside a directory you are about to hand to the renderer
  publishes temporaries as products.

Under the door, unchanged: `src/hexcore/drivers/run_cuda_v841_forecast.py` is the
arbitrary-case driver whose `execute_forecast` is the integration loop,
and `src/hexcore/drivers/mpas_mesh_binding.py` is the registry it binds through. Both
remain runnable directly, and
`tools/run_cuda_v841_forecast_mesh.py --selftest` still validates the mesh
binding in both directions: the correct bind must pass and a wrong name,
an unregistered name and a swapped file must each be refused by name.
Run for this manual: 5/5 PASS.

## 6.2 Will it fit?: the admission gate, and `--preflight`

**Before anything else, ask the card.** The door does, on every run:

```
ADMISSION mesh=x1.40962 cells=40,962 card=68 SM row=measured global predicted=5,682.0 MiB margin=884.0 MiB free=9,097.0 MiB of 10,239.5 MiB -> admitted
```

That is an RTX 3080 answering **on its own row, with no flags**: the card,
its 68 multiprocessors and its 9,097.0 MiB free all read from the driver. The door reads the
card's multiprocessor count from the driver at the moment of the decision
and selects or derives that card's row; free memory is measured at the same
moment, because the memory your desktop is already holding is part of the
answer and is never a budget carried from a previous run or a previous
card.

**You no longer type your card's row on the command line.** Until
2026-08-27 you did: the door priced every card with the 170 SM part's fixed
term unless you passed `--device-fixed-mib` and `--device-bytes-per-cell`
yourself, so a 10 GiB desktop refused `x1.40962` and `v15.150.38857`, two
meshes it had been measured running with 2,244 MiB to spare. That was
the borrowed-row defect and it is fixed; the flags survive for a card whose own ledger
you have run and want to override with.

**What the footprint model is.** Not a line in cell count. Chapter 4.6 has
the shape; the short version is a card core, plus the physics workspaces
that are sized to the threads the *card* can hold in flight (so they stop
growing once your mesh is bigger than the card), plus a per-cell term. The
margin held back is not a flat number either: it is this card's RRTMG
shortwave workspace (the largest block in the footprint that does not scale
with the mesh) plus 11.2 MiB of instrument convention. Both are measured
and both name what they prevent.

**One admission surface.** The door and the driver's own free-memory floor
answer from the same sum, through `woof.hex.device_admission`, and the door
forwards the identical requirement to the driver
(`--required-free-bytes`), so the two gates cannot disagree. It was not
always so: the driver once floored at an asserted 24 GiB scaled linearly
per cell, about 6.0 GiB on `x1.40962` against a then-measured 9,948 MiB
peak. A card between those numbers passed the floor, loaded the mesh,
compiled the kernels, and *then* died inside a CuPy allocation part-way
through the integration, after burning the time it took to get there. That
is the concrete breakage this gate prevents, and the floor is the measured
row (re-derived 2026-08-25 and re-fitted at the merged tip 2026-08-26) so
the two gates are one.

The refusal names the shortfall, the margin's own two components, the row it
priced with, and the fitted alternative: this one is the same RTX 3080 and
the same free reading, asked for the 163,842-cell `x4.163842`:

```
device memory admission refused --mesh x4.163842: the fitted footprint for
163,842 cells is 17,000.2 MiB and the decision holds back 884.0 MiB, so it
needs 17,884.2 MiB free; this device reports 9,097.0 MiB free of 10,239.5 MiB,
short by 8,787.2 MiB. ... This card fits the registered mesh(es)
conus-x1.2971, r4.75.11020, ... u96.64002, v15.150.38857, x1.40962 at this
moment (68,440 cells fit) ...
```

When a smaller registered mesh *does* fit the memory measured, the refusal
names it instead, and on this card that list now includes the two meshes
the retired row refused.

**On Windows, the number the door reads is not the number `nvidia-smi`
prints.** Measured in one process on the RTX 3080 above, at one moment: the
CUDA driver reported 9,097.0 MiB free where `nvidia-smi` reported
6,778 MiB, a 2,319 MiB disagreement. That is WDDM: the driver reports what
it believes it could *obtain*, including memory the OS could evict from
other clients, not what is unallocated right now. The door reads the CUDA
driver deliberately, because that is the allocator the run allocates
through, but it means an admission on Windows is optimistic relative to
`nvidia-smi`, and on a marginal card a run can be admitted that WDDM then
cannot deliver. If a run the door admitted dies in CuPy on Windows, this is
the first thing to check.

**The row flags are an override now, not a procedure.**
`--device-fixed-mib` and `--device-bytes-per-cell` still exist and still
have to be given together (half a row mixes this card's core with another
card's slope, which is a footprint nothing ever measured) but you reach
for them only when you have run this card's own ledger
(`tools/device_memory_ledger/hex_ledger_probe.py`) and want to override
what the door selected. `--headroom-mib` overrides the margin the same
way; its default is no longer a flat 512 MiB but the model's own two named
terms, priced from your card.

**Architecture is admission too, and every NVIDIA architecture NVRTC can
compile the kernels for is admitted, with no flag.** The kernels are
compiled by NVRTC for your card when the run starts, and they ask for
nothing an older card lacks, so the floor is the installed CUDA toolkit's
(CUDA 12 or 13): compute capability 7.5 (Turing) and newer, and compute
capability 12.0 from CUDA 12.8 on. A card NVRTC cannot compile for
is refused by name, because no kernel could be built for it. The one other
architecture refusal is the IEEE check below, on an unanchored card, when
ordinary arithmetic is not IEEE; nothing else about the architecture
refuses a run.

What differs between architectures is the evidence behind the numbers. The
port's numerical contract was first measured on sm_120; an architecture
holds an **anchor** when the same contract has been measured again on real
hardware of it, a contract receipt and a frozen-authority anchor, each row
pinned to its evidence by SHA-256
(`src/hexcore/cuda_backend/arch_admission.py`). **sm_86 holds one**
(2026-08-25, a 10 GiB RTX 3080) and **sm_89 holds one** (2026-09-24, an
RTX 4090, so the RTX 40 series and the L40S), each with the guard-cost
ceiling measured on its card (1.75 and 1.55). The preflight prints one line
saying which it is, with the evidence pin of an anchored row:

```
ARCHITECTURE sm=sm_89 -> admitted, anchored: the 2026-09-24 anchor (evidence/sm89-tier-20260924/RECEIPT.md, evidence pin 15357eb4d576853f...)
```

On a card of any other architecture the line reads `admitted, unanchored`,
and before the run the door measures the numeric route on that card in one
tiny compile and launch: what the route does with an FP32 subnormal, what
the port's guarded fallback returns for it, whether `--fmad=false` keeps
`a*b+c` separately rounded, and whether ordinary FP32 and FP64 arithmetic
agree with IEEE 754. The line ends with what it found. Only the last check
refuses, because on such a card every number the run produced would be
wrong; the others are recorded. Every receipt (the door's, the driver's and
the proof harness's) records the compute capability and either `anchored`
or `unanchored`. An anchored row carries its evidence pin; sm_120, the
proven contract floor where the contract was first measured, is anchored
and carries no pin. No receipt claims an anchor its architecture does not
hold, and `woof hex doctor` names the card, its compute capability and
the same status before you run. `--preflight` answers this half from a
device-properties read and NVRTC's own target list, so it and the run
agree about the architecture.

**`--preflight` answers the whole question.** It runs every check
(inputs, the mesh bind, the admission decision, and the driver's own
source-pin and host preparation) reports all of them, and exits 1 if any
would stop the run. It is the one mode where a missing input or an
unadmitted card is *reported* rather than raised, because a user asking
"will this run?" has often not built every input yet, and the card
question is the one no file fixes:

```
INPUT MISSING --init names a missing file: .../x1.40962.init.nc.  Build one with `woof hex init` (chapter 5 of the manual).
BIND mesh=x1.40962 rebound=True dt=120.0 s
ADMISSION mesh=x1.40962 cells=40,962 card=68 SM row=measured global predicted=5,682.0 MiB margin=884.0 MiB free=9,097.0 MiB of 10,239.5 MiB -> admitted
ARCHITECTURE sm=sm_86 -> admitted, anchored: the 2026-08-25 anchor (evidence/sm86-tier-20260825/RECEIPT.md, evidence pin 75437c663c9894dc...)
PREFLIGHT mesh=x1.40962 problems=1 status=preflight_refused
```

The registered row's TIMESTEP is collected the same way (2026-08-26).  A
row declaring a timestep that holds no anchor used to end the preflight on
the spot, so the answer to "will this mesh fit my card?" was never printed
for exactly the meshes people ask it about.  Both answers now come back in
one pass: here for `v15.150.38857`, which declares 60 s:

```
INPUT MISSING --mesh v15.150.38857 declares dt=60 s and selects GF (resolution) with surface/PBL every step (welded).  config_dt=60 s with GF holds no timestep anchor: ... a schedule receipt and an integration anchor at 5 s (GF), 5 s (convection off), 20 s (GF), 20 s (convection off), 75 s (GF), 100 s (GF), 120 s (GF) and at nothing else ...
ADMISSION mesh=v15.150.38857 cells=38,857 card=68 SM row=measured global predicted=5,488.1 MiB margin=884.0 MiB free=9,097.0 MiB of 10,239.5 MiB -> admitted
PREFLIGHT mesh=v15.150.38857 problems=4 status=preflight_refused
```

**The timestep is not pinned to 120 s.** Five configurations hold an earned
anchor (120, 100, 75, 20 and 5 s) and the registry is keyed to the
cumulus selection as well as the timestep, because how often a scheme is
CALLED is part of what an anchor's forecasts measured
(`woof.hex.dt_admission`). Only 120 s carries a native MPAS-A reference
and only it ever can. For a new mesh the rule of thumb is `dt ≈ 6 × dx` in
km. Read the anchor you are about to use before trusting its weather: each
one records what its band did against a 120 s control on the same card,
mesh and init, and the 20 s and 5 s rows record a divergence rather than a
match. An anchor certifies that a timestep integrates finitely and
deterministically at the cadences it names, and nothing more.

Preflight writes the same receipt a run does, with a `preflight_problems`
list. It touches no CUDA beyond the memory and device-properties queries.

## 6.3 What the door writes

`--out` receives, on a clean run:

- `cuda-history.<valid-time>.nc`, one per capture, the files the render
  door takes;
- `cuda-v841-forecast-receipt.json`, the driver's own receipt (6.4);
- `forecast-receipt.json`, the door's receipt: the resolved request, the
  admission decision with every number it was made from, the mesh-binding
  receipt, the driver's receipt embedded whole, the history files, and the
  exact `woof hex render` command for them.

The door prints that command as its last line, so the next step is a
paste rather than a lookup. Its form:

```
NEXT woof hex render --history <out>/cuda-history.<valid-time>.nc --mesh <grid> --out <out>/png --simulation-start <start>
```

## 6.4 What a run claims: read the receipt

Every run writes `cuda-v841-forecast-receipt.json`. Its `claim` names
exactly what happened (the mesh, the physics set, dt, the init). Its
`nonclaims` and `dropped_guarantees` are just as binding: among them:

- **no comparison against a native MPAS CPU run** is performed for these
  forecasts;
- **no checkpoint is written** and restart identity is not re-established
  per run (see 6.5);
- **forecast skill is not established**, these are engineering forecasts;
- the receipt carries `gf_native_parity_claim: false` beside
  `gf_declared_divergence`, the declaration of chapter 3's GF generation
  gap, and so does every history file written.

Determinism is a real guarantee: the same run twice produces byte-identical
output, which is also the memory-corruption screen on cards without ECC
(dual-run byte comparison). This manual's local-time-stepping check ran the
same case twice through two configurations that must coincide on a uniform
mesh and got byte-identical history, on hardware, in one session.

### The step's host path: what runs on the card, what the host reads

Every forecast step is a two-owner transaction (the dycore, then the
physics seam) followed by a host health gate that refuses the moment the
state stops being finite and physical. The dycore's kernels are the step's
cost and are the only part that touches the prognostic arithmetic; the
checks around them are meant to be free, and on the 43,884-cell point cull
they were not: a fifth of every 0.38 s step was the health gate reading 31
scalars one drain at a time, every recovery kernel ran twice under a timed
launch, the CUDA admission was re-measured 59 times a step, and the
garbage discipline (7.4) issued 3,328 tiny fills a step.

Since the hostpath change every check runs on the device over the solve
region and is read back once (`woof.hex.cuda_solve_region_v841`): the gate's
envelope of every field in one launch pair and one read, the density and
recovered-state validations as device kernels that set the step's flag,
the discipline's restoration of a launch's padded columns as one launch,
each recovery kernel launched once and untimed, and the admission measured
once per process. None of those kernels computes a value that reaches a
field (they compare, select and store what they are handed) so a run's
history is byte-identical to the run before the change; that identity was
measured on the point cull, two arms in one session on the same card, and
the contract deck on the cull's rings passed at the new kernel-set digest.
The cost: the composite step 0.3143 s before, 0.2957 s after (1.06x) on
an RTX 5090; the health gate 56.2 s to 0.2 s an hour; the loop as you wait
for it (integration, gate and history) 0.3973 to 0.3011 s per step, 1.32x. The receipt still carries the same
`step_health` entries (the same numbers, from the same exact reductions)
and the discipline's receipt gains `fused_restoration_launches` beside
`garbage_columns_restored`.

What the step still pays for on the host, with the reason: one read of
the dycore's validation flag before publication, the two preparation
flags (each raises with the offending field), and the radiation call's
host round trip inside the engine's legacy RRTMG.

The physics seam's rollback export is no longer one of them. Before phase
one of every step the adapter used to copy every persisted seam array to
the host (447 MB and 192 copies a step on the point cull, about 24 ms of
the host blocked) so that a refused step could be rolled back. Only
`--stop-on-refusal` ever reads that export: it keeps the last committed
state and writes its frame. So the forecast takes the export only under
`--stop-on-refusal`; without it a refused step ends the run as it always
did, and the receipt says the seam was retired rather than restored
(`physics_rollback` in the driver receipt, `transaction_rollback` in every
step receipt). No output byte depends on it.

## 6.5 Restarting a run: what exists and what does not

**What is proven:** checkpoint/restart with bitwise-identical continuation
is a property of the port, established in the full-physics proof harness
(`src/hexcore/drivers/run_cuda_v841_full_physics_x4.py`): a host checkpoint is taken
mid-run in a fresh process, restored, and the restarted history file
matches the uninterrupted run's byte for byte. The checkpoint schema is
versioned (v3: it additionally carries GF's per-step advective-forcing
pair, whose omission was found and closed by exactly this gate; CHANGELOG). A pre-v3 checkpoint is
refused by name instead of resuming wrong.

**What does not exist in 0.2.0:** a user-facing resume flag on the
forecast driver. Engineering forecasts run uninterrupted and write no
checkpoint: that is a stated dropped guarantee in every receipt. If a
forecast stops (power, refusal, operator), you re-run it from its init;
determinism means the re-run reproduces the original trajectory exactly up
to where it stopped. Treat "restart" in this release as a proven property
of the engine and a capability of the proof harness, not a workflow
feature of the forecast lane.

## 6.6 Local time stepping: opt-in, with its measured cost

On a variable-resolution mesh most columns are far coarser than the finest
one, yet the acoustic sub-step is sized for the finest.
`--local-timestep` lets coarse columns take fewer, longer acoustic
sub-steps, chosen from the grid file's own `dcEdge`:

```sh
... tools/run_cuda_v841_forecast_mesh.py --repo <checkout>/tree --mesh x1.40962 \
  -- ... --local-timestep
```

`--local-timestep-rates` sets the ladder (default `1,3`) and
`--local-timestep-buffer-rings` the buffer width at class boundaries
(default 1).

**Off by default, deliberately.** Native MPAS-A v8.4.1 has no local time
stepping, so there is no byte-identical implementation and never can be:
turning it on takes a declared divergence from native; leaving it off gets
the pinned arithmetic unchanged. Two facts follow from the mesh, and both
were verified on hardware:

- On a **quasi-uniform** mesh every column lands in one class and the
  option is inert: the run above (x1.40962, `--local-timestep`) produced
  history **byte-identical** to the default run, re-verified for this
  manual on the proving node.
- On the published variable-resolution mesh **the option does not pay**:
  measured one model hour, x4.163842, dry lane, RTX 5070 Ti, 1.268 s/step
  default against 1.283 s/step with the option, 0.988x. The acoustic loop
  is 23.5 % of a model step and this mesh admits a 23.0 % acoustic saving,
  so the whole-step ceiling is 1.057x before bookkeeping, and the released
  `(1,3,6)` schedule caps the feature at 1.16x even in the
  every-column-coarse limit.

- On a **limited-area cull** the option is inert: every driven boundary
  cell is held at rate 1 and on the culls `mesh-plan` and `cull` make that
  is the whole coarse class, so the run is byte-identical to the default
  (point cull, RTX 5090, 2026-09-14: 0.992x the step, both history frames
  SHA-256-identical). An interface forced into the interior with
  `--local-timestep-classing` (an instrument, not a feature) is 1.07x
  slower, because on a 44,000-cell mesh every kernel runs in one wave and a
  coarse-class launch is a further wave rather than a saving
  [`docs/local-timestep-lam.md`](../local-timestep-lam.md).

Conservation shape: class boundaries are refluxed (Berger–Colella on the
acoustic mass flux), so mass and passive vapour are conservative in the
flux and approximate at binary32 rounding, measured drifts 2.6e-11 (dry
mass) and 6.4e-10 (passive qv) over a model hour against a 2.0e-8 bound,
on the x4 mesh with 1,063 interface edges (README, *Limitations*). Runs
with the option on remain bit-reproducible run to run, so the dual-run
corruption screen still works.

The stronger per-class whole-step form was measured to a no-go on current
hardware (projected 1.254x on the one mesh big enough to profit, a
slowdown on small steep meshes); the README carries
the full measurement; `tools/probe_lts_fullstep_projection.py` is the
instrument to re-run if meshes an order of magnitude larger become real.

## 6.7 When a run stops early

Mid-run validation failures are refusals, not crashes (chapter 3.4). The
one seen in practice: a vertical-velocity divergence at levels 44–47 tripped
the 200 m/s validation bound and the case stopped rather than publish a bad
step. `--stop-on-refusal` converts that into a receipted stop with the
committed frames preserved. If a run refuses on *sources* (a pinned module
or seam file whose digest moved) nothing was integrated at all; restore
the pinned bytes rather than editing the pin.

## 6.8 Limited-area forecasting: `--lbc-dir`

> **This section used to open by telling you the lane could not be opened
> with a published engine. That was true of 2.5.7, and it is no longer the
> current answer.** Walked 2026-08-27 from an installed 0.2.0 wheel against
> gpuwm 2.5.7: the `rw_mpas_mesh` that `woof fetch-bridges` staged had no
> `--cull-parent` (it dropped the flag silently and then asked for `--spec`),
> and `rw_mpas_lbc` was in no published bundle and no published source.
>
> The engine published 2.5.8, which this distribution now requires. Measured
> 2026-08-28 against the published artefacts: `woof fetch-bridges` on a 2.5.8
> install stages 26 of 26 artifacts against its packaged pins, `rw_mpas_lbc`
> among them, and `woof hex cull` drove the **staged published**
> `rw_mpas_mesh` through two real cuts of the 40,962-cell global parent, 338
> and 606 cells, grid, static and initial condition each written, 0.9 s.
>
> **What is still unmeasured:** a complete boundary set written by the
> published `rw_mpas_lbc`, and a `--lbc-dir` forecast behind it. That binary
> runs and refuses correctly by name on inputs that do not satisfy it, and
> nothing here has yet driven it to a written boundary file. Every
> limited-area number in this chapter was taken with a source-built engine and
> none has been reproduced from the published bundle.

The same door runs a **limited-area** case: a mesh cut out of a global
parent and integrated behind lateral boundary conditions instead of around
a sphere. `woof hex cull` cuts the grid, static and initial condition out
of a global case in one command; `rw_mpas_lbc` builds the boundary files
from the parent's own history frames; `--lbc-dir <directory>` hands them to
the forecast. Everything else on the command line is unchanged.

**The physics is the whole stack, not a dycore with one tracer**: WSM6 +
Grell-Freitas + YSU + YSU-GWDO + revised-MO + NoahMP + cloud fraction +
legacy RRTMG, the same set the global lane runs. Measured on a 10 GiB
RTX 3080: six hours on an
11,020-cell cull, **1,080 of 1,080 steps**, 13 history frames, median
0.271 s/step, peak **6,224 MiB**, and 343 rendered products across seven
frames.

Against a full-physics global run over the same ground, same init and same
six hours, at t+6 h over the free interior: theta **1.117 K** RMS
(r = 0.999973), precipitation r = 0.95, reflectivity r = 0.82, vertical
velocity r = 0.621. The disagreement does not grow inward from the
boundary: theta RMS by ring is nearly flat and largest at the driven
rings, where a coarse parent's interpolated state is imposed on fine cells.
`w` is the smallest and fastest field here, and a limited-area `w` product
is that run's own answer rather than the parent's.

The limited-area device stack is its own row of the admission table (6.2):
it carries a padded atmosphere, two levels of lateral-boundary state and 22
more kernels, so the door prices it as `row=... limited-area` rather than
scaling the global row. That row's core is an ENVELOPE over five measured
culls rather than a fit, so it over-predicts the smaller ones on purpose.

### The vertical grid on RRFS meshes: `--invariant`

RRFS v2 runs MPAS with an `invariant` stream: the model reads `zgrid`, `zz`,
`zb`, `zb3` and the rest of the vertical metric set from
`<mesh>.invariant.nc_L60_GFS`, not from the init file. The `conus12km` and
`conus3km` rows name that file, and when it sits beside `--grid` or
`--static` the door passes it to the driver without being asked. The console
prints a `VERTICAL GRID` line and the receipt's `vertical_grid` block says
which file was used and why.

Why it matters, measured on conus12km with physics off: the init file's grid
is 2.7 mm off in `zgrid` and 8e-5 relative in `zz`, and running on it put hex
2.0e-4 K RMS theta away from NOAA's MPAS at hour 1. With the invariant file
and the cold-start recompute of the coupled diagnostics, the difference is
3.3e-5 K, against NOAA's own rounding envelope of 2.8e-5 K.

`--invariant FILE` names a different file. `--no-invariant` reads the grid
from the init file, which reproduces runs made before this default. An
invariant file with a different level count from the init is refused before
the card is touched.

## 6.9 Cycling: following weather from one cycle to the next

One forecast is a snapshot. `woof hex cycle run` is the loop: detect in a
coarse forecast, place the fine swaths, decide per slot whether to move or
stay, cut the culls, build the boundaries, run the per-geometry contract
deck, run the full-physics limited-area forecast behind it, render, and
carry the decision into the next cycle. `woof hex cycle plan` answers what
it would do with no device opened and nothing cut. `docs/cycle-door.md` is
the page for it; this section is what a forecast reader needs.

Measured over two cycles on one real case: corridors at dt 20 s under
an anchored configuration class, both **1,080 of 1,080 steps**, and a
**delayed start**, a cycle beginning mid-window takes the parent's state
at the hour the swath wanted instead of integrating the fine grid from
hour 0. The state transplant took **0.83 s** and saved **273.8 s of card,
43 %**, against a real baseline arm on the same card and the same cull
geometry.

**Its cost is named, and it is the first hour.** The initial-condition
stream has no slot for cloud ice, snow or graupel, so a corridor started
inside an ice cloud starts without it: hour-0 reflectivity does not
correlate with the baseline at all. One hour later WSM6 has re-formed the
ice and r = 0.863; theta agrees to five decimals throughout. If your first
frame matters, that is the frame to distrust.


## 6.10 The surface/PBL cadence: selectable, welded by default

The surface layer, the land-surface model and the PBL are called on every
model step by default (`config_bldt_seconds = config_dt`, the native x4
v8.4.1 reference's own semantics). At 120 s that is 30 calls an hour; at
the 5 s a sub-kilometre mesh declares it is 720, and the 2026-09-13 per-step
profile on the point mesh put that phase at a fifth of every step.

`--pbl-cadence SECONDS` calls the stack once every `SECONDS / dt` steps and
holds its tendency on the steps between: the engine's positive-`bldt`
path, ARW's `stepbl`. Radiation keeps its own 600 s cadence either way.

```sh
woof hex forecast --mesh <row> --grid ... --static ... --init ...   --start-time ... --hours 1 --out <dir> --pbl-cadence 60
```

**Selectable, never silently the default.** It changes the forecast, so
`auto` (the weld) stays the default until a measured recommendation moves
it, and an explicit cadence records itself as `source: "explicit"` with
its calls per hour and held steps in the run receipt. The driver receipt
adds the seam's own count of due and held steps against the count the
declared cadence predicts (`physics.cadence` in `forecast-receipt.json`),
so the receipt itself is the evidence that the hold happened.

**Refused by name, before any card is touched:**

- a cadence that is not a whole number of the row's steps: the refusal
  names both numbers, the multiples of `dt` on either side, and `auto`;
  nothing is rounded for you;
- a request that is not `auto` or a positive number of seconds;
- a held cadence at a timestep whose welded configuration holds no anchor
  (anchoring the welded configuration admits the held cadence with it).

**Admitted through a derived row.** A held cadence at an anchored
`(dt, cumulus)` is admitted on the welded anchor's dycore evidence: the
outer step, its RK schedule, its clock and its byte-identical dual run are
properties of the timestep. The physics band is stamped NOT MEASURED at
the held cadence rather than borrowed, and `surface_pbl_anchor_derived_from`
in the receipt names the welded row the admission came from. `--preflight`
answers the same question from the row alone.

The measurement of record (the every-step arm against 30, 60 and 120 s on
the point mesh, per-step cost and 1 h field differences) is in
[`docs/hex-point-hrrr.md`](../hex-point-hrrr.md): every held cadence buys
the same five per cent per step and every one moves the one-hour forecast
well past the 2.6.5 to 2.7.3 seam movement, so none qualified as the
default and the weld stays.


## 6.11 Forecast presets: `--preset reference` and `--preset fast`

A preset is a row of `woof.hex.forecast_preset`, and the door turns it into
the knobs above. Two rows ship:

- `reference`, the default: the proven configuration (legacy RRTMG,
  Noah-MP, revised MO and YSU, the surface layer, Noah-MP and YSU called
  every step);
- `fast`: the same schemes, with the surface/PBL stack held for up to 30 s
  between calls (six steps at dt 5 s; the weld where dt is 20 s or longer,
  so no mesh is refused for a cadence its timestep does not divide).

```sh
woof hex forecast --mesh <row> --grid ... --static ... --init ...   --start-time ... --hours 3 --out <dir> --preset fast
```

An explicit `--pbl-cadence` wins over the row, and the run receipt says
which decided (`configuration.preset` and `pbl_cadence_source` in
`forecast-receipt.json`). A row that declared a radiation, land-surface,
surface-layer or PBL scheme other than the one the engine's column seam
builds would be refused by name: the seam takes none of them as an
argument, so such a row would run the engine's suite under its own name.

**Why `fast` is not the default.** Graded against MRMS reflectivity,
Stage-IV hourly precipitation and ASOS on four 3 h HRRR-driven forecasts of
the 937.5 m point cull, it took about 6 per cent less time per step and
scored the same on reflectivity and precipitation thresholds, and it
forecast 10 m wind worse on all four cases and 2 m dewpoint and temperature
worse on three. Use it when time matters more than the near-surface
forecast. The measurement is in
[`docs/hex-point-hrrr.md`](../hex-point-hrrr.md), the forecast presets
section.
