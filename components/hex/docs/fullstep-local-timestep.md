# Full-step local time stepping

The full-step switch is separate from acoustic local time stepping. It defaults
off. An enabled request defaults to `shadow`: the native fixed integrator still
advances every column, and an instance-local observer records the actual
split-averaged mass flux supplied to each scalar RK stage. This route is an
admission instrument, with no qualified speed or forecast skill claim.

An LTS request uses one card by default, even on a large mesh with several
visible cards. Automatic capacity selection never opts LTS into multiple
cards. Active multi-card LTS requires explicit `--samehost-devices`; its
`--samehost-layouts` and `--samehost-part` assets are verified when given and
built from the grid with the same builder automatic selection uses when both are
omitted. A moving region runs on these blocks too (docs/moving-nest.md,
`run --cards K`). Every multi-card
run used to stop at its first third RK stage ("the third-stage theta edge flux
was not observed before recovery"); the capture is repaired and unit-tested
(tests/test_fullstep_lts_multi_matched.py), and with the engine's land kernels
held per card (`woof.hex.engine_per_card`) a 2 h two-card forecast runs. It is not
identical to the one-card run (surface pressure 1.0 Pa RMS after 2 h). Shadow mode remains
one-card only. LTS itself remains off unless requested. Phase-one-ahead
physics is not enabled by this stage-4 integration.

```
woof hex forecast ... --fullstep-local-timestep --fullstep-lts-mode shadow
```

The pilot supports one resident card, RK2, three dynamics splits, any even
number of acoustic sub-steps (MPAS-A v8.4.2's default of four, or two with
`--acoustic-substeps 2`) and monotonic transport. An odd count is refused:
the middle of a coarser class's step would not be a plane of its clock, so
the scalar interface could not publish the finer class's mid-step scalars. The supported geometric ladder is
`1,2,4,8,16`, with two simultaneous neighbor-minimum buffer passes. A geometric
class is a ceiling. The receipt's transport ceiling is also optimistic until
advection, sound speed, explicit mixing and physics duration bounds are measured.
It is never labeled a complete class admission.

For each cell and level the donor reduction adds positive outward horizontal
mass flux times edge length divided by area, and positive outward vertical
mass flux times `rdzw`, then divides by source-old coupled density. It takes
the level maximum after adding those terms. The observer uses actual `uhAvg`
and `wwAvg`, with the scalar stage duration recorded separately. Physical
vertical velocity is not a substitute. Invalid geometry or nonfinite density
is a named refusal, never excluded from the maximum.

`fullstep-lts-stages.jsonl` records the source-old controlling cell and level,
stage and outer depletion, required base step and optimistic populations. The
forecast receipt adds `fullstep_lts_treatment` only when a multi-rate shadow is
attached. Fixed receipts and native numerical source files stay unchanged.
The two dry identity pilot arms compare the evolved prognostic and saved arrays;
they do not qualify physics sidecars, output files or radiation events.

`--fullstep-lts-mode active` runs a full-physics forecast with every rate class
on its own whole step, on one card or on several cards of one host
(`--samehost-devices`; section "Several cards"). It is not the default. A mesh
that classes entirely at rate one, or a forced maximum rate of one, uses the
unchanged native route (the same-host ranks when several cards are given). The
shadow still refuses more than one card: the partitioned shadow cannot count
owners alone.

The CPU clock uses integer forty-eighth seconds and eighth-second base steps.
It clips each dyadic macro before the next exact global event. At 18 seconds,
a steady 1,800-second window contains six 288-second macros and a 72-second
tail. After two native startup steps the first tail is 36 seconds. These plan
tests prove interval coverage, not stability of an active atmosphere solver.

Box-only pilot and test commands:

```
python tools/census_fullstep_lts.py --grid GRID --static STATIC --init INIT \
  --out NEW_DIRECTORY --cache CACHE --dt 18 --steps 2 --arm shadow
python -m pytest tests/test_fullstep_lts_shadow.py tests/test_fullstep_lts_pilot_hashes.py
HEX_LTS_GPU_TESTS=1 python -m pytest tests/test_fullstep_lts_cuda_monitor.py
```

The GPU commands need one CUDA card to themselves (name it with `CUDA_VISIBLE_DEVICES`). The pilot carries
six native scalar slots with physics and explicit mixing off. It writes compact
array hashes and receipts instead of raw histories. A successful pilot does not
replace the six-hour reference-physics shadow, interface conservation/order
proofs, matched full-cost timings or paired 48-hour valid-time analysis scores.
Default-on requires those gates on both target meshes and a second initialization.

The 2026-10-11 implementation stopped at the reference-physics capacity gate.
On the public 60-to-3 km case, the pinned fixed route and the shadow route both
exhausted a supplied 96 GB card in native monotonic transport. The shadow run
reached 97,025 MiB of 97,887 MiB at one-second sampling, leaving at most 0.88%
headroom against the required 10%. The two-step dry donor census did not reject
the base duration: its optimistic cell-work ratio was 1.560905. The memory failure
does not establish that full-step LTS cannot pay after baseline memory work.
The complete coupled integrator, interface reference, physics column bank,
multi-card protocol and forecast scoring were not built or qualified.

## Active stepping of the dry dynamics (stage "dynamics", one device)

`woof.hex.cuda_fullstep_lts_active` advances the dry dynamics with every rate
class at its own whole step; `tools/run_fullstep_lts_dry.py` is its box-only
front door (`--arm fixed` and `--arm lts`, same dry configuration).

Each class is one resident native driver on its own sub-mesh (the same-host
partition slicer), running the unchanged pinned step at `rate * dt`. No native
kernel or pinned numerical source is edited. The classes step FINE FIRST:

* Toward coarser neighbours a class carries a scratch zone of coarser cells and
  advances them itself, at its own shorter step, which is always stable for a
  coarser cell. A zone cell is reset to its owner's accepted state at the
  owner's step boundaries. The four rings beyond the zone are frozen at the
  owner's accepted state. The zone is deeper than a 400 m/s signal travels
  between two resets (times 1.5), so the frozen ring cannot reach an owned cell.
* Toward finer neighbours a class carries the plain four-ring halo (with three,
  the outer ring's clamped stencils reached the computed cells within a step, so
  a forecast depended on how a layout numbered its cells). The finer
  class has already covered the interval. With RK2, three dynamics splits and
  N acoustic sub-steps, every stage and acoustic time of a class is a
  multiple of 1/(3N) of its step: a plane. The clock counts ticks of
  base_dt/(3N). A finer class records its accepted split ends on the planes
  they land on; a plane no split end lands on is the linear-in-time
  interpolation of the two recorded split ends around it
  (`fullstep_lts_classes.plane_weights`). Nothing is extrapolated.
  With N = 2 every plane is a split end of every strictly finer dyadic class,
  so the halo receives states the finer class actually accepted; only two
  classes that share a step in an event tail read the mean of the
  neighbouring accepted states at odd sixths. With N = 4 the odd planes of a
  class fall in the middle of the next finer class's splits and read the mean
  of its two split ends around them; a class two levels finer supplies every
  plane directly.
* The interface register is every edge between two classes. The finer class
  owns the edge. Its FP64 mass and theta integrals over the interval replace
  the coarser cell's own estimate after the coarser class has stepped, one
  thread per cell in ascending register order. Momentum has one owner per edge
  and is not refluxed.
* Matched interface flux (default). Each acoustic sub-step of a class spans
  exactly one plane of its step. A split of its dyadic child spans N/2 planes
  (N when the two share a step in an event tail; with N = 2 one plane, the
  "sixths" of the original fix). The child records per split its accepted
  mean momentum on every interface edge (saved momentum plus mean acoustic
  perturbation, what its own density update used) into every plane the split
  covers (`fullstep_lts_classes.child_flux_intervals`). The coarser class's
  acoustic momentum kernel is tapped in its own kernel cache, and on its
  interface edges the continuity and heat equations of sub-step j of split s
  read the child's mean for plane N*s + j - 1; the third-stage heat flux
  takes the child's heat integral over the split's N planes. Over a child
  split the coarser class therefore moves exactly the integral the child
  moved. The fix was measured at N = 2 (below); its even-N port was gated at
  N = 4 for 0.4.0 (release notes). The coarser class's recovered flux and register then equal the
  child's, so the reflux is left with rounding and the large-step theta
  difference instead of the whole interface flux error. `--interface-flux
  own`, an option of the dry measurement tool `tools/run_fullstep_lts_dry.py`
  (the forecast door always uses the matched flux), restores the earlier
  coupling as a diagnostic arm.

An earlier coarse-first arrangement (a coarse class stepping before its finer
neighbour, reading an extrapolated halo) failed within thirty base steps in
every overlap layout tried; it is not in the tree.

Not covered by the dynamics stage: scalar transport (next section), physics, more than one device,
restart, re-classing during a run, and reflux of the horizontal-mixing theta
flux at an interface. Classes are static: the initial winds admit each rate
with margin, and nothing demotes a class later.

```
python tools/run_fullstep_lts_dry.py --arm lts --grid GRID --static STATIC --init INIT \
  --out NEW_DIRECTORY --cache CACHE --dt 18 --hours 6
python -m pytest tests/test_fullstep_lts_classes.py
```

## Scalar transport per class (stage "transport", one device)

`tools/run_fullstep_lts_dry.py --transport` turns on the native scalar transport
with the monotonic limiter in both arms. Every class runs the unchanged native
transport (two predictor stages, the flux-corrected third stage) on its own
sub-mesh at its own whole step. The interface law is
`woof.hex.cuda_fullstep_lts_scalars`; no native kernel or pinned source is
edited. The class driver's own kernel-cache entry for five native kernels is
replaced by "run a hook, then the same kernel", and the hooks edit the third
stage's two edge arrays (low-order flux, antidiffusive residual) at interface
edges before the native low-order, scale, limit and finish kernels read them.

* **Mass.** On an edge the finer class owns, a class's step-mean mass flux is
  replaced by the mass the finer class moved through that edge over the
  interval (the FP64 register of the dynamics stage). The class's transport
  target density then equals its refluxed density to rounding.
* **Packets.** The finer class owns the edge. The limited scalar flux of each
  of its steps (low-order plus limited residual, the two numbers its own cells
  are updated with) is added to an FP64 register. When the coarser class steps,
  its low-order flux on that edge is the register and its residual there is
  zero, so the packet is in the coarse cell's provisional mass before its
  limiter factors are formed: the limiter treats it as already spent. There is
  no scalar reflux afterwards.
* **Reserve.** Fine first means the coarse column's donor budget is drawn
  before the column's own step exists. Each coarse column on an interface
  carries a reserve, the scalar mass it held at its last accepted state. The
  finer class may not move more than that out of the column through the
  interface between two of the column's accepted states: its interface fluxes
  are scaled when it would (low-order first, then the outward residuals), and
  every accepted packet is debited. When the column steps, its own low-order
  outflow, horizontal and vertical, is reduced by what the packets took beyond
  the column's own upwind export. Both safeguards are measured against the
  mass-consistent export (the column's start mixing ratio times the dry mass
  that left it), not against the content alone: a thin level over steep
  terrain exports more dry mass in a long class step than it holds, and
  scaling its tracer without its mass broke a constant mixing ratio (0.896 to
  1.093 after 208 base steps on the 48-3 km patch mesh before this rule). The
  receipt (`scalar_interface`) counts both safeguards and books what they
  withheld against the packet throughput, per species.
* **Halo scalars.** A class's scalars are accepted at its step ends. A finer
  dyadic class ends a step at a coarser class's step start, middle and end,
  which are the three times the scalar stages read. A coarser owner supplies
  its frozen accepted state, and the scratch zone is reset at the owner's step
  boundaries as for the dynamics.

Classes are still static. With `--transport` the initial admission also asks
that the initial fluxes carry no more than 0.9 of any level's dry mass out of
a column in one class step; nothing re-admits a class later.

The native third stage ends with `max(0, value / target)`. That clamp creates
tracer mass wherever a value would be negative, in the fixed arm too (the GFS
initial file itself holds negative water vapour). `--clamp-audit` adds up the
clamped mass per species, so the change of a tracer total splits into clamped
mass and rounding in both arms.

The retired refusal: the active path refused `config_scalar_advection` because
a transported scalar would have crossed a rate interface with two different
mass integrals. It now refuses only transport with the limiter off, where the
stage that accepts one packet per edge does not run. The separate acoustic
attachment (`--local-timestep`) keeps its own limiter refusal; it is a
different schedule and was not changed.

### The rate-16/8 interface instability and the matched interface flux

Until 2026-10-12 the top rate defaulted to 8: on the 48 km mesh with a 3 km
patch (294,799 cells, base step 18 s, dry dynamics) a rate-16/8 interface
under a jet of 77 to 96 m/s was unstable. The guard is retired with its defect.

Cause, measured: the reflux. The coarser class moved mass and heat through an
interface edge with its own estimate of the momentum there (the child's
accepted value one acoustic sub-step ago plus the class's own pressure-gradient
increment), and the reflux replaced that integral after the step: an impulse
in density and theta in every interface column once per class step. At rate 16
the sub-step is 48 s. With the reflux switched off (a diagnostic arm that does
not conserve) the same class map is stable; the zone depth made no difference
(eight rings gave the wind series of four).

Fix: the matched interface flux of the dynamics section (default). The coarser
class's continuity and heat equations read the child's accepted mean momentum
over each sub-step, so its register equals the child's and the reflux repairs
only rounding (largest relative density correction 3.2e-8 in six hours).

| Six hours, dry, 48-3 km patch mesh | Largest wind over the run, m/s (fixed step: 111) | Dry mass drift |
|---|---:|---:|
| Rates from the mesh, top 16, earlier coupling (`--interface-flux own`) | 193 (152 at 6 h) | 7.4e-12 |
| Same, reflux off (diagnostic) | 108 | 3.6e-8 |
| Same, matched interface flux | 108 | 1.2e-11 |
| 1,800 km ring plus two rate-8 islands under the southern jet, earlier coupling | 216 (177 at 6 h) | |
| Same islands, matched interface flux | 108 | 6.7e-13 |

The ladder reaches rate 32 in active mode. On a 48 km mesh with a 0.75 km
patch (327,797 cells, base step 4.5 s, 259,613 cells at rate 32) six dry hours
at top rate 32 end at 109 m/s with a dry mass drift of 2.9e-11.
`tools/run_fullstep_lts_dry.py --max-rate` defaults to 32. The forecast door's
top rate stays at 8 for a different, full-physics reason (below).

Not covered by the transport stage: physics (next section), more than one
device, restart, re-classing during a run.

```
python tools/run_fullstep_lts_dry.py --arm lts --transport --clamp-audit --grid GRID --static STATIC \
  --init INIT --out NEW_DIRECTORY --cache CACHE --dt 18 --hours 6
python -m pytest tests/test_fullstep_lts_scalars.py
HEX_LTS_GPU_TESTS=1 python -m pytest tests/test_fullstep_lts_scalars.py
```

#### A uniform-mesh test of the interface under a jet

`--forced-cap=-90,0,4780 --max-rate 2 --dt 90` on the uniform 30 km mesh puts
one 2:1 interface (class steps 90 s and 180 s, the step lengths of a rate-16/8
pair at a base step of 11.25 s) on a ring at 47 S, under the southern jet.
Dry, GFS 2026-10-03 12Z, twelve hours:

| Arm | Largest wind over the run, m/s | At 12 h | Dry mass drift |
|---|---:|---:|---:|
| Earlier coupling (`--interface-flux own`) | 426 at 7.5 h, then stopped by the validation flag | | |
| Matched interface flux | 110.3 | 107.8 | 1.8e-11 |
| Same cap around the north pole (ring at 40 N), matched | 109.7 | 107.2 | 1.4e-11 |
| One fixed 90 s step | 110.3 | 107.6 | 8.0e-13 |
| One fixed 180 s step | 109.7 | 107.2 | 2.4e-11 |

With the matched flux the largest wind of either class is never at an
interface cell.

Use class steps near the natural step of the cells when building such a test.
A cap whose inner classes step at 11.25 s to 45 s on 30 km cells is not an
interface test: the native fixed step itself grows at the model top there
(one fixed 11.25 s step on the 30 km mesh: 110 m/s to 146 m/s in four hours,
largest vertical velocity 1.6 to 3.6 m/s), and the cap runs grow with it at
the top two levels whether the cap sits under a jet or over the calm equatorial
Pacific (201 and 166 m/s after 2.5 to 3 h; the earlier coupling 245 m/s). The
same three-class cap over the calm Pacific at 45/90/180 s is quiet for six
hours (largest wind 109.7 m/s, the initial value).

#### Twenty-four hours, and what the coarsest class's step does at the surface

| 24 h, GFS 2026-10-03 12Z | Largest wind over the run, m/s | At 24 h | Dry mass drift | Unit tracer drift |
|---|---:|---:|---:|---:|
| 48-3 km patch mesh, top rate 16, transport | 131.2 | 114.8 | 4.3e-11 | 6.5e-7 |
| 48-3 km patch mesh, top rate 8, transport | 115.7 | 114.0 | 6.2e-11 | 7.6e-7 |
| 48-3 km patch mesh, fixed 18 s step, dry | 118.1 | 113.5 | 9.3e-11 | |
| 48-0.75 km patch mesh, top rate 16, transport | 118.1 | 114.6 | 1.7e-11 | 9.0e-7 |
| 48-0.75 km patch mesh, top rate 32, transport | 115.7 | 114.2 | 1.8e-11 | 7.5e-7 |

The 131 m/s of the first row is not at an interface when it appears and it
does not grow. From hour 21.8 a wind at the LOWEST model level near 63 S 128 E
(dry dynamics: no surface drag) leads the rate-16 class for two hours and
flickers between 113 and 130 m/s from one 288 s step to the next, at an
advective Courant number of 0.72 to 0.81 for the whole step; the class's
largest wind elsewhere stays at 112 to 115 m/s. No arm that steps the same
48 km cells at 144 s or less (top rate 8; the fixed step; the 0.75 km mesh at
rates 16 and 32, whose coarsest steps are 72 s and 144 s) exceeds 118.1 m/s
anywhere in 24 hours. Lowering the advective target from 0.80 to 0.65 gave the identical
class map and run: the classes are admitted from the initial state, and this
wind was not there at the start. So a class at the ladder's top step runs its
cells at the native step limit, and winds that grow during a run have no
margin; the rate-8 cap had been hiding that as well. Classes are static;
re-admission during a run is not built.

#### Full physics at top rate 16 stops: the door keeps 8

48-3 km patch mesh, GFS 2026-10-03 12Z, base step 18.75 s, through the
forecast door with `--fullstep-lts-mode active`:

| Top rate, code | Result |
|---|---|
| 8, matched flux | 24 h passed (4,608 steps, largest vertical velocity 9.7 m/s) |
| 16, matched flux | 6 h passed; a 24 h run is stopped by the validation flag (non-finite) at step 1984, 10.3 h; the same step on two boxes |
| 16, matched momentum flux without the matched heat flux | stopped at step 1168, 6.1 h |
| 16, earlier coupling (c06273b) | stopped at step 2112, 11.0 h |

Every top-rate-16 run with physics stops, with either coupling, inside a class
step with physics and without a rising vertical velocity before it (4 to 7 m/s
in the last two hours). The dry dynamics with transport ran the same mesh and
date 24 h at top rate 16. So this is not the interface instability above. The
cause is not found; nothing with physics had run at rate 16 before, and the
rate-16 class steps its column physics at 300 s. The door's top rate therefore
defaults to 8 and the guard in `drivers/fullstep_lts_forecast_v841.py` names
this breakage; 16 and 32 remain selectable.

## Column physics per class (stage "physics and clocks", one device)

`woof hex forecast ... --fullstep-local-timestep --fullstep-lts-mode active`
runs a full-physics forecast with every rate class on its own whole step. The
executor is `woof.hex.drivers.fullstep_lts_forecast_v841`; the forecast driver
hands it the request when the mode is active and the top rate is above one.

* **Classes.** `woof.hex.fullstep_lts_admission.admit_mesh_rates` gives every
  cell a dyadic rate from the mesh spacing, capped by the initial advective,
  acoustic and donor bounds at the class step, buffered and graded. It is the
  rule the dry and transport stages measured; the instrument
  `tools/run_fullstep_lts_dry.py` now calls the same function. The door's top
  rate defaults to 8 (a full-physics run at top rate 16 stops non-finite after 6 to 11 h, see above; 16 and 32 are selectable). A
  mesh that classes entirely at rate one is handed back to the unchanged
  native executor before any class object exists, so on a uniform mesh "on"
  is "off".
* **Physics on the class's own step.** Each class owns one resident column
  backend, built by the same-host executor's slicer and stack builder for the
  class's local columns, and sealed at the class step `rate * dt`. Surface,
  boundary layer, drag and cumulus run once per class step with that duration
  and their tendencies are held over it; the microphysics runs once at the
  class endpoint with the same duration. This is DESIGN.md section 5. A
  cadence welded to the step (surface/PBL and cumulus in the shipped
  configuration) is welded to the class step; an independent interval
  (radiation, an explicit PBL cadence) is kept.
* **Radiation.** Each backend counts its own steps. The executor first
  requires every global event (the radiation interval, every history time and
  the run length) to be a whole number of the coarsest class step, so every
  class calls radiation at the same model times. DESIGN.md section 4.3 asked
  for event tails (a shortened macro before an event) instead; a sealed
  backend refuses any step but its own, so the tail is replaced by a refusal
  that names a base step that works. With the shipped 1800 s radiation and
  top rate 8, an 18 s base step refuses (144 s does not divide 1800 s) and
  18.75 s runs (150 s does).
* **Physics halo round.** Before the microphysics reads every local column,
  the halo of the unpublished endpoint is filled from the owners' accepted end
  states (round `P` of `cuda_fullstep_lts_active`), then the species are
  clamped. The proof module clamps first; here a coarser owner's frozen state
  can still hold the initial file's negative vapour, which the column
  preparation refuses.
* **History.** Frames are written only at macro boundaries, where every class
  stands at the same instant; an interface register that is not empty at a
  frame is a refusal.
* **Restart identity.** At every frame `fullstep-lts-restart-identity.jsonl`
  records the SHA-256 of the owned-union rho, rho_theta, rho_u, rho_w and
  scalars, every class's clock, and each class backend's phase, step index
  and restart-identity digest. A resume reproduces these lines (see
  "Restart, DA cycling and limited-area meshes" below).

Scratch-zone columns run the finer class's physics on that backend's own
land-surface state. Their atmosphere is reset from the owner at the owner's
step boundaries; their land state is never published and is not reset.

The step-length constant of the drag scheme
(`cuda_gwdo_v841.X4_DT_SECONDS_F32`) is bound once per run. Classes step one
after another on one stream, so it is rebound to the stepping class's length
for exactly the duration of that class step and restored afterwards.

Refusals that remain, each naming its breakage: a limited-area mesh, restart
and DA cycling, a parked physics tier, the acoustic local-time-step
attachment, an event off the coarsest class step, and a physics backend that
is not a resident column seam. (More than one card is carried since the
multi-card stage, below.)
Refusals that remain, each naming its breakage: more than one device, a
parked physics tier, the acoustic local-time-step attachment, an event off the
coarsest class step (a boundary-file time included), and a physics backend
that is not a resident column seam.

## Restart, DA cycling and limited-area meshes

**Checkpoint and resume** (`woof.hex.drivers.fullstep_lts_restart_v841`).
`--checkpoint-hours H` writes `checkpoint.<label>.pkl` at a history frame, which
is always a macro boundary: every class stands at the same instant and every
interface register is empty. The class checkpoint
(`gpuwm-hex.fullstep-lts-checkpoint/v1`) holds, per class, the driver's state
and saved diagnostics on every local column, the backend's `restart_state()`
and the GF forcing carrier; the accepted-state mailboxes and the scalar
mailbox, reserves and hold array; and the class clock. Two of these are not
reconstructible from the owned state, which is why a single-stack checkpoint
could not carry the run:

* a scratch column's atmosphere and land state feed the class's owned cells
  inside the next class step;
* the trajectory planes hold a finer class's committed endpoint from before
  its own reflux, and the next macro's coarser steps start from them.

`--restart-from` validates the schema, base step, mesh size, rate map, class
decomposition digest, macro boundary and valid time, and refuses by name what
it cannot resume (a single-stack checkpoint, another rate map: "every class
would resume on the wrong columns"). `--start-time/--hours` name the original
run, as on the native route.

**DA-cycling restart.** `--restart-from CKPT --da-analysis FILE` recouples the
analysis with the native `jedi_contract.recouple_da_cycling`, places it on
every local column of every class (owned, scratch and halo) and builds the
interface from that state exactly as a cold start does; every class's physics
backend resumes from the checkpoint. `--jedi-background` writes
`mpasout.<label>.nc` with the native writer on global views assembled from
each class's owned columns.

**The door.** `woof hex forecast` forwards `--checkpoint-hours`,
`--restart-from`, `--da-analysis` and `--jedi-background` to the driver (fixed
step and LTS alike); they used to be reachable only past the door's mesh bind.

**Limited-area meshes.** A regional mesh's outer edges and vertices name an
absent cell (-1). The class planner, grading, buffering and the admission
carry one "absent" slot for it; before, the sentinel indexed the last cell of
the mesh (grading coupled the first cell to the last) and the planner refused
open meshes outright. Each class driver is the native regional driver on the
class's local sub-mesh: it keeps native's garbage column (the interface
kernels index by the device row stride), reads the boundary files through its
own layout, and advances its own boundary clock by its own class step.

* **The zone's class.** The specified and relaxation zones (`bdyMaskCell >= 1`)
  plus two interior rings are held in ONE class at rate 1
  (`fullstep_lts_admission.BOUNDARY_RATE`). The relaxation coefficients are
  per-step quantities -- the Rayleigh term `(mask-1)/5/(50 dt)` and the
  Laplacian filter's `10 dt` clock -- so a zone at rate r would relax toward
  the boundary data r times more weakly per unit time than the fixed-step run;
  at rate 1 the zone is forced exactly as the fixed run forces it. The two
  rings keep every relaxation stencil inside the class.
  `WOOF_HEX_LTS_BOUNDARY_RATE` is an experiment knob for the A/B.
* **The matched interface flux on a regional driver.** A regional class
  driver launches `acoustic_ru_regional_v841` (the specified-zone pgrad mask)
  through its own regional kernel table, not `acoustic_ru_v841`, so the
  coarse-side tap goes on the kernel the driver launches
  (`cuda_fullstep_lts_scalars.tap_acoustic_momentum`). Before 0.4.5 the tap
  sat only on the global kernel, never fired on a region, and the first class
  step of any region with an interior interface stopped with "class 0 imposed
  the child's interface flux 0 times". Measured 2026-10-24 (one RTX
  PRO 6000): a 1400 km cap cut from a 120-15 km globe (7,173 cells; 4,082 at
  rate 1, 3,091 at rate 2), GFS 2024-05-06 18Z, ArWen physics, 1 h. Stock
  0.4.4 stopped at the first class step; fixed, it runs, the interior budget
  closes to 8.3e-11 (the parent globe's own LTS run: dry mass drift 2.4e-10),
  and against the uniform 90 s step at +1 h: surface pressure RMS 3.5 Pa,
  10 m wind RMS 0.07 m/s, lowest-level theta RMS 0.04 K, all of it in the
  rate-2 ring, none at the class interface or in the zone. LTS off: the
  history files are byte-identical before and after (strict numerics).
* **Timing.** Every boundary-file time inside the run is a class-clock event:
  a class step that straddled one would integrate part of a step on the
  expired interval's tendency, and classes would disagree about when the
  interval changed.
* **Mass budget.** The dry mass of the interior (`bdyMaskCell == 0`) changes
  only by flux through the interior/zone edges. Each such edge's owner adds its
  accepted split flux to an FP64 register; the receipt's `interior_budget`
  reports the interior mass, the integrated outflow and the closure of their
  sum.

**Measured (2026-10-12, tree bbefb27, GFS 2026-10-03 12Z, one card each).**

* **Resume identity, limited-area.** `lam6000` is a 6000 km cap cut from the
  48-3 km patch mesh (97,504 cells; classes 8/4/2/1 with 54,095 / 8,106 /
  8,172 / 27,131 owned cells; GFS boundary files at 12, 15 and 18Z). A 6 h
  active-LTS run wrote its class checkpoint at 3 h (4.38 GB); the run resumed
  from it wrote the 16, 17 and 18Z frames with the same file SHA-256 as the
  uninterrupted run (45 of 45 variables byte-equal, 39 of 39 published field
  hashes, state hash, class ticks and class physics identity equal).
* **Resume identity, whole mesh.** The 48-3 km patch mesh (294,799 cells),
  1 h with the checkpoint at 30 min (6.64 GB): the resumed 13Z frame has the
  uninterrupted run's SHA-256 (46 of 46 variables, 39 of 39 field hashes).
* **JEDI background and DA restart.** A 3 h `--jedi-background` run on
  `lam6000` wrote `mpasout` at every frame and its four history frames are
  byte-equal to the plain run's, so writing the background does not touch the
  trajectory. Resuming the 15Z checkpoint with `--da-analysis` set to that
  run's own 15Z `mpasout` (a zero increment) ran to 18Z; against the
  uninterrupted run at 18Z: theta RMS 0.012 K, zonal wind RMS 0.017 m/s,
  surface pressure RMS 0.64 Pa. It is not bit-identical and is not meant to
  be: the analysis file carries rho, theta, scalars, u and w, and the coupled
  state and the interface are rebuilt from it as at a cold start.
* **Limited-area LTS against the fixed-step regional run** (`lam6000`, 6 h,
  dt 18.75 s, both through `woof hex forecast --lbc-dir`). Both passed the
  physical gates. At 18Z, LTS minus fixed: theta RMS 0.40 K, zonal wind RMS
  0.34 m/s, surface pressure RMS 11.5 Pa, 2 m temperature RMS 0.27 K,
  accumulated rain RMS 0.41 mm. By zone (theta RMS): specified zone 0,
  relaxation zone 0.10 K, the 3 km core 0.09 K, 250-1000 km 0.31 K, beyond
  1000 km 0.50 K; the difference grows with the class step, which is where the
  physics runs on a longer step (150 s against 18.75 s). The largest point
  differences (13 K, 18 m/s) are in convective cells over the western Gulf in
  the 48 km region.
* **Mass.** Whole-domain dry mass changed by +4.890e-4 (LTS) and +4.828e-4
  (fixed) over the 6 h. The LTS interior budget closes: interior dry mass
  +5.9375e-4, integrated accepted outflow through the 1,493 interior/zone
  edges -5.9375e-4, largest closure residual 1.5e-10 of the interior mass.
* **Speed.** Warm, LTS took 70.7 s per forecast hour and the fixed-step run
  69.6 s (0.98x). The plan's local cell work per base step is 58,751 against
  97,504 fixed (1.66x), and the zone at rate 1 is part of that work; at this
  size the per-class-step cost that does not scale with cells takes the rest
  (15 class steps per macro in place of 8 base steps).

```
woof hex forecast --mesh MESH --grid GRID --static STATIC --init INIT \
  --start-time 2026-10-03_12:00:00 --hours 24 --history-every-minutes 180 --dt 18.75 \
  --out OUT --scratch SCRATCH --fullstep-local-timestep --fullstep-lts-mode active
python -m pytest tests/test_fullstep_lts_physics_clock.py tests/test_fullstep_lts_restart.py

# checkpoint at 1 h, then resume (bit-identical frames), or resume from an analysis
woof hex forecast ... --hours 2 --fullstep-local-timestep --fullstep-lts-mode active --checkpoint-hours 1
woof hex forecast ... --hours 2 --fullstep-local-timestep --fullstep-lts-mode active \
  --restart-from OUT/checkpoint.2026-10-03_13.00.00.pkl [--da-analysis ANALYSIS.nc]
# limited-area: a culled mesh and its boundary series
woof hex forecast --mesh CULL --grid ... --lbc-dir LBC --dt 18.75 --fullstep-local-timestep --fullstep-lts-mode active
```

## Several cards (stage "multi-card")

`woof hex forecast ... --samehost-devices 0,1,... --samehost-layouts L
--samehost-part P --fullstep-local-timestep --fullstep-lts-mode active` runs
the class executor across the cards of one host. A mesh that classes entirely
at rate one runs the unchanged same-host ranks instead. The dry and transport
instrument takes `--devices 0,1,...` (or `all`); its fixed arm is then one
rate-1 class cut the same way, so both arms share the partition method and the
exchange code.

* **Blocks.** `woof.hex.fullstep_lts_blocks` cuts each class's computed set
  (its owned cells and its scratch zone) into blocks, one per card the class
  uses: one block per `--block-cells` owned cells (default 16,000; it was
  50,000, which left the finer classes whole on one card while the others
  idled), at most one per card. Owned cells are cut by recursive coordinate bisection; a zone
  cell goes to the block of its nearest owned cell. A block holds the
  three-ring halo of the partition law. An entity another block of the same
  class computes is a peer entity; everything else in the halo belongs to
  another class.
* **Edge ownership.** Inside a class an edge belongs to the block of its
  lower-numbered cell, except an edge to the next coarser class, which belongs
  to the block that carries the edge's coarse cell. All interface edges of a
  coarse column are therefore in one block, which holds the column's donor
  reserve.
* **Peer exchange.** The blocks of a class step at the same time, one thread
  per card. At every native round site (A, B, C, D, E, P) a block first runs
  the one-card mailbox fill, then the same-host byte transport refreshes its
  peer halo. Three rounds are new: two `H` rounds inside the flux-corrected
  scalar stage carry the donor-guard factor of every coarse interface column
  and the reserve-cap factors of every finer one, so every copy of an edge
  flux is scaled by one number (the one-card kernels are split into a factor
  kernel and an apply kernel with the same expressions); one `R` round at the
  class's sync carries the refluxed cell state before velocities next to it
  are recovered.
* **Between classes.** Every card holds a full mailbox and a full register.
  Published planes are copied, through pinned host memory, to exactly the
  cards whose blocks read those cells, before the next class step. Before a
  class steps, the register slots its blocks read are copied from the cards
  whose finer blocks wrote them, and the writer's copy is cleared.
* **Physics.** Each block owns one column backend sealed at its class step.
  The drag scheme's step constant is bound around each class event (every
  block of the class shares the length).

### Measured (2026-10-11, public 60-3 km mesh, dynamics and transport of six species, 1 h)

Warm seconds of integration per forecast hour, base step 18 s, rates 8/4/2/1,
RTX PRO 6000 cards (one card and two cards on one host, five cards on another). The
fixed arm on several cards is one rate-1 class cut the same way:

| Cards | Fixed | LTS | Fixed / LTS |
|---:|---:|---:|---:|
| 1 | 279.46 | 195.54 | 1.43 |
| 2 | 143.74 | 118.79 | 1.21 |
| 5 | 58.66 | 88.02 | 0.67 |

The LTS state after the hour is byte-identical on one, two and five cards
(rho and the six scalars hash the same), as is the fixed state on one and two
cards of the 48-3 km patch mesh. The speedup shrinks with the card count
because every class step pays a host launch floor of roughly 70 to 100 ms
beyond its GPU time (eager launches; the subcycle graphs refuse a halo
exchanger), and LTS has 375 class steps per hour against the fixed arm's 200
steps: at five cards a rate-1 block of 89,000 local columns costs 0.25 s per
step and a rate-4 class of 76,000 columns, not cut at all, 0.22 s.

Conservation on five cards, 6 h, same arms (LTS, `--clamp-audit`): dry mass
relative drift 1.97e-11; tracer mass change less the clamped mass qv
3.7e-11, unit 4.5e-7, band 8.1e-7, ramp 1.5e-10; withheld share of interface
throughput qv 8.2e-8, band 6.7e-6; registers empty at the end. These are the
one-card stage-2 numbers, as the byte identity requires.

Full physics through the door (48-3 km patch mesh, 1 h, 18.75 s). The
owned-union state hashes of the restart record at every frame:

| Run | Frames | State at 1 h |
|---|---|---|
| one card | hourly | A |
| two cards, every class whole | hourly | A |
| two cards, rate-8 class cut | hourly | A |
| two cards, finest class cut (three runs, one with round `T`) | hourly | B |
| one card | every 5 min | B (identical to the cut run at all 12 frames) |
| two cards, finest class cut | every 5 min | B |

So every run agrees except those whose finest class is one block and that
write only hourly frames (one card or two): with the finest class whole, the
forecast depends on how often history is captured; with it cut, it does not,
and the cut runs equal the one-card run with 5-minute frames. The cause is open (a host-device ordering that the extra synchronisation
of a capture or of a peer exchange hides is the leading suspect; the test is
the one-card hourly run under `CUDA_LAUNCH_BLOCKING=1`). Exchanging the
phase-one column rates on the peer halo (`WOOF_HEX_FULLSTEP_LTS_RAW_EXCHANGE=1`,
round `T`) changes nothing: the halo physics is exact. The native same-host
fixed step on one and two cards with hourly frames is byte-identical.

Full physics, public 60-3 km mesh, four RTX PRO 6000 cards, 18.75 s:
fixed step (same-host ranks, no per-step fingerprints) 108.7 s per forecast
hour warm, peak 34.8 to 35.3 GB per card; LTS 128.4 s per forecast hour warm
over 3 h, peak 29.6 to 33.1 GB per card, dry-mass drift 8.0e-11, registers
empty, radiation 6 times per block at the same model times.

Default: off. On this mesh LTS is slower than the fixed step at four cards
with physics (0.85) and at five cards without (0.67), the 48 h scored
forecast has not been run, and the one-card physics run is not yet
independent of the history cadence.

## On the MPAS suite route (`--physics-suite`)

Full-step LTS runs on the MPAS-physics route as on the WOOF-ArWen route (`woof.hex.mpasphys.lts_v842`):

* `--fullstep-local-timestep --fullstep-lts-mode active` steps every rate class at its own whole step with the
  MPAS suite as each class's physics. The coordinator is the same `FullStepLtsActive` (mailboxes, matched
  interface flux, reflux). Each class runs the ported route on its own local mesh, and the physics runs on every
  local column. Class 0 writes the global history and diag frames from every class's owned words.
* `--fullstep-local-timestep` alone is the shadow: the same observer as on the WOOF-ArWen route, on the fixed step.

Measured on a 54,422-cell 3-to-192 km variable mesh (one card, 1 h, GFS 2026-10-24 00Z, four classes at
rates 1/2/4/8): the MPAS route's LTS-minus-fixed difference matches the WOOF-ArWen route's band by band (surface
pressure 21.2 Pa RMS on cells of 40 km and coarser on both, theta 0.198 K RMS in the 20-40 km band on both),
and the maps show no seam at a class boundary. On a mesh this small LTS does not pay on either route: each class
step costs a fixed host time. The MPAS route takes about 1.9 times the WOOF-ArWen route's time with LTS on and with
it off.

The door chooses between the fixed step and LTS at each top class, on both physics routes
(`woof.hex.fullstep_lts_choice`). It counts the mesh's classes from geometry alone (the admission's ladder, the
convective hold, the buffer and the grading) for every top class from the mesh's own down to 2, gives each the step
its class clock allows (a default step moves down only as far as that class needs; an explicit step is priced only
where it already fits), and predicts every plan and the fixed step from the per-step cost measured on the run's
route, one row per route in `woof/hex/data/fullstep-lts-step-cost.v1.json`. It runs the cheapest. A lower top class
sets `--fullstep-lts-max-rate` and its step; where the fixed step wins, the run is the plain fixed run (the fixed
step's own timestep, no LTS argument left). `fullstep-lts-choice.json` beside the frames records every prediction.
One card, closed meshes and runs without a moving region only; `WOOF_HEX_FULLSTEP_LTS_FORCE=1` runs LTS at the top
class anyway.

Measured on the 96 -> 3 km zoom globe (149,322 cells, GFS 2025-05-19 18Z, 3 h, one RTX 5090 per arm, hex 0.4.9 fold
tree, seconds per forecast hour, warm hours 2-3; each LTS arm ran at the same time as its fixed arm on a second card):

| Route | Fixed, 18 s | LTS top class 8, 15 s | LTS top class 4, 18 s | Row: per class step + per cell-step |
|---|---:|---:|---:|---|
| MPAS suite | 77.2 / 75.2 | 75.4 | 64.0 (1.18x) | 0.086 s + 2.0 us |
| WOOF-ArWen | 80.2 | 83.0 | 73.2 (1.10x) | 0.106 s + 2.0 us |

Why the top class matters on that mesh: radiation every 1800 s, and 8 x 18 s = 144 s does not divide it, so top
class 8 needs a 15 s step (20 % more base steps than the fixed step) and 15 class steps per 120 s; top class 4 keeps
18 s (4 x 18 s = 72 s) and takes 7 class steps per 72 s. The zoom globe keeps 43 % of its cells at the finest rate
(the 3 km core and every cell finer than 8 km), so its cell work per base step is at most 1.95 times less than the
fixed step's, and every class step also pays a fixed part (the step loop's host work and its smallest kernels).
Before hex 0.4.9 every LTS class on that mesh also ran the slower element stencil kernels (the hoisted kernels
started at 400,000 edges; the fixed step's 447,960 edges were above it, no LTS class was), and the MPAS route paid
about twice today's per-step host time: LTS then ran 0.81 times the fixed step's speed on both routes. A mesh whose
cells are mostly coarse (the 48-3 km patch mesh, ceiling 5.0, 2.0x on one card in October) keeps a larger win.

The class clock moves a default step so that the coarsest class's step lands on every event. That class is the
mesh's own top class (`woof.hex.fullstep_lts_census`), at most `--fullstep-lts-max-rate` (default 8, the executors' default), so the 96 -> 3 km
zoom globe steps 15 s (8 x 15 s = 120 s), not 12.5 s.

Refused by name on this route: `--local-timestep` (the acoustic ladder needs RK3 with six sub-steps and no FCT
limiter), several cards, restart, DA cycling, IAU, stochastic forcing, chemistry, the balance series, nest
triggers, timed surface updating, limited-area meshes and the forked RRFS dycore. A moving segment keeps the
fixed step on this route (its move hooks are single-mesh).
