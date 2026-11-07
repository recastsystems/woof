# Adaptive time stepping: what it cost, what it bought

Measured 2026-09-02 against gpuwm 2.6.5 on an RTX 5070 Ti, on a
two-domain tree (d01 311x229 at 10 km, d02 300x300 at 2 km, nz 61, 5:1
ratio, d02 relocating), 30-minute forecast, 9 wrfout frames.

This implements WRF's `use_adaptive_time_step` with its real namelist
surface, a `dt` that survives restarts and moving nests, and a bitwise
gate at every step.

---

## The answer

Three interleaved replicates each, medians (wall clock on the
measurement host drifts +/-20% run to run and a single run cannot
resolve the difference):

| | wall s | frames | `w_damp` |
|---|---|---|---|
| fixed dt = 30 | 78.89 | 9/9 | idle |
| **adaptive, `target_cfl` 1.0** | **53.80** | 9/9 | idle |

**1.466x, 31.8% less wall clock, SE ~0.1% on both arms.** This is a
cost result only. The forecast change from the larger step has not
been measured against a small-step reference run (solution
verification) or observations (validation); see What is open. `dt` climbs
30 -> ~55 s, the CFL holds at target, every frame lands on its exact time
and the run ends on the stop boundary.

Three bitwise properties, all gated by `cmp` on every wrfout file:

- **Feature off is byte-identical** to the tree before the work, at every
  step.  The refusal-to-run was deleted only after this held.
- **`calc_dt` is exact.**  1728/1728 rows against a Fortran oracle
  captured from `dyn_em/adapt_timestep_em.F`, `max_ulp == 0`.
- **The restart carry is bitwise.**  A checkpoint written mid-adaptation
  (900 s, while `dt` was between values) resumes **4/4 frames
  byte-identical**, on a tree whose nest relocates.

VRAM went **down**, 11.259 -> 11.171 GiB.

---

## The configuration surface

Twelve new `RunConfig` fields, all carrying WRF's own namelist names and
defaults, and honouring WRF's Registry scoping
(`Registry.EM_COMMON:2269-2281`):

| key | WRF scope | default | per-domain in woof |
|---|---|---|---|
| `use_adaptive_time_step` | 1 | `false` | no |
| `step_to_output_time` | 1 | **`true`** | no |
| `adaptation_domain` | 1 | 1 | no |
| `target_cfl` | `max_domains` | 1.2 | yes |
| `target_hcfl` | `max_domains` | 0.84 | yes |
| `max_step_increase_pct` | `max_domains` | 5 | yes |
| `starting_time_step` | `max_domains` | −1 | yes |
| `starting_time_step_den` | `max_domains` | 0 | yes |
| `max_time_step` | `max_domains` | −1 | yes |
| `max_time_step_den` | `max_domains` | 0 | yes |
| `min_time_step` | `max_domains` | −1 | yes |
| `min_time_step_den` | `max_domains` | 0 | yes |

The three scope-1 keys are deliberately **absent** from
`_DOMAIN_RUN_OVERRIDES`: upstream has one scalar per run for those, and
a per-domain override would be inventing surface WRF does not have.  The
other nine are per-domain.

With `use_adaptive_time_step = true`, history intervals need only be
positive whole seconds. They do not have to divide the declared
`time_step`. The root shortens its live step to each domain's history
alarm, including `history_begin`, and divides that step down to its
children. Positive radiation, cumulus, and surface/PBL periods likewise
use exact tick deadlines at their requested intervals. Zero physics
periods still mean every step. Fixed-clock cadence validation and its
WRF step-count predicates are unchanged.

One more field sits beside them and is not a WRF key:
`min_time_step_sound` (default 0, even, per domain).  The clock derives
each domain's acoustic substep count from its live step, as upstream's
`time_step_sound = 0` asks, and that count is 4 at every short step (under
about 3.3 s at 1 km).  `min_time_step_sound` is a floor under it; 0 keeps
upstream's count.  The steep-terrain rules (`woof/acoustic_adaptation.py`,
`woof/terrain_clock.py`) set it on each adaptive domain whose count they
raise to 6, because without it the clock put the count back to 4 and a
1 km forecast under a strong crest-level jet stopped.  It is in
`ADAPTIVE_TIMESTEP_RUN_FIELDS`, so a fixed-clock identity never sees it and
a resume may change it like any other clamp.

`−1` on the three step limits means *unset*, which is WRF's own
encoding: it substitutes a grid-spacing-derived value in
`start_em.F:939-953`:

| key | value WRF substitutes for −1 |
|---|---|
| `starting_time_step` | `NINT(4 * MIN(dx,dy) / 1000)` |
| `max_time_step` | `NINT(8 * MIN(dx,dy) / 1000)` |
| `min_time_step` | `NINT(3 * MIN(dx,dy) / 1000)` |

### What the `_den` companions are

Each of those three is **two namelist keys forming one value**.  WRF does
not store these limits as floats: it stores an exact rational and hands
it to ESMF as a numerator and denominator, `WRFU_TimeIntervalSet(Sn=...,
Sd=...)`:

| `starting_time_step` | `_den` | resulting step |
|---|---|---|
| 30 | 0 | 30 s exactly (`Sd=1` is substituted) |
| 30 | 1 | 30 s exactly |
| 100 | 3 | 33.333... s, held exactly |

`den = 0` is a **sentinel meaning "the numerator is whole seconds"**, not
a zero denominator: `adapt_timestep_em.F:132-136` branches on it and
passes `Sd=1`, and `:186-214` does the same for the two limits.  Reading
it as arithmetic would divide by zero.

The point of the rational is drift: a step of 100/3 s stays exact over
thousands of steps, where a float accumulates error and the frame times
walk off their alarms.  WOOF's clock is an integer tick lattice for the
same reason, so the pair maps onto it directly: `_adaptive_interval`
returns a `Fraction`, and `tick_den` is the LCM of every domain's
rational-dt denominator.  Turning the adaptive clock on folds a further
100 into that LCM, matching the `precision = 100` upstream's `calc_dt`
uses, so every adapted step lands on a whole hundredth of a second.

Every field is registered in **three** places in the same commit that
adds it:

- `ingest.prepared_cache.DEFAULT_TOLERANT_IDENTITY_FIELDS`, so a tree
  prepared before the field existed still loads.  A field that joins
  `RunConfig` without joining that table refuses every previously
  prepared tree and blames the experiment file: that has happened
  **four times** in this package's history.
- `experiment._DOMAIN_RUN_OVERRIDES` for the nine per-domain keys.
- `namelist_import`, so a real WRF namelist round-trips.  The refusal
  that used to reject `use_adaptive_time_step` on import is gone, and the
  nine per-domain keys are read as columns: a domain that differs from
  the root gets its own `[[domain]]` row.  The native HRRR route writes
  all twelve into both of its namelists from the configuration.

**From `woof domain`.**  `--clock adaptive` writes
`use_adaptive_time_step = true` into `[shared]` and nothing else: every
bound stays at −1, so each domain gets WRF's fill-ins for its own
spacing (the shared-clamp trap below cannot arise), and the first step
is the `time_step` the wizard derived.  The terrain clock and the
steep-ground substep rule still write their ceiling and substep floor at
launch.  `--clock auto`, the door's default, is adaptive only when every
domain's first step sits inside those fill-ins (so never on the tropical
2.5 s per km clock, which starts under the 3 s per km floor) and every
spacing lies within the terrain clock's measured 500 m to 12 km;
`--clock adaptive` outside the fill-ins is refused naming the domain.

**Tolerance covers absence, not divergence.**  A cache prepared *before*
these fields existed loads fine.  A cache prepared *with* adaptive on is
still refused against a config with it off, because both values are
present and differ, which is correct, and is why each arm of the
measurement below needed its own prepared tree.

---

## Observability: `dt` in the step log, only when it varies

A varying step that nothing records is a run you cannot debug.  Before
this, `progress.jsonl` carried `model_seconds` but not `dt`, and the only
way to recover the step series was to difference consecutive
`model_seconds`, which `StepLog.domain_step` already documents as wrong
for a delayed-start nest, whose clock begins offset.

So the `step` record now carries `dt`, **but only under an adaptive
clock**.  Under a fixed step it would print the same constant on every
line for the life of the run, and the human-facing line is read by
humans.

That conditionality is what keeps it compatible.  The log's schema string
is a published contract, and its rule is that a consumer meeting a field
it was never told about must *refuse loudly* rather than skip it.  So:

| run | schema | `dt` in record | measured |
|---|---|---|---|
| fixed | `gpuwm.step-log/v3`: unchanged | absent | 360 steps, 0 carrying `dt` |
| adaptive | `gpuwm.step-log/v4` | present | 207 steps, 207 carrying `dt`, 29.8–78.0 s over 34 distinct values |

A fixed run therefore emits exactly the stream it always did and every
existing consumer keeps working untouched; an adaptive run declares v4,
and a v3-only consumer refuses it, which is the contract behaving as
designed rather than a break.

The caller passes `dt` unconditionally and the *log* decides whether to
report it, so the schema and the field cannot disagree.  The value is the
step just taken, from the same integer-tick lattice as `model_seconds`,
not the step the clock is about to adopt: under an adaptive dt those
differ on most steps.

---

## Why a fixed dt cannot bank the same headroom

The headroom is real.  WRF's own CFL, reduced by `w_cfl_stat` over the
same index range `w_damp` uses:

| domain | dx | dt | max WRF vert_cfl | `w_damp` fired |
|---|---|---|---|---|
| d01 | 10 km | 30 s | **0.643** | 0 / 769,165,200 cells |
| d02 | 2 km | 6 s | **0.414** | 0 / 4,860,000,000 cells |

Against `target_cfl = 1.2` that is **1.87x of headroom on d01 and 2.9x on
d02**.  The damper never fired once across 5.6 billion cell-visits, so
"`w_damping` is buying the stability" is dead as well: it is idle.

Horizontal sits at ~0.20 against `target_hcfl = 0.84` (**4.2x**) and never
binds; vertical is the constraint on every sample.

But the CFL climbs **3.2x over a 4-hour run** as the storm organises, so
a *fixed* dt must be sized for the worst step:

| policy | dt | speed |
|---|---|---|
| fixed, never exceeds target | 36.2 s | 1.21x |
| adaptive | 49.0 s mean | 1.63x |

That gap is the entire justification for a controller.

---

## The instrument was measuring a different quantity

This is the finding that reversed the project, and the one most worth
carrying upstream.

`health.cu` reports a **geometric** Courant number: `max |w|/dz` over
*every* level.  WRF's `target_cfl` grades the eta-coordinate,
mass-weighted `|ww/(c1f*mut + c2f)*rdnw|` over `k = 2..kde-1`.  With a
26 m first mass level the geometric form is dominated by a layer WRF's
loop excludes, and the two differ by roughly 2x: **1.18 geometric against
0.643 WRF on the same run**.

Mining the reported number from 169 run-domain samples gave d01
**1.720 ± 0.027** and d02 **1.290 ± 0.022**: read against a 1.2 target
that says the model already runs 1.4x *over*, and a controller would
**cut** dt.  The projection was a 40% slowdown, and the project was
nearly abandoned on it.  Both figures are the wrong instrument; the true
values are the 0.643 and 0.414 above.

Sizing a timestep decision off the reported number gets the sign wrong,
and gets it wrong in the direction that kills the work.

`apply_w_damping` already computed the correct quantity per cell
(`kernels/openbc.cu:123`) and discarded it.  It is now reduced by
`w_cfl_stat`, which is what the controller reads.

---

## What the port required beyond `calc_dt`

Porting the arithmetic was the easy half.  WOOF's clock is an integer
tick lattice with a **precomputed periodic** schedule, and eleven places
assumed a uniform step.  Each was silent or nearly so; only the first
four were foreseeable from reading:

| # | assumption | how it failed |
|---|---|---|
| 1 | `step_ticks` frozen on `DomainTicks` |, (design) |
| 2 | the schedule is a precomputed periodic table |, (design) |
| 3 | `dt_fp32` frozen | boundaries forced on a different clock from the interior |
| 4 | cadences must divide `dt` | frames skipped |
| 5 | nest divide as an exact rational | 827 substeps on a nearly-prime tick count |
| 6 | `step_to_time` halving | half-tick |
| 7 | the run end | stepped past the finish after computing everything |
| 8 | radiation's `itimestep % stepra` | 373 s or 660 s against a 360 s target |
| 9 | force-coverage audit divides ticks | 50 expected against 36 correct |
| 10 | Davies-weight cache keyed on dt | unbounded slot growth, digest refused |
| 11 | root steps to its OWN frames only | **run PASSED having written 7 of 9 frames** |

**Number 11 is the one to remember: it did not fail, it under-delivered
and exited zero.**  A gate that only checks the exit status would have
shipped it.

One of these is a correctness bug in the *running* model, not only in the
adaptive path: `bldt_seconds` was computed once at driver construction
and never refreshed.  It is the timestep handed to MYJ, Noah, the surface
layer and the 10 m diagnostics.

### The one reader that had to stay on the CONFIGURED step

The sweep that produced the table above converted one reader too many.
`woof/core/inflow_perturbation.py`'s draw-refresh index is not a
per-step physical rate like rows 3, 8 or 10: it is a **quantizer** of
absolute model time into 100 s buckets, and `clock.ticks` is absolute.
Moving its divisor to the live pair left `floor(T / D(t))` with a
breathing `D`, which is neither monotone nor onto: on a parent whose dt
breathes 4–8 s over 3 h, 356 backward index transitions in 1,910 forces,
largest jump 8 indices, and holds running 4.0 s to 148.7 s against a
pinned 100.0 s.  A backward index re-emits a Philox draw the boundary
face has already imprinted.

The test for **any** reader is which quantity it is: a RATE integrates
and must be live; a BUCKET partitions model time and must be a run
constant, or the map stops being a function of time.  The bucket now
comes from `clock.spec` and the index refuses a decrease by name, so a
future re-conversion raises instead of silently repeating a pattern.

---

## An upstream bug, reproduced rather than tidied

`adapt_timestep_em.F:394-406` intends to reach back to `last_max_*_cfl`
after a step shortened to land on an output time, because such a step has
an artificially small CFL.  It works for the horizontal only.  The
`use_last2` branch (`:394-402`) preserves both with no-op
self-assignments, then `:406` assigns
`grid%last_max_vert_cfl = grid%max_vert_cfl` **unconditionally**, four
lines below the `endif` that just protected it.

So the vertical memory is clobbered every step and the protection never
applies to it.  With `step_to_output_time` defaulting true, that is every
adaptive run.

The port reproduces this.  A version that "fixed" the asymmetry would
diverge from WRF on every run: it was written the corrected way first,
and the vertical arm produced dt=3 where the protected reading predicts
30+.  Pinned as-is by `tests/test_adaptive_timestep_controller.py`.

---

## Measured cost

The controller needs one device-to-host readback per step, which is a
synchronisation point and the one thing that could have made an adaptive
dt slower than the steps it saves.

**+1.1% on the median against a 2.5% SE**: under the measurement
host's noise floor.

The nest coupling costs nothing extra: every domain runs its own
controller, and the nest's step is snapped to a *divisor* of its
parent's.  `parent_time_step_ratio` was already independent of
`parent_grid_ratio`; they are equal only by convention.

---

## The distribution behind the max

`w_cfl_stat` reported only a domain **maximum**: an order statistic that
one cell decides.  A controller reading it cannot separate "the flow got
faster everywhere" from "one column is having a moment", and upstream's
`+5%` per-step growth clamp (`:174`) exists to survive exactly that
ambiguity: a rate limiter standing in for a distribution nobody could
afford to measure.

A GPU can afford it.  The same reduction now folds a 32-bin histogram of
the same quantity: one shared `atomicAdd` per thread, 32 global per
*block*, and the **frame stays 0 B**, so the local-memory reservation law
still does not touch this kernel.  Shared 12 -> 140 B; device buffer
4.7 MB per domain.

The reported `median_max_over_p9999` is the number that settles whether a
quantile-based controller is worth building: near 1.0 and the max *is*
the distribution.

---

## Where this port diverges from upstream, deliberately

- **`starting_time_step = -1` keeps the CONFIGURED `time_step`.**
  Upstream substitutes `NINT(4*dx km)` (`start_em.F:939-941`).  A woof
  tree is PREPARED: its output alarms, nest step ratios and boundary
  interval are built around the configured step and bound into the
  prepared-cache identity, so a first period taken at some other step is
  the one period nothing was prepared for.  An explicit positive
  `starting_time_step` is honoured exactly as upstream honours it, and
  `max_time_step` / `min_time_step` keep WRF's `-1` fill-ins, where no
  lattice is involved.
- **`adaptation_domain` must be 1.**  Upstream's values above 1 select a
  child domain to drive the tree's step.  woof drives every domain from
  its own measured CFL and snaps the nest to a divisor of its parent's
  step, so the selection has nothing to act on; the value is refused by
  name rather than accepted and ignored.
- **Radiation and cumulus fire on elapsed MODEL TIME, not a step count.**
  Both of WRF's predicates are `itimestep % stepN`, and under a varying
  dt neither number counts what it says.  See "What the port required
  beyond `calc_dt`".

---

## Resuming a dead run: seven gates, one shape

The restart carry was bitwise for the state it carried, and an adaptive
run still could not resume from its own checkpoints.  Not the retuned
resume: any resume.  A three-domain relocating run wrote five
checkpoint sets and could use none of them.

Seven gates refused it in sequence, and **one shape produced six of
them**: a rule lives in two places, one that APPLIES it and one that
RESTATES it by hand, and the two drift.

| # | applies the rule | restates it | the drift |
|---|---|---|---|
| 1 | the clock derives `dt` AND `time_step_sound` | the identity walk exempts `dt` | compared against a value the driver overwrites every step |
| 2 | the walk permits a retune | the prepared cache binds the same fields | the cache refuses what the walk allows |
| 3 | the walk permits a retune | `_configuration_fingerprint` binds them | the fingerprint refuses what the walk allows |
| 4 | the reading build's exemptions | a hash the WRITING build computed | asks "was this hashed by this build", not "is this the same run" |
| 5 | the walk permits a retune | the experiment identity binds them | the identity refuses what the walk allows |
| 6 | live components drop the fields | stored components still carry them | one-sided normalisation: `1.2 vs 1.1` became `1.2 vs absent` |

Each fix replaces a restatement with a shared constant, or normalises
both sides instead of one.

**The reported defect (1).**  `AdaptiveClockDriver._apply` overwrites two
`RunConfig` fields every root step (`dt` and `time_step_sound`) while
the walk exempted only the first, so a checkpoint stored the DERIVED
sound-step count and the walk compared it against the config's declared
one:

    time_step_sound: restart=6 run=4

`wrf_num_sound_steps` leaves 4 as soon as `300*dt/spacing >= 2`, so on a
sub-km nest this trips almost at once.  Adding the missing name would
have left the same trap for the next derived field, so the set is
single-sourced: `ADAPTIVE_DERIVED_RUN_FIELDS` names them, `_apply` builds
its replacement by iterating it, and deriving an unnamed field is
impossible while naming an underived one raises immediately.  It is not
fixed by exempting `time_step_sound` unconditionally: under a **fixed**
clock it is a real model difference and a resume that changes it must
still refuse.  Both halves are pinned.

**Recovery means changing what killed the run (2, 3, 5).**  For the
prepared cache this was simply wrong: nothing on the preparation path
reads a target or a clamp (the only prepare-side module naming
`target_cfl` or `min_time_step` was the identity check itself) so a tree
prepared under one value is byte-for-byte a tree prepared under another.
Those eleven fields are now skipped outright in the prepared-cache
comparison, in both directions and at any value, through the same
`PREPARATION_INERT_RUN_FIELDS` partition the inflow-perturbation keys
use; the standing check for anything proposed for it is to grep the
preparation path for a reader, and if preparation reads it, it belongs
in the identity instead.

For the restart walk and the experiment identity the binding was
deliberate, and the stated reasoning was that a resume changing a target
integrates on a different clock and so is not the same experiment.  The
premise is true and the conclusion was wrong: **integrating on a
different clock from here on is what a recovery resume IS**, and
refusing it made `restart_interval_s` useless for the case that most
wants it, a run that died because of the setting now being changed.  It
is allowed and REPORTED: the walk warns, naming each field with its old
and new value, so a resumed leg still says where its seam is and what
moved.  Silence was never the requirement; the refusal was doing the job
of a record.

`use_adaptive_time_step` itself stays refused in every gate.  Flipping it
leaves the carried controller state describing a clock that no longer
runs.

**A stored hash is not a comparable artifact (4, 6).**  Both sides are
normalised under the current rules before comparison, conservatively: it
can only ever ALLOW a resume the digest already rejected, and only when
every component matches once the fields this build no longer binds are
dropped from BOTH sides.

**The relocation chain (7).**  A moved tree's fingerprint is the build's
base with every move record hashed into it, one way.  The restore
replayed those records from THIS build's base, so any change to what the
base binds orphaned every relocation checkpoint ever written, and the
refusal blamed the move history:

    it was written after 184 nest relocation(s) ... resumes only into the
    run that wrote it

Read at face value that says relocating runs cannot be restarted at all.
It is not true, and the move history was never the obstacle: proved by
restoring the checkpoint's own original configuration exactly and
watching it refuse anyway.  The header carries the components the writing
run hashed, so its base is recoverable: drop the `relocation` block and
hash the rest.  On the reported checkpoint that base replays through all
184 records to the stored fingerprint exactly.  Integrity is unchanged:
the records hash one-way, so a forged or truncated chain still cannot
reproduce it.  The refusal now names the components that differ as well
as the move count, so the next mismatch on a moved tree does not send the
reader back into the move history.

**Self-consistency is not identity, and the replay alone is only the
first half.**  The replayed value IS the stored fingerprint (that is
what makes it a match) so adopting it on the strength of the replay
alone hands the gate in `woof.io.restart` the header to compare against
itself.  That gate is the only place a tree's preparation receipt, cache
content, execution plan and runtime source identity are ever compared, so
a checkpoint from a *different prepared tree* would have resumed
silently: its chain replays perfectly too, because the chain proves the
move history was not tampered with and says nothing about which tree made
the moves.  Adoption therefore requires BOTH: the replay reproduces the
stored fingerprint, AND the stored components agree with the live ones
once both are normalised under this build's rules, the same instrument
gates 4 and 6 already use.  `fingerprint_across_stored_chain` is the one
function, so both halves are testable without a model, and
`tests/test_restart_across_move.py` pins the widened case, the foreign
tree and a truncated chain.

**A resumed leg inherits its itinerary, it does not replay it.**  A
`[[relocation.move]]` queue is strictly ordered and its head is matched
EXACTLY, so a row whose opportunity is behind the clock can never match
again, and while it sits at the head it blocks every later row with it.
MEASURED on a 3 / 1 km tree with moves at 1200 s and 2400 s, resuming
from the 1800 s checkpoint: **neither** move executed, the nest stayed at
its 1200 s footprint, and the two frames after the missed 2400 s move
differed from the uninterrupted run, while the restore banner said the
resume reproduces that run bit for bit.  Not adaptive-specific: a
fixed-clock resume across the same move lost the same two frames.  Rows
behind the clock are now retired at the first consultation and named in
the summary receipt as `moves_behind_the_clock`, which is the reading
that was always correct: a row behind a checkpoint executed *before*
that checkpoint was written, and its result is already in the restored
state.

## The nest divide collapsed a child's step to arithmetic

Found only because the restart above finally worked.

A nest's step is the largest divisor of its parent's tick count at or
below what its CFL asks for.  When the parent is poor in factors, the
smallest usable divisor is enormous.  Measured on a 10 / 2 / 0.667 km
tree at lattice 15: a root of 5595 ticks is `15 x 373`, so d02 took
`1119 = 3 x 373`, divisors 1, 3, 373, 1119 and nothing between.  d03
asked for about 1 s, got **0.03 s**, and took **373 substeps inside one
parent step** where the grid ratio is 3.

**The run did not fail.**  It slowed by the factor (11.5 model-seconds
per wall-second fell to 0.23) which from outside is indistinguishable
from a controller answering a violent flow.  Every field was healthy
while it happened:

| d03 | start | end |
|---|---|---|
| max abs w | 38.7 | **23.0** m/s (falling) |
| cells with abs w > 20 | 1145 | **62** |
| surface pressure | 919.0 | 917.5 hPa (1.5 mb in 2 h) |

It read as a diverging storm until the fields were checked.  **Read dt
alongside the fields, never alone.**

Two defences, different in kind:

1. **The root is snapped to a SMOOTH tick count**, not merely one on the
   ratio lattice.  Being on the lattice makes the NOMINAL divide exact
   and nothing more: the cofactor can still be prime, which 373 was.
   The multiplier is 4, and 4 is measured rather than asserted.  Sweeping
   the three-level 10 / 2 / 0.667 km shape (lattice 15) over every root
   request from 20 s to 90 s at tick resolution, with d03 asking for a
   fifteenth to a sixtieth of the root: 35,005 pairs per factor:

   | multiplier | root resolution lost (mean) | d03 requests the divide cannot meet |
   |---|---|---|
   | 1 | 0.07 s | 6,705 / 35,005 |
   | 2 | 0.15 s | 2,640 / 35,005 |
   | 3 | 0.22 s | 1,980 / 35,005 |
   | **4** | **0.30 s** | **0 / 35,005** |
   | 6 | 0.45 s | 0 / 35,005 |
   | 8 | 0.60 s | 0 / 35,005 |
   | 12 | 0.90 s | 0 / 35,005 |

   4 is the smallest multiplier that clears the collapse on this shape,
   and every larger one costs root resolution in proportion for no
   further benefit.  The guarantee is bounded and worth stating plainly:
   it is that the root's quotients carry interior divisors, **not** that
   every descendant is divisor-rich at every depth.  Graded on the
   two-level and three-level shapes above; a deeper or more exotic
   ratio set has not been swept.

   The smoothing is applied only to a tree that HAS nests.  A
   single-domain run has no divide to keep rich in factors, so
   coarsening its root buys nothing and costs resolution: it was handing
   22.52 s for a request of 22.53 s, and honouring an explicit
   `starting_time_step` only to within four ticks.
2. **The divide PREFERS a lattice-friendly quotient, and degrades when
   it cannot afford one.**  `_quantise_root` gives the root that
   guarantee and nothing gave the middle domain, which on a three-level
   tree is a parent AND adapts.  The preference is a preference: when
   the smallest lattice-friendly divisor is past the substep ceiling the
   plain divide answers instead.  Returning through the ceiling check
   made a *preference* into a refusal that killed the middle domain of a
   three-level tree mid-run, parent 246 ticks, child asking 62,
   lattice 3: the lattice-friendly divisor is 41 substeps, past a ceiling
   of 24, while the plain divide answers 6.
3. **A substep count past 8x the grid ratio is REFUSED**, naming the
   domain, the counts, the measured collapse and the remedy.  Slow and
   silent is the worst way for this to fail.  It is a mid-run refusal,
   so the door prints it as one sentence and exits 2 with a failed-run
   receipt, rather than as the traceback a crashed forecast takes.

Measured on the three-level 10 / 2 / 0.667 km tree the collapse was
found on, same checkpoint and same config before and after:

|  | collapsed | fixed |
|---|---|---|
| d03 dt min | 0.03 s | **1.74 s** |
| d03 dt mean | 0.67 s | **3.10 s** |
| most common step | 0.03 s | **3.0 s** |
| throughput | 0.23 model-s/wall-s | **12.01** |

Reproduced on the release line, on a two-level 3 / 1 km tree with the
condition constructed through the config alone (d01 pinned at 1119 ticks
= 3 x 373, d02 pinned at 3.0 s, which asks the divide for four or more
substeps).  Same config file for both arms, both exit 0 with four frames
and no failure, which is the point:

|  | before | after |
|---|---|---|
| root dt | 11.19 s | 11.16 s |
| d02 dt | 0.03 s | **2.79 s** |
| substeps per parent step | 373 | **4** |
| d02 steps in 120 model s | 3,372 | **44** |
| throughput | 2.007 model-s/wall-s | **31.08 / 31.99** |

Two runs of the fixed arm, identical dt series and substep counts in both,
so the throughput spread is wall clock.  About 15x, lost silently, with
every field healthy.

## A shared clamp is a per-domain trap

`min_time_step` is scoped `max_domains`, so a value written once in
`[shared]` reaches every domain unchanged.  That reads as a sensible
default and is not one: the floor is an absolute number of seconds while
each domain's step is a fraction of the root's.

Measured on a 10 / 2 / 0.667 km tree with `min_time_step = 6` in
`[shared]`, written as "one fifth of the start", which it is, for the
root alone:

| domain | own step | inherited floor | floor / step |
|---|---|---|---|
| d01 | 30 s | 6 | 0.2 |
| d02 | 6 s | 6 | 1.0 |
| d03 | **2 s** | 6 | **3.0** |

On the inner nest the floor sat ABOVE the step it was meant to protect.
The controller could never ask for less than 6 s, the nest divide settled
on parent/2, and every checkpoint recorded d03 at 5–6 s against its
configured 2 s.  At 94.8 m/s over 667 m that is a horizontal Courant of
**0.853 against a 0.84 target**: the clamp produced the number the
controller was then blamed for holding.  At the configured 2 s the same
flow is 0.284.

The controller was never the problem; it was forbidden from acting.  A
clamp expressed in seconds and applied to a tree needs to be set per
domain, or scaled by the refinement ratio, and `[shared]` is the wrong
place for it on anything but a single domain.

## Per-domain coverage, audited against the Registry

The split between tree-wide and per-domain settings was arbitrary in
places rather than principled.  Audited against WRF's own
`Registry.EM_COMMON` scoping: `diff_6th_opt` and `diff_6th_factor` were
per domain while `diff_6th_slopeopt` and `diff_6th_thresh` (the same
filter) were not; `epssm` was while `emdiv` and `smdiv` were not; the
damping coefficients were not exposed at all.  Eleven numerics WRF
declares `max_domains` are now per domain, taking the surface from 42
keys to 53:

    diff_6th_slopeopt  diff_6th_thresh  dampcoef  zdamp
    emdiv  smdiv  khdif  kvdif  h_sca_adv_order  moist_adv_opt
    tke_budget

Each goes through the same three registrations any per-domain key needs:
a prepared-cache ruling (all eleven predate that table's baseline or
already carry one), `woof.experiment._DOMAIN_RUN_OVERRIDES`, and
`woof.namelist_import`, which now reads each as a per-domain column and
emits a `[[domain]]` override only where a domain differs from the root.
A namelist whose columns are uniform therefore imports to a
byte-identical TOML; a namelist whose columns are not uniform used to be
refused outright and now imports.

All eleven are READ as columns, which is what stops a tail being
silently dropped: `emdiv`, `smdiv` and `h_sca_adv_order` were typed
`dyn.scalar`, which took element 1 and discarded the rest.  Nine of them
can then actually DIFFER between domains.  The other two cannot, and it
is the scheme rather than the scoping that says so: WOOF implements one
value of each, so the importer refuses any element of
`h_sca_adv_order` that is not 5 and any element of `moist_adv_opt` that
is not 1, per domain and naming the domain.  They are columns so that a
namelist which varies them is refused BY NAME rather than accepted with
its tail dropped.

Additive and backward compatible: a domain naming none of them still
takes the `[shared]` value, so no existing experiment moves.  The point
is that a tree can tune damping or diffusion on the nest that needs it
without moving the parent, and on a 10 / 2 / 0.667 km tree those want
different values, since the relaxation sponge alone is 40 km on the root
and 2.7 km on the inner nest at the same cell count.

Deliberately still tree-wide: geometry (`dx`, `dy`, `ztop`, `grid_id`,
`nested`, `specified`), which the domain tree authors rather than the
operator chooses; and the scheme SELECTORS WRF also scopes `max_domains`
(`ra_lw_physics`, `ra_sw_physics`, `sf_surface_physics`, the `bl_mynn_*`
block), on the same grounds the SASE closures already are: a tree whose
domains ran different schemes cannot be compared across its own
boundary, and two-way feedback already requires one microphysics
tree-wide.

## A relocation cadence is a STEP COUNT, not a duration

Worth stating because it is invisible in a config file, and because it is
a consequence of this work rather than a pre-existing defect.

`cadence_seconds` never reaches the runtime gate as seconds.
`_cadence_periods` converts it **once, at runner construction**, into a
whole number of root periods.  The gate `is_due` then fires on
`ticks % (period_ticks * cadence_periods) == 0`, and `period_ticks`
resolves to `clock.root.step_ticks`: the root step **right now**, which
under an adaptive clock floats.

So the count is frozen and the modulus is not:

| configured | means | at dt = 30 | at a 60 s ceiling |
|---|---|---|---|
| `cadence_seconds = 180` | 6 root steps | 180 s | **360 s** |

**Anything gated in real model seconds alongside it quietly stops binding
as the step grows.**  `cooldown_seconds` is the one that bites: at a
360 s cadence and dt = 60, consultations fall 720 s apart, a
`cooldown_seconds = 360` never binds again, and the achievable tracking
speed drops with it.

Admission cannot catch this.  It validates `cadence_seconds` against the
root's **nominal** `time_step`, so a config passes and then means
something else at run time.  That is not a check that can be tightened:
under an adaptive clock there is no single root step to validate against.

**Measured, and it is not firing off-grid.**  On a 3-domain tree
(10 / 2 / 0.667 km) running adaptive dt and a follow source together,
three separate cadences were each checked over the first ~57 model
minutes and all three landed exactly on their configured lattices: a
180 s mover, a 720 s containment slide, and a 60 s track interval across
42 rows.  Boundaries are *skipped* while dt ramps, never displaced.

## What is ruled out

- **Fixed dt = 60.**  Buys 1.45x and is **unsafe**: 86% of steps on the
  damper at 4 h, against 0/480 at dt=30.  It looks like a free win for
  the first hour, which is how long a careless test runs.
- **The geometric CFL as the control variable**: a different quantity,
  off by ~2x, and it inverts the conclusion.
- **`microphysics + cumulus + pbl` as a clock proxy.**  The standing
  normalisation for run-to-run drift here, and invalid for this change:
  when `dt` is global the proxy scales with the very thing being
  measured, so it reports a speedup that is partly its own denominator
  moving.  Interleaved absolute wall time is the only sound form for a
  dt change.
- **A step-count ratio as a wall-clock ratio.**  At 2x dt everything
  step-bound scales 1.99x, but radiation and output are on *time*
  cadences and scale 1.00x, and microphysics/PBL cost *more* per larger
  step (1.66x/1.83x).

## What is open

- **`median_max_over_p9999` has not been measured.**  The run that would
  have produced it was ended at ~15%; the probe dump is `atexit`-only and
  a hard kill loses it.
- **`target_cfl` above 1.0 is unexplored.**  The 1.466x median was taken
  at 1.0.  The shipped default is WRF's 1.2 and it runs (207 steps,
  47.7 s on the same tree), but it has not been graded for intensity, and
  the two are different settings: they must not be cross-quoted.
- **`target_cfl` does not bind at the shipped clamps at all.**  Measured
  on a 3 km one-hour case: with `max_time_step` at WRF's derived 24 s,
  and again at 60 s, the 1.2 and 1.0 arms produced identical step counts,
  identical dt series and seven of seven byte-identical output frames.
  dt is held by the 5%/step growth cap and the ceiling for the whole
  hour, and the CFL target never gets a say.  It separates the arms only
  at `max_time_step = 300` (dt ceiling 121.20 s against 120.04 s, two of
  seven frames differing), so the shipped `target_cfl` is not yet a
  setting a short run can feel, which is also why it is ungraded.
- **Why relocation's awake-boundaries land on exactly 60 s once dt
  settles near 50 s.**  Two periods of a 49.95 s root step is 99.9 s, not
  60, yet the rows are exactly 60 s apart and exactly on the lattice.
  Something is regularising it that the plain
  `ticks % (period_ticks x cadence_periods)` reading does not capture, so
  the "boundaries are skipped while dt ramps" account fits the data
  without being proven by it.
- **Whether any of that survives `step_to_output_time = false`.**  The
  measured run had it on, with root output every 720 s, and the plausible
  story is that snapping the step to land on output times keeps `ticks`
  an exact multiple of the current step and so keeps the modulus
  meaningful.  If that is what holds it together, then
  `step_to_output_time` is a **prerequisite for relocation** under an
  adaptive dt and must be stated rather than left as a default.  The
  deciding experiment is cheap: the same tree with the flag off, a config
  edit and a re-run, no re-prepare.
- **Whether WRF's default targets are safe on a sub-km nest in an extreme
  storm.**  A reported run reached 950 mb / 139 kt with 94.8 m/s on a
  667 m nest at dt = 6 s: a horizontal Courant of ~0.85 against
  `target_hcfl = 0.84`.  The controller was holding exactly the target it
  was given, and the run then died in the PBL with a non-finite tendency.
  Whether the targets are wrong there, or whether something else fails
  first, is unresolved; the storm deepened 9.3 mb in its final hour,
  which is not a physical rate, so it may be a runaway no timestep policy
  would have caught.
- **A dt change needs a time-step sensitivity test of storm intensity
  against a small-step reference run (solution verification), not only
  a wall-clock measurement.** Comparison with observed intensity would
  be validation and answers a separate question.
  Barrett et al. (2019, JAMES 11, 641) measured a 53% precipitation
  reduction going 1 s -> 15 s in COSMO, independent of stability.

---

## How this was measured

Two runs of the same two-domain tree, 30 minutes of forecast, differing
only in `use_adaptive_time_step`: **three interleaved replicates of
each**, medians taken.  Wall clock drifts +/-20% run to run with no
hardware throttle, so a single pair cannot resolve a difference this size, and the
replicates are interleaved rather than batched so a slow patch cannot land
entirely on one arm.

Each arm needs its **own prepared tree**: `use_adaptive_time_step` is part
of the prepared-cache identity, so crossing an adaptive config with a
fixed tree is refused (see the tolerance note above).

Bitwise gates `cmp` every wrfout file, and the **corpus count is asserted
first**: an empty glob has silently passed a gate in this repo before,
so a gate that matches nothing now fails instead of reporting success.

The per-step CFL series a controller reads comes from an environment
probe on the same reduction the controller uses.  It writes from an
`atexit` hook, so it only lands on a clean exit: a hard kill loses the
whole series, and `SIGINT` does not help on Windows for a non-console
child.  That cost one run's worth of data during this campaign.

Tests: `tests/test_adaptive_timestep_{oracle,controller,surface,checkpoint,executor}.py`,
`tests/test_adaptive_clock_driver.py`, `tests/test_wrf_cfl_histogram.py`,
**1818 together**.  The oracle harness is `tools/calc_dt_wrf_oracle/`
(`run_calc_dt.F90` + `stubs.F90`), which compiles `calc_dt` against a
local WRF checkout and captures **1728 rows**; the port is graded against
them at `max_ulp == 0`.
