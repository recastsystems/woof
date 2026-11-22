# Spawn-at-trigger nests: the design, the contract, the seams

Status: implemented on `lane/nest-spawn` (task #102 + the spawn half of
#103).  The storm-following program's birth event: a user declares a
dormant high-res nest plus a trigger in config; when the trigger fires
mid-run, the nest materializes at the tracker-chosen position with its
own-grid statics and a parent-derived atmosphere, and follows the storm
from then on ([relocation.follow], leg 1/leg 2).

## The reservation contract (decision 1, stated plainly)

A dormant nest is **pre-declared, not mid-run allocated**.  The
`[[domain]]` carries its full size, resolution and physics, so:

- the memory plan prices it **exactly as if it existed**
  (`estimate_experiment` iterates every declared domain, dormant
  included, and the shared scratch arena / dycore-state workspace are
  sized over the full domain set at startup);
- preflight can **refuse accurately at planning time** -- `woof check`'s
  envelope already contains the reservation, and prints one advisory
  line per dormant nest naming what its declaration costs
  (`woof.core.preflight.spawn_reservation_advisories`);
- spawning is **ACTIVATION, not allocation**: the fired nest's memory
  was priced before the first step ran.

**A declared-but-never-triggered nest costs its reserved VRAM for the
whole run and zero compute.**  That is the contract, not a leak.

## Config surface (decision 4; governance: honored or refused)

```toml
[[domain]]                    # the dormant nest: full geometry + physics
grid_id = 3
parent_id = 2
i_parent_start = 40           # PLACEHOLDER placement: prices the plan,
j_parent_start = 30           # anchors the default search box, and is
parent_grid_ratio = 3         # the manual time-trigger's position
parent_time_step_ratio = 3
e_we = 121
e_sn = 121
history_interval_s = 60.0
spawn = { trigger = "uh", threshold = 60.0, earliest_s = 900.0, latest_s = 7200.0 }
```

- `trigger = "uh" | "reflectivity" | "pressure" | "time"`: the
  tracker's signal vocabulary (the spawn watch's own updraft-helicity
  window / composite `refl_10cm` / the pressure field reduced from the
  live prognostic column, read through the same
  `woof.core.storm_tracking.signal_plane` with the same missing-slot
  refusals), plus the manual deterministic form.
- Field triggers require `threshold` (field units), `earliest_s`,
  `latest_s` (model-time window; after `latest_s` the watch closes and
  the nest never spawns: the reservation stays held).  Optional
  `search_box = [i_lo, j_lo, i_hi, j_hi]` (1-based inclusive parent
  cells); absent, the box is the declared footprint plus the
  `[relocation.follow]` margin when a follow block exists, else the
  whole parent.
- `trigger = "pressure"` is the one INVERTED trigger: a cyclone is a
  minimum, not a maximum. It takes the same two optional knobs
  `[relocation.follow]` carries, with the same meanings and the same
  refusals, through the same validator:
  - `level_hpa`, which surface the vortex is looked for ON. Absent is
    `850` (the low-level circulation centre), where `threshold` is
    **metres** of geopotential height above the search box's own
    minimum. `level_hpa = 0` is the sea-level form, and is the only one
    whose `threshold` is an absolute **hPa** ceiling. The two threshold
    bands are disjoint (1–500 m against 800–1100 hPa), so a config that
    means one and would be read as the other refuses at load.
  - `radius_km`: how far from the extremum the centroid may draw
    (default 50 km; a tropical cyclone's core is 20–100 km).

  One surface only: the extremum cell is what claims a storm under the
  exclusion rule below, and a deep-layer mean of per-level centres has
  no extremum cell to claim one with. `[relocation.follow]` is where a
  deep-layer mean steers a nest that already exists.

  Under `level_hpa` the threshold is relative, so *some* cell always
  qualifies: the watch therefore fires only when the search box's own
  signal **span** reaches `threshold`. A flat box is a no-signal, not a
  spawn at the box's own centre.
- `trigger = "pressure"` reads no scratch stash. It is reduced from the
  live prognostic column, which is valid at every cycle boundary, so a
  pressure spawn has no cadence contract with the output knobs (the
  line `woof.core.storm_tracking.STASH_BACKED_FIELDS` already draws).
- `trigger = "time"` requires `at_s` alone (whole parent steps,
  spawning is a cycle-boundary operation) and spawns at the DECLARED
  placement: the testable, bit-deterministic form.
- Refusals: unknown keys (with did-you-mean); spawn on the root; spawn
  beside `start_time` (the trigger owns activation time); a child
  declared under a dormant parent (cascading activation is not
  implemented); `at_s` off the parent-step lattice; a window that can
  never open.
- Multiple dormant `[[domain]]` blocks are first-class; each carries
  its own trigger and its own reservation.

## Position choice (decision 2)

At trigger time the watch reads the parent plane, takes the strongest
qualifying cell in its search box (the loudest under `uh` and
`reflectivity`; the **deepest** under `pressure`), and computes the
storm-core weighted centroid over a footprint-sized window around it,
so with two storms in one box the placement lands on the *stronger
storm*, never between them.  That centroid comes from
`woof.core.storm_tracking.locate_signal`, the tracker's own arithmetic,
which owns the minimum inversion in one place: negating both the plane
and the ceiling turns "deepest low" into the same problem as "loudest
cell", so the weight becomes the pressure deficit below the ceiling and
there is no second centre-finder to drift.  The
footprint is centered on that centroid, whole-parent-cell aligned per
leg 1's relocation convention, clamped so every side keeps
`spec_bdy_width + blend_width` parent rows clear of the parent edge
(the loader's own admission rule, re-validated at activation by
`woof.experiment.validate_spawn_placement`).

Two nests, two storms: a watch **ignores signal inside another active
nest's footprint on the same parent grid** (the exclusion rule: the
masked cells are filled with an infinity of the losing sign, so they
lose the extremum search on either convention), and
`SpawnController.evaluate_all` feeds each event fired at a boundary
into the exclusion set of the watches on the same parent grid evaluated
after it, so two triggers crossing threshold at one boundary cannot
claim one storm.  A footprint is cell numbers on one grid
(`NestFootprint.parent_id`), so it never masks a watch on another grid:
a permanent d02 placed on d01 says nothing about which cells of d02 a
dormant d03 may fire on.

## Spawn initialization (decision 3)

`woof.ingest.nest_spawn_init.spawn_child_from_parent`:

1. **Own-grid statics** for the fired footprint through the
   footprint-parametric static path (`build_static_for_domain` on the
   child grid nested at the fired placement; `[static.highres]` overlay
   when the case enables it: the tile cache makes the trigger-time
   build fast).  `prepare_spawn_statics` is the one-call form.
2. **Atmosphere from the CURRENT parent** through `parent_only_init`'s
   full-parent SINT fill: the spawned nest is born inside the storm
   its trigger saw, not inside a stale analysis (which is what the
   delayed-start path would have used, and why it is not used here).
3. **Terrain adoption**, byte-for-byte the real-data child adjustment
   sequence with the analysis's role played by the parent: fine-frame
   ht/mub/phb from the analytic hydrostatic real base on the own-grid
   `HGT_M` (`woof.ingest.real._make_real_base`), `blend_terrain` on
   all three operands (parent near the boundary, fine interior), in
   float64 on the fine base so the re-derived base keeps its FP32 EOS
   correction (`dphb_resid`),
   `adjust_tempqv` for the column-mass change, the `start_domain`
   base/EOS re-derivation, and the real-nest `press_adj` MU correction
   (`_adjust_and_rederive`), then the RK time-t reseed.
   - Calibration: with fine terrain identical to the parent-SINT
     terrain the adoption is the identity (bitwise except the real
     path's own theta −300 K/+300 K FP32 roundtrip, ≤1 ULP on `thp`).
   - Treatment: a 300 m hill shifts the column dry mass by −g/α₀ per
     metre (measured −11.6 Pa m⁻¹), a valley by the mirror image.
4. **Receipts**: trigger evidence (field, threshold, peak, centroid,
   search box and its source, exclusions), fired placement, static
   source and highres provenance, atmosphere provenance (parent state
   sha), adoption numbers, RK reseed, and the measured
   parent-bitwise-unchanged claim.

Physics/land DRIVER state follows the same rule as every leg boundary
and relocation: re-initialised through the `on_child_built` seam, never
invented by the initializer.

## Land-surface state for a nest with no analysis (2026-08-07)

A newborn has no prior self to continue from and no `real.exe` product
at a footprint nobody knew about until the trigger fired, so neither the
relocation transplant nor the t = 0 soil path applies to it.  WRF has
exactly two routes for a nest that starts later than its parent (Users'
Guide chapter 5, "Nesting"):

- `fine_input_stream = 2`: 3-D meteorology interpolated from the
  parent, static AND masked surface fields (soil temperature and
  moisture among them) read from the nest's own `wrfinput`.  That file
  is a `real.exe` product at the nest's footprint and start time, so the
  route presupposes knowing both in advance.
- `input_from_file = .false.`: "the model interpolates all variables
  required in the nest from the coarse domain fields".

A trigger-spawned nest cannot have the first, so it takes the second,
and that is not a degradation.  It is the operator WRF runs
unconditionally at EVERY nest birth (`med_nest_initial` calls
`med_interp_domain(parent, nest)` before any input file is consulted,
`share/mediation_integrate.F:670`) and, via the identical call after
each `shift_domain_em`, at every moving-nest leading edge
(`share/mediation_nest_move.F:186`).  HAFS's storm-following nest states
the rule in words: interpolation at the leading edge "taking into
account the land/sea/ice mask to only consider values from the same
surface type".

WOOF-ArWen keeps the half of `fine_input_stream = 2` it CAN have (the
own-grid statics) which is strictly more than `input_from_file =
.false.` gives, and is exactly why the mask is essential: the child's
land-use categories are its OWN, resolved at the child's dx, so they
disagree with the parent's wherever finer terrain resolves a coast, lake
or island the parent smoothed away.

- **Operator**: `woof.core.nest_interp.interp_mask_field`, a
  transliteration of `share/interp_fcn.F:4075-4275`, the interpolator
  the Registry names per field
  (`i02rhd=(interp_mask_field:lu_index,iswater)` on TSLB/SMOIS/SH2O/
  SMCREL/TSK/SNOW/SNOWH/CANWAT, `,isice` on XICE).  Uniform-class cells
  are plain 4-point bilinear; mixed cells average only the corners
  matching the CHILD's class; a cell whose four corners are all the
  other class takes WRF's one-cell island/lake compromise (plain
  bilinear, its own comment: "no better way").
- **Inventory**: `LAND_SURFACE_CONTINUATION_FIELDS`, the same set a
  relocation carries.  The two events differ in where the state comes
  from, never in which state.
- **Receipts**: `spawn_land_state_from_parent` counts every branch per
  mask family and publishes `land_class_conflict_cells` (the island/
  lake fallbacks) the way `donor_fill_plan` counts `class_fallback_
  filled`.  It reaches the birth certificate under `land_surface`
  through the preparer's duck-typed `last_receipt`.
- **Driver rebuild**: `runtime.rebuild_child_driver_from_land_state`,
  shared with the relocation preparer.  Accumulators are re-initialised
  at the new footprint and the receipt says so.

This is what lifted the real-data spawn refusal on the run route.
Routes that neither reserve nor watch still refuse by name.

## Runtime: what runs today, what awaits leg 2

- `woof.core.model.build_experiment` prices and fingerprints the FULL
  experiment but builds clocks, schedule and child inits from
  `pre_spawn_experiment(exp)`: the dormant nest is reserved and never
  integrated.
- Activation is leg-boundary schedule surgery:
  `active_experiment(exp, {grid_id: (i, j)})` is the validated view the
  post-spawn leg integrates (the same sequence-of-static-trees design
  the relocation demo uses).  The in-loop runner that evaluates
  `SpawnController` each relocation cadence, calls
  `spawn_child_from_parent`, attaches the node/clock/coupler and
  continues the walk belongs to the leg-2 relocation runner; the seams
  it consumes are `SpawnController.evaluate_all` /
  `active_experiment` / `spawn_child_from_parent`, all documented in
  their docstrings.
- Routes that neither reserve nor watch (prepared caches, DA, native)
  refuse a spawn declaration by name (`refuse_unrouted_spawn`), never
  drop it.

## Restart, retire

A restart across a spawn is an exact contract, not a posture. The
checkpoint persists the spawn watches, the fired placements, the episode
counts and the retirement timers, and a run split at any checkpoint
resumes bit-identical to the unbroken run: see `docs/nest-lifecycle.md`
and `tests/test_lifecycle_restart_identity.py`. Absent `spawn` is still
byte-inert on every pre-feature fingerprint and prepared-cache identity
(absent-stays-absent in `restart_identity_payload`;
`DEFAULT_TOLERANT_IDENTITY_FIELDS` tolerance); a declared spawn binds.

Retire/despawn shipped with the lifecycle tables: `retire` deactivates a
spawned nest at a completed leg boundary and `rearm` brings the slot back
for a later episode.
