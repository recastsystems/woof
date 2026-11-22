"""THE STORM-FOLLOWING MOVING NEST, under each of WOOF-ArWen's execution modes.

Run it::

    python -m tilestream.test_moving_nest            # the whole matrix
    python -m tilestream.test_moving_nest --probe    # just the UH calibration

WHAT IS UNDER TEST
------------------
A moving nest is two whole-domain reductions and one state swap:

1. :class:`woof.core.storm_tracking.StormTracker` takes a domain-wide
   weighted centroid of the PARENT's ``uh_follow_window`` plane
   (storm_tracking.py:461-474) and proposes a whole-parent-cell shift;
2. :class:`woof.core.relocation_runner.RelocationRunner` RESETS that
   window on the parent state at every evaluation, so the window means
   "strongest rotation since I last looked" and nothing else;
3. :func:`woof.core.nest_relocation.relocate_child` rebuilds the child on
   the new footprint and REPLACES ``node.state``.

All three read or write ``node.state``.  A STREAMED domain's arrays do not
live on ``node.state``: :func:`woof.core.streaming.attach` copies the
carriers into a store (``store="host"`` makes that store pinned host RAM,
which is the entire point of the mode) and every sweep gathers, steps and
scatters THE STORE.  The state object is left behind at the values it held
when it was attached, and nothing in the tree knows that.

So this module runs the SAME two-domain relocation configuration three
ways and compares what the nest actually did:

``resident``          the control: ``steppers = {}``, every domain on
                      ``dycore.step`` itself.
``streamed-device``   d01 streamed with ``store="device"``.  The store IS
                      the state's own arrays (streaming.py:735-738), so
                      this isolates the TILING arithmetic from the
                      TRANSPORT: whatever differs here is not a copy.
``streamed-host``     d01 streamed with ``store="host"`` -- the out-of-core
                      mode, and the one a user turns on.

The comparison is the EXECUTED SHIFT SEQUENCE -- ``(di, dj)`` or ``None``
at every relocation cadence -- plus the carrier digests of both domains at
the end.  A shift sequence is the right observable because it is what the
feature is FOR, it is discrete (so a difference is unambiguous rather than
a tolerance question), and it is a function of the whole parent plane at
every cadence, which is exactly the information a tile cannot see.

WHAT THIS LANE FOUND, IN THE ORDER IT BITES
-------------------------------------------
1. NO PRODUCTION ROUTE CAN TURN STREAMING ON AT ALL.  Both tree routes call
   ``streaming.steppers_for_tree(model, exp.tiles)`` with no builders, so
   ``[tiles] mode = "on"`` refuses at startup with "this route wired no
   streamed-domain builder".  ``[tiles]`` is also a TREE-WIDE block --
   ``ExperimentConfig.tiles``, one options object, no per-domain key --
   so "stream d01, keep d02 resident", which is the only shape a moving nest
   wants, is not expressible.  Both are asserted here as ``route_control``.

2. THE MODEL CLOCK DID NOT REACH THE STREAMED DOMAIN.  ``refresh_model_time``
   makes ``woof.core.clock`` the calendar authority for a resident domain
   and writes a state a streamed one never reads, so a streamed domain ran a
   SECOND free-running clock.  Fixed: ``StreamedDomain.impose_clock``, called
   from ``on_step`` beside ``refresh_model_time``.

3. THE TRACKER'S WINDOW WAS NOT A CARRIER, and the plane it reduces was the
   state's, not the store's.  Fixed: ``physics_inventory
   .STREAMED_ONLY_SCRATCH_SLOTS`` plus ``StreamedDomain.sync_to_state`` /
   ``sync_from_state``, called around the relocation runner and before every
   FORCE.

4. THE STATE SWAP IS STILL A HARD STOP, by design and unchanged.  A
   ``StreamedDomain`` handed a different state refuses; relocation's whole
   job is to hand back a different one.  It does not fire in the tested
   shape (the PARENT is streamed and relocation replaces the CHILD), and it
   fires immediately in the shape ``steppers_for_tree`` actually produces,
   which streams every domain in the tree.

THE CONTROLS
------------
``the isolating control``  ``parent_bitexact``: the SAME parent, alone, no
                           nest and no relocation, resident vs streamed.
                           Without it a d01 difference in the two-domain
                           legs cannot be attributed -- streaming and the
                           executor would both be candidates and neither
                           would be excluded.
``the projection``         each streamed leg runs at ``projection="none"``
                           (as shipped), ``"read-only"`` (reads projected,
                           the reset not pushed back) and ``"full"``.  The
                           first two MUST differ from resident and the third
                           MUST match; a one-sided assertion would pass on a
                           domain that happened not to move.
``the state swap``         ``StreamingRefused`` is provoked and its message
                           quoted rather than paraphrased.
``physics fired``          radiation / cumulus / PBL / LSM call counts are
                           printed for BOTH sides of every comparison, and
                           for a streamed domain they are read from the
                           SWEEP's scalar carriers -- the state's driver
                           holds only the warmup step, which is how a census
                           can say "radiation fired once" about a run in
                           which it fired nine times.

WHY A SEEDED STATE AND NOT A WK82 SUPERCELL
-------------------------------------------
``tilestream/REAL-DATA.md``'s argument, and one extra reason specific to
this lane.  The WK82 quarter-shear case is the natural moving-nest case,
but it runs on OPEN lateral boundaries, and ``dycore.step`` applies the
Klemp-Wilhelmson radiative condition at the WINDOW's edges
(dycore.py:2048-2125, called at :2384) -- so every interior seam of every
tile would take a radiative boundary condition, which no halo can undo.
Streaming has been proven bit-exact on PERIODIC and on SPECIFIED domains
and on neither open one; running the moving nest on an open-boundary
parent would measure that gap instead of this feature.  The configuration
here is the one the join gate proved: specified lateral boundaries, a real
Lambert projection, real terrain, moist physics -- with
``nwp_diagnostics = 1`` added, which no tilestream configuration has ever
carried before, because that is the switch that makes the UH accumulator
exist at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from woof.core import streaming
from tilestream import driver, gather, harness
from tilestream import physics_inventory as physinv
from tilestream import test_join

# --------------------------------------------------------------------------
# the configuration
# --------------------------------------------------------------------------

#: The parent.  256 divides by 128 / 64 exactly, and the tile MUST divide the
#: domain (a ragged trailing tile costs ~22% in ring mode instead of ~2%).
PARENT_NX = PARENT_NY = 256
NZ = 49
#: Whole seconds, because the root clock is rational (time_step +
#: time_step_fract_num/den) and the child's dt is the FP32 chained quotient.
PARENT_DT = 30.0
RATIO = 3
#: 72 child cells at ratio 3 is a 24-parent-cell footprint: small enough to
#: move several times inside the parent without ever approaching the
#: keepout, large enough that ``register_nest``'s +-2 SINT stencil is not
#: the thing under test.
CHILD_NX = CHILD_NY = 72
I_PARENT_START = J_PARENT_START = 116

TILE = 128
NBUFFERS = 2
#: Free VRAM a leg waits for before it starts.  MEASURED peak for the
#: 256^2 x 49 tree at this rung plus two 160^2 tile buffers is under 4 GiB;
#: the margin is for the co-tenant, not for this run.
VRAM_NEEDED_GIB = 6.0

PARENT_STEPS = 30
CADENCE_S = 150.0          # 5 parent steps -> 6 evaluations in 30 steps
RUN_SECONDS = PARENT_STEPS * PARENT_DT

START_TIME = datetime(2011, 4, 27, 18, 0, 0, tzinfo=timezone.utc)
SEED = 20_260_731

#: The physics rung.  Morrison + Smagorinsky + MM5 sfclay + YSU + Noah is
#: the "+Noah LSM" rung of ``test_join.RUNGS``; radiation and cumulus are
#: added on top so that both slow cadences fire inside the window (see
#: ``cadence_census`` -- the numbers are printed, never assumed).
#:
#: ``ra_*_physics = 90`` is the analytic clear-sky proxy, not RTE+RRTMGP.
#: The choice is about MEMORY and nothing else: this lane runs a whole
#: RESIDENT 256^2 tree beside a streamed one plus 2 tile buffers on a
#: shared card, and RRTMGP's column chunking OOMs there (measured, 4090
#: with 9 GiB already taken by another tenant).  What the cadence has to do
#: here is FIRE, on both sides, at a rhythm the streamed clock has to carry
#: -- ``radt_minutes = 2`` at dt = 30 s is every 4 steps and
#: ``cudt_minutes = 1`` every 2 -- and an analytic scheme tests the carried
#: clock exactly as a spectral one does.  The RRTMGP rung is proven streamed
#: bit-exact by ``tilestream.test_join`` and is not re-derived here.
_RUNG = dict(moist=True, mp_physics=10, ztop=20000.0,
             km_opt=4, sf_sfclay_physics=91, bl_pbl_physics=1, bldt=0.0,
             sf_surface_physics=2,
             ra_sw_physics=90, ra_lw_physics=90, radt_minutes=2.0,
             cu_physics=1, cudt_minutes=1.0,
             # The 2.0 carrier contract refuses Noah consuming a GLW no
             # producer ever wrote, and the analytic ra_*=90 proxy is not
             # in GLW's producer ledger -- so the RESIDENT control of this
             # matrix died at model second 150 and every comparison with
             # it was void (measured on the base tree, gatelogs/
             # moving_nest_base.log).  This harness runs a seeded synthetic
             # state, which is exactly the case the contract's remedy (3)
             # names: the policy consumes unsourced carriers at their
             # allocation fill -- the SAME zeros the pre-contract tree
             # consumed silently -- and labels them in the receipt.  Both
             # sides of every leg carry it, so no comparison is skewed.
             surface_radiation_policy="wrf_compat_zero")

#: The follow block.  ``max_shift_cells`` / ``max_move_parent_cells`` are 12
#: -- HALF the child's 24-parent-cell footprint -- because a move as large as
#: the footprint leaves no shared ground and ``relocate_child`` refuses it by
#: name ("that is a new domain, not a relocation"), which it did at 32.  The
#: clamp does not blunt the comparison: the tracker's unclamped weighted
#: CENTROID is recorded at every cadence beside the executed shift, and that
#: is the number the whole-domain reduction actually produces.
#: ``search_margin_cells`` is deliberately larger than
#: the domain so the search box IS the whole plane: this feature's risk is
#: a whole-domain reduction, and a search box that happened to sit inside
#: one tile would not exercise it.
FOLLOW = dict(field="uh", threshold=None, fallback_threshold=1.0,
              search_margin_cells=4096, min_shift_cells=1,
              max_shift_cells=12, cooldown_seconds=0.0)


def parent_cfg(nx=None, ny=None, nz=NZ, **over):
    """Specified BCs + real Lambert + real terrain + moist physics + UH."""
    kwargs = dict(harness.GEOGRAPHY_OVERRIDES)     # map_proj=1, terrain_opt=1
    kwargs.update(test_join.SPEC_BC)               # specified, spec_bdy_width=5
    kwargs.update(_RUNG)
    kwargs.update(dt=PARENT_DT, nwp_diagnostics=1, grid_id=1,
                  run_seconds=RUN_SECONDS, output_interval_s=RUN_SECONDS)
    kwargs.update(over)
    return harness.make_config(PARENT_NX if nx is None else nx,
                               PARENT_NY if ny is None else ny,
                               nz, periodic=False, **kwargs)


def child_cfg(parent, nx=None, ny=None):
    """The nest: ``nested=True``, dx/3, dt/3, the parent's physics."""
    nx = CHILD_NX if nx is None else nx
    ny = CHILD_NY if ny is None else ny
    return replace(parent, nx=int(nx), ny=int(ny),
                   dx=parent.dx / RATIO, dy=parent.dy / RATIO,
                   dt=float(np.float32(parent.dt) / np.float32(RATIO)),
                   specified=False, open_x=False, open_y=False, nested=True,
                   grid_id=2, case="movnest_child")


def experiment(parent, child, *, cadence_s=CADENCE_S, threshold=None,
               streaming_options=None):
    """The two-domain ``ExperimentConfig`` with ``[relocation.follow]`` on d02."""
    from woof.experiment import (DomainConfig, ExperimentConfig,
                                  ProjectionConfig, VerticalConfig,
                                  _build_relocation)

    root_dc = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=float(cadence_s), run=parent,
        time_step=int(parent.dt))
    child_dc = DomainConfig(
        grid_id=2, parent_id=1,
        i_parent_start=I_PARENT_START, j_parent_start=J_PARENT_START,
        parent_grid_ratio=RATIO, parent_time_step_ratio=RATIO,
        history_interval_s=float(cadence_s), run=child)
    follow = dict(FOLLOW)
    follow["threshold"] = float(threshold)
    relocation = _build_relocation(
        {"relocation": {"enabled": True, "grid_id": 2,
                        "max_move_parent_cells": 12,
                        "cadence_seconds": float(cadence_s),
                        "follow": follow}},
        "test_moving_nest", (root_dc, child_dc), RUN_SECONDS)
    kwargs = dict(
        name="movnest", start_time=START_TIME, run_seconds=RUN_SECONDS,
        vertical=VerticalConfig((), 0.0, 1, 0.2),
        projection=ProjectionConfig("lambert", **{
            "ref_lat": harness.REAL74_PROJECTION["ref_lat"],
            "ref_lon": harness.REAL74_PROJECTION["ref_lon"],
            "truelat1": harness.REAL74_PROJECTION["truelat1"],
            "truelat2": harness.REAL74_PROJECTION["truelat2"],
            "stand_lon": harness.REAL74_PROJECTION["stand_lon"]}),
        restart_interval_s=0.0, domains=(root_dc, child_dc),
        relocation=relocation)
    if streaming_options is not None:
        kwargs["streaming"] = streaming_options
    return ExperimentConfig(**kwargs)


# --------------------------------------------------------------------------
# building one tree
# --------------------------------------------------------------------------

def build_parent(cfg, *, seed=SEED, boundaries=None, warmup=1):
    """The prepared, resident parent on the real projection and terrain."""
    from woof.ingest.lateral_bc import attach_lateral_boundaries

    geo = harness.make_geography(cfg, terrain=True, periodic_faces=False)
    state, _drv = harness.make_physics_state(cfg, seed, geography=geo,
                                             start_time=START_TIME)
    if boundaries is not None:
        attach_lateral_boundaries(state, boundaries)
    if warmup:
        harness.run_steps(state, cfg, int(warmup))
    # Allocate the tracker's FALLBACK plane, zeroed, so that "no UH signal"
    # is a HOLD in every leg instead of a hold in one and a TrackerRefusal
    # in another.  ``_plane_from_state`` raises outright when the slot is
    # absent (storm_tracking.py:407-420), and under streaming the slot is
    # never allocated at all: ``compute_refl_10cm`` calls
    # ``state.scratch(..., "refl_10cm")`` on whichever state ran the
    # microphysics, and under streaming that is a TILE BUFFER.  Without
    # this, the stale-plane legs would die at the first evaluation for a
    # reason that is real but is NOT the one under test, and the shift
    # sequences could not be compared at all.  The slot is rebuild-class
    # scratch ("refl_" prefix, woof/io/restart.py:524) so it is in no
    # carrier set and no digest.
    state.scratch((int(cfg.nz), int(cfg.ny), int(cfg.nx)), "refl_10cm")
    return state, geo


def domain_boundaries(cfg, *, seconds=21600.0):
    """The parent's specified forcing, from two differently seeded states.

    Two genuinely different snapshots give a NONZERO time tendency, so
    ``dtbc`` -- and therefore ``elapsed_seconds`` -- actually reaches the
    answer; a single repeated snapshot would quietly disarm the clock
    carrier and pass on a build that never carried it.
    """
    import gc

    import cupy as cp

    from woof.ingest.lateral_bc import StateBoundaryFrames

    # ONE state at a time.  ``build_state_lateral_boundaries`` needs both
    # snapshots simultaneously, and each is a whole physics-on 256^2 x 49
    # domain, so building the tables that way costs three resident domains
    # where the comparison needs one -- a straight OOM on a shared card, and
    # measured as one twice on the 4090 this lane was developed on.
    # ``StateBoundaryFrames`` keeps only the four spec_bdy_width perimeter
    # frames and is documented exact, element for element, against the
    # all-at-once builder.
    geo = harness.make_geography(cfg, terrain=True, periodic_faces=False)
    frames = StateBoundaryFrames(spec_bdy_width=int(cfg.spec_bdy_width),
                                 spec_zone=int(cfg.spec_zone),
                                 relax_zone=int(cfg.relax_zone))
    for seed in (SEED, SEED + 1):
        snap, _ = harness.make_physics_state(cfg, seed, geography=geo,
                                             start_time=START_TIME)
        frames.add_state(snap)
        del snap
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    return frames.build([0.0, float(seconds)])


#: Moist mass and number fields SINT can drive slightly negative.
_POSITIVE_DEFINITE = (
    "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop",
    "nc", "nr", "ni", "ns", "ng", "nwfa", "nifa",
    "qnr", "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh",
    "effc", "effr", "effi", "effs")


def _clip_positive_definite(state) -> int:
    """Clamp the child's moist mass/number fields at zero after SINT.

    WRF's nest interpolation is a smoothing stencil with NEGATIVE weights,
    so interpolating a field that is zero over most of its extent -- which
    every Morrison number concentration is on a seeded state after one step
    -- undershoots: MEASURED -1.86e-9 in ``nr`` at (0, 8, 54) on this
    configuration, which the full-state health gate refuses by name
    (``violates lower bound >= 0.0; classes=('moment',)``).  Clamping is
    what the schemes themselves do to their own number moments
    (module_mp_morr_two_moment.F:1528-1635) and it is applied through the
    preparer, so the t=0 child and every RELOCATION-rebuilt child get the
    identical treatment and the health gate stays ARMED for the whole run
    rather than being switched off to make the case build.
    """
    import cupy as cp

    fixed = 0
    for name in _POSITIVE_DEFINITE:
        arr = getattr(state, name, None)
        if arr is None:
            continue
        bad = int(cp.count_nonzero(arr < 0))
        if bad:
            cp.maximum(arr, 0, out=arr)
            fixed += bad
    return fixed


def _preparer(state, dc, grid, start_time):
    """``assemble_idealized_tree``'s domain preparer, driver-aware.

    The parent arrives from ``harness.make_physics_state`` with its driver
    already attached; the CHILD arrives from ``parent_only_init`` with none,
    and a child without a PhysicsDriver integrates nothing.
    """
    from woof.verify.cases.nest_ideal_common import prepare_idealized_domain

    if getattr(state, "physics", None) is not None:
        return state.physics
    _clip_positive_definite(state)
    return prepare_idealized_domain(state, dc, grid, start_time)


def build_tree(exp, parent_state, parent_grid):
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree

    return assemble_idealized_tree(
        exp, parent_state, grids=[parent_grid, parent_grid],
        domain_preparer=_preparer)


def make_runner(exp, model, *, staging="device", receipts=None):
    """The route's job: the runner, its child preparer and its tracker."""
    from woof.core.relocation_runner import RelocationRunner

    def on_child_built(initialized, new_dc, parent_node):
        _preparer(initialized.state, new_dc, initialized.grid, START_TIME)

    return RelocationRunner.from_experiment(
        exp, schedule=model.schedule, on_child_built=on_child_built,
        staging=staging)


# --------------------------------------------------------------------------
# the streamed parent: the construction a route would own
# --------------------------------------------------------------------------

def parent_builder(domain, geo, boundaries, *, seam="zeros"):
    """``make_stepper``'s ``build``: the domain-specific half of the seam.

    Verbatim the shape ``tilestream.test_join`` uses -- the domain's
    geography gathered once per buffer, the lateral tables windowed once
    per tile, buffers built on POISON geography with the domain's own
    physics selectors and one warmup step for the lazily allocated
    carriers.
    """

    def build(state, run_cfg, decision):
        geo_inv = {k: gather.pinned_copy(v) for k, v in
                   driver.geography_inventory(domain).items()}
        per_tile = streaming.tile_boundary_tables(
            boundaries, streaming.tile_specs(run_cfg, decision), seam=seam)
        # REFL priming + the refl-carrying inventory, exactly as the
        # production builder (streaming.prepared_domain_builder) does.
        # The 2.2 train made a refl-due sweep REFUSE when the store has no
        # 'scratch/refl_10cm' to join the tiles' windows into; this
        # harness's history handler raises refl_10cm_due on the history
        # cadence, so without the priming every streamed leg of the matrix
        # died at its first due step -- on the untouched base tree too
        # (gatelogs/moving_nest_base.log), which is what dates the break
        # to the refusal, not to this lane.
        streaming.prime_lazy_carriers(state, run_cfg)
        base_make = test_join.tile_factory(run_cfg, per_tile[0])

        def factory(tile_cfg):
            buffer_state = base_make(tile_cfg)
            streaming.prime_lazy_carriers(buffer_state, tile_cfg)
            return buffer_state

        return streaming.attach(
            state, run_cfg, decision, tile_state_factory=factory,
            geography=geo_inv, boundary_tables=per_tile,
            scalars=physinv.carrier_scalars(state), check_geography=False,
            inventory_fn=streaming.refl_inventory(
                physinv.streaming_inventory))

    return build


def stream_parent(model, geo, boundaries, *, store="host", tile=TILE):
    """Attach a :class:`StreamedDomain` to d01 and return ``{1: stepper}``."""
    node = model.root
    cfg = node.cfg.run
    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(tile), tile_ny=int(tile),
        nbuffers=NBUFFERS, store=store)
    decision = streaming.decide(cfg, options)
    stepper = streaming.make_stepper(
        node.state, cfg, options, decision=decision,
        build=parent_builder(node.state, geo, boundaries))
    if not streaming.is_streaming(stepper):
        raise RuntimeError("the parent did not stream")
    return {1: stepper}


# --------------------------------------------------------------------------
# observables
# --------------------------------------------------------------------------

def shift_sequence(runner) -> list:
    """The executed shift at every cadence, in order: ``[di, dj]`` or None."""
    out = []
    for row in runner.receipts:
        if row.get("event") == "relocated":
            out.append(list(row["executed_shift_parent_cells"]))
        elif row.get("event") == "held":
            out.append(None)
    return out


def hold_reasons(runner) -> list:
    return [row.get("reason") for row in runner.receipts
            if row.get("event") == "held"]


def centroids(runner) -> list:
    """The weighted centroid the tracker computed at every evaluation.

    The most direct expression of the whole-domain reduction under test: two
    floats per cadence, straight out of ``weighted_centroid``, before the
    dead band, the cooldown, the clamp and the keepout have had any say.  An
    executed shift can agree by accident once it has been clamped to the
    move limit; a centroid pair agreeing to three decimals at every cadence
    cannot.
    """
    out = []
    for row in runner.receipts:
        for entry in row.get("tracker_receipts", ()):
            if "centroid_parent_ij" in entry:
                out.append(entry["centroid_parent_ij"])
            elif entry.get("decision") == "no-signal":
                out.append(None)
    return out


def tracker_fields(runner) -> list:
    """Which signal the tracker actually reduced at each evaluation.

    ``"uh"`` is the configured field; ``"reflectivity"`` means the UH window
    showed nothing at all and the handoff fired -- which is what a stale,
    all-zero plane looks like from the tracker's side.
    """
    out = []
    for row in runner.receipts:
        for entry in row.get("tracker_receipts", ()):
            if "field_used" in entry:
                out.append(entry["field_used"])
    return out


def placements(runner) -> list:
    return [row["placement_to"] for row in runner.receipts
            if row.get("event") == "relocated"]


def digests(arrays) -> dict:
    return test_join.digest_arrays(
        {k: v for k, v in arrays.items()})


def carrier_digest(obj) -> tuple[str, dict]:
    """One SHA-256 over a whole carrier set, plus the per-field map.

    ONE DIGEST BASIS FOR BOTH REPRESENTATIONS OF A DOMAIN.  A resident
    domain is digested off its state, whose ``carrier_manifest`` already
    excludes REBUILD-class scratch by the restart classifier; a streamed
    domain is digested off its STORE, a plain mapping whose keys pass
    through ``carrier_inventory`` unfiltered -- and since f54558672
    brought the harness builders to production parity, that mapping
    legitimately carries ``scratch/refl_10cm``: a primed, transport-owned
    REBUILD slot the state-side digest never sees.  Measured on the 5080
    (streamed-child matrix run 2): every one of 158 classified carriers
    bit-identical in both domains, both coupling directions, host AND
    device store, with this one-sided rebuild row the only difference --
    three FAIL rows about a field that is not trajectory state.  So the
    mapping side now applies the SAME production classifier the state
    side always applied.  This cannot mask a real defect: rebuild-class
    scratch is recomputed before every legitimate consume, and the
    streamed REFL transport has its own dedicated gates
    (``parent_bitexact --refl-due-every`` and the 2.2 frame field-set
    gate).
    """
    import hashlib

    from woof.io.restart import classify_scratch_slot

    per = digests(physinv.carrier_inventory(obj))
    per = {name: value for name, value in per.items()
           if not (name.startswith("scratch/") and classify_scratch_slot(
               name.split("/", 1)[1]) == "rebuild")}
    h = hashlib.sha256()
    for name in sorted(per):
        h.update(name.encode())
        h.update(per[name].encode())
    return h.hexdigest(), per


def cadence_census(model, steppers=None) -> dict:
    """How many times each slow scheme fired, per domain.

    Printed on BOTH sides of every comparison.  Three of this project's six
    false results were a number measured in a window where radiation and
    cumulus never fired, and a window that gives one side a free ride looks
    exactly like a result.

    A STREAMED domain's counters are NOT on ``node.state.physics``: the
    schemes run on tile buffers, and the DOMAIN's counts are the scalar
    carriers the sweep maintains (``TiledRun._advance_clock``, cross-checked
    per buffer).  Reading the state's driver for a streamed domain reports
    the one warmup step its preparation took -- which is how a census can
    say "radiation fired once" about a run in which it fired nine times.
    """
    steppers = steppers or {}
    out = {}
    for node in model.walk_parent_first():
        gid = int(node.cfg.grid_id)
        stepper = steppers.get(gid)
        if streaming.is_streaming(stepper):
            out[f"d{gid:02d}"] = dict(
                (stepper.scalars or {}).get("call_counts", {}))
            continue
        driver_ = getattr(node.state, "physics", None)
        out[f"d{gid:02d}"] = ({} if driver_ is None
                              else dict(driver_.call_counts))
    return out


def uh_plane(state, slot="uh_follow_window") -> np.ndarray:
    import cupy as cp

    buf = state.existing_scratch(slot)
    if buf is None:
        return np.zeros((0, 0))
    return cp.asnumpy(buf)


# --------------------------------------------------------------------------
# the legs
# --------------------------------------------------------------------------

#: What the store-to-state projection is allowed to do in one leg.
#:
#: ``"none"``      exactly what the tree did before this lane: the executor
#:                 never projects, so every whole-domain consumer reads the
#:                 state as ``attach`` left it.  This is the AS-SHIPPED
#:                 behaviour and it is also the negative control for the
#:                 fix -- the shift sequence must differ from resident.
#: ``"read-only"`` the reads are projected but the runner's RESET is not
#:                 pushed back, so the UH window grows from t=0 instead of
#:                 from the last evaluation.  The narrower control: it is
#:                 the half of the defect that is easiest to fix by
#:                 accident and hardest to see, because the plane it
#:                 produces is real data, just over the wrong window.
#: ``"full"``      the fix.
PROJECTIONS = ("none", "read-only", "full")



def wait_for_vram(min_gib: float = 8.0, timeout_s: float = 5400.0,
                  poll_s: float = 30.0) -> float:
    """Block until the card has ``min_gib`` free, or give up loudly.

    These boxes are shared with other agents, and a leg that OOMs halfway
    through is not a measurement -- it is a missing row that a reader could
    mistake for a refusal.  ``cupy.cuda.runtime.memGetInfo`` is the number
    trusted here; ``nvidia-smi``'s used/free is unreliable under WSL2 and
    tells you nothing about another tenant's allocator anyway.
    """
    import cupy as cp

    deadline = time.time() + float(timeout_s)
    while True:
        try:
            free, total = cp.cuda.runtime.memGetInfo()
        except Exception:                              # noqa: BLE001
            free, total = 0, 1
        gib = free / 2 ** 30
        if gib >= float(min_gib):
            return gib
        if time.time() > deadline:
            raise RuntimeError(
                f"waited {timeout_s:.0f} s for {min_gib} GiB of VRAM and "
                f"only {gib:.1f} GiB of {total / 2**30:.1f} came free; the "
                "card is in use by another tenant")
        time.sleep(float(poll_s))


def retry_on_oom(fn, *args, tries: int = 12, need_gib=None, **kwargs):
    """Run ``fn``, and on an out-of-memory death wait and run it again.

    These cards are shared, and a co-tenant can take several GiB between
    :func:`wait_for_vram` returning and the first allocation landing.  The
    retry exists so a contended box produces a MISSING-FOR-NOW row rather
    than a FAILED row -- a distinction that matters, because "streamed-host
    raised OutOfMemoryError" reads like a property of streaming and is a
    property of the neighbour.
    """
    import gc

    import cupy as cp

    last = None
    for attempt in range(int(tries)):
        try:
            wait_for_vram(VRAM_NEEDED_GIB if need_gib is None else need_gib)
            return fn(*args, **kwargs)
        except (cp.cuda.memory.OutOfMemoryError,
                cp.cuda.runtime.CUDARuntimeError,
                cp.cuda.driver.CUDADriverError) as err:
            last = err
            print(f"  (attempt {attempt + 1}/{tries} died on VRAM: "
                  f"{type(err).__name__}; freeing and waiting)")
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            time.sleep(60.0)
    raise last


def run_leg(mode: str, *, threshold: float, tile=None, staging="device",
            projection: str = "full",
            steps=PARENT_STEPS, verbose=True) -> dict:
    """One complete two-domain relocation run in one execution mode.

    ``mode`` is ``"resident"``, ``"streamed-device"`` or ``"streamed-host"``.
    ``projection`` selects how much of :data:`PROJECTIONS` the executor is
    allowed to do; it is applied by DISABLING methods on the live
    ``StreamedDomain`` rather than by a flag inside the executor, so the
    production path has no test-only branch in it.
    """
    import cupy as cp

    from woof.core.model import execute_experiment

    tile = TILE if tile is None else int(tile)
    wait_for_vram(VRAM_NEEDED_GIB)
    pcfg = parent_cfg()
    ccfg = child_cfg(pcfg)
    bnd = domain_boundaries(pcfg)
    parent_state, geo = build_parent(pcfg, boundaries=bnd)
    exp = experiment(pcfg, ccfg, threshold=threshold)
    model = build_tree(exp, parent_state, geo.grid)
    runner = make_runner(exp, model, staging=staging)

    if projection not in PROJECTIONS:
        raise ValueError(f"projection must be one of {PROJECTIONS}")
    steppers = {}
    streamed = None
    if mode != "resident":
        store = "device" if mode.endswith("device") else "host"
        steppers = stream_parent(model, geo, bnd, store=store, tile=tile)
        streamed = steppers[1]
        if projection in ("none", "read-only"):
            streamed.sync_from_state = lambda *a, **k: 0
        if projection == "none":
            streamed.sync_to_state = lambda *a, **k: 0

    # A history handler is not decoration.  ``refl_10cm_due`` is
    # ``history_handler is not None and clock.history_rings_within_step()``
    # (model.py:866-874), and the tracker's UH path falls back to the
    # reflectivity plane whenever the UH window shows nothing at all
    # (storm_tracking.py:537-542).  With no handler that slot is never
    # allocated, so a no-signal evaluation raises ``TrackerRefusal``
    # instead of holding -- which would turn every stale-plane leg into a
    # crash and hide the shift sequence this lane exists to compare.
    from woof.verify.cases.nest_ideal_common import (
        consume_history_reflectivity)

    refl_refusals: list = []

    def history(_model, node, ticks):
        # Called, never skipped, and its refusal RECORDED: the REFL_10CM
        # producer/consumer handoff is a second, independent thing streaming
        # breaks.  ``refl_10cm_due`` reaches every tile of the sweep and
        # each tile stashes the field on ITS OWN driver
        # (woof/core/refl.py:565-583), so the DOMAIN's driver has nothing
        # and the frame's consumer refuses by name.  Recording it rather
        # than skipping it is the difference between a measured gap and an
        # assumed one.
        try:
            consume_history_reflectivity(node, ticks)
        except RuntimeError as err:
            refl_refusals.append(
                {"grid_id": int(node.cfg.grid_id), "ticks": int(ticks),
                 "message": str(err)})

    t0 = time.perf_counter()
    execution = execute_experiment(
        model, relocation_runner=runner, steppers=steppers,
        history_handler=history,
        validate_state=True, pool_trim_per_period=False)
    cp.cuda.runtime.deviceSynchronize()
    wall = time.perf_counter() - t0
    runner.close_receipt(model)

    d01_source = streamed.store if streamed is not None else model.root.state
    d01_sha, d01_per = carrier_digest(d01_source)
    d02_sha, d02_per = carrier_digest(model.node(2).state)
    out = {
        "mode": mode,
        "projection": (None if streamed is None else projection),
        "tile": None if streamed is None else tile,
        "shifts": shift_sequence(runner),
        "holds": hold_reasons(runner),
        "placements": placements(runner),
        "moves_executed": int(runner.moves_executed),
        "d01_sha256": d01_sha,
        "d02_sha256": d02_sha,
        "d01_fields": d01_per,
        "d02_fields": d02_per,
        "cadence": cadence_census(model, steppers),
        "steps": int(execution.steps),
        "forces": int(execution.forces),
        "wall_s": round(wall, 2),
        "final_placement": [int(model.node(2).cfg.i_parent_start),
                            int(model.node(2).cfg.j_parent_start)],
        "uh_state_max": float(uh_plane(model.root.state).max(initial=0.0)),
        "tracker_fields_used": tracker_fields(runner),
        "centroids": centroids(runner),
        "refl_handoff_refusals": len(refl_refusals),
        "refl_handoff_message": (refl_refusals[0]["message"]
                                 if refl_refusals else None),
    }
    if streamed is not None:
        store_uh = streamed.store.get("scratch/uh_follow_window")
        out["uh_store_max"] = (None if store_uh is None
                               else float(test_join._as_numpy(store_uh).max()))
        out["store_has_follow_window"] = store_uh is not None
    del model, runner, steppers, streamed, parent_state
    import gc

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    if verbose:
        print(json.dumps({k: v for k, v in out.items()
                          if not k.endswith("_fields")}, indent=2))
    return out


# --------------------------------------------------------------------------
# the isolating control: is THIS parent bit-exact streamed, on its own?
# --------------------------------------------------------------------------

def parent_bitexact(steps=8, *, refl_due_every=0, store="host",
                    tile=None, verbose=True) -> dict:
    """One domain, no nest, no relocation: resident vs streamed, bit for bit.

    THE CONTROL THAT DECIDES WHAT A d01 DIFFERENCE MEANS.  The two-domain
    legs compare a streamed parent against a resident one inside a tree that
    also moves a nest, and with ``feedback = 0`` the parent's trajectory does
    not depend on the nest at all -- so a d01 difference there is either the
    moving nest reaching the parent (which would be a defect in relocation)
    or the parent simply not being bit-exact streamed in THIS configuration
    (which would be a defect in streaming, and nothing to do with this
    feature).  Those are different findings and only this run separates them.

    Two things here that the join gate never carried, either of which could
    be the difference:

    ``nwp_diagnostics = 1``   the UP_HELI_MAX lane.  Its accumulator is a
                              carrier, its two tracker windows are carriers
                              only after this branch, and its kernel does a
                              boundary MIRROR (``uh[:, 0] = uh[:, 1]``,
                              uh_diag.py:427-428) at the edges of whatever
                              window it is handed -- which for a tile is the
                              tile's edge, not the domain's.
    ``refl_10cm_due``         the history handshake, forwarded to every tile
                              of the sweep.

    ``refl_due_every`` = 0 never raises the flag; a positive value raises it
    every k steps on BOTH sides, so the flag is either in or out of both runs
    and can be attributed.
    """
    from woof.core.dycore import step as dycore_step

    import cupy as cp

    tile = TILE if tile is None else int(tile)
    wait_for_vram(VRAM_NEEDED_GIB)
    pcfg = parent_cfg()
    bnd = domain_boundaries(pcfg)

    def due(n):
        return bool(refl_due_every) and (n + 1) % int(refl_due_every) == 0

    ref_state, _geo = build_parent(pcfg, boundaries=bnd)
    for n in range(int(steps)):
        dycore_step(ref_state, pcfg, refl_10cm_due=due(n))
        if due(n):
            from woof.core.refl import consume_refl_10cm
            consume_refl_10cm(ref_state)
    cp.cuda.runtime.deviceSynchronize()
    ref = {k: test_join._as_numpy(v)
           for k, v in physinv.carrier_inventory(ref_state).items()}
    ref_counts = dict(ref_state.physics.call_counts)
    del ref_state
    import gc

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    state, geo = build_parent(pcfg, boundaries=bnd)
    options = streaming.StreamingOptions(
        mode="on", tile_nx=tile, tile_ny=tile, nbuffers=NBUFFERS, store=store)
    decision = streaming.decide(pcfg, options)
    stepper = streaming.make_stepper(
        state, pcfg, options, decision=decision,
        build=parent_builder(state, geo, bnd))
    refl_errors = []
    for n in range(int(steps)):
        try:
            stepper(state, pcfg, refl_10cm_due=due(n))
        except RuntimeError as err:
            refl_errors.append(str(err))
            raise
        if due(n):
            # The stash contract both sides are under: a due step's field
            # is consumed before the next due step, which is what every
            # history call site does.  The RESIDENT arm above already
            # consumes; without this the STREAMED arm died at its second
            # due step on the stash-reuse guard -- an asymmetry in the
            # CONTROL, not in the transport it isolates.
            from woof.core.refl import consume_refl_10cm
            consume_refl_10cm(state)
    cp.cuda.runtime.deviceSynchronize()
    got = {k: test_join._as_numpy(v) for k, v in stepper.store.items()}
    cmp = test_join.compare(ref, got)
    out = {"steps": int(steps), "store": store, "tile": tile,
           "refl_due_every": int(refl_due_every),
           "resident_call_counts": ref_counts,
           "streamed_call_counts": dict(
               (stepper.scalars or {}).get("call_counts", {})),
           "bitexact": bool(cmp["bitexact"]),
           "ndiff": int(cmp["ndiff"]), "ntotal": int(cmp["ntotal"]),
           "differing": cmp["differing"], "max_abs": float(cmp["max_abs"]),
           "all_differing": sorted(
               n for n in ref
               if test_join.digest_arrays({n: ref[n]})[n]
               != test_join.digest_arrays({n: got[n]})[n]) if not cmp[
                   "bitexact"] else []}
    del state, stepper, ref, got
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    if verbose:
        print(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------
# calibration: what threshold makes the tracker fire?
# --------------------------------------------------------------------------

def probe(steps=PARENT_STEPS, verbose=True) -> dict:
    """Step the parent alone and report the UH window it actually produces.

    A threshold is a physical number and this harness's air is not the
    atmosphere, so it is MEASURED here rather than guessed: a threshold
    above the plane's maximum makes every evaluation a no-signal hold and
    the whole comparison degenerates to comparing two empty sequences.
    """
    from woof.core.dycore import step
    from woof.core.uh_diag import UH_FOLLOW_WINDOW_SLOT

    wait_for_vram(VRAM_NEEDED_GIB)
    pcfg = parent_cfg()
    bnd = domain_boundaries(pcfg)
    state, _geo = build_parent(pcfg, boundaries=bnd)
    samples = []
    for n in range(int(steps)):
        step(state, pcfg, refl_10cm_due=False)
        if (n + 1) % 5 == 0:
            plane = uh_plane(state, UH_FOLLOW_WINDOW_SLOT)
            samples.append({
                "step": n + 1,
                "max": float(plane.max()),
                "p99": float(np.percentile(plane, 99.0)),
                "p999": float(np.percentile(plane, 99.9)),
                "cells_above_p99": int((plane >= np.percentile(
                    plane, 99.0)).sum()),
            })
    out = {"samples": samples,
           "up_heli_max_max": float(uh_plane(state, "up_heli_max").max())}
    if verbose:
        print(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------
# the state-swap control
# --------------------------------------------------------------------------

def state_swap_control() -> dict:
    """A StreamedDomain stepped after something replaced its state.

    This is the guaranteed hard stop the feature map predicted, isolated so
    the message can be quoted rather than paraphrased.  It does not need a
    relocation to produce it -- a relocation is simply the operation that
    hands back a different object -- so it is exercised here with the same
    substitution relocation performs.
    """
    from woof.core.streaming import StreamingRefused

    pcfg = parent_cfg(nx=128, ny=128)
    bnd = domain_boundaries(pcfg)
    state, geo = build_parent(pcfg, boundaries=bnd)
    options = streaming.StreamingOptions(mode="on", tile_nx=64, tile_ny=64,
                                         nbuffers=2, store="host")
    decision = streaming.decide(pcfg, options)
    stepper = streaming.make_stepper(
        state, pcfg, options, decision=decision,
        build=parent_builder(state, geo, bnd))
    stepper(state, pcfg, refl_10cm_due=False)
    other, _geo2 = build_parent(pcfg, boundaries=bnd, seed=SEED + 7, warmup=0)
    try:
        stepper(other, pcfg, refl_10cm_due=False)
    except StreamingRefused as err:
        return {"refused": True, "message": str(err)}
    return {"refused": False, "message": None}


def route_control() -> dict:
    """What a production route does when ``[tiles] mode = 'on'``.

    ``woof.prepared_domain_tree_forecast`` and
    ``woof.prepared_single_domain_forecast`` both call
    ``streaming.steppers_for_tree(model, exp.tiles)`` with NO builders.
    """
    from types import SimpleNamespace

    from woof.core.streaming import StreamingRefused

    cfg = parent_cfg(nx=128, ny=128)
    node = SimpleNamespace(cfg=SimpleNamespace(grid_id=1, run=cfg),
                           state=object())
    model = SimpleNamespace(walk_parent_first=lambda: iter([node]))
    options = streaming.StreamingOptions(mode="on", tile_nx=64, tile_ny=64)
    try:
        streaming.steppers_for_tree(model, options)
    except StreamingRefused as err:
        return {"refused": True, "message": str(err)}
    return {"refused": False, "message": None}


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------

def _line(label: str, ok: bool, detail: str = "") -> str:
    return f"  {'PASS' if ok else 'FAIL':4s}  {label:56s} {detail}"


def main(argv=None) -> int:
    import cupy as cp

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--steps", type=int, default=PARENT_STEPS)
    parser.add_argument("--parent", type=int, default=None,
                        help="d01 nx=ny (the tile must divide it)")
    parser.add_argument("--child", type=int, default=None)
    parser.add_argument("--tile", type=int, default=None)
    parser.add_argument("--vram-gib", type=float, default=None,
                        help="free VRAM a leg waits for; these cards are "
                             "shared and the default is sized for the "
                             "default domain")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    global PARENT_NX, PARENT_NY, CHILD_NX, CHILD_NY, TILE, \
        I_PARENT_START, J_PARENT_START, VRAM_NEEDED_GIB
    if args.parent:
        PARENT_NX = PARENT_NY = int(args.parent)
        I_PARENT_START = J_PARENT_START = int(args.parent) // 2 - 12
    if args.child:
        CHILD_NX = CHILD_NY = int(args.child)
        I_PARENT_START = J_PARENT_START = (
            PARENT_NX // 2 - int(args.child) // (2 * RATIO))
    if args.tile:
        TILE = int(args.tile)
    if args.vram_gib:
        VRAM_NEEDED_GIB = float(args.vram_gib)
    if PARENT_NX % TILE:
        raise SystemExit(
            f"tile {TILE} does not divide the domain {PARENT_NX}; a ragged "
            "trailing tile is read right through in ring mode (22.7% of the "
            "store instead of 2.4% in one measured case)")

    # The card can be so full that CONTEXT CREATION fails, which is what
    # memGetInfo raising cudaErrorMemoryAllocation means; wait it out rather
    # than reporting a co-tenant as a result.
    while True:
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            break
        except Exception as err:                       # noqa: BLE001
            print(f"waiting for a CUDA context ({type(err).__name__})")
            time.sleep(60.0)
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    print(f"parent {PARENT_NX}x{PARENT_NY}x{NZ} dx={parent_cfg().dx/1000} km "
          f"dt={PARENT_DT} s  child {CHILD_NX}x{CHILD_NY} ratio {RATIO}  "
          f"tile {TILE}  halo {harness.halo_radius(parent_cfg())}")
    print()

    if args.probe:
        probe(steps=args.steps)
        return 0

    report: dict = {}
    print("CONTROLS")
    report["route"] = route_control()
    print(_line("a production route refuses [tiles] mode = 'on'",
                report["route"]["refused"]))
    print(f"        {report['route']['message']}")
    report["state_swap"] = retry_on_oom(state_swap_control,
                                        need_gib=VRAM_NEEDED_GIB)
    print(_line("a streamed domain refuses a swapped state",
                report["state_swap"]["refused"]))
    print(f"        {report['state_swap']['message']}")
    print()
    if args.controls_only:
        if args.json:
            open(args.json, "w").write(json.dumps(report, indent=2))
        return 0

    threshold = args.threshold
    if threshold is None:
        got = retry_on_oom(probe, steps=args.steps, verbose=False)
        threshold = float(got["samples"][-1]["p999"])
        print(f"UH threshold from the probe: {threshold:.4f} m2 s-2 "
              f"(p99.9 of the window after {args.steps} steps)")
        report["probe"] = got
    report["threshold"] = threshold
    print()

    legs = {}
    plan = [
        ("resident", "resident", "full"),
        # AS SHIPPED: the executor projects nothing, so every whole-domain
        # consumer reads the state ``attach`` left behind.
        ("streamed-host/AS-SHIPPED", "streamed-host", "none"),
        # The narrower control: reads projected, the RESET not pushed back.
        ("streamed-host/no-reset", "streamed-host", "read-only"),
        # The fix, on the transport-free store first so a failure there is
        # a tiling failure and cannot be a copy.
        ("streamed-device", "streamed-device", "full"),
        ("streamed-host", "streamed-host", "full"),
    ]
    for label, mode, projection in plan:
        print(f"--- {label} ---")
        try:
            legs[label] = retry_on_oom(
                run_leg, mode, threshold=threshold,
                projection=projection, steps=args.steps)
        except Exception as err:                       # noqa: BLE001
            legs[label] = {"error": f"{type(err).__name__}: {err}"}
            print(f"  RAISED  {type(err).__name__}: {err}")
        print()
    report["legs"] = legs

    ref = legs["resident"]
    print("VERDICT")
    for label, _mode, _proj in plan[1:]:
        got = legs[label]
        if "error" in got:
            print(_line(f"{label}: completed", False, got["error"]))
            continue
        same_shift = (got["shifts"] == ref["shifts"]
                      and got["centroids"] == ref["centroids"])
        same_d01 = got["d01_sha256"] == ref["d01_sha256"]
        same_d02 = got["d02_sha256"] == ref["d02_sha256"]
        must_match = got["projection"] == "full"
        print(_line(f"{label}: shift+centroid sequence "
                    f"{'==' if must_match else '!='} resident",
                    same_shift is must_match,
                    f"{got['shifts']} vs {ref['shifts']}"))
        print(f"        centroids {got['centroids']}")
        print(_line(f"{label}: d01 carriers == resident", same_d01))
        print(_line(f"{label}: d02 carriers == resident", same_d02))
        if not same_d01:
            diff = sorted(n for n in ref["d01_fields"]
                          if ref["d01_fields"].get(n)
                          != got["d01_fields"].get(n))
            print(f"        d01 differs in {len(diff)}/"
                  f"{len(ref['d01_fields'])}: {diff[:8]}")
        if not same_d02:
            diff = sorted(n for n in ref["d02_fields"]
                          if ref["d02_fields"].get(n)
                          != got["d02_fields"].get(n))
            print(f"        d02 differs in {len(diff)}/"
                  f"{len(ref['d02_fields'])}: {diff[:8]}")
    print()
    print("physics fired, per leg (radiation / cumulus / microphysics must "
          "be nonzero on BOTH sides of every comparison)")
    for label in legs:
        if "error" not in legs[label]:
            print(f"  {label:28s} {legs[label]['cadence']}")
    if args.json:
        open(args.json, "w").write(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
