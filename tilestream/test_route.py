"""THE FRONT DOOR: a forecast that streams because its TOML said so.

``tilestream/test_join.py`` proves that a real forecast configuration is
bit-exact when streamed.  It proves it through a loop of its own -- the
model's loop with the output, restart and nest ops removed -- and through a
builder of its own, defined inside the test.  Neither of those is what a user
runs, and the gap between them was total:

    ``woof/prepared_single_domain_forecast.py`` and
    ``woof/prepared_domain_tree_forecast.py``, the only two routes the CLI
    can launch a forecast through, both called
    ``streaming.steppers_for_tree(model, exp.tiles)`` with NO builders
    argument.  ``make_stepper(..., build=None)`` refuses.  So every
    ``[tiles] mode = "on"`` forecast raised ``StreamingRefused`` and NO
    configuration the front door accepts was capable of streaming.

This module is the gate for closing that.  It builds a WOOF-ArWen
:class:`~woof.core.model.ExperimentState` out of an experiment TOML the
schema validates, hands it to :func:`woof.core.model.execute_experiment` --
the executor both routes call and the only thing between them and the dycore
-- and runs the same experiment twice through the ROUTES' OWN pair of calls::

    steppers = streaming.steppers_for_tree(
        model, exp.tiles,
        builders=streaming.builders_for_tree(model, exp.tiles))

once with ``[tiles]`` absent and once with ``mode = "on"``, then compares
every carrier.

    python -m tilestream.test_route             # the gate
    python -m tilestream.test_route --quick     # 24 steps, controls only
    python -m tilestream.test_route --steps 120

WHAT THIS GATE ADDS OVER test_join, AND WHY EACH ADDITION IS NOT COSMETIC
-------------------------------------------------------------------------
*The builder is the product's, not the test's.*  ``test_join`` defines
``_make_builder`` inside itself; if that function is the only correct one, a
user has no way to reach it.  Here the builder is
:func:`woof.core.streaming.prepared_domain_builder`, reached through
:func:`~woof.core.streaming.builders_for_tree`, which is what the routes now
call.  A defect in the shipped builder fails this gate; a defect in a test's
private builder never could.

*The loop is the executor's.*  ``execute_experiment`` refreshes the state
clock either side of every STEP, runs the health validators, walks the op
table and trims the pool.  A streamed stepper has to survive all of that,
and one of those interactions is a finding this gate MEASURES rather than
assumes -- see THE FROZEN STATE below.

*The cadence is a real forecast's.*  ``radt = 12 min`` and ``cudt = 5 min``
at ``dt = 30 s`` fire radiation every 24 steps and cumulus every 10, so a
window shorter than about 30 steps gives one side a free ride on the two most
expensive schemes in the stack.  This project has produced three false
results that way.  The fire counts are PRINTED on both sides and a run in
which either side shows zero refuses to report a verdict.

THE FROZEN STATE -- measured here, and it is the reason this gate exists
------------------------------------------------------------------------
With ``store = "host"`` the domain's arrays live in pinned host RAM and the
``DomainState`` the route is holding is a SNAPSHOT taken when the stepper was
attached.  Nothing writes back to it.  ``execute_experiment`` reads
``node.state`` for the health validators, and both routes read it again for
the history frames, the stability gate and the final canonical digest.

So this gate digests ``node.state`` before and after the streamed run as a
first-class result.  It does not assert that they differ or that they agree:
it prints which, because that single fact decides whether the mode is usable
through the front door or only through this gate.

THE CONTROLS
------------
``no builder``          the routes' call without ``builders`` -> must REFUSE
``auto on a fit``       ``mode = "auto"`` on a domain that fits -> must NOT
                        stream, and a gate that reported PASS from that run
                        would be reporting a resident run
``geography rebuilt``   each buffer rebuilds geography from its tile_cfg ->
                        must DIFFER
``tables unwindowed``   every tile gets tile 0's forcing -> must DIFFER
``the clock dropped``   the domain clock is not carried -> must DIFFER
``cadence fired``       radiation and cumulus fire counts, both sides,
                        printed; zero on either side refuses the verdict
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import sys
import time
import warnings

import numpy as np

from woof.core import streaming
from tilestream import harness
from tilestream import physics_inventory as physinv

# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------

#: 192x144 at 12 km is a 2304 x 1728 km regional footprint, split 4x3 into
#: 48x48 tiles.  The tile DIVIDES the domain in both axes, which is the only
#: tiling a forecast should ever run: a ragged trailing tile is read right
#: through in ring mode (22.7% of the store instead of 2.4% in one measured
#: case).  The plan still contains tiles of all three kinds -- 4 corners own
#: two true domain edges, 6 sides own one, 2 own none at all -- so the
#: true-edge/interior-seam split this mode rests on is exercised rather than
#: assumed.  A 64-cell compute window is far below the ~500 cells a TIMING
#: measurement needs; nothing here is a timing.
NX, NY, NZ = 192, 144, 49
TX, TY = 48, 48

#: 12 km's own step is 60 s, and the dry rung is clean there.  This gate runs
#: FULL PHYSICS on the harness's seeded state, and test_join MEASURED that
#: state going non-finite at 60 s in the moist rungs (20 non-finite cells at
#: mp10, 12 at full+MYNN+Noah-MP) while being clean at 30 s and below.  That
#: is a property of the initial condition, not of streaming -- it is the
#: MONOLITHIC run that blows up -- and a reference that is not finite has
#: nothing to compare against.
DT = 30.0

#: 120 steps = 3600 s = one forecast hour, and at radt = 12 min / cudt =
#: 5 min that is 5 radiation fires and 12 cumulus fires.  The counts are
#: printed rather than trusted.
STEPS = 120
QUICK_STEPS = 24

SEED = 20_260_809
#: One step before the compared window opens, for the same reason a tile
#: buffer takes one: Kain-Fritsch's ``cumulus/w0avg`` is allocated lazily on
#: first use, so a domain state that has never stepped is MISSING a carrier
#: the tile buffers hold and the inventory comparison would refuse the run.
WARMUP = 1

#: The forcing series has to span every ``elapsed_seconds`` the run reaches:
#: ``LateralBoundaries.interval_at`` raises outside it, and ``dtbc`` -- the
#: interpolation weight the clock carrier feeds -- is ``elapsed - start``.
BDY_SECONDS = 21600.0


def raw_experiment(streaming_table: dict | None = None) -> dict:
    """One experiment TOML, as a parsed dict, that the schema validates.

    The physics is ``configs/real74_d01.toml``'s full stack -- Morrison
    mp10, MM5 sfclay, YSU, Noah, RRTMGP and Kain-Fritsch, 154 carriers at
    229.29 B/cell -- on the Lambert projection ``tilestream.harness`` builds
    its geography from, so the state below and the tree here describe the
    same grid.
    """
    raw: dict = {
        "experiment": {
            "name": "streaming-route-gate",
            "start_time": datetime(1974, 4, 3, 12, 0, 0),
            "run_seconds": 3600.0,
            "feedback": 0, "smooth_option": 0, "blend_width": 5,
            "spec_bdy_width": 5, "restart_interval_s": 0.0,
        },
        "projection": {
            "map_proj": "lambert",
            "ref_lat": harness.REAL74_PROJECTION["ref_lat"],
            "ref_lon": harness.REAL74_PROJECTION["ref_lon"],
            "truelat1": harness.REAL74_PROJECTION["truelat1"],
            "truelat2": harness.REAL74_PROJECTION["truelat2"],
            "stand_lon": harness.REAL74_PROJECTION["stand_lon"],
        },
        "shared": {
            "nz": NZ, "ztop": 20000.0, "hybrid_opt": 2, "etac": 0.2,
            "base_temp": 290.0, "time_step_sound": 4,
            "terrain_opt": 1, "map_proj": 1,
            "spec_zone": 1, "relax_zone": 4, "spec_exp": 0.0,
            "open_x": False, "open_y": False,
            "sf_surface_physics": 2,
            "ra_lw_physics": 4, "ra_sw_physics": 4,
            "damp_opt": 3, "zdamp": 5000.0, "dampcoef": 0.2,
            "w_damping": 1, "diff_6th_opt": 2,
        },
        "domain": [{
            "grid_id": 1, "parent_id": 0,
            "i_parent_start": 1, "j_parent_start": 1,
            "parent_grid_ratio": 1, "parent_time_step_ratio": 1,
            "nx": NX, "ny": NY, "dx": 12000.0, "dy": 12000.0,
            "time_step": int(DT),
            "specified": True, "nested": False,
            "history_interval_s": 900.0,
            "moist": True, "mp_physics": 10,
            "km_opt": 4, "sf_sfclay_physics": 91,
            "bl_pbl_physics": 1, "bldt": 0.0,
            "cu_physics": 1, "cudt_minutes": 5.0,
            "radt_minutes": 12.0,
            "diff_6th_factor": 0.12, "epssm": 0.5,
        }],
    }
    if streaming_table is not None:
        raw["streaming"] = dict(streaming_table)
    return raw


def build_experiment_config(streaming_table: dict | None = None):
    from woof.experiment import build_experiment

    return build_experiment(raw_experiment(streaming_table),
                            source="<tilestream.test_route>")


# --------------------------------------------------------------------------
# the prepared domain, and the tree ArWen's executor walks
# --------------------------------------------------------------------------

def domain_boundaries(cfg, state_a, state_b):
    """The DOMAIN's specified forcing, from two genuinely different states.

    Two seeds rather than one snapshot repeated, so the time tendency is
    NONZERO and ``dtbc`` -- and therefore ``elapsed_seconds`` -- actually
    reaches the answer.  A zero tendency would quietly disarm the clock
    control, which would then pass on a build that never carried the clock.
    """
    from woof.ingest.lateral_bc import build_state_lateral_boundaries

    return build_state_lateral_boundaries(
        [state_a, state_b], [0.0, BDY_SECONDS],
        spec_bdy_width=int(cfg.spec_bdy_width),
        spec_zone=int(cfg.spec_zone), relax_zone=int(cfg.relax_zone))


def prepared_state(cfg, start_time, *, seed: int = SEED):
    """A prepared, resident domain: real Lambert, real terrain, real LBCs.

    ``periodic_faces=False`` is not a detail.  On a SPECIFIED domain column
    ``nx`` is a real east boundary face ``dx*nx`` away from column 0, and
    duplicating the closing map-factor face -- ``make_geography``'s default,
    which a periodic domain requires -- would install the WEST edge's map
    factor on the EAST edge.
    """
    from woof.ingest.lateral_bc import attach_lateral_boundaries

    geo = harness.make_geography(cfg, terrain=True, periodic_faces=False)
    state, _drv = harness.make_physics_state(cfg, seed, geography=geo,
                                             start_time=start_time)
    other, _drv2 = harness.make_physics_state(cfg, seed + 1, geography=geo,
                                              start_time=start_time)
    bnd = domain_boundaries(cfg, state, other)
    del other
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()

    attach_lateral_boundaries(state, bnd)
    if WARMUP:
        harness.run_steps(state, cfg, WARMUP)
    # The warmup advanced the state clock; the executor's clock starts at
    # zero and ``refresh_model_time`` will impose that on the RESIDENT state
    # before its first step.  A streamed domain's clock is the ``scalars``
    # dict ``attach`` captures HERE, so the two would start the compared
    # window one step apart unless this is reset -- and one step of dtbc and
    # of physics cadence apart is exactly the difference the clock control
    # is built to detect.  Reset once, before either run, so both start at
    # the same instant.
    state.elapsed_seconds = 0.0
    return state, bnd


def build_model(exp, state):
    """An ``ExperimentState`` around one prepared domain, as a route builds it."""
    from types import SimpleNamespace

    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.model import (DomainNode, ExperimentState,
                                  ModelRuntimeStatus)
    from woof.static.lambert import grids_from_projection_config

    clock = resolve_clock(exp)
    schedule = build_schedule(exp, clock)
    grids = grids_from_projection_config(exp)
    root = DomainNode(exp.domains[0], grids[0], state, clock.clocks()[1],
                      None, [], None)
    model = ExperimentState(root, {1: root}, schedule, None,
                            "tilestream-route-gate")
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    return model


# --------------------------------------------------------------------------
# digests
# --------------------------------------------------------------------------

def _host(value):
    import cupy as cp

    return (cp.asnumpy(value) if isinstance(value, cp.ndarray)
            else np.asarray(value))


def digests(arrays) -> dict[str, str]:
    return physinv.field_digests({k: _host(v) for k, v in arrays.items()})


def compare(ref: dict, got: dict) -> dict:
    """Carrier by carrier, member for member, plus the worst difference."""
    da, db = digests(ref), digests(got)
    missing = sorted(set(da) ^ set(db))
    differing = sorted(n for n in da if n in db and da[n] != db[n])
    worst = 0.0
    for name in differing:
        a, b = _host(ref[name]), _host(got[name])
        if a.dtype.kind == "f" and a.shape == b.shape:
            worst = max(worst, float(np.nanmax(np.abs(
                a.astype(np.float64) - b.astype(np.float64)))))
    nonfinite = sum(int(np.count_nonzero(~np.isfinite(_host(v))))
                    for v in got.values() if _host(v).dtype.kind == "f")
    return dict(bitexact=(not differing and not missing),
                ncarriers=len(da), ndiff=len(differing),
                missing=missing[:8], differing=differing[:8],
                max_abs=worst, nonfinite=nonfinite)


# --------------------------------------------------------------------------
# the two runs, both through execute_experiment
# --------------------------------------------------------------------------

class Result(dict):
    """One run's carriers, its fire counts and what its state looked like."""


def run(exp, *, steps: int, streaming_table=None, builders=None,
        validate_state: bool = True, refresh_history: bool = False) -> Result:
    """One ``execute_experiment`` over ``steps`` model steps.

    ``streaming_table=None`` is the RESIDENT control and it is a control in
    the strong sense: ``steppers_for_tree`` returns ``{}``, the executor
    binds ``woof.core.dycore.step`` itself, and there is no wrapper in the
    call at all.

    ``refresh_history`` installs a ``history_handler`` that calls
    :meth:`woof.core.streaming.StreamedDomain.refresh_state` before it
    reads ``node.state`` -- what a route publishing wrfout frames from a
    streamed domain has to do.  The frames it digests are compared against
    the resident run's at the same tick, so this MEASURES whether the
    mechanism closes the gap rather than asserting that it would.
    """
    import cupy as cp

    from woof.core.model import execute_experiment

    exp_run = replace(exp, run_seconds=float(steps) * DT)
    state, _bnd = prepared_state(exp_run.domains[0].run, exp_run.start_time)
    model = build_model(exp_run, state)
    before = digests(physinv.carrier_inventory(state, None))

    options = (streaming.OFF if streaming_table is None
               else streaming.StreamingOptions(**streaming_table))
    # THE ROUTES' OWN PAIR OF CALLS, verbatim.
    if builders is None:
        builders = streaming.builders_for_tree(model, options)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        steppers = streaming.steppers_for_tree(model, options,
                                               builders=builders)
    warned = [w for w in caught if "will NOT advance" in str(w.message)]

    streamed = steppers.get(1)
    frames: list[tuple[int, str]] = []
    handler = None
    if refresh_history:
        import hashlib

        from woof.core.refl import consume_refl_10cm, refl_10cm_is_stashed

        def handler(_model, node, ticks):
            if streamed is not None:
                streamed.refresh_state(node.state)
            # The route's own REFL_10CM handshake
            # (``_consume_due_native_refl_10cm``).  A frame that does not
            # consume the stash makes the NEXT output-due microphysics call
            # raise "REFL_10CM stash was not consumed before reuse", which
            # is a producer/consumer contract and not a nicety.
            refl = (consume_refl_10cm(node.state)
                    if refl_10cm_is_stashed(node.state) else None)
            rolled = hashlib.sha256()
            for key, value in sorted(digests(
                    physinv.carrier_inventory(node.state, None)).items()):
                rolled.update(key.encode())
                rolled.update(value.encode())
            frames.append((int(ticks), rolled.hexdigest()[:16],
                           refl is not None))

    t0 = time.perf_counter()
    report = execute_experiment(model, history_handler=handler,
                                progress_callback=None,
                                validate_state=validate_state,
                                skip_feedback_path=True,
                                pool_trim_per_period=True,
                                steppers=steppers)
    cp.cuda.runtime.deviceSynchronize()
    wall = time.perf_counter() - t0

    if streamed is None:
        carriers = {k: _host(v) for k, v in
                    physinv.carrier_inventory(state, None).items()}
        counts = dict(state.physics.call_counts)
        elapsed = float(state.elapsed_seconds)
    else:
        carriers = {k: _host(v) for k, v in streamed.store.items()}
        # ``scalars`` is None for the CLOCK CONTROL -- attach(scalars=None)
        # means CARRY NOTHING -- so this must survive it, or the control
        # dies in the harness before it can be compared and reads as a
        # crash rather than as the difference it is there to produce.
        counts = dict((streamed.scalars or {}).get("call_counts", {}))
        elapsed = float(streamed.elapsed_seconds)
    after = digests(physinv.carrier_inventory(state, None))
    refreshed = None
    if streamed is not None:
        # What a route must do before its FINAL read of the state --
        # canonical_state_digest and the final health validation.
        ncopied = streamed.refresh_state(state)
        refreshed = (ncopied, digests(physinv.carrier_inventory(state, None)))
    out = Result(carriers=carriers, call_counts=counts, wall=wall,
                 steps=int(report.steps), model_elapsed=elapsed,
                 streamed=streamed is not None, frames=frames,
                 state_moved=sorted(n for n in before
                                    if before[n] != after.get(n)),
                 state_carriers=len(before), refreshed=refreshed,
                 warned=bool(warned),
                 decision=(None if streamed is None
                           else streamed.decision.explain()))
    del model, state
    cp.get_default_memory_pool().free_all_blocks()
    return out


# --------------------------------------------------------------------------
# the controls
# --------------------------------------------------------------------------

def control_no_builder(exp) -> tuple[bool, str]:
    """The routes' call WITHOUT builders must refuse.  This is the bug."""
    state, _bnd = prepared_state(exp.domains[0].run, exp.start_time)
    model = build_model(replace(exp, run_seconds=DT), state)
    options = streaming.StreamingOptions(mode="on", tile_nx=TX, tile_ny=TY)
    try:
        streaming.steppers_for_tree(model, options)      # builders unset
    except streaming.StreamingRefused as exc:
        ok = "wired no streamed-domain builder" in str(exc)
        detail = str(exc).split(".")[0][-60:]
    else:
        ok, detail = False, "DID NOT REFUSE"
    import cupy as cp
    del model, state
    cp.get_default_memory_pool().free_all_blocks()
    return ok, detail


def control_auto_does_not_fire(exp) -> tuple[bool, str]:
    """``mode = "auto"`` on a domain that fits must NOT stream.

    Stated as a control because the failure it guards is a PASSING test: an
    ``auto`` run that silently returned ``dycore.step`` would compare a
    resident run against a resident run and report bit-exactness.
    """
    from woof.core.dycore import step as dycore_step

    cfg = exp.domains[0].run
    # An explicit budget rather than ``Machine.detect()``: this control is
    # about the DECISION, and detect reads FREE VRAM, so on a card another
    # tenant is already sitting on it measures the neighbour rather than
    # the logic.  (MEASURED: with 3.38 GiB free the planner refuses this
    # domain outright, which is correct and is not what this asks.)
    stepper = streaming.make_stepper(
        object(), cfg, streaming.StreamingOptions(
            mode="auto", vram_budget_bytes=24 * 2 ** 30), build=None)
    ok = stepper is dycore_step
    return ok, ("auto returned dycore.step itself" if ok
                else "auto STREAMED a domain that fits")


def control_auto_records_that_it_declined(exp) -> tuple[bool, str]:
    """``auto`` that ran resident must SAY so, in the run's own receipt.

    The control above proves ``auto`` correctly declines to stream a domain
    that fits.  This one proves the operator can TELL -- which is a separate
    property, and the one that was missing.  ``steppers_for_tree`` returns
    ``{}`` for a declined ``auto``, and ``{}`` is also what ``off`` returns
    and what a config with no ``[tiles]`` table returns, so the run that
    declined and the run that was never asked were the same run as far as
    anything downstream could see.

    That is the exact shape of a false pass this project has produced
    before: point a streaming test at ``auto``, watch it go green, and
    conclude the mode works when what was compared is a resident run against
    a resident run.
    """
    state, _bnd = prepared_state(exp.domains[0].run, exp.start_time)
    model = build_model(replace(exp, run_seconds=DT), state)
    options = streaming.StreamingOptions(mode="auto",
                                         vram_budget_bytes=24 * 2 ** 30)
    decisions: dict = {}
    steppers = streaming.steppers_for_tree(
        model, options, decisions=decisions,
        builders=streaming.builders_for_tree(model, options))
    receipt = streaming.streaming_receipt(options, decisions)

    import cupy as cp
    del model, state
    cp.get_default_memory_pool().free_all_blocks()

    if steppers:
        return False, "auto STREAMED a domain that fits"
    ok = (receipt.get("streamed_any") is False
          and receipt.get("configured_mode") == "auto"
          and "NO domain streamed" in receipt.get("summary", ""))
    return ok, (receipt.get("summary", "NO RECEIPT AT ALL")[:64] if ok
                else f"receipt cannot distinguish this from OFF: {receipt}")


def broken_builder(model, options, *, what: str):
    """A builder with exactly one capability removed.  Each MUST differ."""
    node = model.root
    accurate = streaming.prepared_domain_builder(node)

    def build(state, cfg, decision):
        from woof.ingest.lateral_bc import LateralBoundaries
        from tilestream import driver as _driver

        boundaries = state.lateral_boundaries
        assert isinstance(boundaries, LateralBoundaries)
        tables = streaming.tile_boundary_tables(
            boundaries, streaming.tile_specs(cfg, decision), seam="zeros")
        if what == "tables unwindowed":
            # Every tile gets TILE 0's tables.  Tile 0 owns the west and
            # south domain edges, so this hands the domain's west and south
            # forcing to tiles whose west and south are interior seams
            # hundreds of kilometres away.  The SHAPES are identical, which
            # is the point: no shape check can catch this.
            tables = [tables[0]] * len(tables)
        geography = (None if what == "geography rebuilt"
                     else _driver.geography_store(state, host=True))
        factory = streaming.prepared_tile_state_factory(
            state, cfg, tables0=tables[0])
        if what == "geography rebuilt":
            # Each buffer keeps the geography it was BUILT with.  The
            # shipped factory builds on neutral_geography -- identity map
            # factors, zero Coriolis, flat terrain -- so with the gather
            # removed the buffers integrate a domain with no projection at
            # all.  That is the defect in its purest form.
            pass
        scalars = (None if what == "the clock dropped" else streaming.UNSET)
        return streaming.attach(
            state, cfg, decision, tile_state_factory=factory,
            geography=geography, boundary_tables=tables,
            scalars=scalars, check_geography=False)

    return accurate if what == "accurate" else build


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def _line(label: str, ok: bool, detail: str = "") -> str:
    return f"  {'PASS' if ok else 'FAIL':4s}  {label:32s} {detail}"


def cadence_expectation(steps: int) -> str:
    return (f"radt 12 min / cudt 5 min at dt={DT:g} s fire every "
            f"{int(12 * 60 / DT)} and {int(5 * 60 / DT)} steps; "
            f"{steps} steps expects ~{steps // int(12 * 60 / DT)} and "
            f"~{steps // int(5 * 60 / DT)}")


def main(argv=None) -> int:
    import cupy as cp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--tile", type=int, default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    steps = args.steps or (QUICK_STEPS if args.quick else STEPS)
    global NX, NY, TX, TY
    NX = args.nx or NX
    NY = args.ny or NY
    if args.tile:
        TX = TY = int(args.tile)
    if NX % TX or NY % TY:
        print(f"REFUSED: tile {TX}x{TY} does not divide {NX}x{NY}; a ragged "
              "trailing tile is read right through in ring mode")
        return 2

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    print("=" * 78)
    print("THE FRONT DOOR -- [tiles] mode = 'on' through "
          "execute_experiment,")
    print(f"    {NX}x{NY}x{NZ} at dx=12 km, tile {TX}x{TY}, "
          f"dt={DT:g} s, {steps} steps "
          f"({steps * DT / 3600.0:.2f} forecast hours)")
    print("=" * 78)

    exp = build_experiment_config()
    cfg = exp.domains[0].run
    halo = harness.halo_radius(cfg)
    specs = streaming.tile_specs(
        cfg, streaming.StreamingDecision(True, "gate", TX, TY, 2, halo))
    census = {0: 0, 1: 0, 2: 0}
    for s in specs:
        census[sum(streaming.owned_edges(s).values())] += 1
    print(f"  halo {halo} = 10 + 3*{cfg.time_step_sound}//2, from "
          "harness.halo_radius and never from a measurement")
    print(f"  {len(specs)} tiles, {NX % TX == 0 and NY % TY == 0 and 'EXACT' or 'RAGGED'}"
          f" tiling: {census[0]} own no true edge, {census[1]} own one, "
          f"{census[2]} own two")
    print(f"  {cadence_expectation(steps)}")
    print()

    failures: list[str] = []

    # ---------------------------------------------------------- the controls
    if not args.skip_controls:
        print("-- THE CONTROLS THAT MUST FIRE BEFORE ANYTHING ELSE COUNTS")
        ok, detail = control_no_builder(exp)
        print(_line("no builder -> REFUSED", ok, detail))
        if not ok:
            failures.append("the no-builder refusal did not fire")
        ok, detail = control_auto_does_not_fire(exp)
        print(_line("auto on a fit -> resident", ok, detail))
        if not ok:
            failures.append("auto streamed a domain that fits")
        ok, detail = control_auto_records_that_it_declined(exp)
        print(_line("auto that declined SAYS so", ok, detail))
        if not ok:
            failures.append("a declined auto is indistinguishable from off")
        print()

    # ---------------------------------------------------------- the two runs
    print("-- THE POSITIVE: resident vs streamed, same experiment, same loop")
    resident = run(exp, steps=steps)
    print(f"  resident: {resident['steps']} steps, "
          f"{resident['wall']:.1f} s, "
          f"elapsed {resident['model_elapsed']:.0f} s, "
          f"{len(resident['carriers'])} carriers")
    print(f"            call_counts {resident['call_counts']}")

    table = dict(mode="on", tile_nx=TX, tile_ny=TY, nbuffers=2)
    streamed = run(exp, steps=steps, streaming_table=table)
    print(f"  streamed: {streamed['steps']} steps, "
          f"{streamed['wall']:.1f} s, "
          f"elapsed {streamed['model_elapsed']:.0f} s, "
          f"{len(streamed['carriers'])} carriers")
    print(f"            {streamed['decision']}")
    print(f"            call_counts {streamed['call_counts']}")
    print()

    # ------------------------------------------------- the cadence, both sides
    print("-- THE CADENCE (a window where these are zero measures nothing)")
    cadence_ok = True
    for scheme in ("radiation", "cumulus", "sfclay", "noah", "ysu"):
        a = int(resident["call_counts"].get(scheme, 0))
        b = int(streamed["call_counts"].get(scheme, 0))
        fired = a > 0 and b > 0
        if scheme in ("radiation", "cumulus") and not fired:
            cadence_ok = False
        print(_line(f"{scheme} fired", fired and a == b,
                    f"resident {a}, streamed {b}"))
        if a != b:
            failures.append(f"{scheme} fired {a} times resident and {b} "
                            "streamed")
    if not cadence_ok:
        print("\n  REFUSING TO REPORT A VERDICT: radiation or cumulus never "
              "fired on one side.")
        return 2
    print()

    # --------------------------------------------------- the bit-exactness
    print("-- BIT-EXACTNESS, member for member")
    res = compare(resident["carriers"], streamed["carriers"])
    ok = res["bitexact"] and res["nonfinite"] == 0
    if not ok:
        failures.append("the streamed run is not bit-exact")
    print(_line(f"{res['ncarriers']} carriers identical", ok,
                f"ndiff={res['ndiff']} max|d|={res['max_abs']:.3e} "
                f"nonfinite={res['nonfinite']} {res['differing']}"))
    if res["missing"]:
        print(f"        carriers on one side only: {res['missing']}")
    print()

    # ----------------------------------------------- the frozen DomainState
    print("-- THE DOMAIN STATE THE ROUTE STILL HOLDS (measured, not asserted)")
    print(f"  resident: {len(resident['state_moved'])} of "
          f"{resident['state_carriers']} carriers on node.state moved")
    print(f"  streamed: {len(streamed['state_moved'])} of "
          f"{streamed['state_carriers']} carriers on node.state moved")
    if streamed["state_moved"]:
        print("  -> the state advances; output, health and the final digest "
              "read a live domain")
    else:
        print("  -> node.state DID NOT MOVE.  The forecast is in the store.  "
              "Everything")
        print("     the routes read off node.state -- history frames, the "
              "stability gate,")
        print("     the health validators, canonical_state_digest -- would "
              "read the")
        print("     INITIAL CONDITION for the whole run.")
    print(_line("the host store WARNS about it", streamed["warned"],
                "attach emitted the frozen-state RuntimeWarning"
                if streamed["warned"] else "NO WARNING -- silent and wrong"))
    if not streamed["warned"]:
        failures.append("the host store did not warn that node.state is "
                        "frozen")
    ncopied, back = streamed["refreshed"]
    resident_final = digests(resident["carriers"])
    ok = back == resident_final
    print(_line("refresh_state() -> node.state", ok,
                f"{ncopied} carriers copied back; "
                f"{'equals' if ok else 'DIFFERS FROM'} the resident final "
                "state"))
    if not ok:
        failures.append("refresh_state did not reproduce the resident state")
    print()

    # ------------------------------------- the same thing on HISTORY cadence
    # A history handler is not decoration here: the executor threads
    # ``refl_10cm_due`` into the STEP the moment one is attached, and that
    # flag arms a one-frame producer/consumer handshake -- microphysics
    # stashes REFL_10CM and the frame must consume it before the next due
    # call, or ``stash_refl_10cm`` raises.  A streamed sweep runs that
    # handshake inside the TILE BUFFERS, so this leg is the first thing in
    # this project to ask what happens to it under tiling.
    nhist = max(8, min(steps, 60))
    print(f"-- HISTORY FRAMES at the executor's own history op ({nhist} "
          "steps, REFL_10CM armed)")
    hist_res = run(exp, steps=nhist, refresh_history=True)
    print(f"  resident: {hist_res['frames']}   (tick, digest, REFL staged)")
    # And now the streamed one, which MUST refuse: REFL_10CM is a one-frame
    # handoff whose producer under tiling is the tile buffer's driver and
    # whose consumer is the domain state.  MEASURED before the refusal
    # existed, 96x72x49 tile 24x24: resident staged it for 1 of 2 frames,
    # streamed for 0 of 2 -- no error, no NaN, a missing output field.
    try:
        hist_str = run(exp, steps=nhist, streaming_table=table,
                       refresh_history=True)
    except streaming.StreamingRefused as exc:
        named = "REFL_10CM is a ONE-FRAME" in str(exc)
        print(_line("streamed + history output -> REFUSED", named,
                    "the seam names REFL_10CM" if named
                    else f"refused for the wrong reason: {str(exc)[:60]}"))
        if not named:
            failures.append("the streamed history refusal named the wrong "
                            "thing")
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as exc:                            # noqa: BLE001
        print(_line("streamed run with history output", False,
                    f"{type(exc).__name__}: {str(exc)[:70]}"))
        failures.append(
            "a streamed domain with history output failed for a reason "
            f"other than the REFL_10CM refusal: {type(exc).__name__}: "
            f"{str(exc)[:70]}")
        cp.get_default_memory_pool().free_all_blocks()
    else:
        # If this branch is ever reached, REFL under tiling was solved and
        # the refusal above should go -- so compare, do not celebrate.
        print(f"  streamed: {hist_str['frames']}")
        same = ([f[:2] for f in hist_res["frames"]]
                == [f[:2] for f in hist_str["frames"]])
        refl_res = [f[2] for f in hist_res["frames"]]
        refl_str = [f[2] for f in hist_str["frames"]]
        if not same or refl_res != refl_str:
            failures.append("the streamed history run neither refused nor "
                            "reproduced the resident frames")
        print(_line("refreshed frames identical", same, ""))
        print(_line("REFL_10CM reached the frame", refl_res == refl_str,
                    f"resident {refl_res}, streamed {refl_str}"))
    print()

    # ----------------------------------------- the mode a forecast actually uses
    # ``mode = "on"`` with a pinned tiling is what a PROOF runs.  What a user
    # runs is ``mode = "auto"`` on a domain that does not fit, where the
    # tiling comes from the planner rather than from the config -- a
    # different path through ``decide`` and one no test had ever taken with
    # a builder attached, because there were no builders.  The domain here
    # fits comfortably, so the budget is narrowed until it does not; that is
    # what ``vram_budget_bytes`` is for.
    print("-- MODE 'AUTO': the planner picks the tiling, not the config")
    # MEASURED on this domain, after the budget-override fix: below 3.8 GiB
    # no tile fits at all (the full rung's per-process fixed cost is
    # 3.16 GiB before any tile), at 4.0-4.2 GiB the planner tiles it, and
    # from 4.5 GiB up it correctly declines because a domain this small is
    # cheaper resident.  The window is narrow because the DOMAIN is small;
    # that is a property of the test, not of the mode.
    nauto = max(8, min(steps, 24))
    auto_ok = False
    for budget_gib in (4.2, 4.0):
        table_auto = dict(mode="auto",
                          vram_budget_bytes=int(budget_gib * 2 ** 30))
        try:
            got = run(exp, steps=nauto, streaming_table=table_auto)
        except Exception as exc:                       # noqa: BLE001
            print(f"        budget {budget_gib:.0f} GiB: "
                  f"{type(exc).__name__}: {str(exc)[:80]}")
            continue
        if not got["streamed"]:
            print(f"        budget {budget_gib:.0f} GiB: fits resident, "
                  "auto declined (correctly)")
            continue
        ref = run(exp, steps=nauto)
        res_auto = compare(ref["carriers"], got["carriers"])
        auto_ok = res_auto["bitexact"] and res_auto["nonfinite"] == 0
        print(f"        {got['decision']}")
        print(_line(f"auto streamed, {res_auto['ncarriers']} carriers",
                    auto_ok, f"ndiff={res_auto['ndiff']} "
                             f"max|d|={res_auto['max_abs']:.3e}"))
        break
    if not auto_ok:
        failures.append("mode='auto' never streamed a domain it was told "
                        "would not fit, so the planner path is unproven")
    print()

    # ------------------------------------------------------ broken builders
    if not args.skip_controls:
        print("-- THE BUILDER'S CAPABILITIES, one removed at a time")
        ncontrol = max(8, min(steps, 24))
        base = run(exp, steps=ncontrol, streaming_table=table)
        for what in ("geography rebuilt", "tables unwindowed",
                     "the clock dropped"):
            exp_run = replace(exp, run_seconds=float(ncontrol) * DT)
            state, _b = prepared_state(exp_run.domains[0].run,
                                       exp_run.start_time)
            model = build_model(exp_run, state)
            options = streaming.StreamingOptions(**table)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = run(exp, steps=ncontrol, streaming_table=table,
                          builders={1: broken_builder(model, options,
                                                      what=what)})
            del model, state
            cp.get_default_memory_pool().free_all_blocks()
            diff = compare(base["carriers"], got["carriers"])
            fired = not diff["bitexact"]
            if not fired:
                failures.append(f"the control {what!r} did NOT fire")
            print(_line(what, fired,
                        f"ndiff={diff['ndiff']}/{diff['ncarriers']} "
                        f"max|d|={diff['max_abs']:.3e}"))
        print()

    print("=" * 78)
    if failures:
        print(f"FAIL -- {len(failures)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS -- a [tiles] mode = 'on' experiment runs through "
          "execute_experiment")
    print("        with the SHIPPED builder and is bit-exact against the "
          "resident run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
