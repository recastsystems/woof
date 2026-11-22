"""SPAWN-AT-TRIGGER UNDER STREAMING: does the trigger see the running domain?

``woof.core.nest_spawn.SpawnWatch.evaluate`` decides WHEN a dormant nest is
born and WHERE it lands, from ONE whole-parent plane -- the parent's own
``uh_spawn_window`` (or its composite reflectivity) -- through an argmax and a
footprint-sized weighted centroid around the peak.  It reads that plane off
``parent_state``: the ``DomainState`` object the model tree holds
(``woof/core/spawn_runner.py:200`` builds ``{grid_id: node.state}``).

That is exactly the object a STREAMED domain stops writing to.
``woof.core.streaming.attach`` copies the prepared state's carriers into a
pinned host store and the sweep integrates the STORE; the ``DomainState``
survives only as an identity check (``StreamedDomain.__call__`` compares
``state is self._state`` and never touches its arrays).  A whole-domain
consumer that reads the state therefore reads the domain as it was at attach
time and not as it is now.

For the spawn trigger it is worse than stale, and this module measures the
difference, because the two failures need different fixes:

    STALE   the plane was written once and never again -- a consumer sees
            t=0 values and makes a t=0 decision.

    DEAD    the plane is not a streaming carrier AT ALL.  ``uh_spawn_window``
            and ``uh_follow_window`` are classified ``rebuild`` in
            ``woof/io/restart.py:REBUILT_SCRATCH_SLOTS`` -- deliberately, and
            for a good reason that is about RESTART: a window means "max since
            that consumer last looked" and a checkpoint cannot know when the
            consumer will next look.  ``tilestream.physics_inventory
            .carrier_manifest`` builds the streaming inventory by CALLING
            restart's manifest builders, so a slot restart does not serialize
            is a slot streaming does not carry.  The domain-level window is
            never gathered, never scattered and never folded; each tile
            buffer folds into its OWN tile-shaped window and the next tile
            served by that buffer overwrites it.

    So the plane the spawn trigger reads on a streamed parent is the ZERO
    plane ``DomainState.__init__`` allocated (state.py:720), for the whole
    run.  ``np.max(plane) == 0.0`` for every threshold a user could
    configure, the watch records ``decision: no-signal`` at every leg
    boundary, and at ``latest_s`` it records ``window-closed`` with the note
    that says the reservation was held and the nest never fired -- which is a
    sentence about the WEATHER.  Nothing anywhere says the trigger was blind.

WHAT THIS MODULE RUNS
---------------------
The joined configuration of :mod:`tilestream.test_join` -- real Lambert
projection, real terrain, specified lateral boundaries, full physics -- with
``nwp_diagnostics = 1`` added, which is the switch the spawn trigger's field
depends on and which no lane in this project had ever turned on.  The same
domain is integrated twice, resident and streamed, and ONE
:class:`~woof.core.nest_spawn.SpawnWatch` is evaluated against each result.
PASS is the fired grid_id, the fired flag and the chosen ``(i_parent_start,
j_parent_start)`` being IDENTICAL.

The threshold is not a guess.  It is taken from the RESIDENT run's own plane
at a high percentile, so it is crossed by a small, localized set of cells and
the argmax/centroid is a sharp decision rather than a domain-wide average --
and it is then applied unchanged to both sides.  A threshold that half the
domain crosses would place the nest at the domain centre on any plane at all,
including a plane of zeros, and would prove nothing.

THE CONTROLS
------------
``--no-fix``          the shipped behaviour: the tracker windows are not
                      streamed.  The streamed watch MUST fail to reproduce the
                      resident placement.  It ships as a required FAILURE, so
                      the gate fails if it ever starts passing -- that is the
                      only thing that says the positive is measuring anything.
``resident argmax``   printed, with the number of qualifying cells, so a
                      threshold that qualifies everything is visible.
``radiation/cumulus`` fire counts printed on BOTH sides of the comparison.
                      This project has produced three false results from a
                      timed window in which radiation and cumulus never fired.
``carrier census``    the store's carrier count and whether the window slots
                      are in it, printed before any comparison.

WHAT THIS MODULE DOES NOT TEST
------------------------------
The ``reflectivity`` trigger.  ``refl_10cm`` is a ``(nz, ny, nx)`` ephemeral
handoff produced only on an output-due microphysics call and consumed by the
writer (``woof/core/refl.py``), classified ``rebuild`` on the DRIVER rather
than in the scratch pool.  It is dead under streaming for the same reason the
UH window is, but its fix is not the same fix -- it is an output-path
question, and ``tilestream/test_io.py`` owns the output path.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from woof.core import streaming
from tilestream import driver, gather, harness
from tilestream import physics_inventory as physinv
from tilestream.test_join import (SEED, build_domain, compare,
                                  domain_boundaries, join_cfg, tile_factory)

#: The joined gate's own geometry.  256x192 at 12 km is a real regional
#: forecast footprint; the 32x32 tiling puts tiles with no true edge, one and
#: two into the same plan.  No timing is quoted from this module, so the
#: compute window being below the ~500 cells a TIMING needs is not a defect
#: here -- a small window makes the halo and the seams work harder per cell.
NX, NY, NZ = 256, 192, 49
TX, TY = 32, 32

#: Enough steps that the UH window has structure and that the radiation and
#: cumulus cadences fire on both sides.  At dt = 30 s, radt = 12 min fires
#: every 24 steps and cudt = 5 min every 10, so 32 steps crosses both -- the
#: counts are printed rather than assumed.
NSTEPS = 32

#: The rung.  ``full(real74)+KF`` is configs/real74_d01.toml's own selector
#: set: Morrison, 2-D Smagorinsky, MM5 surface layer, YSU, Noah, RRTMGP and
#: Kain-Fritsch.  The trigger's field needs w and the vertical vorticity of a
#: real physics trajectory, not of a dry one.
RUNG = "full(real74)+KF"

#: The percentile of the RESIDENT plane the threshold is taken at.  99.9 on a
#: 256x192 plane is ~49 cells: a localized peak, not a plateau.
THRESHOLD_PCT = 99.9

#: The declared dormant nest.  A 60x60 child at ratio 3 spans 20 parent cells,
#: so the footprint-sized centroid window around the peak is 21x21 parent
#: cells -- big enough that the centroid is genuinely a weighted average of a
#: region and not a restatement of the argmax.
CHILD_NX, CHILD_NY, RATIO = 60, 60, 3
KEEPOUT = 10          # spec_bdy_width 5 + blend_width 5, the loader's rule

#: The trigger's search box, 1-based inclusive parent cells, and it is not
#: cosmetic.  MEASURED on the first run of this module with no box: the
#: argmax landed at parent (i=0, j=3) -- the domain's own west edge, where
#: WRF's cal_helicity mirrors the inward neighbour into the boundary column
#: (uh_diag's ratified divergence 4) -- and ``_placement_from_centroid``
#: then CLAMPED the placement to the keepout minimum (11, 11).  A clamped
#: placement is the same (11, 11) for every peak in that corner, so an
#: identity claim about it is much weaker than it looks.  Boxing the search
#: into the interior makes the fired placement a genuine function of the
#: centroid, and ``clamped_to_keepout`` is printed so the next run can see
#: whether that stayed true.
SEARCH_BOX = (48, 40, 208, 152)


# --------------------------------------------------------------------------
# the configuration
# --------------------------------------------------------------------------

def spawn_cfg(nx: int = NX, ny: int = NY, nz: int = NZ, rung: str = RUNG):
    """The joined forecast configuration with the UH diagnostic ON.

    ``nwp_diagnostics = 1`` is what allocates ``up_heli_max`` and the two
    consumer-owned tracking windows (state.py:711-720) and what makes
    ``dycore.step`` fold into them every step (dycore.py:2474).  Every other
    lane in this project ran with it OFF, which is why the UH window's
    absence from the streaming inventory had never been observable: with the
    diagnostic off the slots do not exist and there is nothing to miss.
    """
    return join_cfg(nx, ny, nz, rung=rung, nwp_diagnostics=1)


# --------------------------------------------------------------------------
# the plane, read the way each consumer really reads it
# --------------------------------------------------------------------------

def state_window(state, slot: str) -> np.ndarray:
    """The tracker window as ``woof.core.storm_tracking.signal_plane`` sees
    it: off the ``DomainState``'s scratch pool, on the host, float64."""
    import cupy as cp

    buf = state.existing_scratch(slot)
    if buf is None:
        raise AssertionError(
            f"the state carries no {slot!r} slot; nwp_diagnostics=1 "
            "allocates it eagerly, so this config never turned it on")
    return np.asarray(cp.asnumpy(buf), dtype=np.float64)


def store_window(store, slot: str) -> np.ndarray | None:
    """The same plane out of a streamed domain's STORE, or ``None`` when the
    store does not carry it -- which is the whole finding."""
    key = f"scratch/{slot}"
    if key not in store:
        return None
    return np.asarray(store[key], dtype=np.float64)


# --------------------------------------------------------------------------
# the watch, evaluated identically against each result
# --------------------------------------------------------------------------

def make_watch(cfg, threshold: float, *, earliest_s=0.0, latest_s=1.0e9):
    """One dormant d02 watching its parent's ``uh_spawn_window``.

    Built through the shipped :class:`~woof.core.nest_spawn.SpawnConfig` and
    :class:`~woof.core.storm_tracking.NestFootprint` so the object under test
    is the object the runner constructs, not a lookalike.
    """
    from woof.core.nest_spawn import SpawnConfig, SpawnWatch
    from woof.core.storm_tracking import NestFootprint

    return SpawnWatch(
        SpawnConfig(trigger="uh", threshold=float(threshold),
                    search_box=SEARCH_BOX,
                    earliest_s=float(earliest_s), latest_s=float(latest_s)),
        NestFootprint(grid_id=2, i_parent_start=1, j_parent_start=1,
                      child_nx=CHILD_NX, child_ny=CHILD_NY,
                      parent_grid_ratio=RATIO),
        keepout_cells=KEEPOUT)


class _PlaneState:
    """A minimal state whose ONLY job is to answer ``existing_scratch``.

    ``SpawnWatch.evaluate`` reaches its plane through
    ``signal_plane(parent_state, ...)``, which calls
    ``state.existing_scratch(slot)``.  Handing the watch a plane through this
    shim rather than a whole ``DomainState`` keeps the two evaluations
    identical in everything except the plane, which is the variable under
    test.
    """

    def __init__(self, plane, slot: str) -> None:
        self._planes = {slot: plane}

    def existing_scratch(self, slot):
        return self._planes.get(slot)


def evaluate(plane, cfg, threshold, *, t: float = 1.0e4) -> dict:
    """Run the shipped watch against one plane; return the receipt fields."""
    from woof.core.uh_diag import UH_SPAWN_WINDOW_SLOT

    watch = make_watch(cfg, threshold)
    event = watch.evaluate(_PlaneState(plane, UH_SPAWN_WINDOW_SLOT), float(t))
    receipts = watch.drain_receipts()
    last = receipts[-1] if receipts else {}
    return {
        "fired": event is not None,
        "grid_id": None if event is None else int(event.grid_id),
        "placement": None if event is None else list(event.position),
        "decision": last.get("decision"),
        "cells_above_threshold": last.get("cells_above_threshold"),
        "peak_parent_ij": last.get("peak_parent_ij"),
        "max_value": last.get("max_value"),
        "centroid_parent_ij": last.get("centroid_parent_ij"),
        # Printed, not asserted: a CLAMPED placement is the keepout minimum
        # for every peak in that corner, so an identity claim about it says
        # far less than an unclamped one.  See SEARCH_BOX.
        "clamped_to_keepout": last.get("clamped_to_keepout"),
    }


# --------------------------------------------------------------------------
# the two runs
# --------------------------------------------------------------------------

def physics_fire_counts(state) -> dict:
    """How many times each cadence-gated scheme actually ran.

    Printed on BOTH sides of every comparison.  Three of this project's six
    false results were the same mistake -- a number measured in a window
    where radiation and cumulus never fired -- and the only defence that
    works is showing the counts rather than reasoning about the cadence.
    """
    driver_obj = getattr(state, "physics", None)
    if driver_obj is None:
        return {}
    return {k: int(v) for k, v in sorted(dict(driver_obj.call_counts).items())}


def resident(cfg, nsteps, *, boundaries, seed=SEED, warmup=1) -> dict:
    """The control: one resident domain, WOOF-ArWen's own ``dycore.step``."""
    import cupy as cp
    from woof.core.dycore import step as dycore_step
    from woof.core.uh_diag import UH_SPAWN_WINDOW_SLOT

    state, _geo = build_domain(cfg, seed=seed, boundaries=boundaries,
                              warmup=warmup)
    stepper = streaming.make_stepper(state, cfg, streaming.OFF)
    assert stepper is dycore_step, (
        "the resident control must be the dycore's own step, not a wrapper")
    for _ in range(int(nsteps)):
        stepper(state, cfg, refl_10cm_due=False)
    cp.cuda.runtime.deviceSynchronize()
    plane = state_window(state, UH_SPAWN_WINDOW_SLOT)
    carriers = {name: cp.asnumpy(arr) for name, arr
                in physinv.carrier_inventory(state, None).items()}
    counts = physics_fire_counts(state)
    manifest_has_window = any(
        UH_SPAWN_WINDOW_SLOT in key for key in carriers)
    del state
    cp.get_default_memory_pool().free_all_blocks()
    return {"plane": plane, "carriers": carriers, "call_counts": counts,
            "manifest_has_window": bool(manifest_has_window)}


def _builder(cfg, domain, geo, *, boundaries, tracker_windows: bool):
    """The route-owned construction :func:`streaming.make_stepper` needs.

    ``tracker_windows`` is the fix under test: with it FALSE the streaming
    inventory is exactly what ships (restart's manifest), and with it TRUE the
    two consumer-owned tracking windows join the store.
    """
    def build(state, run_cfg, decision):
        geo_inv = {k: gather.pinned_copy(v) for k, v
                   in driver.geography_inventory(domain).items()}
        per_tile = streaming.tile_boundary_tables(
            boundaries, streaming.tile_specs(run_cfg, decision), seam="zeros")
        factory = tile_factory(run_cfg, per_tile[0])
        # THE CONTROL is the SHIPPED inventory -- restart's manifest, which
        # excludes the tracker windows by name.  The fix is the streaming
        # inventory, which adds them back.
        inventory_fn = (physinv.streaming_inventory if tracker_windows
                        else physinv.carrier_inventory)
        return streaming.attach(
            state, run_cfg, decision, tile_state_factory=factory,
            geography=geo_inv, boundary_tables=per_tile,
            inventory_fn=inventory_fn,
            scalars=physinv.carrier_scalars(domain), check_geography=False)

    return build


def streamed(cfg, nsteps, *, boundaries, seed=SEED, warmup=1,
             tile_nx=TX, tile_ny=TY, tracker_windows: bool) -> dict:
    """The streamed run, driven one sweep per model step through the seam."""
    import cupy as cp
    from woof.core.uh_diag import UH_SPAWN_WINDOW_SLOT

    domain, geo = build_domain(cfg, seed=seed, boundaries=boundaries,
                               warmup=warmup)
    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(tile_nx), tile_ny=int(tile_ny), nbuffers=2,
        store="host")
    decision = streaming.decide(cfg, options)
    stepper = streaming.make_stepper(
        domain, cfg, options, decision=decision,
        build=_builder(cfg, domain, geo, boundaries=boundaries,
                       tracker_windows=tracker_windows))
    assert streaming.is_streaming(stepper)
    for _ in range(int(nsteps)):
        stepper(domain, cfg, refl_10cm_due=False)
    cp.cuda.runtime.deviceSynchronize()
    store = {name: np.asarray(arr) for name, arr in stepper.store.items()}
    out = {
        # What the SPAWN RUNNER reads: the DomainState the model tree holds.
        "state_plane": state_window(domain, UH_SPAWN_WINDOW_SLOT),
        # What the domain actually IS: the store.
        "store_plane": store_window(store, UH_SPAWN_WINDOW_SLOT),
        "carriers": store,
        "ncarriers": len(store),
        "tiles": int(stepper.report.get("tiles", 0)),
        "decision": stepper.decision.explain(),
    }
    del domain, stepper
    cp.get_default_memory_pool().free_all_blocks()
    return out


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def _line(label: str, ok: bool, detail: str = "") -> str:
    return f"  {'PASS' if ok else 'FAIL':4s}  {label:56s} {detail}"


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    nsteps = NSTEPS
    for arg in argv:
        if arg.startswith("--steps="):
            nsteps = int(arg.split("=", 1)[1])
    want_fix = "--no-fix" not in argv
    #: ``--fix-only`` skips the negative control's streamed run.  It is a
    #: RERUN convenience and never the first run: with it, nothing in the
    #: output says the shipped inventory fails, and a positive with no
    #: control beside it is what this project has been burned by six times.
    want_control = "--fix-only" not in argv
    if not want_control and not want_fix:
        raise SystemExit("--fix-only and --no-fix select nothing")

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    print("=" * 78)
    print("SPAWN TRIGGER vs STREAMING -- does the watch see the running "
          "domain?")
    print(f"  {NX}x{NY}x{NZ} at dx=12 km, tile {TX}x{TY}, rung {RUNG!r}, "
          f"N={nsteps}, nwp_diagnostics=1")
    print("=" * 78)

    cfg = spawn_cfg()
    halo = harness.halo_radius(cfg)
    print(f"  halo {halo} = 10 + 3*{cfg.time_step_sound}//2, from "
          "harness.halo_radius and never from a measurement")
    print(f"  dt = {cfg.dt:g} s; radt = {cfg.radt_minutes:g} min -> every "
          f"{cfg.radt_minutes * 60 / cfg.dt:.0f} steps; cudt = "
          f"{cfg.cudt_minutes:g} min -> every "
          f"{cfg.cudt_minutes * 60 / cfg.dt:.0f} steps")
    print()

    failures: list[str] = []

    # ---------------------------------------------------- the inventory fact
    print("-- THE STRUCTURAL FACT: is the trigger's plane a carrier at all?")
    probe, _ = build_domain(cfg, seed=SEED, warmup=0)
    manifest = physinv.carrier_manifest(probe)
    has_uh = "scratch/up_heli_max" in manifest
    has_spawn = "scratch/uh_spawn_window" in manifest
    has_follow = "scratch/uh_follow_window" in manifest
    del probe
    cp.get_default_memory_pool().free_all_blocks()
    print(f"        carrier_manifest has {len(manifest)} entries")
    print(f"        scratch/up_heli_max        in manifest: {has_uh}")
    print(f"        scratch/uh_spawn_window    in manifest: {has_spawn}")
    print(f"        scratch/uh_follow_window   in manifest: {has_follow}")
    print("        (up_heli_max is SERIALIZED_SCRATCH_SLOTS; the two tracker")
    print("         windows are REBUILT_SCRATCH_SLOTS, and the streaming")
    print("         inventory is restart's manifest -- so a slot restart does")
    print("         not serialize is a slot streaming does not carry.)")
    print()

    # ---------------------------------------------------------- the two runs
    seed_b = SEED + 1
    bnd_a, _ = build_domain(cfg, seed=SEED, warmup=0)
    bnd_b, _ = build_domain(cfg, seed=seed_b, warmup=0)
    bnd = domain_boundaries(cfg, bnd_a, bnd_b)
    del bnd_a, bnd_b
    cp.get_default_memory_pool().free_all_blocks()

    t0 = time.perf_counter()
    ref = resident(cfg, nsteps, boundaries=bnd)
    t_res = time.perf_counter() - t0
    plane = ref["plane"]
    finite = np.isfinite(plane)
    if not finite.all():
        failures.append("the RESIDENT control's UH plane is not finite; "
                        "there is nothing to compare against")
    # The threshold is a percentile of the plane INSIDE THE SEARCH BOX, not
    # of the whole plane: the box is where the argmax will be taken, and a
    # percentile of the whole domain would be set by cells the trigger is
    # never going to look at.
    i_lo, j_lo, i_hi, j_hi = SEARCH_BOX
    box = plane[j_lo - 1:j_hi, i_lo - 1:i_hi]
    threshold = float(np.percentile(box[np.isfinite(box)], THRESHOLD_PCT))
    qualifying = int(np.count_nonzero(box >= threshold))
    if not threshold > 0.0:
        # A zero threshold qualifies a plane of zeros, so a DEAD plane would
        # fire at the same cell as a live one and the negative control below
        # would pass for the wrong reason.  Named here rather than left to be
        # noticed in the numbers.
        failures.append(
            f"the P{THRESHOLD_PCT} threshold is {threshold!r}: the resident "
            "UH plane has no localized signal, so this run cannot "
            "discriminate a live plane from a dead one")
    print(f"-- THE CONTROL (resident, {t_res:.1f} s)")
    print(f"        UH window: max {plane.max():.4g}, mean "
          f"{plane.mean():.4g}, nonzero cells "
          f"{int(np.count_nonzero(plane))} of {plane.size}")
    print(f"        threshold = P{THRESHOLD_PCT} of the resident plane inside "
          f"search box {SEARCH_BOX} = {threshold:.6g} m2 s-2")
    print(f"        -> {qualifying} qualifying cells of {box.size} in the box "
          f"({100.0 * qualifying / box.size:.2f}%)")
    print(f"        physics fire counts: {ref['call_counts']}")
    for scheme in ("radiation", "cumulus"):
        n = ref["call_counts"].get(scheme, 0)
        if n < 1:
            failures.append(
                f"the resident control never fired {scheme} in {nsteps} "
                "steps; a window in which a cadence never fires cannot "
                "certify anything about it")
    control = evaluate(plane, cfg, threshold)
    print(f"        watch -> {control}")
    if not control["fired"]:
        failures.append("the RESIDENT control's watch did not fire; the "
                        "threshold is wrong and there is no placement to "
                        "compare against")
    print()

    # ------------------------------------------------------------- streamed
    arms = tuple(w for w, want in ((False, want_control), (True, want_fix))
                 if want)
    for tracker_windows in arms:
        label = ("streamed, tracker windows CARRIED (the fix)"
                 if tracker_windows else
                 "streamed, shipped inventory (the NEGATIVE CONTROL)")
        print(f"-- {label}")
        t0 = time.perf_counter()
        try:
            got = streamed(cfg, nsteps, boundaries=bnd,
                           tracker_windows=tracker_windows)
        except Exception as exc:                          # noqa: BLE001
            print(_line(label, False, f"raised {exc!r}"))
            failures.append(f"{label}: raised {exc!r}")
            import traceback
            traceback.print_exc()
            continue
        dt_run = time.perf_counter() - t0
        sp = got["state_plane"]
        st = got["store_plane"]
        print(f"        {got['ncarriers']} carriers streamed, "
              f"{got['tiles']} tiles, {dt_run:.1f} s")
        print(f"        the plane the SPAWN RUNNER reads (node.state): "
              f"max {sp.max():.4g}, nonzero cells "
              f"{int(np.count_nonzero(sp))} of {sp.size}")
        if st is None:
            print("        the plane in the STORE: ABSENT -- the domain's "
                  "window is not a streamed carrier")
        else:
            print(f"        the plane in the STORE: max {st.max():.4g}, "
                  f"nonzero cells {int(np.count_nonzero(st))}")

        runner_view = evaluate(sp, cfg, threshold)
        print(f"        watch on node.state -> {runner_view}")
        same = (runner_view["fired"] == control["fired"]
                and runner_view["placement"] == control["placement"])

        if not tracker_windows:
            # THE NEGATIVE CONTROL.  It ships as a required FAILURE: if the
            # streamed watch ever reproduces the resident placement WITHOUT
            # the windows being carried, then the trigger is not sensitive to
            # the plane and nothing above it means anything.
            ok = not same
            print(_line("negative control: shipped streaming DIFFERS", ok,
                        "" if ok else
                        "the streamed watch reproduced the resident "
                        "placement with a dead plane -- the trigger is not "
                        "sensitive enough to be a control"))
            if not ok:
                failures.append("negative control did not fire")
        else:
            # The bit-exactness of the rest of the domain still has to hold:
            # a fix that carries one more array and perturbs the trajectory
            # is not a fix.
            res = compare(ref["carriers"],
                          {k: v for k, v in got["carriers"].items()
                           if k in ref["carriers"]})
            ok_bits = res["bitexact"] and res["nonfinite"] == 0
            print(_line("streamed trajectory still bit-exact", ok_bits,
                        f"{res['ntotal'] - res['ndiff']}/{res['ntotal']} "
                        f"carriers identical"
                        + ("" if ok_bits else
                           f", differing {res['differing']}, "
                           f"max|d| {res['max_abs']:.3g}")))
            if not ok_bits:
                failures.append("the fix perturbed the trajectory")
            ok_plane = st is not None and np.array_equal(st, plane)
            worst = (float("nan") if st is None
                     else float(np.max(np.abs(st - plane))))
            print(_line("the UH window plane is bit-identical", ok_plane,
                        f"max|d| {worst:.6g}"))
            if not ok_plane:
                failures.append("the streamed UH window is not bit-identical")
            print(_line("the watch fires at the SAME placement", same,
                        f"resident {control['placement']} vs streamed "
                        f"{runner_view['placement']}"))
            if not same:
                failures.append("the streamed watch chose a different "
                                "placement")
        print()

    print("=" * 78)
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for row in failures:
            print(f"  - {row}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
