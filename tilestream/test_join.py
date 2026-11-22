"""THE JOIN, through WOOF-ArWen's own seam: a real forecast configuration streamed.

Real Lambert projection, real terrain, SPECIFIED lateral boundaries, full
physics, the whole domain in pinned host RAM, one tile of it on the card at a
time -- and bit-exact against the same configuration integrated as one
resident block.  Geography had been proven on PERIODIC domains; lateral
boundaries had been proven on FLAT UNPROJECTED ones; a real forecast is the
intersection and it had never been run.

It is driven through :func:`woof.core.streaming.make_stepper`, which is the
callable WOOF-ArWen's own run loops step a domain with
(:func:`woof.core.model.execute_experiment`'s STEP op and
:func:`woof.runtime.integrate_prepared_case`'s substep both call it and
nothing else).  So the object under test here is the object the model runs,
not a lookalike: the loop below is the model's loop with the output, restart,
health and nest-coupling ops removed, because this module is a bit-exactness
gate and those ops are proven elsewhere (``tilestream/test_io.py``,
``tilestream/test_restart_gate.py``).

    python -m tilestream.test_join            # the whole gate
    python -m tilestream.test_join --quick    # dry + the top rung only


THREE THINGS THAT ARE TRUE HERE AND WERE TRUE IN NEITHER PREVIOUS LANE
-----------------------------------------------------------------------
1.  ``periodic_faces=False``.  ``harness.make_geography``'s default
    duplicates the closing map-factor faces (``msfu[:, nx] = msfu[:, 0]``)
    because on a periodic domain ``TileSpec._axis_gather`` reduces every
    window mod nx and never reads the alias slot, so the domain's column
    ``nx`` must equal its column 0 or a tile and a monolithic run disagree
    about a face that physically IS the same face.  On a SPECIFIED domain
    column ``nx`` is a real east boundary face ``dx*nx`` away from column 0
    and ``TileSpec.owns_x_alias`` becomes ``i1 == nx`` -- the east-edge tile
    really does gather and scatter it.  Duplicating would install the WEST
    edge's map factor on the EAST edge.  :func:`periodic_face_lie` MEASURES
    how big that lie is instead of asserting that it matters: 0.245% of
    msfv at the closing face, equating two columns 1800 km and 21.1 degrees
    of longitude apart.

2.  Tiles that own TRUE DOMAIN EDGES and tiles that own only interior seams
    exist in the same plan (measured on the gate's own geometry: 24 tiles
    with no true edge, 20 with one, 4 with two).  A tile's true edges take
    the domain's own boundary tables windowed along the tangential axis; its
    interior seams take inert tables.

3.  ``elapsed_seconds`` drives ``dtbc`` (lateral_bc.py:373-378) on top of
    every physics cadence, so it must be carried per tile even at the DRY
    rung, where the previous lanes had no scalar carriers at all.


THE CONTROLS, AND WHAT EACH ONE IS FOR
--------------------------------------
This project has produced six false results and every one of them was caught
by a control.  Each capability here ships with a test that fails when the
capability is removed:

``periodic faces``      duplicate the closing faces -> must DIFFER
``per-tile windowing``  give every tile the domain's tables -> must DIFFER
``true-edge tables``    scale a true edge by 1.000001 -> must DIFFER
``the clock``           stop carrying elapsed_seconds -> must DIFFER
``geography``           let each tile rebuild its own -> must DIFFER
``the halo``            one cell below the measured margin -> must DIFFER
``interior seams``      zeros / self-consistent / GARBAGE -> must AGREE

The last is the one the mode rests on, and it is stated the strong way
round: a seam filled with 1e6 in coupled units and 1e4/s of tendency gives
the bit-identical answer, so the seam relaxation provably cannot reach the
interior at the halo ``harness.halo_radius`` prescribes.
"""

from __future__ import annotations

import hashlib
import sys
import time
import warnings

import numpy as np

from woof.core import streaming
from tilestream import driver, gather, harness
from tilestream import physics_inventory as physinv
from tilestream import spec as tspec

# --------------------------------------------------------------------------
# the joined configuration
# --------------------------------------------------------------------------

#: WRF's specified lateral-boundary selector set.  ``periodic=False`` in
#: ``harness.make_config`` merely DECLINES to force the flags off; these are
#: the flags that must then be on.
SPEC_BC = dict(specified=True, open_x=False, open_y=False, nested=False,
               spec_bdy_width=5, spec_zone=1, relax_zone=4, spec_exp=0.0)

_MOIST = dict(moist=True, mp_physics=10, ztop=20000.0)
_FULL = dict(_MOIST, km_opt=4, sf_sfclay_physics=91, bl_pbl_physics=1,
             bldt=0.0, sf_surface_physics=2, ra_sw_physics=4,
             ra_lw_physics=4, radt_minutes=12.0, cu_physics=1,
             cudt_minutes=5.0)

#: The ladder, worked UP, with the same selector sets as
#: ``test_gate.PHYSICS_RUNGS`` so a failure here can be compared against a
#: failure there rung for rung.
RUNGS: dict[str, dict] = {
    "dry":               dict(ztop=20000.0),
    "mp10":              dict(_MOIST),
    "+YSU PBL":          dict(_MOIST, km_opt=4, sf_sfclay_physics=91,
                              bl_pbl_physics=1, bldt=0.0),
    "+Noah LSM":         dict(_MOIST, km_opt=4, sf_sfclay_physics=91,
                              bl_pbl_physics=1, bldt=0.0,
                              sf_surface_physics=2),
    "full(real74)+KF":   dict(_FULL),
    "full+MYNN+Noah-MP": dict(_FULL, sf_sfclay_physics=5, bl_pbl_physics=5,
                              sf_surface_physics=4),
    "full fast cadence": dict(_FULL, sf_sfclay_physics=5, bl_pbl_physics=5,
                              sf_surface_physics=4, radt_minutes=0.05,
                              cudt_minutes=0.1, bldt=1.0),
}

NZ = 49
SEED = 20_260_731
#: A 12 km grid's own time step (WRF's rule of thumb is ~6 s per km).  The
#: geography lane ran dx = 12 km at the harness's dt = 3 s, which is a 500 m
#: domain's step on a 12 km grid; a forecast configuration has to use the
#: real one, and dtbc is measured in it.
DT = 60.0
#: The step each rung actually runs at, and why it is not always ``DT``.
#: MEASURED stability ladder, 150x120x49, N=8, on this harness's random
#: initial state: dry is clean at every step from 3 s to 60 s, but mp10 and
#: full+MYNN+Noah-MP go NON-FINITE at 60 s (20 and 12 non-finite cells
#: respectively) and are clean at 30 s and below.  That is a property of the
#: INITIAL CONDITION, not of streaming: it is the MONOLITHIC reference that
#: blows up, and a reference that is not finite has nothing to compare
#: against.  A balanced analysis at 12 km runs at 60 s; a field of random
#: perturbations does not, and pretending otherwise would turn this gate into
#: a test of whether two runs produce the same NaNs.
RUNG_DT: dict[str, float] = {"dry": 60.0}
DT_MOIST = 30.0
#: The boundary interval must cover every ``elapsed_seconds`` the run reaches:
#: ``LateralBoundaries.interval_at`` raises outside it and ``dtbc`` is
#: ``elapsed - interval.start``.
BDY_SECONDS = 21600.0


def join_cfg(nx: int, ny: int, nz: int = NZ, rung: str = "dry", **over):
    """Specified BCs + real Lambert + real terrain + one physics rung."""
    kwargs = dict(harness.GEOGRAPHY_OVERRIDES)   # map_proj=1, terrain_opt=1,
    kwargs.update(SPEC_BC)                       # dx = dy = 12000
    kwargs.update(RUNGS[rung])
    kwargs.setdefault("dt", RUNG_DT.get(rung, DT_MOIST))
    kwargs.update(over)
    return harness.make_config(nx, ny, nz, periodic=False, **kwargs)


def periodic_face_lie(cfg) -> dict:
    """How wrong ``periodic_faces=True`` is on a SPECIFIED domain.

    Returns the largest absolute and relative difference between the raw
    Lambert closing faces and the duplicated ones, plus the physical
    separation of the two columns the duplication equates.  A measurement,
    not an assertion that "a specified domain has real edges, so obviously
    do not duplicate".
    """
    accurate = harness.make_geography(cfg, terrain=True, periodic_faces=False)
    lie = harness.make_geography(cfg, terrain=True, periodic_faces=True)
    du = np.abs(accurate.msfu[:, -1] - lie.msfu[:, -1])
    dv = np.abs(accurate.msfv[-1, :] - lie.msfv[-1, :])
    return {
        "msfu_face_max_rel": float((du / np.abs(accurate.msfu[:, -1])).max()),
        "msfv_face_max_rel": float((dv / np.abs(accurate.msfv[-1, :])).max()),
        "columns_equated_km": float(cfg.dx * cfg.nx / 1000.0),
        "lon_span_deg": float(accurate.lon[:, -1].mean()
                              - accurate.lon[:, 0].mean()),
    }


def domain_boundaries(cfg, state_a, state_b, *, seconds: float = BDY_SECONDS):
    """The DOMAIN's specified forcing, from two coupled snapshots.

    Two genuinely different states (different seeds on the same geography)
    give a NONZERO time tendency, so ``dtbc`` -- and therefore
    ``elapsed_seconds`` -- actually reaches the answer.  A single repeated
    snapshot would give a zero tendency and quietly disarm the clock control,
    which would then pass on a build that never carried the clock at all.
    """
    from woof.ingest.lateral_bc import build_state_lateral_boundaries

    return build_state_lateral_boundaries(
        [state_a, state_b], [0.0, float(seconds)],
        spec_bdy_width=int(cfg.spec_bdy_width), spec_zone=int(cfg.spec_zone),
        relax_zone=int(cfg.relax_zone))


def coupled_snapshot(state):
    """The domain's coupled boundary fields, on the host, for seam='self'."""
    from woof.ingest.lateral_bc import _coupled_device_fields

    import cupy as cp

    return {name: cp.asnumpy(arr)
            for name, arr in _coupled_device_fields(state).items()}


# --------------------------------------------------------------------------
# states
# --------------------------------------------------------------------------

def build_domain(cfg, *, seed: int = SEED, periodic_faces: bool = False,
                 boundaries=None, warmup: int = 1):
    """A prepared, resident domain on the real projection and real terrain."""
    from woof.ingest.lateral_bc import attach_lateral_boundaries

    geo = harness.make_geography(cfg, terrain=True,
                                 periodic_faces=periodic_faces)
    state, _drv = harness.make_physics_state(cfg, seed, geography=geo)
    if boundaries is not None:
        attach_lateral_boundaries(state, boundaries)
    if warmup:
        harness.run_steps(state, cfg, int(warmup))
    return state, geo


def tile_factory(cfg, boundaries0, *, geography_fn=None, seed: int = 4242,
                 warmup: int = 1):
    """A ``tile_state_factory`` for a joined tile buffer.

    Three things beyond ``driver.make_physics_tile_state``, each of which a
    joined tile would otherwise get wrong:

    * built with a :class:`harness.Geography` at the TILE's extents, so
      ``terrain_opt=1``'s 3-D ``thb/pb/alb/phb`` and ``mub2d`` have the right
      SHAPES.  The default builder is ``harness.neutral_geography`` -- the
      POISON buffer -- so an array the gather fails to write is obviously
      wrong rather than plausibly wrong;
    * lateral boundaries ATTACHED before the warmup step, because
      ``cfg.specified=True`` makes ``dycore.step`` call
      ``apply_state_lateral_boundaries``, which raises without an attachment.
      The tables attached here are tile 0's and are replaced by the
      ``tile_hook`` before the buffer serves anyone;
    * the warmup step, still required for the lazily allocated carriers.
    """
    from woof.ingest.lateral_bc import attach_lateral_boundaries

    build = geography_fn or harness.neutral_geography

    def make(tile_cfg):
        state, _drv = harness.make_physics_state(
            tile_cfg, seed, geography=build(tile_cfg))
        if boundaries0 is not None:
            attach_lateral_boundaries(state, boundaries0)
        if warmup:
            harness.run_steps(state, tile_cfg, int(warmup))
        return state

    return make


# --------------------------------------------------------------------------
# digests
# --------------------------------------------------------------------------

def _as_numpy(value):
    import cupy as cp

    return cp.asnumpy(value) if isinstance(value, cp.ndarray) \
        else np.asarray(value)


def digest_arrays(arrays) -> dict[str, str]:
    out = {}
    for name in sorted(arrays):
        host = np.ascontiguousarray(_as_numpy(arrays[name]))
        h = hashlib.sha256()
        h.update(name.encode())
        h.update(host.dtype.str.encode())
        h.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        h.update(host.tobytes(order="C"))
        out[name] = h.hexdigest()
    return out


def compare(ref: dict, got: dict) -> dict:
    """Carrier-by-carrier, plus the worst absolute difference over all."""
    da, db = digest_arrays(ref), digest_arrays(got)
    differing = sorted(n for n in da if da.get(n) != db.get(n))
    worst = 0.0
    for name in differing:
        a, b = _as_numpy(ref[name]), _as_numpy(got[name])
        if a.dtype.kind == "f" and a.shape == b.shape:
            worst = max(worst, float(np.nanmax(np.abs(
                a.astype(np.float64) - b.astype(np.float64)))))
    nonfinite = sum(int(np.count_nonzero(~np.isfinite(_as_numpy(v))))
                    for v in got.values()
                    if _as_numpy(v).dtype.kind == "f")
    return dict(bitexact=not differing, ndiff=len(differing),
                ntotal=len(da), differing=differing[:8], max_abs=worst,
                nonfinite=nonfinite)


# --------------------------------------------------------------------------
# the two runs
# --------------------------------------------------------------------------

class UnstableReference(RuntimeError):
    """The reference run itself went non-finite, so there is no comparison."""


def bound_lbc_clock(cfg, *, nsteps: int,
                    lbc_interval_seconds: float = BDY_SECONDS):
    """One production integer-tick DomainClock, ready to bind (task #219).

    The exact constructor the offline child binds its root with; every
    production driver-forced root gets the same shape from the tree build.
    A FRESH instance per arm: the clock is mutable per-step state.
    """
    from woof.offline_child_run import _child_boundary_clock

    return _child_boundary_clock(
        cfg, lbc_interval_seconds=float(lbc_interval_seconds),
        steps=int(nsteps), output_steps=int(nsteps))


def _drive(stepper, state, cfg, clock, nsteps) -> None:
    """One loop for BOTH arms.  With a clock this is the executor's exact
    per-step recurrence (core/clock.py execute_schedule; the offline child
    runs the same lines): seam reset -> prepare_step -> solve -> advance.
    Without one it is the loop this gate has always run, byte for byte."""
    for _ in range(int(nsteps)):
        if clock is not None:
            if clock.lbc_reset_due():
                clock.mark_force()
            clock.prepare_step()
        stepper(state, cfg, refl_10cm_due=False)
        if clock is not None:
            clock.advance()


def monolithic(cfg, nsteps, *, seed=SEED, boundaries=None, warmup=1,
               periodic_faces=False, clock=None) -> dict:
    """The reference: one resident domain, WOOF-ArWen's ordinary ``dycore.step``.

    Deliberately the SAME loop shape as the streamed run below, so the only
    difference between the two is which callable ``make_stepper`` returned.
    ``clock`` binds the production DomainClock to the external LBC mirror
    (WRF's post-increment dtbc, the semantics every real-data root runs
    under); ``None`` keeps the retired elapsed-based compatibility path.
    """
    from woof.core.dycore import step as dycore_step
    from woof.ingest.lateral_bc import bind_lateral_boundary_clock

    state, _geo = build_domain(cfg, seed=seed, boundaries=boundaries,
                               warmup=warmup, periodic_faces=periodic_faces)
    if clock is not None:
        bind_lateral_boundary_clock(state, clock)
    stepper = streaming.make_stepper(state, cfg, streaming.OFF)
    assert stepper is dycore_step, (
        "the resident reference must run the dycore's own step, not a "
        "wrapper around it")
    _drive(stepper, state, cfg, clock, nsteps)
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    out = {name: _as_numpy(arr)
           for name, arr in physinv.carrier_inventory(state, None).items()}
    bad = sum(int(np.count_nonzero(~np.isfinite(v))) for v in out.values()
              if v.dtype.kind == "f")
    if bad:
        # Loud, because the silent version of this is a gate that compares
        # one run's NaNs against another's and reports agreement.
        raise UnstableReference(
            f"the MONOLITHIC reference has {bad} non-finite cells after "
            f"{nsteps} steps at dt={cfg.dt} s; there is nothing to compare "
            "a streamed run against.  See RUNG_DT for the measured "
            "stability ladder of this harness's initial state.")
    return out


def streamed(cfg, nsteps, tile_nx, tile_ny, *, seed=SEED, boundaries=None,
             warmup=1, nbuffers=2, halo=None, seam="zeros", snapshot=None,
             store="host", periodic_faces=False, geography=True,
             carry_clock=True, window_tables=True, clock=None,
             report: dict | None = None) -> dict:
    """The streamed run, driven one sweep per model step through the seam.

    ``geography=False``, ``carry_clock=False``, ``window_tables=False``,
    ``periodic_faces=True`` and a short ``halo`` are the negative controls;
    each disables exactly one capability and each MUST make the answer
    differ.  ``clock`` binds the production DomainClock to the DOMAIN's
    external mirror BEFORE the streamed domain is built, exactly where the
    production routes bind theirs -- the tile hook must then carry that
    binding onto every buffer (task #219).
    """
    from woof.ingest.lateral_bc import bind_lateral_boundary_clock

    domain, geo = build_domain(cfg, seed=seed, boundaries=boundaries,
                               warmup=warmup, periodic_faces=periodic_faces)
    if clock is not None:
        bind_lateral_boundary_clock(domain, clock)
    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(tile_nx), tile_ny=int(tile_ny),
        nbuffers=int(nbuffers), halo=halo, store=store)
    decision = streaming.decide(cfg, options)

    build = _make_builder(
        cfg, domain, geo, boundaries=boundaries, seam=seam, snapshot=snapshot,
        geography=geography, carry_clock=carry_clock,
        window_tables=window_tables)
    stepper = streaming.make_stepper(domain, cfg, options, decision=decision,
                                     build=build)
    assert streaming.is_streaming(stepper)
    _drive(stepper, domain, cfg, clock, nsteps)
    if report is not None:
        report.update(stepper.report)
        report["decision"] = stepper.decision.explain()
    return {name: np.asarray(arr) for name, arr in stepper.store.items()}


def _make_builder(cfg, domain, geo, *, boundaries, seam,
                  snapshot, geography, carry_clock, window_tables):
    """The route-owned construction :func:`streaming.make_stepper` needs."""

    def build(state, run_cfg, decision):
        if geography:
            # The real tiled run: buffers are built on NEUTRAL geography --
            # a poison fill -- and the domain's own arrays are gathered into
            # them, once per buffer, never scattered back.
            geo_inv = {k: gather.pinned_copy(v) for k, v in
                       driver.geography_inventory(domain).items()}
            geo_fn = None
        else:
            # THE CONTROL: each buffer REBUILDS the geography from its own
            # tile_cfg, so it holds the map factors, Coriolis and terrain of
            # a domain centred on the TILE.  Plausible, self-consistent and
            # wrong by up to 1022 km of displacement and 20.6% in Coriolis.
            geo_inv = None

            def geo_fn(tile_cfg):
                return harness.make_geography(tile_cfg, terrain=True,
                                              periodic_faces=False)

        scalars = (physinv.carrier_scalars(domain) if carry_clock else None)
        per_tile = None
        if boundaries is not None:
            # Windowed ONCE, here, and used twice: tile 0's tables give the
            # buffers something correctly shaped to take their warmup step
            # with, and the whole list becomes the tile hook.
            per_tile = streaming.tile_boundary_tables(
                boundaries, streaming.tile_specs(run_cfg, decision),
                seam=seam, snapshot=snapshot)
            if not window_tables:
                # THE CONTROL: every tile gets TILE 0's tables.  Tile 0 owns
                # the west and south domain edges, so this hands the domain's
                # west and south forcing to tiles whose west and south are
                # interior seams hundreds of kilometres away, and hands
                # nothing to the tiles that own the east and north edges.
                # The SHAPES are identical, so nothing complains -- which is
                # the point: the defect this control reproduces is one that
                # cannot be caught by a shape check.
                per_tile = [per_tile[0]] * len(per_tile)
        factory = tile_factory(run_cfg,
                               None if per_tile is None else per_tile[0],
                               geography_fn=geo_fn)
        return streaming.attach(
            state, run_cfg, decision, tile_state_factory=factory,
            geography=geo_inv, boundary_tables=per_tile,
            scalars=scalars, check_geography=False)

    return build


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

#: 256x192 at 12 km is a 3072 x 2304 km domain -- a real regional forecast
#: footprint -- split 8x6 into 32x32 tiles, which is the geometry that puts
#: 24 tiles with NO true edge, 20 with one and 4 with two into the same plan.
#: The compute window is 64 cells, deliberately far below the ~500 cells a
#: TIMING measurement needs: this is a bit-exactness gate and a small window
#: makes the halo, the seams and the ragged arithmetic do more work per cell,
#: not less.
NX, NY = 256, 192
TX, TY = 32, 32
NSTEPS = 8


def _line(label: str, ok: bool, detail: str = "") -> str:
    return f"  {'PASS' if ok else 'FAIL':4s}  {label:52s} {detail}"


def edge_census(cfg, tile_nx, tile_ny, halo) -> dict:
    specs = tspec.plan_tiles(int(cfg.nx), int(cfg.ny), int(tile_nx),
                             int(tile_ny), int(halo), False)
    counts = {0: 0, 1: 0, 2: 0}
    for s in specs:
        counts[sum(owned for owned in streaming.owned_edges(s).values())] += 1
    return {"tiles": len(specs), "no_true_edge": counts[0],
            "one_true_edge": counts[1], "two_true_edges": counts[2]}


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    rungs = (["dry", "full+MYNN+Noah-MP"] if quick else list(RUNGS))

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    print()
    print("=" * 78)
    print("THE JOIN -- real Lambert + real terrain + specified LBC + physics,")
    print(f"            streamed from pinned host RAM, {NX}x{NY}x{NZ} at "
          f"dx=12 km, tile {TX}x{TY}, N={NSTEPS}")
    print("=" * 78)

    cfg0 = join_cfg(NX, NY, rung="dry")
    halo = harness.halo_radius(cfg0)
    census = edge_census(cfg0, TX, TY, halo)
    lie = periodic_face_lie(cfg0)
    print(f"  halo {halo} = 10 + 3*{cfg0.time_step_sound}//2, taken from "
          "harness.halo_radius and never from a measurement")
    print(f"  {census['tiles']} tiles: {census['no_true_edge']} own no true "
          f"edge, {census['one_true_edge']} own one, "
          f"{census['two_true_edges']} own two")
    print(f"  periodic_faces=True would equate two columns "
          f"{lie['columns_equated_km']:.0f} km and "
          f"{lie['lon_span_deg']:.1f} deg of longitude apart, "
          f"for {lie['msfv_face_max_rel'] * 100:.3f}% of msfv")
    print()

    failures: list[str] = []

    # -------------------------------------------------- the ladder, worked up
    print("-- THE POSITIVE: streamed == monolithic, rung by rung")
    for rung in rungs:
        cfg = join_cfg(NX, NY, rung=rung)
        t0 = time.perf_counter()
        seed_b = SEED + 1
        bnd_a, _ = build_domain(cfg, seed=SEED, warmup=0)
        bnd_b, _ = build_domain(cfg, seed=seed_b, warmup=0)
        bnd = domain_boundaries(cfg, bnd_a, bnd_b)
        del bnd_a, bnd_b
        cp.get_default_memory_pool().free_all_blocks()

        try:
            ref = monolithic(cfg, NSTEPS, boundaries=bnd)
        except UnstableReference as exc:
            failures.append(f"joined {rung}: {exc}")
            print(_line(f"{rung} (dt={cfg.dt:g} s)", False,
                        "MONOLITHIC REFERENCE NON-FINITE"))
            cp.get_default_memory_pool().free_all_blocks()
            continue
        report: dict = {}
        got = streamed(cfg, NSTEPS, TX, TY, boundaries=bnd, report=report)
        res = compare(ref, got)
        ok = res["bitexact"] and res["nonfinite"] == 0
        if not ok:
            failures.append(f"joined {rung} is not bit-exact")
        print(_line(f"{rung} (dt={cfg.dt:g} s)", ok,
                    f"{res['ntotal']} carriers, ndiff={res['ndiff']}, "
                    f"nonfinite={res['nonfinite']}, "
                    f"{time.perf_counter() - t0:.1f} s"))
        if not ok:
            print(f"        max|d| = {res['max_abs']:.6g}  "
                  f"first differing: {res['differing']}")
        del ref, got
        cp.get_default_memory_pool().free_all_blocks()

    # --------------------------------------------- the seams must be inert
    print()
    print("-- INTERIOR SEAMS ARE INERT: three completely different fillings")
    cfg = join_cfg(NX, NY, rung="dry")
    bnd_a, _ = build_domain(cfg, seed=SEED, warmup=0)
    bnd_b, _ = build_domain(cfg, seed=SEED + 1, warmup=0)
    bnd = domain_boundaries(cfg, bnd_a, bnd_b)
    snap = coupled_snapshot(bnd_a)
    del bnd_a, bnd_b
    cp.get_default_memory_pool().free_all_blocks()
    ref = monolithic(cfg, NSTEPS, boundaries=bnd)
    for seam in ("zeros", "self", "poison"):
        got = streamed(cfg, NSTEPS, TX, TY, boundaries=bnd, seam=seam,
                       snapshot=snap)
        res = compare(ref, got)
        ok = res["bitexact"] and res["nonfinite"] == 0
        if not ok:
            failures.append(f"seam={seam} is not bit-exact")
        print(_line(f"seam={seam}", ok,
                    f"ndiff={res['ndiff']}, nonfinite={res['nonfinite']}"))
        del got

    # ------------------------------------------- the BOUND clock (task #219)
    #
    # Every production driver-forced root binds a DomainClock to its external
    # LBC mirror, switching Davies consumers to WRF's post-increment dtbc
    # (dt..T_bdy).  This gate ran BOTH arms unbound (0..T-dt) for its whole
    # life, so a tile hook that silently dropped a binding -- which is what
    # shipped -- passed it bit-exact while every clock-bound route forced one
    # step late.  Both arms are warmup=0 here: a one-step warmup shifts the
    # retired path's dtbc by exactly +dt and makes the two semantics
    # numerically coincide at dt-multiple times, which would quietly disarm
    # all three checks below.
    print()
    print("-- THE BOUND CLOCK (production dtbc semantics; both arms warmup=0)")
    ref_bound = monolithic(cfg, NSTEPS, boundaries=bnd, warmup=0,
                           clock=bound_lbc_clock(cfg, nsteps=NSTEPS))
    got_bound = streamed(cfg, NSTEPS, TX, TY, boundaries=bnd, warmup=0,
                         clock=bound_lbc_clock(cfg, nsteps=NSTEPS))
    res = compare(ref_bound, got_bound)
    ok = res["bitexact"] and res["nonfinite"] == 0
    if not ok:
        failures.append(
            f"clock-bound streamed differs from clock-bound resident on "
            f"{res['ndiff']}/{res['ntotal']} carriers -- the tile buffers "
            "are not consuming the bound dtbc recurrence")
    print(_line("bound both arms (must be bit-exact)", ok,
                f"ndiff={res['ndiff']}/{res['ntotal']}, "
                f"max|d|={res['max_abs']:.4g}, "
                f"nonfinite={res['nonfinite']}"))
    del got_bound
    ref_unbound = monolithic(cfg, NSTEPS, boundaries=bnd, warmup=0)
    res = compare(ref_bound, ref_unbound)
    differs = not res["bitexact"]
    if not differs:
        failures.append(
            "bound and unbound resident references are bit-identical; the "
            "dtbc semantics is disarmed and the bound-clock checks above "
            "prove nothing")
    print(_line("bound vs unbound semantics (must differ)", differs,
                f"ndiff={res['ndiff']}/{res['ntotal']}, "
                f"max|d|={res['max_abs']:.4g}"))
    got_unbound = streamed(cfg, NSTEPS, TX, TY, boundaries=bnd, warmup=0)
    res = compare(ref_unbound, got_unbound)
    ok = res["bitexact"] and res["nonfinite"] == 0
    if not ok:
        failures.append(
            f"UNBOUND streamed stopped matching unbound resident "
            f"({res['ndiff']}/{res['ntotal']} differ); the legacy "
            "compatibility semantics moved")
    print(_line("unbound both arms (must stay bit-exact)", ok,
                f"ndiff={res['ndiff']}/{res['ntotal']}"))
    res = compare(ref_bound, got_unbound)
    differs = not res["bitexact"]
    if not differs:
        failures.append(
            "a streamed arm with NO binding matched the bound reference; "
            "this gate cannot see a dropped clock binding at all")
    print(_line("dropped binding on the streamed arm (must differ)", differs,
                f"ndiff={res['ndiff']}/{res['ntotal']}, "
                f"max|d|={res['max_abs']:.4g}"))
    del ref_bound, ref_unbound, got_unbound
    cp.get_default_memory_pool().free_all_blocks()

    # ------------------------------------------------------- the negatives
    print()
    print("-- EVERY NEGATIVE CONTROL (each MUST differ)")
    negatives = [
        ("geography rebuilt per tile", dict(geography=False)),
        ("elapsed_seconds not carried", dict(carry_clock=False)),
        ("boundary tables not windowed", dict(window_tables=False)),
        ("periodic closing faces", dict(periodic_faces=True)),
        # HALF the dependency radius, not one cell below it.  The minimum
        # halo that happens to pass is a MEASUREMENT -- it grows with step
        # count, shrinks with domain size and differs by GPU -- so a control
        # pinned to it tests the measurement rather than the mechanism.  The
        # join lane measured halo 14 failing at this exact geometry on the
        # other card of this box; it is bit-exact here.  The margin is
        # reported below as data; the CONTROL is a halo the dependency
        # argument says cannot work.
        (f"halo {halo // 2} (half the dependency radius)",
         dict(halo=halo // 2)),
    ]
    for label, kwargs in negatives:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                got = streamed(cfg, NSTEPS, TX, TY, boundaries=bnd, **kwargs)
            res = compare(ref, got)
            differs = (not res["bitexact"]) or res["nonfinite"] > 0
            detail = (f"ndiff={res['ndiff']}/{res['ntotal']}, "
                      f"max|d|={res['max_abs']:.4g}, "
                      f"nonfinite={res['nonfinite']}")
            del got
        except Exception as exc:                      # noqa: BLE001
            differs, detail = True, f"refused: {type(exc).__name__}"
        if not differs:
            failures.append(f"negative control '{label}' did not fire")
        print(_line(label, differs, detail))
        cp.get_default_memory_pool().free_all_blocks()

    # ------------------------------------------------- the margin, as data
    print()
    print("-- THE HALO MARGIN, MEASURED AND NEVER ASSERTED")
    print(f"   (the operative halo is harness.halo_radius = {halo}; what "
          "follows is where")
    print("    this geometry and this step count happen to break, which is "
          "not a rule)")
    for h in range(halo, max(halo - 6, 1), -1):
        with warnings.catch_warnings():
            # The short-halo warning is the point of this sweep, not news.
            warnings.simplefilter("ignore", RuntimeWarning)
            got = streamed(cfg, NSTEPS, TX, TY, boundaries=bnd, halo=h)
        res = compare(ref, got)
        verdict = ("bit-exact" if res["bitexact"] else
                   f"DIFFERS {res['ndiff']}/{res['ntotal']}, "
                   f"max|d|={res['max_abs']:.4g}")
        print(f"    halo {h:2d}  {verdict}")
        del got
        cp.get_default_memory_pool().free_all_blocks()

    # ------------------------------------- the true edges must be essential
    print()
    print("-- THE TRUE-EDGE TABLES REACH THE ANSWER")
    scaled = _scaled_boundaries(bnd, 1.000001)
    got = streamed(cfg, NSTEPS, TX, TY, boundaries=scaled)
    res = compare(ref, got)
    differs = not res["bitexact"]
    if not differs:
        failures.append("true-edge tables scaled by 1e-6 changed nothing; "
                        "the forcing is not reaching the interior")
    print(_line("true-edge tables x1.000001", differs,
                f"ndiff={res['ndiff']}/{res['ntotal']}, "
                f"max|d|={res['max_abs']:.4g}"))
    del got, scaled
    got = streamed(cfg, NSTEPS, TX, TY,
                   boundaries=_scaled_boundaries(bnd, 1.0))
    res = compare(ref, got)
    ok = res["bitexact"]
    if not ok:
        failures.append("re-materialising the tables at x1.0 changed the "
                        "answer; the scaling control is not clean")
    print(_line("true-edge tables x1.0 (must be exact)", ok,
                f"ndiff={res['ndiff']}"))

    print()
    print("=" * 78)
    if failures:
        print(f"JOIN GATE FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  * {f}")
        return 1
    print("JOIN GATE PASSED -- a real forecast configuration streams "
          "bit-exact, every negative control fired.")
    return 0


def _scaled_boundaries(bnd, factor: float):
    """``bnd`` with every side value scaled, for the true-edge control."""
    from woof.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                         LateralBoundaries, SideBoundary)

    intervals = []
    for iv in bnd.intervals:
        fields_out = {}
        for name, fb in iv.fields.items():
            sides = {}
            for side_name in ("west", "east", "south", "north"):
                side = getattr(fb, side_name)
                sides[side_name] = SideBoundary(
                    np.ascontiguousarray(np.asarray(side.value) * factor),
                    np.ascontiguousarray(np.asarray(side.tendency)))
            fields_out[name] = FieldBoundary(**sides)
        intervals.append(BoundaryInterval(iv.start_seconds, iv.end_seconds,
                                          fields_out))
    return LateralBoundaries(tuple(intervals), bnd.spec_bdy_width,
                             bnd.spec_zone, bnd.relax_zone)


if __name__ == "__main__":
    raise SystemExit(main())
