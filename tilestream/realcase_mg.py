"""The REAL forced decomposition gate: an ERA5 case through MultiGPUDomain.

:func:`tilestream.multigpu.gate_forced` proves the specified-BC decomposition
on a synthetic Lambert case whose forcing is two seeded harness states.  That
is the right instrument for the geometry and the windowing arithmetic, and it
is silent on the question a user actually asks: does a REAL analysis -- ERA5
ingested through the production preparation, full physics attached, real
terrain-following base state, real Davies relaxation toward real analysed air
-- hold the SAME bits decomposed that it holds resident?

This module asks exactly that, with the same three-part shape the streamed
real gate (:mod:`tilestream.run_realcase` ``gate``) uses:

1.  The case is PREPARED ONCE (``realcase.prepare``, the production
    ``prepare_root_experiment_case``), settled one step so every lazily
    allocated carrier exists, and its full carrier inventory, geography
    inventory and scalars are captured to host.  That capture feeds every arm,
    so any difference between arms is the decomposition's.

2.  The RESIDENT arm is ``dycore.step`` on the whole domain with the domain's
    own attached boundaries -- what a single-GPU production run of this case
    is -- and its digest is the reference.

3.  Each DECOMPOSED arm rebuilds the same start through
    :class:`tilestream.multigpu.MultiGPUDomain`: rank buffers built by
    ``realcase.tile_state_builder`` (the case's own vertical coordinate and
    base-state arithmetic), the domain's REAL geography scattered per rank
    with the same gather rectangles ``load_from_host`` uses, the domain's
    ``LateralBoundaries`` windowed per rank by the constructor (real tables
    on true edges, inert tables on seams), carriers loaded from the shared
    host capture, scalars imposed, and the run stepped in lock-step.  The
    assembled interiors must digest bit-identically to the resident arm.

Rank count and device count are independent: ``--device 0`` runs every rank
on one card, which gates the full geometry, windowing and forcing arithmetic
(the transport degenerates to same-device copies).  That is the proven
pattern from the periodic gate and it is what this module runs by default.

NEGATIVE CONTROLS, four of them, because a bit-exact green on real data is
exactly the shape of good news this project has been wrong about before:

``poison``    seam tables poisoned with a large finite value MUST still
              match -- the halo pad, not luck, quarantines seam-side
              boundary fiction (the ``realcase.py`` proof shape).
``scaled``    every TRUE side table scaled by 1.000001 MUST differ -- an
              arm that matches because the forcing never reached the answer
              is a green light on nothing.
``skip``      one edge rank's boundary application suppressed (the dycore's
              ``apply_state_lateral_boundaries`` no-opped for that rank's
              state only) MUST differ -- the per-rank application is
              essential, in the direction of doing less.
``double``    another edge rank's application run twice per RK stage MUST
              differ -- essential in the direction of doing more.  The
              spec-zone overwrite is idempotent, so what this control
              measures is the relaxation-zone blend, which is not.

Run::

    python -m tilestream.realcase_mg gate --config CFG            # 1x1, 1x2, 2x2
    python -m tilestream.realcase_mg gate --config CFG --no-controls
    python -m tilestream.realcase_mg forecast --config CFG --grid 1x2 \
        --hours 6 --wrfout DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def _digest(arrays) -> str:
    """Order-independent digest over a host carrier inventory (16 hex)."""
    from tilestream import physics_inventory as physinv

    per = physinv.field_digests(arrays)
    h = hashlib.sha256()
    for name in sorted(per):
        h.update(name.encode())
        h.update(bytes.fromhex(per[name]))
    return h.hexdigest()[:16]


def _host_copy(mapping) -> dict:
    import cupy as cp

    return {k: (cp.asnumpy(v).copy() if isinstance(v, cp.ndarray)
                else np.asarray(v).copy())
            for k, v in mapping.items()}


def scatter_geography(dom, geo_home: dict) -> int:
    """Install the DOMAIN's real geography into every rank, windowed.

    The rank buffers were built on ``harness.neutral_geography`` -- correct
    shapes, wrong world.  The streamed lane fixes that per tile by gathering
    ``driver.geography_inventory`` from the geography store; this is the same
    fix applied once per rank, with the same gather rectangles
    ``load_from_host`` uses, so halo cells get the neighbour's real geography
    and clamped true edges get the domain's own.

    The DOMAIN's ``has_msf``/``rotational`` flags are imposed alongside the
    arrays, exactly as every other install site does (``mgstream``,
    ``decomp``, ``bench``, ``restart_stream`` -- all via
    ``driver.geography_scalars``).  Writing the arrays alone is the bypass
    ``set_map_coriolis`` warns about: the rank keeps the flags its
    neutral-geography construction derived (both ``False``), every
    msf-weighted dycore path degenerates to identity and the
    Coriolis+curvature kernel never launches -- ~0.3% everywhere in the
    momentum/theta tendencies on this 24 km Lambert case, bit-reproducible,
    NaN-free, and wrong.
    """
    import cupy as cp

    from tilestream import driver as _driver
    from tilestream.multigpu import MultiGPUError, _variant_of

    flags = _driver.geography_scalars(geo_home)
    n = None
    for g, dev in enumerate(dom.devices):
        sp = dom.specs[g]
        inv = _driver.geography_inventory(dom.states[g])
        missing = set(inv) - set(geo_home)
        if missing:
            raise MultiGPUError(
                f"rank {g} carries geography {sorted(missing)[:8]} the "
                "domain capture does not -- the rank builder and the "
                "prepared case disagree about the scheme set")
        with cp.cuda.Device(dev):
            for name, dst in inv.items():
                src = np.asarray(geo_home[name])
                if dst.ndim < 2:
                    dst[...] = cp.asarray(src) \
                        if isinstance(dst, cp.ndarray) else src
                    continue
                v = _variant_of(name, src, dom.nz, dom.ny, dom.nx)
                for t in sp.gather(v):
                    chunk = np.ascontiguousarray(src[t.full_key])
                    dst[t.tile_key] = cp.asarray(chunk) \
                        if isinstance(dst, cp.ndarray) else chunk
            cp.cuda.runtime.deviceSynchronize()
        for key, value in flags.items():
            setattr(dom.states[g], key, value)
        n = len(inv) if n is None else n
    return int(n or 0)


def _device_list(spec, nranks: int) -> list[int]:
    """``--device`` as a card list ranks round-robin over.

    ``"0"`` is the historical behaviour -- every rank on one card, which
    gates the geometry and forcing arithmetic completely.  ``"0,1"`` puts
    rank ``r`` on ``cards[r % len(cards)]``, so the seam exchange crosses a
    PHYSICAL device boundary -- the same contract ``test_forced_gate``'s
    ``DEVICES`` uses.
    """
    cards = [int(v) for v in str(spec).split(",") if str(v).strip() != ""]
    if not cards:
        raise SystemExit("--device lists no cards")
    return [cards[r % len(cards)] for r in range(int(nranks))]


def _rank_factory(exp, boundaries):
    """A ``(sub_cfg, spec) -> state`` MultiGPUDomain factory for a REAL case.

    ``realcase.tile_state_builder`` needs SOME rank-shaped boundary tables
    attached before its warmup step (``cfg.specified`` refuses a bare step);
    the constructor rebinds the rank's true windowed tables afterwards, so
    the warmup tables' values never survive -- only their layout matters.
    """
    from woof.core.streaming import window_boundaries
    from tilestream import realcase

    def make(sub_cfg, *, spec):
        warm = window_boundaries(boundaries, spec, seam="zeros")
        state, _driver = realcase.tile_state_builder(exp, warm)(sub_cfg)
        return state

    return make


def _sabotaged_run(dom, steps: int, *, mode: str, victim: int) -> None:
    """Run with rank ``victim``'s boundary application broken by ``mode``.

    The patch is identity-gated on the victim's ``DomainState``, so the other
    ranks -- and any resident state alive in the process -- apply their
    boundaries exactly as before.  ``skip`` answers "is the application
    essential at all"; ``double`` answers "is the CADENCE of application
    essential" (one blend per RK stage, not two).
    """
    import woof.core.dycore as dyc

    real = dyc.apply_state_lateral_boundaries
    target = dom.states[int(victim)]

    def wrapped(state, *a, **kw):
        if state is target:
            if mode == "skip":
                return None
            real(state, *a, **kw)
        return real(state, *a, **kw)

    dyc.apply_state_lateral_boundaries = wrapped
    try:
        dom.run(int(steps), step_mode="sequential", exchange_mode="blocking")
    finally:
        dyc.apply_state_lateral_boundaries = real


def _fired(dom, start_scalars) -> dict:
    """Per-rank physics fire counts since the shared start."""
    from tilestream import physics_inventory as physinv

    base = start_scalars.get("call_counts", {})
    out = {}
    for g, st in enumerate(dom.states):
        now = physinv.carrier_scalars(st).get("call_counts", {})
        out[f"rank{g}"] = {k: int(v) - int(base.get(k, 0))
                           for k, v in now.items()
                           if int(v) != int(base.get(k, 0))}
    return out


def _mg_arm(cfg, exp, boundaries, start, start_scalars, geo_home, grid, *,
            steps: int, device="0", seam: str = "zeros", bnd=None,
            sabotage=None, victim: int = 0):
    import cupy as cp

    from tilestream import multigpu, physics_inventory as physinv

    gy, gx = int(grid[0]), int(grid[1])
    devs = _device_list(device, gy * gx)
    dom = multigpu.MultiGPUDomain(
        cfg, grid=(gy, gx), devices=devs,
        boundaries=(boundaries if bnd is None else bnd), seam=seam,
        state_factory=_rank_factory(exp, boundaries),
        inventory_fn=physinv.carrier_inventory)
    transport = dom.transport_report() if len(set(devs)) > 1 else []
    if transport:
        print(dom.describe_transport())
    ngeo = scatter_geography(dom, geo_home)
    dom.load_from_host(start)
    for st in dom.states:
        physinv.set_carrier_scalars(st, start_scalars)
    try:
        if sabotage:
            _sabotaged_run(dom, steps, mode=sabotage, victim=victim)
        else:
            dom.run(int(steps), step_mode="sequential",
                    exchange_mode="blocking")
    except Exception:
        # A sabotaged arm may legitimately refuse (non-finite physics
        # inputs once the boundary application is broken); release the
        # arm's device memory so the next arm starts clean, then let the
        # caller decide what the refusal means.
        dom.close()
        for d in set(devs):
            with cp.cuda.Device(d):
                cp.get_default_memory_pool().free_all_blocks()
        raise
    host = dom.assemble_host()
    fired = _fired(dom, start_scalars)
    halo = int(dom.halo)
    dom.close()
    for d in set(devs):
        with cp.cuda.Device(d):
            cp.get_default_memory_pool().free_all_blocks()
    return _digest(host), host, fired, ngeo, halo, transport


def _prepare_settled(config: str):
    """``(prep, exp, cfg, state, boundaries, start, scalars, geo)`` settled."""
    import cupy as cp

    from woof.core.dycore import step
    from tilestream import driver, physics_inventory as physinv, realcase

    prep, exp, _peak = realcase.prepare(config)
    cfg = prep.cfg
    state = prep.initial_result.state
    boundaries = state.lateral_boundaries
    if boundaries is None:
        raise SystemExit("the prepared case carries no LateralBoundaries -- "
                         "this gate is about specified forcing and cannot "
                         "run without it")
    # Settle the lazily-allocated carriers before anything is copied (the
    # run_realcase gate's own discipline).  Both arms start from the settled
    # state, so the settle step is shared, not a difference.
    step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()
    start = _host_copy(physinv.carrier_inventory(state))
    scalars = physinv.carrier_scalars(state)
    geo = _host_copy(driver.geography_inventory(state))
    return prep, exp, cfg, state, boundaries, start, scalars, geo


def _diff_report(mono: dict, host: dict, limit: int = 12) -> list[str]:
    from tilestream import physics_inventory as physinv

    pm, ph = physinv.field_digests(mono), physinv.field_digests(host)
    differ = sorted(k for k in pm if pm[k] != ph.get(k))
    for name in differ[:limit]:
        a = np.asarray(mono[name], dtype=np.float64)
        b = np.asarray(host[name], dtype=np.float64)
        d = np.abs(a - b)
        print(f"      {name:38s} max|d| {np.nanmax(d):.3e}, "
              f"{int((d > 0).sum())} of {d.size} cells")
    return differ


def stage_gate(args) -> None:
    import cupy as cp

    from woof.core.dycore import step
    from tilestream import physics_inventory as physinv
    from tilestream.multigpu import _scaled_boundaries, forced_halo

    print(f"=== REAL FORCED GATE: {args.config}, {args.steps} steps ===")
    (prep, exp, cfg, state, boundaries,
     start, scalars, geo) = _prepare_settled(args.config)
    print(f"    domain {cfg.nx}x{cfg.ny}x{cfg.nz} dx={cfg.dx/1000:.0f} km, "
          f"dt={cfg.dt:.0f} s, specified={cfg.specified} "
          f"(spec_zone={cfg.spec_zone}, relax_zone={cfg.relax_zone}), "
          f"forced halo {forced_halo(cfg)}")
    print(f"    start digest {_digest(start)}  ({len(start)} carriers, "
          f"{len(geo)} geography arrays)  clock {scalars['elapsed_seconds']}")

    t0 = time.perf_counter()
    for _ in range(int(args.steps)):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()
    mono = _host_copy(physinv.carrier_inventory(state))
    mono_scalars = physinv.carrier_scalars(state)
    h_mono = _digest(mono)
    base_counts = scalars.get("call_counts", {})
    mono_fired = {k: int(v) - int(base_counts.get(k, 0))
                  for k, v in mono_scalars.get("call_counts", {}).items()
                  if int(v) != int(base_counts.get(k, 0))}
    print(f"    RESIDENT {args.steps} steps in {time.perf_counter()-t0:.1f}s"
          f"  digest {h_mono}  fired {mono_fired}")

    grids = [tuple(int(v) for v in g.split("x"))
             for g in args.grids.split(",")]
    results = {"config": args.config, "steps": int(args.steps),
               "hash_start": _digest(start), "hash_resident": h_mono,
               "resident_fired": mono_fired, "runs": {}, "controls": {}}
    ok = True
    for grid in grids:
        t0 = time.perf_counter()
        h, host, fired, ngeo, halo, transport = _mg_arm(
            cfg, exp, boundaries, start, scalars, geo, grid,
            steps=args.steps, device=args.device)
        same = (h == h_mono)
        ok = ok and same
        key = f"{grid[0]}x{grid[1]}"
        results["runs"][key] = {"hash": h, "match": same, "halo": halo,
                                "fired": fired}
        if transport:
            results["runs"][key]["transport"] = transport
        print(f"    {key} ({grid[0]*grid[1]} rank(s), card {args.device}, "
              f"halo {halo}, {ngeo} geo arrays/rank): {h} "
              f"{'MATCH' if same else 'DIFFER'} "
              f"[{time.perf_counter()-t0:.1f}s]  fired {fired}")
        if not same:
            results["runs"][key]["differ"] = _diff_report(mono, host)

    if args.controls:
        multi = next((g for g in grids if g[0] * g[1] > 1), (1, 2))

        h, _, _, _, _, _ = _mg_arm(cfg, exp, boundaries, start, scalars,
                                   geo, multi, steps=args.steps,
                                   device=args.device, seam="poison")
        poison_ok = (h == h_mono)
        results["controls"]["poison_seam_matches"] = poison_ok
        print(f"    control poison seams {multi[0]}x{multi[1]}: {h} "
              f"{'MATCH (quarantine holds)' if poison_ok else 'DIFFER -- SEAM FICTION REACHED OWNED CELLS'}")

        def must_differ(label: str, **kw):
            """A must-DIFFER arm fires by digest or by LOUD refusal.

            With the geography flags imposed the decomposed run carries the
            real Coriolis/map-factor dynamics, and an edge rank whose
            boundary application is broken for the whole run no longer
            drifts quietly to a different answer -- its unforced frame goes
            non-finite and the YSU input validation refuses.  That refusal
            IS the control firing (the sabotage could not be missed); only
            a sabotaged arm that silently REPRODUCES the resident digest is
            a dead control.
            """
            try:
                h, _, _, _, _, _ = _mg_arm(cfg, exp, boundaries, start,
                                           scalars, geo, multi,
                                           steps=args.steps,
                                           device=args.device, **kw)
            except FloatingPointError as exc:
                print(f"    control {label}: NON-FINITE REFUSAL -- the "
                      f"sabotaged run cannot even complete "
                      f"({str(exc).splitlines()[0]})")
                return "nan-refusal", True
            return h, (h != h_mono)

        h, scaled_fired = must_differ(
            f"scaled edges {multi[0]}x{multi[1]}",
            bnd=_scaled_boundaries(boundaries, 1.000001))
        results["controls"]["scaled_edge_differs"] = scaled_fired
        if h != "nan-refusal":
            print(f"    control scaled edges {multi[0]}x{multi[1]}: {h} "
                  f"{'DIFFER (forcing is live)' if scaled_fired else 'MATCH -- FORCING NEVER REACHED THE ANSWER'}")

        h, skip_fired = must_differ("skip rank0 application",
                                    sabotage="skip", victim=0)
        results["controls"]["skip_rank0_differs"] = skip_fired
        if h != "nan-refusal":
            print(f"    control skip rank0 application: {h} "
                  f"{'DIFFER (application is essential)' if skip_fired else 'MATCH -- THE APPLICATION DID NOTHING'}")

        last = multi[0] * multi[1] - 1
        h, double_fired = must_differ(f"double rank{last} application",
                                      sabotage="double", victim=last)
        results["controls"][f"double_rank{last}_differs"] = double_fired
        if h != "nan-refusal":
            print(f"    control double rank{last} application: {h} "
                  f"{'DIFFER (cadence is essential)' if double_fired else 'MATCH -- A SECOND APPLICATION IS INVISIBLE'}")

        ok = ok and poison_ok and scaled_fired and skip_fired and double_fired

    results["ok"] = bool(ok)
    print("RESULT " + json.dumps(results))
    if not ok:
        raise SystemExit(1)


def stage_forecast(args) -> None:
    import cupy as cp

    from woof.core.dycore import step
    from tilestream import (multigpu, physics_inventory as physinv,
                            realcase, realcase_wrfout)

    grid = tuple(int(v) for v in args.grid.split("x"))
    print(f"=== REAL FORCED FORECAST: {args.config}, grid {args.grid}, "
          f"{args.hours:.0f} h, wrfout -> {args.wrfout} ===")
    (prep, exp, cfg, state, boundaries,
     start, scalars, geo) = _prepare_settled(args.config)

    history = realcase_wrfout.History(
        prep, exp, args.wrfout, slab_rows=int(args.refl_slab),
        title=args.title)
    history.capture(state, cfg)

    store = {k: np.ascontiguousarray(v) for k, v in start.items()}

    class _Case:
        pass

    shim = _Case()
    shim.cfg, shim.store, shim.geo_store = cfg, store, geo
    # This lane's store is the plain carrier inventory -- no REFL_10CM slot
    # -- so the composite is the slab recompute, for the schemes the
    # operator dispatches.  Said out loud because ``composite_reflectivity``
    # asks the case, not the store: a case that claimed "computed" with no
    # slot would be refused, and one that stayed silent would be a shim
    # missing a field rather than a lane making a choice.
    shim.refl_stash = "absent"
    history.bind(shim)

    elapsed = float(scalars["elapsed_seconds"])
    devs = _device_list(args.device, grid[0] * grid[1])
    dom = multigpu.MultiGPUDomain(
        cfg, grid=grid, devices=devs,
        boundaries=boundaries, seam="zeros",
        state_factory=_rank_factory(exp, boundaries),
        inventory_fn=physinv.carrier_inventory)
    if len(set(devs)) > 1:
        print(dom.describe_transport())
    scatter_geography(dom, geo)
    dom.load_from_host(start)
    for st in dom.states:
        physinv.set_carrier_scalars(st, scalars)

    path = history.write(elapsed)
    print(f"    f000 {path.name}  ({path.stat().st_size/2**20:.0f} MiB)")

    steps_per_dump = max(1, int(round(args.every * 3600.0 / cfg.dt)))
    ndumps = max(1, int(round(args.hours / args.every)))
    wall0 = time.perf_counter()
    for i in range(ndumps):
        t0 = time.perf_counter()
        dom.run(steps_per_dump, step_mode="sequential",
                exchange_mode="blocking")
        host = dom.assemble_host()
        for k in store:
            store[k][...] = host[k]
        elapsed += steps_per_dump * float(cfg.dt)
        path = history.write(elapsed)
        refl = realcase.composite_reflectivity(shim,
                                               slab_rows=int(args.refl_slab))
        print(f"    t+{elapsed/3600:.2f} h  {path.name}  refl max "
              f"{np.nanmax(refl):.1f} dBZ  "
              f"[{time.perf_counter()-t0:.1f}s wall]")
        sys.stdout.flush()
    dom.close()
    wall = time.perf_counter() - wall0
    h_final = _digest(store)
    print(f"    forecast complete: {args.hours:.1f} h decomposed over "
          f"{grid[0]*grid[1]} ranks in {wall/60:.1f} min  "
          f"final digest {h_final}")
    print("RESULT " + json.dumps({
        "grid": args.grid, "hours": args.hours, "devices": devs,
        "hash_final": h_final,
        "wrfouts": [p.name for p in history.written],
        "wall_seconds": wall}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["gate", "forecast"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=24,
                    help="gate steps; the default covers a radt=24 min "
                         "radiation cadence at dt=120 s twice")
    ap.add_argument("--grids", default="1x1,1x2,2x2",
                    help="comma list of GYxGX rank grids for the gate")
    ap.add_argument("--grid", default="1x2",
                    help="rank grid for the forecast")
    ap.add_argument("--device", default="0",
                    help="card list ranks round-robin over: 0 is the "
                         "single-card gate, 0,1 puts the seam across a "
                         "physical device boundary")
    ap.add_argument("--no-controls", dest="controls", action="store_false")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--every", type=float, default=1.0,
                    help="hours between wrfout frames")
    ap.add_argument("--refl-slab", type=int, default=48)
    ap.add_argument("--wrfout", default="mg_wrfout")
    ap.add_argument("--title", default="WOOF-ArWen specified-BC decomposed run")
    args = ap.parse_args(argv)
    if args.stage == "gate":
        stage_gate(args)
    else:
        stage_forecast(args)
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
