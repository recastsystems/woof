"""Drive a REAL case through the streaming path, and measure every stage.

    python -m tilestream.run_realcase gate     --config CFG --tile 64
    python -m tilestream.run_realcase ceiling  --config CFG
    python -m tilestream.run_realcase forecast --config CFG --tile 256 \
        --hours 12 --every 30 --out DIR

``gate`` is the one that decides whether anything else here is worth
believing: the SAME real initial state and the SAME real boundary series,
integrated monolithically and streamed, compared carrier by carrier.  It is
the specified-boundary analogue of ``test_gate``'s periodic ladder and it is
the only evidence that :func:`tilestream.realcase.halo_for`'s quarantine
argument is arithmetic rather than optimism.

``ceiling`` is the control a "vanilla cannot run this" caption needs: it
tries to prepare and step the same configuration resident and reports how it
refuses, in this run's own log, on this run's own card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np


def _vram():
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo()
    return free / 2 ** 30, total / 2 ** 30


def _banner(title: str) -> None:
    free, total = _vram()
    from tilestream import hoststore
    mem = hoststore.host_memory()
    print(f"\n=== {title} ===")
    print(f"    VRAM {free:.2f}/{total:.2f} GiB free | host "
          f"{mem['available'] / 2**30:.1f}/{mem['total'] / 2**30:.1f} GiB")
    sys.stdout.flush()


def _prepare(args):
    """The case's preparation, low-water by default.

    ``--prepare inorder`` restores ``runtime.prepare_root_experiment_case``'s
    own ordering; it is the control that shows what the reordering is worth,
    and on any domain that fits it must produce the same case.
    """
    from tilestream import realcase
    if args.prepare == "inorder":
        return realcase.prepare(args.config)
    if args.prepare == "slabbed":
        return realcase.prepare_slabbed(args.config,
                                        rows_per_slab=args.prepare_rows)
    return realcase.prepare_low_water(args.config)


def _digest(arrays) -> str:
    from tilestream import physics_inventory as physinv
    per = physinv.field_digests(arrays)
    h = hashlib.sha256()
    for name in sorted(per):
        h.update(name.encode())
        h.update(bytes.fromhex(per[name]))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------

def stage_gate(args) -> None:
    """Monolithic specified vs streamed specified, on real data.

    Both arms start from the SAME prepared state -- prepared once, digested,
    then copied -- so any difference is the streaming path's and not the
    ingest's.  The monolithic arm is the reference by construction: it is
    ``dycore.step`` on the whole domain with the domain's own attached
    boundaries, which is what a resident run of this case is.
    """
    import cupy as cp

    from tilestream import driver, gather, physics_inventory as physinv
    from tilestream import realcase

    _banner(f"GATE: real specified boundaries, monolithic vs streamed, "
            f"{args.steps} steps")
    prep, exp, _peak = _prepare(args)
    cfg = prep.cfg
    state = prep.initial_result.state
    boundaries = state.lateral_boundaries
    print(f"    specified={cfg.specified} open_x={cfg.open_x} "
          f"open_y={cfg.open_y} spec_zone={cfg.spec_zone} "
          f"relax_zone={cfg.relax_zone} spec_bdy_width={cfg.spec_bdy_width}")
    print(f"    {len(boundaries.intervals)} boundary intervals, "
          f"{sorted(boundaries.intervals[0].fields)}")

    # Settle the lazily-allocated carriers before anything is copied.
    from woof.core.dycore import step
    step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()

    start = {k: cp.asnumpy(v).copy()
             for k, v in physinv.carrier_inventory(state).items()}
    start_scalars = physinv.carrier_scalars(state)
    print(f"    start scalars {start_scalars}")
    geo_home = {k: (cp.asnumpy(v).copy() if isinstance(v, cp.ndarray)
                    else np.asarray(v).copy())
                for k, v in driver.geography_inventory(state).items()}
    print(f"    start digest {_digest(start)}  "
          f"{len(start)} carriers, {len(geo_home)} geography arrays")

    t0 = time.perf_counter()
    for _ in range(args.steps):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()
    mono = {k: cp.asnumpy(v).copy()
            for k, v in physinv.carrier_inventory(state).items()}
    mono_scalars = physinv.carrier_scalars(state)
    print(f"    MONOLITHIC {args.steps} steps in "
          f"{time.perf_counter() - t0:.1f}s  digest {_digest(mono)}")
    # ``del state`` alone is NOT a release: ``prep.initial_result.state``
    # still holds the whole resident arm -- its carriers plus every
    # domain-shaped physics workspace the monolithic steps grew -- and
    # ``prep.final_analysis`` holds the interpolated analysis beside it.
    # Measured on a 15.5 GiB card at 648x576x49: with prep retained the
    # STREAMED arm's tile factory refused inside its warmup radiation at
    # 14.7 GiB allocated; with ``release_prepared`` the same battery leg
    # runs.  The boundary tables survive through ``boundaries`` above,
    # which is the only piece of prep the streamed arm still needs.
    del state
    realcase.release_prepared(prep)
    prep = None

    halo = realcase.halo_for(cfg) if args.halo is None else int(args.halo)
    specs, halo = realcase.plan(cfg, args.tile, args.tile, halo)
    print(f"    halo {halo} = radius {halo - realcase.seam_fiction_width(cfg)} + "
          f"seam fiction {realcase.seam_fiction_width(cfg)};  {len(specs)} "
          f"tiles of {specs[0].cny}x{specs[0].cnx} (non-periodic plan)")

    store = {}
    for name, arr in start.items():
        host = gather.pinned_empty(arr.shape, arr.dtype)
        host[...] = arr
        store[name] = host
    geo_store = {}
    for name, arr in geo_home.items():
        host = gather.pinned_empty(arr.shape, arr.dtype)
        host[...] = arr
        geo_store[name] = host

    binder = realcase.tile_boundary_binder(boundaries, specs, cfg,
                                           verbose=print)
    kwargs = _run_kwargs(cfg, exp, geo_store, binder.per_tile[0])
    kwargs["scalars"] = dict(start_scalars)

    t0 = time.perf_counter()
    report: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver.run_tiled(store, cfg, args.tile, args.tile, halo=halo,
                         nsteps=args.steps, nbuffers=args.nbuffers,
                         periodic=False, write_mode="ring",
                         overlap=args.overlap,
                         tile_hook=binder, report=report, **kwargs)
    cp.cuda.runtime.deviceSynchronize()
    print(f"    STREAMED   {args.steps} steps in "
          f"{time.perf_counter() - t0:.1f}s  digest {_digest(store)}  "
          f"({report['tile_hook_calls']} boundary binds)")

    per_mono = physinv.field_digests(mono)
    per_str = physinv.field_digests(store)
    differ = sorted(k for k in per_mono if per_mono[k] != per_str.get(k))
    print(f"    scalars monolithic {mono_scalars} vs streamed "
          f"{kwargs['scalars']}")
    if differ:
        print(f"    {len(differ)} of {len(per_mono)} carriers DIFFER:")
        for name in differ[:24]:
            a = np.asarray(mono[name], dtype=np.float64)
            b = np.asarray(store[name], dtype=np.float64)
            d = np.abs(a - b)
            scale = max(1e-30, float(np.nanmax(np.abs(a))))
            print(f"      {name:38s} max|d| {np.nanmax(d):.3e} "
                  f"({np.nanmax(d)/scale:.2e} rel), "
                  f"{int((d > 0).sum())} of {d.size} cells")
    else:
        print(f"    BIT-EXACT over all {len(per_mono)} carriers -- the "
              f"quarantine holds")
    print("RESULT " + json.dumps({
        "steps": args.steps, "tile": args.tile, "halo": halo,
        "tiles": len(specs), "carriers": len(per_mono),
        "differ": differ, "bitexact": not differ}))
    if differ:
        raise SystemExit(1)


def _run_kwargs(cfg, exp, geo_store, tile_lb=None) -> dict:
    """``run_tiled`` keywords for a REAL domain.  Three overrides matter.

    ``geography_fn=harness.neutral_geography``
        the buffer's own geography is POISON, overwritten by the gather.
        The default is ``make_geography``, i.e. the per-tile REBUILD, which
        is the gate's negative control and not a forecast setting.

    ``coord_fn``
        the case's own eta table.  See ``harness.make_physics_state``: the
        purely vertical setup arrays are rebuilt by the tile, not gathered,
        so a buffer on the default stretch runs a different vertical grid
        from the domain and nothing checks it.

    ``geography=`` the domain's arrays, already pinned in host RAM.
    """
    from tilestream import driver, realcase

    kwargs = driver.physics_run_kwargs(
        cfg, None, builder=realcase.tile_state_builder(exp, tile_lb))
    kwargs["geography"] = geo_store
    return kwargs


# --------------------------------------------------------------------------
# ceiling
# --------------------------------------------------------------------------

def stage_ceiling(args) -> None:
    """Prepare and step this configuration RESIDENT, and report the refusal."""
    import cupy as cp

    from tilestream import realcase

    _banner(f"CEILING: monolithic prepare + step, {args.config}")
    free, total = _vram()
    verdict = {"config": args.config, "vram_total_gib": total,
               "vram_free_gib": free}
    try:
        prep, exp, peak = _prepare(args)
        verdict["prepare"] = "ran"
        verdict["prepare_peak_gib"] = peak / 2 ** 30
        cfg = prep.cfg
        verdict["nx"], verdict["ny"], verdict["nz"] = cfg.nx, cfg.ny, cfg.nz
        from woof.core.dycore import step
        pool = cp.get_default_memory_pool()
        # Step by step, reporting WHICH step refuses.  A domain whose first
        # step fits and whose second does not is still one vanilla cannot
        # run, but "the second step refused" and "no step ran" are different
        # facts and the log should not conflate them.
        done = 0
        for k in range(args.steps):
            t0 = time.perf_counter()
            step(prep.initial_result.state, cfg)
            cp.cuda.runtime.deviceSynchronize()
            done = k + 1
            verdict[f"step{done}_seconds"] = time.perf_counter() - t0
            verdict["peak_gib"] = pool.total_bytes() / 2 ** 30
            print(f"    monolithic step {done}: "
                  f"{verdict[f'step{done}_seconds']:.1f} s, pool "
                  f"{verdict['peak_gib']:.2f} GiB", flush=True)
        verdict["step"] = "ran"
        verdict["steps_run"] = done
        print(f"    MONOLITHIC RAN {done} steps "
              f"-- this domain is NOT past the resident ceiling")
    except Exception as exc:                       # noqa: BLE001 - reporting
        verdict["refused"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        verdict["refused_on_step"] = verdict.get("steps_run", 0) + 1
        print(f"    MONOLITHIC REFUSED: {verdict['refused']}")
    print("RESULT " + json.dumps(verdict))


# --------------------------------------------------------------------------
# twin
# --------------------------------------------------------------------------

def stage_twin(args) -> None:
    """RESIDENT and STREAMED wrfout histories of one case, from one prepare.

    The gate proves the carriers identical; this stage produces the thing a
    person can LOOK at.  Both arms start from the same settled initial state
    (the gate's own discipline), both write frames through the same
    :class:`tilestream.realcase_wrfout.History` writer at the same valid
    times, and the only difference between the two output directories is the
    transport: ``dycore.step`` on the whole resident domain against
    ``run_tiled`` out of the pinned host store (``--overlap`` selects the
    streamed arm's mode; the default is the build's default).  Pixel-equal
    galleries from the two directories are therefore the streaming path's
    visual receipt, not a coincidence of rendering.

    The resident arm runs first, while the prepared state exists; the
    streamed arm is rebuilt from the start snapshot afterwards, exactly as
    ``stage_gate`` does it.
    """
    import types

    import cupy as cp

    from tilestream import driver, gather, physics_inventory as physinv
    from tilestream import realcase, realcase_wrfout

    out_dir = Path(args.out)
    res_dir = out_dir / "resident"
    str_dir = out_dir / "streamed"
    _banner(f"TWIN: resident vs streamed wrfout history, {args.hours:.1f} h")

    prep, exp, _peak = _prepare(args)
    cfg = prep.cfg
    state = prep.initial_result.state
    boundaries = state.lateral_boundaries
    title = args.title or (
        f"WOOF-ArWen twin receipt {cfg.nx}x{cfg.ny} dx={cfg.dx/1000:.1f} km "
        f"-- {Path(args.config).stem}")

    histories = {
        "resident": realcase_wrfout.History(
            prep, exp, res_dir, slab_rows=args.refl_slab, title=title),
        "streamed": realcase_wrfout.History(
            prep, exp, str_dir, slab_rows=args.refl_slab, title=title),
    }

    from woof.core.dycore import step
    step(state, cfg)                     # settle lazy carriers, both arms
    cp.cuda.runtime.deviceSynchronize()
    for h in histories.values():
        h.capture(state, cfg)

    start = {k: cp.asnumpy(v).copy()
             for k, v in physinv.carrier_inventory(state).items()}
    start_scalars = physinv.carrier_scalars(state)
    geo_home = {k: (cp.asnumpy(v).copy() if isinstance(v, cp.ndarray)
                    else np.asarray(v).copy())
                for k, v in driver.geography_inventory(state).items()}
    print(f"    start scalars {start_scalars}; {len(start)} carriers, "
          f"{len(geo_home)} geography arrays", flush=True)

    steps_per_frame = max(1, int(round(args.wrfout_every * 60.0 / cfg.dt)))
    nframes = max(1, int(round(args.hours * 3600.0
                               / (args.wrfout_every * 60.0))))
    print(f"    {nframes} frames every {steps_per_frame} steps "
          f"(dt {cfg.dt:.0f} s)", flush=True)

    # -- RESIDENT, while the prepared state exists ----------------------
    res_store = {k: v.copy() for k, v in start.items()}
    res_case = types.SimpleNamespace(cfg=cfg, store=res_store,
                                     geo_store=geo_home)
    histories["resident"].bind(res_case)
    histories["resident"].write(0.0)
    t0 = time.perf_counter()
    for i in range(nframes):
        for _ in range(steps_per_frame):
            step(state, cfg)
        cp.cuda.runtime.deviceSynchronize()
        for k, v in physinv.carrier_inventory(state).items():
            res_store[k][...] = cp.asnumpy(v)
        elapsed = (i + 1) * steps_per_frame * float(cfg.dt)
        path = histories["resident"].write(elapsed)
        print(f"    [resident] t+{elapsed/3600:.2f} h -> {path.name} "
              f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    res_digest = _digest(res_store)
    res_scalars = physinv.carrier_scalars(state)
    print(f"    RESIDENT final digest {res_digest}  scalars {res_scalars}",
          flush=True)
    # The gate's own release discipline: drop every reference to the
    # resident state (prep holds one too, and ``final_analysis`` beside
    # it) and return the cached blocks.  Both History objects captured
    # everything they read from prep at construction, so the release
    # costs the writers nothing.
    del state
    realcase.release_prepared(prep)

    # -- STREAMED, from the same snapshot --------------------------------
    store = {}
    for name, arr in start.items():
        host = gather.pinned_empty(arr.shape, arr.dtype)
        host[...] = arr
        store[name] = host
    geo_store = {}
    for name, arr in geo_home.items():
        host = gather.pinned_empty(arr.shape, arr.dtype)
        host[...] = arr
        geo_store[name] = host

    halo = realcase.halo_for(cfg) if args.halo is None else int(args.halo)
    specs, halo = realcase.plan(cfg, args.tile, args.tile, halo)
    print(f"    {len(specs)} tiles of {specs[0].cny}x{specs[0].cnx}, halo "
          f"{halo}, nbuffers {args.nbuffers}, overlap {args.overlap}",
          flush=True)
    binder = realcase.tile_boundary_binder(boundaries, specs, cfg,
                                           verbose=print)
    kwargs = _run_kwargs(cfg, exp, geo_store, binder.per_tile[0])
    kwargs["scalars"] = dict(start_scalars)

    str_case = types.SimpleNamespace(cfg=cfg, store=store,
                                     geo_store=geo_store)
    histories["streamed"].bind(str_case)
    histories["streamed"].write(0.0)
    t0 = time.perf_counter()
    for i in range(nframes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            driver.run_tiled(store, cfg, args.tile, args.tile, halo=halo,
                             nsteps=steps_per_frame, nbuffers=args.nbuffers,
                             periodic=False, write_mode="ring",
                             overlap=args.overlap, tile_hook=binder,
                             **kwargs)
        cp.cuda.runtime.deviceSynchronize()
        elapsed = (i + 1) * steps_per_frame * float(cfg.dt)
        path = histories["streamed"].write(elapsed)
        print(f"    [streamed] t+{elapsed/3600:.2f} h -> {path.name} "
              f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    str_digest = _digest(store)
    print(f"    STREAMED final digest {str_digest}  scalars "
          f"{kwargs['scalars']}", flush=True)

    per_res = {k: v for k, v in physinv.field_digests(res_store).items()}
    per_str = physinv.field_digests(store)
    differ = sorted(k for k in per_res if per_res[k] != per_str.get(k))
    verdict = {"hours": args.hours, "frames": nframes + 1,
               "tile": args.tile, "halo": halo, "tiles": len(specs),
               "overlap": args.overlap,
               "resident_digest": res_digest, "streamed_digest": str_digest,
               "differ": differ, "bitexact": not differ,
               "resident_dir": str(res_dir), "streamed_dir": str(str_dir)}
    print("RESULT " + json.dumps(verdict), flush=True)
    if differ:
        raise SystemExit(1)


# --------------------------------------------------------------------------
# pair
# --------------------------------------------------------------------------

def stage_pair(args) -> None:
    """Streamed ``overlap='off'`` against streamed ``overlap='on'``, one
    prepare, past the resident ceiling.

    ``gate`` needs a monolithic reference arm, and past the ceiling that
    arm is exactly what the card refuses (measured here: 864x768x49 and
    960x864x49 both OOM inside the settling step's radiation on 15.5 GiB).
    What the overlap build must still prove at those sizes is ITS OWN
    identity: the ring with the event chain and the ring without it, fed
    the same snapshot through the same window, must land the same digest
    in every carrier.  Both arms restore the same start store and scalars,
    so the only degree of freedom is the ordering machinery under test.
    """
    import copy

    import cupy as cp

    from tilestream import driver, physics_inventory as physinv
    from tilestream import realcase

    _banner(f"PAIR: streamed overlap off vs on, {args.steps} steps")
    prep, exp, _peak = _prepare(args)
    cfg = prep.cfg

    # The overlap_attrib wiring, which is the proven envelope at these
    # sizes: stores in pinned host RAM, no settling step (that is the
    # full-domain step that does not fit; run_tiled's inventory match is
    # the guard), prepared state released before the first arm.
    case = realcase.build_stores(prep, exp, settle=args.settle)
    realcase.release_prepared(prep)
    prep = None

    halo = realcase.halo_for(cfg) if args.halo is None else int(args.halo)
    specs, halo = realcase.plan(cfg, args.tile, args.tile, halo)
    print(f"    halo {halo};  {len(specs)} tiles of "
          f"{specs[0].cny}x{specs[0].cnx}, nbuffers {args.nbuffers}")

    snap = {k: np.asarray(v).copy() for k, v in case.store.items()}
    snap_scalars = copy.deepcopy(case.scalars)

    digests: dict[str, dict] = {}
    whole: dict[str, str] = {}
    scalars_by: dict[str, dict] = {}
    for mode in ("off", "on"):
        for k, v in snap.items():
            case.store[k][...] = v
        arm_scalars = copy.deepcopy(snap_scalars)
        binder = realcase.tile_boundary_binder(case.boundaries, specs, cfg,
                                               verbose=print)
        kwargs = _run_kwargs(cfg, exp, case.geo_store, binder.per_tile[0])
        kwargs["scalars"] = arm_scalars
        # The store carries the REFL_10CM slot, so this arm writes it too --
        # both arms identically, which is the point of an A/B on the
        # transport: a carrier one arm computes and the other does not would
        # differ for a reason that is not the overlap mode.
        kwargs = realcase.reflectivity_run_kwargs(kwargs, case)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            driver.run_tiled(case.store, cfg, args.tile, args.tile,
                             halo=halo, nsteps=args.steps,
                             nbuffers=args.nbuffers, periodic=False,
                             write_mode="ring", overlap=mode,
                             tile_hook=binder, **kwargs)
        cp.cuda.runtime.deviceSynchronize()
        case.reflectivity_swept()
        digests[mode] = physinv.field_digests(case.store)
        whole[mode] = _digest(case.store)
        scalars_by[mode] = arm_scalars
        print(f"    STREAMED overlap={mode:3s} {args.steps} steps in "
              f"{time.perf_counter() - t0:.1f}s  digest {whole[mode]}",
              flush=True)
        cp.get_default_memory_pool().free_all_blocks()

    differ = sorted(k for k in digests["off"]
                    if digests["off"][k] != digests["on"].get(k))
    print(f"    scalars off {scalars_by['off']} vs on {scalars_by['on']}")
    if differ:
        print(f"    {len(differ)} of {len(digests['off'])} carriers DIFFER "
              f"between overlap modes:")
        for name in differ[:24]:
            print(f"      {name}")
    else:
        print(f"    BIT-EXACT over all {len(digests['off'])} carriers -- "
              f"the chain changes nothing but the clock")
    print("RESULT " + json.dumps({
        "steps": args.steps, "tile": args.tile, "halo": halo,
        "tiles": len(specs), "carriers": len(digests["off"]),
        "digest_off": whole["off"], "digest_on": whole["on"],
        "differ": differ, "bitexact": not differ}))
    if differ:
        raise SystemExit(1)


# --------------------------------------------------------------------------
# forecast
# --------------------------------------------------------------------------

def stage_forecast(args) -> None:
    import cupy as cp

    from tilestream import driver, realcase

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _banner(f"FORECAST {args.config} streamed, tile {args.tile}")

    prep, exp, peak = _prepare(args)
    cfg = prep.cfg
    history = None
    if args.wrfout:
        from tilestream import realcase_wrfout
        history = realcase_wrfout.History(
            prep, exp, args.wrfout, slab_rows=args.refl_slab,
            title=(args.title
                   or f"WOOF-ArWen streamed out-of-core {cfg.nx}x{cfg.ny} "
                      f"dx={cfg.dx/1000:.1f} km -- {Path(args.config).stem}"))
    case = realcase.build_stores(
        prep, exp, on_state=(None if history is None else history.capture),
        settle=args.settle)
    if history is not None:
        history.bind(case)
    realcase.release_prepared(prep)
    prep = None
    print(f"    domain {cfg.nx}x{cfg.ny}x{cfg.nz}, dx {cfg.dx/1000:.0f} km "
          f"({cfg.nx*cfg.dx/1000:.0f} x {cfg.ny*cfg.dy/1000:.0f} km), "
          f"dt {cfg.dt:.0f} s")
    print(f"    lat {case.lat.min():.2f}..{case.lat.max():.2f}  "
          f"lon {case.lon.min():.2f}..{case.lon.max():.2f}  terrain "
          f"{case.terrain.min():.0f}..{case.terrain.max():.0f} m")
    print(f"    init {case.start_time}  forcing every "
          f"{(case.forcing_times[1]-case.forcing_times[0]).total_seconds()/3600:.0f} h"
          f" through {case.forcing_times[-1]}")

    specs, halo = realcase.plan(cfg, args.tile, args.tile)
    red = sum(s.cnx * s.cny for s in specs) / (cfg.nx * cfg.ny)
    print(f"    {len(specs)} tiles of {specs[0].cny}x{specs[0].cnx}, "
          f"halo {halo}, redundancy {red:.3f}x")
    sys.stdout.flush()

    binder = realcase.tile_boundary_binder(case.boundaries, specs, cfg,
                                           verbose=print)
    kwargs = _run_kwargs(cfg, exp, case.geo_store, binder.per_tile[0])
    kwargs["scalars"] = case.scalars
    kwargs["tile_hook"] = binder
    # AND THE REFLECTIVITY SLOT (audit R-052).  ``build_stores`` put
    # ``scratch/refl_10cm`` in the store, so every sweep has to WRITE it:
    # these keywords prime it on each buffer, name it in the inventory the
    # transport gathers and scatters by, ask every tile step for it, and
    # clear each tile's one-frame stash.  Without them the slot is scattered
    # unwritten over the domain field and the composite this lane's whole
    # product is would be zeros for the schemes that compute their own dBZ.
    kwargs = realcase.reflectivity_run_kwargs(kwargs, case)

    steps_per_dump = max(1, int(round(args.every * 60.0 / cfg.dt)))
    ndumps = max(1, int(round(args.hours * 3600.0 / (args.every * 60.0))))

    wrfout_every = max(1, int(round(args.wrfout_every / args.every))) \
        if args.wrfout else 0
    ndumped = [0]

    def dump(tag: str) -> None:
        t = time.perf_counter()
        snap = realcase.surface_snapshot(
            case, elapsed_s=case.scalars["elapsed_seconds"],
            slab_rows=args.refl_slab)
        np.savez_compressed(out_dir / f"real_{cfg.nx}_{tag}.npz", **snap)
        wrote = ""
        if history is not None and ndumped[0] % wrfout_every == 0:
            tw = time.perf_counter()
            path = history.write(case.scalars["elapsed_seconds"])
            wrote = (f" wrfout {path.name} "
                     f"{path.stat().st_size / 2**30:.2f} GiB "
                     f"[{time.perf_counter() - tw:.0f}s]")
        ndumped[0] += 1
        refl = snap["REFL_COMPOSITE"]
        print(f"    [{tag}] valid {snap['valid_time']}  "
              f"t+{snap['elapsed_s']/3600:.2f} h  "
              f"refl max {np.nanmax(refl):.1f} dBZ, frac>35 "
              f"{float((refl > 35).mean()):.5f}  "
              f"w {snap['WMIN'].min():+.1f}..{snap['WMAX'].max():+.1f} m/s  "
              f"UH max {np.nanmax(snap['UH25']):.0f}  "
              f"rain {float(snap['RAINNC'].max()):.1f} mm "
              f"[{time.perf_counter()-t:.1f}s]{wrote}")
        sys.stdout.flush()

    dump("f000")
    wall0 = time.perf_counter()
    peak_vram = 0
    prev_calls: dict = {k: int(v) for k, v
                        in kwargs["scalars"].get("call_counts", {}).items()}
    for i in range(ndumps):
        t = time.perf_counter()
        report: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            driver.run_tiled(case.store, cfg, args.tile, args.tile,
                             halo=halo, nsteps=steps_per_dump,
                             nbuffers=args.nbuffers, periodic=False,
                             write_mode="ring", report=report, **kwargs)
        cp.cuda.runtime.deviceSynchronize()
        # The sweep carried refl_10cm_due on every step, so the store's
        # REFL_10CM is now the field this window's last microphysics call
        # wrote, and the dump below may read it instead of recomputing one
        # scheme's formulation from another scheme's species.
        case.reflectivity_swept()
        peak_vram = max(peak_vram,
                        int(cp.get_default_memory_pool().total_bytes()))
        dt_wall = time.perf_counter() - t
        moved = report.get("gathered_bytes", 0) + report.get(
            "scattered_bytes", 0)
        # WHICH SCHEMES ACTUALLY FIRED in this window.  A cadenced scheme
        # (radiation at radt, cumulus at cudt) can fire zero times inside a
        # short window, and a timing quoted from such a window is not a
        # full-physics timing.  The counts are the driver's own, carried
        # across the sweep by physics_inventory.carrier_scalars.
        calls = kwargs["scalars"].get("call_counts", {})
        fired = {k: int(v) - int(prev_calls.get(k, 0))
                 for k, v in calls.items() if int(v) != int(prev_calls.get(k, 0))}
        prev_calls.update({k: int(v) for k, v in calls.items()})
        print(f"    swept {steps_per_dump} steps in {dt_wall:.1f}s "
              f"({dt_wall/steps_per_dump*1e3:.0f} ms/step)  moved "
              f"{moved/1e9:.1f} GB @ {moved/dt_wall/1e9:.1f} GB/s  "
              f"VRAM peak {peak_vram/2**30:.2f} GiB  fired {fired}")
        sys.stdout.flush()
        dump(f"f{int(round((i + 1) * args.every)):04d}")
    wall = time.perf_counter() - wall0
    print(f"    forecast complete: {args.hours:.1f} h of weather in "
          f"{wall/60:.1f} min of wall clock "
          f"({wall/60/args.hours:.1f} min per forecast hour)")
    print("RESULT " + json.dumps({
        "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz, "dx": cfg.dx, "dt": cfg.dt,
        "tile": args.tile, "halo": halo, "tiles": len(specs),
        "hours": args.hours, "wall_seconds": wall,
        "prepare_peak_gib": peak / 2 ** 30,
        "forecast_peak_vram_gib": peak_vram / 2 ** 30,
        "store_gib": case.store_bytes / 2 ** 30,
        "geography_gib": case.geo_bytes / 2 ** 30,
        "boundary_host_gib": binder.nbytes / 2 ** 30}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage",
                    choices=["gate", "ceiling", "forecast", "twin", "pair"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--nbuffers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--halo", type=int, default=None,
                    help="override realcase.halo_for; the gate's negative "
                         "control is the bare dependency radius, which drops "
                         "the boundary quarantine")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--every", type=float, default=30.0,
                    help="minutes between product dumps")
    ap.add_argument("--refl-slab", type=int, default=64)
    ap.add_argument("--no-settle", dest="settle", action="store_false",
                    help="skip build_stores' one monolithic settling step. "
                         "REQUIRED past the resident ceiling: that step is a "
                         "full-domain dycore.step and is exactly what does "
                         "not fit. run_tiled's carrier-inventory match is "
                         "the guard that keeps this accurate")
    ap.set_defaults(settle=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--wrfout", default=None,
                    help="also write wrfout NetCDF history here, which is "
                         "what 'woof render --engine rust' reads")
    ap.add_argument("--wrfout-every", type=float, default=30.0,
                    help="minutes between wrfout frames; rounded to a "
                         "multiple of --every, which is the only cadence a "
                         "frame may be taken on (sweep boundaries)")
    ap.add_argument("--overlap", choices=("on", "off", "unchained"),
                    default="on",
                    help="TiledRun's overlap mode for the gate's and the "
                         "twin's streamed arm.  The digest must hold under "
                         "BOTH 'on' and 'off' -- that pair is the overlap "
                         "build's correctness receipt on real data")
    ap.add_argument("--title", default=None,
                    help="wrfout TITLE for forecast/twin frames; the "
                         "default names the grid and the config stem "
                         "rather than any particular case")
    ap.add_argument("--prepare",
                    choices=["slabbed", "lowwater", "inorder"],
                    default="slabbed")
    ap.add_argument("--prepare-rows", type=int, default=64,
                    help="rows per ingest slab; sets the ingest's peak")
    args = ap.parse_args()
    {"gate": stage_gate, "ceiling": stage_ceiling,
     "forecast": stage_forecast, "twin": stage_twin,
     "pair": stage_pair}[args.stage](args)


if __name__ == "__main__":
    main()
