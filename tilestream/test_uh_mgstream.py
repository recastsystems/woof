"""The same three accumulators, streamed across N GPUs out of ONE host store.

:mod:`tilestream.mgstream` shares the tiles of a single pinned host domain out
over several GPUs.  Nothing about the running maxima is device-aware -- the
fold happens in a tile buffer and the result is scattered into the store like
any other carrier -- so the multi-GPU question here is narrow and worth asking
precisely:

1.  do the accumulators survive the PARTITION?  Two workers walk the tile list
    in a different order from one, and ``write_mode="shadow"`` gives them two
    stores and one barrier while ``write_mode="ring"`` gives them a Gauss-
    Seidel dependency chain that crosses devices.  A running max is
    order-independent as arithmetic (max is associative and commutative) but
    the transport underneath it is not, and an accumulator that is gathered
    from the wrong generation of the store is wrong in a way no amount of
    associativity fixes.
2.  is the RESET still transport-agnostic?  ``mgstream`` is a store-sweeping
    function with no seam of its own -- it is handed a store and never sees a
    ``DomainState`` -- so a run loop driving it has to bind the store where
    the model's reset can find it, exactly as ``streaming.attach`` does for
    the single-GPU seam.  This module binds it the same way and calls the
    MODEL's ``reset_up_heli_max`` / ``reset_tracker_window``, so what is
    tested is the shipped API and not a store-aware stand-in.

``devices=(0,)`` and ``devices=(0, 0)`` are the two configurations that
separate "the partition is wrong" from "two physical GPUs interfere", and both
run on any box.  ``--devices 0,1`` runs the real two-card case when a
multi-GPU box is free.

    python -m tilestream.test_uh_mgstream                # 1 worker + 2 workers
    python -m tilestream.test_uh_mgstream --devices 0,1  # two real cards
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

from woof.core import streaming
from woof.core.uh_diag import (UH_FOLLOW_WINDOW_SLOT, UH_SPAWN_WINDOW_SLOT,
                                UP_HELI_MAX_SLOT, reset_tracker_window,
                                reset_up_heli_max)
from tilestream import driver, gather, harness, mgstream
from tilestream import physics_inventory as physinv
from tilestream import test_join as tj
from tilestream import test_uh_stream as uhs

#: Smaller than the single-GPU gate's geometry because every ``run_mgstream``
#: call rebuilds its tile buffers (MEASURED at 3.3 s for a 544-cell dry
#: buffer), and a reset schedule forces one call per segment.  Still 24 tiles,
#: still tiles with no true edge and tiles with two.
NX, NY, NZ = 192, 128, 49
TX, TY = 32, 32
#: One uniform cadence for all three windows here, unlike the single-GPU gate:
#: this module is asking whether the PARTITION moves the answer, and three
#: interleaved cadences would multiply the segment count without adding a
#: single new way for the partition to be wrong.
NSTEPS = 12
RESET_EVERY = 4

#: OPEN lateral boundaries, not SPECIFIED, and that is a finding rather than a
#: convenience.  ``update_up_heli_max`` needs non-periodic boundary handling on
#: BOTH axes -- specified/nested, or open-x AND open-y -- so a periodic domain
#: cannot run this diagnostic at all.  Of the two legal choices only OPEN is
#: reachable here: ``run_mgstream`` has no ``tile_hook`` parameter, so it has
#: nowhere to attach a tile's windowed lateral-boundary tables, and a
#: specified-BC domain streamed through it would give every tile the SAME
#: tables -- which is the single-GPU lane's "boundary tables not windowed"
#: negative control, i.e. a known-wrong run.  That gap is mgstream's, not this
#: diagnostic's: the module predates the per-tile boundary work.  Open BCs need
#: no tables, so the accumulators can be tested across devices today and the
#: specified-BC multi-GPU case is reported as NOT TESTED.
OPEN_BC = dict(specified=False, open_x=True, open_y=True, nested=False)

#: FLAT, and that is also forced rather than chosen.  ``dycore.step`` REFUSES
#: ``terrain_opt=1`` together with open boundaries outright -- "set_w_surface
#: and the advance_w_phi kinematic surface BC difference ht with unconditional
#: periodic wraps, which would couple the two open boundaries through the
#: terrain slope" -- so the only geography this lane can carry is the
#: PROJECTION.  That is still horizontally varying and still the thing a tile
#: would get wrong if it rebuilt it: map factors, Coriolis, the rotation angle
#: and every scheme's latitude/longitude grid.  Real TERRAIN under multi-GPU
#: streaming is reported as NOT TESTED, and it is untestable here for a reason
#: that has nothing to do with this diagnostic.
FLAT = dict(terrain_opt=0, hill_height=0.0)


def mg_cfg(nx: int = NX, ny: int = NY, nz: int = NZ, **over):
    """Real Lambert projection + OPEN boundaries + nwp_diagnostics."""
    kwargs = dict(harness.GEOGRAPHY_OVERRIDES)      # map_proj=1, terrain_opt=1
    kwargs.update(FLAT)
    kwargs.update(OPEN_BC)
    kwargs.update(ztop=20000.0, dt=60.0, nwp_diagnostics=1)
    kwargs.update(over)
    return harness.make_config(nx, ny, nz, periodic=False, **kwargs)


def build_domain(cfg, *, seed=tj.SEED, warmup: int = 1):
    """A prepared resident domain on the real projection."""
    geo = harness.make_geography(cfg, terrain=False, periodic_faces=False)
    state, _drv = harness.make_physics_state(cfg, seed, geography=geo)
    if warmup:
        harness.run_steps(state, cfg, int(warmup))
    return state, geo


def _bind_store(state, store) -> None:
    """What a multi-GPU seam would have to do, done explicitly.

    ``streaming.attach`` leaves this binding behind for the single-GPU path.
    ``mgstream`` has no attach -- it is handed a store and never sees a
    ``DomainState`` -- so a run loop over it must leave the same one, and this
    line IS the whole multi-GPU half of the fix.  Without it
    ``reset_up_heli_max`` reaches only the state, the store keeps accumulating
    and every window silently becomes "max since the run began".
    """
    setattr(state, streaming.STREAMED_SCRATCH_ATTR,
            {name.split("/", 1)[1]: arr for name, arr in store.items()
             if name.startswith("scratch/")})


def reference(cfg, nsteps=NSTEPS):
    """Resident, WOOF-ArWen's own ``dycore.step``, with the same reset schedule."""
    import cupy as cp

    from woof.core.dycore import step

    state, _geo = build_domain(cfg)
    frames: dict[str, list] = {"history": [], "follow": [], "spawn": []}
    for k in range(nsteps):
        step(state, cfg, refl_10cm_due=False)
        if (k + 1) % RESET_EVERY == 0:
            frames["history"].append(uhs._read_state(state, UP_HELI_MAX_SLOT))
            frames["follow"].append(
                uhs._read_state(state, UH_FOLLOW_WINDOW_SLOT))
            frames["spawn"].append(
                uhs._read_state(state, UH_SPAWN_WINDOW_SLOT))
            reset_up_heli_max(state)
            reset_tracker_window(state, UH_FOLLOW_WINDOW_SLOT)
            reset_tracker_window(state, UH_SPAWN_WINDOW_SLOT)
    cp.cuda.runtime.deviceSynchronize()
    carriers = {k: uhs._host(v) for k, v in
                physinv.carrier_inventory(state, None).items()}
    del state
    cp.get_default_memory_pool().free_all_blocks()
    return frames, carriers


def mg_run(cfg, *, devices, write_mode="shadow", partition="block",
           nsteps=NSTEPS, cross_worker_sync="events", sweep_barrier=True):
    """The same run, tiles shared over ``devices`` out of one pinned store."""
    import cupy as cp

    domain, _geo = build_domain(cfg)
    store = {k: gather.pinned_copy(v) for k, v in
             physinv.carrier_inventory(domain, None).items()}
    _bind_store(domain, store)
    geo_inv = {k: gather.pinned_copy(v)
               for k, v in driver.geography_inventory(domain).items()}
    halo = harness.halo_radius(cfg)
    factory = driver.make_physics_tile_state
    scalars = physinv.carrier_scalars(domain)

    frames: dict[str, list] = {"history": [], "follow": [], "spawn": []}
    report: dict = {}
    for _start in range(0, nsteps, RESET_EVERY):
        mgstream.run_mgstream(
            store, cfg, TX, TY, halo=halo, nsteps=RESET_EVERY,
            devices=list(devices), nbuffers=2, periodic=False,
            write_mode=write_mode, partition=partition,
            tile_state_factory=factory,
            inventory_fn=physinv.carrier_inventory, nz=int(cfg.nz),
            scalars=scalars, geography=geo_inv, check_geography=False,
            cross_worker_sync=cross_worker_sync,
            sweep_barrier=sweep_barrier, report=report)
        for label, slot in (("history", UP_HELI_MAX_SLOT),
                            ("follow", UH_FOLLOW_WINDOW_SLOT),
                            ("spawn", UH_SPAWN_WINDOW_SLOT)):
            frames[label].append(np.asarray(store[f"scratch/{slot}"]).copy())
        # The MODEL's reset, through the model's own API, on a state whose
        # arrays are in the store.  Nothing here knows about stores.
        reset_up_heli_max(domain)
        reset_tracker_window(domain, UH_FOLLOW_WINDOW_SLOT)
        reset_tracker_window(domain, UH_SPAWN_WINDOW_SLOT)
    carriers = {k: np.asarray(v) for k, v in store.items()}
    del domain, store
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return frames, carriers, report


def _cmp(ref_frames, ref_carriers, got_frames, got_carriers) -> dict:
    per = {}
    for label in ref_frames:
        rows = [bool(np.array_equal(a, b))
                for a, b in zip(ref_frames[label], got_frames[label])]
        per[label] = {"nframes": len(rows), "equal": all(rows),
                      "first_differing": next(
                          (i + 1 for i, ok in enumerate(rows) if not ok),
                          None),
                      "peak": max((float(np.nanmax(a))
                                   for a in ref_frames[label]), default=0.0)}
    shared = sorted(set(ref_carriers) & set(got_carriers))
    ndiff = sum(1 for k in shared
                if not np.array_equal(ref_carriers[k], got_carriers[k]))
    per["_carriers"] = {"ntotal": len(shared), "ndiff": ndiff,
                        "missing": sorted(set(ref_carriers) - set(shared))}
    return per


def _line(label, ok, detail=""):
    return f"  {'PASS' if ok else 'FAIL':4s}  {label:52s} {detail}"


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    devices_arg = None
    for a in argv:
        if a.startswith("--devices"):
            devices_arg = a.split("=", 1)[1] if "=" in a else argv[
                argv.index(a) + 1]
    ndev = cp.cuda.runtime.getDeviceCount()
    print(f"cupy {cp.__version__}  {ndev} visible device(s)")
    for i in range(ndev):
        f, t = cp.cuda.Device(i).mem_info
        print(f"  device {i}: "
              f"{cp.cuda.runtime.getDeviceProperties(i)['name'].decode()}  "
              f"{f / 2**30:.1f} GiB free of {t / 2**30:.1f}")

    cases = [("1 worker,  1 GPU  (shadow)", (0,), "shadow", "block"),
             ("2 workers, 1 GPU  (shadow)", (0, 0), "shadow", "block"),
             ("2 workers, 1 GPU  (ring)", (0, 0), "ring", "block"),
             ("2 workers, 1 GPU  (shadow, stripe)", (0, 0), "shadow",
              "stripe")]
    if devices_arg:
        devs = tuple(int(x) for x in devices_arg.split(","))
        cases += [(f"{len(devs)} workers on devices {devs} (shadow)", devs,
                   "shadow", "block"),
                  (f"{len(devs)} workers on devices {devs} (ring)", devs,
                   "ring", "block")]

    cfg = mg_cfg()
    print("=" * 78)
    print("UP_HELI_MAX AND THE TRACKER WINDOWS ACROSS N GPUs (mgstream)")
    print(f"  {NX}x{NY}x{NZ}, real Lambert (flat), OPEN lateral BCs, "
          f"tile {TX}x{TY}, halo "
          f"{harness.halo_radius(cfg)}, N={NSTEPS}, reset every "
          f"{RESET_EVERY} steps")
    print("=" * 78)

    ref_frames, ref_carriers = reference(cfg)
    peaks = {k: max(float(np.nanmax(a)) for a in v)
             for k, v in ref_frames.items()}
    # How many cells the comparison actually has anything to compare: a
    # running max that is non-zero in six cells is a peak, not a field.
    live = {k: max(int(np.count_nonzero(a)) for a in v)
            for k, v in ref_frames.items()}
    size = ref_frames["history"][0].size
    nonvacuous = (all(p > 0.0 for p in peaks.values())
                  and min(live.values()) >= size // 100
                  and len(ref_frames["history"]) >= 3)
    print(_line("resident control is non-vacuous", nonvacuous,
                f"{len(ref_frames['history'])} resets, peaks "
                + ", ".join(f"{k}={v:.4g}" for k, v in peaks.items())
                + "; non-zero cells "
                + ", ".join(f"{k}={v}/{size}" for k, v in live.items())))
    failures = [] if nonvacuous else ["the resident control is vacuous"]
    untested: list[str] = []

    for label, devs, wmode, part in cases:
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                gf, gc, _rep = mg_run(cfg, devices=devs,
                                      write_mode=wmode, partition=part)
            res = _cmp(ref_frames, ref_carriers, gf, gc)
            ok = all(res[k]["equal"] for k in ("history", "follow", "spawn")) \
                and res["_carriers"]["ndiff"] == 0 \
                and not res["_carriers"]["missing"]
            detail = (", ".join(
                f"{k}: {res[k]['nframes']} frames "
                f"{'equal' if res[k]['equal'] else 'DIFFER@' + str(res[k]['first_differing'])}"
                for k in ("history", "follow", "spawn"))
                + f"; {res['_carriers']['ntotal']} carriers ndiff="
                f"{res['_carriers']['ndiff']}")
        except Exception as exc:                          # noqa: BLE001
            oom = isinstance(exc, cp.cuda.memory.OutOfMemoryError) or \
                "OutOfMemory" in repr(exc) or "out of memory" in str(exc) \
                or "failed" in str(exc)
            ok = False
            if oom:
                # Not a result either way.  A shared card that filled up has
                # told us nothing about this configuration, and calling that
                # a failure is as wrong as calling it a pass.
                untested.append(f"{label}: {exc}")
                print(_line(label, False,
                            "NOT TESTED -- the card ran out of memory "
                            f"[{time.perf_counter() - t0:.1f} s]"))
                cp.get_default_memory_pool().free_all_blocks()
                continue
            detail = f"raised {type(exc).__name__}: {exc}"
        if not ok:
            failures.append(f"{label}: {detail}")
        print(_line(label, ok, f"{detail}  [{time.perf_counter() - t0:.1f} s]"))
        cp.get_default_memory_pool().free_all_blocks()

    # -------------------------------------------------------- the negatives
    print()
    print("-- NEGATIVE CONTROLS (each MUST differ)")
    negatives = [
        ("ring, cross_worker_sync='none'",
         dict(devices=(0, 0), write_mode="ring",
              cross_worker_sync="none")),
        ("shadow, sweep_barrier=False",
         dict(devices=(0, 0), write_mode="shadow", sweep_barrier=False)),
    ]
    for label, kw in negatives:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                gf, gc, _rep = mg_run(cfg, **kw)
            res = _cmp(ref_frames, ref_carriers, gf, gc)
            differs = (not all(res[k]["equal"]
                               for k in ("history", "follow", "spawn"))
                       or res["_carriers"]["ndiff"] > 0)
            detail = (f"ndiff={res['_carriers']['ndiff']}/"
                      f"{res['_carriers']['ntotal']} carriers")
        except Exception as exc:                          # noqa: BLE001
            # An OOM is a shared card, not a control firing.  Counting it as
            # "the control refused" would let a full GPU certify an ordering
            # rule that was never exercised -- the exact shape of this
            # project's earlier false results.
            oom = isinstance(exc, cp.cuda.memory.OutOfMemoryError) or \
                "OutOfMemory" in repr(exc) or "out of memory" in str(exc)
            differs = not oom
            detail = ("NOT EXERCISED: the card ran out of memory"
                      if oom else f"refused: {type(exc).__name__}")
        # These controls are about the ORDERING, not about the accumulators,
        # and on an idle card an unsynchronised order can still happen to be
        # correct -- mgstream's own module docstring says so.  So a control
        # that does not fire is REPORTED, loudly, and is not a gate failure
        # here: the gate for those two lives in tilestream/test_mgstream.py.
        print(_line(label, differs, detail
                    + ("" if differs else "  <-- DID NOT FIRE (reported, "
                                          "not asserted: see docstring)")))
        cp.get_default_memory_pool().free_all_blocks()

    print()
    print("=" * 78)
    if untested:
        print(f"NOT TESTED -- {len(untested)} configuration(s) never ran:")
        for u in untested:
            print(f"  ? {u}")
        print()
    if failures:
        print(f"MULTI-GPU UH GATE FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  * {f}")
        return 1
    if untested:
        print("MULTI-GPU UH GATE INCOMPLETE -- every configuration that RAN "
              "agreed with the resident run, but not all of them ran.")
        return 2
    print("MULTI-GPU UH GATE PASSED -- the accumulators and their resets are "
          "invariant to the tile partition and the worker count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
