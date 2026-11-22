"""A FORECAST-LENGTH streamed run: thousands of steps, and what breaks over them.

Every bit-exactness result in this project was taken over 8 to 48 steps.  That
is enough to prove the ARITHMETIC -- the gather rectangles, the halo, the
clock, the per-tile boundary windowing -- and it proves nothing whatsoever
about a run that lasts four hours.  An 18-hour forecast at dt = 15 s is 4320
steps; at 672^2 x 49 streamed that is 4320 sweeps, 4320 x 4 tile gathers,
4320 x 4 scatters and 58 TB across PCIe.  The failure modes that live at that
length are invisible at 48 steps and all of them are silent until they are
fatal:

*A pinned leak.*  Page-locked pages cannot be swapped, so a store that grows
1 MiB per sweep does not get slower, it kills the box.  ``/proc/meminfo`` on
these rented machines reports the HOST's memory rather than the container's,
and page-locked pages are ordinary anonymous RSS as far as
``/proc/self/status`` is concerned (``VmLck`` counts ``mlock``, which
``cudaHostAlloc`` does not use).  So the only accurate instrument is the
process's own ledger, which is why :func:`tilestream.hoststore.pinned_ledger`
exists and why this module samples it every step rather than at the end: a
leak has to be visible as a TREND, because the alternative -- inferring one
from a final allocation failure -- cannot distinguish a leak from a store
that was always too big.

*Device fragmentation.*  ``PhysicsDriver`` REPLACES whole tendency bundles
when the owning scheme runs, so a run at forecast cadence allocates and frees
device memory in a pattern that a 48-step gate never reaches: radiation fires
every 48 steps and cumulus every 20, and each firing churns the pool.  The
question is not whether VRAM grows -- CuPy's pool grows on purpose and
plateaus -- but whether it plateaus.

*Clock drift.*  ``elapsed_seconds`` is a float accumulated by ``+= dt`` inside
``dycore.step``, once per CALL, and it is the argument to every cadence test.
Over 4320 steps at dt = 15 s it reaches 64800.0; the check here is EXACT
equality against ``nsteps * dt``, not a tolerance, because dt = 15 is
representable and any drift at all would mean the clock is being reconstructed
rather than carried.

*Step-time decay.*  Answered by comparing deciles, not endpoints.

THE INSTRUMENT THAT NEVER FIRED IS NOT AN INSTRUMENT
----------------------------------------------------
``--leak-mib`` retains that many MiB of pinned memory every ``--leak-every``
steps.  It exists so the leak detector can be shown to SEE a leak: with it on
the reported slope must match the injected rate, and with it off the slope
must be zero.  A memory-flatness verdict from a detector that has never
reported anything else is not evidence.

WHY THE TWO LEGS RUN IN ONE PROCESS
-----------------------------------
``--mode pair`` integrates the same forecast twice from the same t0 snapshot:
once resident through ``woof.core.dycore.step`` and once streamed through
:class:`woof.core.streaming.StreamedDomain`, and compares carrier digests at
every checkpoint rather than only at the end.  Comparing only the end says
"they agree" or "they do not"; comparing the series says WHICH step they
stopped agreeing at, which is the difference between a result and a bug
report.  Both legs are driven by the callable
:func:`woof.core.streaming.make_stepper` returned -- ``dycore.step`` itself
for the resident leg, by the OFF contract -- so the loop below is the same
loop twice and the only variable is that object.

ONE SWEEP PER CALL, WHICH IS ALSO WHY THE CLOCK CAN BE READ AT ALL
------------------------------------------------------------------
``driver.run_tiled(store, cfg, ..., nsteps=N)`` writes the domain clock back
into the caller's ``scalars`` dict only after ALL N steps (driver.py, end of
``_sweep``), so a progress callback inside such a call reads a frozen clock --
measured: a 240-step call reported ``model 0.2 min`` at every one of its
240 steps while the run really did advance to 60.2 minutes.  Nothing was
wrong with the model; the caller had asked for one sweep of 240 steps rather
than 240 sweeps of one.  This module drives ``stepper(state, cfg)`` once per
model step, which is what WOOF-ArWen's own run loops do, and the clock is live.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings

import numpy as np


SEED = 20_260_731

#: full + MYNN surface layer/PBL + Noah-MP land surface -- the 229-carrier
#: rung, the same selector set ``hourrun``'s A/B hour used so the two results
#: are comparable.
_MOIST = dict(moist=True, mp_physics=10, ztop=20000.0)
_FULL = dict(_MOIST, km_opt=4, sf_sfclay_physics=91, bl_pbl_physics=1,
             bldt=0.0, sf_surface_physics=2, ra_sw_physics=4,
             ra_lw_physics=4, radt_minutes=12.0, cu_physics=1,
             cudt_minutes=5.0)
RUNG = dict(_FULL, sf_sfclay_physics=5, bl_pbl_physics=5,
            sf_surface_physics=4)
#: real Lambert conformal projection + real terrain at dx = 3 km.
GEO = dict(map_proj=1, terrain_opt=1, dx=3000.0, dy=3000.0)


def build_cfg(n, nz, dt):
    from tilestream import harness

    return harness.make_config(n, n, nz, dt=float(dt), **RUNG, **GEO)


# --------------------------------------------------------------------------
# instrumentation
# --------------------------------------------------------------------------

def _proc_gib(field: str) -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(field):
                    return float(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("nan")


def _cgroup_gib() -> float:
    """Memory charged to this process's own cgroup, in GiB; NaN when unread.

    ``/proc/meminfo`` reports the HOST's on these boxes.  The usage is the
    deepest level of the planner's walk that publishes one
    (``tilestream.autoplan._cgroup_memory_walk``): the process's own cgroup,
    or its nearest ancestor that reports usage.

    THE BREAKAGE: this read the mount root's ``memory.current``, which a
    cgroup v2 root does not have, so inside a systemd scope (or on any bare
    host) the trace's cgroup column was NaN from the first step.
    """
    from tilestream import autoplan

    for level in autoplan._cgroup_memory_walk():
        if level.usage is not None:
            return level.usage / 2 ** 30
    return float("nan")


def sample() -> dict:
    """One row of the memory trace.  Cheap enough to take every step.

    MEASURED at 0.14 ms, against a 660 ms step: sampling every step rather
    than every twentieth costs 0.02% and is what makes a slope fit over 4320
    points possible instead of 216.  Reading the process's own cgroup
    (:func:`_cgroup_gib`) added 0.11 ms, measured in a systemd scope six
    levels deep, so a row is about 0.25 ms, still under 0.04% of the step.
    """
    import cupy as cp
    from tilestream import hoststore

    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    led = hoststore.pinned_ledger()
    return dict(
        vram_used=float(total - free), vram_total=float(total),
        pool_used=float(pool.used_bytes()), pool_total=float(pool.total_bytes()),
        pinned=float(led["total_bytes"]),
        pinned_blocks=float(led["block_bytes"]),
        pinned_pool=float(led["pool_total_bytes"]),
        rss=_proc_gib("VmRSS:") * 2 ** 30, cgroup=_cgroup_gib() * 2 ** 30)


def slope_per_1k(y, x=None) -> float:
    """Least-squares slope of ``y`` against step index, per 1000 steps.

    Deliberately a plain fit rather than last-minus-first: CuPy's device pool
    grows in jumps as new block sizes are first requested, so an endpoint
    difference reports a step change as a trend and a trend as noise.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size < 8:
        return float("nan")
    x = np.arange(y.size, dtype=np.float64) if x is None \
        else np.asarray(x, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom <= 0.0:
        return float("nan")
    return float(((x - xm) * (y - ym)).sum() / denom * 1000.0)


def deciles(values) -> list[float]:
    """Median of each tenth of the run -- the step-time drift instrument.

    Endpoints cannot answer "is step 4000 as fast as step 100": the first
    step carries pipeline priming and the last carries the closing frame
    write, and a single sample of each is noise.  Ten medians show the shape.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size < 10:
        return [float(np.median(v))] if v.size else []
    return [float(np.median(chunk)) for chunk in np.array_split(v, 10)]


def digest_of(inv) -> tuple[str, dict]:
    from tilestream import physics_inventory as physinv

    per = physinv.field_digests(inv)
    acc = hashlib.sha256()
    for key in sorted(per):
        acc.update(key.encode())
        acc.update(per[key].encode())
    return acc.hexdigest()[:32], per


def nonfinite_count(inv) -> int:
    import cupy as cp

    bad = 0
    for arr in inv.values():
        host = cp.asnumpy(arr) if isinstance(arr, cp.ndarray) \
            else np.asarray(arr)
        if host.dtype.kind == "f":
            bad += int(np.count_nonzero(~np.isfinite(host)))
    return bad


class Leak:
    """A deliberate pinned leak, so the leak detector can be shown to fire.

    Holds every block it allocates.  ``mib=0`` is the ordinary run and
    allocates nothing at all -- the object still exists so that the control
    and the real run take the identical code path.
    """

    def __init__(self, mib: float = 0.0, every: int = 0):
        self.mib = float(mib)
        self.every = int(every)
        self._held: list = []

    @property
    def per_1k_bytes(self) -> float:
        """What the detector must report if it works, in bytes per 1000 steps."""
        if self.mib <= 0 or self.every <= 0:
            return 0.0
        return self.mib * 2 ** 20 * 1000.0 / self.every

    def maybe(self, step: int) -> None:
        if self.mib <= 0 or self.every <= 0 or step % self.every:
            return
        from tilestream import hoststore

        self._held.append(hoststore.alloc_pinned_array(
            (int(self.mib * 2 ** 20),), np.uint8))

    def release(self) -> None:
        self._held.clear()


# --------------------------------------------------------------------------
# the domain, and the streamed attachment
# --------------------------------------------------------------------------

def build_domain(cfg, seed=SEED):
    """A prepared resident domain on the real projection and real terrain."""
    from tilestream import harness

    geo = harness.make_geography(cfg)
    state, _drv = harness.make_physics_state(cfg, seed, geography=geo)
    harness.run_steps(state, cfg, 1)       # itimestep 1: the lazy carriers
    return state, geo


def snapshot_t0(state):
    """Pinned host copies of every carrier, so both legs start identical."""
    from tilestream import gather
    from tilestream import physics_inventory as physinv

    return {name: gather.pinned_copy(arr)
            for name, arr in physinv.carrier_inventory(state).items()}


def restore_t0(state, start, scalars0):
    """Put the snapshot back on the device, in place -- ARRAYS AND SCALARS.

    Restoring only the arrays is the obvious version and it is wrong.  After
    the resident leg has run N steps ``state.elapsed_seconds`` is ``N*dt`` and
    ``PhysicsDriver.call_counts`` has counted every firing; a streamed leg
    started on those would evaluate ``itimestep`` N steps ahead of where its
    fields are and take a different radiation and cumulus cadence from the
    leg it is supposed to be compared against.  The two legs would then
    differ for a reason that has nothing to do with streaming, which is
    exactly the kind of false result this project has produced six times.
    """
    import cupy as cp
    from tilestream import physics_inventory as physinv

    live = physinv.carrier_inventory(state)
    if set(live) != set(start):
        raise RuntimeError(
            f"the t0 snapshot and the live state disagree on carriers: "
            f"{sorted(set(live) ^ set(start))[:8]}")
    for name, arr in live.items():
        if isinstance(arr, cp.ndarray):
            arr.set(np.ascontiguousarray(start[name]))
        else:
            np.copyto(arr, start[name])
    physinv.set_carrier_scalars(state, scalars0)
    cp.cuda.runtime.deviceSynchronize()


def streamed_stepper(state, cfg, tile, nbuffers, geo_store, log):
    """``make_stepper`` for this domain, plus the construction it needs.

    The ``build`` closure is what a ROUTE owns: the seam decides when a
    streamed domain is constructed and what it must satisfy, never how the
    store is filled or where the geography comes from.  This is the harness
    version of the tile-state factory WOOF-ArWen's real-data preparation still has
    to supply.
    """
    from woof.core import streaming
    from tilestream import driver, harness

    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(tile), tile_ny=int(tile),
        nbuffers=int(nbuffers), store="host")
    decision = streaming.decide(cfg, options)
    log(f"  decision: {decision.explain()}")

    def build(st, run_cfg, dec):
        kwargs = driver.geography_run_kwargs(
            run_cfg, None, geography=geo_store,
            geography_fn=harness.neutral_geography)
        return streaming.attach(
            st, run_cfg, dec,
            tile_state_factory=kwargs["tile_state_factory"],
            geography=kwargs["geography"],
            inventory_fn=kwargs["inventory_fn"], nz=kwargs["nz"],
            check_geography=False)

    stepper = streaming.make_stepper(state, cfg, options, decision=decision,
                                     build=build)
    if not streaming.is_streaming(stepper):
        raise RuntimeError("make_stepper declined to stream")
    return stepper


# --------------------------------------------------------------------------
# output at forecast cadence
# --------------------------------------------------------------------------

class DeviceMirror:
    """A read-only mapping that pulls a RESIDENT state's carriers to the host.

    :class:`tilestream.output.StoreFrame` reads a mapping, which on the
    streamed leg is the pinned store itself.  The resident leg has no such
    mapping -- its carriers are on the card -- and the two obvious
    substitutes are both wrong:

    * a host copy of all 229 carriers moves 3.34 GiB across PCIe at 512^2 to
      write a frame that names 15 fields;
    * a dict built from ``plan.names(SOURCE_CARRIER)`` raises ``KeyError:
      'state/thp'`` at the first frame, because ``StoreFrame.fields`` reaches
      PAST the plan for the three carriers its derived rules read directly
      (``state/thp`` for T, ``state/p`` for P, ``state/php`` for PSFC).  That
      is measured, not anticipated: it is what the first smoke run did.

    So this copies on demand, into buffers allocated once, and
    :meth:`refresh` invalidates between frames.  The frame then costs exactly
    the fields it touches, and the two legs are writing the same file through
    the same writer.
    """

    def __init__(self, state):
        from tilestream import physics_inventory as physinv

        self._state = state
        self._live = physinv.carrier_inventory(state)
        self._host: dict[str, np.ndarray] = {}
        self._fresh: set[str] = set()

    def refresh(self) -> None:
        """Next :meth:`__getitem__` of each key re-reads the device."""
        from tilestream import physics_inventory as physinv

        # Re-taken rather than cached: PhysicsDriver REPLACES whole tendency
        # bundles when the owning scheme runs, so a device pointer held
        # across a step can be the previous generation's.
        self._live = physinv.carrier_inventory(self._state)
        self._fresh.clear()

    def __contains__(self, key) -> bool:
        return key in self._live

    def keys(self):
        return self._live.keys()

    def __getitem__(self, key) -> np.ndarray:
        import cupy as cp

        if key in self._fresh:
            return self._host[key]
        src = self._live[key]
        dst = self._host.get(key)
        if dst is None or dst.shape != src.shape or dst.dtype != src.dtype:
            dst = np.empty(src.shape, src.dtype)
            self._host[key] = dst
        np.copyto(dst, cp.asnumpy(src))
        self._fresh.add(key)
        return dst

    @property
    def resident_bytes(self) -> int:
        return sum(int(a.nbytes) for a in self._host.values())


def open_writer(state, cfg, setup, store):
    """The production wrfout writer, fed from wherever the domain lives.

    Same writer, same field order, same file layout on both legs -- which is
    the only way the frame COST is comparable between them.  Returns
    ``(writer, mirror)``; ``mirror`` is ``None`` for the streamed leg.
    """
    from tilestream import output

    plan = output.frame_plan(state)
    mirror = None if store is not None else DeviceMirror(state)
    # require_complete=False, DECLARED rather than inherited.  This is an
    # endurance bench: it measures how long a store-backed writer survives,
    # its inventory is the carrier set, and it has never carried the
    # output-only driver diagnostics.  The product route does
    # (woof.core.streaming.diagnostic_inventory), and StoreFrame refuses a
    # short frame there by default; saying so here keeps this bench's
    # unchanged behaviour a stated choice instead of a silent exemption.
    frame = output.StoreFrame(plan, store if mirror is None else mirror,
                              setup, cfg, require_complete=False)
    return output.StoreHistoryWriter(frame, cfg), mirror


# --------------------------------------------------------------------------
# one leg
# --------------------------------------------------------------------------

def run_leg(name, stepper, state, cfg, store, nsteps, *, log, outdir,
            frame_every, digest_every, leak, setup, restart_at=0):
    """Integrate ``nsteps`` model steps, sampling everything, every step.

    ``store`` is the mapping the digests and the frames are taken from -- the
    pinned store for the streamed leg, ``None`` (meaning the live device
    inventory) for the resident one -- so both legs are measured through the
    same instruments on the same quantities.
    """
    import cupy as cp
    from tilestream import physics_inventory as physinv

    dt = float(cfg.dt)
    streamed = store is not None
    inv_of = (lambda: store) if streamed else \
        (lambda: physinv.carrier_inventory(state))

    writer = mirror = None
    if frame_every:
        writer, mirror = open_writer(state, cfg, setup, store)

    steps_ms: list[float] = []
    trace: list[dict] = []
    digests: list[dict] = []
    frames: list[dict] = []
    restart_note: dict = {}
    # The domain has already taken its warmup step, so the clock starts at dt
    # and not at zero.  Drift is measured against THIS, not against zero: a
    # baseline of zero reports the warmup as 15 s of drift on every line and
    # buries the thing the number exists to detect.
    clock0 = (float(stepper.scalars["elapsed_seconds"])
              if streamed else float(state.elapsed_seconds))

    cp.cuda.runtime.deviceSynchronize()
    t_run = time.perf_counter()
    for k in range(nsteps):
        t = time.perf_counter()
        stepper(state, cfg, refl_10cm_due=False)
        cp.cuda.runtime.deviceSynchronize()
        ms = (time.perf_counter() - t) * 1e3
        steps_ms.append(ms)
        leak.maybe(k + 1)

        s = sample()
        model = (float(stepper.scalars["elapsed_seconds"])
                 if hasattr(stepper, "scalars") and stepper.scalars
                 else float(state.elapsed_seconds))
        s.update(step=k + 1, ms=ms, model_s=model,
                 wall_s=time.perf_counter() - t_run)
        trace.append(s)

        if frame_every and (k + 1) % frame_every == 0:
            tf = time.perf_counter()
            if mirror is not None:
                mirror.refresh()
            hh = int(model // 3600)
            writer.submit(os.path.join(outdir, f"wrfout_{name}_{k + 1:05d}.nc"),
                          f"0001-01-01_{hh:02d}:00:00")
            el = time.perf_counter() - tf
            frames.append(dict(step=k + 1, seconds=el, model_s=model))
            log(f"    [{name}] frame at step {k + 1} ({model / 3600:.2f} fh) "
                f"in {el:.2f} s")

        if digest_every and (k + 1) % digest_every == 0:
            d, _per = digest_of(inv_of())
            bad = nonfinite_count(inv_of())
            digests.append(dict(step=k + 1, digest=d, nonfinite=bad,
                                model_s=model))
            log(f"    [{name}] digest @{k + 1:>5d}  {d}  nonfinite={bad}")
            if bad:
                log(f"    [{name}] *** NON-FINITE: the forecast has gone "
                    f"unstable at step {k + 1}; the run continues so the "
                    f"memory and timing trace stays complete, but nothing "
                    f"after this point is a forecast")

        if restart_at and (k + 1) == restart_at:
            restart_note = restart_round_trip(
                stepper, state, cfg, store, setup, outdir, name, k + 1, log)

        if (k + 1) % 100 == 0 or k == 0:
            drift = model - (clock0 + (k + 1) * dt)
            log(f"  [{name}] {k + 1:>5d}/{nsteps} {ms:8.1f} ms  "
                f"model {model / 3600:6.3f} fh  wall {s['wall_s'] / 60:6.2f} m  "
                f"clock-drift {drift:+.6g} s  "
                f"VRAM {s['vram_used'] / 2 ** 30:5.2f}  "
                f"pinned {s['pinned'] / 2 ** 30:5.2f}  "
                f"cgroup {s['cgroup'] / 2 ** 30:5.1f} GiB")

    wall = time.perf_counter() - t_run
    if writer is not None:
        writer.close()
    scal = (dict(stepper.scalars) if hasattr(stepper, "scalars")
            and stepper.scalars else physinv.carrier_scalars(state))
    final, per = digest_of(inv_of())
    return dict(name=name, nsteps=nsteps, wall_s=wall, steps_ms=steps_ms,
                trace=trace, digests=digests, frames=frames,
                final_digest=final, per_field=per, scalars=scal,
                nonfinite=nonfinite_count(inv_of()),
                restart=restart_note, streamed=bool(streamed),
                clock0=clock0)


def restart_round_trip(stepper, state, cfg, store, setup, outdir, name, step,
                       log) -> dict:
    """Checkpoint the streamed domain, read it back, and prove it came back.

    Two assertions and they are opposite in sign, because either alone is
    satisfiable by a broken implementation: reading the checkpoint back must
    reproduce the digest EXACTLY, and a single carrier perturbed by one ULP
    must make it differ.  Without the second, a digest that ignored the
    arrays would pass the first.
    """
    from tilestream import checkpoint

    if store is None or setup is None:
        return {}
    path = os.path.join(outdir, f"wrfrst_{name}_{step:05d}.npz")
    before, _ = digest_of(store)
    t = time.perf_counter()
    checkpoint.write_store_restart(path, store, dict(stepper.scalars), setup,
                                   cfg)
    w = time.perf_counter() - t

    # Perturb, then read back: the read has to REPAIR the perturbation, which
    # proves it wrote real bytes and not a header.
    victim = sorted(store)[0]
    arr = store[victim]
    keep = np.array(arr.flat[0])
    arr.flat[0] = np.nextafter(arr.flat[0], np.float32(np.inf)) \
        if arr.dtype.kind == "f" else arr.flat[0] + 1
    perturbed, _ = digest_of(store)

    t = time.perf_counter()
    scal = checkpoint.read_store_restart(path, store, setup, cfg)
    r = time.perf_counter() - t
    after, _ = digest_of(store)
    size = os.path.getsize(path)

    # The clock comes back out of the header and is put back on the domain,
    # in place, because TiledRun holds this very dict: a round trip that
    # restored 5.76 GiB of fields and left elapsed_seconds where it was would
    # resume with the fields of step N and the cadence of step N, which looks
    # right and is only right by accident.
    clock_ok = float(scal["elapsed_seconds"]) == float(
        stepper.scalars["elapsed_seconds"])
    stepper.scalars.clear()
    stepper.scalars.update(scal)

    ok = after == before
    sensitive = perturbed != before
    log(f"    [{name}] RESTART round trip at step {step}: "
        f"write {w:.2f} s, read {r:.2f} s, {size / 1e9:.2f} GB, "
        f"fields {'IDENTICAL' if ok else 'DIFFER -- FAIL'}; "
        f"clock {scal['elapsed_seconds']:.1f} s "
        f"{'preserved' if clock_ok else 'LOST -- FAIL'}; "
        f"1-ULP control {'fired' if sensitive else 'DID NOT FIRE -- FAIL'}")
    if not ok:
        arr.flat[0] = keep
    return dict(step=step, path=path, bytes=size, write_s=w, read_s=r,
                digest_before=before, digest_after=after,
                identical=bool(ok), ulp_control_fired=bool(sensitive),
                clock_preserved=bool(clock_ok),
                scalars_read={k: v for k, v in scal.items()
                              if k != "call_counts"})


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def save_trace(leg, outdir, tag) -> str:
    """The per-step trace, as arrays, next to the log.

    The summary in the JSON is a dozen numbers and they are the answer only
    while the answer is "flat".  The moment something is NOT flat, the
    question becomes WHEN it stopped being flat, and that is a 4320-point
    series, not a slope.  Written as ``.npz`` because a failed run's trace is
    the deliverable and a CSV of 4320 x 8 float64 is 1.4 MB of text to
    transfer off a rented box that may be about to be destroyed.
    """
    tr = leg["trace"]
    cols = ("step", "ms", "model_s", "wall_s", "vram_used", "pool_total",
            "pinned", "pinned_blocks", "pinned_pool", "rss", "cgroup")
    arrays = {c: np.asarray([r[c] for r in tr], dtype=np.float64)
              for c in cols}
    path = os.path.join(outdir, f"trace{tag}_{leg['name']}.npz")
    np.savez_compressed(path, **arrays)
    return path


def report_leg(leg, cfg, cells, leak, log) -> dict:
    ms = np.asarray(leg["steps_ms"], dtype=np.float64)
    tr = leg["trace"]
    dt = float(cfg.dt)
    dec = deciles(ms)
    pin = [r["pinned"] for r in tr]
    vram = [r["vram_used"] for r in tr]
    pool = [r["pool_total"] for r in tr]
    cg = [r["cgroup"] for r in tr]
    # Two windows, because one cannot answer the question.  The first 2% of
    # steps carry pipeline priming and first-touch page faults, so fitting
    # from step 0 reports a startup transient as a leak; but fitting the
    # whole run from step k0 still lets a big early jump dominate a long flat
    # tail.  The LAST HALF is the window a leak has to show up in -- a run
    # that is still acquiring memory halfway through its life is leaking,
    # whatever the first half did.
    k0 = max(8, len(pin) // 50)
    half = len(pin) // 2
    out = dict(
        name=leg["name"], wall_s=leg["wall_s"], nsteps=leg["nsteps"],
        median_ms=float(np.median(ms)), mean_ms=float(ms.mean()),
        p95_ms=float(np.percentile(ms, 95)), min_ms=float(ms.min()),
        max_ms=float(ms.max()), deciles_ms=dec,
        decile_ratio=(dec[-1] / dec[0] if dec and dec[0] else float("nan")),
        ns_per_cell=float(np.median(ms)) * 1e6 / cells,
        pinned_slope_per_1k=slope_per_1k(pin[k0:]),
        vram_slope_per_1k=slope_per_1k(vram[k0:]),
        pool_slope_per_1k=slope_per_1k(pool[k0:]),
        cgroup_slope_per_1k=slope_per_1k(cg[k0:]),
        pinned_slope_last_half=slope_per_1k(pin[half:]),
        vram_slope_last_half=slope_per_1k(vram[half:]),
        cgroup_slope_last_half=slope_per_1k(cg[half:]),
        vram_max=float(np.max(vram)), pinned_max=float(np.max(pin)),
        pinned_first=pin[k0], pinned_last=pin[-1],
        vram_first=vram[k0], vram_last=vram[-1],
        cgroup_first=cg[k0], cgroup_last=cg[-1],
        model_s=tr[-1]["model_s"],
        clock_drift_s=tr[-1]["model_s"] - (leg["clock0"] + leg["nsteps"] * dt),
        final_digest=leg["final_digest"], nonfinite=leg["nonfinite"],
        scalars={k: v for k, v in leg["scalars"].items()},
        frames=leg["frames"], restart=leg["restart"],
        leak_injected_per_1k=leak.per_1k_bytes)
    log("")
    log(f"  {leg['name']}: {leg['wall_s'] / 60:.2f} min for {leg['nsteps']} "
        f"steps = {leg['nsteps'] * dt / 3600:.2f} forecast hours "
        f"({leg['nsteps'] * dt / leg['wall_s']:.1f}x real time)")
    log(f"      median {out['median_ms']:.1f} ms  mean {out['mean_ms']:.1f}  "
        f"p95 {out['p95_ms']:.1f}  min {out['min_ms']:.1f}  "
        f"max {out['max_ms']:.1f}  ns/cell {out['ns_per_cell']:.1f}")
    log("      decile medians (ms): "
        + " ".join(f"{v:.0f}" for v in dec)
        + f"   last/first {out['decile_ratio']:.4f}x")
    log(f"      pinned  {out['pinned_first'] / 2 ** 30:7.3f} -> "
        f"{out['pinned_last'] / 2 ** 30:7.3f} GiB (max "
        f"{out['pinned_max'] / 2 ** 30:.3f})   slope "
        f"{out['pinned_slope_per_1k'] / 2 ** 20:+9.3f} MiB/1k, last half "
        f"{out['pinned_slope_last_half'] / 2 ** 20:+9.3f} MiB/1k")
    log(f"      VRAM    {out['vram_first'] / 2 ** 30:7.3f} -> "
        f"{out['vram_last'] / 2 ** 30:7.3f} GiB (max "
        f"{out['vram_max'] / 2 ** 30:.3f})   slope "
        f"{out['vram_slope_per_1k'] / 2 ** 20:+9.3f} MiB/1k, last half "
        f"{out['vram_slope_last_half'] / 2 ** 20:+9.3f} MiB/1k")
    log(f"      cgroup  {out['cgroup_first'] / 2 ** 30:7.3f} -> "
        f"{out['cgroup_last'] / 2 ** 30:7.3f} GiB   slope "
        f"{out['cgroup_slope_per_1k'] / 2 ** 20:+9.3f} MiB/1k, last half "
        f"{out['cgroup_slope_last_half'] / 2 ** 20:+9.3f} MiB/1k")
    log(f"      model clock {out['model_s']:.1f} s vs clock0+nsteps*dt "
        f"{leg['clock0'] + leg['nsteps'] * dt:.1f} s   "
        f"drift {out['clock_drift_s']:+.6g} s")
    log(f"      scalars {out['scalars']}")
    if leak.per_1k_bytes:
        seen = out["pinned_slope_per_1k"]
        want = leak.per_1k_bytes
        log(f"      LEAK CONTROL: injected {want / 2 ** 20:.1f} MiB/1000 "
            f"steps, detector reported {seen / 2 ** 20:.1f} -- "
            f"{'SEEN' if seen > 0.5 * want else 'MISSED -- the detector is blind'}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pair",
                    choices=("pair", "stream", "resident"))
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--nz", type=int, default=49)
    ap.add_argument("--dt", type=float, default=15.0)
    ap.add_argument("--steps", type=int, default=4320)
    ap.add_argument("--nbuffers", type=int, default=2)
    ap.add_argument("--frame-every", type=int, default=240)
    ap.add_argument("--digest-every", type=int, default=480)
    ap.add_argument("--restart-at", type=int, default=0)
    ap.add_argument("--leak-mib", type=float, default=0.0)
    ap.add_argument("--leak-every", type=int, default=0)
    ap.add_argument("--out", default="endure-out")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fh = open(os.path.join(args.out, f"endure{args.tag}.log"), "a",
              buffering=1)

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        fh.write(msg + "\n")

    import cupy as cp
    from woof.core import streaming
    from tilestream import checkpoint, driver, gather, harness, hoststore
    from tilestream import physics_inventory as physinv

    dev = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    cfg = build_cfg(args.n, args.nz, args.dt)
    halo = harness.halo_radius(cfg)
    cells = args.n * args.n * args.nz
    tile = args.tile or args.n // 2
    if args.n % tile:
        raise SystemExit(f"tile {tile} does not divide n {args.n}")
    win = tile + 2 * halo
    dt = float(cfg.dt)

    stepra = int(round(cfg.radt_minutes * 60.0 / dt))
    stepcu = int(round(cfg.cudt_minutes * 60.0 / dt))
    nrad = sum(1 for i in range(2, 2 + args.steps)
               if i == 1 or (stepra > 1 and i % stepra == 1))
    ncu = sum(1 for i in range(2, 2 + args.steps)
              if i == 1 or (stepcu > 1 and i % stepcu == 0))

    log("=" * 78)
    log(f"ENDURE  {dev}  {time.strftime('%Y-%m-%d %H:%M:%S')}  mode={args.mode}")
    log(f"domain {args.n}^2 x {args.nz} = {cells / 1e6:.1f} Mcell, "
        f"{args.n * cfg.dx / 1000:.0f} km per side at dx={cfg.dx / 1000:.0f} km")
    log(f"dt={dt:.0f}s  halo={halo}  tile {tile} -> "
        f"{(args.n // tile) ** 2} tiles, window {win}^2, "
        f"redundancy {(win / tile) ** 2:.4f}x, nbuffers={args.nbuffers}")
    log(f"{args.steps} steps = {args.steps * dt / 3600:.2f} FORECAST HOURS")
    log(f"cadence inside the timed window: radiation {nrad}x "
        f"(every {stepra} steps), cumulus {ncu}x (every {stepcu}), "
        f"microphysics/surface/PBL every step")
    log(f"history every {args.frame_every} steps "
        f"({args.frame_every * dt / 3600:.2f} fh), digest every "
        f"{args.digest_every}, restart round trip at {args.restart_at or 'none'}")
    leak = Leak(args.leak_mib, args.leak_every)
    if leak.per_1k_bytes:
        log(f"LEAK CONTROL ARMED: {args.leak_mib} MiB every "
            f"{args.leak_every} steps = {leak.per_1k_bytes / 2 ** 20:.1f} "
            f"MiB / 1000 steps must appear in the pinned slope")

    out: dict = dict(mode=args.mode, device=dev, argv=vars(args),
                     n=args.n, nz=args.nz, cells=cells, dt=dt, halo=halo,
                     tile=tile, window=win, steps=args.steps,
                     forecast_hours=args.steps * dt / 3600.0,
                     rad_fires=nrad, cu_fires=ncu)

    t0 = time.perf_counter()
    state, geo = build_domain(cfg)
    cp.cuda.runtime.deviceSynchronize()
    log(f"built + warmed in {time.perf_counter() - t0:.0f} s; resident "
        f"{sample()['vram_used'] / 2 ** 30:.2f} GiB")
    inv = physinv.carrier_inventory(state)
    bpc = sum(v.nbytes for v in inv.values()) / cells
    log(f"{len(inv)} carriers, {bpc:.1f} B/cell -> "
        f"{bpc * cells / 2 ** 30:.2f} GiB of domain state")
    out.update(carriers=len(inv), bytes_per_cell=bpc)

    start = snapshot_t0(state)
    geo_store = {k: gather.pinned_copy(v)
                 for k, v in driver.geography_inventory(state).items()}
    led = hoststore.pinned_ledger()
    log(f"t0 snapshot + geography pinned: ledger says "
        f"{led['total_bytes'] / 2 ** 30:.2f} GiB "
        f"({led['blocks']} raw blocks {led['block_bytes'] / 2 ** 30:.2f} GiB, "
        f"pool {led['pool_total_bytes'] / 2 ** 30:.2f} GiB)")
    d0, _ = digest_of(start)
    scalars0 = physinv.carrier_scalars(state)
    log(f"shared t0 digest {d0}   scalars {scalars0}")
    out.update(t0_digest=d0, pinned_after_snapshot=led["total_bytes"],
               scalars0=scalars0)
    cp.get_default_memory_pool().free_all_blocks()

    setup = checkpoint.DomainSetup.capture(state, cfg)
    legs = {}

    if args.mode in ("pair", "resident"):
        log("-" * 78)
        log("LEG R  resident, woof.core.dycore.step itself")
        from woof.core.dycore import step as dycore_step

        stepper = streaming.make_stepper(state, cfg, streaming.OFF)
        if stepper is not dycore_step:
            raise RuntimeError("the OFF contract is broken: make_stepper "
                               "returned a wrapper, not dycore.step itself")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            legR = run_leg("R", stepper, state, cfg, None, args.steps,
                           log=log, outdir=args.out,
                           frame_every=args.frame_every,
                           digest_every=args.digest_every, leak=leak,
                           setup=setup)
        legs["R"] = report_leg(legR, cfg, cells, leak, log)
        legs["R"]["digests"] = legR["digests"]
        legs["R"]["trace_path"] = save_trace(legR, args.out, args.tag)
        log(f"      trace: {legs['R']['trace_path']}")
        leak.release()

    if args.mode in ("pair", "stream"):
        log("-" * 78)
        log("LEG S  streamed, woof.core.streaming.StreamedDomain")
        if args.mode == "pair":
            restore_t0(state, start, scalars0)
            d, _ = digest_of(physinv.carrier_inventory(state))
            got = physinv.carrier_scalars(state)
            log(f"  t0 restored onto the device: digest {d} "
                f"{'== t0' if d == d0 else '!= t0 -- RESTORE FAILED'}; "
                f"scalars {'== t0' if got == scalars0 else f'!= t0 {got}'}")
            if d != d0 or got != scalars0:
                raise RuntimeError("the t0 restore did not reproduce t0; the "
                                   "two legs would not start from the same "
                                   "state and any comparison would be void")
            cp.get_default_memory_pool().free_all_blocks()
        t = time.perf_counter()
        stepper = streamed_stepper(state, cfg, tile, args.nbuffers,
                                   geo_store, log)
        cp.cuda.runtime.deviceSynchronize()
        setup_s = time.perf_counter() - t
        led = hoststore.pinned_ledger()
        log(f"  attached in {setup_s:.1f} s; pinned ledger now "
            f"{led['total_bytes'] / 2 ** 30:.2f} GiB "
            f"(raw {led['block_bytes'] / 2 ** 30:.2f}, "
            f"pool {led['pool_total_bytes'] / 2 ** 30:.2f}); VRAM "
            f"{sample()['vram_used'] / 2 ** 30:.2f} GiB")
        out["stream_setup_s"] = setup_s
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            legS = run_leg("S", stepper, state, cfg, stepper.store,
                           args.steps, log=log, outdir=args.out,
                           frame_every=args.frame_every,
                           digest_every=args.digest_every, leak=leak,
                           setup=setup, restart_at=args.restart_at)
        legs["S"] = report_leg(legS, cfg, cells, leak, log)
        legs["S"]["digests"] = legS["digests"]
        legs["S"]["trace_path"] = save_trace(legS, args.out, args.tag)
        log(f"      trace: {legs['S']['trace_path']}")

    out["legs"] = legs

    if args.mode == "pair":
        log("=" * 78)
        R, S = legs["R"], legs["S"]
        same = R["final_digest"] == S["final_digest"]
        log(f"R final {R['final_digest']}")
        log(f"S final {S['final_digest']}")
        log(f"AFTER {args.steps} STEPS ({out['forecast_hours']:.2f} FORECAST "
            f"HOURS): {'BIT-FOR-BIT IDENTICAL' if same else 'DIFFERENT'}")
        # The series, not just the end: a mismatch names the checkpoint it
        # started at, which is the difference between a result and a report
        # that something is wrong somewhere.
        first_bad = None
        for a, b in zip(R["digests"], S["digests"]):
            if a["digest"] != b["digest"]:
                first_bad = a["step"]
                break
        log(f"  checkpoint digests compared: {len(R['digests'])} pairs, "
            + ("all equal" if first_bad is None
               else f"FIRST DIVERGENCE AT STEP {first_bad}"))
        for a, b in zip(R["digests"], S["digests"]):
            log(f"    step {a['step']:>5d}  R {a['digest']}  S {b['digest']}  "
                f"{'ok' if a['digest'] == b['digest'] else 'DIFFER'}  "
                f"nonfinite R={a['nonfinite']} S={b['nonfinite']}")
        log(f"  scalars R {R['scalars']}")
        log(f"  scalars S {S['scalars']}")
        log(f"  wall R {R['wall_s'] / 60:.2f} min   S {S['wall_s'] / 60:.2f} "
            f"min   S/R {S['wall_s'] / R['wall_s']:.3f}x")
        out.update(bitexact=bool(same), first_divergence=first_bad)

    path = os.path.join(args.out, f"endure{args.tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
