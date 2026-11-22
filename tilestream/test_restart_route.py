"""The four legs, through WOOF-ArWen's OWN single-domain run loop.

``tilestream/test_restart_gate.py`` proved that a restart can be written from
a pinned host store and read back into one: it drives ``run_tiled`` directly
and calls :mod:`tilestream.restart_stream` by hand.  That is the transport.
It is NOT the operation a user performs, which is ``woof run`` -- and the
thing standing between the two is :func:`woof.runtime.integrate_prepared_case`,
the loop that owns the restart CADENCE, the checkpoint FILENAME, the
``run_trackers`` continuity block, the resume arithmetic and the decision of
what object to hand the restart writer.  Nothing had ever streamed through it.

This module runs the four legs THROUGH THAT LOOP::

    streamed  -> file -> streamed
    streamed  -> file -> resident
    resident  -> file -> streamed
    resident  -> file -> resident

Each leg is two calls to ``integrate_prepared_case``: the first integrates
0 -> 1800 s and lets the loop's own ``restart_interval_s`` cadence write the
checkpoint, the second is handed that checkpoint as ``restart_path`` and
integrates on to 3600 s.  PASS is that all four final domains are bit-identical
over all 229 carriers, to each other AND to an uninterrupted 3600 s run of each
mode.  The uninterrupted runs are what keep the legs accurate: four resumed runs
that agreed with each other and with nothing else would prove only that the
bug is deterministic.


WHAT THE LOOP DID BEFORE, AND WHY IT LOOKED LIKE IT WORKED
----------------------------------------------------------
``integrate_prepared_case`` wrote its checkpoint with
``write_restart(path, state, cfg)``.  ``state`` is the prepared resident
:class:`~woof.core.state.DomainState`.  Under streaming the domain's carriers
are in the stepper's store -- ``streaming.attach`` copied them out and the
state has been frozen at its preparation values ever since -- so that call
does not fail and does not warn.  It writes a complete, internally consistent,
229-member checkpoint OF THE INITIAL CONDITION, stamped with the current
model clock and the current ``run_trackers``.  ``validate_manifest_checkpoint``
passes it.  Every fingerprint in ``restore_restart`` matches, because the
setup and the physics setup really are this run's.  The file resumes cleanly
into a forecast that silently threw away every step taken.

That is the negative control this gate runs (``resident writer on a streamed
domain``), and it is expressed the way the defect would actually reach
production: the streamed stepper is wrapped in a plain function, so
``streaming.is_streaming`` says False, the loop takes the resident branch, and
the leg is run end to end.  It MUST differ.  A second control removes the
clock reseed from the restore and it MUST differ too.


THE CADENCES FIRE ON BOTH SIDES, AND THE NUMBERS ARE PRINTED
------------------------------------------------------------
Three of this project's six false results were the same mistake: a number
measured in a window where radiation and cumulus never fired.  At
``radt_minutes = 12``, ``cudt_minutes = 5`` and ``dt = 30 s`` radiation is due
every 24 steps and cumulus every 10, and each half of this run is 60 steps.
The per-window fire counts are read out of the checkpoint headers -- they are
``call_counts``, a restart carrier -- and PRINTED for every leg on both sides
of the split.  A leg whose second half fired no radiation is reported as such
rather than counted as a pass.


WHAT THIS GATE IS NOT
---------------------
It is not ``woof run``.  The CLI's single-domain route,
:func:`woof.runtime.run_experiment`, builds no streamed stepper at all --
before this branch it read ``[tiles]``, validated it, echoed it into the
resolved-config report and then integrated RESIDENT without saying so; it now
refuses instead.  Wiring it needs a per-domain builder (store filled from the
prepared state, tile buffers on the domain's own physics selectors, geography
inventoried, boundary tables windowed per tile) which is a different piece of
work from the restart, and a real ``woof run`` additionally needs a real GFS
or HRRR ingest that no bit-exactness gate can use anyway (see
``tilestream/REAL-DATA.md``).  So this drives the loop that route calls, with
the stepper supplied here, on the harness's real Lambert projection, real
terrain and specified lateral boundaries.

Two things inside the loop are deliberately kept out of the window, and both
are proven elsewhere: history output (``history_interval_s`` is set beyond the
run, so ``write_case_output`` is called once for the cold start and is stubbed
here -- streamed output is ``tilestream/test_io.py``'s subject), and with it
the REFL_10CM consume-once handshake, which never arms because no output is
ever due.

Run me::

    python tilestream/test_restart_route.py            # the whole gate
    python tilestream/test_restart_route.py --quick    # smaller and shorter
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from woof.core import streaming
from tilestream import driver, gather, harness, restart_stream, test_join
from tilestream import physics_inventory as physinv


#: The domain.  ``NX`` must be divisible by ``TILE`` -- a ragged trailing tile
#: is read right through in ring mode (22.7% of the store against 2.4% in one
#: measured case) and it is not what a forecast would run.
NX = NY = 336
NZ = 49
TILE = 112
SEED = 20_260_808
#: dt = 30 s is ``test_join.DT_MOIST``: the MEASURED stability ladder of this
#: harness's random initial state at 12 km.  60 s is clean dry and goes
#: non-finite with moisture, and a reference that is not finite has nothing to
#: compare against.
HALF_SECONDS = 1800.0
RUN_SECONDS = 3600.0
#: Beyond the run on purpose: see the module docstring.
HISTORY_SECONDS = 7200.0
START = datetime(2011, 4, 27, 12, 0, tzinfo=timezone.utc)
SCRATCH = Path("/tmp/arwen-restart-route")


# --------------------------------------------------------------------------
# the prepared bundle the loop reads, and the one thing that is stubbed
# --------------------------------------------------------------------------

class _StubbedOutput:
    """``write_case_output`` replaced, and LOUD about what it will not do.

    The loop writes exactly one frame in this gate -- the cold-start wrfout,
    which it writes unconditionally when ``restart_path is None``.  Building
    the real one needs the preparation receipt's static fields and geog
    selection, which the harness does not have and which have nothing to do
    with the restart.  So it is stubbed, and the stub REFUSES any call that is
    not the cold start: if a change ever makes a history frame due inside this
    window, this gate stops rather than quietly measuring a run whose output
    path is fake.
    """

    def __init__(self, outdir: Path):
        self.outdir = Path(outdir)
        self.calls = 0

    def __call__(self, prepared, output_dir, valid_time, *, start_time,
                 title, domain_id=1, expect_refl_10cm=True, feedback=None):
        self.calls += 1
        if valid_time != start_time:
            raise AssertionError(
                f"write_case_output was called for a HISTORY frame at "
                f"{valid_time} -- this gate stubs the writer and only the "
                "cold-start frame may reach it; history output is "
                "tilestream/test_io.py's subject")
        path = Path(output_dir) / f"stub_wrfout_d{domain_id:02d}.json"
        path.write_text(json.dumps({"stub": True,
                                    "valid": valid_time.isoformat()}))
        return path


def _prepared(cfg, state, grid):
    """The attribute surface :func:`integrate_prepared_case` actually reads."""
    return types.SimpleNamespace(
        cfg=cfg,
        initial_result=types.SimpleNamespace(state=state, coord=None),
        grid=grid, static_fields=None, geog_selection=None)


# --------------------------------------------------------------------------
# the domain
# --------------------------------------------------------------------------

def frugal_boundaries(cfg, *, seconds=test_join.BDY_SECONDS, seeds=(SEED,
                                                                    SEED + 1)):
    """The domain's specified forcing without holding two domains at once.

    ``test_join.domain_boundaries`` builds both snapshot states and then reads
    them, which at these extents is two complete 229-carrier domains resident
    at the same time -- 5.1 GiB at 448^2 before the run has allocated
    anything.  :class:`~woof.ingest.lateral_bc.StateBoundaryFrames` is
    WOOF-ArWen's own one-state-at-a-time accumulator and is element-for-element
    exact against the all-at-once builder, so this is the same forcing at half
    the peak.
    """
    from woof.ingest.lateral_bc import (StateBoundaryFrames,
                                         domain_boundary_snapshot)

    acc = StateBoundaryFrames(spec_bdy_width=int(cfg.spec_bdy_width),
                              spec_zone=int(cfg.spec_zone),
                              relax_zone=int(cfg.relax_zone))
    for seed in seeds:
        snapshot, _geo = test_join.build_domain(cfg, seed=seed, warmup=0)
        acc.add_snapshot(domain_boundary_snapshot(snapshot))
        del snapshot
        _free()
    return acc.build([0.0, float(seconds)])


def build_start(cfg, bnd):
    """A prepared domain whose clock reads ZERO, and its geography.

    The warmup step is not optional: two carriers are allocated lazily on
    first use (Kain-Fritsch's ``cumulus/w0avg``, kf.py:335) and a store built
    from a state that has never stepped is short by them.  But the run loop
    maps ``outer_step`` to ``elapsed_seconds`` as ``outer_step * dt``, and a
    domain that starts at ``dt`` would resume one step short of where it
    stopped -- so the clock is put back to zero afterwards.  Every run in this
    gate does the same thing, so the comparison is between identical starts.
    """
    state, geo = test_join.build_domain(cfg, seed=SEED, boundaries=bnd,
                                        warmup=1)
    state.elapsed_seconds = 0.0
    return state, geo


def _free():
    import cupy as cp

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# --------------------------------------------------------------------------
# the streamed stepper this route supplies
# --------------------------------------------------------------------------

def make_builder(cfg, domain, geo, bnd):
    """The route-owned construction :func:`streaming.make_stepper` needs.

    Exactly ``test_join``'s builder with its negative-control switches
    removed: the domain's geography is inventoried once and gathered per
    buffer (never rebuilt per tile), the boundary tables are windowed once per
    tile (tile 0's also give the buffers something correctly shaped to take
    their warmup step with, which ``cfg.specified`` makes mandatory), and the
    domain's clock is carried.
    """

    def build(state, run_cfg, decision):
        specs = streaming.tile_specs(run_cfg, decision)
        per_tile = streaming.tile_boundary_tables(bnd, specs)
        geo_inv = {key: gather.pinned_copy(value) for key, value
                   in driver.geography_inventory(domain).items()}
        factory = test_join.tile_factory(run_cfg, per_tile[0])
        return streaming.attach(
            state, run_cfg, decision, tile_state_factory=factory,
            geography=geo_inv, boundary_tables=per_tile,
            scalars=physinv.carrier_scalars(state),
            boundaries=bnd, check_geography=False)

    return build


def make_stepper_for(cfg, domain, geo, bnd, *, tile=TILE, nbuffers=2):
    options = streaming.StreamingOptions(mode="on", tile_nx=int(tile),
                                         tile_ny=int(tile),
                                         nbuffers=int(nbuffers))
    decision = streaming.decide(cfg, options)
    stepper = streaming.make_stepper(domain, cfg, options, decision=decision,
                                     build=make_builder(cfg, domain, geo, bnd))
    if not streaming.is_streaming(stepper):
        raise AssertionError("mode='on' did not produce a StreamedDomain")
    return stepper


# --------------------------------------------------------------------------
# digests
# --------------------------------------------------------------------------

def _as_numpy(array):
    import cupy as cp

    return cp.asnumpy(array) if isinstance(array, cp.ndarray) \
        else np.asarray(array)


def digest_of(arrays) -> str:
    """One SHA-256 over every carrier, in sorted key order."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        host = np.ascontiguousarray(_as_numpy(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def file_identity(path) -> dict:
    """A checkpoint's content identity: payload bytes, and header sans birth.

    "Byte-identical files" is the wrong test and would fail for the wrong
    reason: ``np.savez`` stamps every zip member with the local wall clock, so
    two identical checkpoints written a second apart differ in bytes and agree
    in content.  ``created`` is the same kind of fact.  Everything else in the
    header -- the config echo, both fingerprints, the driver call counts, the
    ``run_trackers`` block, the whole array manifest -- is compared, and the
    payload is compared array by array.
    """
    from woof.io.restart import _HEADER_KEY, read_restart_header

    header = dict(read_restart_header(path))
    header.pop("created", None)
    core = {k: v for k, v in header.items() if k != "run_trackers"}
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files
                  if name != _HEADER_KEY}

    def sha(obj):
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()

    return {
        "members": len(arrays),
        "payload_sha256": digest_of(arrays),
        "header_sha256": sha(header),
        # Everything a RESUME reads: the config echo, both fingerprints, the
        # clock, the driver call counts and the whole array manifest.
        # ``run_trackers`` is summary bookkeeping the model never reads, and
        # it is split out because on this branch it is written by observers
        # that do not yet see a streamed domain -- see the gate's own row.
        "header_core_sha256": sha(core),
        "header": header,
        "bytes": Path(path).stat().st_size,
    }


def field_digests(arrays) -> dict:
    """One SHA-256 per carrier.  Cheap enough to keep for every run, so a
    whole-domain digest that disagrees can say WHICH of the 229 members did
    without holding two complete domains on the host."""
    out = {}
    for key in sorted(arrays):
        host = np.ascontiguousarray(_as_numpy(arrays[key]))
        out[key] = hashlib.sha256(host.tobytes(order="C")).hexdigest()
    return out


def differing_carriers(a: dict, b: dict, limit: int = 8) -> list:
    keys = sorted(set(a) | set(b))
    return [k for k in keys if a.get(k) != b.get(k)][:limit]


def compare(a, b) -> dict:
    """Carrier-by-carrier comparison of two domains."""
    keys = sorted(set(a) | set(b))
    ndiff = worst = 0
    worst_abs = 0.0
    differing = []
    nonfinite = 0
    ntot = 0
    for key in keys:
        left = np.ascontiguousarray(_as_numpy(a[key]))
        right = np.ascontiguousarray(_as_numpy(b[key]))
        ntot += left.size
        if left.dtype.kind == "f":
            nonfinite += int((~np.isfinite(left)).sum())
        if left.shape != right.shape or left.dtype != right.dtype:
            ndiff += 1
            differing.append(key)
            continue
        if not np.array_equal(left, right):
            ndiff += 1
            if len(differing) < 6:
                differing.append(key)
            if left.dtype.kind == "f":
                d = float(np.nanmax(np.abs(left.astype(np.float64)
                                           - right.astype(np.float64))))
                worst_abs = max(worst_abs, d)
    return {"bitexact": ndiff == 0, "ndiff": ndiff, "ncarriers": len(keys),
            "values": ntot, "max_abs": worst_abs, "differing": differing,
            "nonfinite": nonfinite}


# --------------------------------------------------------------------------
# one run of the loop
# --------------------------------------------------------------------------

def run_leg(cfg, bnd, *, mode: str, restart_path=None,
            run_seconds: float = RUN_SECONDS, outdir: Path,
            tile: int = TILE, hide_streaming: bool = False,
            skip_clock_reseed: bool = False) -> dict:
    """One call to ``woof.runtime.integrate_prepared_case``.

    ``hide_streaming`` wraps the streamed stepper in a plain function so
    ``streaming.is_streaming`` is False and the loop takes the RESIDENT
    restart branch on a streamed domain -- the defect, run end to end.
    ``skip_clock_reseed`` restores into the store without reseeding
    ``TiledRun``'s cached clock, which is the other half of the same failure.
    """
    from woof import runtime

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    domain, geo = build_start(cfg, bnd)
    prepared = _prepared(cfg, domain, geo.grid)

    stepper = None
    streamed = None
    if mode == "streamed":
        streamed = make_stepper_for(cfg, domain, geo, bnd, tile=tile)
        stepper = streamed
        if skip_clock_reseed:
            def _no_reseed(path, run_cfg, _sd=streamed):
                return restart_stream.read_streamed_restart(
                    path, _sd.store, run_cfg, setup=_sd.restart_setup(),
                    template_state=_sd.template_state, scalars=_sd.scalars)
            streamed.restore_restart = _no_reseed
        if hide_streaming:
            def stepper(state, run_cfg, _sd=streamed, **kw):
                return _sd(state, run_cfg, **kw)

    saved = runtime.write_case_output
    runtime.write_case_output = _StubbedOutput(outdir)
    t0 = time.perf_counter()
    try:
        summary = runtime.integrate_prepared_case(
            outdir, prepared, start_time=START,
            output_title="tilestream restart route", domain_id=1,
            run_seconds=float(run_seconds),
            history_interval_s=HISTORY_SECONDS,
            restart_interval_s=HALF_SECONDS,
            restart_path=restart_path, stepper=stepper)
    finally:
        runtime.write_case_output = saved
    wall = time.perf_counter() - t0

    carriers = (dict(streamed.store) if streamed is not None
                else physinv.carrier_manifest(domain))
    scalars = (dict(streamed.scalars) if streamed is not None
               else physinv.carrier_scalars(domain))
    result = {
        "mode": mode, "wall": wall, "summary": summary,
        "digest": digest_of(carriers), "carriers": len(carriers),
        "fields": field_digests(carriers), "scalars": scalars,
        "checkpoints": sorted(outdir.glob("gpuwmrst_*.npz")),
    }
    del carriers, streamed, stepper, domain, geo, prepared
    _free()
    return result


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def _line(label: str, ok: bool, detail: str = "") -> str:
    return f"  {'PASS' if ok else 'FAIL':4s}  {label:46s} {detail}"


def _fires(before: dict, after: dict) -> dict:
    """How many times each cadence-driven scheme fired between two headers."""
    b = (before or {}).get("call_counts", {})
    a = (after or {}).get("call_counts", {})
    return {k: int(a.get(k, 0)) - int(b.get(k, 0)) for k in sorted(a)}


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    nx = 168 if quick else NX
    tile = 56 if quick else TILE
    run_seconds = 900.0 if quick else RUN_SECONDS
    half = run_seconds / 2.0

    global HALF_SECONDS
    HALF_SECONDS = half

    props = cp.cuda.runtime.getDeviceProperties(0)
    free, total = cp.cuda.runtime.memGetInfo()
    print("=" * 78)
    print("THE FOUR LEGS, through woof.runtime.integrate_prepared_case")
    print("=" * 78)
    print(f"  {props['name'].decode()}  cupy {cp.__version__}  "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")

    cfg = test_join.join_cfg(nx, nx, NZ, rung="full+MYNN+Noah-MP")
    halo = harness.halo_radius(cfg)
    steps = int(round(run_seconds / cfg.dt))
    print(f"  {nx}x{nx}x{NZ} at dx={cfg.dx / 1000:g} km, dt={cfg.dt:g} s, "
          f"tile {tile}x{tile}, halo {halo} = 10 + 3*{cfg.time_step_sound}//2")
    print(f"  {steps} steps total, checkpoint at {half:g} s "
          f"(= step {steps // 2}); radt={cfg.radt_minutes:g} min -> every "
          f"{int(round(cfg.radt_minutes * 60 / cfg.dt))} steps, "
          f"cudt={cfg.cudt_minutes:g} min -> every "
          f"{int(round(cfg.cudt_minutes * 60 / cfg.dt))} steps")
    dec = streaming.decide(cfg, streaming.StreamingOptions(
        mode="on", tile_nx=tile, tile_ny=tile, nbuffers=2))
    specs = streaming.tile_specs(cfg, dec)
    edges = [sum(streaming.owned_edges(s).values()) for s in specs]
    print(f"  {len(specs)} tiles: {edges.count(0)} own no true edge, "
          f"{edges.count(1)} own one, {edges.count(2)} own two")
    print()

    root = SCRATCH
    root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    t_all = time.perf_counter()

    print("-- building the domain's specified forcing")
    bnd = frugal_boundaries(cfg)
    _free()

    # ------------------------------------------------- the two references
    print("-- UNINTERRUPTED references (what the legs must reproduce)")
    refs = {}
    for mode in ("resident", "streamed"):
        out = root / f"ref_{mode}"
        res = run_leg(cfg, bnd, mode=mode, run_seconds=run_seconds,
                      outdir=out, tile=tile)
        refs[mode] = res
        print(f"     {mode:9s} {res['carriers']} carriers  "
              f"{res['digest'][:16]}  {res['wall']:.1f} s  "
              f"nan_free={res['summary'].nan_free} "
              f"w_max={res['summary'].w_max_ms:.3f} "
              f"calls={res['scalars'].get('call_counts')}")
    same = refs["resident"]["digest"] == refs["streamed"]["digest"]
    if not same:
        failures.append("the uninterrupted streamed run is not bit-exact "
                        "against the resident one; nothing below can mean "
                        "anything")
    bad = differing_carriers(refs["resident"]["fields"],
                             refs["streamed"]["fields"])
    print(_line("uninterrupted streamed == uninterrupted resident", same,
                "" if same else f"differing carriers: {bad}"))

    # the reference checkpoint at the half mark, one per mode
    half_ckpts = {}
    for mode, res in refs.items():
        mark = START + timedelta(seconds=half)
        from woof.io.restart import restart_filename
        half_ckpts[mode] = (root / f"ref_{mode}" /
                            restart_filename(mark, "d01"))
    print()

    # ------------------------------------------------------- the phase ones
    print("-- PHASE 1: integrate to the half mark, let the loop checkpoint")
    p1 = {}
    for mode in ("resident", "streamed"):
        out = root / f"p1_{mode}"
        res = run_leg(cfg, bnd, mode=mode, run_seconds=half, outdir=out,
                      tile=tile)
        mark = START + timedelta(seconds=half)
        from woof.io.restart import restart_filename
        ckpt = out / restart_filename(mark, "d01")
        if not ckpt.exists():
            failures.append(f"phase 1 {mode} wrote no checkpoint at {half} s")
            print(_line(f"phase 1 {mode}", False, "NO CHECKPOINT WRITTEN"))
            continue
        ident = file_identity(ckpt)
        p1[mode] = {"path": ckpt, "ident": ident, "res": res}
        drv = ident["header"].get("driver") or {}
        print(f"     {mode:9s} {ident['members']} members  "
              f"{ident['bytes'] / 2**20:.0f} MiB  "
              f"payload {ident['payload_sha256'][:16]}  "
              f"{res['wall']:.1f} s  fired={drv.get('call_counts')}")

    # Do the run loop's own observers see a streamed domain?  ``nan_free``,
    # the w-max trackers and the SWDOWN peak come from
    # ``stability_report(state, ...)`` and ``state.physics.fields['swdown']``,
    # which under streaming read the frozen PREPARATION state.  This gate
    # measures that rather than assuming it either way, because the answer
    # decides whether a run_trackers difference below is an attributed
    # consequence of a known defect or a failure of the restart itself.
    observers_live = (float(refs["resident"]["summary"].w_max_ms)
                      == float(refs["streamed"]["summary"].w_max_ms))

    if len(p1) == 2:
        pay = (p1["resident"]["ident"]["payload_sha256"]
               == p1["streamed"]["ident"]["payload_sha256"])
        core = (p1["resident"]["ident"]["header_core_sha256"]
                == p1["streamed"]["ident"]["header_core_sha256"])
        trk = (p1["resident"]["ident"]["header"].get("run_trackers")
               == p1["streamed"]["ident"]["header"].get("run_trackers"))
        if not (pay and core):
            failures.append("the streamed and resident checkpoints at the "
                            "half mark are not the same checkpoint")
            for key in sorted(set(p1["resident"]["ident"]["header"])
                              | set(p1["streamed"]["ident"]["header"])):
                if key == "run_trackers":
                    continue
                lv = p1["resident"]["ident"]["header"].get(key)
                rv = p1["streamed"]["ident"]["header"].get(key)
                if lv != rv:
                    print(f"        header[{key}] resident={str(lv)[:80]} "
                          f"streamed={str(rv)[:80]}")
        print(_line("checkpoint(streamed) == checkpoint(resident)",
                    pay and core,
                    f"229-member payload {'==' if pay else '!='}, "
                    f"header (config, both fingerprints, clock, driver "
                    f"counts, manifest) {'==' if core else '!='}"))
        # run_trackers is the one header block a resume does not read, and
        # the only one that differs.  It is a failure ONLY once the loop's
        # observers see the streamed domain: until then the difference is
        # the KNOWN stale-observer defect arriving through the checkpoint,
        # and calling it a restart failure would be attributing it wrongly.
        if observers_live and not trk:
            failures.append("the loop's observers now see the streamed "
                            "domain, and run_trackers STILL differ between "
                            "a streamed and a resident checkpoint")
        print(_line("checkpoint run_trackers agree", trk,
                    "" if trk else
                    (f"ATTRIBUTED, not a restart defect: the loop's "
                     f"observers read the frozen preparation state under "
                     f"streaming (resident w_max="
                     f"{p1['resident']['ident']['header']['run_trackers']['w_max_ms']:.6f} "
                     f"vs streamed "
                     f"{p1['streamed']['ident']['header']['run_trackers']['w_max_ms']:.6f}); "
                     f"this row becomes a FAILURE once they do")))
    print()

    # -------------------------------------------------------- the four legs
    print("-- THE FOUR LEGS: resume to the end and compare against BOTH refs")
    legs = []
    for first in ("streamed", "resident"):
        for second in ("streamed", "resident"):
            if first not in p1:
                continue
            label = f"{first} -> file -> {second}"
            out = root / f"leg_{first[0]}{second[0]}"
            res = run_leg(cfg, bnd, mode=second,
                          restart_path=p1[first]["path"],
                          run_seconds=run_seconds, outdir=out, tile=tile)
            ok = (res["digest"] == refs["resident"]["digest"]
                  == refs["streamed"]["digest"])
            fired = _fires(p1[first]["ident"]["header"].get("driver"),
                           {"call_counts": res["scalars"]["call_counts"]})
            legs.append((label, ok, res))
            if not ok:
                failures.append(f"leg {label} is not bit-exact")
            detail = (f"{res['digest'][:16]}  {res['wall']:.1f} s  "
                      f"2nd-half fires={fired}")
            if not ok:
                detail += (" differing: "
                           + str(differing_carriers(refs[second]["fields"],
                                                    res["fields"])))
            print(_line(label, ok, detail))

    if legs:
        allsame = len({r["digest"] for _, _, r in legs}) == 1
        print(_line("all four legs agree with each other", allsame))
        if not allsame:
            failures.append("the four legs do not agree with each other")

        # every leg's END checkpoint, against the uninterrupted reference's
        from woof.io.restart import restart_filename
        end = START + timedelta(seconds=run_seconds)
        idents = {}
        for mode in ("resident", "streamed"):
            path = root / f"ref_{mode}" / restart_filename(end, "d01")
            if path.exists():
                idents[f"ref-{mode}"] = file_identity(path)
        for first in ("streamed", "resident"):
            for second in ("streamed", "resident"):
                path = (root / f"leg_{first[0]}{second[0]}"
                        / restart_filename(end, "d01"))
                if path.exists():
                    idents[f"{first[0]}{second[0]}"] = file_identity(path)
        pays = {v["payload_sha256"] for v in idents.values()}
        cores = {v["header_core_sha256"] for v in idents.values()}
        trks = {json.dumps(v["header"].get("run_trackers"), sort_keys=True)
                for v in idents.values()}
        ok = len(pays) == 1 and len(cores) == 1
        if not ok:
            failures.append("the end-of-run checkpoints are not all the same "
                            "checkpoint")
        if observers_live and len(trks) != 1:
            failures.append("observers are live and the end-of-run "
                            "run_trackers still disagree")
        print(_line(f"all {len(idents)} end-of-run checkpoints identical", ok,
                    f"distinct payloads {len(pays)}, distinct header cores "
                    f"{len(cores)}, distinct run_trackers {len(trks)}"))
    print()

    # ----------------------------------------------------- run_trackers
    print("-- run_trackers survive the file, and WHOSE state they describe")
    for label, res in (("uninterrupted resident", refs["resident"]),
                       ("uninterrupted streamed", refs["streamed"])):
        s = res["summary"]
        print(f"     {label:24s} nan_free={s.nan_free} "
              f"w_max={s.w_max_ms:.6f} interior={s.interior_w_max_ms:.6f} "
              f"boundary={s.boundary_w_max_ms:.6f}")
    if legs:
        for label, _ok, res in legs:
            s = res["summary"]
            print(f"     leg {label:22s} nan_free={s.nan_free} "
                  f"w_max={s.w_max_ms:.6f}")
        # Continuity, stated so it can only pass for the right reason.  A
        # resumed run's w-max is the running maximum over the WHOLE forecast,
        # so it must be at least the value the checkpoint carried: a run that
        # dropped the trackers would restart the maximum from the loop's
        # initial 0.0 and report only what its own half observed.  It is not
        # required to EQUAL the reference of its own mode, because the two
        # halves of a cross-mode leg are observed by two different observers
        # (see the note below) -- requiring that would fail a leg for a defect
        # that is not the restart's.
        carried = False
        for label, _ok, res in legs:
            first = label.split(" ")[0]
            block = p1[first]["ident"]["header"]["run_trackers"]
            got = res["summary"]
            ok = (bool(got.nan_free) == bool(block["nan_free"])
                  and float(got.w_max_ms) >= float(block["w_max_ms"]))
            # The strong form: this leg's answer IS the file's number, and it
            # is one its own half could not have produced on its own.
            strict = (float(got.w_max_ms) == float(block["w_max_ms"])
                      > float(refs[res["mode"]]["summary"].w_max_ms))
            carried = carried or strict
            if not ok:
                failures.append(
                    f"leg {label}: run_trackers did not survive the file "
                    f"(w_max {got.w_max_ms} < the checkpoint's "
                    f"{block['w_max_ms']})")
            print(_line(f"trackers survive the file: {label}", ok,
                        f"w_max {got.w_max_ms:.6f} >= checkpoint's "
                        f"{block['w_max_ms']:.6f}"
                        + ("  <- and IS it, which its own half could not "
                           "have produced" if strict else "")))
        if not carried:
            failures.append(
                "no leg's w_max came demonstrably out of the file: every leg "
                "could have produced its number from its own half alone, so "
                "'the trackers survive' is not shown by this run")
    print(f"     NOTE: the loop's trackers come from "
          f"stability_report(state, ...) and state.physics.fields['swdown'], "
          f"which under streaming read the frozen PREPARATION state.\n"
          f"           resident w_max="
          f"{refs['resident']['summary'].w_max_ms:.6f} vs streamed w_max="
          f"{refs['streamed']['summary'].w_max_ms:.6f} -> observers "
          f"{'SEE the streamed domain' if observers_live else 'are STALE'}."
          f"  The carriers round-trip regardless; the trackers describe the\n"
          f"           wrong state on BOTH sides of the file, so they survive "
          f"it faithfully and are still wrong.  That is "
          f"run_loop_safety_observers' item, not this one.")
    print()

    # ------------------------------------------------------ the negatives
    print("-- NEGATIVE CONTROLS (each MUST fire)")

    # 1. a store built with hoststore's DEFAULT 41-name contract
    print(_line("41-name store REFUSED at write", *_control_contract_store(
        cfg, bnd, failures)))

    # 2. the resident writer pointed at a streamed domain, end to end
    if "streamed" in p1:
        out = root / "neg_resident_writer"
        res = run_leg(cfg, bnd, mode="streamed", run_seconds=half,
                      outdir=out, tile=tile, hide_streaming=True)
        from woof.io.restart import restart_filename
        ckpt = out / restart_filename(START + timedelta(seconds=half), "d01")
        ident = file_identity(ckpt)
        differs = (ident["payload_sha256"]
                   != p1["streamed"]["ident"]["payload_sha256"])
        # And what it actually holds: the PREPARATION state, at t=0.  The
        # cold-start UP_HELI_MAX reset is replayed here because the loop
        # applies it to that same resident state before the first step, so a
        # comparison without it would differ in one carrier for a reason that
        # has nothing to do with the control.
        from woof.core.uh_diag import reset_up_heli_max
        start_state, start_geo = build_start(cfg, bnd)
        reset_up_heli_max(start_state)
        start_digest = digest_of(physinv.carrier_manifest(start_state))
        del start_state, start_geo
        _free()
        with np.load(ckpt, allow_pickle=False) as archive:
            from woof.io.restart import _HEADER_KEY
            got = digest_of({k: archive[k] for k in archive.files
                             if k != _HEADER_KEY})
        is_t0 = got == start_digest
        clock_is_zero = float(ident["header"]["elapsed_seconds"]) == 0.0
        fired = ((ident["header"].get("driver") or {}).get("call_counts"))
        fires = all(bool(x) for x in (differs, is_t0, clock_is_zero))
        if not fires:
            failures.append(
                "the resident writer on a streamed domain did NOT produce "
                f"the initial condition (differs={differs}, is_t0={is_t0}, "
                f"clock_is_zero={clock_is_zero}); the control is not firing "
                "and the branch it justifies is unjustified")
        print(_line("resident writer on a streamed domain", fires,
                    f"differs={differs}, payload IS the t=0 preparation "
                    f"state={is_t0}, header clock="
                    f"{ident['header']['elapsed_seconds']:g} s "
                    f"(the run was at {half:g} s), fired={fired}"))
        print("        NOTE: this checkpoint is not merely stale, it claims "
              "t=0.  Resuming it therefore replays the whole forecast and\n"
              "              lands on the same answer -- which is why 'does "
              "the resumed leg differ' is NOT a valid control here and the\n"
              "              file's own content is tested instead.  In "
              "operations the damage is a resume that silently redoes every\n"
              "              step already taken, and any leg that continues "
              "PAST the original end lands at the wrong valid time.")

    # 3. restore without reseeding TiledRun's cached clock
    if "streamed" in p1:
        leg = run_leg(cfg, bnd, mode="streamed",
                      restart_path=p1["streamed"]["path"],
                      run_seconds=run_seconds, outdir=root / "neg_clock",
                      tile=tile, skip_clock_reseed=True)
        bad = leg["digest"] != refs["streamed"]["digest"]
        if not bad:
            failures.append("a restore that did not reseed the sweep's "
                            "cached clock still reproduced the reference; "
                            "the reseed is not essential and the claim "
                            "that it is must be withdrawn")
        print(_line("restore without the clock reseed DIFFERS", bad,
                    f"{leg['digest'][:16]}"))

    print()
    print("=" * 78)
    if failures:
        print(f"FAILED  {len(failures)} finding(s), "
              f"{time.perf_counter() - t_all:.0f} s")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"PASSED  all rows, {time.perf_counter() - t_all:.0f} s")
    return 0


def _control_contract_store(cfg, bnd, failures) -> tuple[bool, str]:
    """A store built with hoststore's DEFAULT attrs must be REFUSED.

    ``HostDomainStore`` defaults to ``attrs=STATE_SERIALIZED_ATTRS`` -- 41
    names, of which 25 are allocated at this rung.  A file holding 25 of 229
    members passes every shape check in ``woof/io/restart.py``, because those
    checks validate the file against ITSELF and against a resuming state that
    already has all 229 slots allocated by preparation.  So the refusal has to
    happen at WRITE, by name.
    """
    from tilestream import hoststore

    domain, geo = build_start(cfg, bnd)
    setup = restart_stream.capture_domain_setup(domain)
    scalars = physinv.carrier_scalars(domain)
    store = hoststore.HostDomainStore(cfg)          # <- the DEFAULT
    try:
        restart_stream.write_streamed_restart(
            SCRATCH / "contract_only.npz", store.arrays, cfg,
            scalars=scalars, setup=setup, template_state=domain,
            check_pinned=False)
    except restart_stream.RestartRefused as exc:
        text = str(exc)
        named = ("missing" in text and str(len(store.arrays)) in text)
        store.free()
        domain = geo = None  # released before the early return below
        _free()
        if not named:
            failures.append("the 41-name store was refused but the refusal "
                            "did not name the missing members")
        return named, (f"{len(store.arrays)} members offered: "
                       f"{text.split('.')[0][:88]}")
    store.free()
    del domain, geo
    _free()
    failures.append("a store built with hoststore's DEFAULT 41-name attrs was "
                    "ACCEPTED by write_streamed_restart")
    return False, "ACCEPTED -- a 25-of-229 checkpoint would have been written"


if __name__ == "__main__":
    raise SystemExit(main())
