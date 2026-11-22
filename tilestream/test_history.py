"""HISTORY OUTPUT: a streamed forecast must write the SAME wrfout frames.

Run it::

    python -m tilestream.test_history         # from the repository root

:mod:`tilestream.test_join` proves the streamed TRAJECTORY equals the resident
one, carrier by carrier, in the store.  :mod:`tilestream.test_io` proves that
ONE frame taken off that store equals the device frame, byte for byte.  Neither
of them proves that a RUN writes the right files, because neither of them runs
the loop that writes them -- and that loop is where this feature was broken.

WHAT WAS BROKEN, AND WHY IT LOOKED FINE
---------------------------------------
Three history call sites exist in WOOF-ArWen and all three read the domain's
resident :class:`~woof.core.state.DomainState`::

    woof/runtime.py:2156,2263        write_case_output(prepared, ...)
                                      -> prepared.initial_result.state
    woof/runtime.py:2702             _submit_tree_history_frame -> node.state
    woof/prepared_single_domain_forecast.py:3999
                                      history_handler -> writers.submit(current)

A streamed domain's arrays do not live on that state.  ``streaming.attach``
copies the carriers into a pinned host store (``gather.pinned_copy``) and the
sweep writes the STORE; the resident state is never touched again.  So a
streamed run wrote the cold-start frame correctly and then wrote **the same
t = 0 field values into every subsequent frame**, with the correct Times
string, the correct global attributes and the correct variable inventory.  In
ncview that is a forecast in which nothing happens -- which is not obviously a
bug at 15 minute output over a quiet domain, and is not a bug at all if you
only ever open frame 1.  How much of a frame that freezes is MEASURED by
:func:`negative_frame_from_state` and printed per frame by the gate, rather
than asserted here.

The second defect is one level down and is the reason the first one could not
simply be papered over.  :meth:`tilestream.driver.TiledRun.sweep` documents
``step_kwargs`` as "forwarded verbatim to every tile's ``dycore.step`` ...
WOOF-ArWen's own loop passes ``refl_10cm_due`` through it, and it has to reach
EVERY tile of a sweep" -- and the sweep called ``step(tiles[b], tile_cfg)``
with the kwargs dropped on the floor.  ``refl_10cm_due=True`` therefore never
reached a tile, no tile ever ran WRF's ``calc_refl10cm``, the ``refl_10cm``
scratch slot was never even allocated, and the consumer on the domain state
raised ``REFL_10CM output is due but no microphysics-time field is stashed``.
A streamed forecast could not publish reflectivity at all.

Fixing the drop is one line and immediately exposes the third defect, which
is the one the feature brief predicted: ``stash_refl_10cm`` refuses to
overwrite an unconsumed handoff, and a tile BUFFER serves several tiles per
sweep, so the second tile a buffer serves raises ``REFL_10CM stash was not
consumed before reuse``.  The handoff is per TILE and the frame is per DOMAIN;
the two have to be joined by the same scatter that joins everything else.

The fourth is the same mistake as the first, one field further on, and it is
about what a history write RESETS rather than what it reads.
``woof.core.uh_diag.reset_up_heli_max`` zeroes the ``up_heli_max``
accumulator immediately after each frame is durable -- the WRF
history-interval reset, and a reset every publishing route already performs.
``up_heli_max`` is a SERIALIZED scratch slot, so it is a streaming CARRIER
and it is in the store; the reset was aimed at the resident state, hit
nothing, and left the running max growing across the whole run.  That is the
identical defect ``prepared_single_domain_forecast`` fixed once for the
resident path, with the note still in its ``history_handler``, re-introduced
by the mode and invisible because a monotone maximum is exactly what a
severe-weather diagnostic is supposed to look like.

WHAT IS UNDER TEST HERE
-----------------------
One configuration, integrated twice through
:func:`woof.core.streaming.make_stepper` -- once resident, once streamed --
inside the same history loop, writing through the same
:class:`woof.io.wrfout.AsyncDomainWrfoutWriter`, at the same cadence, to two
directories.  PASS is the strong form: **per-variable SHA-256 and whole-file
SHA-256 of every frame identical**, cold-start frame included.

Whole-file, not only per-variable, because the frame dict doubles as the
writer's schema and HDF5 lays its name heap out in variable-creation order:
:func:`tilestream.test_io.negative_field_order` measured a file 189 bytes
larger in which every variable compares equal.  A store-backed frame that
assembled the same numbers in a different order would pass a per-variable
comparison and hand a byte-different archive to everything downstream.

THE CONTROLS, AND WHAT EACH ONE CATCHES
----------------------------------------
``naive`` (frame from ``node.state``)
    The shipped behaviour, run on purpose.  MUST differ from frame 2 onward
    on every 3-D field except the base state (``PB``/``PHB``, which are
    input), and MUST agree on frame 1 -- both halves, because a control that
    differed at frame 1 too would be reporting a broken run rather than a
    frozen state.
``history-interval reset``
    Run the reset on the resident state only.  MUST make ``UP_HELI_MAX``
    differ from the second frame onward.
``refl dropped``
    Sweep with ``step_kwargs`` dropped again.  MUST raise the consumer's own
    "no microphysics-time field is stashed", not produce a zero field.
``refl from the last tile``
    Publish the tile buffer's own ``refl_10cm`` window instead of the
    scattered domain field -- the naive fix.  MUST differ, and it differs over
    3 of 4 tiles, which is what makes it a plausible-looking radar image.
``async, no drain``
    Submit frames to the background writer with ``overlap=False`` and let the
    solver sweep on WITHOUT waiting.  MUST differ: the carriers are zero-copy
    views of the pinned store and the sweep scatters into them mid-write.
    This is the one failure mode that is a race, so it is asserted as the
    PROPERTY that produces it (views change under a held frame) rather than
    timed.

CADENCE, PRINTED ON BOTH SIDES
-------------------------------
Three of this project's six false results were the same mistake: a number
measured in a window where radiation and cumulus never fired.  Every leg
reports ``state.physics.call_counts`` for radiation and cumulus and the
per-step ``refl_10cm_due`` schedule, and the two legs' counts are ASSERTED
equal.  At ``full fast cadence`` (radt 0.05 min, cudt 0.1 min, dt 30 s) both
fire on every step of the window; at ``mp10`` neither exists and the run is
reported as such rather than silently counted as coverage.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from woof.core import streaming
from tilestream import driver, gather, harness
from tilestream import output as tsout
from tilestream import checkpoint as tsck
from tilestream import physics_inventory as physinv
from tilestream import test_join as tj

# --------------------------------------------------------------------------
# the configuration
# --------------------------------------------------------------------------

#: 2x2 tiles that DIVIDE the domain.  A ragged trailing tile is a ring-mode
#: cost question (22% of the store read through instead of 2%), not a
#: correctness one, and it is measured in ``tilestream.rings``; here an exact
#: tiling keeps the comparison about the frames.
NX, NY, NZ = 192, 160, 49
TILE_NX, TILE_NY = 96, 80

#: 16 steps at ``history_every`` 4 gives the cold-start frame plus four
#: history frames -- the same five-frame shape as a 1 h forecast at 900 s
#: output, at a size two legs of it fit on a shared card.
NSTEPS = 16
HISTORY_EVERY = 4

#: The rungs.  Both carry a REFL_10CM-producing microphysics (mp_physics=10
#: is in ``runtime.REFL_10CM_MICROPHYSICS``) so the one-frame stash is
#: exercised on every history step; ``full fast cadence`` additionally fires
#: radiation and cumulus inside the window.
RUNGS = ("mp10", "full fast cadence")

START_TIME = datetime(2011, 4, 27, 12, 0, 0)

#: 3-D frame rows that are INPUT and therefore identical in every frame of a
#: correct run: the terrain-following base pressure and geopotential.  Named
#: so :func:`negative_frame_from_state` can say "every 3-D field except the
#: base state" and mean it, rather than weakening its assertion to "most".
STATIC_3D = frozenset({"PB", "PHB"})


def history_cfg(rung: str, nx: int = NX, ny: int = NY, nz: int = NZ):
    return tj.join_cfg(nx, ny, nz, rung)


def _free_device() -> None:
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _as_numpy(value):
    import cupy as cp

    return cp.asnumpy(value) if isinstance(value, cp.ndarray) \
        else np.asarray(value)


# --------------------------------------------------------------------------
# digests
# --------------------------------------------------------------------------

def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def variable_digests(path: Path) -> dict[str, str]:
    """SHA-256 per variable, so a difference is localised to a field."""
    import netCDF4

    out: dict[str, str] = {}
    with netCDF4.Dataset(path, "r") as ds:
        for name in sorted(ds.variables):
            data = np.ascontiguousarray(
                np.ma.getdata(ds.variables[name][:]))
            h = hashlib.sha256()
            h.update(name.encode())
            h.update(data.dtype.str.encode())
            h.update(np.asarray(data.shape, dtype=np.int64).tobytes())
            h.update(data.tobytes(order="C"))
            out[name] = h.hexdigest()
    return out


def frame_variable_shapes(path: Path) -> dict[str, tuple]:
    import netCDF4

    with netCDF4.Dataset(path, "r") as ds:
        return {name: tuple(int(s) for s in ds.variables[name].shape)
                for name in sorted(ds.variables)}


def compare_frames(ref_paths, got_paths) -> dict:
    """Per-variable and whole-file digests over a whole run's frames."""
    rows = []
    for index, (a, b) in enumerate(zip(ref_paths, got_paths)):
        da, db = variable_digests(a), variable_digests(b)
        differing = sorted(n for n in set(da) | set(db)
                           if da.get(n) != db.get(n))
        rows.append({
            "frame": index,
            "variables": len(da),
            "differing": differing,
            "ndiff": len(differing),
            "file_equal": file_digest(a) == file_digest(b),
            "sha256": file_digest(b)[:16],
        })
    return {
        "frames": len(rows),
        "rows": rows,
        "all_variables_equal": all(not r["differing"] for r in rows),
        "all_files_equal": all(r["file_equal"] for r in rows),
    }


def three_d(path: Path) -> tuple[str, ...]:
    """Variable names with a vertical dimension, in file order."""
    import netCDF4

    with netCDF4.Dataset(path, "r") as ds:
        return tuple(name for name in sorted(ds.variables)
                     if ds.variables[name].ndim >= 4)


# --------------------------------------------------------------------------
# the writer, shared by both legs
# --------------------------------------------------------------------------

def open_writer(cfg, outdir: Path, title="history-gate"):
    """WOOF-ArWen's own per-domain async writer, constructed identically for both
    legs so that a whole-file digest comparison is a comparison of the FRAME
    and of nothing else."""
    from woof.config import soil_layer_count
    from woof.io.wrfout import AsyncDomainWrfoutWriter

    outdir.mkdir(parents=True, exist_ok=True)
    return AsyncDomainWrfoutWriter(
        nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
        title=title, global_attrs={"GRID_ID": 1},
        soil_layers=soil_layer_count(cfg), grid_id=1)


def frame_path(outdir: Path, index: int, valid) -> Path:
    from woof.io.wrfout import wrfout_filename

    return outdir / wrfout_filename(valid, 1)


# --------------------------------------------------------------------------
# leg 1: RESIDENT -- the control that must work
# --------------------------------------------------------------------------

def resident_leg(cfg, outdir: Path, *, boundaries, nsteps=NSTEPS,
                 history_every=HISTORY_EVERY, seed=tj.SEED,
                 reset_uh: bool = False) -> dict:
    """One resident domain, WOOF-ArWen's ``dycore.step``, WOOF-ArWen's async writer.

    The loop is :func:`woof.runtime.integrate_prepared_case`'s, reduced to
    the parts a history frame depends on: the ``refl_10cm_due`` predicate on
    the step before an output, the frame, the submit, the drain.
    """
    from woof.core.dycore import step as dycore_step
    from woof.core.refl import consume_refl_10cm
    from woof.core.uh_diag import reset_up_heli_max

    state, _geo = tj.build_domain(cfg, seed=seed, boundaries=boundaries,
                                  warmup=1)
    stepper = streaming.make_stepper(state, cfg, streaming.OFF)
    assert stepper is dycore_step, (
        "the resident control must run the dycore's own step function, not "
        "a wrapper around it")

    paths, dues = [], []
    started = time.perf_counter()
    writer = open_writer(cfg, outdir)
    try:
        valid = START_TIME
        writer.submit(frame_path(outdir, 0, valid), valid, state)
        paths.append(frame_path(outdir, 0, valid))
        writer.drain()
        if reset_uh:
            reset_up_heli_max(state)
        for istep in range(nsteps):
            due = (istep + 1) % history_every == 0
            dues.append(due)
            stepper(state, cfg, refl_10cm_due=due)
            if due:
                valid = START_TIME + timedelta(
                    seconds=float(cfg.dt) * (istep + 1))
                refl = consume_refl_10cm(state)
                path = frame_path(outdir, len(paths), valid)
                writer.submit(path, valid, state, refl_field=refl)
                paths.append(path)
                writer.drain()
                if reset_uh:
                    reset_up_heli_max(state)
    finally:
        writer.close()
    counts = dict(state.physics.call_counts)
    record = {
        "leg": "resident", "frames": len(paths),
        "paths": paths, "due_steps": dues,
        "radiation_calls": int(counts.get("radiation", 0)),
        "cumulus_calls": int(counts.get("cumulus", 0)),
        "pbl_calls": int(counts.get("ysu", 0)) + int(counts.get("mynn", 0)),
        "microphysics_calls": int(state.physics.microphysics_updates),
        "elapsed_seconds": float(state.elapsed_seconds),
        "wall_s": time.perf_counter() - started,
    }
    del state
    _free_device()
    return record


# --------------------------------------------------------------------------
# leg 2: STREAMED
# --------------------------------------------------------------------------

def streamed_leg(cfg, outdir: Path, *, boundaries, geo_source,
                 nsteps=NSTEPS, history_every=HISTORY_EVERY, seed=tj.SEED,
                 source="store", tile_nx=TILE_NX, tile_ny=TILE_NY,
                 nbuffers=2, forward_kwargs=True, refl_from="store",
                 drain=True, reset_uh: bool = False) -> dict:
    """The same loop, with ``[tiles] mode = "on"``.

    ``source``
        ``"store"`` writes the frame out of the pinned host store (the fix);
        ``"state"`` writes it out of the resident ``DomainState`` exactly as
        every woof history call site does today (the negative control).
    ``forward_kwargs``
        ``False`` restores the sweep's dropped ``step_kwargs`` (the negative
        control for the one-line driver fix).
    ``refl_from``
        ``"store"`` publishes the scattered domain reflectivity; ``"tile"``
        publishes the last tile buffer's own window (the naive fix).
    ``drain``
        ``False`` submits to the background writer and sweeps on without
        waiting, which is the aliasing control.
    """
    domain, geo = tj.build_domain(cfg, seed=seed, boundaries=boundaries,
                                  warmup=1)
    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(tile_nx), tile_ny=int(tile_ny),
        nbuffers=int(nbuffers))
    decision = streaming.decide(cfg, options)

    build = _history_builder(cfg, domain, geo, boundaries=boundaries)
    stepper = streaming.make_stepper(domain, cfg, options, decision=decision,
                                     build=build)
    assert streaming.is_streaming(stepper)
    assert getattr(domain, "_streamed_domain", None) is stepper, (
        "attach must MARK the resident state it took over; the writers "
        "read that marker to decide where a frame comes from, and without "
        "it every history call site silently publishes t=0")
    run = stepper.tiled_run

    paths, dues = [], []
    started = time.perf_counter()
    writer = open_writer(cfg, outdir)
    try:
        valid = START_TIME
        path = frame_path(outdir, 0, valid)
        _submit(writer, path, valid, domain, stepper, source=source,
                refl=None)
        paths.append(path)
        if drain:
            writer.drain()
        if reset_uh:
            _reset_uh(domain)
        for istep in range(nsteps):
            due = (istep + 1) % history_every == 0
            dues.append(due)
            if forward_kwargs:
                stepper(domain, cfg, refl_10cm_due=due)
            else:
                # THE CONTROL: the sweep as it shipped, with step_kwargs
                # dropped.  Call the TiledRun directly so the drop is this
                # function's doing and not a patched module's.
                run.sweep(1, step_kwargs=None)
                stepper.steps += 1
            if due:
                valid = START_TIME + timedelta(
                    seconds=float(cfg.dt) * (istep + 1))
                refl = _streamed_refl(stepper, domain, cfg, refl_from)
                path = frame_path(outdir, len(paths), valid)
                _submit(writer, path, valid, domain, stepper, source=source,
                        refl=refl)
                paths.append(path)
                if drain:
                    writer.drain()
                if reset_uh:
                    _reset_uh(domain)
    finally:
        writer.close()
    counts = _sweep_call_counts(run)
    record = {
        "leg": f"streamed/{source}", "frames": len(paths),
        "paths": paths, "due_steps": dues,
        "tiles": len(run.specs), "nbuffers": run.nbuffers,
        "halo": run.halo,
        "radiation_calls": counts["call_counts"].get("radiation", 0),
        "cumulus_calls": counts["call_counts"].get("cumulus", 0),
        "pbl_calls": (counts["call_counts"].get("ysu", 0)
                      + counts["call_counts"].get("mynn", 0)),
        "microphysics_calls": counts["microphysics_updates"],
        "elapsed_seconds": float(stepper.elapsed_seconds),
        "decision": stepper.decision.explain(),
        "wall_s": time.perf_counter() - started,
    }
    del stepper, run, domain
    _free_device()
    return record


def _sweep_call_counts(run) -> dict:
    """A tile buffer's physics call counts.

    The scalars carried per tile make every buffer agree, so buffer 0's
    counts ARE the domain's -- asserted here rather than assumed, because a
    buffer that had drifted would be a clock-carrier bug and this is the
    cheapest place it would show.
    """
    counts = [physinv.carrier_scalars(t) for t in run.tiles]
    keys = ("call_counts", "microphysics_updates")
    first = {k: counts[0][k] for k in keys}
    for other in counts[1:]:
        trimmed = {k: other[k] for k in keys}
        if trimmed != first:
            raise AssertionError(
                "tile buffers disagree about how many times physics has "
                f"fired: {first} vs {trimmed}.  The clock is a carrier "
                "and this is what its loss looks like.")
    return first


def _reset_uh(state) -> None:
    """WOOF-ArWen's own history-interval reset, called exactly where it is called.

    ``reset_up_heli_max`` is what every publishing route runs immediately
    after a frame is durable; on a streamed domain it has to reach the
    STORE's ``scratch/up_heli_max``, not the resident state's dead copy.
    Called through the real function rather than reimplemented so the
    streamed leg cannot pass by doing something the model does not do.
    """
    from woof.core.uh_diag import reset_up_heli_max

    reset_up_heli_max(state)


def _submit(writer, path, valid, state, stepper, *, source, refl) -> None:
    """One history frame, through WOOF-ArWen's own async writer.

    ``source="store"`` is the production call: the frame comes from
    :meth:`woof.core.streaming.StreamedDomain.history_fields` and goes in
    through ``AsyncDomainWrfoutWriter.submit(frame=...)`` -- the same two
    objects ``PerDomainWrfoutWriters.submit`` uses for a streamed node, so
    what is gated here is the shipped path and not a stand-in for it.
    """
    if source == "state":
        writer.submit(path, valid, state, refl_field=refl)
        return
    writer.submit(path, valid, None, frame=stepper.history_fields(),
                  refl_field=refl)


def _streamed_refl(stepper, state, cfg, refl_from: str):
    """The domain's REFL_10CM for this frame, or ``None`` when not due.

    The ``"store"`` arm is deliberately ``consume_refl_10cm(state)`` and NOT
    a store lookup: that is the unmodified model function all three of
    WOOF-ArWen's history call sites already call, and the point of
    :meth:`StreamedDomain._stash_domain_refl` is that they keep working.  A
    store lookup here would test my own plumbing and leave the call sites
    untested.
    """
    import cupy as cp

    from woof.core.refl import consume_refl_10cm

    if cfg.mp_physics not in (1, 6, 8, 10, 18, 28):
        return None
    if refl_from == "tile":
        # THE CONTROL: the last tile buffer's own window, published for the
        # whole domain.  Right over its own quarter and wrong everywhere
        # else -- which still looks like a radar image.
        consume_refl_10cm(state)
        tile = stepper.tiled_run.tiles[-1]
        window = tile.existing_scratch("refl_10cm")
        if window is None:
            raise RuntimeError("no tile refl_10cm scratch to publish")
        host = cp.asnumpy(window)
        out = np.zeros((cfg.nz, cfg.ny, cfg.nx), dtype=host.dtype)
        out[:, :host.shape[1], :host.shape[2]] = host
        return out
    return consume_refl_10cm(state)


# --------------------------------------------------------------------------
# the streamed-domain builder
# --------------------------------------------------------------------------

def history_inventory(obj, names=None) -> dict:
    """``diagnostic_inventory`` plus the REFL_10CM producer.

    ``refl_10cm`` is a state-owned SCRATCH slot that the microphysics call
    writes when ``refl_10cm_due`` is set, and the driver holds only a
    one-frame reference to it.  Nothing reads it back into the trajectory --
    ``woof/io/restart.py`` classifies it REBUILT -- so it is not a carrier
    and the streamed run is bit-identical without it.  But it is horizontally
    decomposable like every other diagnostic (each tile's own microphysics
    call gets its own window right), so scattering it is what turns four tile
    windows into one domain frame.  8 B/cell at nz=49; the same trade
    :func:`tilestream.output.scatter_cost` prices for ``XKMH``/``XKHH``.
    """
    from collections.abc import Mapping

    inner = getattr(obj, "arrays", None)
    if isinstance(inner, dict):
        obj = inner
    source = dict(tsout.diagnostic_inventory(obj, None))
    if not isinstance(obj, Mapping):
        slot = getattr(obj, "existing_scratch", lambda _s: None)("refl_10cm")
        if slot is not None:
            source["scratch/refl_10cm"] = slot
    keys = sorted(source) if names is None else list(names)
    return {key: source[key] for key in keys if source.get(key) is not None}


def _history_builder(cfg, domain, geo, *, boundaries):
    """The route-owned construction :func:`streaming.make_stepper` needs.

    Two things beyond :func:`tilestream.test_join._make_builder`:

    * the inventory is :func:`history_inventory`, so the output-only driver
      diagnostics AND the reflectivity slot are gathered and scattered like
      carriers and the frame can be served entirely off the store;
    * the tile buffers take their warmup step with ``refl_10cm_due=True``, so
      the lazily-allocated ``refl_10cm`` scratch slot EXISTS before the
      inventory is taken.  Without it the buffer is missing an array the
      store holds and the inventory comparison refuses the run -- the same
      failure Kain-Fritsch's ``cumulus/w0avg`` produces on an unwarmed
      buffer, for the same reason.
    """
    from woof.ingest.lateral_bc import attach_lateral_boundaries
    from woof.core.refl import consume_refl_10cm, refl_10cm_is_stashed

    refl_capable = cfg.mp_physics in (1, 6, 8, 10, 18, 28)

    def build(state, run_cfg, decision):
        if refl_capable:
            # The DOMAIN's ``refl_10cm`` slot, primed so the store carries a
            # destination for the scatter.  A pure allocation:
            # ``DomainState.scratch`` is a named-slot dictionary of zeros,
            # nothing reads the slot before microphysics writes it, and
            # ``restart.py`` classifies it REBUILT -- so priming it cannot
            # move a value.  It has to happen HERE and not by warming the
            # domain with ``refl_10cm_due=True``, because that would run an
            # extra ``calc_refl10cm`` on one leg of a bit-exactness gate and
            # not the other.
            state.scratch((int(run_cfg.nz), int(run_cfg.ny), int(run_cfg.nx)),
                          "refl_10cm")
        geo_inv = {k: gather.pinned_copy(v) for k, v in
                   driver.geography_inventory(domain).items()}
        scalars = physinv.carrier_scalars(domain)
        per_tile = None
        if boundaries is not None:
            per_tile = streaming.tile_boundary_tables(
                boundaries, streaming.tile_specs(run_cfg, decision),
                seam="zeros")

        def factory(tile_cfg):
            tile, _drv = harness.make_physics_state(
                tile_cfg, 4242, geography=harness.neutral_geography(tile_cfg))
            if per_tile is not None:
                attach_lateral_boundaries(tile, per_tile[0])
            harness.run_steps(tile, tile_cfg, 1,
                              refl_10cm_due=refl_capable)
            if refl_capable and refl_10cm_is_stashed(tile):
                # The warmup exists to ALLOCATE the slot; its values are
                # overwritten on the first due step and the handoff must not
                # survive into the run, or the first real due step raises
                # "stash was not consumed before reuse".
                consume_refl_10cm(tile)
            return tile

        return streaming.attach(
            state, run_cfg, decision, tile_state_factory=factory,
            geography=geo_inv, boundary_tables=per_tile,
            inventory_fn=history_inventory, scalars=scalars)

    return build


# --------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------

# Absolute, not drive-relative: on Windows "/tmp" spells "\tmp", and the
# netCDF-C build in use refuses to OPEN a classic-format file through a
# drive-relative path ("NetCDF: Unknown file format") while opening the
# same bytes absolutely works.  The product writer pins its own paths
# (woof/io/wrfout.py), and this gate's readers must do the same.
WORK = Path(os.path.abspath("/tmp/tilestream-history"))


def case_frames(rung: str, *, nsteps=NSTEPS, history_every=HISTORY_EVERY,
                nx=NX, ny=NY, tile_nx=TILE_NX, tile_ny=TILE_NY,
                verbose: bool = True) -> dict:
    """THE GATE: five frames, resident against streamed, byte for byte."""
    cfg = history_cfg(rung, nx, ny)
    bnd = _boundaries(cfg)
    work = WORK / f"frames-{rung.replace(' ', '_').replace('+', 'p')}"
    shutil.rmtree(work, ignore_errors=True)

    res = resident_leg(cfg, work / "resident", boundaries=bnd,
                       nsteps=nsteps, history_every=history_every)
    got = streamed_leg(cfg, work / "streamed", boundaries=bnd,
                       geo_source=None, nsteps=nsteps,
                       tile_nx=tile_nx, tile_ny=tile_ny,
                       history_every=history_every, source="store")
    cmp = compare_frames(res["paths"], got["paths"])
    cadence_ok = (res["radiation_calls"] == got["radiation_calls"]
                  and res["cumulus_calls"] == got["cumulus_calls"]
                  and res["pbl_calls"] == got["pbl_calls"]
                  and res["microphysics_calls"] == got["microphysics_calls"]
                  # refl_10cm_due on the SAME steps on both sides.  The two
                  # loops compute it independently from the same predicate,
                  # so this is a check that the two loops really are the
                  # same loop and not a restatement of one of them.
                  and res["due_steps"] == got["due_steps"])
    clock_ok = abs(res["elapsed_seconds"] - got["elapsed_seconds"]) < 1e-9
    record = {
        "rung": rung, "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
        "dt": float(cfg.dt), "steps": nsteps,
        "history_every": history_every,
        "frames": cmp["frames"],
        "variables": cmp["rows"][0]["variables"],
        "three_d": len(three_d(res["paths"][0])),
        "tiles": got["tiles"], "halo": got["halo"],
        "bitexact": cmp["all_variables_equal"] and cmp["all_files_equal"],
        "all_variables_equal": cmp["all_variables_equal"],
        "all_files_equal": cmp["all_files_equal"],
        "differing": [(r["frame"], r["differing"][:6]) for r in cmp["rows"]
                      if r["differing"]],
        "resident_radiation": res["radiation_calls"],
        "streamed_radiation": got["radiation_calls"],
        "resident_cumulus": res["cumulus_calls"],
        "streamed_cumulus": got["cumulus_calls"],
        "resident_microphysics": res["microphysics_calls"],
        "streamed_microphysics": got["microphysics_calls"],
        "resident_pbl": res["pbl_calls"],
        "streamed_pbl": got["pbl_calls"],
        "refl_due_steps": [i + 1 for i, d in enumerate(res["due_steps"]) if d],
        "cadence_agrees": cadence_ok,
        "clock_agrees": clock_ok,
        "resident_wall_s": res["wall_s"],
        "streamed_wall_s": got["wall_s"],
        "sha256": [r["sha256"] for r in cmp["rows"]],
    }
    record["ok"] = bool(record["bitexact"] and cadence_ok and clock_ok)
    if verbose:
        shutil.rmtree(work, ignore_errors=True)
    return record


def negative_frame_from_state(rung: str = "mp10", *, nsteps=NSTEPS,
                              history_every=HISTORY_EVERY) -> dict:
    """THE CONTROL THAT MUST FIRE: write the frame from ``node.state``.

    This is not a hypothetical mistake, it is what every history call site in
    WOOF-ArWen does today.  Both halves are asserted: frame 1 must AGREE (the
    cold-start frame is taken before any sweep, so a disagreement there would
    mean the run itself was broken and the control would be measuring the
    wrong thing), and every frame from 2 onward must differ on every 3-D
    variable.
    """
    cfg = history_cfg(rung)
    bnd = _boundaries(cfg)
    work = WORK / "negative-state"
    shutil.rmtree(work, ignore_errors=True)

    res = resident_leg(cfg, work / "resident", boundaries=bnd, nsteps=nsteps,
                       history_every=history_every)
    naive = streamed_leg(cfg, work / "naive", boundaries=bnd, geo_source=None,
                         nsteps=nsteps, history_every=history_every,
                         source="state")
    cmp = compare_frames(res["paths"], naive["paths"])
    volume = set(three_d(res["paths"][0]))
    rows = []
    for row in cmp["rows"]:
        differing = set(row["differing"])
        rows.append({
            "frame": row["frame"],
            "ndiff": row["ndiff"],
            "three_d_differing": len(volume & differing),
            "three_d_total": len(volume),
        })
    frame0_agrees = not cmp["rows"][0]["differing"]
    # Every 3-D field EXCEPT the base state.  PB and PHB are the
    # terrain-following reference pressure and geopotential: input, not
    # state, identical in every frame of a correct run too, so requiring
    # them to differ would be requiring the control to be wrong in a way the
    # defect does not produce.  MEASURED at mp10: 17 of 19.
    later_all_3d_differ = all(
        r["three_d_differing"] == r["three_d_total"] - len(STATIC_3D)
        for r in rows[1:])
    shutil.rmtree(work, ignore_errors=True)
    return {
        "rung": rung, "rows": rows,
        "frame0_agrees": frame0_agrees,
        "later_all_3d_differ": later_all_3d_differ,
        "frozen_variables": rows[-1]["ndiff"],
        "variables": cmp["rows"][0]["variables"],
        "fires": bool(frame0_agrees and later_all_3d_differ),
    }


def negative_dropped_step_kwargs(rung: str = "mp10") -> dict:
    """THE CONTROL FOR THE ONE-LINE DRIVER FIX, in both its halves.

    Sweep with ``step_kwargs`` dropped, exactly as ``TiledRun._sweep``
    shipped for the whole life of this prototype.  Two things must then be
    true, and the SILENT one is the reason this control exists at all:

    * **silent** -- ``refl_10cm_due`` never reaches a tile, no tile calls
      ``calc_refl10cm``, and the domain's ``scratch/refl_10cm`` in the store
      is still EXACTLY ZERO after four due sweeps.  A frame written off that
      store is a complete, valid, correctly-timestamped wrfout with a radar
      field of 0 dBZ everywhere, and nothing anywhere reports a problem;
    * **loud** -- the model's own ``consume_refl_10cm`` then raises, at the
      history call site, which is the only symptom the defect ever produced
      and which points at the consumer rather than at the sweep.

    Asserting only the loud half would let a future "fix" that quietly
    supplies a stale or zero field pass.  MEASURED at mp10: the tile buffers
    DO own a ``refl_10cm`` slot (2 of 2 -- their warm-up step allocated it),
    the scatter faithfully assembles a domain field out of it every sweep,
    and that field is exactly zero everywhere (absmax 0.0) against a correct
    run that differs from it in 1,470,980 of 1,505,280 cells.  Zero dBZ over
    the whole domain in an otherwise complete, correctly-timestamped wrfout
    is what "no radar echo" looks like, which is why the loud half is the
    only reason this defect was ever noticed.
    """
    cfg = history_cfg(rung)
    bnd = _boundaries(cfg)
    outcome: dict = {"rung": rung}
    fields = {}
    for label, forward in (("forwarded", True), ("dropped", False)):
        domain, geo = tj.build_domain(cfg, seed=tj.SEED, boundaries=bnd,
                                      warmup=1)
        options = streaming.StreamingOptions(mode="on", tile_nx=TILE_NX,
                                             tile_ny=TILE_NY, nbuffers=2)
        decision = streaming.decide(cfg, options)
        build = _history_builder(cfg, domain, geo, boundaries=bnd)
        stepper = streaming.make_stepper(domain, cfg, options,
                                         decision=decision, build=build)
        run = stepper.tiled_run
        for istep in range(HISTORY_EVERY):
            due = (istep + 1) == HISTORY_EVERY
            if forward:
                stepper(domain, cfg, refl_10cm_due=due)
            else:
                run.sweep(1, step_kwargs=None)     # the shipped drop
        fields[label] = np.array(run.store[streaming.REFL_STORE_KEY],
                                 copy=True)
        if not forward:
            outcome["tile_buffers_with_a_refl_slot"] = sum(
                t.existing_scratch("refl_10cm") is not None
                for t in run.tiles)
            try:
                from woof.core.refl import consume_refl_10cm

                consume_refl_10cm(domain)
                outcome["raised"] = None
            except BaseException as exc:           # noqa: BLE001
                outcome["raised"] = f"{type(exc).__name__}: {exc}"
        del stepper, run, domain
        _free_device()
    differ = not np.array_equal(fields["forwarded"], fields["dropped"])
    outcome.update(
        store_refl_differs=differ,
        cells_differing=int(np.count_nonzero(
            fields["forwarded"] != fields["dropped"])),
        cells=int(fields["dropped"].size),
        dropped_is_finite=bool(np.isfinite(fields["dropped"]).all()),
        dropped_absmax=float(np.abs(fields["dropped"]).max()),
        fires=bool(differ and outcome.get("raised") is not None))
    return outcome


def negative_refl_from_tile(rung: str = "mp10") -> dict:
    """THE CONTROL FOR THE REFL SCATTER: publish the last tile's window.

    The naive fix -- the buffer has a ``refl_10cm`` slot, so publish it.  It
    is right over that buffer's own tile and wrong over the rest of the
    domain, which is a plausible-looking radar image and the reason this
    needs a control rather than an inspection.
    """
    cfg = history_cfg(rung)
    bnd = _boundaries(cfg)
    work = WORK / "negative-refl-tile"
    shutil.rmtree(work, ignore_errors=True)
    res = resident_leg(cfg, work / "resident", boundaries=bnd,
                       nsteps=HISTORY_EVERY, history_every=HISTORY_EVERY)
    tile = streamed_leg(cfg, work / "tile", boundaries=bnd, geo_source=None,
                        nsteps=HISTORY_EVERY, history_every=HISTORY_EVERY,
                        source="store", refl_from="tile")
    cmp = compare_frames(res["paths"], tile["paths"])
    refl_differs = any("REFL_10CM" in r["differing"] for r in cmp["rows"][1:])
    others = [n for r in cmp["rows"] for n in r["differing"]
              if n != "REFL_10CM"]
    shutil.rmtree(work, ignore_errors=True)
    return {
        "rung": rung,
        "refl_differs": refl_differs,
        "other_differing": sorted(set(others))[:6],
        "fires": bool(refl_differs and not others),
    }


def case_history_interval_reset(rung: str = "mp10", *, nsteps=12,
                                history_every=4) -> dict:
    """THE OTHER HALF OF A HISTORY FRAME: what the write RESETS.

    ``UP_HELI_MAX`` is WRF's ``nwp_diagnostics`` running max and woof's
    ratified placement zeroes it immediately after each frame is durable, so
    frame *k* reports the maximum over history interval *k* and not over the
    whole run.  It is a SERIALIZED scratch slot, which makes it a streaming
    CARRIER: it lives in the store, is gathered into each tile and scattered
    back every sweep, and the resident state's copy is dead memory.  A reset
    aimed at the state therefore zeroes nothing, and every frame from the
    second onward reports the max since MODEL START -- monotone, plausible,
    and wrong in the direction that never comes back down.

    Both legs run the full contract -- write the frame, then reset -- and
    all frames must be byte-identical.  The negative half turns the reset
    off on the streamed leg only and requires UP_HELI_MAX to differ, which
    is what a reset that missed the store looks like.
    """
    cfg = history_cfg(rung)
    import dataclasses

    cfg = dataclasses.replace(cfg, nwp_diagnostics=1)
    bnd = _boundaries(cfg)
    work = WORK / "uh-reset"
    shutil.rmtree(work, ignore_errors=True)
    res = resident_leg(cfg, work / "resident", boundaries=bnd, nsteps=nsteps,
                       history_every=history_every, reset_uh=True)
    good = streamed_leg(cfg, work / "streamed", boundaries=bnd,
                        geo_source=None, nsteps=nsteps,
                        history_every=history_every, source="store",
                        reset_uh=True)
    bad = streamed_leg(cfg, work / "no-reset", boundaries=bnd,
                       geo_source=None, nsteps=nsteps,
                       history_every=history_every, source="store",
                       reset_uh=False)
    ok = compare_frames(res["paths"], good["paths"])
    control = compare_frames(res["paths"], bad["paths"])
    uh_present = "UP_HELI_MAX" in variable_digests(res["paths"][0])
    control_uh = [r["frame"] for r in control["rows"]
                  if "UP_HELI_MAX" in r["differing"]]
    shutil.rmtree(work, ignore_errors=True)
    return {
        "rung": rung, "frames": ok["frames"],
        "UP_HELI_MAX_in_frame": uh_present,
        "all_variables_equal": ok["all_variables_equal"],
        "all_files_equal": ok["all_files_equal"],
        "differing": [(r["frame"], r["differing"][:6]) for r in ok["rows"]
                      if r["differing"]],
        "control_frames_with_UP_HELI_MAX_differing": control_uh,
        "ok": bool(uh_present and ok["all_variables_equal"]
                   and ok["all_files_equal"] and control_uh),
    }


class _StubNode:
    """The three attributes ``PerDomainWrfoutWriters.submit`` reads."""

    def __init__(self, state, cfg, grid_id=1, tick_den=1):
        self.state = state
        self.cfg = _StubDomainCfg(cfg, grid_id)
        self.clock = _StubClock(tick_den)


class _StubDomainCfg:
    def __init__(self, run, grid_id):
        self.run = run
        self.grid_id = int(grid_id)


class _StubClock:
    def __init__(self, tick_den):
        self.tick_den = int(tick_den)


def case_production_dispatch(rung: str = "mp10", *, nsteps=HISTORY_EVERY
                             ) -> dict:
    """THE PRODUCTION CALL SITE, not a lookalike of it.

    ``PerDomainWrfoutWriters.submit(node, ticks)`` is what ``woof go``'s
    ``history_handler`` and the tree runner's
    ``runtime._submit_tree_history_frame`` both call, and it is the function
    that used to read ``node.state`` unconditionally.  This drives THAT
    method -- through a writers object stood up field by field, which is the
    same ``object.__new__`` shell ``tests/test_wrfout.py`` already uses --
    once for a resident node and once for a streamed one, and requires the
    two files to be byte-identical.

    So what is gated is the dispatch: that the marker
    ``streaming.attach`` leaves on the state routes the frame to the store,
    that the writer's ``frame=`` path publishes it, and that the drain
    happens before the caller can sweep again.
    """
    from woof.io.wrfout import PerDomainWrfoutWriters

    cfg = history_cfg(rung)
    bnd = _boundaries(cfg)
    work = WORK / "production-dispatch"
    shutil.rmtree(work, ignore_errors=True)
    paths = {}
    for leg in ("resident", "streamed"):
        outdir = work / leg
        outdir.mkdir(parents=True, exist_ok=True)
        writers = object.__new__(PerDomainWrfoutWriters)
        writers.output_dir = outdir
        writers.start_time = START_TIME
        writers.last_durable_wrfout = None
        writers._metadata_by_grid_id = {1: {}}
        writers._writers = {1: open_writer(cfg, outdir)}
        try:
            if leg == "resident":
                state, _geo = tj.build_domain(cfg, seed=tj.SEED,
                                              boundaries=bnd, warmup=1)
                stepper = streaming.make_stepper(state, cfg, streaming.OFF)
                domain = state
            else:
                domain, geo = tj.build_domain(cfg, seed=tj.SEED,
                                              boundaries=bnd, warmup=1)
                options = streaming.StreamingOptions(
                    mode="on", tile_nx=TILE_NX, tile_ny=TILE_NY, nbuffers=2)
                decision = streaming.decide(cfg, options)
                stepper = streaming.make_stepper(
                    domain, cfg, options, decision=decision,
                    build=_history_builder(cfg, domain, geo, boundaries=bnd))
            node = _StubNode(domain, cfg)
            written = []
            # The cold-start frame, then one history frame, exactly as
            # `integrate_prepared_case` orders them.
            writers.submit(node, 0)
            written.append(outdir / _filename(START_TIME))
            for istep in range(nsteps):
                due = (istep + 1) == nsteps
                stepper(domain, cfg, refl_10cm_due=due)
            from woof.core.refl import consume_refl_10cm
            seconds = float(cfg.dt) * nsteps
            writers.submit(node, int(seconds),
                           refl_field=consume_refl_10cm(domain))
            written.append(outdir / _filename(
                START_TIME + timedelta(seconds=seconds)))
            paths[leg] = written
        finally:
            writers._writers[1].close()
            del stepper, domain
            _free_device()
    cmp = compare_frames(paths["resident"], paths["streamed"])
    record = {
        "rung": rung,
        "frames": cmp["frames"],
        "all_variables_equal": cmp["all_variables_equal"],
        "all_files_equal": cmp["all_files_equal"],
        "differing": [(r["frame"], r["differing"][:6]) for r in cmp["rows"]
                      if r["differing"]],
        "sha256": [r["sha256"] for r in cmp["rows"]],
    }
    record["ok"] = bool(cmp["all_variables_equal"] and cmp["all_files_equal"])
    shutil.rmtree(work, ignore_errors=True)
    return record


def _filename(valid) -> str:
    from woof.io.wrfout import wrfout_filename

    return wrfout_filename(valid, 1)


def negative_async_no_drain(rung: str = "mp10", *, nsteps=8) -> dict:
    """THE ALIASING CONTROL, asserted as a property and not as a race.

    ``StoreFrame`` with ``overlap=False`` hands out ZERO-COPY VIEWS of the
    pinned store -- that is what makes an out-of-core frame 2.2x-2.4x cheaper
    than the resident one -- so a caller that submits a frame and sweeps on
    without draining hands the writer thread a buffer the scatter is writing
    into.  Timing that against a background writer gives a control that
    passes by luck on a fast disk.  This one takes the frame, sweeps once,
    and asks whether the arrays the frame handed out CHANGED, which is the
    property that produces the corruption.
    """
    cfg = history_cfg(rung)
    bnd = _boundaries(cfg)
    domain, geo = tj.build_domain(cfg, seed=tj.SEED, boundaries=bnd, warmup=1)
    options = streaming.StreamingOptions(mode="on", tile_nx=TILE_NX,
                                         tile_ny=TILE_NY, nbuffers=2)
    decision = streaming.decide(cfg, options)
    setup = tsck.DomainSetup.capture(domain, cfg)
    build = _history_builder(cfg, domain, geo, boundaries=bnd)
    stepper = streaming.make_stepper(domain, cfg, options, decision=decision,
                                     build=build)
    plan = tsout.frame_plan(domain, extra_available=stepper.store.keys())

    outcomes = {}
    for overlap in (False, True):
        frame = tsout.StoreFrame(plan, stepper.store, setup, cfg,
                                 overlap=overlap)
        held = frame.fields()
        before = {n: np.array(v, copy=True) for n, v in held.items()}
        stepper(domain, cfg, refl_10cm_due=False)
        moved = sorted(n for n, v in held.items()
                       if not np.array_equal(np.asarray(v), before[n]))
        outcomes["overlap" if overlap else "views"] = {
            "moved": len(moved), "total": len(held),
            "examples": moved[:4],
            "snapshot_MB": frame.snapshot_bytes / 1e6,
        }
    del stepper, domain
    _free_device()
    return {
        "rung": rung, "outcomes": outcomes,
        "fires": bool(outcomes["views"]["moved"] > 0
                      and outcomes["overlap"]["moved"] == 0),
    }


def _boundaries(cfg):
    """The domain's specified forcing, from two genuinely different draws."""
    import cupy as cp

    a, _ = tj.build_domain(cfg, seed=tj.SEED, warmup=0)
    b, _ = tj.build_domain(cfg, seed=tj.SEED + 1, warmup=0)
    bnd = tj.domain_boundaries(cfg, a, b)
    del a, b
    cp.get_default_memory_pool().free_all_blocks()
    return bnd


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _ok(flag: bool) -> str:
    return "  PASS" if flag else "  FAIL"


def _opt(argv, name, default):
    return type(default)(argv[argv.index(name) + 1]) if name in argv \
        else default


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    nx = _opt(argv, "--nx", NX)
    ny = _opt(argv, "--ny", NY)
    tile_nx = _opt(argv, "--tile-nx", TILE_NX if nx == NX else nx // 3)
    tile_ny = _opt(argv, "--tile-ny", TILE_NY if ny == NY else ny // 3)
    nsteps = _opt(argv, "--steps", NSTEPS)
    every = _opt(argv, "--every", HISTORY_EVERY)
    only = "--frames-only" in argv
    failures = 0

    print("=" * 74)
    print("HISTORY OUTPUT GATE -- wrfout frames from a streamed domain")
    print(f"{nx}x{ny}x{NZ}, tile {tile_nx}x{tile_ny}, "
          f"{nsteps} steps, history every {every}")
    print("=" * 74)

    rungs = RUNGS[:1] if quick else RUNGS
    for rung in rungs:
        print(f"\n--- {rung} ---")
        record = case_frames(rung, nsteps=nsteps, history_every=every,
                             nx=nx, ny=ny, tile_nx=tile_nx, tile_ny=tile_ny)
        print(f"  {record['frames']} frames, {record['variables']} variables "
              f"({record['three_d']} 3-D), {record['tiles']} tiles, "
              f"halo {record['halo']}")
        print(f"  physics fired inside the window: radiation "
              f"{record['resident_radiation']}/{record['streamed_radiation']}"
              f"  cumulus "
              f"{record['resident_cumulus']}/{record['streamed_cumulus']}"
              f"  PBL {record['resident_pbl']}/{record['streamed_pbl']}"
              f"  microphysics "
              f"{record['resident_microphysics']}"
              f"/{record['streamed_microphysics']}   (resident/streamed)")
        print(f"  refl_10cm_due on steps {record['refl_due_steps']}")
        print(f"  wall: resident {record['resident_wall_s']:.1f} s, "
              f"streamed {record['streamed_wall_s']:.1f} s "
              "(NOT a speed result -- see the module docstring; the "
              "compute window here is far below the 500-cell floor)")
        print(f"  frame sha256 {record['sha256']}")
        if not record["ok"]:
            print(f"  differing: {record['differing'][:3]}")
        print(_ok(record["ok"]) + "  per-variable AND whole-file identical "
              f"on all {record['frames']} frames"
              if record["ok"] else
              _ok(False) + "  frames DIFFER")
        failures += not record["ok"]

    if only:
        print("\n(--frames-only: negative controls skipped)")
        return 1 if failures else 0

    print("\n--- the history-interval reset (UP_HELI_MAX) ---")
    record = case_history_interval_reset()
    for key, value in record.items():
        if key in ("ok", "rung"):
            continue
        print(f"      {key}: {value}")
    print(_ok(record["ok"]) + "  the reset reaches the store, and a reset "
          "that misses it is caught")
    failures += not record["ok"]

    print("\n--- the production dispatch "
          "(PerDomainWrfoutWriters.submit) ---")
    record = case_production_dispatch()
    print(f"      frames: {record['frames']}  sha256 {record['sha256']}")
    print(f"      variables equal: {record['all_variables_equal']}  "
          f"files equal: {record['all_files_equal']}")
    if record["differing"]:
        print(f"      differing: {record['differing']}")
    print(_ok(record["ok"]) + "  the shipped history call site publishes a "
          "streamed domain identically")
    failures += not record["ok"]

    print("\n--- negative controls (each MUST fire) ---")
    for label, fn in (
            ("frame written from node.state", negative_frame_from_state),
            ("sweep drops step_kwargs", negative_dropped_step_kwargs),
            ("REFL_10CM from the last tile", negative_refl_from_tile),
            ("frame held across a sweep", negative_async_no_drain)):
        record = fn()
        print(f"  {label}")
        for key, value in record.items():
            if key in ("fires", "rung"):
                continue
            print(f"      {key}: {value}")
        print(_ok(record["fires"]) + "  control fired"
              if record["fires"] else _ok(False) + "  CONTROL DID NOT FIRE")
        failures += not record["fires"]

    print("\n" + "=" * 74)
    print("HISTORY GATE: " + ("PASS" if not failures
                              else f"FAIL ({failures})"))
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
