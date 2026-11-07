"""CUDA graph capture of the tiled step: pay the launch cost once, not per tile.

WOOF's step issues ~1,203 kernel launches and the profile in PERF-SURVEY.md
measures a 26% dead gap between them -- 6.34 us of idle per launch against
24.1 us of device work.  That gap is host submission cost, and a tiled sweep
pays it ONCE PER TILE: the same 1,203 launches are re-issued for every tile in
the plan, so an 8x8 tiling submits 77,000 launches to do one domain-step's
worth of arithmetic.  It is worst exactly where the streaming lane hurts most,
on the small compute windows a 12 GB card is forced into, because a small
window shrinks the device work per launch without shrinking the launch.

A CUDA graph replaces the whole submission sequence with one
``cuGraphLaunch``.  This module captures a tile's step once and replays it for
every other tile.  The per-tile sequence is IDENTICAL -- same config, same
buffer, same arrays, only the gathered *contents* differ -- which is what
makes the tiled loop a better graph candidate than the resident loop it was
proposed for.

Nothing here changes arithmetic.  A replayed graph re-executes the recorded
kernels on the recorded pointers; the gate's job is to prove that claim rather
than accept it, and ``tilestream.test_gate`` runs every rung through this path
with ``--graph``.


THE FOUR WAYS THIS GOES SILENTLY WRONG
--------------------------------------
Each one is a real failure that was hit, reproduced, or defended against here,
and each has a mechanism in this module that answers it.  A graph that
replays stale work does not crash: it produces a plausible forecast.

1. **An empty capture.**  ``Stream.begin_capture()`` does NOT redirect CuPy's
   launches on its own.  The stream must also be made CURRENT (``with s:``),
   and PERF-FINDINGS records a sweep that reported a fake 99.9% speedup
   because capture without it produced an empty graph that launched, did
   nothing, and raised nothing.  :func:`capture_step` therefore refuses a
   graph whose node count is below :data:`MIN_PLAUSIBLE_NODES`, and records
   the node count in every report.

2. **A stale step.**  Two ways, and the second is the one that surprises.
   Radiation, the surface/PBL bundle and cumulus fire on a CADENCE, so
   replaying a no-radiation graph on a radiation-due step silently drops the
   heating; :func:`cadence_key` asks the driver's own predicates about that.
   And a graph bakes in every kernel's scalar ARGUMENTS, several of which
   are the absolute model time -- RRTMGP's solar hour angle above all -- so
   a graph is not re-usable across steps even when the cadence agrees.  The
   default therefore keys on the SWEEP as well, which is exactly the reuse a
   tiled run needs and no more: one capture per step, replayed once per
   tile.  ``graph_reuse="run"`` lifts that and the gate runs it as a
   negative control, where it is caught by 33 differing carriers at maxabs
   2.0e+04.  MEASURED, not assumed -- and note the consequence for the
   cadence key: under sweep keying the sweep index already separates every
   step, so the cadence half of the key only becomes essential under
   ``reuse="run"``.

3. **Stale HOST state.**  A replay executes device work only.  Every Python
   side effect a step has -- the clock, the call counters, an array rebound
   on the driver -- happens at CAPTURE time and never again.  If any of that
   feeds the next step, replaying is wrong.  This module does not assume it
   does not: :func:`capture_step` captures the same step several times from
   the same clock and requires the array SHAPES, the scalar CARRIERS and
   object identity (as a set) to settle.  What settles is replayable; what
   does not is refused, with the attribute named.  The scalar carriers are
   then re-applied after every replay as the increment the capture measured
   (:attr:`StepGraph.scalars_delta`), because the tiled driver's sweep
   bookkeeping reads them.  Diagnostic counters that move every step and
   feed nothing -- RRTMGP's ``update_count``, Kain-Fritsch's cached history
   -- are reported as drift and allowed: the tiled driver ALREADY
   desynchronises exactly those from the domain, a buffer serving k tiles
   advancing them k times a sweep, and the gate has proved rung by rung that
   the forecast does not depend on them.

4. **Pointer reuse across concurrent graphs.**  A graph bakes in the
   addresses of the temporaries the step allocated while it was being
   captured.  CuPy hands those addresses back to its pool at the end of the
   capture, so the NEXT capture -- buffer 1's, say -- gets the same
   addresses, and then buffer 0 and buffer 1 replay concurrently on two
   streams into ONE set of scratch buffers.  The answer is wrong and nothing
   reports it.  Every :class:`GraphStepper` -- one per tile buffer --
   therefore owns a PRIVATE ``cupy.cuda.MemoryPool`` that its captures run
   under and that outlives them; the blocks stay reserved to that pool, so
   no other allocation can be handed them.  Never call ``free_all_blocks``
   on it: that returns the addresses to the driver while the graph still
   points at them, which is the same corruption with an extra step.


WHAT IS NOT CAPTURABLE, AND WHY THAT IS A LIST AND NOT A SHRUG
--------------------------------------------------------------
Capture fails, loudly, on any host/device transfer or synchronisation, which
is the same thing as saying it fails on any host branch over device data.
Measured over the gate's fourteen physics rungs at 64x64x49 (each rung in its
own process, first failure only):

===========================  ==========================================
rung                         first refusal
===========================  ==========================================
dry                          none -- captures and replays
mp10 Morrison .. +sfclay     ``microphysics.py`` status readback
+YSU PBL                     ``ysu.py`` status readback
+Noah LSM .. full+NSSL       ``noah.py`` per-step upload of a CONSTANT table
full fast cadence            ``rrtmgp.py`` status readback, then two more
                             constant uploads, then ``physics.py``'s
                             per-output finite check
full+Noah-MP                 ``noahmp_runtime.py`` per-step latitude upload
full+MYNN                    ``mynn_pbl_gpu.py`` per-chunk guard readbacks
===========================  ==========================================

:mod:`woof.core.health_ledger` answers the status readbacks (they are
error-only: on a healthy run nothing downstream reads them, so they can be
accumulated on the device and drained at a synchronisation the caller already
pays for).  The constant uploads are memoised at their sites.  With those
done, ELEVEN of the fourteen rungs capture, including the ship config
``full(real74) +KF`` at 876 nodes and the radiation-every-step lane at 1,414
to 1,471 nodes.  Three do not:

``full+MYNN`` (and ``full+MYNN+Noah-MP``)
    ``mynn_pbl_gpu._flag_mask`` reads a block of validation words back to the
    host once per guard group per COLUMN CHUNK -- so a domain walked in
    chunks pays a multiple of them per step.  Every one of them is
    error-only, exactly like the four sites the ledger already covers, so
    this is the same mechanical change again and nothing more.

``full+Noah-MP``
    ``noahmp_runtime._lsm_step_slab`` uploads the latitude grid and the soil
    coordinate every step (cacheable), and then does something that is not
    just an upload: ``cp.nonzero`` over the land mask, ``cp.asnumpy`` of the
    land columns' vegetation and soil identity, and ``np.unique`` on the host
    to build the parameter rows -- a data-dependent gather whose size the
    host has to know, EVERY STEP.  The inputs are static geography, so the
    index and the row table are cacheable across the whole run, and that is
    worth doing for its own sake.  It is a Noah-MP change and not a graph
    change.

``use_graph=True`` falls back to stream launching for those, reports the
reason, and stays bit-exact; the gate runs ``full+MYNN+Noah-MP`` as a
positive case precisely to prove that the fallback is exact and that it
really is falling back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable
import weakref

import numpy as np


#: A captured step with fewer nodes than this is not a step.  The dry lane at
#: the gate's smallest window captures 559 nodes; full physics captures
#: thousands.  The bound exists for failure 1 above -- an empty or truncated
#: capture -- and is deliberately far below any real step so it never fires on
#: a legitimately small configuration.
MIN_PLAUSIBLE_NODES = 32


class GraphCaptureError(RuntimeError):
    """A capture that must not be silently downgraded to stream launching."""


# ---------------------------------------------------------------------------
# The topology key
# ---------------------------------------------------------------------------

def cadence_key(state, cfg) -> tuple:
    """The host-visible facts that decide WHICH kernels a step launches.

    Two steps with the same key issue the same sequence; two steps with
    different keys may not, and a graph captured under one key must never be
    replayed under the other.  Every entry is a pure function of the domain
    clock and the config -- which is only true because the blocking device
    reads have been removed from the step path.  A host branch over device
    data would make the topology depend on the VALUES in the arrays, and no
    key computable here could see it.  That is the deeper reason
    :mod:`woof.core.health_ledger` exists; the launch saving is the smaller
    half of it.

    The answers come from the physics driver's own predicates rather than
    from a re-derivation of the cadence rules, because a re-derivation is a
    second copy of a rule that can drift from the first.  What this cannot
    check is whether the list is COMPLETE.  That is what
    ``GraphStepper(verify_topology=True)`` is for: it recaptures and compares
    the graph's structural fingerprint, so a step whose topology moved for a
    reason not in this tuple is caught rather than replayed.
    """
    import math

    from woof.core.physics import (
        _cumulus_step_due, _model_clock_dt, _physics_interval_steps,
        _radiation_step_due, _surface_pbl_step_due, physics_enabled,
        radiation_enabled)

    cold = (("wrf-mass-cold",)
            if getattr(state, "_wrf_mass_carriers_cold", False) else ())
    if not physics_enabled(cfg) or getattr(state, "physics", None) is None:
        return ("dry",) + cold

    drv = state.physics
    # dycore.step -> PhysicsDriver.compute, physics.py:3399-3400.  The
    # rounding matters: elapsed_seconds accumulates in float and the +0.5
    # floor is what makes step k's itimestep exactly k+1.
    epoch = float(getattr(state, "domain_start_offset", 0.0) or 0.0)
    itimestep = int(np.floor((float(state.elapsed_seconds) - epoch) / cfg.dt + 0.5)) + 1
    stepra = _physics_interval_steps(drv.radt_minutes, cfg.dt)
    override = getattr(drv, "radiation_due_override", None)
    radiation = bool(radiation_enabled(cfg) and (
        _radiation_step_due(itimestep, stepra, drv.radt_minutes)
        if override is None else override))
    surface = bool(drv.surface_enabled and _surface_pbl_step_due(
        itimestep, drv.stepbl, cfg.bldt,
        getattr(drv, "surface_pbl_due_override", None)))
    override = getattr(drv, "cumulus_due_override", None)
    stepcu = _physics_interval_steps(drv.cudt_minutes, cfg.dt)
    cumulus = bool(cfg.cu_physics and (
        _cumulus_step_due(itimestep, stepcu, drv.cudt_minutes)
        if override is None else override))
    # physics.py:2103-2107, ``_advance_cumulus_clock``: WRF's advance_ppt
    # runs once per MODEL CLOCK step, which is not once per dycore step when
    # cfg.clock_dt > cfg.dt (the real74 compatibility integrator).  It is a
    # host branch, so it belongs here; it was found by reading, and the
    # ``verify_topology`` mode is what would have found it otherwise.
    cu_clock = False
    if cfg.cu_physics:
        clock_dt = _model_clock_dt(cfg)
        end = float(state.elapsed_seconds) + float(cfg.dt)
        remainder = abs(math.fmod(end, clock_dt))
        cu_clock = min(remainder, clock_dt - remainder) <= 1.0e-6 * clock_dt
    return ("physics", itimestep == 1, radiation, surface, cumulus, cu_clock) + cold


# ---------------------------------------------------------------------------
# The host fingerprint
# ---------------------------------------------------------------------------

_SCALARISH = (bool, int, float, str, bytes, type(None))


class ObjectLifetimes:
    """Names an object by its LIFETIME rather than by its address.

    ``id()`` is an address, and CPython hands a freed address to the next
    object of the same size.  :func:`capture_step` asks whether an object a
    step rebinds comes BACK on a later capture; with bare ids, a fresh
    tendency bundle allocated where the previous capture's already freed
    bundle lived reads as the same object, and a correct rebinding is
    refused as a double buffer.  MEASURED on the graph section: the ship
    config and the real Lambert projection row were each refused that way
    on some runs and not others.

    A weak reference tells the two apart.  A serial is reused only while
    the object it was issued to is still alive and is the very object
    asked about; an address reused by a new object gets a new serial.  An
    object that cannot be weakly referenced is held strongly for as long as
    this ledger lives, so its address cannot be reused while it is compared.
    A ledger lives for one :func:`capture_step` call.
    """

    def __init__(self) -> None:
        self._entries: dict[int, tuple[Callable[[], Any], int]] = {}
        self._issued = 0

    def serial(self, obj) -> int:
        entry = self._entries.get(id(obj))
        if entry is not None and entry[0]() is obj:
            return entry[1]
        try:
            ref = weakref.ref(obj)
        except TypeError:
            ref = (lambda held=obj: held)
        self._issued += 1
        self._entries[id(obj)] = (ref, self._issued)
        return self._issued


def _fingerprint_value(value, depth: int = 0, *, ids: bool = True,
                       shapes_only: bool = False,
                       identity: Callable[[Any], int] | None = None) -> str:
    """A stable text rendering of one host attribute.

    Device arrays are rendered as their identity and shape, NOT their
    contents: the point of the fingerprint is to catch host state that a
    replay would leave stale, and array contents are exactly what the replay
    does update.  An array REBOUND to a different object is host state and is
    caught, because the id moves.

    ``ids=False`` renders the same thing with the object identities left out,
    which is what separates "this attribute now holds a different KIND of
    thing" from "this attribute holds a freshly allocated one of the same
    thing".  :func:`capture_step` needs both, and treats them very
    differently -- see the rebinding discussion there.

    ``identity`` replaces the bare ``id()`` in that rendering, and
    :func:`capture_step` passes :meth:`ObjectLifetimes.serial` so a freed
    address that a new object reuses does not read as the old object.
    """
    if isinstance(value, _SCALARISH):
        return "" if shapes_only else repr(value)
    if isinstance(value, (np.generic,)):
        return "" if shapes_only else f"np:{value!r}"
    if depth > 3:
        return "" if shapes_only else f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{k!r}:{part}" for k, part in (
                (k, _fingerprint_value(v, depth + 1, ids=ids,
                                       shapes_only=shapes_only,
                                       identity=identity))
                for k, v in sorted(value.items(), key=lambda kv: repr(kv[0])))
            if part or not shapes_only) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(
            part for part in (
                _fingerprint_value(v, depth + 1, ids=ids,
                                   shapes_only=shapes_only,
                                   identity=identity) for v in value)
            if part or not shapes_only) + "]"
    if not ids or shapes_only:
        tag = ""
    elif identity is None:
        tag = f" id={id(value):x}"
    else:
        tag = f" id=#{identity(value)}"
    shape = getattr(value, "shape", None)
    if shape is not None:
        dtype = getattr(value, "dtype", "")
        return (f"<{type(value).__name__}{tag} shape={tuple(shape)} "
                f"dtype={dtype}>")
    fields = getattr(value, "__dict__", None)
    if fields is not None and (shapes_only or not ids):
        # A rebound CONTAINER (a tendency bundle, say) is described by what
        # it holds, so a fresh instance around the same shapes reads as the
        # same structure while a different bundle does not.
        return ("<" + type(value).__name__ + " " + ",".join(
            f"{k}={part}" for k, part in (
                (k, _fingerprint_value(v, depth + 1, ids=ids,
                                       shapes_only=shapes_only,
                                       identity=identity))
                for k, v in sorted(fields.items()))
            if part or not shapes_only) + ">")
    return f"<{type(value).__name__}{tag}>"


def host_fingerprint(state, *, ids: bool = True,
                     shapes_only: bool = False,
                     identity: Callable[[Any], int] | None = None) -> str:
    """Digest the host-side state a replay would NOT update.

    Walks the ``DomainState``'s and the ``PhysicsDriver``'s ``__dict__`` --
    every Python attribute, one level into containers -- and renders each as
    text.  Two fingerprints differ exactly when some host attribute moved.

    This is the mechanism behind failure 3 in the module docstring.  A step's
    Python side effects happen only at capture; a replay is sound only if
    repeating the step leaves the host state where it already is.  Comparing
    the fingerprint after two consecutive captures from the SAME clock tests
    that directly, and names the attribute when it fails.

    ``identity`` is passed through to :func:`_fingerprint_value`.
    """
    parts = []
    driver = getattr(state, "physics", None)
    for label, obj in (("state", state), ("driver", driver)):
        if obj is None:
            continue
        for name, value in sorted(vars(obj).items()):
            if obj is state and value is driver:
                # Walked separately, and walking it twice renders the whole
                # driver as ONE line of ``state.physics``, so any attribute
                # that moves anywhere inside it reports as "state.physics
                # changed" with a kilobyte of context.  The first failure
                # this check caught printed 883 KB.
                continue
            part = _fingerprint_value(value, ids=ids,
                                      shapes_only=shapes_only,
                                      identity=identity)
            if part or not shapes_only:
                parts.append(f"{label}.{name}={part}")
    return "\n".join(parts)


def fingerprint_diff(a: str, b: str, limit: int = 6, width: int = 160) -> list[str]:
    """The attribute lines that differ, for an error message worth reading.

    Truncated on purpose.  A container attribute renders its whole contents,
    and an untruncated diff of a physics driver is a kilobyte per line -- the
    first real failure this check caught printed 883 KB of it into the gate
    log, which is the same as printing nothing.
    """
    def _short(line: str) -> str:
        return line if len(line) <= width else line[:width] + "..."

    left, right = a.split("\n"), b.split("\n")
    keyed = {line.split("=", 1)[0]: line for line in left}
    out = []
    for line in right:
        key = line.split("=", 1)[0]
        if keyed.get(key) != line:
            out.append(f"{_short(keyed.get(key, '<absent>'))}"
                       f"\n      ->  {_short(line)}")
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# One captured step
# ---------------------------------------------------------------------------

def scalar_delta(before: dict, after: dict) -> dict:
    """What ONE step added to the scalar carriers, as an increment.

    Stored as a delta rather than as the post-step values because the two
    callers need different things from it and a delta is right for both: a
    straight-line loop replays the graph from a clock that keeps advancing,
    while a tiled sweep resets every buffer to the DOMAIN clock before each
    tile and so replays from the same clock every time.  Post-step VALUES
    would freeze the straight-line loop's clock at step one -- which is
    exactly what the first version of this did, and the probe caught it by
    comparing the carriers as well as the field digest.
    """
    out: dict = {}
    for name, new in after.items():
        old = before.get(name)
        if isinstance(new, dict):
            if any(isinstance(v, dict) for v in new.values()):
                # A TABLE, not a counter: the carrier contract's records
                # ({carrier: {source, last_update_model_time}}).  A
                # produced-at stamp has no meaningful increment -- the
                # captured step's producer wrote it AT a model second --
                # so the post-step value is carried whole and re-applied
                # as an absolute.  Right for the tiled sweep by
                # construction: every replay of this graph starts from
                # the same domain clock the capture did (reuse="sweep"
                # keys on the sweep), so the capture's stamps are the
                # replayed step's stamps.
                out[name] = new
            else:
                out[name] = {k: v - old.get(k, 0) for k, v in new.items()}
        elif isinstance(new, (int, float)) and not isinstance(new, bool):
            out[name] = new - (old or 0)
        else:
            out[name] = new
    return out


def apply_scalar_delta(current: dict, delta: dict) -> dict:
    """``current + delta``, matching :func:`scalar_delta`'s shapes."""
    out: dict = {}
    for name, inc in delta.items():
        have = current.get(name)
        if isinstance(inc, dict):
            if any(isinstance(v, dict) for v in inc.values()):
                out[name] = inc          # absolute table; see scalar_delta
            else:
                out[name] = {k: (have or {}).get(k, 0) + v
                             for k, v in inc.items()}
        elif isinstance(inc, (int, float)) and not isinstance(inc, bool):
            out[name] = (have or 0) + inc
        else:
            out[name] = inc
    return out


@dataclass
class StepGraph:
    """One step of one buffer at one topology, captured and replayable.

    ``pool`` is not a detail: see failure 4 in the module docstring.  It is
    the private allocator the capture ran under and it must outlive the
    graph, which is why it is held here rather than dropped after capture.
    """

    graph: Any
    pool: Any
    key: tuple
    nodes: int
    fingerprint: str
    scalars_delta: dict
    host_after: str
    #: The graph of the FIRST capture, kept only when it differs from the
    #: settled one -- the lazily-allocated scratch case documented in
    #: :func:`capture_step`.  Launched exactly once, then dropped.
    settle: Any = None
    #: Host attributes that moved between two captures of the same step and
    #: are NOT re-applied by a replay -- diagnostic counters and cached
    #: histories.  Reported rather than refused; see :func:`capture_step`.
    drift: list = field(default_factory=list)
    capture_seconds: float = 0.0
    replays: int = 0

    def launch(self, stream) -> None:
        if self.settle is not None:
            self.settle.launch(stream)
            self.settle = None
        else:
            self.graph.launch(stream)
        self.replays += 1


#: ``debug_dot_str`` names every node after the graph's own serial number
#: (``graph_7_node_12``), which increments with each capture in the process.
#: Two captures of the SAME step therefore differ in every line unless the
#: serial is normalised away -- and the first run of the two-capture check
#: reported exactly that: 232 nodes both times, two different digests.
_GRAPH_SERIAL = __import__("re").compile(r"(graph|cluster)_\d+")


def _dot_fingerprint(graph) -> tuple[int, str]:
    """``(node count, sha256 of the structure)`` for a captured graph.

    ``debug_dot_str(0)`` prints one node per launch with the kernel's name
    and the dependency edges, and NO pointers -- which is what makes it a
    topology fingerprint rather than an address fingerprint.  Two captures of
    the same step must produce the same string even though their temporaries
    sit at different addresses.
    """
    dot = _GRAPH_SERIAL.sub(r"\1", graph.debug_dot_str(0))
    nodes = dot.count("[style=")
    return nodes, hashlib.sha256(dot.encode()).hexdigest()


def capture_step(state, cfg, stream, *, step_fn=None, scalars_fn=None,
                 set_scalars_fn=None, verify_host: bool = True,
                 pool=None) -> StepGraph:
    """Capture ONE step of ``state`` on ``stream`` and return it replayable.

    The capture does not execute anything: it records.  The caller must
    launch the returned graph to actually perform the step -- which is the
    one thing about graph capture that reads backwards and, left unnoticed,
    silently drops a step.

    ``verify_host=True`` (the default, and the reason this is not four lines)
    captures the same step SEVERAL times from the same clock and requires the
    result to settle.  Two distinct things come out of that, and the second
    was found by running it rather than by thinking about it.

    **The fixed-point check.**  A replay executes device work only, so
    replaying is sound only if repeating the step leaves the HOST state where
    it already is.  Two captures from the same clock must therefore produce
    the same host fingerprint; a graph that fails is refused, and the message
    names the attribute that moved.

    **The settling capture.**  MEASURED, on the dry lane at a 224 window: the
    first capture of a fresh tile buffer had 241 nodes and the second 232.
    The nine extra were the memsets of arrays the step allocates LAZILY --
    ``state.scratch`` slots that did not exist until the first step asked for
    them.  Capture allocates them (that is host-side Python and it really
    runs) but only RECORDS their zeroing, so a graph captured second, cached
    and replayed would leave that scratch holding whatever the pool last put
    there.  Worse in the other direction too: caching the FIRST graph would
    re-zero, every tile, scratch that legitimately carries information across
    steps -- ``state._scratch`` is in the streamed carrier manifest precisely
    because it does.  So when the first capture differs from the second, the
    first is kept as a one-shot ``settle`` graph that is launched exactly
    once (performing this step, zeroing included) and the SETTLED graph is
    what gets cached.  That is what the stream path does: zero on first use,
    not on every step.
    """
    import time

    import cupy as cp

    from woof.core.dycore import step as _step

    run = step_fn or (lambda: _step(state, cfg))
    key = cadence_key(state, cfg)
    pool = pool or cp.cuda.MemoryPool()
    t0 = time.perf_counter()
    # Identities in the host fingerprints below are LIFETIMES, not
    # addresses: a fresh object that lands on a freed predecessor's
    # address must not read as that predecessor coming back.
    lifetimes = ObjectLifetimes()

    before = None if scalars_fn is None else dict(scalars_fn(state))

    def _one_capture():
        with cp.cuda.using_allocator(pool.malloc):
            with stream:
                stream.begin_capture()
                try:
                    run()
                except Exception:
                    try:
                        stream.end_capture()
                    except Exception:                      # noqa: BLE001
                        pass
                    raise
                return stream.end_capture()

    def _rewind():
        if before is not None and set_scalars_fn is not None:
            set_scalars_fn(state, before)

    # Capture up to four times from the SAME clock.  Two things have to be
    # separated out of the sequence and both were found by running it:
    #
    #   * a SETTLING capture.  The first capture of a fresh buffer allocates
    #     whatever the step allocates lazily and records the memsets that
    #     zero it; the second does not.  Measured on the dry lane at a 224
    #     window: 241 nodes then 232.  The first graph is kept and launched
    #     exactly once (it owes that zeroing); the settled graph is cached.
    #     Caching the first instead would re-zero, on every tile, scratch
    #     that legitimately carries information across steps.
    #
    #   * a REBINDING.  With ``bldt=0`` the PBL slot builds a fresh
    #     ``PhysicsTendencies`` every step and the driver rebinds
    #     ``pbl_tendencies``/``tendencies`` to it, so the host fingerprint
    #     moves even though nothing about the step changed.  That is safe to
    #     replay -- the graph's pointers and the driver's attribute agree
    #     with each other, and the bundle is written before it is read -- but
    #     only if the rebinding is a FRESH object each time.  A driver that
    #     PING-PONGED between two bundles would look identical after two
    #     captures and be wrong on every second replay, so the identities are
    #     required to be pairwise distinct across the captures rather than
    #     merely different.
    # Capture up to four times from the SAME clock and require the result to
    # settle.  Three separate things come out of the sequence and every one
    # of them was found by running it, not by thinking about it.
    #
    # 1. A SETTLING capture.  The first capture of a fresh buffer allocates
    #    whatever the step allocates lazily and records the memsets that zero
    #    it; the second does not.  Measured on the dry lane at a 224 window:
    #    241 nodes then 232.  The first graph is kept and launched exactly
    #    once -- it owes that zeroing -- and the SETTLED graph is cached.
    #    Caching the first instead would re-zero, on every tile, scratch that
    #    legitimately carries information across steps: ``state._scratch`` is
    #    in the streamed carrier manifest precisely because it does.
    #
    # 2. What must be STATIONARY, and what need not be.  A replay updates
    #    device memory and runs no Python, so the question is which host
    #    state going stale would change an answer.  Three things are checked
    #    and they are the three that can:
    #      * the SHAPES.  If a step's second execution allocates a
    #        differently shaped bundle, the graph's baked pointers describe
    #        the wrong array.  Fatal.
    #      * the SCALAR CARRIERS.  These are the project's own answer to
    #        "which host state is trajectory-relevant" -- the domain clock
    #        and the call counters every physics cadence is a function of --
    #        and a replay re-applies their increment, so the increment has to
    #        be the same every time.  Fatal if it is not.
    #      * OBJECT IDENTITY, but only for repeats.  With ``bldt=0`` the PBL
    #        slot builds a fresh ``PhysicsTendencies`` every step and the
    #        driver rebinds to it; that replays correctly, because the
    #        graph's pointers and the driver's attribute agree with each
    #        other and the bundle is written before it is read.  A driver
    #        that PING-PONGED between two bundles would look identical after
    #        two captures and be wrong on every second replay, so an identity
    #        that comes BACK is fatal while an identity that merely moves is
    #        not.
    #    Everything else that moves -- ``RRTMGPRadiation.update_count``,
    #    Kain-Fritsch's ``_history_time`` -- is reported as drift and
    #    allowed, because the tiled driver ALREADY desynchronises exactly
    #    those from the domain (a buffer serving k tiles advances them k
    #    times per sweep while the domain advances once) and the gate has
    #    proved, rung by rung, that the forecast does not depend on them.
    #    Refusing them would refuse every physics rung for a reason the gate
    #    says is not a reason.
    captures = []
    for attempt in range(4):
        if attempt:
            _rewind()
        graph = _one_capture()
        nodes, fp = _dot_fingerprint(graph)
        if nodes < MIN_PLAUSIBLE_NODES:
            raise GraphCaptureError(
                f"captured only {nodes} nodes, which is not a step.  The "
                "usual cause is capturing without making the stream CURRENT, "
                "which produces an empty graph that launches successfully "
                "and does nothing")
        captures.append((graph, nodes, fp,
                         host_fingerprint(state, identity=lifetimes.serial),
                         host_fingerprint(state, shapes_only=True),
                         None if scalars_fn is None else dict(scalars_fn(state))))
        if attempt == 0:
            scalars_after = captures[0][5]
        if not verify_host:
            break
        trailing = 0
        for older in reversed(captures):
            if older[1:3] != captures[-1][1:3]:
                break
            trailing += 1
        if trailing >= 2 and captures[-1][3] == captures[-2][3]:
            break                       # identical down to object identity
        if trailing >= 3:
            break                       # a rebinding: three captures are
                                        # what tells a fresh object from a
                                        # ping-pong

    settle = None
    graph, nodes, fp, host_after, shapes, _sc = captures[-1]
    drift: list[str] = []
    if verify_host:
        if captures[0][1:3] != captures[-1][1:3]:
            settle = captures[0][0]
        stable = [c for c in captures if c[1:3] == captures[-1][1:3]]
        if len(stable) < 2:
            shown = " then ".join(f"{c[1]} nodes/{c[2][:12]}" for c in captures)
            raise GraphCaptureError(
                f"repeated captures of the same step from the same clock "
                f"never settled: {shown}.  A single settling pass explains "
                "two different graphs; more than that means the launch "
                "sequence is not a function of the host clock, and no cache "
                "key computed on the host can be correct for it")
        if stable[-1][4] != stable[-2][4]:
            raise GraphCaptureError(
                "the step's array SHAPES are not stationary, so the graph's "
                "baked pointers would describe the wrong arrays on a "
                "replay:\n  " + "\n  ".join(
                    fingerprint_diff(stable[-2][4], stable[-1][4])))
        if stable[-1][5] != stable[-2][5]:
            raise GraphCaptureError(
                f"the step's scalar-carrier increment is not repeatable "
                f"({stable[-2][5]} then {stable[-1][5]}), so no single "
                "increment can be re-applied after a replay")
        if stable[-1][3] != stable[-2][3]:
            drift = [line.split("=", 1)[0].strip()
                     for line in fingerprint_diff(stable[-2][3], stable[-1][3],
                                                  limit=99)]
            ids = [c[3] for c in stable]
            if len(stable) < 3 or len(set(ids)) != len(ids):
                raise GraphCaptureError(
                    "the step rebinds " + ", ".join(drift[:4]) + " and the "
                    "same object comes back on a later capture, which is a "
                    "double buffer: replaying one captured generation would "
                    "be wrong on every second step")

    delta = ({} if before is None or scalars_after is None
             else scalar_delta(before, scalars_after))
    return StepGraph(graph=graph, pool=pool, key=key, nodes=nodes,
                     fingerprint=fp, scalars_delta=delta,
                     host_after=host_after, settle=settle, drift=drift,
                     capture_seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# The per-buffer cache the tiled loop drives
# ---------------------------------------------------------------------------

@dataclass
class GraphStepper:
    """Capture-once/replay-per-tile for ONE tile buffer.

    The tiled loop calls :meth:`run` where it used to call ``dycore.step``.
    The first tile a buffer serves at a given key pays a capture; every later
    tile under that key pays one ``cuGraphLaunch``.

    ``reuse``, and why the default is the cautious one
        A graph bakes in every kernel's scalar ARGUMENTS as well as its
        pointers, and several schemes pass the absolute model time as one:
        RRTMGP takes ``xtime``/``julian`` for the solar geometry
        (rrtmgp.py:2045, :2201), Noah takes ``itimestep``, Noah-MP takes the
        Julian day.  A graph captured at time t and replayed at time t + dt
        would silently re-use t's sun.  So the default,
        ``reuse="sweep"``, puts the CLOCK in the cache key: one capture per
        step, replayed for every tile of that step.  That is the whole win
        for tiling -- the launch cost is what is multiplied by the tile
        count -- and it costs nothing in soundness.

        ``reuse="run"`` keys on the cadence flags alone and so re-uses a
        graph across steps.  It is much cheaper -- MEASURED at 96x80 with
        2x2 tiles, N=8: 2 captures totalling 130 ms instead of 16 totalling
        423 ms -- and it is correct only for a configuration whose kernels
        take no absolute time.  Which configurations those are is a
        measurement and not a guess.  At 48x40 tiles, N=8, against the
        monolithic answer::

            reuse   rung                     result
            sweep   every rung tried         BIT-EXACT
            run     mp10 Morrison            BIT-EXACT
            run     +YSU PBL                 BIT-EXACT
            run     full(real74) +KF         BIT-EXACT   (radt 12 min:
                                                          radiation fires on
                                                          1 step in 240 and
                                                          not inside this run)
            run     full fast cadence        MISMATCH    (radt 0.05 min:
                                                          radiation fires
                                                          EVERY step and the
                                                          solar hour angle is
                                                          baked in)

        So the accurate rule is that ``reuse="run"`` is safe for exactly as
        long as no radiation-due step is inside the run, which is not a
        property a config can promise.  The default stays ``"sweep"``; a
        caller who knows the run is short and radiation is not due can take
        the cheaper key with their eyes open, and the fast-cadence row is
        what a wrong guess looks like.

    ``mode``
        ``"auto"`` falls back to stream launching when a step cannot be
        captured, and records why.  ``"require"`` raises instead -- which is
        what a benchmark wants, because a benchmark that silently measures
        the fallback measures nothing.

    ``key_fn``
        :func:`cadence_key`, or a deliberately broken one.  The gate passes
        ``lambda *_: ("fixed",)`` as a negative control: one graph for every
        step regardless of cadence, which must FAIL at a rung whose
        radiation or cumulus fires part-way through the run.

    ``replay_scalars``
        Re-apply the scalar-carrier increment the capture measured.  A
        replay runs no Python, so without this the buffer's clock and call
        counters stay at the pre-step values.  The gate turns it off as the
        second negative control.

    One memory pool serves ALL of a stepper's graphs, and that is deliberate
    in both directions: two graphs of the SAME buffer never run at the same
    time (they belong to different steps, and the sweep synchronises between
    steps), so sharing addresses between them is safe; two graphs of
    DIFFERENT buffers do run at the same time, which is why the pool lives on
    the stepper and a stepper serves one buffer.
    """

    cfg: Any
    mode: str = "auto"
    reuse: str = "sweep"
    key_fn: Callable = cadence_key
    replay_scalars: bool = True
    verify_host: bool = True
    verify_topology: bool = False
    max_graphs: int = 4
    scalars_fn: Callable | None = None
    set_scalars_fn: Callable | None = None
    graphs: dict = field(default_factory=dict)
    captures: int = 0
    replays: int = 0
    fallbacks: int = 0
    capture_seconds: float = 0.0
    reason: str | None = None
    ledger: Any = None
    _uncapturable: set = field(default_factory=set)
    #: The sweep the driver is currently in; part of the cache key under
    #: ``reuse="sweep"``.  Set by :func:`tilestream.driver.run_tiled`.
    sweep: int = 0
    _pool: Any = None
    _verified: set = field(default_factory=set)

    def _health(self):
        """This buffer's health ledger, created on first use.

        One per stepper, i.e. one per tile buffer, for the same reason the
        memory pool is: two buffers step concurrently on two streams, and a
        shared slot word would be a read-modify-write race that can only ever
        LOSE a fault flag.
        """
        from woof.core import health_ledger

        if self.ledger is None:
            self.ledger = health_ledger.HealthLedger()
        return self.ledger

    def _fallback(self, state, before) -> None:
        """Step the ordinary way after a capture refused.

        The restore is not tidiness.  A capture RUNS the step's Python -- it
        records the device work but really executes the host bookkeeping --
        so a capture that refuses part-way through has already incremented
        the driver's call counters and, on a later attempt, its clock.
        Running the real step on top of that would leave the buffer's scalar
        carriers ahead of the domain, and the tiled driver's cross-buffer
        clock check would refuse the sweep (or worse, the buffers would
        agree with each other and all be wrong together).  Found by reading;
        the MYNN rung is where it fires.
        """
        from woof.core.dycore import step as _step

        if before is not None and self.set_scalars_fn is not None:
            self.set_scalars_fn(state, before)
        self.fallbacks += 1
        # NOT under the ledger: a fallback step is an ordinary step and its
        # health belongs where it happens.
        _step(state, self.cfg)

    def drain(self) -> None:
        """Report anything the schemes flagged since the last drain.

        The caller owes this at every point it synchronises anyway -- for
        the tiled driver, the end of a sweep.  A ledger that is never
        drained turns a fatal non-finite field into no error at all, which
        is why :func:`tilestream.driver.run_tiled` drains unconditionally
        and this method is cheap enough to call when nothing was recorded.
        """
        if self.ledger is not None:
            self.ledger.drain()

    def set_sweep(self, sweep: int) -> None:
        """Expire graphs whose absolute-time operands belong to the prior step.

        TiledRun synchronizes every graph sweep before entering this boundary;
        the private pool can then reuse the released graphs' workspace blocks.
        """
        if self.reuse == "sweep" and self.sweep != sweep:
            self.graphs.clear()
        self.sweep = sweep

    def set_config(self, cfg) -> None:
        """Rebind scalar kernel arguments after the caller drains the buffer.

        Captured graphs retain dt/acoustic-loop operands and workspaces.
        They cannot be replayed after those operands change.
        """
        if cfg == self.cfg:
            return
        self.graphs.clear()
        self._verified.clear()
        self._uncapturable.clear()
        self.reason = None
        self.cfg = cfg

    def _key(self, state) -> tuple:
        base = self.key_fn(state, self.cfg)
        from woof.core.dycore import wrf_cfl_capture_key
        base += (wrf_cfl_capture_key(self.cfg.grid_id),)
        # The cadence booleans choose topology; these intervals are scalar
        # kernel operands even when that topology is unchanged.
        driver = getattr(state, "physics", None)
        base += tuple(getattr(driver, name, None) for name in (
            "radt_seconds", "cudt_seconds", "bldt_seconds"))
        if self.reuse == "run":
            return base
        if self.reuse != "sweep":
            raise ValueError(
                f"reuse must be 'sweep' or 'run', got {self.reuse!r}")
        # The SWEEP index, supplied by the driver, and not the buffer's own
        # clock.  They are the same thing for a physics run, where every tile
        # is reset to the domain clock before it steps -- but a DRY run
        # carries no scalars at all, so each buffer's clock free-runs and
        # keying on it gave every tile its own graph and no reuse whatever.
        # Measured before the fix: 16 captures for 16 tile-steps, and the
        # graph path 12% SLOWER than the stream path.
        return base + (self.sweep,)

    def run(self, state, stream) -> str:
        """Advance ``state`` one step.  Returns ``"capture"``/``"replay"``/``"stream"``."""
        import cupy as cp

        from woof.core.dycore import step as _step

        if self.mode == "off":
            _step(state, self.cfg)
            return "stream"

        key = self._key(state)
        base = self.key_fn(state, self.cfg)
        if base in self._uncapturable:
            # Already established, on this buffer, that this topology cannot
            # be captured.  Retrying every step would pay a failed capture
            # per step -- and each failed capture runs part of the step's
            # HOST bookkeeping before it refuses, which then has to be undone.
            self._fallback(state, None)
            return "stream"

        held = self.graphs.get(key)
        if held is None:
            if self._pool is None:
                self._pool = cp.cuda.MemoryPool()
            from woof.core import health_ledger

            before = (None if self.scalars_fn is None
                      else dict(self.scalars_fn(state)))
            try:
                with health_ledger.deferring(self._health()):
                    held = capture_step(
                        state, self.cfg, stream,
                        scalars_fn=self.scalars_fn,
                        set_scalars_fn=self.set_scalars_fn,
                        pool=self._pool,
                        # The fixed-point check is a property of the CONFIG
                        # at a topology, not of the particular clock, so it
                        # is paid once per topology, not once per step.
                        verify_host=(self.verify_host
                                     and base not in self._verified))
            except Exception as exc:                       # noqa: BLE001
                if self.mode == "require":
                    if isinstance(exc, GraphCaptureError):
                        raise
                    raise GraphCaptureError(
                        f"the step could not be captured as a graph: {exc}"
                    ) from exc
                self.reason = f"{type(exc).__name__}: {exc}"
                self._uncapturable.add(base)
                self._fallback(state, before)
                return "stream"
            self._verified.add(base)
            while len(self.graphs) >= max(1, int(self.max_graphs)):
                self.graphs.pop(next(iter(self.graphs)))
            self.graphs[key] = held
            self.captures += 1
            self.capture_seconds += held.capture_seconds
            held.launch(stream)
            self.replays += 1
            return "capture"

        if self.verify_topology:
            from woof.core import health_ledger

            with health_ledger.deferring(self._health()):
                self._verify(state, stream, held)

        held.launch(stream)
        self.replays += 1
        if self.replay_scalars and held.scalars_delta and self.set_scalars_fn:
            self.set_scalars_fn(state, apply_scalar_delta(
                self.scalars_fn(state), held.scalars_delta))
        return "replay"

    def _verify(self, state, stream, held) -> None:
        """Recapture this step and demand the cached graph still describes it.

        The expensive, paranoid mode.  It is what proves :func:`cadence_key`
        is COMPLETE rather than merely plausible: a step whose true launch
        sequence differs from the graph cached under its key is caught here
        instead of being replayed.

        The scalar carriers are put back afterwards.  A capture RUNS the
        step's host bookkeeping -- that is the whole reason a replay has to
        re-apply it -- so a verification capture advances the buffer's clock
        and call counters, and the replay that follows would then advance
        them a second time.  Restoring makes the verification invisible to
        everything except the time it takes, which is what a check has to be.
        """
        before = (None if self.scalars_fn is None
                  else dict(self.scalars_fn(state)))
        # And the BINDINGS, which is the part that is not obvious and which
        # produced a wrong answer the first time this mode ran.  A
        # verification capture executes the step's Python, so a driver that
        # allocates a fresh tendency bundle every step rebinds to a NEW one
        # -- and then the cached graph, whose pointers name the OLD bundle,
        # writes arrays that the scatter no longer ships, while the arrays it
        # does ship are never written.  Bit-exactness went to maxabs 9.0e+06
        # on state/nc.  Restoring the attribute bindings undoes the rebinding
        # without touching a byte of device memory.
        bound = [(obj, dict(vars(obj)))
                 for obj in (state, getattr(state, "physics", None))
                 if obj is not None]
        fresh = capture_step(state, self.cfg, stream,
                             scalars_fn=self.scalars_fn,
                             set_scalars_fn=self.set_scalars_fn,
                             pool=self._pool, verify_host=False)
        for obj, snapshot in bound:
            vars(obj).clear()
            vars(obj).update(snapshot)
        if before is not None and self.set_scalars_fn is not None:
            self.set_scalars_fn(state, before)
        if (fresh.nodes, fresh.fingerprint) != (held.nodes, held.fingerprint):
            raise GraphCaptureError(
                f"topology key {held.key} is incomplete: the cached graph has "
                f"{held.nodes} nodes / {held.fingerprint[:16]}, this step "
                f"captures {fresh.nodes} nodes / {fresh.fingerprint[:16]}.  "
                "Replaying the cached graph would run the wrong kernels")

    def report(self) -> dict:
        return dict(graphs=len(self.graphs), captures=self.captures,
                    replays=self.replays, fallbacks=self.fallbacks,
                    capture_seconds=self.capture_seconds,
                    nodes=sorted({g.nodes for g in self.graphs.values()}),
                    reuse=self.reuse, reason=self.reason,
                    health_records=(0 if self.ledger is None
                                    else self.ledger.records),
                    health_drains=(0 if self.ledger is None
                                   else self.ledger.drains))
