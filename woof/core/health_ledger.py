"""Deferred readback of the scheme health status words.

Four schemes end their step by writing a bitmask of "this output went
non-finite / out of range" into a one-word device buffer and reading it
straight back to the host:

============================  ==============================================
site                          what the word means
============================  ==============================================
``microphysics.py``           per-output finite flags for the canonical
                              surface diagnostics, plus SR below/above range
``ysu.py``                    per-output finite flags in launcher order
``kf.py``                     per-output finite flags for the KF arrays
``rrtmgp.py``                 the fused input-plausibility guard
============================  ==============================================

Each readback is a blocking device-to-host copy, and PERF-FINDINGS.md measures
one at ~94 us of pipeline drain -- 50 queued full-size axpy kernels went from
0.566 to 0.660 ms/iteration with a single trailing ``int(status[0].item())``.
That is the small half of the cost.  The large half is that a blocking read is
a HOST BRANCH OVER DEVICE DATA, and a host branch over device data means the
step's launch sequence is a function of the VALUES in the arrays.  While that
is true no CUDA graph can be captured (capture refuses the transfer outright)
and, more fundamentally, no cache key computable on the host can be correct,
because the host cannot see what the branch saw.

So this module does not merely make the reads cheaper.  It is what makes the
step's topology a function of the clock, which is the precondition
:mod:`tilestream.graphcap` rests on.


HOW THE DEFERRAL WORKS, AND WHAT IT COSTS
-----------------------------------------
A :class:`HealthLedger` owns a small uint32 device buffer, one slot per site.
``record`` ORs the site's status word into its slot -- a device-to-device
operation, capturable, and commutative, so many steps (or many TILES) can
accumulate into one slot without ordering.  ``drain`` reads the whole buffer
in ONE copy at a synchronisation the caller already pays for, and raises the
same exception the blocking site would have raised, built by the same closure.

What changes for a HEALTHY run: nothing.  Not the arithmetic, not the field
bytes, not the number of steps.  The gate proves it.

What changes for a SICK run: the failure is reported later -- at the next
drain rather than inside the step -- so the model executes more work before it
aborts.  The exception type and the offending field name are preserved; the
message says when it was detected, because a report that pretends to be
instantaneous when it is not is worse than a late one.  This is the same
trade-off the codebase already accepted when it replaced per-substep mass-flux
readbacks with :class:`~woof.core.dycore.MassFluxAccumulator`.

EXECUTOR POLICY
---------------
The resident forecast executor installs one ledger per domain and polls a
bounded pinned snapshot after each step. Polling only queries completed
CUDA events. History, checkpoint and final-return boundaries drain every
ledger before publishing or accepting a run. A failure can therefore advance
through internal nest work before it is reported. ``health_debug`` retains
immediate attribution. Other callers choose their own deferring scopes.
With no ledger installed, sites retain their immediate read and refusal.

DRAIN OR IT DID NOT HAPPEN
--------------------------
A ledger that is recorded into and never drained silently converts a fatal
error into no error at all. :meth:`HealthLedger.close` therefore drains.
The owner of a :func:`deferring` scope must drain or close its ledger on
all exit paths; that scope only routes records and restores the context.
The forecast executor drains in its finalizer as well as before publication.
Sites reach the ledger through :func:`read_status`.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable


#: The ledger the schemes report into, or ``None`` for the historical
#: read-immediately behaviour.  Module-level because the reporting sites are
#: four levels down inside three schemes and threading a parameter through
#: every one of them would be a larger and more fragile change than this.
#:
#: A context variable, not a plain global.  Breakage it prevents: concurrent
#: ensemble members (woof.ensemble.execution runs each on its own thread)
#: each wrap their domain step in :func:`deferring` with their OWN step
#: ledger (woof.core.model, since ac988f1fd).  With one process-wide slot,
#: member A's scheme sites recorded into member B's ledger while the steps
#: overlapped, so a sick member's refusal was raised by another member's
#: drain or lost, and the out-of-order exits restored a stale ledger for good:
#: every later run in the process then deferred its health reads into a ledger
#: nobody drains, and an immediate refusal became no refusal at all (the 2.8.7
#: GPU shard failures in test_mynn_validation_batching, test_mynn_pbl_gpu,
#: test_gf_engine_smoke and test_graphcap, after
#: test_ensemble_runtime_forecast_gpu's four-member run).  A context variable
#: gives every thread, and every ``copy_context()`` member scope, its own slot.
_ACTIVE: "ContextVar[HealthLedger | None]" = ContextVar(
    "gpuwm_health_ledger", default=None)


class HealthLedger:
    """Accumulate scheme status words on the device; read them once, later.

    ``capacity`` bounds the number of distinct SITES, not the number of
    calls: a site's slot is ORed into, so a hundred steps and a hundred tiles
    share one word.  A full-physics step reports from about twenty -- four
    scheme status words plus one per validated radiation/cumulus output --
    and the default leaves room.
    """

    def __init__(self, capacity: int = 64, *, label: str = "") -> None:
        import cupy as cp

        self._slots = cp.zeros(int(capacity), dtype=cp.uint32)
        self._index: dict[str, int] = {}
        self._describe: dict[str, Callable[[int], None]] = {}
        self._label = label
        self.records = 0
        self.drains = 0
        self._device = int(cp.cuda.runtime.getDevice())
        self._pending = None
        self._poll_stream = None
        self.polls = 0

    def poll(self) -> None:
        """Queue one pinned status copy and inspect completed copies only.

        The event is queried, never waited on. While a copy is pending,
        subsequent records accumulate in the device slots. At most one
        snapshot and its diagnostic closures are retained per ledger.
        Output/checkpoint/final boundaries still call drain().
        """
        import cupy as cp
        import numpy as np

        with cp.cuda.Device(self._device):
            self._poll_stream = cp.cuda.get_current_stream()
            self.polls += 1
            self._finish_pending(wait=False)
            if self._pending is not None or not self._describe:
                return
            memory = cp.cuda.alloc_pinned_memory(self._slots.nbytes)
            host = np.frombuffer(memory, dtype=np.uint32,
                                 count=self._slots.size)
            self._slots.get(out=host, blocking=False)
            event = cp.cuda.Event(disable_timing=True)
            event.record()
            # Copy, clear and the next record use the same ordered stream.
            self._slots.fill(cp.uint32(0))
            self._pending = (event, host, self._describe.copy())
            self._describe.clear()

    def _finish_pending(self, *, wait: bool) -> None:
        if self._pending is None:
            return
        event, host, descriptions = self._pending
        if wait:
            event.synchronize()
        elif not event.done:
            return
        self._pending = None
        self._raise_flags(host, descriptions)

    def _raise_flags(self, host, descriptions) -> None:
        for site, describe in descriptions.items():
            flags = int(host[self._index[site]])
            if flags:
                describe(flags)
                raise FloatingPointError(
                    f"{site} reported health flags {flags:#x} but its "
                    "description did not raise; a health site must refuse")

    @property
    def slots(self):
        return self._slots

    def _slot(self, site: str) -> int:
        i = self._index.get(site)
        if i is None:
            if len(self._index) >= int(self._slots.size):
                raise RuntimeError(
                    f"health ledger is full at {self._slots.size} sites; "
                    f"cannot add {site!r}")
            i = len(self._index)
            self._index[site] = i
        return i

    def record(self, site: str, status, describe: Callable[[int], None]) -> None:
        """OR ``status[0]`` into ``site``'s slot.  Device-to-device, capturable."""
        import cupy as cp

        i = self._slot(site)
        flags = status[:1].reshape(1)
        if flags.dtype != self._slots.dtype:
            # A site's status word may be a signed 32-bit integer (BEP's
            # error_flags carries a Fortran line number).  uint32 | int32
            # promotes to int64, which cannot be written back into the
            # uint32 slot, so every BEP and BEM run under a deferred ledger
            # crashed at its first urban call (17 combo-sweep runs,
            # 2026-10-07 round 2, hidden until then by the door's urban
            # refusal).  The words are flags: reinterpret the same 32 bits,
            # which keeps the record a view (no allocation, capturable).
            if (flags.dtype.kind not in "iu"
                    or flags.dtype.itemsize != self._slots.dtype.itemsize):
                raise TypeError(
                    f"health site {site!r} reported a {flags.dtype} status; "
                    "a status word is a 32-bit integer")
            flags = flags.view(self._slots.dtype)
        cp.bitwise_or(self._slots[i:i + 1], flags, out=self._slots[i:i + 1])
        self._describe[site] = describe
        self.records += 1

    def drain(self) -> None:
        """One D2H over every slot; raise for the first site that flagged.

        Clearing happens BEFORE the raise so a caller that catches and
        continues does not re-raise the same historical failure for ever.
        """
        import cupy as cp

        with cp.cuda.Device(self._device), (
                self._poll_stream or cp.cuda.get_current_stream()):
            self._finish_pending(wait=True)
            if not self._describe:
                return
            self.drains += 1
            host = cp.asnumpy(self._slots)
            self._slots.fill(cp.uint32(0))
            self._raise_flags(host, self._describe)
        # The synchronized read found every site healthy. Its diagnostic
        # closures are no longer needed and may own full atmospheric input
        # generations (YSU's producer-forensics mapping, for example).
        # Keep the slot indices, but release those arrays between steps;
        # record() installs the next description with the next status.
        # Failed drains retain their descriptions for inspection.
        self._describe.clear()

    def close(self) -> None:
        self.drain()


def active() -> "HealthLedger | None":
    return _ACTIVE.get()


@contextmanager
def deferring(ledger: "HealthLedger | None"):
    """Route the health sites into ``ledger`` for the duration.

    ``None`` restores the historical immediate reads, which is what makes
    this safe to wrap around anything.
    """
    token = _ACTIVE.set(ledger)
    try:
        yield ledger
    finally:
        _ACTIVE.reset(token)


def read_status(status, *, site: str, describe: Callable[[int], None] | None) -> int:
    """The status word -- read now, or recorded for later and reported as 0.

    ``describe=None`` forces the immediate read, so a caller that has not
    said how to report a failure never gets a deferral it cannot describe.
    Returning ``0`` on the deferred path is exactly right: zero means "no
    fault seen", which is what every caller does nothing about, and the fault
    -- if there is one -- is reported by the drain instead.
    """
    ledger = _ACTIVE.get()
    if ledger is None or describe is None:
        return int(status[0].item())
    ledger.record(site, status, describe)
    return 0


def _array_module(cp, array):
    """``cupy`` or ``numpy``, whichever OWNS ``array``.

    ``cupy.get_array_module`` is the canonical answer and is used when it is
    there.  It is not always there: this package is driven from CPU harnesses
    that substitute a numpy-backed stand-in for the ``cp`` name
    (tests/test_wrf_legacy_radiation.py's ``_NumpyCupy`` is one), and such a
    stand-in implements the array API without implementing cupy's dispatch
    helper -- so the canonical call raises AttributeError from inside a
    finiteness CHECK, which is the one place an exception is indistinguishable
    from the thing being checked for.  The fallback answers the same question
    from the array's own type.
    """
    getter = getattr(cp, "get_array_module", None)
    if getter is not None:
        return getter(array)
    import numpy as _np

    return _np if isinstance(array, _np.ndarray) else cp


def check_finite(array, *, site: str, message: str) -> bool:
    """``bool(isfinite(array).all())`` -- read now, or recorded for later.

    The same trade as :func:`read_status`, for the eight-to-fifteen scheme
    outputs the driver validates one at a time on a radiation- or
    cumulus-due step (``physics.py:_validated_array``).  Each one was a full
    reduction FOLLOWED BY A BLOCKING READ; deferred, the reduction still
    runs -- the check is not weakened -- and only the read moves.

    Returns True when the array is known finite (or when the verdict has
    been deferred, which is the same thing as far as every caller's
    behaviour on a healthy step is concerned).
    """
    import cupy as cp

    # The array's OWN module, not this one's.  ArWen drives several of these
    # sites from a CPU harness that monkeypatches the scheme module's ``cp``
    # to numpy (tests/test_physics_driver.py's WSM6 SR case), and a hard
    # ``cupy.isfinite`` there raises ``TypeError: Unsupported type
    # numpy.ndarray``.  Found by their own suite, which is what it is for.
    xp = _array_module(cp, array)
    ledger = _ACTIVE.get()
    if ledger is None or xp is not cp:
        return bool(xp.isfinite(array).all())
    status = cp.logical_not(cp.isfinite(array).all()).astype(
        cp.uint32).reshape(1)
    ledger.record(site, status,
                  lambda _flags: _raise_nonfinite(message))
    return True


def _raise_nonfinite(message: str) -> None:
    raise FloatingPointError(message + deferred_note())


def deferred_note(ledger: "HealthLedger | None" = None) -> str:
    """The sentence appended to a deferred failure, so nobody is misled."""
    return (" (detected at a deferred health drain, so the run advanced past "
            "the step that produced it; re-run with the ledger disabled to "
            "stop on the exact step)")


def masked_clear(mask, array) -> None:
    """``array[...] = where(mask, 0, array)`` as ONE predicated in-place pass.

    Same written values, no temporary, and -- the reason it is here -- no
    host branch: the caller that used to skip this work when nothing was
    masked can now run it unconditionally, which is what makes its launch
    sequence independent of the data.  When the mask is empty the kernel
    reads the mask and writes nothing.

    Bit-exact by construction: it is a select-and-store with no arithmetic,
    so a lane that is not cleared keeps the bits it already had.
    """
    import cupy as cp

    xp = _array_module(cp, array)
    if xp is not cp:
        # Same reason as :func:`check_finite`: a CPU harness reaches here
        # with numpy arrays, and the point of this function is the written
        # VALUES, which numpy's own ``where`` produces identically.
        array[...] = xp.where(mask, xp.float32(0.0), array)
        return

    global _MASKED_CLEAR
    if _MASKED_CLEAR is None:
        _MASKED_CLEAR = cp.ElementwiseKernel(
            "bool m", "float32 a", "if (m) { a = 0.0f; }",
            "gpuwm_masked_clear")
    # The KF mask arrives as (1, ny, nx) so it broadcasts over the 3-D rates;
    # a 2-D carrier needs it back at (ny, nx), because an ElementwiseKernel
    # broadcasts its INPUTS but writes its output in place and will not
    # broadcast the destination.
    if mask.ndim == array.ndim + 1 and mask.shape[0] == 1:
        mask = mask[0]
    _MASKED_CLEAR(mask, array)


_MASKED_CLEAR = None
