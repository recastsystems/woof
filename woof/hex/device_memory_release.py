# SPDX-License-Identifier: Apache-2.0
"""Give back the device memory nothing is using, at physics phase boundaries.

WHAT THIS IS FOR
----------------
A hex process holds two kinds of device memory that no live array uses:

1. **Cached pool blocks.**  CuPy's default pool keeps every block an array
   gave back, so the next request of a similar size is served without a
   cudaMalloc.  A request that none of those blocks can serve is met with
   fresh memory *on top of* them.  The 2026-10-14 memory census
   measured 3,960 MiB of such free
   blocks held at the route-B shortwave peak on x1.40962 (35 % of the pool)
   and 733 MiB at the route-A radiation peak; between radiation steps route A
   held 3.2 GiB of free radiation blocks through every dycore step.
2. **The CUDA local-memory store.**  The driver sizes it to the widest
   per-thread frame any launched kernel needed, times every thread the card
   can hold, and never shrinks it by itself.  Measured on route B: 9,036 MiB,
   7,980 of it for Thompson's 43,440-byte frame; on route A 1,546 MiB, 1,170
   of it for WSM6.  It stays held through the dycore half of every step,
   where no kernel needs it, and so it stacks on the dycore's transport peak.

Both are sized by the phase that needed them and both are then held through
every later phase, so a card had to hold the largest pool high-water of any
phase *plus* the largest frame store of any phase at the same time.

WHAT IT DOES
------------
:meth:`DevicePhaseRelease.phase` brackets one physics phase (a route-B
physics slot, or route A's phase one or phase two):

* on entry, the pool's wholly free cached blocks are returned to the driver
  (``free_all_blocks``), so the phase's own requests are met from memory the
  previous phase gave back instead of being stacked on top of it;
* on entry, a phase that needed a wide frame last time has its local-memory
  store grown back to that frame at once (``cudaDeviceSetLimit``), at a point
  where a failure can be handled (below), instead of inside its first kernel
  launch, where it cannot;
* on exit, a phase that grew the per-thread stack limit (WSM6 7,216 B,
  GWDO 2,528 B, longwave 2,048 B) has the limit set back to its value at entry, which makes the driver shrink the store to
  what it was before the phase, and the phase's free blocks are returned.
  ``GPUWM_HEX_LMEM_RELEASE_MIN_FRAME=<bytes>`` keeps frames narrower than
  that (default 4096): a kept narrow store sits on the dycore's transport
  peak (measured: keeping the 2,528 B store cost 160 MiB at that peak on a
  70-SM card), but growing it back synchronizes the device every step (see
  "WHAT A BOUNDARY COSTS" below).

Measured on an RTX 5090: a 40,000-byte frame grows a 9,706 MiB store;
setting the limit back returns all of it in about 4 ms.  Growing it with
cudaDeviceSetLimit took 0.4 ms for 3,996 MiB on an RTX 5070 Ti; a
kernel launch that grows it lazily took about 8 ms on the 5090.

WHICH PHASES STILL USE THE LOCAL-MEMORY HALF
--------------------------------------------
Thompson (43,440 B) and Noah (11,408 B) were the two frames that made the
store several GiB.  Thompson now keeps its column arrays in a pooled global
workspace tiled by column (``mpasphys/thompson_v842.py``: step frame 64 B,
radar 0 B on sm_120) and Noah reads its tables through one shared device copy
(``mpasphys/noah_v842.py``: widest frame 960 B, no recursion, no 8 KiB module
reserve).  Both now fit inside the driver's default 1,024 B limit, so their
phases never grow the store and this bracket does no reset or regrow for them:
the per-step cudaDeviceSetLimit pair, and its regrow of a multi-GiB store, is
gone from those phases without a knob.  The local-memory half is kept because
it still saves memory where a frame stays wider than the default: GWDO
(2,528 B, every step), longwave (2,048 B, radiation steps) and WSM6 (7,216 B,
suites that run it).  On an RTX 5090 a 2,528 B store held through the dycore
is about 0.3 GiB more than the default store.

A CARD SHARED WITH OTHER PROCESSES
----------------------------------
Memory given back to the driver can be taken by another process on the same
card before this one needs it again.  For the pool that window is one phase;
for the local-memory store it is the rest of the step.  So the store is grown
back on entry, and if the driver refuses (cudaErrorMemoryAllocation), this
process first returns its own cached blocks and retries, then waits for the
memory to come back (``GPUWM_HEX_LMEM_REGROW_WAIT_S``, default 1,800 s),
saying so on stderr.  Once it has the store again it keeps it for the rest of
the run (the receipt says ``pinned_after_regrow_wait``), as the run would
have without this module.  If the wait runs out the run stops with an error
that names the knob.  On a card the process has to itself none of this
happens.  Breakage it prevents: a kernel launch failing with
CUDA_ERROR_OUT_OF_MEMORY inside a physics scheme, with nothing to retry, when
a co-tenant took the store's memory between steps (seen with a co-tenant
job on the proving host, 2026-10-15).

BIT-IDENTITY
------------
Nothing here changes an operand, a kernel, a launch, or launch order.  Freed
memory is returned only when no array refers to it (``free_all_blocks`` frees
wholly unused chunks; cudaFree and cudaDeviceSetLimit wait for the device).
What changes is *where* later allocations land and what bytes an
uninitialised allocation happens to hold.  An output bit can only move if a
kernel reads memory it never wrote, which is a defect in that kernel, so every
route this is on is proved by a before/after byte comparison (route B: hl-3,
G1, G2; route A: history files), never by this argument.

KNOBS (each saving can be switched off for an A/B run)
------------------------------------------------------
``WOOF_HEX_POOL_RELEASE=0``  keep cached pool blocks across phase boundaries
``GPUWM_HEX_LMEM_RELEASE=0``  keep the local-memory store across phase boundaries
``GPUWM_HEX_LMEM_RELEASE_MIN_FRAME=<bytes>``  narrowest frame whose store is released (4096)
``GPUWM_HEX_POOL_RELEASE_MIN_BYTES=<bytes>``  idle cached pool bytes a boundary keeps whatever the headroom (8 GiB)
``GPUWM_HEX_RELEASE_HEADROOM=<fraction>``  free share of the card above which bigger idle sets and wide stores are kept (0.15; 1 releases them at every boundary)
``GPUWM_HEX_LMEM_REGROW_WAIT_S=<seconds>``  how long to wait for a refused regrow (1800)

WHAT A BOUNDARY COSTS (U-MPASPHYS-1, measured 2026-10-24)
---------------------------------------------------------
Both returns synchronize the device: cudaFree and cudaDeviceSetLimit wait for
every queued kernel, so the single host thread cannot run ahead of the card.
On the 216,187-cell MPAS-physics zoom (RTX PRO 6000, 55 levels) the release
returned about 14 GB per step, 1.2 to 4.6 GB from each of microphysics, PBL,
convection, surface layer and GWDO, and the same sizes were allocated again
the next step; the GWDO and PBL stores (2,736 B frames) were reset and regrown
every step.  Profiled: 41 s of free_all_blocks and 32 s of cudaDeviceSetLimit
over 39 steps, 1.9 s per step of a 4 s outer-physics stage; with a 4 GiB keep
the PBL, convection, microphysics and surface-layer transients (1.9-6.6 GiB a
step each) still went back and came again, 74 % of the remaining outer-physics
host time.  So a boundary keeps up to ``GPUWM_HEX_POOL_RELEASE_MIN_BYTES``
(8 GiB) of idle cached blocks, and returns them anyway whenever the card has
under a tenth of its memory free (another process needs it; cudaMemGetInfo
does not synchronize).  Measured cost of keeping every block on that mesh:
peak 32.3 GB instead of 24.6 GB.  Frames narrower than 4 KiB keep their store
(GWDO 2,528-2,736 B, longwave 2,048 B; WSM6's 7,216 B store is still
released).  0 restores the release of every block or frame.

RETURN ONLY UNDER PRESSURE (PROFILE-3KM item 4, measured 2026-10-28)
--------------------------------------------------------------------
The fixed 8 GiB keep broke at 3 km.  On mv3km (hrrrv5, 206,726 cells, 59
levels, RTX PRO 6000) the dycore's transients are bigger than 8 GiB, so every
step the surface-layer boundary returned them: 25.3 synchronising cudaFree per
step (430 ms of host time waiting on the card), 27 cudaMalloc for the same
blocks in the next dycore, and the card idle 19.9 ms before the surface layer
and 6.0 ms before GWDO, on a card with 55 GB free.  So an idle set above the
8 GiB keep is now returned only when the card is short: when it has less free
than ``GPUWM_HEX_RELEASE_HEADROOM`` (15 %) of its memory, or less free than
the idle set itself (the step's transient: a card that could not hold that
much again gives it back).  A store grown to a frame of 4 KiB or more (WSM6)
follows the same rule: it is kept while the card has the headroom free, and
reset at the first later boundary that finds the card short.  A tight card
(conus3km on 96 GB cards, a co-tenant taking memory) still releases at every
boundary as before; the free reading is the non-synchronising cudaMemGetInfo,
at most a second old, raised by what a release just returned.  Pool requests
the cached blocks cannot serve are safe without a boundary: CuPy's allocator
returns its idle blocks and retries before it reports out of memory.
``GPUWM_HEX_RELEASE_HEADROOM=1`` restores the release at every boundary.

Both savings are on by default (fixed means default).  Inside a CUDA stream
capture, and in a phase nested inside another on the same thread, the bracket
does nothing: the outermost phase owns the boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import os
import sys
import threading
import time
from typing import Any, Callable, Iterator

__all__ = [
    "HEADROOM_KNOB",
    "LMEM_KNOB",
    "LMEM_MIN_FRAME_KNOB",
    "LMEM_WAIT_KNOB",
    "POOL_KNOB",
    "DevicePhaseRelease",
    "phase_release",
    "process_release",
]

POOL_KNOB = "WOOF_HEX_POOL_RELEASE"
LMEM_KNOB = "GPUWM_HEX_LMEM_RELEASE"
LMEM_MIN_FRAME_KNOB = "GPUWM_HEX_LMEM_RELEASE_MIN_FRAME"
LMEM_WAIT_KNOB = "GPUWM_HEX_LMEM_REGROW_WAIT_S"
POOL_MIN_BYTES_KNOB = "GPUWM_HEX_POOL_RELEASE_MIN_BYTES"
HEADROOM_KNOB = "GPUWM_HEX_RELEASE_HEADROOM"
#: Defaults; see "WHAT A BOUNDARY COSTS" above (U-MPASPHYS-1).
LMEM_MIN_FRAME_DEFAULT = 4096
POOL_MIN_BYTES_DEFAULT = 8 << 30
#: See "RETURN ONLY UNDER PRESSURE" above (PROFILE-3KM item 4): idle sets over the keep, and wide stores, are kept
#: while the card has at least this share of its memory free.
HEADROOM_DEFAULT = 0.15
#: A boundary returns idle blocks regardless of the keep when the card's free memory is below this fraction.
POOL_PRESSURE_FREE_FRACTION = 0.10
#: How old that free-memory reading may be (seconds) before a boundary asks the driver again.
POOL_PRESSURE_RECHECK_SECONDS = 1.0
_OFF = frozenset({"0", "off", "false", "no"})
_CUDA_ERROR_MEMORY_ALLOCATION = 2


MEMBER_MEMORY_KNOB = "WOOF_HEX_MEMBER_MEMORY"


def shared_card_memory(environ=None) -> bool:
    """Run policy: single keeps release defaults; shared-card bounds member scratch.

    Launchers select shared-card only when multiple members reside on one card.
    Set before constructing a forecast; existing low-level budget overrides keep
    their precedence. No physics option or arithmetic changes with this policy.
    """
    env = os.environ if environ is None else environ
    value = env.get(MEMBER_MEMORY_KNOB, "single").strip().lower()
    if value not in ("single", "shared-card"):
        raise ValueError(f"{MEMBER_MEMORY_KNOB} must be single or shared-card, got {value!r}")
    return value == "shared-card"


def knob_enabled(name: str, environ: Any = None) -> bool:
    """A saving is on unless its knob says 0/off/false/no."""

    value = (os.environ if environ is None else environ).get(name, "1")
    return str(value).strip().lower() not in _OFF


def _knob_number(name: str, default: float, environ: Any = None) -> float:
    value = (os.environ if environ is None else environ).get(name)
    if value is None or not str(value).strip():
        return default
    number = float(value)
    if number < 0:
        raise ValueError(f"{name}={value!r}: must be zero or more")
    return number


class DevicePhaseRelease:
    """Return cached pool blocks and shrink the local-memory store per phase."""

    def __init__(self, cp: Any = None, *, pool: bool | None = None,
                 lmem: bool | None = None, environ: Any = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._cp = cp
        self.pool = knob_enabled(POOL_KNOB, environ) if pool is None else bool(pool)
        self.lmem = knob_enabled(LMEM_KNOB, environ) if lmem is None else bool(lmem)
        self.min_frame = int(_knob_number(LMEM_MIN_FRAME_KNOB, LMEM_MIN_FRAME_DEFAULT, environ))
        shared = shared_card_memory(environ)
        self.member_memory = "shared-card" if shared else "single"
        self.pool_min_bytes = int(_knob_number(
            POOL_MIN_BYTES_KNOB, 1 << 30 if shared else POOL_MIN_BYTES_DEFAULT, environ))
        # Release gained a pressure-only policy after the memory branch. Shared
        # members need its original phase trimming even on an otherwise empty card.
        self.headroom = _knob_number(HEADROOM_KNOB, 1.0 if shared else HEADROOM_DEFAULT, environ)
        self.regrow_wait_s = _knob_number(LMEM_WAIT_KNOB, 1800.0, environ)
        self._sleep, self._clock = sleep, clock
        # Nesting is tracked per thread: same-host ranks step on their own
        # threads, each on its own card (drivers/samehost_forecast_v841.py),
        # and one rank's open phase must not turn another rank's bracket into
        # a pass-through.  The counters below are shared and only reported.
        self._local = threading.local()
        self._lock = threading.Lock()
        # Widest stack limit each phase grew to, per device: the frame its
        # store is grown back to on the phase's next entry.
        self._widest: dict[tuple[int, str], int] = {}
        # A wide store kept past its phase because the card had room, per
        # device: (the limit to set back when the card turns short, its phase).
        self._held: dict[int, tuple[int, str]] = {}
        # Last free-memory reading per device: (clock, free bytes, total bytes).
        self._pressure: dict[int, tuple[float, int, int]] = {}
        self.lmem_pinned = False
        self.phases = 0
        self.pool_releases = 0
        self.pool_keeps = 0
        self.pool_bytes_returned = 0
        self.lmem_kept = 0
        self.lmem_resets = 0
        self.lmem_regrows = 0
        self.lmem_refused_regrows = 0
        self.lmem_regrow_wait_s = 0.0
        self.lmem_widest_stack_bytes = 0
        self.by_phase: dict[str, dict[str, int]] = {}

    @property
    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @_depth.setter
    def _depth(self, value: int) -> None:
        self._local.depth = value

    def _cupy(self) -> Any:
        if self._cp is None:
            import cupy

            self._cp = cupy
        return self._cp

    def _row(self, name: str) -> dict[str, int]:
        return self.by_phase.setdefault(
            name, {"pool_bytes_returned": 0, "lmem_resets": 0, "lmem_regrows": 0, "lmem_kept": 0})

    def _card_free(self, cp: Any) -> tuple[int, int]:
        """The current card's (free, total) bytes, read at most every ``POOL_PRESSURE_RECHECK_SECONDS``
        per card.

        THE BREAKAGE THIS PREVENTS (hex host-bound, V5090 2026-10-27): every slot boundary under the
        keep asked the driver (cudaMemGetInfo), two per slot, about fourteen per physics step; on a
        shared 8-card host each call took about a millisecond on the thread that issues the card's
        work (py-spy: 4 % of the MPAS-physics LTS class threads' step-loop CPU).  Another process
        needing this card's memory is a condition that lasts seconds, so a reading this fresh still
        returns the idle blocks in time.
        """
        device = int(cp.cuda.runtime.getDevice())
        now = self._clock()
        with self._lock:
            seen = self._pressure.get(device)
        if seen is not None and now - seen[0] < POOL_PRESSURE_RECHECK_SECONDS:
            return seen[1], seen[2]
        free, total = (int(value) for value in cp.cuda.runtime.memGetInfo())
        with self._lock:
            self._pressure[device] = (now, free, total)
        return free, total

    def _credit_free(self, cp: Any, returned: int) -> None:
        """Add bytes this process just gave back to the card's last free reading (no driver call)."""

        device = int(cp.cuda.runtime.getDevice())
        with self._lock:
            seen = self._pressure.get(device)
            if seen is not None:
                self._pressure[device] = (seen[0], min(seen[1] + returned, seen[2]), seen[2])

    def _keep_idle(self, cp: Any, idle: int) -> bool:
        """Whether a boundary keeps ``idle`` cached bytes; "RETURN ONLY UNDER PRESSURE" above."""

        free, total = self._card_free(cp)
        if free < POOL_PRESSURE_FREE_FRACTION * total:
            return False
        if idle < self.pool_min_bytes:
            return True
        return free >= max(self.headroom * total, idle)

    def _keep_store(self, cp: Any) -> bool:
        """Whether a boundary keeps a store grown to a frame of ``min_frame`` bytes or more."""

        free, total = self._card_free(cp)
        return free >= max(self.headroom, POOL_PRESSURE_FREE_FRACTION) * total

    def _return_pool(self, cp: Any, name: str) -> None:
        # The default pool acts on the calling thread's current device.
        pool = cp.get_default_memory_pool()
        held = int(pool.total_bytes())
        idle = held - int(pool.used_bytes())
        if idle == 0:
            return
        if self._keep_idle(cp, idle):
            with self._lock:
                self.pool_keeps += 1
            return
        pool.free_all_blocks()
        returned = held - int(pool.total_bytes())
        if returned:
            self._credit_free(cp, returned)
            with self._lock:
                self.pool_releases += 1
                self.pool_bytes_returned += returned
                self._row(name)["pool_bytes_returned"] += returned

    def _reset_store(self, runtime: Any, device: int, to: int, name: str) -> None:
        runtime.deviceSetLimit(runtime.cudaLimitStackSize, to)
        with self._lock:
            self._held.pop(device, None)
            self.lmem_resets += 1
            self._row(name)["lmem_resets"] += 1

    def _release_held_store(self, cp: Any, runtime: Any, device: int, stack: int) -> int:
        """At a boundary, set a store kept past its phase back once the card is short; returns the limit now."""

        with self._lock:
            held = self._held.get(device)
        if held is None:
            return stack
        restore_to, owner = held
        if stack <= restore_to:
            # Something outside the brackets set the limit down already.
            with self._lock:
                self._held.pop(device, None)
            return stack
        if self._keep_store(cp):
            return stack
        self._reset_store(runtime, device, restore_to, owner)
        return restore_to

    def _regrow(self, cp: Any, runtime: Any, frame: int, name: str) -> None:
        """Grow the store back to ``frame`` bytes per thread before the phase runs."""

        limit = runtime.cudaLimitStackSize
        started = None
        while True:
            try:
                runtime.deviceSetLimit(limit, frame)
            except runtime.CUDARuntimeError as error:
                if getattr(error, "status", None) != _CUDA_ERROR_MEMORY_ALLOCATION:
                    raise
                now = self._clock()
                if started is None:
                    started = now
                    with self._lock:
                        self.lmem_refused_regrows += 1
                    print(f"woof hex: device memory release: the driver refused to regrow the "
                          f"local-memory store of {name} to {frame} bytes per thread (another process "
                          f"on this card took the memory this run gave back); returning this process's "
                          f"cached blocks and waiting up to {self.regrow_wait_s:.0f} s "
                          f"({LMEM_WAIT_KNOB}); the store is then kept for the rest of the run",
                          file=sys.stderr, flush=True)
                    gc.collect()
                    cp.get_default_memory_pool().free_all_blocks()
                    continue
                if now - started >= self.regrow_wait_s:
                    raise RuntimeError(
                        f"device memory release: the local-memory store of {name} could not be "
                        f"regrown to {frame} bytes per thread in {now - started:.0f} s: another process "
                        f"on this card holds the memory this run gave back between phases.  Run "
                        f"alone on the card, or set {LMEM_KNOB}=0 so the store is never given back"
                    ) from error
                self._sleep(2.0)
                continue
            break
        with self._lock:
            self.lmem_regrows += 1
            self._row(name)["lmem_regrows"] += 1
            if started is not None:
                self.lmem_regrow_wait_s += self._clock() - started
                self.lmem_pinned = True

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Bracket one physics phase; see the module docstring."""

        if self._depth or not (self.pool or self.lmem):
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        cp = self._cupy()
        if cp.cuda.get_current_stream().is_capturing():
            # cudaFree and cudaDeviceSetLimit are not legal inside a capture.
            yield
            return
        runtime = cp.cuda.runtime
        lmem = self.lmem and not self.lmem_pinned
        device = int(runtime.getDevice()) if lmem else 0
        stack_at_entry = (int(runtime.deviceGetLimit(runtime.cudaLimitStackSize))
                          if lmem else 0)
        if lmem:
            stack_at_entry = self._release_held_store(cp, runtime, device, stack_at_entry)
        if self.pool:
            self._return_pool(cp, name)
        if lmem:
            frame = self._widest.get((device, name), 0)
            if frame > stack_at_entry:
                self._regrow(cp, runtime, frame, name)
        self._depth += 1
        try:
            yield
        except BaseException:
            # The phase failed: leave the device as it is, so the caller's
            # rollback and the original error are what the run reports.
            self._depth -= 1
            raise
        self._depth -= 1
        with self._lock:
            self.phases += 1
        if lmem:
            # The limit, and the store it sizes, belong to the current device.
            stack = int(runtime.deviceGetLimit(runtime.cudaLimitStackSize))
            if stack > stack_at_entry and stack >= self.min_frame:
                with self._lock:
                    key = (device, name)
                    self._widest[key] = max(self._widest.get(key, 0), stack)
                    self.lmem_widest_stack_bytes = max(self.lmem_widest_stack_bytes, stack)
                    pinned = self.lmem_pinned
                if not pinned:
                    if self._keep_store(cp):
                        # The card has room: keep the store (no synchronising
                        # reset now, no regrow next step); a later boundary
                        # that finds the card short sets it back.
                        with self._lock:
                            self._held.setdefault(device, (stack_at_entry, name))
                            self.lmem_kept += 1
                            self._row(name)["lmem_kept"] += 1
                    else:
                        with self._lock:
                            held = self._held.get(device)
                        self._reset_store(runtime, device, held[0] if held else stack_at_entry, name)
            elif stack < stack_at_entry:
                # Something inside the phase set the limit down below a kept store.
                with self._lock:
                    self._held.pop(device, None)
        if self.pool:
            self._return_pool(cp, name)

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "hex-device-phase-release-v3",
            "member_memory": self.member_memory,
            "headroom": {"fraction": self.headroom, "knob": HEADROOM_KNOB,
                         "pressure_fraction": POOL_PRESSURE_FREE_FRACTION},
            "pool_release": {"enabled": self.pool, "knob": POOL_KNOB,
                             "releases": self.pool_releases,
                             "keeps": self.pool_keeps,
                             "min_idle_bytes": self.pool_min_bytes,
                             "bytes_returned": self.pool_bytes_returned},
            "lmem_release": {"enabled": self.lmem, "knob": LMEM_KNOB,
                             "min_frame_bytes": self.min_frame,
                             "kept": self.lmem_kept,
                             "resets": self.lmem_resets,
                             "regrows": self.lmem_regrows,
                             "refused_regrows": self.lmem_refused_regrows,
                             "regrow_wait_seconds": round(self.lmem_regrow_wait_s, 3),
                             "pinned_after_regrow_wait": self.lmem_pinned,
                             "widest_stack_bytes": self.lmem_widest_stack_bytes},
            "phases": self.phases,
            "by_phase": {k: dict(v) for k, v in sorted(self.by_phase.items())},
        }


_PROCESS: DevicePhaseRelease | None = None
_PROCESS_LOCK = threading.Lock()


def process_release() -> DevicePhaseRelease:
    """The process's one release object; knobs are read on first use."""

    global _PROCESS
    with _PROCESS_LOCK:
        if _PROCESS is None:
            _PROCESS = DevicePhaseRelease()
        return _PROCESS


def phase_release(name: str):
    """``with phase_release("slot:radiation_sw"): ...`` on the process object."""

    return process_release().phase(name)
