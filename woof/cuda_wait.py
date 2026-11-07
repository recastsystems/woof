"""How a host thread waits for the card: blocking sync by default.

CUDA's default ("auto") scheduling spins the waiting host thread whenever the
process has fewer active contexts than the host has cores, which is every
WOOF run.  So a GPU-bound forecast held one host core at 100 % for its whole
step loop while the card did the work: every ``synchronize``, ``.get()``,
event wait and ``cudaFree`` busy-polled.  Measured on an RTX 5070 Ti host (
gpuwm 2.9.8, a 3 km GFS nest forecast, card at 100 %): the forecast's main
thread sat at 99.9 % of one core, ``perf`` put 67 % of it in libcuda, 16 % in
the kernel (``sched_yield``) and 5 % in the vDSO clock -- the spin loop.  On
a box running many members or ranks per host (mass ensembles, co-resident
members, DA legs) every one of those spinning cores is a core taken from the
host work that has to run beside the card (history writes, products, the
next member's preparation).  The hex engine fixed the same signature the
same way (woof hex ``cuda_backend/schedule.py``).

This sets the scheduling bits of every ACTIVE primary context in the process
(``cuCtxSetFlags`` on the pushed context, driver 12.1+, legal on an active
context), and of every visible device's primary context that is not active
yet (``cuDevicePrimaryCtxSetFlags``, which applies when the context is
created), so a card the run binds later waits the same way.  It never creates
a context: a device whose primary context is inactive only has its creation
flags set.  Waiting is a host-side choice: kernels, their order, their
launch configuration and their results are unchanged.

KNOB
``WOOF_CUDA_SCHEDULE=blocking`` (default) | ``spin`` | ``yield`` | ``auto``
(``auto`` leaves CUDA's own default untouched; ``spin`` restores the old
behaviour for a timed comparison).
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from typing import Any

KNOB = "WOOF_CUDA_SCHEDULE"
DEFAULT = "blocking"
_FLAGS = {"auto": 0x00, "spin": 0x01, "yield": 0x02, "blocking": 0x04}
_SCHED_MASK = 0x07
_CUDA_SUCCESS = 0

_lock = threading.Lock()
_driver: Any = None
_applied: dict[int, str] = {}
_warned = False


def requested(environ: Any = None) -> str:
    """The wait mode the environment asks for (default ``blocking``)."""

    raw = (os.environ if environ is None else environ).get(KNOB, DEFAULT)
    mode = str(raw).strip().lower() or DEFAULT
    if mode not in _FLAGS:
        raise ValueError(
            f"{KNOB}={raw!r}: expected one of {', '.join(sorted(_FLAGS))}")
    return mode


def _load_driver() -> Any:
    global _driver
    if _driver is None:
        name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
        try:
            _driver = ctypes.CDLL(name)
        except OSError:
            _driver = False
    return _driver or None


def _sym(cu: Any, name: str) -> Any:
    # cuda.h maps these names to their ``_v2`` entry points; the bare
    # symbols are the pre-11 ABI kept for old binaries.
    return getattr(cu, f"{name}_v2", None) or getattr(cu, name)


def _set_one(cu: Any, ordinal: int, want: int) -> str:
    dev = ctypes.c_int()
    if cu.cuDeviceGet(ctypes.byref(dev), int(ordinal)) != _CUDA_SUCCESS:
        return "no-device"
    flags = ctypes.c_uint()
    active = ctypes.c_int()
    if cu.cuDevicePrimaryCtxGetState(
            dev, ctypes.byref(flags), ctypes.byref(active)) != _CUDA_SUCCESS:
        return "unreadable"
    if not active.value:
        # Not created yet: set the flags it will be created with.  This
        # call never creates the context.
        if flags.value & _SCHED_MASK == want:
            return "inactive (already)"
        if _sym(cu, "cuDevicePrimaryCtxSetFlags")(
                dev, ctypes.c_uint((flags.value & ~_SCHED_MASK) | want)) == _CUDA_SUCCESS:
            return "inactive (creation flags)"
        return "unchanged"
    # Active: push it, set its live flags, pop it.  Retain/release keep the
    # reference count balanced; the context already exists, so retain only
    # adds a reference and never creates one.
    setter = getattr(cu, "cuCtxSetFlags", None)
    ctx = ctypes.c_void_p()
    if cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) != _CUDA_SUCCESS:
        return "unchanged"
    how = "unchanged"
    try:
        if _sym(cu, "cuCtxPushCurrent")(ctx) != _CUDA_SUCCESS:
            return how
        try:
            live = ctypes.c_uint()
            if cu.cuCtxGetFlags(ctypes.byref(live)) == _CUDA_SUCCESS:
                if live.value & _SCHED_MASK == want:
                    how = "active (already)"
                elif setter is not None and setter(ctypes.c_uint(
                        (live.value & ~_SCHED_MASK) | want)) == _CUDA_SUCCESS:
                    how = "active (context)"
        finally:
            popped = ctypes.c_void_p()
            _sym(cu, "cuCtxPopCurrent")(ctypes.byref(popped))
    finally:
        _sym(cu, "cuDevicePrimaryCtxRelease")(dev)
    return how


def apply(*, environ: Any = None, driver: Any = None) -> dict[int, str]:
    """Set the requested wait mode on every visible device's primary context.

    Called by a process that is already using CUDA (the forecast step loop,
    the CUDA preparation backend), so ``cuInit`` has run; the call is cheap
    and idempotent (a device already in the requested mode is left alone).
    Returns ``{ordinal: how}`` for receipts.
    """

    global _warned
    mode = requested(environ)
    if mode == "auto":
        return {}
    from woof.gpu_smi import is_amd
    if driver is None and is_amd():
        from woof.hip_runtime import wait_mode
        result = wait_mode(mode)
        with _lock:
            _applied.update(result)
        return result
    cu = driver if driver is not None else _load_driver()
    if cu is None:
        return {}
    want = _FLAGS[mode]
    with _lock:
        if cu.cuInit(0) != _CUDA_SUCCESS:
            return {}
        count = ctypes.c_int()
        if cu.cuDeviceGetCount(ctypes.byref(count)) != _CUDA_SUCCESS:
            return {}
        result = {}
        for ordinal in range(count.value):
            how = _set_one(cu, ordinal, want)
            result[ordinal] = f"{mode} {how}" if how != "unchanged" else how
            _applied[ordinal] = result[ordinal]
        if (not _warned and any(how == "unchanged" for how in result.values())):
            _warned = True
            print(f"woof: could not set CUDA wait mode {mode} on every device "
                  f"({result}); a host thread waiting on such a card keeps "
                  f"CUDA's default spin ({KNOB})", file=sys.stderr, flush=True)
        return result


def applied() -> dict[str, str]:
    """What each device got, for receipts."""

    with _lock:
        return {str(key): how for key, how in _applied.items()}
