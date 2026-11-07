"""How a host thread waits for the card: blocking sync by default.

CUDA's default ("auto") scheduling spins the waiting host thread whenever the
process has fewer active contexts than the host has cores, which is every hex
run.  So a GPU-bound forecast held one host core at 100 % for its whole step
loop while the card did the work: every ``synchronize``, ``asnumpy`` and
``cudaFree`` busy-polled.  Measured (RTX 5070 Ti, hex 0.4.7, the
98,073-cell VR mesh, 1 h at dt 18.75 s, MPAS convection_permitting, GPU at
100 %): the main thread sat at 99-101 % of one core for the whole loop; with
blocking sync it sat at 4-5 % (GPU still 93-100 %), the step-loop wall went
125.7 s -> 127.1 s (driver probe) and 128.6 s (this module), 1-2 % for the
wake-up after each wait, and all 1,050 history/diag variables were
byte-identical.  ``spin`` stays selectable for a timed proof run.  The spinning core is a
single-core stage on the critical path in name and on every box sentinel, and
on shared boxes (co-resident members, same-host ranks, host workers such as
history writes and products) it is a core taken from real work.

This sets the scheduling bits of the CURRENT context (``cuCtxSetFlags``,
driver 12.1+, legal on an active context).  On an older driver it falls back to
``cuDevicePrimaryCtxSetFlags``, which applies only while the primary context is
not yet active; if neither applies the run keeps CUDA's default and says so
once.  Waiting is a host-side choice: kernels, their order and their results
are unchanged.

KNOB
``WOOF_HEX_CUDA_SCHEDULE=blocking`` (default) | ``spin`` | ``yield`` | ``auto``
(``auto`` leaves CUDA's own default untouched).
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from typing import Any

KNOB = "WOOF_HEX_CUDA_SCHEDULE"
DEFAULT = "blocking"
_FLAGS = {"auto": 0x00, "spin": 0x01, "yield": 0x02, "blocking": 0x04}
_SCHED_MASK = 0x07
_CUDA_SUCCESS = 0

_lock = threading.Lock()
_driver: Any = None
_done: dict[Any, str] = {}
_warned = False


def requested(environ: Any = None) -> str:
    raw = (os.environ if environ is None else environ).get(KNOB, DEFAULT)
    mode = str(raw).strip().lower() or DEFAULT
    if mode not in _FLAGS:
        raise ValueError(f"{KNOB}={raw!r}: expected one of {', '.join(sorted(_FLAGS))}")
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


def apply(device_id: int, *, environ: Any = None, driver: Any = None) -> str:
    """Set the wait mode on the context current for ``device_id``; returns how it was applied.

    Called right after the device is made current.  Idempotent per context.
    """

    global _warned
    mode = requested(environ)
    if mode == "auto":
        return "auto"
    from woof.gpu_smi import is_amd
    if driver is None and is_amd():
        from woof.hip_runtime import wait_mode
        how = wait_mode(mode, [int(device_id)]).get(int(device_id), 'no-device')
        with _lock:
            _done[int(device_id)] = how
        return how
    cu = driver if driver is not None else _load_driver()
    if cu is None:
        return "no-driver"
    want = _FLAGS[mode]
    with _lock:
        ctx = ctypes.c_void_p()
        key: Any = int(device_id)
        if cu.cuCtxGetCurrent(ctypes.byref(ctx)) == _CUDA_SUCCESS and ctx.value:
            key = ctx.value
        if key in _done:
            return _done[key]
        how = "unchanged"
        flags = ctypes.c_uint()
        setter = getattr(cu, "cuCtxSetFlags", None)
        if (ctx.value and setter is not None
                and cu.cuCtxGetFlags(ctypes.byref(flags)) == _CUDA_SUCCESS):
            if flags.value & _SCHED_MASK == want:
                how = f"{mode} (already)"
            elif setter(ctypes.c_uint((flags.value & ~_SCHED_MASK) | want)) == _CUDA_SUCCESS:
                how = f"{mode} (context)"
        if how == "unchanged":
            dev = ctypes.c_int()
            if (cu.cuDeviceGet(ctypes.byref(dev), int(device_id)) == _CUDA_SUCCESS
                    and cu.cuDevicePrimaryCtxSetFlags(dev, ctypes.c_uint(want)) == _CUDA_SUCCESS):
                how = f"{mode} (primary context)"
        if how == "unchanged" and not _warned:
            _warned = True
            print(f"woof hex: could not set CUDA wait mode {mode} on device {device_id} "
                  f"(driver lacks cuCtxSetFlags and the context is already active); the host "
                  f"thread keeps CUDA's default wait ({KNOB})", file=sys.stderr, flush=True)
        _done[key] = how
        return how


def applied() -> dict[str, str]:
    """What each context got, for receipts."""

    with _lock:
        return {str(key): how for key, how in _done.items()}
