"""Device-side guards read once per radiation call instead of once per chunk.

THE BREAKAGE THIS PREVENTS (A3, 2026-10-05): the legacy RRTMG call runs its
columns in chunks sized to the card's resident threads (2,048 LW and 2,560 SW
columns on an RTX 5090), and five guards inside each chunk read a flag back to
the host: the LW and SW cloud-property abort words, the SW day-column
contract, the prep night gate and the McICA seed guard.  Each read waits for
the card to drain, so the host could never queue the next chunk while the card
worked on this one.  Measured on one RTX 5090 at 360,000 columns: 317 chunks,
1,164 device reads and 141 device-wide synchronizations, 0.94 s of a 3.1 s
call spent waiting in them.

Inside :func:`deferred` a guard records its device flag and its verdict and
returns at once; the scope reads every recorded flag in ONE transfer when it
closes and runs the verdicts in the order they were recorded, so the first
failing guard raises the same exception with the same message it raised
before.  Nothing the call returns is visible to the caller before the scope
closes, so a failed guard still means no result is used.  Outside a scope
:func:`check` reads the flag at once, exactly as before.  A guard flag must
be a value nothing writes after the check is recorded (copy reused scratch).
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

_SINK: ContextVar[list | None] = ContextVar("gpuwm_deferred_device_checks",
                                            default=None)


def check(value, verdict: Callable) -> None:
    """Run ``verdict(host_value)`` now, or when the enclosing scope closes.

    ``value`` is a CuPy array (any shape and dtype); ``verdict`` receives its
    host copy and raises when the guard fails.
    """
    sink = _SINK.get()
    if sink is None:
        import cupy as cp
        verdict(cp.asnumpy(value))
        return
    sink.append((value, verdict))


@contextmanager
def deferred():
    """Collect :func:`check` calls in this context; read them all on exit."""
    sink: list = []
    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)
    if not sink:
        return
    import cupy as cp
    import numpy as np
    parent = _SINK.get()
    if parent is not None:
        parent.extend(sink)
        return
    flat = cp.concatenate([cp.ravel(value).astype(cp.float64)
                           for value, _ in sink])
    host = cp.asnumpy(flat)
    offset = 0
    for value, verdict in sink:
        size = int(value.size)
        part = host[offset:offset + size].astype(value.dtype).reshape(value.shape)
        offset += size
        verdict(np.asarray(part))


__all__ = ["check", "deferred"]


@contextmanager
def collecting(sink):
    """Retain immutable guard values until a compiled pass has completed."""
    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)


def drain_collected(sink):
    """Evaluate every captured guard after the pass boundary synchronization."""
    if not sink:
        return
    with deferred():
        for value,verdict in sink:
            check(value,verdict)
    sink.clear()
