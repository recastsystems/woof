"""What an engine offers the bridge, and the few helpers that move arrays between it and PyTorch.

THE INTERFACE.  Arrays are level-major: a per-level field
is C-contiguous (nz, ncol) with row k = model level k from the ground; a surface field is (ncol,).  A state field may
be one column wider than the slot (a limited-area pool's last, unused column); the bridge reads the first ncol.

  xp, dtype            the array module (numpy or cupy) and the slot's float type (float32 on the real engine)
  ncol, nz             slot columns and levels
  slot_fields          the slot's fields, a subset of SLOT_FIELDS
  neighbours()         (ncol, m) int array of each column's neighbours, -1 padded (the multi-scale loss)
  run_forward(window, callback, record=True)
                       the window's forecast.  Before model step k (at boundary k) the engine calls
                       callback(k, state) -> {slot field: array (nz, ncol), units per second} or None; it reads the
                       returned arrays in place during that step only.  state: {name: array} at boundary k (views,
                       read-only).  record=False: a forecast only (no reverse will follow).  An absent field, or None,
                       makes no launch: with no callback the run is the engine's own, byte for byte
  state()              {name: array} at the end of the last run_forward (boundary n)
  reverse_window(g_by_boundary)
                       the whole reverse of the last recorded window.  g_by_boundary: {k: {name: array}}, the
                       cotangents to add into the named state at boundary k (k = n is the end).  Returns the slot's
                       cotangent at every step, [ {slot field: array (nz, ncol)} for k = 0 .. n-1 ]
  reverse_step(g_add), reverse_end(g_add)   optional, a reverse that pauses at each boundary:
                       reverse_step adds g_add at boundary k+1, reverses step k and returns step k's slot cotangent
                       (called for k = n-1 .. 0); reverse_end adds at boundary 0 and returns the state's cotangent
  state_at(k)          optional: the named state at boundary k from the step store (keep="engine")
  times                optional: {"record": s, "replay": s, ...} seconds the engine spent that the bridge cannot see

The slot is added once a step after physics and before the dynamics takes the tendencies, so its derivative is the
identity add: step k's slot cotangent is the cotangent of the physics tendency sum at step k.
"""
from __future__ import annotations

from dataclasses import dataclass

SLOT_FIELDS = ("theta", "qv", "u", "v")


@dataclass(frozen=True)
class WindowSpec:
    """One training window: a case, its start (an index or a time the engine understands), its length in steps."""

    case: int | str
    start: int | str
    length: int


def to_torch(a):
    """An engine array as a torch tensor on the same memory (DLPack; the protocol orders the two streams)."""
    import torch

    return torch.from_dlpack(a)


def to_engine(t, xp):
    """A torch tensor as an engine array on the same memory.  A non-contiguous tensor is made contiguous first."""
    t = t.detach()
    if not t.is_contiguous():
        t = t.contiguous()
    return xp.from_dlpack(t)


def reverse_window(engine, g_by_boundary, n_steps, inject=None):
    """The reverse of a whole window: [slot cotangent per step], step 0 first.

    inject(k, g_slot_k) -> {name: array} or None, called right after step k is reversed, adds into the cotangent at
    boundary k before step k-1 is reversed (how the network's inputs carry their share back).  It needs
    reverse_step; without inject, an engine's reverse_window is used when it has one.
    """
    if inject is None and hasattr(engine, "reverse_window"):
        return list(engine.reverse_window(g_by_boundary))
    out = [None] * n_steps
    pending = None
    for k in range(n_steps - 1, -1, -1):
        out[k] = engine.reverse_step(merge(g_by_boundary.get(k + 1), pending))
        pending = inject(k, out[k]) if inject is not None else None
    engine.reverse_end(merge(g_by_boundary.get(0), pending))
    return out


def merge(a, b):
    """Two {name: array} cotangent sets added name by name (either may be None)."""
    if not a:
        return b or None
    if not b:
        return a
    m = dict(a)
    for name, g in b.items():
        m[name] = m[name] + g if name in m else g
    return m
