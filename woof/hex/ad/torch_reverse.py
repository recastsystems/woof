# SPDX-License-Identifier: Apache-2.0
"""The reverse side of the torch bridge: the runner's per-step slot cotangent, the state cotangent in, and the window cut
at step boundaries, through the libhexad-phys C ABI (rust/hexad, ``hexad_phys.h``).

The runner never sees PyTorch. Arrays cross on the card without copies: a device address from the runner is wrapped
as a CuPy array (``cupy.cuda.UnownedMemory``) and handed to PyTorch with DLPack; PyTorch tensors go the other way
through ``__cuda_array_interface__`` / DLPack. PyTorch (BSD) is an optional dependency: only the autograd functions at
the bottom import it, and nothing here is copied from it.

Entry points (a ``Reverse`` over an open session, ``hexad_open`` with ``{"context": "primary"}`` so CuPy, PyTorch and
the runner share the card's primary context):

* ``set_tap(fn, taps=None)``: ``fn(step, function, argument, view)`` at every reversed launch named in ``taps``
  (``"function:arg,.."``; None = the physics tendency sum, ``torch_bridge::SLOT_DEFAULT``), before the launch's
  adjoint entries; ``view`` is a float32 CuPy array over the launch's cotangent view, dJ/d(the argument as the launch
  left it), valid only during the call. Copy it (a per-step card buffer) or add into it (a cotangent at the slot).
* ``init(seed)`` / ``init_at(step, seed)``: start the reverse at the window end / at boundary ``step``; ``seed`` is
  None (zero), a dict of named state fields (the loss's gradient on the card: theta, rho, u, w and species, see
  ``hexad_field``) or a whole cotangent arena from ``arena()``.
* ``put(fields)``: add a cotangent on named state fields at the current boundary (the network reads the state there).
* ``step()``: reverse one step. ``arena(out=None)``: the whole cotangent arena at the current boundary (the state
  cotangent a piece hands to the next). ``get(fields)``: at boundary 0, add T0^T(arena) into the caller's fields.
* ``shadow(alloc, offset, nbytes)``: the float32 shadow of a plan allocation (e.g. a recorded external-tendency
  table's slice k once the reverse has passed step k: the slot cotangent of form (b) of the contract).

``reverse_window`` is form v0 (one call for the whole window); ``reverse_pieces`` is form v1 (pieces [k1, k0) chained
through the arena, so PyTorch can chain the steps and the network's input dependence flows back exactly).
"""
from __future__ import annotations

import ctypes as C
from typing import Callable, Mapping, Optional, Sequence

from woof.hex import hexad

VEC_ZERO, VEC_INCREMENT, VEC_SEED, VEC_ARENA = 0, 1, 2, 3
F64, F32 = 0, 1
TAP_FN = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_int, C.c_char_p, C.c_char_p, C.c_uint64, C.c_uint64)


class HexadField(C.Structure):
    _fields_ = [("name", C.c_char_p), ("dptr", C.c_uint64), ("dtype", C.c_int32), ("nlev", C.c_int32),
                ("nhoriz", C.c_int64), ("lev_stride", C.c_int64), ("horiz_stride", C.c_int64)]


class HexadVec(C.Structure):
    _fields_ = [("kind", C.c_int32), ("nfields", C.c_int32), ("fields", C.POINTER(HexadField)),
                ("seed", C.c_uint64), ("arena", C.c_uint64), ("words", C.c_uint64)]


class ReverseError(RuntimeError):
    """The runner refused or failed (its own line, hexad_last_error)."""


def _lib(path=None):
    lib = C.CDLL(str(path or hexad.library()))
    sig = {
        "hexad_last_error": ([], C.c_char_p),
        "hexad_steps": ([C.c_void_p], C.c_int),
        "hexad_arena_words": ([C.c_void_p], C.c_uint64),
        "hexad_ad_init": ([C.c_void_p, C.POINTER(HexadVec)], C.c_int),
        "hexad_ad_init_at": ([C.c_void_p, C.c_int, C.POINTER(HexadVec)], C.c_int),
        "hexad_ad_put": ([C.c_void_p, C.POINTER(HexadVec)], C.c_int),
        "hexad_ad_step": ([C.c_void_p], C.c_int),
        "hexad_ad_get": ([C.c_void_p, C.POINTER(HexadVec)], C.c_int),
        "hexad_position": ([C.c_void_p], C.c_int),
        "hexad_shadow_ptr": ([C.c_void_p, C.c_uint64, C.c_uint64, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)], C.c_int),
        "hexad_set_tap": ([TAP_FN, C.c_void_p, C.c_char_p], C.c_int),
    }
    for name, (args, res) in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, res
    return lib


def _f32_view(cp, dptr: int, nbytes: int, owner=None):
    mem = cp.cuda.UnownedMemory(int(dptr), int(nbytes), owner)
    return cp.ndarray((int(nbytes) // 4,), dtype=cp.float32, memptr=cp.cuda.MemoryPointer(mem, 0))


def fields_vec(fields: Mapping[str, object], kind=VEC_INCREMENT):
    """A hexad_vec over named device arrays in the plan layout (level-major (nlev, nhoriz), C-contiguous)."""
    rows, keep = [], []
    for name, a in fields.items():
        ci = a.__cuda_array_interface__
        shape, typestr = ci["shape"], ci["typestr"]
        if len(shape) != 2 or ci.get("strides") not in (None, (shape[1] * int(typestr[-1]), int(typestr[-1]))):
            raise ValueError(f"{name}: want a C-contiguous (nlev, nhoriz) array, got shape {shape} strides {ci.get('strides')}")
        dt = {"<f4": F32, "<f8": F64}.get(typestr)
        if dt is None:
            raise ValueError(f"{name}: float32 or float64, not {typestr}")
        b = name.encode()
        keep.append(b)
        rows.append(HexadField(b, ci["data"][0], dt, shape[0], shape[1], shape[1], 1))
    arr = (HexadField * len(rows))(*rows)
    keep.append(arr)
    v = HexadVec(kind, len(rows), C.cast(arr, C.POINTER(HexadField)), 0, 0, 0)
    v._keep = keep
    return v


class Reverse:
    """The reverse of one open session (``session``: the hexad_session* from hexad_open, as an int)."""

    def __init__(self, session: int, lib=None):
        import cupy as cp
        self.cp, self.s, self.lib = cp, C.c_void_p(session), lib or _lib()
        self.words = int(self.lib.hexad_arena_words(self.s))
        self.steps = int(self.lib.hexad_steps(self.s))
        self._tap = None

    def _ok(self, rc: int, what: str):
        if rc != 0:
            raise ReverseError(f"{what}: {(self.lib.hexad_last_error() or b'').decode()}")

    def _vec(self, seed):
        if seed is None:
            return HexadVec(VEC_ZERO, 0, None, 0, 0, 0)
        if hasattr(seed, "__cuda_array_interface__"):
            n = seed.size if hasattr(seed, "size") else seed.numel()
            if n != self.words:
                raise ValueError(f"an arena seed has {self.words} words, not {n}")
            return HexadVec(VEC_ARENA, 0, None, 0, seed.__cuda_array_interface__["data"][0], self.words)
        return fields_vec(seed)

    def set_tap(self, fn: Optional[Callable], taps: Optional[str] = None):
        if fn is None:
            self._ok(self.lib.hexad_set_tap(TAP_FN(), None, None), "set_tap")
            self._tap = None
            return
        cp = self.cp

        def call(_user, step, function, argument, dptr, nbytes):
            try:
                fn(int(step), function.decode(), argument.decode(), _f32_view(cp, dptr, nbytes))
                return 0
            except Exception:  # the runner stops the reverse and names the step
                return 1
        self._tap = TAP_FN(call)
        self._ok(self.lib.hexad_set_tap(self._tap, None, taps.encode() if taps else None), "set_tap")

    def init(self, seed=None):
        v = self._vec(seed)
        self._ok(self.lib.hexad_ad_init(self.s, C.byref(v)), "ad_init")

    def init_at(self, step: int, seed=None):
        v = self._vec(seed)
        self._ok(self.lib.hexad_ad_init_at(self.s, int(step), C.byref(v)), "ad_init_at")

    def put(self, fields: Mapping[str, object]):
        v = fields_vec(fields)
        self._ok(self.lib.hexad_ad_put(self.s, C.byref(v)), "ad_put")

    def step(self):
        self._ok(self.lib.hexad_ad_step(self.s), "ad_step")

    def position(self) -> int:
        return int(self.lib.hexad_position(self.s))

    def arena(self, out=None):
        out = self.cp.empty(self.words, self.cp.float32) if out is None else out
        v = HexadVec(VEC_ARENA, 0, None, 0, out.data.ptr, self.words)
        self._ok(self.lib.hexad_ad_get(self.s, C.byref(v)), "ad_get(arena)")
        return out

    def get(self, fields: Mapping[str, object]):
        v = fields_vec(fields)
        self._ok(self.lib.hexad_ad_get(self.s, C.byref(v)), "ad_get")

    def shadow(self, alloc: int, offset: int = 0, nbytes: Optional[int] = None):
        p, n = C.c_uint64(), C.c_uint64()
        self._ok(self.lib.hexad_shadow_ptr(self.s, int(alloc), int(offset), C.byref(p), C.byref(n)), "shadow_ptr")
        return _f32_view(self.cp, p.value, n.value if nbytes is None else min(int(nbytes), n.value))


def reverse_window(rev: Reverse, final_seed, state_cotangent: Optional[Callable[[int], Optional[Mapping]]] = None,
                   taps: Optional[str] = None):
    """Form v0: the whole window in one reverse. ``state_cotangent(k)`` may return named fields to add at boundary k
    (the network's input dependence, taken as given: exact while the network is zero). Returns {step: {name: copy}}
    of the slot cotangents and the arena at boundary 0."""
    out: dict = {}

    def keep(step, function, argument, view):
        out.setdefault(step, {})[f"{function}:{argument}"] = view.copy()
    rev.set_tap(keep, taps)
    try:
        rev.init(final_seed)
        for k in range(rev.steps, 0, -1):
            if state_cotangent is not None and k < rev.steps:
                add = state_cotangent(k)
                if add:
                    rev.put(add)
            rev.step()
        if state_cotangent is not None:
            add = state_cotangent(0)
            if add:
                rev.put(add)
        return out, rev.arena()
    finally:
        rev.set_tap(None)


def reverse_pieces(rev: Reverse, final_seed, cuts: Sequence[int], taps: Optional[str] = None):
    """Form v1: the reverse cut at the boundaries ``cuts`` (descending, inside (0, steps)); each piece [k1, k0) starts
    from the arena the previous one handed out. Returns the same as reverse_window (no state cotangent: PyTorch adds
    it between pieces with ``init_at(k0, arena + T^T(...))``)."""
    out: dict = {}

    def keep(step, function, argument, view):
        out.setdefault(step, {})[f"{function}:{argument}"] = view.copy()
    rev.set_tap(keep, taps)
    try:
        bounds = [rev.steps, *sorted(set(int(c) for c in cuts), reverse=True), 0]
        arena = None
        for i, (k1, k0) in enumerate(zip(bounds, bounds[1:])):
            if i == 0:
                rev.init(final_seed)
            else:
                rev.init_at(k1, arena)
            for _ in range(k1 - k0):
                rev.step()
            arena = rev.arena()
        return out, arena
    finally:
        rev.set_tap(None)
