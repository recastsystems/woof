"""Moving bytes between a full-domain store and a halo tile.

This is the transport layer of the out-of-core tiling prototype.  It owns two
operations and nothing else::

    gather_tile(store, tile_state, spec, stream=None)   # domain -> tile
    scatter_tile(tile_state, store, spec, stream=None)  # tile interior -> domain

``store`` holds the full domain's persisted inventory -- the names in
``woof.state_serialization_contract.STATE_SERIALIZED_ATTRS`` (47 as of the
mp9/P3 restart fixes; ``test_gather`` pins the exact count) -- as arrays that
may live in PINNED host RAM (the out-of-core case) or on the device (the
in-core gate).  ``tile_state`` is a ``DomainState`` built on the tile config.
``spec`` is a ``TileSpec`` from ``tilestream.spec`` (see "Spec contract").

Why cudaMemcpy3D and not slice assignment
-----------------------------------------
A gather writes an ``(nz, ty, tx)`` block into a tile array from an
``(nz, ny, nx)`` domain array.  Neither side is contiguous: rows are ``tx``
elements long with a source pitch of ``nx`` and a destination pitch of ``tx``,
and planes are strided again.  The options were:

* ``cupy`` slice assignment -- device-to-device only.  Cannot touch host RAM
  without CuPy staging a contiguous temporary, which allocates per call and
  puts a pageable buffer (15 GB/s) in the middle of a 57 GB/s path.
* one ``cudaMemcpy2DAsync`` per k-plane -- correct, but ``nz`` launches per
  field per segment.  At nz=49 and 9 live fields that is 441 launches for one
  tile whose payload only takes ~0.8 ms to move; launch overhead would be the
  same order as the transfer.
* ``cudaMemcpy3DAsync`` -- ONE call per (field, segment), handling row pitch,
  plane pitch and a 3-D source/destination offset natively.  ``cupy`` exposes
  it (``cupy.cuda.runtime.memcpy3DAsync``) but only as a raw pointer to a
  ``cudaMemcpy3DParms``, so the struct is declared here in ctypes.  The layout
  (sizeof 160, field offsets 0/8/32/64/72/96/128/152) is asserted at import
  and is exercised bitwise by ``test_gather.py``; a layout error cannot pass
  silently.

The same call serves all four directions (H2D, D2H, D2D, H2H), so gather and
scatter are one code path with the ``kind`` flipped, and an on-device store
(the bit-exact gate) and a pinned-host store (the real pipeline) differ only
in a pointer type.

Staggering is derived FROM SHAPE
--------------------------------
Nothing here hard-codes which field is which.  :func:`classify` compares an
array's trailing two axes against the domain's ``(ny, nx)`` and its leading
axis against ``nz``:

    (.., ny,   nx  ) and lead nz    -> 'mass'   (thp, p, al, qv, ... and 2-D mup)
    (.., ny,   nx+1)                -> 'u'
    (.., ny+1, nx  )                -> 'v'
    (nz+1, ny, nx  )                -> 'w'      (w, php)

so a field added to ``STATE_SERIALIZED_ATTRS`` tomorrow is handled the day it
appears, and a field whose shape does not match ANY stagger raises instead of
being copied wrong.  2-D ``(ny, nx)`` surface fields (``mup``, ``nwfa2d``,
``nifa2d``) fall out of the same rule with a depth of 1.

Spec contract
-------------
``tilestream.spec.TileSpec`` is written by another agent and owns ALL index
arithmetic; this module does no geometry of its own.  What is consumed is:
for each stagger kind in ``('mass', 'u', 'v', 'w')``, the rectangles to copy,
separately for gather and for scatter.  ``tilestream.spec`` spells that
``spec.gather(variant) -> tuple[Transfer, ...]``, where ``Transfer`` names its
two sides ``full``/``tile`` rather than src/dst -- better, because it says
which array each slice indexes instead of leaving it to the direction.
:func:`segments_for` also accepts mapping attributes and already-oriented
``(src_slices, dst_slices)`` pairs, so a second spec implementation with a
different spelling still works; the test file exercises both.

ONLY THE TRAILING TWO AXES of a spec slice tuple are used.  A tile always
takes the full vertical column, and the vertical extent differs per field
(``nz`` for mass, ``nz+1`` for w/phi, absent for 2-D surface fields), so a
single horizontal segment list has to serve all of them.  Any leading entries
in a spec tuple must be full slices (``slice(None)``) or ``Ellipsis`` and are
ignored; a partial vertical slice raises rather than being silently dropped.

Two structural checks run at plan time, independent of which spec is in use
(:func:`_check_destination_rects`): destination rectangles must be pairwise
DISJOINT, and a GATHER's must cover the tile array exactly.  The second is
the one that matters -- a tile array whose horizontal shape does not match
the plan leaves cells holding stale halo values, every rectangle is still in
bounds, and the run produces a plausible forecast instead of an error.

Ordering
--------
Nothing here synchronizes.  Copies are issued on the caller's stream and
return immediately.  A ``non_blocking=True`` stream does NOT synchronize with
the default stream, so anything that wrote the store or the tile buffers on
another stream must be fenced with an event (or a device sync) before the
gather.  This bit the test suite before it bit the pipeline: see
``test_gather._sync``.

Pinning
-------
Host arrays are REQUIRED to be pinned.  Pageable host memory measures 15 GB/s
against 57 GB/s pinned on this machine, and it is also not truly asynchronous,
so a pageable buffer would quietly destroy both the bandwidth and the
copy/compute overlap the whole design depends on.  Every host pointer is
checked with ``cudaPointerGetAttributes`` when a plan is built and
:class:`PageableHostMemoryError` is raised if it is not registered.  Use
:func:`pinned_empty` / :func:`make_host_store` to allocate.

Allocation
----------
No device or host memory is allocated per call.  A :class:`TilePlan` resolves
the spec, classifies every field, validates every extent and preallocates the
single ``cudaMemcpy3DParms`` struct it mutates in place; plans are cached on
the spec object keyed by a structural signature of the two inventories.  Call
:func:`make_plan` once and reuse the plan in a hot loop to skip even the
signature build.
"""

from __future__ import annotations

import ctypes
import math
from typing import Any, Mapping, Sequence

import numpy as np


KINDS: tuple[str, ...] = ("mass", "u", "v", "w")

#: Stagger kinds that fall back to 'mass' segments when a spec omits them.
#: 'w' is horizontally a mass point -- it differs only in vertical extent, and
#: this module never takes a vertical slice from the spec -- so a spec that
#: only defines 'mass' is still complete.
_KIND_FALLBACK: dict[str, str] = {"w": "mass"}


class PageableHostMemoryError(RuntimeError):
    """A host array in the transfer path is not page-locked."""


class SpecError(ValueError):
    """The TileSpec could not be interpreted, or disagrees with the arrays."""


# --------------------------------------------------------------------------
# cudaMemcpy3DParms
# --------------------------------------------------------------------------

class _Pos(ctypes.Structure):
    _fields_ = [("x", ctypes.c_size_t),      # BYTES, not elements
                ("y", ctypes.c_size_t),      # rows
                ("z", ctypes.c_size_t)]      # planes


class _PitchedPtr(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p),
                ("pitch", ctypes.c_size_t),  # row pitch in BYTES
                ("xsize", ctypes.c_size_t),  # logical width  (elements)
                ("ysize", ctypes.c_size_t)]  # logical height (rows); the
                                             # plane stride is pitch * ysize


class _Extent(ctypes.Structure):
    _fields_ = [("width", ctypes.c_size_t),  # BYTES
                ("height", ctypes.c_size_t),
                ("depth", ctypes.c_size_t)]


class _Memcpy3DParms(ctypes.Structure):
    _fields_ = [("srcArray", ctypes.c_void_p),
                ("srcPos", _Pos),
                ("srcPtr", _PitchedPtr),
                ("dstArray", ctypes.c_void_p),
                ("dstPos", _Pos),
                ("dstPtr", _PitchedPtr),
                ("extent", _Extent),
                ("kind", ctypes.c_int)]


# Guards against a silently different ABI.  These offsets are the x86-64
# layout of CUDA's cudaMemcpy3DParms and have been stable since CUDA 3.
assert ctypes.sizeof(_Memcpy3DParms) == 160, ctypes.sizeof(_Memcpy3DParms)
assert _Memcpy3DParms.srcPos.offset == 8
assert _Memcpy3DParms.srcPtr.offset == 32
assert _Memcpy3DParms.dstArray.offset == 64
assert _Memcpy3DParms.dstPos.offset == 72
assert _Memcpy3DParms.dstPtr.offset == 96
assert _Memcpy3DParms.extent.offset == 128
assert _Memcpy3DParms.kind.offset == 152


# --------------------------------------------------------------------------
# memory classification
# --------------------------------------------------------------------------

_CUDA_MEMORY_TYPE_UNREGISTERED = 0
_CUDA_MEMORY_TYPE_HOST = 1
_CUDA_MEMORY_TYPE_DEVICE = 2
_CUDA_MEMORY_TYPE_MANAGED = 3


def _runtime():
    from cupy.cuda import runtime
    return runtime


def is_device_array(array) -> bool:
    """True for a ``cupy.ndarray`` (device-resident)."""
    import cupy as cp
    return isinstance(array, cp.ndarray)


def array_pointer(array) -> int:
    """Base device/host pointer of a cupy or numpy array."""
    if is_device_array(array):
        return int(array.data.ptr)
    return int(np.asarray(array).ctypes.data)


def is_pinned(array) -> bool:
    """True iff ``array`` is host memory that CUDA has page-locked.

    Device arrays return False (they are not host memory at all); use
    :func:`is_device_array` to distinguish.  Managed memory also returns
    False -- it is not the fast path this module wants and silently accepting
    it would hide a footgun.
    """
    if is_device_array(array):
        return False
    try:
        attrs = _runtime().pointerGetAttributes(array_pointer(array))
    except Exception:
        return False
    return int(attrs.type) == _CUDA_MEMORY_TYPE_HOST


def _memcpy_kind(src_is_device: bool, dst_is_device: bool) -> int:
    rt = _runtime()
    if src_is_device and dst_is_device:
        return rt.memcpyDeviceToDevice
    if src_is_device:
        return rt.memcpyDeviceToHost
    if dst_is_device:
        return rt.memcpyHostToDevice
    return rt.memcpyHostToHost


# --------------------------------------------------------------------------
# pinned host allocation
# --------------------------------------------------------------------------

def pinned_empty(shape, dtype) -> np.ndarray:
    """An uninitialised page-locked ``numpy`` array of ``shape``/``dtype``.

    The returned array's ``.base`` chain keeps the CuPy pinned-memory object
    alive, so the caller only has to hold the array itself.
    """
    import cupy as cp

    if isinstance(shape, (int, np.integer)):
        shape = (int(shape),)
    else:
        shape = tuple(int(s) for s in shape)
    dtype = np.dtype(dtype)
    count = 1
    for s in shape:
        count *= s
    mem = cp.cuda.alloc_pinned_memory(max(count * dtype.itemsize, 1))
    out = np.frombuffer(mem, dtype=dtype, count=count).reshape(shape)
    assert is_pinned(out), "cupy.alloc_pinned_memory returned unpinned memory"
    return out


def pinned_empty_like(array) -> np.ndarray:
    """A pinned host array with ``array``'s shape and dtype."""
    return pinned_empty(tuple(array.shape), array.dtype)


def pinned_copy(array) -> np.ndarray:
    """A pinned host copy of a cupy or numpy array (synchronous)."""
    import cupy as cp

    out = pinned_empty_like(array)
    if is_device_array(array):
        with array.device:
            out[...] = cp.asnumpy(array)
    else:
        out[...] = np.asarray(array)
    return out


class HostStore:
    """A pinned host mirror of a state's persisted inventory.

    Exposes each held field as an attribute (``store.thp``) so it satisfies
    the "``store`` exposes the ``STATE_SERIALIZED_ATTRS``" contract, and
    also behaves as a read-only mapping.  Names the configuration did not
    allocate are simply absent -- that is the serialization contract's own
    convention (``getattr(state, name, None) is None``), and
    :func:`inventory` follows it on both ends.
    """

    __slots__ = ("_arrays",)

    def __init__(self, arrays: Mapping[str, np.ndarray]):
        self._arrays = dict(arrays)
        for name, arr in self._arrays.items():
            if not is_pinned(arr):
                raise PageableHostMemoryError(
                    f"HostStore field {name!r} is not page-locked")

    def __getattr__(self, name: str):
        try:
            return self.__getattribute__("_arrays")[name]
        except KeyError:
            raise AttributeError(name) from None

    def __contains__(self, name: str) -> bool:
        return name in self._arrays

    def __getitem__(self, name: str):
        return self._arrays[name]

    def __len__(self) -> int:
        return len(self._arrays)

    def __iter__(self):
        return iter(self._arrays)

    def keys(self):
        return self._arrays.keys()

    def items(self):
        return self._arrays.items()

    def values(self):
        return self._arrays.values()

    def as_dict(self) -> dict[str, np.ndarray]:
        return dict(self._arrays)

    def nbytes(self) -> int:
        return sum(int(a.nbytes) for a in self._arrays.values())


def make_host_store(source, *, copy: bool = True) -> HostStore:
    """Build a pinned :class:`HostStore` mirroring ``source``'s inventory.

    ``source`` may be a ``DomainState``, a :class:`HostStore`, or a mapping.
    With ``copy=True`` the current values are transferred in (synchronously);
    with ``copy=False`` the buffers are left uninitialised, which is what a
    write-only destination wants.
    """
    arrays = inventory(source)
    out: dict[str, np.ndarray] = {}
    for name, arr in arrays.items():
        out[name] = pinned_copy(arr) if copy else pinned_empty_like(arr)
    return HostStore(out)


# --------------------------------------------------------------------------
# inventory and staggering
# --------------------------------------------------------------------------

def serialized_attrs() -> tuple[str, ...]:
    """The enforced persist list, imported (never copied)."""
    from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS
    return tuple(STATE_SERIALIZED_ATTRS)


def _is_array(value) -> bool:
    import cupy as cp
    return isinstance(value, (cp.ndarray, np.ndarray))


def inventory(obj, names: Sequence[str] | None = None) -> dict[str, Any]:
    """``{name: array}`` over ``STATE_SERIALIZED_ATTRS``, in contract order.

    Works on a ``DomainState``, a :class:`HostStore` or a plain mapping.
    Names that are missing or ``None`` are omitted, which is exactly how the
    serialization contract itself treats an unallocated species -- so a dry
    milestone-one state yields 9 entries and a Thompson-aerosol LES state
    yields far more, with no change here.
    """
    if names is None:
        names = serialized_attrs()
    out: dict[str, Any] = {}
    is_mapping = isinstance(obj, Mapping) or isinstance(obj, HostStore)
    for name in names:
        if is_mapping:
            value = obj[name] if name in obj else None
        else:
            value = getattr(obj, name, None)
        if value is not None and _is_array(value):
            out[name] = value
    return out


def domain_extents(arrays: Mapping[str, Any], *,
                   nz: int | None = None) -> tuple[int, int, int]:
    """Infer ``(nz, ny, nx)`` mass-point extents from an inventory.

    Every staggered variant is >= the mass extent on its staggered axis and
    equal on the others (``u`` adds one in x, ``v`` one in y, ``w``/``phi``
    one in z), and nothing in the inventory is ever SMALLER than a mass
    point.  So the horizontal mass extents are the per-axis minima, and that
    is true whatever a leading axis means.

    The VERTICAL extent is only inferable while every leading axis really is
    vertical.  A physics carrier set breaks that: Noah's soil columns are
    ``(4, ny, nx)``, Noah-MP's snow layers ``(3, ny, nx)`` and its combined
    snow-soil coordinate ``(7, ny, nx)``.  The old rule ``nz = min(leading)``
    then returned ``nz = 3`` for a 49-level domain -- silently, which is the
    worst possible failure for a routine whose answer feeds
    :func:`classify`.  So: pass ``nz`` when the inventory holds layered
    fields, and when it is not passed this now RAISES on any leading extent
    that is neither ``nz`` nor ``nz+1`` instead of inventing one.
    """
    horiz = [a for a in arrays.values() if a.ndim >= 2]
    if not horiz:
        raise SpecError(
            "cannot infer domain extents: inventory has no array with a "
            "horizontal extent")
    ny = min(int(a.shape[-2]) for a in arrays.values() if a.ndim >= 2)
    nx = min(int(a.shape[-1]) for a in arrays.values() if a.ndim >= 2)
    if nz is not None:
        return int(nz), ny, nx
    lead = [int(a.shape[0]) for a in arrays.values() if a.ndim >= 3]
    nz = min(lead) if lead else 0
    stray = sorted({v for v in lead if v not in (nz, nz + 1)})
    if stray:
        raise SpecError(
            f"cannot infer the vertical extent: leading extents {stray} are "
            f"neither nz={nz} nor nz+1={nz + 1}. This inventory holds fields "
            "whose leading axis is not vertical (soil/snow layers); pass "
            "nz=cfg.nz explicitly rather than letting the minimum leading "
            f"extent ({nz}) stand in for it.")
    return nz, ny, nx


def classify(shape: Sequence[int], nz: int, ny: int, nx: int, *,
             layers_ok: bool = False) -> str:
    """The stagger kind of an array of ``shape`` on an ``(nz, ny, nx)`` grid.

    Raises :class:`SpecError` for a shape that matches no WRF-ARW stagger --
    deliberately, because the alternative to raising is copying the wrong
    bytes into a field nobody is watching.

    ``layers_ok`` admits a leading axis that is NOT vertical: the physics
    carrier set contains ``(4, ny, nx)`` soil columns, ``(3, ny, nx)`` snow
    layers and a ``(7, ny, nx)`` snow-soil coordinate.  Those are ordinary
    horizontal windows -- the transport carries the whole leading axis
    through as the copy's ``depth`` and ``tile_shape_for`` keeps it verbatim,
    so the block copy is identical to a mass field's -- and they classify as
    ``"mass"``.  It is opt-in because with it off, a leading extent that is
    neither ``nz`` nor ``nz+1`` is a genuine shape bug and must still raise.
    Note ``"w"`` and ``"mass"`` differ ONLY in vertical extent: both have
    stagger ``(0, 0)``, so which one a layered field gets cannot change a
    single copied byte.
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) < 2:
        raise SpecError(f"array of shape {shape} has no horizontal extent")
    h, w = shape[-2], shape[-1]

    if w == nx:
        x_face = False
    elif w == nx + 1:
        x_face = True
    else:
        raise SpecError(
            f"shape {shape}: x extent {w} is neither nx={nx} nor nx+1={nx + 1}")

    if h == ny:
        y_face = False
    elif h == ny + 1:
        y_face = True
    else:
        raise SpecError(
            f"shape {shape}: y extent {h} is neither ny={ny} nor ny+1={ny + 1}")

    if x_face and y_face:
        raise SpecError(
            f"shape {shape} is staggered in BOTH x and y; no ARW field is")
    if x_face:
        return "u"
    if y_face:
        return "v"

    if len(shape) >= 3:
        if shape[0] == nz + 1:
            return "w"
        if shape[0] != nz and not layers_ok:
            raise SpecError(
                f"shape {shape}: z extent {shape[0]} is neither nz={nz} "
                f"nor nz+1={nz + 1}")
    return "mass"


def tile_shape_for(kind: str, domain_shape: Sequence[int],
                   tile_ny: int, tile_nx: int) -> tuple[int, ...]:
    """The shape the tile array of ``kind`` must have.

    Vertical extent is carried through from the domain array unchanged (a
    tile is a horizontal window; it never subsets z).
    """
    lead = tuple(int(s) for s in domain_shape[:-2])
    h = int(tile_ny) + (1 if kind == "v" else 0)
    w = int(tile_nx) + (1 if kind == "u" else 0)
    return lead + (h, w)


# --------------------------------------------------------------------------
# spec interpretation
# --------------------------------------------------------------------------

def _looks_like_pair(obj) -> bool:
    """True for a single ``(src_tuple, dst_tuple)`` rather than a list."""
    if not isinstance(obj, (tuple, list)) or len(obj) != 2:
        return False
    return all(isinstance(half, (tuple, list))
               and all(isinstance(e, slice) or e is Ellipsis
                       or isinstance(e, (int, np.integer))
                       for e in half)
               and len(half) > 0
               for half in obj)


def _full_tile_keys(entry):
    """``(full_slices, tile_slices)`` if ``entry`` is a full/tile Transfer.

    ``tilestream.spec.Transfer`` names its two sides FULL-domain and TILE
    rather than src/dst -- which is strictly better, because it says which
    array each slice indexes instead of leaving it to be inferred from the
    direction.  Both spellings are accepted: ``.full_key`` / ``.tile_key``
    (already ``(Ellipsis, y, x)`` tuples) or the raw
    ``.full_y/.full_x/.tile_y/.tile_x`` fields.  Returns ``None`` when the
    entry is not of this form.
    """
    full = getattr(entry, "full_key", None)
    tile = getattr(entry, "tile_key", None)
    if full is not None and tile is not None:
        return tuple(full), tuple(tile)
    fy = getattr(entry, "full_y", None)
    fx = getattr(entry, "full_x", None)
    ty = getattr(entry, "tile_y", None)
    tx = getattr(entry, "tile_x", None)
    if None not in (fy, fx, ty, tx):
        return (Ellipsis, fy, fx), (Ellipsis, ty, tx)
    return None


def _as_segment_list(value, direction: str) -> list[tuple[tuple, tuple]]:
    """Normalise a spec's per-kind segments to ``[(src_slices, dst_slices)]``.

    A full/tile ``Transfer`` is oriented by ``direction``: a gather reads the
    full domain and writes the tile, a scatter does the reverse.  A bare
    ``(src, dst)`` pair is taken as already oriented.
    """
    if value is None:
        return []
    ft = _full_tile_keys(value)
    if ft is not None:
        value = [value]
    elif _looks_like_pair(value):
        return [(tuple(value[0]), tuple(value[1]))]
    out: list[tuple[tuple, tuple]] = []
    for entry in value:
        ft = _full_tile_keys(entry)
        if ft is not None:
            full, tile = ft
            out.append((full, tile) if direction == "gather" else (tile, full))
            continue
        if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
            raise SpecError(
                f"segment {entry!r} is neither a (src_slices, dst_slices) pair "
                "nor a full/tile Transfer")
        out.append((tuple(entry[0]), tuple(entry[1])))
    return out


def _lookup(container, kind: str):
    if container is None:
        return None
    if callable(container) and not isinstance(container, (Mapping, list, tuple)):
        try:
            return container(kind)
        except (KeyError, TypeError, ValueError):
            # e.g. spec.stagger() raising ValueError on an unknown variant --
            # treat as "this spec does not define that kind" and let the
            # caller's fallback chain decide.
            return None
    try:
        if kind in container:
            return container[kind]
    except TypeError:
        return None
    return None


def segments_for(spec, direction: str, kind: str) -> list[tuple[tuple, tuple]]:
    """``[(src_slices, dst_slices), ...]`` for ``direction`` and ``kind``.

    ``direction`` is ``'gather'`` or ``'scatter'``.  The container is located
    by trying, in order:

    1. ``spec.gather(kind)`` / ``spec.scatter(kind)`` -- a method taking the
       variant name.  THIS IS WHAT ``tilestream.spec.TileSpec`` DOES.
    2. ``spec.gather_segments(kind)`` / ``spec.gather_segments[kind]``;
    3. ``spec.gather[kind]`` / ``spec.scatter[kind]`` -- a mapping attribute;
    4. ``spec.segments[direction][kind]`` or ``spec.segments[direction, kind]``;
    5. ``spec[direction][kind]`` when ``spec`` is itself a mapping.

    Each entry may be a ``tilestream.spec.Transfer`` (``full_key``/``tile_key``,
    or ``full_y/full_x/tile_y/tile_x``), which :func:`_as_segment_list`
    orients by direction, or an already-oriented ``(src_slices, dst_slices)``
    pair.

    A missing ``'w'`` entry falls back to ``'mass'``: ``w``/``phi`` are mass
    points horizontally and this module never consumes a vertical slice, so
    the mass segments are exactly right for them.  A missing entry for any
    other kind raises rather than silently copying nothing -- an empty
    gather looks precisely like a speedup.
    """
    if direction not in ("gather", "scatter"):
        raise ValueError(f"direction must be 'gather' or 'scatter', got {direction!r}")

    sought = [kind]
    fallback = _KIND_FALLBACK.get(kind)
    if fallback is not None:
        sought.append(fallback)

    for want in sought:
        for container in (
            getattr(spec, f"{direction}_segments", None),
            getattr(spec, direction, None),
            _lookup(getattr(spec, "segments", None), direction),
            (spec.get(direction) if isinstance(spec, Mapping) else None),
        ):
            found = _lookup(container, want)
            if found is not None:
                return _as_segment_list(found, direction)
        seg_map = getattr(spec, "segments", None)
        if isinstance(seg_map, Mapping) and (direction, want) in seg_map:
            return _as_segment_list(seg_map[(direction, want)], direction)

    raise SpecError(
        f"TileSpec {type(spec).__name__} exposes no {direction!r} segments for "
        f"kind {kind!r}; expected one of spec.{direction}_segments(kind), "
        f"spec.{direction}[{kind!r}], or spec.segments[{direction!r}][{kind!r}]")


def _range_from_slices(slices: tuple, shape: tuple[int, ...],
                       what: str) -> tuple[int, int, int, int]:
    """Reduce a spec slice tuple to ``(y0, y1, x0, x1)`` on ``shape``.

    Leading (vertical) entries must be full slices or ``Ellipsis`` and are
    dropped: the vertical extent is field-dependent (nz vs nz+1 vs 2-D) while
    one horizontal segment list has to serve every field.  A partial vertical
    slice is an error, not something to quietly ignore.
    """
    parts = list(slices)
    if Ellipsis in parts:
        if parts.count(Ellipsis) > 1:
            raise SpecError(f"{what}: more than one Ellipsis in {slices!r}")
        parts = [p for p in parts if p is not Ellipsis]
    if len(parts) < 2:
        raise SpecError(
            f"{what}: slice tuple {slices!r} has fewer than 2 horizontal axes")
    lead, horiz = parts[:-2], parts[-2:]
    for entry in lead:
        if not (isinstance(entry, slice)
                and entry.start in (None, 0) and entry.stop is None
                and entry.step in (None, 1)):
            raise SpecError(
                f"{what}: leading (vertical) entry {entry!r} in {slices!r} is "
                "not a full slice; tiles never subset z")

    out: list[int] = []
    for axis, entry in zip((-2, -1), horiz):
        dim = int(shape[axis])
        if isinstance(entry, slice):
            if entry.step not in (None, 1):
                raise SpecError(f"{what}: step {entry.step} unsupported in {slices!r}")
            start, stop, _ = entry.indices(dim)
        elif isinstance(entry, (int, np.integer)):
            start = int(entry) % dim
            stop = start + 1
        else:
            raise SpecError(f"{what}: unsupported index {entry!r} in {slices!r}")
        if stop < start:
            stop = start
        if start < 0 or stop > dim:
            raise SpecError(
                f"{what}: range [{start}:{stop}) is outside axis of length {dim} "
                f"(from {slices!r})")
        out.extend((start, stop))
    return out[0], out[1], out[2], out[3]


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------

def _rect_overlap(a, b) -> bool:
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    return (ay0 < by1 and by0 < ay1) and (ax0 < bx1 and bx0 < ax1)


def _check_destination_rects(rects, shape, *, require_full: bool,
                             what: str) -> None:
    """Structural check on where a kind's segments WRITE.

    Two invariants, both spec-independent:

    * DISJOINT -- no destination point is written twice.  Two rectangles
      landing on the same cell means one of them is wrong, and which one wins
      is a race between queued copies.
    * COMPLETE (gather only) -- the rectangles cover the destination array
      exactly.  ``tilestream.spec`` states this itself ("together they fill
      the whole tile array for variant, every point, halo included, exactly
      once"), and it is the invariant that catches the failure this transport
      layer cannot otherwise see: a tile array whose horizontal shape does
      not match the plan leaves cells holding whatever was there before, and
      stale halo cells produce a plausible-looking forecast rather than a
      crash.  Disjoint + total area == array area + in bounds implies an
      exact cover, so no mask is needed.

    A scatter writes only the tile's interior, so completeness there is a
    property of the whole PLAN, not of one tile; only disjointness is checked.
    """
    h, w = int(shape[-2]), int(shape[-1])
    area = 0
    for i, r in enumerate(rects):
        y0, y1, x0, x1 = r
        area += (y1 - y0) * (x1 - x0)
        for j in range(i):
            if _rect_overlap(r, rects[j]):
                raise SpecError(
                    f"{what}: destination rectangles {j} and {i} overlap "
                    f"({rects[j]} vs {r}); a point would be written twice")
    if require_full and area != h * w:
        raise SpecError(
            f"{what}: destination rectangles cover {area} of {h * w} points "
            f"in a {h}x{w} array. A gather must fill the tile completely -- "
            "uncovered cells keep stale values, which reads as a working "
            "forecast rather than an error. Check that the tile array's "
            "horizontal shape matches the plan's tile_ny/tile_nx + 2*halo.")


class _FieldCopy:
    """One (field, segment) 3-D block copy, resolved except for base pointers."""

    __slots__ = ("name", "kind", "itemsize", "memkind",
                 "src_pitch", "src_xsize", "src_ysize",
                 "dst_pitch", "dst_xsize", "dst_ysize",
                 "src_x", "src_y", "dst_x", "dst_y",
                 "w_bytes", "height", "depth", "nbytes")

    def __init__(self, name, kind, itemsize, memkind,
                 src_shape, dst_shape,
                 src_y0, src_x0, dst_y0, dst_x0,
                 height, width, depth):
        self.name = name
        self.kind = kind
        self.itemsize = itemsize
        self.memkind = memkind
        self.src_pitch = int(src_shape[-1]) * itemsize
        self.src_xsize = int(src_shape[-1])
        self.src_ysize = int(src_shape[-2])
        self.dst_pitch = int(dst_shape[-1]) * itemsize
        self.dst_xsize = int(dst_shape[-1])
        self.dst_ysize = int(dst_shape[-2])
        self.src_x = int(src_x0) * itemsize
        self.src_y = int(src_y0)
        self.dst_x = int(dst_x0) * itemsize
        self.dst_y = int(dst_y0)
        self.w_bytes = int(width) * itemsize
        self.height = int(height)
        self.depth = int(depth)
        self.nbytes = self.w_bytes * self.height * self.depth


class TilePlan:
    """A resolved, reusable gather or scatter for one tile.

    Holds no buffers of its own and allocates nothing when executed: the
    single ``cudaMemcpy3DParms`` is preallocated here and mutated in place
    per block (the driver reads the struct before ``cudaMemcpy3DAsync``
    returns, so reuse across queued copies is safe).
    """

    __slots__ = ("direction", "spec", "copies", "field_names",
                 "_parms", "_parms_addr", "_src_names", "_dst_names",
                 "nbytes", "signature")

    def __init__(self, direction: str, spec, copies: list[_FieldCopy],
                 signature, field_names: tuple[str, ...]):
        self.direction = direction
        self.spec = spec
        self.copies = copies
        self.field_names = field_names
        self.signature = signature
        self.nbytes = sum(c.nbytes for c in copies)
        self._parms = _Memcpy3DParms()
        self._parms_addr = ctypes.addressof(self._parms)

    def __repr__(self) -> str:
        return (f"<TilePlan {self.direction} fields={len(self.field_names)} "
                f"blocks={len(self.copies)} bytes={self.nbytes}>")

    def execute(self, src: Mapping[str, Any], dst: Mapping[str, Any],
                stream=None) -> None:
        """Issue every block copy on ``stream`` (non-blocking).

        ``src``/``dst`` are the inventories the plan was built for, or any
        inventories with the identical structure -- only base pointers are
        read here, so a double-buffered pipeline can reuse one plan across
        two tile buffers with the same shapes.
        """
        import cupy as cp

        rt = _runtime()
        if stream is None:
            stream = cp.cuda.get_current_stream()
        stream_ptr = int(getattr(stream, "ptr", stream))

        parms = self._parms
        addr = self._parms_addr
        from woof.hip_runtime import memcpy3d_function
        memcpy3d = memcpy3d_function(rt)
        for copy in self.copies:
            src_arr = src[copy.name]
            dst_arr = dst[copy.name]
            parms.srcPtr.ptr = ctypes.c_void_p(array_pointer(src_arr))
            parms.srcPtr.pitch = copy.src_pitch
            parms.srcPtr.xsize = copy.src_xsize
            parms.srcPtr.ysize = copy.src_ysize
            parms.srcPos.x = copy.src_x
            parms.srcPos.y = copy.src_y
            parms.srcPos.z = 0
            parms.dstPtr.ptr = ctypes.c_void_p(array_pointer(dst_arr))
            parms.dstPtr.pitch = copy.dst_pitch
            parms.dstPtr.xsize = copy.dst_xsize
            parms.dstPtr.ysize = copy.dst_ysize
            parms.dstPos.x = copy.dst_x
            parms.dstPos.y = copy.dst_y
            parms.dstPos.z = 0
            parms.extent.width = copy.w_bytes
            parms.extent.height = copy.height
            parms.extent.depth = copy.depth
            parms.kind = copy.memkind
            memcpy3d(addr, stream_ptr)


def _memkind_tag(array) -> str:
    if is_device_array(array):
        return "device"
    return "pinned" if is_pinned(array) else "pageable"


def _signature(direction: str, src: Mapping[str, Any],
               dst: Mapping[str, Any]) -> tuple:
    return (direction,
            tuple((n, tuple(a.shape), a.dtype.str, _memkind_tag(a))
                  for n, a in src.items()),
            tuple((n, tuple(a.shape), a.dtype.str, _memkind_tag(a))
                  for n, a in dst.items()))


def _check_host(array, name: str, role: str, allow_pageable: bool) -> None:
    if is_device_array(array):
        return
    if is_pinned(array):
        return
    msg = (f"{role} array {name!r} is pageable host memory. Pinned host RAM "
           "transfers at 57 GB/s here; pageable transfers at 15 GB/s and is "
           "not truly async, which destroys copy/compute overlap. Allocate "
           "with tilestream.gather.pinned_empty / make_host_store.")
    if not allow_pageable:
        raise PageableHostMemoryError(msg)


def build_plan(src_arrays: Mapping[str, Any], dst_arrays: Mapping[str, Any],
               spec, direction: str, *,
               allow_pageable: bool = False,
               require_full_gather: bool = True,
               domain_arrays: Mapping[str, Any] | None = None,
               nz: int | None = None, global_keys=()) -> TilePlan:
    """Resolve ``spec`` against two inventories into a :class:`TilePlan`.

    ``domain_arrays`` names which of the two sides is the FULL domain (it is
    ``src_arrays`` for a gather and ``dst_arrays`` for a scatter); it is what
    the stagger classification is done against, and the tile side is then
    checked to have exactly the shape that stagger implies.

    ``require_full_gather`` (default on) demands that a gather's rectangles
    cover the tile array exactly -- see :func:`_check_destination_rects`.
    Turn it off only for a spec that deliberately leaves part of a tile
    untouched, and know that you are giving up the check that catches stale
    halo cells.
    """
    if direction == "gather":
        domain_arrays = src_arrays if domain_arrays is None else domain_arrays
        tile_arrays = dst_arrays
    elif direction == "scatter":
        domain_arrays = dst_arrays if domain_arrays is None else domain_arrays
        tile_arrays = src_arrays
    else:
        raise ValueError(f"direction must be 'gather' or 'scatter', got {direction!r}")

    names = [n for n in src_arrays if n in dst_arrays]
    missing = [n for n in src_arrays if n not in dst_arrays]
    extra = [n for n in dst_arrays if n not in src_arrays]
    if missing or extra:
        raise SpecError(
            f"{direction}: inventories disagree; only in source {missing}, "
            f"only in destination {extra}. The store and the tile state must "
            "hold the same field set.")
    if not names:
        raise SpecError(f"{direction}: no fields in common between store and tile")

    # ``nz`` given means the caller owns the vertical extent (it comes from
    # cfg.nz) and the inventory may therefore hold layered fields whose
    # leading axis is not vertical.  See :func:`domain_extents`.
    layers_ok = nz is not None
    nz, ny, nx = domain_extents(domain_arrays, nz=nz)
    tnz, tny, tnx = domain_extents(tile_arrays, nz=nz if layers_ok else None)
    if tnz != nz:
        raise SpecError(
            f"tile vertical extent {tnz} != domain {nz}; tiling is horizontal only")

    segment_cache: dict[str, list[tuple[tuple, tuple]]] = {}
    copies: list[_FieldCopy] = []

    for name in names:
        src_arr = src_arrays[name]
        dst_arr = dst_arrays[name]
        if name in global_keys:
            from tilestream.global_inventory import global_keys as domain_global_keys
            if domain_global_keys((name,)) != (name,):
                raise SpecError(f"unknown global carrier {name} would bypass tile transport")
            if src_arr.shape != dst_arr.shape or src_arr.dtype != dst_arr.dtype:
                raise SpecError(f"global carrier {name} differs from its single domain owner")
            continue
        dom_arr = domain_arrays[name]
        til_arr = tile_arrays[name]

        if src_arr.dtype != dst_arr.dtype:
            raise SpecError(
                f"{name}: dtype {src_arr.dtype} != {dst_arr.dtype}")
        for arr, role in ((src_arr, "source"), (dst_arr, "destination")):
            if not arr.flags.c_contiguous:
                raise SpecError(
                    f"{name}: {role} array is not C-contiguous; the pitched "
                    "copy math assumes a dense C-order layout")
            _check_host(arr, name, role, allow_pageable)

        from tilestream.fire_inventory import fire_field_layout, fire_transfer_segments, fire_wind_segments
        fire_layout = fire_field_layout(name, dom_arr.shape, ny, nx)
        fire_wind = name in ("fire/uah", "fire/vah")
        if fire_layout is None:
            kind = classify(dom_arr.shape, nz, ny, nx, layers_ok=layers_ok)
            want = tile_shape_for(kind, dom_arr.shape, tny, tnx)
        else:
            ry, rx, py, px = fire_layout
            kind = f"fire:{ry}:{rx}:{py}:{px}"
            want = (tny * ry + 2 * py, tnx * rx + 2 * px)
        if fire_wind:
            kind = f"fire-wind:{kind}"
        if tuple(til_arr.shape) != want:
            raise SpecError(
                f"{name}: domain shape {tuple(dom_arr.shape)} is stagger "
                f"{kind!r}, so the tile array must be {want}, got "
                f"{tuple(til_arr.shape)}")

        if kind not in segment_cache:
            segment_cache[kind] = (fire_wind_segments(spec, name, direction) if fire_wind else
                segments_for(spec, direction, kind) if fire_layout is None else
                fire_transfer_segments(spec, fire_layout, direction))
        segments = segment_cache[kind]
        if not segments:
            raise SpecError(
                f"{direction}: kind {kind!r} has zero segments; an empty "
                "transfer is indistinguishable from a speedup")

        # Every axis ahead of (y, x) is depth.  A rank-4 carrier -- Grell
        # 3D's ``g3_lev`` is (9, nz, ny, nx) -- is a stack of 9 * nz planes
        # of one C-contiguous (ny, nx) pitch, which is exactly what the 3-D
        # block copy walks.  ``shape[0]`` alone copied its first 9 planes
        # and left the rest of the store stale: measured, the streamed
        # g3_lev differed from the resident one in all 9 slots after one
        # step while every rank-2 and rank-3 G3 carrier was bit-exact.
        depth = math.prod(int(n) for n in src_arr.shape[:-2]) if src_arr.ndim >= 3 else 1
        if (math.prod(int(n) for n in dst_arr.shape[:-2])
                if dst_arr.ndim >= 3 else 1) != depth or (
                tuple(src_arr.shape[:-2]) != tuple(dst_arr.shape[:-2])):
            raise SpecError(
                f"{name}: vertical extents differ, {src_arr.shape} vs "
                f"{dst_arr.shape}")
        itemsize = int(src_arr.dtype.itemsize)
        memkind = _memcpy_kind(is_device_array(src_arr), is_device_array(dst_arr))

        dst_rects: list[tuple[int, int, int, int]] = []
        for si, (src_sl, dst_sl) in enumerate(segments):
            sy0, sy1, sx0, sx1 = _range_from_slices(
                src_sl, src_arr.shape, f"{direction} {name} seg{si} src")
            dy0, dy1, dx0, dx1 = _range_from_slices(
                dst_sl, dst_arr.shape, f"{direction} {name} seg{si} dst")
            if (sy1 - sy0) != (dy1 - dy0) or (sx1 - sx0) != (dx1 - dx0):
                raise SpecError(
                    f"{direction} {name} seg{si}: src extent "
                    f"{(sy1 - sy0, sx1 - sx0)} != dst extent "
                    f"{(dy1 - dy0, dx1 - dx0)} (kind {kind!r}; check that the "
                    "spec pairs are (src, dst) in that order)")
            dst_rects.append((dy0, dy1, dx0, dx1))
            if sy1 == sy0 or sx1 == sx0:
                continue                      # empty block; cudaMemcpy3D rejects it
            copies.append(_FieldCopy(
                name, kind, itemsize, memkind,
                src_arr.shape, dst_arr.shape,
                sy0, sx0, dy0, dx0,
                height=sy1 - sy0, width=sx1 - sx0, depth=depth))

        _check_destination_rects(
            dst_rects, dst_arr.shape,
            require_full=(direction == "gather" and require_full_gather and fire_layout is None and not fire_wind),
            what=f"{direction} {name} (kind {kind!r})")

    if not copies:
        raise SpecError(f"{direction}: plan resolved to zero block copies")

    signature = _signature(direction, src_arrays, dst_arrays)
    return TilePlan(direction, spec, copies, signature, tuple(names))


# --------------------------------------------------------------------------
# plan cache
# --------------------------------------------------------------------------

_FALLBACK_CACHE: dict[int, tuple[Any, dict]] = {}


def _plan_cache(spec) -> dict:
    try:
        cache = spec.__dict__.get("_tilestream_plan_cache")
        if cache is None:
            cache = {}
            spec.__dict__["_tilestream_plan_cache"] = cache
        return cache
    except (AttributeError, TypeError):
        entry = _FALLBACK_CACHE.get(id(spec))
        if entry is None or entry[0] is not spec:
            entry = (spec, {})           # strong ref keeps id(spec) valid
            _FALLBACK_CACHE[id(spec)] = entry
        return entry[1]


def make_plan(src_arrays: Mapping[str, Any], dst_arrays: Mapping[str, Any],
              spec, direction: str, *, allow_pageable: bool = False,
              require_full_gather: bool = True,
              nz: int | None = None, global_keys=()) -> TilePlan:
    """Cached :func:`build_plan`, keyed by a structural signature.

    The signature covers every field's name, shape, dtype and memory class on
    both sides, so a cached plan can only be reused where it is exactly
    valid.  Base pointers are NOT part of the plan, so two tile buffers of
    identical shape (the double-buffered pipeline) share one entry.  ``nz``
    is part of the key because it decides the stagger classification.
    """
    key = (_signature(direction, src_arrays, dst_arrays),
           allow_pageable, require_full_gather, nz, tuple(sorted(global_keys)))
    cache = _plan_cache(spec)
    plan = cache.get(key)
    if plan is None:
        plan = build_plan(src_arrays, dst_arrays, spec, direction,
                          allow_pageable=allow_pageable,
                          require_full_gather=require_full_gather, nz=nz, global_keys=global_keys)
        cache[key] = plan
    return plan


# --------------------------------------------------------------------------
# the two public operations
# --------------------------------------------------------------------------

def gather_tile(store, tile_state, spec, stream=None, *,
                allow_pageable: bool = False, require_full_gather: bool = True,
                names=None, inventory_fn=None, nz: int | None = None, global_keys=()
                ) -> TilePlan:
    """Full-domain ``store`` -> ``tile_state`` (halo included).

    Copies, for every field in the persisted inventory, the horizontal
    window(s) the spec's gather segments name -- including the wraparound
    pieces a periodic halo needs -- at full vertical extent.  Async on
    ``stream`` (the current stream when ``None``); nothing is synchronized
    here, so the caller must order the subsequent ``dycore.step`` on the same
    stream or via an event.

    Returns the :class:`TilePlan` used, whose ``.nbytes`` is the exact
    payload moved -- useful for a bandwidth number that cannot be inflated by
    a transfer that silently did nothing.

    ``inventory_fn`` replaces :func:`inventory` as the way a store and a tile
    state are turned into ``{name: array}``.  A physics run passes
    ``physics_inventory.carrier_inventory``, which reaches ``state._scratch``
    and ``state.physics`` -- ``getattr`` over the contract names cannot.
    ``nz`` is required with that inventory, because it holds layered
    soil/snow fields whose leading axis is not vertical; see
    :func:`domain_extents`.
    """
    take = inventory_fn or inventory
    src = take(store, names)
    dst = take(tile_state, names)
    plan = make_plan(src, dst, spec, "gather", allow_pageable=allow_pageable,
                     require_full_gather=require_full_gather, nz=nz, global_keys=global_keys)
    plan.execute(src, dst, stream)
    if "fields/lakemask" in dst:
        lake = getattr(getattr(tile_state, "physics", None), "lake", None)
        if lake is not None:
            # No device read here. Rebuild sparse work only on the compute
            # stream, after the caller's transfer event has completed.
            lake.invalidate_columns()
    return plan


def scatter_tile(tile_state, store, spec, stream=None, *,
                 allow_pageable: bool = False, names=None, inventory_fn=None,
                 nz: int | None = None, global_keys=()) -> TilePlan:
    """``tile_state`` interior -> full-domain ``store``.

    The mirror of :func:`gather_tile`, driven by the spec's SCATTER segments,
    which name the halo-free interior on the tile side and where it lands on
    the domain side.  Async on ``stream``; returns the plan.
    ``inventory_fn`` and ``nz`` mean what they do in :func:`gather_tile`.
    """
    take = inventory_fn or inventory
    src = take(tile_state, names)
    dst = take(store, names)
    plan = make_plan(src, dst, spec, "scatter", allow_pageable=allow_pageable,
                     nz=nz, global_keys=global_keys)
    plan.execute(src, dst, stream)
    return plan


__all__ = [
    "HostStore", "KINDS", "PageableHostMemoryError", "SpecError", "TilePlan",
    "array_pointer", "build_plan", "classify", "domain_extents", "gather_tile",
    "inventory", "is_device_array", "is_pinned", "make_host_store", "make_plan",
    "pinned_copy", "pinned_empty", "pinned_empty_like", "scatter_tile",
    "segments_for", "serialized_attrs", "tile_shape_for",
]
