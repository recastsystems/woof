"""The pinned full-domain host store: the authoritative out-of-core state.

This is the module that makes a larger-than-VRAM domain possible.  The
persistent state -- exactly the ``STATE_SERIALIZED_ATTRS`` inventory that
:mod:`tilestream.harness` hashes -- lives here, in PINNED host RAM, at FULL
domain shape.  Only halo tiles ever visit the GPU.

Three things in here are essential and worth reading before you use it.

1.  THE MANIFEST IS PROBED, NOT HARDCODED.  Which of the contract
    attributes (47 as of the mp9/P3 restart fixes; ``test_gather`` pins the
    count) actually exist, and their staggering, depend on the config
    (``moist``, ``mp_physics``, ``km_opt``, ``bl_pbl_physics``).  Rather than
    transcribe ``woof/core/state.py``'s allocation branches -- which would
    silently drift -- :func:`build_manifest` constructs a throwaway
    ``DomainState(cfg', array_module=numpy)`` at a tiny probe shape whose six
    possible extents are pairwise distinct, and reads the staggering off the
    real shapes.  The probe costs no GPU memory and no CUDA context.

2.  PINNING IS ASSERTED, NOT ASSUMED.  Every buffer's device pointer
    attributes are checked (``cudaPointerGetAttributes(...).type == 1``,
    ``cudaMemoryTypeHost``); a pageable pointer reports ``0``.  That is a
    hard programmatic gate, and :meth:`HostDomainStore.measure_h2d_bandwidth`
    is the independent empirical one -- pinned should reach ~50-57 GB/s on
    this box, pageable ~15 GB/s.

3.  THE HASH IS BYTE-IDENTICAL TO THE HARNESS'S.  :meth:`HostDomainStore.hash`
    reuses ``harness._digest_payload`` and walks ``STATE_SERIALIZED_ATTRS`` in
    the same order, so ``store.hash() == harness.hash_state(state)`` is a
    direct comparison and is the correctness gate for the whole round trip.

4.  ALLOCATION BYPASSES CUPY'S PINNED POOL.  ``cupy.cuda.alloc_pinned_memory``
    rounds every request UP TO THE NEXT POWER OF TWO -- measured, see
    :func:`pool_rounded_bytes` -- so a field one byte past 2 GiB consumed
    4 GiB.  That, and nothing about the driver, produced the "2 GiB block
    cliff" this store was previously believed to have.  :class:`PinnedBlock`
    calls ``cudaHostAlloc`` directly for the EXACT byte count.  See
    :func:`probe_pinned_ceiling`.

Everything is float32 -- verified against the probe, not assumed; the store
carries whatever dtype the probe reports and would allocate a non-float32
field correctly if one ever appeared.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import gc
import hashlib
import os
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS


#: ``cudaMemoryTypeHost``.  ``cudaPointerGetAttributes`` reports this for
#: page-locked host memory and ``0`` (``cudaMemoryTypeUnregistered``) for
#: ordinary pageable memory.  Measured both ways on this box.
CUDA_MEMORY_TYPE_HOST = 1

#: Probe extents for :func:`build_manifest`.  The six extents a persisted
#: array can have are ``nz, nz+1, ny, ny+1, nx, nx+1`` = 9, 10, 5, 6, 3, 4 --
#: pairwise distinct, so every axis of every probed shape classifies
#: unambiguously.  Do not "simplify" these to a square.
_PROBE_NZ, _PROBE_NY, _PROBE_NX = 9, 5, 3

#: Refuse to allocate if it would leave the box with less than this much
#: available RAM.  Pinned pages are UNSWAPPABLE: over-allocating here does not
#: degrade gracefully, it destabilises the machine.  This is the reserve's
#: ceiling; :func:`host_reserve_bytes` scales it down on smaller machines.
DEFAULT_RESERVE_BYTES = 8 << 30

#: The reserve as a fraction of the memory this process may use
#: (:func:`host_memory`'s ``total``: MemTotal, or the memory cgroup limit
#: when one binds).  The reserve is the smaller of this share and
#: :data:`DEFAULT_RESERVE_BYTES`.
#:
#: THE BREAKAGE a fixed 8 GiB reserve caused: it comes out of the room under
#: a memory limit, so a container under about 8.5 GiB refused every store,
#: and one under about 20 GiB could refuse a store the planner had sized to
#: :data:`DEFAULT_MAX_TOTAL_FRACTION` of that same limit; a bare machine of
#: the same size refused the same way.  An eighth reaches the full 8 GiB at
#: 64 GiB, so every machine that size or larger keeps today's reserve, and
#: below it an eighth of the machine stays free beside the store while the
#: 0.47 total-fraction gate still caps the store itself.
RESERVE_TOTAL_FRACTION = 0.125


def host_reserve_bytes(total_bytes: int | None = None) -> int:
    """The RAM a pinned store leaves free: the smaller of
    :data:`DEFAULT_RESERVE_BYTES` and :data:`RESERVE_TOTAL_FRACTION` of
    ``total_bytes`` (default :func:`host_memory`'s ``total``)."""
    total = host_memory()["total"] if total_bytes is None else int(total_bytes)
    return min(DEFAULT_RESERVE_BYTES, int(RESERVE_TOTAL_FRACTION * total))

#: The system-wide page-locked ceiling, as a fraction of ``MemTotal``.
#:
#: MEASURED TO 64 MiB RESOLUTION, and it is a remarkably clean number: raw
#: ``cudaHostAlloc`` on this box walls at **46.9375 GiB of 93.9102 GiB
#: MemTotal = 0.4998**, i.e. exactly HALF of RAM.  The refinement matters --
#: filling with 1 GiB blocks stops at 46.00 GiB and looks like "a 46 GiB
#: driver cap", but topping up with 256 MiB and then 64 MiB blocks walks
#: right up to MemTotal/2.  The old 46 GiB figure was 1 GiB quantisation, not
#: a distinct limit.
#:
#: It is a LIMIT ON PAGE-LOCKING, not on an API: ``cudaHostRegister`` over
#: anonymous ``mmap``s hits the same wall (46.0 GiB in 1 GiB chunks), so
#: there is no alternative allocator that reaches further.  It is also
#: invisible to ``/proc/meminfo`` -- ``MemAvailable`` read 92 GiB throughout,
#: so a guard built only on ``MemAvailable`` sails straight past it into a
#: raw ``cudaErrorMemoryAllocation``.
PINNED_CEILING_FRACTION = 0.5

#: Second, independent cap: never pin more than this fraction of total RAM.
#:
#: Sits just below :data:`PINNED_CEILING_FRACTION`.  0.47 is ~44.1 GiB here,
#: which leaves ~2.8 GiB of page-locked headroom for tile buffers and any
#: stray pinned staging elsewhere in the process, and still leaves the box
#: over 49 GiB of ordinary RAM.  It was 0.45 when the store allocated through
#: CuPy's pinned pool, because the pool's power-of-two rounding could make a
#: request consume up to 2.05x its nominal size; :class:`PinnedBlock` removed
#: that, so the nominal number is now the real one and the cap can sit closer
#: to the wall.  Raise it only after re-measuring on the actual machine.
DEFAULT_MAX_TOTAL_FRACTION = 0.47

#: Smallest transfer a bandwidth number may be reported from.  Below this the
#: 10.9 us fixed per-transfer overhead and the measurement noise floor make
#: the number meaningless; PCIe saturates at 32 MB.
MIN_ACCURATE_BANDWIDTH_BYTES = 256 << 20

GIB = 1 << 30


class HostStoreError(RuntimeError):
    """Base class for store allocation and consistency failures."""


class BudgetExceeded(HostStoreError):
    """The requested store is larger than the caller's ``budget_bytes``."""


class HostMemoryExhausted(HostStoreError):
    """The requested store would drive the box into swap.  Refused."""


class PinnedAllocationFailed(HostStoreError):
    """``cudaHostAlloc`` refused, despite the host-memory guard passing.

    The system-wide page-locked ceiling is lower than free RAM and is not
    visible in ``/proc/meminfo`` -- 46.94 GiB (MemTotal/2) of 93.91 GiB on
    this box, while ``MemAvailable`` read 92 GiB.  This exception reports how
    far the store got before the wall.
    """


class InventoryMismatch(HostStoreError):
    """A ``DomainState`` does not carry the inventory this store holds."""


# --------------------------------------------------------------------------
# exact-size pinned allocation
# --------------------------------------------------------------------------

def pool_rounded_bytes(nbytes: int) -> int:
    """What ``cupy.cuda.alloc_pinned_memory(nbytes)`` actually CONSUMES.

    CuPy's ``PinnedMemoryPool`` bins by power of two and hands back a block
    from the bin, so a request consumes ``2**ceil(log2(nbytes))`` bytes of
    page-locked RAM no matter how few of them the caller asked for.

    MEASURED by ``MemAvailable`` delta across 8 (or 4) simultaneous blocks, at
    six sizes spanning 640 MiB to 3 GiB::

        request    consumed/block   ratio   next pow2
          640 MiB      1042 MiB     1.629     1024 MiB
          768 MiB      1030 MiB     1.341     1024 MiB
         1025 MiB      2096 MiB     2.045     2048 MiB
         1536 MiB      2059 MiB     1.340     2048 MiB
         2048 MiB      4118 MiB     2.011     4096 MiB   <- 2 GiB + 1
         3072 MiB      4118 MiB     1.340     4096 MiB

    THIS IS THE ENTIRE "2 GiB BLOCK CLIFF".  A store field of
    ``(nz+1) * n^2 * 4`` bytes crosses 2 GiB at ``n = 3277`` (nz=49), so at
    n=3276 every field rounded to ~2 GiB and 46 GiB of ceiling held about
    23 fields' worth, while at n=3277 each field silently took 4 GiB and the
    same ceiling held half as much.  Nothing in the driver changed at that
    byte; the pool's bin index did.

    :class:`PinnedBlock` does not go through the pool, so for this store the
    function is now purely diagnostic -- it says what the old path would have
    cost.  Verified against the driver directly: raw ``cudaHostAlloc`` reaches
    the SAME 46 GiB cumulative total with 1 GiB, 2 GiB-1, 2 GiB, 2 GiB+1 and
    2 GiB+4096 blocks, i.e. no block-size sensitivity whatsoever.
    """
    nbytes = int(nbytes)
    if nbytes <= 0:
        return 0
    return 1 << (nbytes - 1).bit_length()


#: Live page-locked bytes and blocks from :class:`PinnedBlock`, and the
#: high-water mark of the bytes.  A list rather than three module globals so
#: that :meth:`PinnedBlock.__del__` -- which can run during interpreter
#: teardown, when module globals have already been rebound to ``None`` -- can
#: mutate it through the closed-over object instead of by name lookup.
_PINNED_LEDGER = [0, 0, 0]      # [live_bytes, live_blocks, peak_bytes]


def pinned_ledger() -> dict:
    """Every page-locked byte this process holds, by allocator.

    There is no way to ask the OS.  Page-locked pages are ordinary anonymous
    RSS as far as ``/proc/self/status`` is concerned -- ``VmLck`` counts only
    ``mlock``, which ``cudaHostAlloc`` does not use -- and on these boxes
    ``/proc/meminfo`` reports the HOST's memory, not the container's.  So a
    run that wants to see a pinned leak as a TREND rather than infer one from
    a final ``cudaHostAlloc`` failure has to keep its own count, and it has
    to cover both allocators, because the two paths into pinned memory in
    this package are different:

    ``blocks``
        :class:`PinnedBlock`, i.e. raw ``cudaHostAlloc`` of an exact size.
        :class:`HostDomainStore` and :mod:`tilestream.rings` allocate here.
    ``pool``
        CuPy's ``PinnedMemoryPool``, which :func:`tilestream.gather
        .pinned_empty` uses and therefore so does every
        :func:`tilestream.gather.pinned_copy` -- which is how
        :func:`woof.core.streaming.attach` fills a host store.  The pool
        BINS BY POWER OF TWO (:func:`pool_rounded_bytes`), so its
        ``total_bytes`` can be up to 2.05x the bytes asked for, and it does
        not return freed blocks to the OS: ``used_bytes`` falling while
        ``total_bytes`` holds is the pool caching, not a leak.

    ``peak_block_bytes`` is a high-water mark rather than a sample, because a
    transient store built and freed between two samples is invisible to
    sampling and is exactly the shape of allocation that fragments a run.
    """
    live, blocks, peak = _PINNED_LEDGER
    out = {"block_bytes": int(live), "blocks": int(blocks),
           "peak_block_bytes": int(peak),
           "pool_total_bytes": 0, "pool_used_bytes": 0}
    try:
        import cupy as cp

        pool = cp.get_default_pinned_memory_pool()
        out["pool_total_bytes"] = int(pool.total_bytes())
        out["pool_used_bytes"] = int(pool.used_bytes())
    except Exception:                          # pragma: no cover - no cupy
        pass
    out["total_bytes"] = out["block_bytes"] + out["pool_total_bytes"]
    return out


class PinnedBlock:
    """One page-locked host block of an EXACT size, from ``cudaHostAlloc``.

    Deliberately not ``cupy.cuda.alloc_pinned_memory``: that rounds up to the
    next power of two (:func:`pool_rounded_bytes`) and wasted up to 2.05x.
    This asks the driver for precisely ``nbytes``.

    There is no upper bound on a single block worth designing around --
    MEASURED: single raw pinned allocations of 2, 4, 6, 8, 12, 16, 24 and
    32 GiB all succeed on this box, each reporting
    ``cudaPointerGetAttributes(...).type == 1`` and writable at both
    endpoints.  A domain field therefore stays ONE contiguous ``(nz, ny, nx)``
    array at any size the machine can hold, which is what lets
    :mod:`tilestream.gather` issue one ``cudaMemcpy3DAsync`` per (field,
    segment) with no banding and no boundary case.

    Exposes ``__array_interface__``, so ``numpy.asarray(block)`` yields a
    writable ``uint8`` view whose ``.base`` is the block -- the reference
    chain that keeps the memory alive for as long as any view of it exists.
    Freeing is therefore refcount-driven (see
    :meth:`HostDomainStore.free`); calling :meth:`free` while a view is still
    reachable would leave that view dangling, so :meth:`free` is not called
    behind the caller's back.
    """

    __slots__ = ("ptr", "nbytes", "__array_interface__", "__weakref__")

    def __init__(self, nbytes: int):
        from cupy.cuda import runtime

        nbytes = int(nbytes)
        if nbytes <= 0:
            raise ValueError(f"PinnedBlock needs a positive size, got {nbytes}")
        self.ptr = int(runtime.hostAlloc(nbytes, 0))
        self.nbytes = nbytes
        _PINNED_LEDGER[0] += nbytes
        _PINNED_LEDGER[1] += 1
        _PINNED_LEDGER[2] = max(_PINNED_LEDGER[2], _PINNED_LEDGER[0])
        self.__array_interface__ = {
            "data": (self.ptr, False),      # False == NOT read-only
            "shape": (nbytes,),
            "typestr": "|u1",
            "version": 3,
        }

    def free(self) -> None:
        """Return the block to the OS.  Idempotent.

        Only safe once every NumPy view of it is gone; the store relies on
        refcounting rather than calling this eagerly.
        """
        ptr, self.ptr = getattr(self, "ptr", 0), 0
        if ptr:
            from cupy.cuda import runtime

            runtime.freeHost(ptr)
            # Decremented only on the transition, so the idempotent second
            # call cannot drive the ledger negative.
            _PINNED_LEDGER[0] -= self.nbytes
            _PINNED_LEDGER[1] -= 1

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:                      # pragma: no cover - teardown
            pass

    def __repr__(self) -> str:                 # pragma: no cover - cosmetic
        return f"<PinnedBlock {self.nbytes} B @ 0x{self.ptr:x}>"


def alloc_pinned_array(shape: Sequence[int], dtype) -> np.ndarray:
    """A zeroed, page-locked, C-contiguous array of EXACTLY ``shape`` bytes.

    The returned array's ``.base`` chain owns a :class:`PinnedBlock`, so the
    caller only has to keep the array itself alive.  Unlike
    :func:`tilestream.gather.pinned_empty` (which goes through CuPy's pool and
    therefore rounds to a power of two) this consumes only what it asks for.
    """
    dtype = np.dtype(dtype)
    shape = tuple(int(s) for s in shape)
    count = 1
    for extent in shape:
        count *= extent
    block = PinnedBlock(count * dtype.itemsize)
    array = np.asarray(block).view(dtype).reshape(shape)
    if not array.flags.writeable:              # pragma: no cover - defensive
        raise HostStoreError(
            "pinned block produced a read-only view; the __array_interface__ "
            "read-only flag is wrong")
    array[...] = 0
    return array


def owning_block(array: np.ndarray) -> PinnedBlock | None:
    """The :class:`PinnedBlock` backing ``array``, by walking ``.base``.

    ``alloc_pinned_array`` returns a view of a view (uint8 -> dtype ->
    reshape), so the block is two or three links down the chain.  Returns
    ``None`` for an array this module did not allocate.
    """
    node: Any = array
    for _ in range(8):
        node = getattr(node, "base", None)
        if node is None:
            return None
        if isinstance(node, PinnedBlock):
            return node
    return None


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldSpec:
    """Shape rule for one persisted attribute.

    ``has_z`` is False for the 2-D surface fields (``mup``, ``nwfa2d``,
    ``nifa2d``).  The three stagger flags are the WRF-ARW ``+1`` on that axis:
    ``u`` is x-staggered, ``v`` is y-staggered, ``w``/``php`` are
    z-staggered.  A gather that slices every array identically is wrong, and
    this is the table that says so.
    """

    name: str
    dtype: np.dtype
    has_z: bool
    z_stagger: bool
    y_stagger: bool
    x_stagger: bool
    #: Fixed leading extent for a field whose first axis is NOT vertical --
    #: Noah's 4 soil layers, Noah-MP's 3 snow layers and its 7-deep snow-soil
    #: coordinate.  ``None`` for every ARW-staggered field.  A layered field
    #: does not scale with ``nz``, which is why it needs its own slot rather
    #: than an abused ``has_z``.
    layers: int | None = None
    refinement_y: int = 1
    refinement_x: int = 1
    padding_y: int = 0
    padding_x: int = 0
    fixed_shape: tuple[int, ...] | None = None
    #: Non-spatial axes before the vertical/layer axis (e.g. species).
    leading_shape: tuple[int, ...] = ()

    def shape(self, nz: int, ny: int, nx: int) -> tuple[int, ...]:
        if self.fixed_shape is not None:
            return self.fixed_shape
        horiz = (ny * self.refinement_y + int(self.y_stagger) + 2 * self.padding_y,
                 nx * self.refinement_x + int(self.x_stagger) + 2 * self.padding_x)
        if self.layers is not None:
            return self.leading_shape + (int(self.layers),) + horiz
        if not self.has_z:
            return self.leading_shape + horiz
        return self.leading_shape + (nz + int(self.z_stagger),) + horiz

    def nbytes(self, nz: int, ny: int, nx: int) -> int:
        return int(np.prod(self.shape(nz, ny, nx))) * self.dtype.itemsize

    @property
    def stagger_code(self) -> str:
        """Compact label: ``m`` mass, ``u``/``v``/``w`` staggered, ``2d``."""
        if self.layers is not None:
            return f"L{self.layers}"
        if not self.has_z:
            return "2d"
        if self.x_stagger:
            return "u"
        if self.y_stagger:
            return "v"
        if self.z_stagger:
            return "w"
        return "m"


def _classify(shape: tuple[int, ...], name: str) -> FieldSpec | None:
    """Turn a probed shape into a :class:`FieldSpec`, or raise."""
    if len(shape) > 3:
        return dataclasses.replace(_classify(shape[-3:], name),
                                   leading_shape=shape[:-3])
    axes = {
        _PROBE_NZ: (True, False), _PROBE_NZ + 1: (True, True),
    }
    horiz_y = {_PROBE_NY: False, _PROBE_NY + 1: True}
    horiz_x = {_PROBE_NX: False, _PROBE_NX + 1: True}

    if len(shape) == 3:
        z, y, x = shape
        if z not in axes or y not in horiz_y or x not in horiz_x:
            raise HostStoreError(
                f"cannot classify staggering of {name!r} from probe shape "
                f"{shape} (probe was nz={_PROBE_NZ} ny={_PROBE_NY} "
                f"nx={_PROBE_NX})")
        has_z, z_stag = axes[z]
        return FieldSpec(name, np.dtype(np.float32), has_z, z_stag,
                         horiz_y[y], horiz_x[x])
    if len(shape) == 2:
        y, x = shape
        if y not in horiz_y or x not in horiz_x:
            raise HostStoreError(
                f"cannot classify staggering of {name!r} from probe shape "
                f"{shape}")
        return FieldSpec(name, np.dtype(np.float32), False, False,
                         horiz_y[y], horiz_x[x])
    raise HostStoreError(
        f"persisted attribute {name!r} has unexpected rank {len(shape)} "
        f"(shape {shape}); a field needs two trailing horizontal axes")


def build_manifest(cfg_like,
                   attrs: Sequence[str] = STATE_SERIALIZED_ATTRS,
                   ) -> tuple[FieldSpec, ...]:
    """Probe ``cfg_like``'s physics options for the real persisted inventory.

    Builds a NumPy-backed ``DomainState`` at :data:`_PROBE_NZ` x
    :data:`_PROBE_NY` x :data:`_PROBE_NX` -- no GPU allocation, no CUDA
    context -- and reads which of ``attrs`` exist plus their dtype and
    staggering.  Ordering follows ``attrs`` (i.e. ``STATE_SERIALIZED_ATTRS``),
    which is what makes :meth:`HostDomainStore.hash` line up with
    ``harness.hash_state``.

    Raises ``TypeError`` if ``cfg_like`` is not a dataclass config; pass an
    explicit ``manifest=`` to :class:`HostDomainStore` in that case.
    """
    from woof.core.state import DomainState

    if not dataclasses.is_dataclass(cfg_like):
        raise TypeError(
            "build_manifest needs a dataclass RunConfig to probe (it shrinks "
            "nx/ny/nz with dataclasses.replace); got "
            f"{type(cfg_like).__name__}.  Pass manifest=... explicitly.")
    probe_cfg = dataclasses.replace(
        cfg_like, nz=_PROBE_NZ, ny=_PROBE_NY, nx=_PROBE_NX)
    probe = DomainState(probe_cfg, array_module=np)

    specs: list[FieldSpec] = []
    for name in attrs:
        value = getattr(probe, name, None)
        if not isinstance(value, np.ndarray):
            continue                      # absent or None under this config
        spec = _classify(tuple(value.shape), name)
        if value.dtype != np.float32:     # honour a non-float32 exception
            spec = dataclasses.replace(spec, dtype=np.dtype(value.dtype))
        specs.append(spec)
    if not specs:
        raise HostStoreError(
            "probe found no persisted arrays; cfg allocates nothing to stream")
    return tuple(specs)


def manifest_from_arrays(arrays: Mapping[str, Any], nz: int, ny: int, nx: int,
                         *, square_slab: bool = False) -> tuple[FieldSpec, ...]:
    """Shape RULES read off a live inventory measured at ``(nz, ny, nx)``.

    :func:`build_manifest` probes a throwaway NumPy ``DomainState``, which
    cannot serve a physics carrier set: the ``driver/*`` and ``fields/*``
    arrays only exist once ``initialize_physics`` has run on a CuPy state,
    and two of them (Kain-Fritsch's ``cumulus/w0avg`` above all) are
    allocated lazily on the first call.  So the carrier manifest is derived
    from a state that has ALREADY RUN A STEP, at its real extents.

    The derivation is positional and total: the last two axes are horizontal
    and must equal ``ny``/``ny+1`` and ``nx``/``nx+1``; the preceding axis is
    vertical when it equals ``nz`` or ``nz+1`` and a fixed LAYER count
    otherwise. Any earlier axes (species, process slots) retain their fixed
    extents. Anything else raises rather than being guessed at, because
    the alternative is a store field of the wrong shape and a transport that
    copies the wrong bytes into it.

    ``ny`` and ``nx`` must differ, and neither may collide with ``nz`` or a
    layer count, or the positional read could resolve two ways; that is
    checked here rather than left to the caller.

    ``square_slab``: the inventory is one slab of a real forecast, square only
    because of where the split fell. The read stays positional and gives the
    rules the same forecast's other slabs give. THE BREAKAGE THIS PREVENTS
    (2026-10-31, a 12 -> 3 km run on a four-way box, run_2e85dd03e500d428):
    "ny == nx == 72" stopped the forecast 8 s in.
    """
    nz, ny, nx = int(nz), int(ny), int(nx)
    if ny == nx and not square_slab:
        raise HostStoreError(
            f"ny == nx == {ny}: derive the manifest on a NON-SQUARE domain so "
            "a y/x transposition cannot pass unnoticed")
    if nz <= 8:
        raise HostStoreError(
            f"nz={nz} is not larger than the largest fixed layer count in "
            "this codebase (7, Noah-MP's snow-soil coordinate): a layered "
            "field would be read as a vertical one and would then be sized "
            "wrong at any other nz.  Derive the manifest at the real nz.")
    specs: list[FieldSpec] = []
    for name, value in arrays.items():
        shape = tuple(int(s) for s in value.shape)
        from tilestream.global_inventory import global_keys
        if global_keys((name,)):
            specs.append(FieldSpec(name, np.dtype(value.dtype), False, False, False, False,
                                   fixed_shape=shape))
            continue
        from tilestream.fire_inventory import fire_field_layout
        fire_layout = fire_field_layout(name, shape, ny, nx)
        if fire_layout is not None:
            ry, rx, py, px = fire_layout
            specs.append(FieldSpec(name, np.dtype(value.dtype), False, False,
                                   False, False, refinement_y=ry, refinement_x=rx,
                                   padding_y=py, padding_x=px))
            continue
        if len(shape) < 2:
            raise HostStoreError(
                f"carrier {name!r} has shape {shape}: no horizontal extent, "
                "so it is not a field this store can hold")
        y, x = shape[-2], shape[-1]
        if y not in (ny, ny + 1) or x not in (nx, nx + 1):
            raise HostStoreError(
                f"carrier {name!r} shape {shape}: trailing axes ({y}, {x}) "
                f"are not (ny={ny}|{ny + 1}, nx={nx}|{nx + 1})")
        y_stag, x_stag = y == ny + 1, x == nx + 1
        if len(shape) == 2:
            spec = FieldSpec(name, np.dtype(value.dtype), False, False,
                             y_stag, x_stag)
        elif len(shape) >= 3:
            lead = shape[-3]
            if name in ("fire/grid.moisture.fmc_gc", "fire/grid.moisture.fmc_equi",
                        "fire/grid.moisture.fmc_lag", "fire/grid.moisture.fmep"):
                # Native moisture-class/parameter axes retain their declared
                # allocation even when their length equals the weather nz.
                spec = FieldSpec(name, np.dtype(value.dtype), False, False,
                                 y_stag, x_stag, layers=lead)
            elif lead == nz:
                spec = FieldSpec(name, np.dtype(value.dtype), True, False,
                                 y_stag, x_stag)
            elif lead == nz + 1:
                spec = FieldSpec(name, np.dtype(value.dtype), True, True,
                                 y_stag, x_stag)
            else:
                spec = FieldSpec(name, np.dtype(value.dtype), False, False,
                                 y_stag, x_stag, layers=lead)
            spec = dataclasses.replace(spec, leading_shape=shape[:-3])
        specs.append(spec)
    if not specs:
        raise HostStoreError("inventory is empty; nothing to stream")
    return tuple(specs)


# --------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------

def domain_bytes(manifest: Iterable[FieldSpec], nz: int, ny: int,
                 nx: int) -> int:
    """Exact byte count for a full ``nz x ny x nx`` store of ``manifest``."""
    return sum(spec.nbytes(nz, ny, nx) for spec in manifest)


def bytes_per_cell(manifest: Iterable[FieldSpec], nz: int, ny: int,
                   nx: int) -> float:
    """Measured B/cell for this manifest at this shape.

    Not a constant: the staggered ``+1``s make small domains dearer per cell.
    Use this against a real allocation rather than trusting the headline
    232 B/cell (which is the FULL moist-physics inventory; a dry periodic
    milestone-one state carries far less).
    """
    manifest = tuple(manifest)
    return domain_bytes(manifest, nz, ny, nx) / float(nz * ny * nx)


def max_square_domain(budget_bytes: int, manifest: Iterable[FieldSpec],
                      nz: int = 49) -> dict[str, Any]:
    """Largest square ``n x n x nz`` domain whose store fits ``budget_bytes``.

    Solved by bisection on the EXACT staggered byte count, not by dividing by
    a per-cell constant, so the answer is a guarantee rather than an estimate.
    Returns the winning ``n``, its byte total, and the per-cell figure that
    allocation actually achieves.
    """
    manifest = tuple(manifest)
    budget_bytes = int(budget_bytes)
    if domain_bytes(manifest, nz, 1, 1) > budget_bytes:
        return {"nx": 0, "ny": 0, "nz": nz, "bytes": 0,
                "bytes_per_cell": float("nan"), "budget_bytes": budget_bytes,
                "note": "budget too small for even a 1x1 column"}
    lo, hi = 1, 2
    while domain_bytes(manifest, nz, hi, hi) <= budget_bytes:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:                      # invariant: lo fits, hi does not
        mid = (lo + hi) // 2
        if domain_bytes(manifest, nz, mid, mid) <= budget_bytes:
            lo = mid
        else:
            hi = mid
    total = domain_bytes(manifest, nz, lo, lo)
    return {"nx": lo, "ny": lo, "nz": nz, "bytes": total,
            "bytes_per_cell": total / float(nz * lo * lo),
            "cells": nz * lo * lo, "budget_bytes": budget_bytes}


# --------------------------------------------------------------------------
# host memory accounting
# --------------------------------------------------------------------------

def _parse_meminfo(lines) -> dict[str, int]:
    """``/proc/meminfo`` text -> ``{key: bytes}``.  Split out so the parser
    is testable on a box that has no procfs at all."""
    info: dict[str, int] = {}
    for line in lines:
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            info[key] = int(parts[0]) * 1024   # kB -> bytes
    return info


def _meminfo() -> dict[str, int]:
    # The planner's path, so one stand-in serves every host-memory reader.
    from tilestream import autoplan

    with open(autoplan._PROC_MEMINFO, "r", encoding="ascii") as handle:
        return _parse_meminfo(handle)


def _host_memory_windows() -> dict[str, int]:
    """``GlobalMemoryStatusEx``, the Win32 answer to ``/proc/meminfo``.

    Same source :func:`tilestream.autoplan._windows_memtotal` reads for the
    planner, taken here in full so the store gets ``available`` too.
    ``ullAvailPhys`` is the kernel's count of physical pages it can hand out
    without paging -- the same question ``MemAvailable`` answers on Linux --
    so it serves as both ``available`` and ``free``: Win32 publishes no
    separate ``MemFree``-shaped number, and every consumer in this module
    reads ``available``.
    """
    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise HostStoreError(
            "GlobalMemoryStatusEx refused; without a host-memory reading the "
            "pinned-store guards cannot run, and running them on a guess is "
            "how a box gets driven into swap")
    avail = int(status.ullAvailPhys)
    return {"total": int(status.ullTotalPhys),
            "available": avail, "free": avail}


def host_memory() -> dict[str, int]:
    """``{'total', 'available', 'free'}`` bytes this process may use.

    ``available`` is the number that matters -- the kernel's own estimate of
    what can be handed out without swapping, which is exactly the question a
    pinned allocation asks.  Linux answers it as ``MemAvailable``; Windows as
    ``GlobalMemoryStatusEx``'s ``ullAvailPhys`` (no procfs exists there, and
    the unconditional ``/proc/meminfo`` read this function used to be meant
    every ``HostDomainStore`` construction on the product's own shipping
    platform died in ``check_allocatable`` before pinning a byte).  Anywhere
    with neither source, psutil answers if it is installed; otherwise this
    raises naming the platform rather than guessing, because every guard in
    this module divides by these numbers.

    Those figures are then capped by the memory cgroup limits this process
    runs under, read through the planner's walk
    (``tilestream.autoplan._cgroup_memory_limits``): ``total`` by the
    smallest limit, ``available`` and ``free`` by the least room left under
    any of them.  The same answers the renderer's, the MPAS builder's and
    the planner's readers give, held to one table
    (``rw-host-memory/src/host_memory_cgroup_cases.json``).

    THE BREAKAGE: the OS figures alone describe the whole host.  Inside a
    systemd scope with ``MemoryMax=2G`` on a 30 GiB worker this read
    30.55 GiB total and 25.97 GiB available, so ``check_allocatable`` with
    no ``budget_bytes`` admitted a 3 GiB pinned store and
    :func:`pinned_ceiling_bytes` put the wall at 15.27 GiB.  Page-locked
    pages cannot be reclaimed, so a store past the scope's limit is one the
    kernel cannot hold.
    """
    mem = _host_memory_uncapped()
    from tilestream import autoplan

    levels = autoplan._cgroup_memory_limits()
    if not levels:
        return mem
    limit = min(level for level, _room in levels)
    room = min(room for _level, room in levels)
    return {"total": min(mem["total"], limit),
            "available": min(mem["available"], room),
            "free": min(mem["free"], room)}


def _cgroup_limit_phrase(mem: dict[str, int] | None = None,
                         key: str | None = None) -> str:
    """Names the memory cgroup limit a refusal's figures were read under,
    or nothing when none binds, so the refusal says whose memory it read.

    Given :func:`host_memory`'s reading and the ``key`` a refusal quotes,
    the limit is named only when it (``total``) or the room under it
    (``available``) is below the host's own figure, because naming a limit
    beside a figure the whole host set tells the reader the wrong cause.
    """
    from tilestream import autoplan

    if mem is not None and key is not None:
        if mem[key] >= _host_memory_uncapped()[key]:
            return ""
    limit = autoplan._cgroup_memory_limit()
    return ("" if limit is None else
            f" under this process's {limit / GIB:.2f} GiB memory cgroup limit")


def _host_memory_uncapped() -> dict[str, int]:
    """The OS's own figures with no cgroup cap: the whole host's.

    Only a report of how far the host's figure is from this process's
    limit reads this (``tilestream.da_stream.container_memory_limit``);
    every guard reads :func:`host_memory`.
    """
    if sys.platform == "win32":
        return _host_memory_windows()
    try:
        info = _meminfo()
    except FileNotFoundError:
        import psutil  # a base dependency since 2.9.3
        vm = psutil.virtual_memory()
        return {"total": int(vm.total), "available": int(vm.available),
                "free": int(getattr(vm, "free", vm.available))}
    return {"total": info.get("MemTotal", 0),
            "available": info.get("MemAvailable", info.get("MemFree", 0)),
            "free": info.get("MemFree", 0)}


#: Environment override for :func:`pinned_ceiling_bytes`, in BYTES.
#:
#: The escape hatch for the fact below: a box whose wall has actually been
#: measured (``probe_pinned_ceiling``, or an operator who knows the machine)
#: states it here rather than being held to a fraction derived from a
#: different machine.
PINNED_CEILING_ENV = "WOOF_PINNED_CEILING_BYTES"


def pinned_ceiling_bytes() -> int:
    """The page-locked wall for THIS box, without allocating.

    ``MemTotal`` DOES NOT PREDICT THIS NUMBER, and the two measurements in
    hand disagree about the fraction:

    ===============  ==================  ==========================
    MemTotal         pinned before wall  fraction
    ===============  ==================  ==========================
    93.91 GiB        46.94 GiB           0.4998 (wall reached)
    123 GiB          84 GiB              0.68 (wall NOT reached)
    ===============  ==================  ==========================

    The second run stopped at its own probe limit with ``hit_wall`` false, so
    0.68 is a LOWER BOUND on that box and the real wall is higher still.  A
    single hardcoded fraction is therefore wrong in both directions: it
    under-uses a large box, and on a box whose wall sits lower than the one
    that was measured it would be dangerously permissive.

    So the number is derived, in this order, and the derivation is reported
    by :func:`pinned_ceiling_source`:

    1. :data:`PINNED_CEILING_ENV`, when set -- a measurement, from a probe or
       from an operator who knows the machine.
    2. Otherwise ``PINNED_CEILING_FRACTION * MemTotal``, which is the
       CONSERVATIVE fallback: it is the only fraction that has been seen to
       be an actual wall, so predicting it on an unmeasured box errs toward
       refusing work rather than toward pinning memory the kernel cannot
       reclaim.  ``MemTotal`` here is :func:`host_memory`'s total, so under
       a memory cgroup limit it is the smallest limit, the fraction the
       planner's ``Machine.host_budget_bytes`` takes of the same figure.

    Deliberately still a prediction and not a probe: probing allocates tens
    of GiB and frees them, which is not something a store build should do on
    a shared box on the way to its real allocation.  The prediction is a
    gate, and :class:`PinnedAllocationFailed` remains the backstop for a wall
    that sits below it.
    """
    override = os.environ.get(PINNED_CEILING_ENV, "").strip()
    if override:
        try:
            value = int(float(override))
        except ValueError:
            raise ValueError(
                f"{PINNED_CEILING_ENV}={override!r} is not a byte count"
            ) from None
        if value <= 0:
            raise ValueError(
                f"{PINNED_CEILING_ENV}={override!r} must be positive")
        return value
    return int(PINNED_CEILING_FRACTION * host_memory()["total"])


def pinned_ceiling_source() -> str:
    """Where :func:`pinned_ceiling_bytes` got its number, for a receipt."""
    return ("measured" if os.environ.get(PINNED_CEILING_ENV, "").strip()
            else "predicted")


def probe_pinned_ceiling(block_bytes: int, *, limit_bytes: int = 60 * GIB,
                         allocator: str = "raw",
                         reserve_bytes: int = DEFAULT_RESERVE_BYTES,
                         refine: bool = False) -> dict[str, Any]:
    """MEASURE how much pinned memory this machine will actually give.

    Allocates ``block_bytes`` at a time until the allocator refuses, then
    frees everything.  ``allocator`` selects the path:

    ``'raw'`` (default)
        ``cudaHostAlloc`` for the exact size -- what :class:`PinnedBlock`, and
        therefore this store, now does.
    ``'pool'``
        ``cupy.cuda.alloc_pinned_memory`` -- what the store used to do, kept
        so the difference stays MEASURABLE rather than merely asserted.

    With ``refine=True`` the fill is topped up with 256 MiB and then 64 MiB
    blocks after the main size refuses, which is what turns a quantised
    "46 GiB" into the true wall.

    MEASURED on this box (93.9102 GiB MemTotal, idle, WSL2 kernel
    6.6.87.1-microsoft-standard-WSL2), cumulative ceiling by block size:

    ==================  ==============  ==============
    block size          raw             pool
    ==================  ==============  ==============
    1 GiB               46.00 GiB       46.00 GiB
    2 GiB - 4096        46.00 GiB       --
    2 GiB - 1           46.00 GiB       --
    2 GiB               46.00 GiB       46.00 GiB
    2 GiB + 1           46.00 GiB       22.00 GiB
    2 GiB + 4096        46.00 GiB       --
    3 GiB               45.00 GiB       --
    4 GiB               44.00 GiB       44.00 GiB
    4.63 GiB            --              23.15 GiB
    ==================  ==============  ==============

    Read the ``raw`` column first: it has NO block-size sensitivity at all.
    46, 45 and 44 GiB are just 46 / 15 / 11 whole blocks of 1, 3 and 4 GiB --
    quantisation of one flat ceiling, which ``refine=True`` walks up to
    46.9375 GiB = 0.4998 x MemTotal.  The ``pool`` column is where the famous
    cliff lives, and it is arithmetic on :func:`pool_rounded_bytes`, not on
    the driver: 2 GiB + 1 rounds to 4 GiB so 11 blocks fit and 11 x (2 GiB+1)
    = 22.00 GiB is REQUESTED; 4.63 GiB rounds to 8 GiB so 5 fit and
    5 x 4.63 = 23.15 GiB is requested.  Both predicted exactly.

    NOTE the allocation is transient but real: this will briefly consume up
    to ``limit_bytes``.  It stops early rather than driving ``MemAvailable``
    below ``reserve_bytes``, and it frees everything on the way out.  Do not
    run it while a large store is live.
    """
    import cupy as cp

    if allocator not in ("raw", "pool"):
        raise ValueError(f"allocator must be 'raw' or 'pool', got {allocator!r}")
    block_bytes = int(block_bytes)
    limit_bytes = int(limit_bytes)
    blocks: list[Any] = []
    total = 0
    failed = False
    stopped = "limit"

    def _take(size: int) -> None:
        """One block, on whichever allocator is under test."""
        nonlocal total
        if allocator == "raw":
            blocks.append(PinnedBlock(size))
        else:
            blocks.append(cp.cuda.alloc_pinned_memory(size))
        total += size

    try:
        sizes = [block_bytes]
        if refine:
            sizes += [s for s in (256 << 20, 64 << 20) if s < block_bytes]
        for size in sizes:
            while total + size <= limit_bytes:
                available = host_memory()["available"]
                if available - size < int(reserve_bytes):
                    stopped = "reserve"
                    break
                try:
                    _take(size)
                except Exception:
                    failed = True
                    stopped = "refused"
                    break
    finally:
        blocks.clear()
        gc.collect()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    return {"block_bytes": block_bytes, "ceiling_bytes": total,
            "ceiling_gib": total / GIB, "hit_wall": failed,
            "allocator": allocator, "stopped": stopped,
            "fraction_of_memtotal": total / max(host_memory()["total"], 1),
            "limit_bytes": limit_bytes}


def check_allocatable(nbytes: int, *, budget_bytes: int | None = None,
                      reserve_bytes: int | None = None,
                      max_total_fraction: float = DEFAULT_MAX_TOTAL_FRACTION,
                      ) -> None:
    """Raise unless ``nbytes`` of pinned RAM can be taken safely.

    Three independent gates, all of which must pass:

    * the caller's explicit ``budget_bytes`` cap,
    * ``MemAvailable - reserve_bytes`` (do not drive the box into swap),
    * ``max_total_fraction * MemTotal`` (never pin most of the machine).

    Both machine figures are :func:`host_memory`'s, so under a memory cgroup
    limit they are the limit's room and the limit, not the whole host's.
    ``reserve_bytes`` None is :func:`host_reserve_bytes` of that total, so
    the reserve shrinks with a small machine or limit instead of taking a
    fixed 8 GiB out of it.  A refusal names the cgroup limit only when the
    limit, or the room under it, is what set the figure it quotes.

    The third gate is the one that matters most and the one that is easiest
    to get wrong.  ``MemAvailable`` is NOT an upper bound on pinnable memory:
    page-locking walls at MemTotal/2 -- 46.94 GiB on this 93.91 GiB box --
    while /proc/meminfo still shows 92 GiB free.  See
    :data:`PINNED_CEILING_FRACTION` and :data:`DEFAULT_MAX_TOTAL_FRACTION`.
    Requests that clear all three gates can still hit that wall (another
    process may hold page-locked memory of its own, and the ceiling is
    system-wide), in which case :class:`PinnedAllocationFailed` is raised with
    the partial total; that is a backstop, not the plan.

    This is deliberately a REFUSAL, not a warning.  Pinning tens of GB that
    the kernel cannot reclaim is the failure mode that takes the box down,
    and it is not worth a "best effort" attempt.
    """
    nbytes = int(nbytes)
    if budget_bytes is not None and nbytes > int(budget_bytes):
        raise BudgetExceeded(
            f"store needs {nbytes / GIB:.2f} GiB but budget_bytes is "
            f"{int(budget_bytes) / GIB:.2f} GiB")
    mem = host_memory()
    if reserve_bytes is None:
        reserve_bytes = host_reserve_bytes(mem["total"])
    headroom = mem["available"] - int(reserve_bytes)
    if nbytes > headroom:
        raise HostMemoryExhausted(
            f"store needs {nbytes / GIB:.2f} GiB; only "
            f"{mem['available'] / GIB:.2f} GiB is available"
            f"{_cgroup_limit_phrase(mem, 'available')} and "
            f"{int(reserve_bytes) / GIB:.2f} GiB is reserved "
            f"(usable {headroom / GIB:.2f} GiB).  Pinned pages cannot be "
            f"swapped; refusing rather than destabilising the machine.")
    cap = int(max_total_fraction * mem["total"])
    if nbytes > cap:
        raise HostMemoryExhausted(
            f"store needs {nbytes / GIB:.2f} GiB, more than "
            f"{max_total_fraction:.0%} of the machine's "
            f"{mem['total'] / GIB:.2f} GiB{_cgroup_limit_phrase(mem, 'total')} "
            f"(cap {cap / GIB:.2f} GiB).  "
            f"Page-locking walls at {PINNED_CEILING_FRACTION:.0%} of MemTotal "
            f"= {pinned_ceiling_bytes() / GIB:.2f} GiB on this box -- MEASURED, "
            f"and invisible to /proc/meminfo.  Refusing.")


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------

class HostDomainStore:
    """Full-domain persistent state in pinned host RAM.

    One pinned buffer per persisted attribute, at full domain shape and
    correct WRF-ARW staggering.  ``.arrays`` exposes them as ordinary
    C-contiguous NumPy views, so a tile gather is plain NumPy slicing --
    but the memory underneath is page-locked, which is the whole point:
    pageable H2D runs at 15 GB/s on this box and pinned at 57 GB/s.

    Parameters
    ----------
    cfg_like:
        A ``RunConfig`` (or any dataclass config carrying ``nz/ny/nx`` and the
        physics selectors).  Its ``nz/ny/nx`` set the store's shape; a shrunk
        copy is used to probe the inventory.
    attrs:
        Contract names to consider, in hash order.  Defaults to the enforced
        ``STATE_SERIALIZED_ATTRS``.
    budget_bytes:
        Hard cap on the store's total size.  ``None`` means only the machine
        gates apply.
    manifest:
        Skip probing and use this inventory (for non-dataclass configs).
    allocate:
        ``False`` builds the manifest and sizing without taking any memory --
        useful for capacity planning.
    """

    def __init__(self, cfg_like, *, attrs: Sequence[str] =
                 STATE_SERIALIZED_ATTRS, budget_bytes: int | None = None,
                 reserve_bytes: int | None = None,
                 max_total_fraction: float = DEFAULT_MAX_TOTAL_FRACTION,
                 manifest: Sequence[FieldSpec] | None = None,
                 allocate: bool = True, inventory_fn=None):
        self.cfg = cfg_like
        self.nz = int(cfg_like.nz)
        self.ny = int(cfg_like.ny)
        self.nx = int(cfg_like.nx)
        #: How a ``DomainState``'s matching arrays are found.  ``None`` means
        #: ``harness.state_arrays`` (the STATE_SERIALIZED_ATTRS contract);
        #: a physics carrier store passes
        #: ``physics_inventory.carrier_manifest``.
        self.inventory_fn = inventory_fn
        self.manifest: tuple[FieldSpec, ...] = (
            tuple(manifest) if manifest is not None
            else build_manifest(cfg_like, attrs))
        self.budget_bytes = budget_bytes
        self._pinned: dict[str, Any] = {}     # keep each PinnedBlock alive
        self.arrays: dict[str, np.ndarray] = {}

        self._planned_bytes = domain_bytes(self.manifest, self.nz, self.ny,
                                           self.nx)
        if not allocate:
            return

        if reserve_bytes is None:
            reserve_bytes = host_reserve_bytes()
        # Gate the WHOLE request before taking a single page, so a refusal
        # leaves no partial allocation behind.
        check_allocatable(self._planned_bytes, budget_bytes=budget_bytes,
                          reserve_bytes=reserve_bytes,
                          max_total_fraction=max_total_fraction)
        try:
            self._allocate(reserve_bytes)
        except BaseException:
            self.free()                        # never leak a partial store
            raise

    # -- allocation ------------------------------------------------------

    def _allocate(self, reserve_bytes: int) -> None:
        # ``array`` is nulled in the ``finally`` because a raised exception's
        # traceback keeps THIS FRAME alive, and with it any local still bound
        # to a pinned block.  Without that, a failed allocation leaked exactly
        # one field -- MEASURED as 4.7 GiB of 62.2 GiB not coming back after a
        # refused 36.6 GiB store.  The traceback holds this very frame, so
        # rebinding the name here really does release the buffer.
        array = None
        try:
            for spec in self.manifest:
                shape = spec.shape(self.nz, self.ny, self.nx)
                nbytes = spec.nbytes(self.nz, self.ny, self.nx)
                # Re-check per field: another process may be growing
                # underneath us, and a long allocation loop is exactly when
                # that happens.
                available = host_memory()["available"]
                if nbytes > available - int(reserve_bytes):
                    raise HostMemoryExhausted(
                        f"ran out of host RAM allocating {spec.name!r} "
                        f"({nbytes / GIB:.2f} GiB): only "
                        f"{available / GIB:.2f} GiB available with "
                        f"{int(reserve_bytes) / GIB:.2f} GiB reserved")
                try:
                    # EXACT size, not cupy.cuda.alloc_pinned_memory: the pool
                    # would round this up to the next power of two and could
                    # consume 2.05x what the field needs.
                    array = alloc_pinned_array(shape, spec.dtype)
                except Exception as exc:
                    got = sum(a.nbytes for a in self.arrays.values())
                    raise PinnedAllocationFailed(
                        f"cudaHostAlloc refused {nbytes / GIB:.2f} GiB for "
                        f"{spec.name!r} after {got / GIB:.2f} GiB was "
                        f"already pinned (store wants "
                        f"{self._planned_bytes / GIB:.2f} GiB total), with "
                        f"MemAvailable at {available / GIB:.2f} GiB.  "
                        f"Page-locking walls system-wide at "
                        f"{PINNED_CEILING_FRACTION:.0%} of MemTotal = "
                        f"{pinned_ceiling_bytes() / GIB:.2f} GiB here "
                        f"(MEASURED to 64 MiB, invisible to /proc/meminfo, "
                        f"and NOT sensitive to block size), so this is either "
                        f"that wall or genuine competition from another "
                        f"process holding page-locked memory.  This store's "
                        f"largest block is "
                        f"{self.largest_field_bytes / GIB:.2f} GiB; see "
                        f"probe_pinned_ceiling.  No partial allocation was "
                        f"retained.") from exc
                self._pinned[spec.name] = owning_block(array)
                self.arrays[spec.name] = array
        finally:
            array = None
            del array

    def free(self) -> None:
        """Drop every buffer and return the pages to the OS.

        The store's own blocks are :class:`PinnedBlock`\\ s, which
        ``cudaFreeHost`` themselves as soon as the last reference goes -- so
        this drops the store's two dicts and forces a collection cycle, which
        is what makes the release PROMPT rather than eventual.  Release is
        refcount-driven on purpose: a caller still holding
        ``store.arrays['thp']`` keeps that one block alive instead of being
        handed a dangling pointer.

        ``free_all_blocks`` is still called for CuPy's pinned pool, which this
        store no longer allocates from but neighbouring code
        (``gather.pinned_empty``) does.  Call this between large stores or the
        second one will be refused by a guard that is, correctly, looking at
        real ``MemAvailable``.
        """
        self.arrays.clear()
        self._pinned.clear()
        gc.collect()
        try:
            import cupy as cp

            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:                      # pragma: no cover - teardown
            pass

    def __enter__(self) -> "HostDomainStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.free()

    # -- description -----------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.manifest)

    @property
    def nbytes(self) -> int:
        """Total bytes held (or planned, when built with ``allocate=False``)."""
        if self.arrays:
            return sum(a.nbytes for a in self.arrays.values())
        return self._planned_bytes

    @property
    def breakdown(self) -> dict[str, dict[str, Any]]:
        """Per-attribute shape, staggering, dtype and byte count."""
        out: dict[str, dict[str, Any]] = {}
        for spec in self.manifest:
            shape = spec.shape(self.nz, self.ny, self.nx)
            out[spec.name] = {
                "shape": shape, "stagger": spec.stagger_code,
                "dtype": str(spec.dtype),
                "bytes": spec.nbytes(self.nz, self.ny, self.nx)}
        return out

    @property
    def largest_field_bytes(self) -> int:
        """Size of the biggest single pinned block this store needs.

        No longer a capacity constraint: raw ``cudaHostAlloc`` served single
        blocks of 2 through 32 GiB on this box with no block-size sensitivity
        in the cumulative ceiling (see :func:`probe_pinned_ceiling`).  It used
        to be THE constraint, because CuPy's pool rounded it to a power of two
        -- :attr:`pinned_pool_bytes` quantifies what that cost.
        """
        return max(spec.nbytes(self.nz, self.ny, self.nx)
                   for spec in self.manifest)

    @property
    def pinned_pool_bytes(self) -> int:
        """What this store would consume through CuPy's pinned pool.

        Per-field :func:`pool_rounded_bytes`.  The ratio against
        :attr:`nbytes` is the waste the old allocator paid, and it is what
        made the achievable domain jump at ``n = 3277``.
        """
        return sum(pool_rounded_bytes(spec.nbytes(self.nz, self.ny, self.nx))
                   for spec in self.manifest)

    @property
    def pinned_pool_waste(self) -> float:
        """``pinned_pool_bytes / nbytes`` -- 1.0 means the pool cost nothing."""
        planned = domain_bytes(self.manifest, self.nz, self.ny, self.nx)
        return self.pinned_pool_bytes / float(planned or 1)

    @property
    def bytes_per_cell(self) -> float:
        """Achieved B/cell of THIS allocation -- measured, not assumed."""
        return self.nbytes / float(self.nz * self.ny * self.nx)

    def max_square_domain(self, budget_bytes: int,
                          nz: int | None = None) -> dict[str, Any]:
        """Largest square domain of this inventory fitting ``budget_bytes``."""
        return max_square_domain(budget_bytes, self.manifest,
                                 self.nz if nz is None else int(nz))

    def __repr__(self) -> str:                 # pragma: no cover - cosmetic
        return (f"<HostDomainStore {self.nz}x{self.ny}x{self.nx} "
                f"{len(self.manifest)} fields {self.nbytes / GIB:.3f} GiB>")

    # -- pinning verification -------------------------------------------

    def pinning_report(self) -> dict[str, Any]:
        """Assert page-locking per field via ``cudaPointerGetAttributes``.

        Returns ``{'all_pinned': bool, 'memory_types': {name: int}}`` where
        ``1`` is ``cudaMemoryTypeHost`` (pinned) and ``0`` is unregistered
        pageable memory.  Both values were confirmed experimentally on this
        box, so a ``0`` here is a real failure, not an API quirk.
        """
        import cupy as cp

        types: dict[str, int] = {}
        for name, array in self.arrays.items():
            ptr = array.__array_interface__["data"][0]
            try:
                types[name] = int(
                    cp.cuda.runtime.pointerGetAttributes(ptr).type)
            except Exception:                  # unregistered can also raise
                types[name] = 0
        return {"all_pinned": bool(types) and all(
            t == CUDA_MEMORY_TYPE_HOST for t in types.values()),
            "memory_types": types}

    def assert_pinned(self) -> None:
        report = self.pinning_report()
        if not report["all_pinned"]:
            bad = [n for n, t in report["memory_types"].items()
                   if t != CUDA_MEMORY_TYPE_HOST]
            raise HostStoreError(
                f"pinning did not take effect for {bad}; these buffers are "
                f"pageable and H2D will run at ~15 GB/s instead of ~57")

    # -- state transfer --------------------------------------------------

    def _device_arrays(self, state) -> dict[str, Any]:
        if self.inventory_fn is not None:
            return dict(self.inventory_fn(state))
        from tilestream import harness

        return harness.state_arrays(state)

    def _check_inventory(self, state) -> dict[str, Any]:
        device = self._device_arrays(state)
        mine, theirs = set(self.names), set(device)
        if mine != theirs:
            raise InventoryMismatch(
                f"store holds {sorted(mine)} but the DomainState carries "
                f"{sorted(theirs)}; missing from state "
                f"{sorted(mine - theirs)}, extra on state "
                f"{sorted(theirs - mine)}")
        for spec in self.manifest:
            name = spec.name
            want = spec.shape(self.nz, self.ny, self.nx)
            got = tuple(device[name].shape)
            if want != got:
                raise InventoryMismatch(
                    f"{name!r}: store shape {tuple(want)} != device shape "
                    f"{got}.  Check staggering -- u is (nz, ny, nx+1), v is "
                    f"(nz, ny+1, nx), w/php are (nz+1, ny, nx).")
        return device

    def fill_from(self, state) -> None:
        """Seed the store from a monolithic ``DomainState`` (D2H).

        Only valid when the whole domain fits on the GPU -- this is the
        correctness-gate path, not the production path.  Copies straight into
        the pinned buffers (``cupy.ndarray.get(out=...)``), then synchronises
        so the caller can read immediately.
        """
        import cupy as cp

        if not self.arrays:
            raise HostStoreError("store was built with allocate=False")
        device = self._check_inventory(state)
        for name in self.names:
            src = device[name]
            if not src.flags.c_contiguous:
                src = cp.ascontiguousarray(src)
            src.get(out=self.arrays[name])
        cp.cuda.runtime.deviceSynchronize()

    def to_device_state(self, state) -> None:
        """Write the whole store back into a ``DomainState`` (H2D).

        Only valid when the domain fits in VRAM; used by the bit-exact gate
        to prove the round trip loses nothing.  The production path never
        calls this -- it moves tiles, not domains.
        """
        import cupy as cp

        if not self.arrays:
            raise HostStoreError("store was built with allocate=False")
        device = self._check_inventory(state)
        for name in self.names:
            device[name].set(self.arrays[name])
        cp.cuda.runtime.deviceSynchronize()

    # -- identity --------------------------------------------------------

    def hash(self) -> str:
        """SHA-256 over the persisted set, byte-identical to the harness's.

        Uses ``harness._digest_payload`` itself (name, dtype string, int64
        shape, C-order bytes) over ``STATE_SERIALIZED_ATTRS`` order, so
        ``store.hash()`` and ``harness.hash_state(state)`` are directly
        comparable.  That equality IS the round-trip gate; it is not a
        coincidence to be re-derived, which is why the harness function is
        imported rather than copied.
        """
        from tilestream import harness

        if not self.arrays:
            raise HostStoreError("store was built with allocate=False")
        digest = hashlib.sha256()
        for name in self.names:
            digest.update(harness._digest_payload(f"state.{name}",
                                                  self.arrays[name]))
        return digest.hexdigest()

    def hash_field_map(self) -> dict[str, str]:
        """Per-field digests, for localizing a round-trip failure."""
        from tilestream import harness

        return {name: hashlib.sha256(
            harness._digest_payload(f"state.{name}", self.arrays[name])
        ).hexdigest() for name in self.names}

    # -- bandwidth -------------------------------------------------------

    def measure_h2d_bandwidth(self, *, min_bytes: int =
                              MIN_ACCURATE_BANDWIDTH_BYTES, repeats: int = 5,
                              ) -> dict[str, Any]:
        """Time a real H2D out of THIS store's own pinned memory.

        Transfers whole fields, largest first, until at least ``min_bytes``
        is covered, into one device buffer; times the batch with CUDA events.
        Refuses to report a number below ``min_bytes`` -- with a 10.9 us fixed
        per-transfer cost and PCIe only saturating at 32 MB, a small-store
        figure would be noise dressed as a result.

        Expect ~50-57 GB/s pinned on this box.  ~15 GB/s means pinning did
        not take effect.
        """
        import cupy as cp

        if not self.arrays:
            raise HostStoreError("store was built with allocate=False")
        if self.nbytes < min_bytes:
            return {"accurate": False, "bytes": self.nbytes,
                    "min_bytes": int(min_bytes), "gb_per_s": None,
                    "note": (f"store is only {self.nbytes / (1 << 20):.1f} MB; "
                             f"refusing to report a bandwidth below "
                             f"{min_bytes / (1 << 20):.0f} MB")}

        order = sorted(self.names, key=lambda n: -self.arrays[n].nbytes)
        selected: list[str] = []
        total = 0
        for name in order:
            selected.append(name)
            total += self.arrays[name].nbytes
            if total >= min_bytes:
                break

        buf = cp.cuda.alloc(total)
        stream = cp.cuda.Stream(non_blocking=True)
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        h2d = cp.cuda.runtime.memcpyHostToDevice

        def _batch() -> None:
            offset = 0
            for name in selected:
                array = self.arrays[name]
                cp.cuda.runtime.memcpyAsync(
                    int(buf.ptr) + offset,
                    array.__array_interface__["data"][0],
                    array.nbytes, h2d, stream.ptr)
                offset += array.nbytes

        with stream:
            _batch()                                  # warm up
            stream.synchronize()
            samples = []
            for _ in range(int(repeats)):
                start.record(stream)
                _batch()
                stop.record(stream)
                stop.synchronize()
                elapsed_ms = cp.cuda.get_elapsed_time(start, stop)
                samples.append(total / (elapsed_ms * 1e-3) / 1e9)
        buf = None  # rebound, not deleted: a closure above reads it
        return {"accurate": True, "bytes": total, "fields": selected,
                "n_transfers": len(selected),
                "gb_per_s": max(samples), "gb_per_s_median": float(
                    np.median(samples)), "samples_gb_per_s": samples,
                "pinned": self.pinning_report()["all_pinned"]}


# --------------------------------------------------------------------------
# capacity planning
# --------------------------------------------------------------------------

def _asymptotic_bytes_per_cell(manifest: Iterable[FieldSpec],
                               nz: int) -> float:
    """B/cell in the large-domain limit (staggering ``+1``s amortised away).

    3-D fields contribute ``itemsize`` (times ``(nz+1)/nz`` when
    z-staggered); 2-D surface fields contribute ``itemsize / nz``.
    """
    total = 0.0
    for spec in manifest:
        if spec.fixed_shape is not None:
            continue  # Domain vectors do not scale with horizontal area.
        planes = int(np.prod(spec.leading_shape))
        if spec.has_z:
            zext = nz + int(spec.z_stagger)
            total += planes * spec.dtype.itemsize * zext / nz
        else:
            total += planes * (spec.layers or 1) * spec.dtype.itemsize / nz
    return total


__all__ = [
    "BudgetExceeded", "CUDA_MEMORY_TYPE_HOST", "DEFAULT_MAX_TOTAL_FRACTION",
    "DEFAULT_RESERVE_BYTES", "FieldSpec", "HostDomainStore",
    "RESERVE_TOTAL_FRACTION", "host_reserve_bytes",
    "HostMemoryExhausted", "HostStoreError", "InventoryMismatch",
    "MIN_ACCURATE_BANDWIDTH_BYTES", "PINNED_CEILING_FRACTION", "PinnedBlock",
    "alloc_pinned_array", "build_manifest", "bytes_per_cell",
    "check_allocatable", "domain_bytes", "host_memory",
    "manifest_from_arrays",
    "max_square_domain", "owning_block", "PinnedAllocationFailed",
    "pinned_ceiling_bytes", "pinned_ledger",
    "pool_rounded_bytes", "probe_pinned_ceiling",
]
