"""One arena for the large work buffers the rate classes share on one card (one-card active LTS).

The one-card class executor (``drivers.fullstep_lts_forecast_v841.execute_fullstep_lts``) keeps
one resident driver and one physics stack per rate class and steps the classes one after
another on one stream.  Every class step allocates its own large work buffers -- the
Grell-Freitas column workspace (``woof.core.gf``: 2.81 GB for a class wider than the card's
tile on an RTX 5090, 2.66 GB for an 82,547-column class) and the scheme's per-column slabs --
and hands them back to the pool.  In CuPy's one pool a freed block that size is split by the
classes' smaller arrays as soon as it is free, so the next class step finds no contiguous block
although most of the card is idle.

THE BREAKAGE THIS PREVENTS (hex 48/3 km Milton storm-following, 353,352 cells, four rate
classes, one RTX 5090, reference system 2026-10-27): with the scalar limiter already one species per pass
(cuda_fullstep_lts_active._Exchanger.fct_scale_parts), the run stopped at base step 88 in
Grell-Freitas with "Out of memory allocating 2,807,562,240 bytes (allocated so far:
31,356,196,352 bytes)" while 20.7 GiB was live: 8.5 GiB of the pool was free in pieces no
2.8 GB request could use.

Here every request of at least :data:`THRESHOLD_BYTES` is served from one separate pool that the
classes share and that no small array can split.  A request no free block of that pool can hold
first hands the pool's free blocks back, so it settles on the largest set of work buffers one
class step holds at once instead of keeping one block per class.  Smaller requests stay in the
default pool.  Only addresses change; no launch or arithmetic does.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

#: Requests of at least this many bytes are served from the shared arena.
THRESHOLD_BYTES = 512 << 20

#: Every arena this process made.  An array the executor hands out (a mailbox, a frame) can
#: outlive the executor, and CuPy frees a pool's device blocks when the pool object goes, so
#: an arena lives as long as the process.
_ARENAS: list["SharedLargeWorkBuffers"] = []


class SharedLargeWorkBuffers:
    """The allocator installed while the class executor runs on one card."""

    def __init__(self, cp: Any, threshold: int = THRESHOLD_BYTES) -> None:
        self.cp = cp
        self.threshold = int(threshold)
        self.small_pool = cp.get_default_memory_pool()
        self.small = cp.cuda.get_allocator()
        self.large = cp.cuda.MemoryPool()
        self.requests = 0
        self.regrowths = 0
        self.peak_bytes = 0

    def malloc(self, size: int) -> Any:
        oom = self.cp.cuda.memory.OutOfMemoryError
        size = int(size)
        if size < self.threshold:
            try:
                return self.small(size)
            except oom:
                # The arena's free blocks are card memory too.
                self.large.free_all_blocks()
                return self.small(size)
        self.requests += 1
        if self.large.free_bytes() < size:
            # No free block here can hold it: give the free ones back before growing, so the
            # arena does not keep a smaller block beside the larger one it now needs.
            if self.large.free_bytes():
                self.regrowths += 1
            self.large.free_all_blocks()
        try:
            block = self.large.malloc(size)
        except oom:
            self.small_pool.free_all_blocks()
            block = self.large.malloc(size)
        self.peak_bytes = max(self.peak_bytes, int(self.large.total_bytes()))
        return block

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "gpuwm-hex.class-work-arena/v1",
            "threshold_bytes": self.threshold,
            "large_requests": self.requests,
            "regrowths": self.regrowths,
            "arena_peak_bytes": self.peak_bytes,
            "arena_bytes_at_end": int(self.large.total_bytes()),
            "default_pool_bytes_at_end": int(self.small_pool.total_bytes()),
            "scope": ("one card, rate classes stepping one after another on one stream: requests of at "
                      "least threshold_bytes share one pool no small array can split"),
        }


@contextmanager
def shared_large_work_buffers(cp: Any, *, threshold: int = THRESHOLD_BYTES) -> Iterator[SharedLargeWorkBuffers]:
    """Install the arena for the executor's lifetime and restore the previous allocator after."""

    arena = SharedLargeWorkBuffers(cp, threshold)
    _ARENAS.append(arena)
    previous = cp.cuda.get_allocator()
    cp.cuda.set_allocator(arena.malloc)
    try:
        yield arena
    finally:
        cp.cuda.set_allocator(previous)
