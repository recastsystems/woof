"""Order-preserving thread fan-out for the preparation's bulk host passes.

THE BREAKAGE.  A GFS preparation's main thread spent most of its
initialization at 100 % of one core while every other core sat idle
(shared host 224 cores, 2026-10-27: ``woof.gfs_direct`` at
PREPARING_INITIALIZE_ALL_TIMES 98-101 % of one core for 70-190 s per
3 km nest).  Profiled with an in-process frame sampler on a desktop host (3 km
376x362x50 nest, 13 GFS leads, CPU backend): reading each decoded lead
(``np.fromfile`` + float64 widening), the snapshot's defensive copy of
every field, hashing the decoded arrays and the GRIBs, and the mixed-phase
RH conversion were each one long serial NumPy/hashlib pass over hundreds
of MB per lead.  Every one of those passes is either per-file or
elementwise, and NumPy's fread/cast/copy loops and hashlib release the
GIL on large buffers, so a thread per item runs them on all cores and
gives byte-identical results: each item's bytes are produced by exactly
the code that produced them serially, and results come back in input
order.

ONE POOL PER PROCESS.  The threads are started once and reused: a pool per
call started ~1,000 threads per preparation, and on a loaded host
(load 50-76 on 24 cores) starting them cost 10 % of the main thread.  A
call made from one of these threads runs inline (no pool waits on
itself), and a forked child starts its own pool.

The worker count is the preparation's automatic host count
(``woof.ingest.cpu_backend.automatic_workers``: affinity, the cgroup CPU
quota, host memory and ``WOOF_PREPROCESS_THREADS``), capped at the
number of items.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None
_pool_pid: int | None = None
_pool_size = 0
_inside = threading.local()


def host_threads(items: int, workers: int | None = None) -> int:
    """Threads for ``items`` independent host passes (at least 1)."""

    if items < 2:
        return 1
    if workers is None:
        from woof.ingest.cpu_backend import automatic_workers
        workers = automatic_workers()
    return max(1, min(int(items), int(workers)))


def _mark_worker() -> None:
    _inside.active = True


def _shared_pool(threads: int) -> ThreadPoolExecutor:
    global _pool, _pool_pid, _pool_size
    with _lock:
        if _pool is None or _pool_pid != os.getpid() or _pool_size < threads:
            from woof.ingest.cpu_backend import automatic_workers
            size = max(threads, automatic_workers())
            # A pool from before a fork has no threads in this process, and
            # a smaller one is replaced.  The replaced pool is NOT shut down: a
            # shared_executor block still running holds it and submits to it,
            # and a shut pool refuses ("cannot schedule new futures after
            # shutdown": an HRRR mean prep on a 96-core box, OK-WOFS
            # 2026-11-02 20:40Z).  It finishes its queue, and its idle threads
            # exit once nothing references it.
            _pool = ThreadPoolExecutor(max_workers=size,
                                       thread_name_prefix="gpuwm-host",
                                       initializer=_mark_worker)
            _pool_pid = os.getpid()
            _pool_size = size
        return _pool


def map_ordered(function: Callable[[T], R], items: Iterable[T], *,
                workers: int | None = None) -> list[R]:
    """``[function(item) for item in items]`` on host threads, in order.

    The first exception raised by any item is raised here, as the serial
    loop would raise it (later items may already have run).  One item, one
    worker, or a call from inside a host thread runs inline.
    """

    items = list(items)
    threads = host_threads(len(items), workers)
    if threads == 1 or getattr(_inside, "active", False):
        return [function(item) for item in items]
    pool = _shared_pool(threads)
    if threads >= len(items):
        futures = [pool.submit(function, item) for item in items]
    else:
        # At most ``threads`` at once: contiguous runs of items, one per
        # task, each run in order.
        edges = [len(items) * k // threads for k in range(threads + 1)]
        runs = [items[a:b] for a, b in zip(edges[:-1], edges[1:]) if b > a]
        futures = [pool.submit(lambda run: [function(item) for item in run], run)
                   for run in runs]
    results: list = []
    error = None
    for future in futures:
        try:
            value = future.result()
        except BaseException as caught:  # noqa: BLE001 - re-raised below
            if error is None:
                error = caught
            continue
        if error is None:
            if threads >= len(items):
                results.append(value)
            else:
                results.extend(value)
    if error is not None:
        raise error
    return results


class _ScopedExecutor:
    """``submit``/``map`` onto the process pool; the block waits for what it submitted.

    Stands in for a ``with ThreadPoolExecutor(max_workers=n)`` block: the
    same calls, the same results, and leaving the block waits for every task
    it submitted, as ``shutdown(wait=True)`` did -- without starting and
    joining ``n`` threads each time.  A call from inside a host thread runs
    inline and returns a finished future (a pool never waits on itself).
    """

    def __init__(self, pool: ThreadPoolExecutor):
        self._pool = pool
        self._futures: list = []

    def submit(self, function, *args, **kwargs):
        if getattr(_inside, "active", False):
            from concurrent.futures import Future
            future: Future = Future()
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as error:  # noqa: BLE001 - carried by the future
                future.set_exception(error)
            return future
        future = self._pool.submit(function, *args, **kwargs)
        self._futures.append(future)
        return future

    def map(self, function, *iterables, timeout=None, chunksize=1):
        futures = [self.submit(function, *values) for values in zip(*iterables)]

        def results():
            for future in futures:
                yield future.result(timeout)
        return results()


class _scoped:
    def __init__(self, max_workers: int):
        self.max_workers = max(1, int(max_workers))

    def __enter__(self) -> _ScopedExecutor:
        self.executor = _ScopedExecutor(_shared_pool(self.max_workers))
        return self.executor

    def __exit__(self, *exc) -> None:
        from concurrent.futures import wait
        wait(self.executor._futures)


def shared_executor(max_workers: int) -> _scoped:
    """``with shared_executor(n) as executor:`` -- a ThreadPoolExecutor block on the process pool."""

    return _scoped(max_workers)
