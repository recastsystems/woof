"""Partition-local views of the existing regional LBC file pool, and the process's shared boundary-file reads.

Only entity selection and read scheduling live here. The shipped reader (woof.hex.lbc.read_lbc_file), stream admission
and CUDA derivation/interpolation remain the authorities for field values.
"""
from __future__ import annotations

import itertools
import sys
import threading
import time
import weakref
from concurrent.futures import Future
from pathlib import Path

import numpy as np

from .lbc import LbcFile, LbcPool, read_lbc_file


def project_lbc_file(admitted, layout):
    fields = {}
    for name, array in admitted.fields.items():
        indexes = layout.edge_l2g if name == "lbc_u" else layout.cell_l2g
        expected = layout._global_edges if name == "lbc_u" else layout._global_cells
        if array.shape[0] != expected:
            raise ValueError(f"{admitted.path}: {name} has {array.shape[0]} entities, expected {expected}")
        selected = np.ascontiguousarray(array[np.asarray(indexes, dtype=np.int64)])
        selected.setflags(write=False)
        fields[name] = selected
    return LbcFile(admitted.path, admitted.valid_time, fields)


def layout_rows(layout):
    """``name -> (indexes, global count)``: the rows ``project_lbc_file`` cuts for ``layout`` (read_lbc_file ``rows``)."""
    cells = np.asarray(layout.cell_l2g, dtype=np.int64)
    edges = np.asarray(layout.edge_l2g, dtype=np.int64)

    def rows(name):
        return (edges, int(layout._global_edges)) if name == "lbc_u" else (cells, int(layout._global_cells))

    return rows


def read_lbc_rows(path, extra_scalars=(), layout=None) -> LbcFile:
    """One boundary file, only ``layout``'s rows: ``project_lbc_file(read_lbc_file(path), layout)`` without the whole file."""
    return read_lbc_file(path, extra_scalars=tuple(extra_scalars), rows=layout_rows(layout))


class SharedLbcReads:
    """One read of each boundary file per process, shared by every consumer (same-host rank) of one series.

    A boundary file carries full-mesh fields (woof.hex.lbc), and each rank's pool used to read the whole file, keep it
    and slice its own rows.  THE BREAKAGE THIS FIXES: on NOAA's conus3km (2.34 M cells, 13.3 GB per boundary hour) 8
    rank threads each read 13.3 GB per boundary hour on their step path and each held a full copy until the next hour
    (c3-lam, 2026-10-28).  Here the first consumer to ask for a file reads it and the others wait for that read; the
    full file is dropped once every registered consumer has taken it.  A consumer that registers after a file was
    dropped reads it again: a slower start, the same words.
    """

    def __init__(self, read=read_lbc_file, read_rows=read_lbc_rows):
        self._read = read
        self._read_rows = read_rows
        self._lock = threading.Lock()
        self._ids = itertools.count()
        self._consumers: set[int] = set()
        self._entries: dict[tuple, tuple[Future, set]] = {}
        self.reads: list[tuple[str, float]] = []

    def register(self) -> int:
        with self._lock:
            consumer = next(self._ids)
            self._consumers.add(consumer)
            return consumer

    def sole(self, consumer: int) -> bool:
        """Whether ``consumer`` is the only consumer of this series in this process (one rank process per card)."""
        with self._lock:
            return self._consumers == {consumer}

    def rows(self, consumer: int, path, extra_scalars, layout) -> LbcFile:
        """The sole consumer's rows of one file, read without the whole file (read_lbc_rows).

        THE COST THIS REMOVES (NOAA RRFS v2 conus3km, V8X6000, 2026-10-29): with one process per card every process
        is its series' only consumer, so ``take`` read each 13.3 GB boundary file whole in every rank process to keep
        a quarter or an eighth of it (4 cards: every rank read 12Z, 13Z and 14Z whole in a 1 h run), while the rank
        processes spent more CPU time in the kernel than in user code (8 cards: 149-199 s system per child in 1 h,
        340-390 s in 6 h).  The rows reader cuts a process's boundary read cost about 13x in system time and 4x in
        user time on NOAA's conus12km files (classic_netcdf.read_rows has the numbers).  Same words:
        ``read_lbc_rows`` returns what ``project_lbc_file`` cuts from the whole file.
        """
        began = time.perf_counter()
        admitted = self._read_rows(path, tuple(extra_scalars), layout)
        seconds = time.perf_counter() - began
        with self._lock:
            self.reads.append((str(path), seconds))
        print(f"lbc-rows-read: {Path(path).name} {len(layout.cell_l2g)} of {layout._global_cells} cells read in "
              f"{seconds:.2f} s [thread {threading.current_thread().name}]", file=sys.stderr, flush=True)
        return admitted

    def take(self, consumer: int, path, extra_scalars=()) -> LbcFile:
        key = (str(path), tuple(extra_scalars))
        with self._lock:
            entry = self._entries.get(key)
            owner = entry is None
            if owner:
                entry = self._entries[key] = (Future(), set())
        if owner:
            began = time.perf_counter()
            try:
                entry[0].set_result(self._read(path, extra_scalars=tuple(extra_scalars)))
            except BaseException as error:
                entry[0].set_exception(error)
                with self._lock:
                    if self._entries.get(key) is entry:
                        del self._entries[key]
                raise
            seconds = time.perf_counter() - began
            with self._lock:
                self.reads.append((str(path), seconds))
                consumers = len(self._consumers)
            print(f"lbc-shared-read: {Path(path).name} read once in {seconds:.2f} s for {consumers} consumer(s) "
                  f"[thread {threading.current_thread().name}]", file=sys.stderr, flush=True)
        admitted = entry[0].result()
        with self._lock:
            entry[1].add(consumer)
            if self._entries.get(key) is entry and self._consumers <= entry[1]:
                del self._entries[key]
        return admitted


# One SharedLbcReads per boundary series (LbcInventory.identity) in this process, alive while a consumer holds it.
_SHARED: "weakref.WeakValueDictionary[tuple, SharedLbcReads]" = weakref.WeakValueDictionary()
_SHARED_LOCK = threading.Lock()


def shared_lbc_reads(inventory) -> SharedLbcReads:
    key = getattr(inventory, "identity", None) or ("inventory", id(inventory))
    with _SHARED_LOCK:
        shared = _SHARED.get(key)
        if shared is None:
            shared = SharedLbcReads()
            _SHARED[key] = shared
        return shared


class RankLbcReader:
    """A consumer's boundary reads: the shared file, cut to the consumer's rows, and the next file read ahead.

    Installed as the consumer's LbcPool reader (share_lbc_reads).  After each admission a thread takes the next file
    (LbcInventory.earliest_strictly_after the admitted file's valid time) from the shared reads and cuts it, so the
    next lbc_in admission finds its rows in host memory instead of reading on the step path.  The rows are the file's
    own words either way.  A read-ahead for another file than the one asked for, or one that failed (a live series
    whose next file is not written yet), is dropped and the admission reads as it always did, so the read-ahead can
    change only when a read happens, never what an admission returns or which error it raises.
    """

    def __init__(self, shared: SharedLbcReads, inventory, layout=None):
        self.shared = shared
        self.inventory = inventory
        self.layout = layout
        self.consumer = shared.register()
        self._ahead: Future | None = None

    def _rows(self, path, extra_scalars) -> LbcFile:
        if self.layout is not None and self.shared.sole(self.consumer):
            # The only consumer in this process (a rank process): its rows, never the whole file (SharedLbcReads.rows).
            return self.shared.rows(self.consumer, path, extra_scalars, self.layout)
        admitted = self.shared.take(self.consumer, path, extra_scalars)
        return admitted if self.layout is None else project_lbc_file(admitted, self.layout)

    def __call__(self, path, extra_scalars=()) -> LbcFile:
        extra_scalars = tuple(extra_scalars)
        ahead, self._ahead = self._ahead, None
        admitted = None
        if ahead is not None:
            try:
                ahead_path, ahead_scalars, rows = ahead.result()
                if str(ahead_path) == str(path) and ahead_scalars == extra_scalars:
                    admitted = rows
            except Exception:
                admitted = None
        if admitted is None:
            admitted = self._rows(path, extra_scalars)
        self._read_ahead(admitted.valid_time, extra_scalars)
        return admitted

    def _read_ahead(self, valid_time, extra_scalars) -> None:
        future: Future = Future()

        def work():
            try:
                path = self.inventory.earliest_strictly_after(valid_time)
                future.set_result((path, extra_scalars, self._rows(path, extra_scalars)))
            except BaseException as error:  # the last file of the series, or one not written yet: read on admission
                future.set_exception(error)

        self._ahead = future
        threading.Thread(target=work, name=f"lbc-read-ahead-{self.consumer}", daemon=True).start()


def share_lbc_reads(pool: LbcPool, layout=None) -> LbcPool:
    """Give ``pool`` the process's shared reads of its series, cut to ``layout``'s rows (None: every row)."""

    pool.reader = RankLbcReader(shared_lbc_reads(pool.inventory), pool.inventory, layout)
    return pool


class PartitionLbcPoolV841:
    """Delegate the native two-level admission, then gather each rank's rows.

    A real LbcPool gets the shared reader (share_lbc_reads), which hands it the rank's rows directly, so the pool never
    holds a full-mesh file; any other pool is cut after admission, as before.
    """

    def __init__(self, pool: LbcPool, layout):
        self.pool = pool
        self.layout = layout
        self._cut_after = not isinstance(pool, LbcPool)
        if not self._cut_after:
            share_lbc_reads(pool, layout)

    @property
    def interval_end(self):
        return self.pool.interval_end

    def start(self, when):
        admitted = self.pool.start(when)
        return project_lbc_file(admitted, self.layout) if self._cut_after else admitted

    def advance(self, when=None):
        admitted = self.pool.advance(when)
        return project_lbc_file(admitted, self.layout) if self._cut_after else admitted
