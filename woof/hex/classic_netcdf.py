"""Read whole numeric variables of a classic (CDF-1/2/5) netCDF file on several threads.

THE COST THIS REMOVES (30 km globe, x1.655362, V5090 box, 2026-10-27): ``Mesh.from_netcdf`` read every grid and
init variable through netCDF4-python one after another on the forecast's main thread (netCDF-C is not thread safe),
about 13 s of each 30 km member start while four cards waited.  A classic file stores each variable's words
big-endian at an offset its header names, so the words can be read straight from those offsets on many threads
and byte-swapped there (os.preadv and the NumPy cast release the GIL).

The arrays are the ones netCDF4-python returns for the same variable with masking off: the same native-endian
dtype, the same shape (a record variable of a one-record file keeps its leading record axis) and the same words
(the swap is exact).  Anything this reader does not handle the same way returns None for that variable, and the
caller reads it through netCDF4 as before: character data, 0-d variables, attributes netCDF4 acts on
(``scale_factor``, ``add_offset``, ``_Unsigned``), and record variables of a file with other than one record.

THE COST ``WholeReads`` REMOVES (own24 zoom globe, 167,254 cells x 55 levels, DA member leg on an RTX 5090,
hex 0.4.9, 2026-10-28): after the mesh, the vertical-grid and initial-state loaders (woof.hex.driver) still read
about 1 GB of the init (zb, zb3, zxu, zgrid, zz, dss, rho, theta, qv, u, w, rho_base, theta_base) through
netCDF4 one variable after another on the main thread while the card waited.  ``WholeReads`` starts every one of
those reads at once on host threads and the loader takes each array when it reaches it; netCDF4 still serves the
metadata and every variable this reader does not handle, as before.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping

import numpy as np

_TYPES = {1: "i1", 3: ">i2", 4: ">i4", 5: ">f4", 6: ">f8", 7: "u1", 8: ">u2", 9: ">u4", 10: ">i8", 11: ">u8"}
_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8, 7: 1, 8: 2, 9: 4, 10: 8, 11: 8}
_ACTED_ON = ("scale_factor", "add_offset", "_Unsigned")
_CHUNK = 1 << 30


@dataclass(frozen=True)
class Extent:
    """Where one variable's words lie: ``count`` values of big-endian ``dtype`` from byte ``begin``."""

    dtype: str | None
    shape: tuple[int, ...]
    begin: int
    record: bool
    #: The variable's attribute names as the header lists them (``readable`` needs them).
    attributes: tuple[str, ...] = ()


class _Header:
    def __init__(self, data: bytes, version: int) -> None:
        self.data, self.offset, self.version = data, 0, version

    def take(self, size: int) -> bytes:
        if self.offset + size > len(self.data):
            raise EOFError
        value = self.data[self.offset:self.offset + size]
        self.offset += size
        return value

    def u32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def count(self) -> int:
        """NON_NEG / nelems: 8 bytes in CDF-5, 4 before."""
        return struct.unpack(">Q", self.take(8))[0] if self.version == 5 else self.u32()

    def name(self) -> str:
        size = self.count()
        text = self.take(size)
        self.take(-size % 4)
        return text.decode("utf-8")

    def attributes(self) -> tuple[str, ...]:
        tag, count = self.u32(), self.count()
        if tag == 0:
            return ()
        names = []
        for _ in range(count):
            names.append(self.name())
            kind, values = self.u32(), self.count()
            size = _SIZES[kind] * values
            self.take(size + (-size % 4))
        return tuple(names)


def extents(path: str | Path) -> tuple[dict[str, Extent], int] | None:
    """Every variable's extent and the record count, or None when ``path`` is not a classic netCDF file."""

    path = Path(path)
    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic[:3] != b"CDF" or magic[3] not in (1, 2, 5):
            return None
        version = magic[3]
        # The header holds names, attributes and offsets only; read it in growing pieces until it parses.
        size = 1 << 20
        while True:
            handle.seek(0)
            data = handle.read(size)
            try:
                return _parse(data, version)
            except EOFError:
                if len(data) < size:
                    return None
                size *= 4


def _parse(data: bytes, version: int) -> tuple[dict[str, Extent], int]:
    header = _Header(data, version)
    header.take(4)
    numrecs = header.count()
    dimensions: list[int] = []
    tag, count = header.u32(), header.count()
    if tag not in (0, 0x0A):
        raise ValueError("classic netCDF dimension list tag")
    for _ in range(count if tag else 0):
        header.name()
        dimensions.append(header.count())
    header.attributes()
    tag, count = header.u32(), header.count()
    if tag not in (0, 0x0B):
        raise ValueError("classic netCDF variable list tag")
    found: dict[str, Extent] = {}
    for _ in range(count if tag else 0):
        name = header.name()
        ids = [header.count() for _ in range(header.count())]
        attributes = header.attributes()
        kind = header.u32()
        header.count() if version == 5 else header.u32()  # vsize
        begin = struct.unpack(">Q" if version in (2, 5) else ">I", header.take(8 if version in (2, 5) else 4))[0]
        lengths = [dimensions[i] for i in ids]
        record = bool(lengths) and lengths[0] == 0
        shape = tuple([numrecs] + lengths[1:]) if record else tuple(lengths)
        found[name] = Extent(_TYPES.get(kind), shape, begin, record, attributes)
    return found, numrecs


def readable(extent: Extent, numrecs: int, attributes: Iterable[str]) -> bool:
    """Whether this reader returns netCDF4-python's array for the variable (see the module docstring)."""

    return (extent.dtype is not None and len(extent.shape) > 0
            and not (extent.record and numrecs != 1)
            and not any(name in _ACTED_ON for name in attributes))


def _read(path: Path, extent: Extent) -> np.ndarray:
    words = np.empty(extent.shape, dtype=extent.dtype)
    view = memoryview(words.reshape(-1).view(np.uint8))
    with path.open("rb", buffering=0) as handle:
        fd = handle.fileno()
        done = 0
        while done < len(view):
            got = os.preadv(fd, [view[done:done + _CHUNK]], extent.begin + done)
            if got <= 0:
                raise EOFError(f"{path.name}: short read of a variable")
            done += got
    return words.astype(words.dtype.newbyteorder("="))


#: A variable larger than this is read and byte-swapped in blocks of this size on several threads.
_BLOCK = 1 << 24


class _Pending:
    """One variable being read in blocks: its native-order array and the futures of its blocks."""

    __slots__ = ("array", "blocks")

    def __init__(self, array: np.ndarray, blocks: list[Future[None]]) -> None:
        self.array, self.blocks = array, blocks

    def result(self) -> np.ndarray:
        for block in self.blocks:
            block.result()
        return self.array

    def cancel(self) -> None:
        for block in self.blocks:
            block.cancel()


def _read_block(path: Path, begin: int, flat: np.ndarray, swap: bool) -> None:
    """Read ``flat``'s words from byte ``begin`` and put them in native order, in place."""

    view = memoryview(flat.view(np.uint8))
    with path.open("rb", buffering=0) as handle:
        fd = handle.fileno()
        done = 0
        while done < len(view):
            got = os.preadv(fd, [view[done:done + _CHUNK]], begin + done)
            if got <= 0:
                raise EOFError(f"{path.name}: short read of a variable")
            done += got
    if swap:
        flat.byteswap(inplace=True)


def _submit(pool: ThreadPoolExecutor, path: Path, extent: Extent) -> _Pending:
    """Queue ``extent``'s blocks on ``pool``; the array ends up holding the words ``_read`` returns.

    The file's big-endian words land in a native-order array and are swapped there (``byteswap`` moves the same
    bytes ``astype`` to the native order moves), so each word is copied once and the array owns its memory, as
    ``_read``'s does.
    """

    stored = np.dtype(extent.dtype)
    native = stored.newbyteorder("=")
    array = np.empty(extent.shape, dtype=native)
    flat = array.reshape(-1)
    step = max(1, _BLOCK // native.itemsize)
    blocks = [pool.submit(_read_block, path, extent.begin + start * native.itemsize, flat[start:start + step],
                          native != stored)
              for start in range(0, flat.size, step)]
    return _Pending(array, blocks)


def read_variables(path: str | Path, names: Iterable[str], table: Mapping[str, Extent], *,
                   workers: int) -> dict[str, Any]:
    """The named variables' arrays, read on ``workers`` threads, keyed by name."""

    path = Path(path)
    names = list(names)
    if workers <= 1:
        return {name: _read(path, table[name]) for name in names}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-classic-read") as pool:
        pending = {name: _submit(pool, path, table[name]) for name in names}
        return {name: value.result() for name, value in pending.items()}


#: Rows gathered per thread task in :func:`read_rows` (about 16 MiB of words at a few hundred bytes a row).
_ROW_BLOCK = 1 << 16


def read_rows(path: str | Path, names: Iterable[str], table: Mapping[str, Extent],
              rows: Mapping[str, np.ndarray], *, workers: int) -> dict[str, np.ndarray]:
    """Only the rows ``rows[name]`` (axis 0, after a one-record file's record axis) of each named variable.

    THE COST THIS REMOVES (NOAA RRFS v2 conus3km, one rank process per card, 2026-10-29): every rank process read
    every boundary file whole (13.3 GB a boundary hour, ``read_variables`` into fresh memory on up to 32 threads per
    process) to keep a quarter or an eighth of it, while the ranks spent more CPU time in the kernel than in user
    code.  Measured on an RTX 5090 host with NOAA's conus12km files (two 634 MB boundary hours, 8 processes at once, METIS
    8-way cut): 0.82-0.95 s of system and 0.46-0.50 s of user time per process, 298 k page faults and 600 MB RSS
    each; this reader: 0.06-0.07 s system, 0.12-0.13 s user, 44 k faults, 120 MB, the same rows to the bit.
    Each variable's byte range is mapped read-only, its rows are gathered on host threads straight from the page
    cache (which every rank shares), and only the gathered rows are byte-swapped and kept.  The words are the ones
    ``read_variables`` returns for those rows: the same stored bytes, swapped the same way.

    The caller checks the variables are readable (``readable``) and the indexes are in range.
    """

    import mmap

    path = Path(path)
    out: dict[str, np.ndarray] = {}
    granularity = mmap.ALLOCATIONGRANULARITY
    with path.open("rb", buffering=0) as handle, ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix="hex-classic-rows") as pool:
        fd = handle.fileno()
        for name in names:
            extent = table[name]
            shape = tuple(extent.shape[1:]) if extent.record else tuple(extent.shape)
            stored = np.dtype(extent.dtype)
            native = stored.newbyteorder("=")
            count = int(np.prod(shape, dtype=np.int64))
            indexes = np.asarray(rows[name], dtype=np.int64)
            if indexes.ndim != 1 or (indexes.size and (int(indexes.min()) < 0 or int(indexes.max()) >= shape[0])):
                raise IndexError(f"{path.name}: {name} rows outside 0..{shape[0] - 1}")
            start = extent.begin - extent.begin % granularity
            length = extent.begin - start + count * stored.itemsize
            result = np.empty((indexes.size,) + shape[1:], dtype=stored)
            if indexes.size and count:
                mapping = mmap.mmap(fd, length, offset=start, access=mmap.ACCESS_READ)
                try:
                    if hasattr(mapping, "madvise") and hasattr(mmap, "MADV_WILLNEED"):
                        mapping.madvise(mmap.MADV_WILLNEED)
                    words = np.frombuffer(mapping, dtype=stored, count=count,
                                          offset=extent.begin - start).reshape(shape)
                    futures = [pool.submit(np.take, words, indexes[block:block + _ROW_BLOCK], 0,
                                           result[block:block + _ROW_BLOCK], "clip")
                               for block in range(0, indexes.size, _ROW_BLOCK)]
                    for future in futures:
                        future.result()
                    del futures, words
                finally:
                    mapping.close()
            if native != stored:
                result.byteswap(inplace=True)
                result = result.view(native)
            out[name] = result
    return out


def disabled() -> bool:
    """``WOOF_HEX_CLASSIC_READ=off`` sends every read back through netCDF4."""

    return os.environ.get("WOOF_HEX_CLASSIC_READ", "").strip().lower() in ("off", "0", "no")


class WholeReads:
    """Whole-variable reads of one classic file, all started now on host threads, each taken once when needed.

    ``take`` returns the array netCDF4-python returns for ``variable[...]`` (see the module docstring), or None when
    this reader does not handle the variable, the read failed, or its shape or dtype differ from the caller's
    netCDF4 metadata; the caller then reads that variable through netCDF4 as before.  Blocks are queued in the
    order of ``names``, so name them in the order they are taken.
    """

    def __init__(self, path: str | Path, names: Iterable[str], *, workers: int) -> None:
        self.path = Path(path).expanduser().resolve()
        self._pending: dict[str, _Pending] = {}
        self._pool: ThreadPoolExecutor | None = None
        if workers <= 1 or disabled():
            return
        try:
            found = extents(self.path)
        except (OSError, ValueError, KeyError, struct.error):
            found = None
        if found is None:
            return
        table, numrecs = found
        wanted = [name for name in dict.fromkeys(names)
                  if name in table and readable(table[name], numrecs, table[name].attributes)]
        if not wanted:
            return
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-classic-read")
        for name in wanted:
            self._pending[name] = _submit(self._pool, self.path, table[name])

    def take(self, name: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray | None:
        pending = self._pending.pop(name, None)
        if pending is None:
            return None
        try:
            value = pending.result()
        except (OSError, EOFError, ValueError):
            return None
        if value.shape != tuple(shape) or value.dtype != np.dtype(dtype):
            return None
        return value

    def close(self) -> None:
        """Drop every read not taken; a block still running finishes on its thread and is discarded."""

        pending, self._pending = self._pending, {}
        for value in pending.values():
            value.cancel()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __enter__(self) -> "WholeReads":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
