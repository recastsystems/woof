"""Bit-exact host copies and elementwise expressions, split across threads.

THE COST THIS REMOVES, MEASURED (rented box, 2026-10-13, 20 km global, 1,472,532
cells x 55 levels, warm host preparation sampled on the main thread):
reading the init's level-by-entity fields spent 20.2 s in one line,
``np.transpose(value, axes).copy()``, a strided copy NumPy runs on one core,
and the coupled ``ru`` gather over 4,417,590 edges spent 8.9 s, also on one
core, while 380 cores sat idle and every card waited for the first step.

WHAT IS EXACT ABOUT IT.  A transpose copy moves every element to one fixed
place and does no arithmetic, so splitting it into column blocks writes the
same bytes.  An elementwise expression (``+``, ``*``, a gather) computes each
output element from the same input elements with the same NumPy ufuncs in
the same order whatever block that element falls in; no reduction or
reassociation crosses a block, so the bytes are the ones the whole-array
expression writes.  Callers keep their single-call form for small arrays.

The interpreter runs without the GIL (``PYTHON_GIL=0``) and NumPy releases it
inside its loops anyway, so blocks run concurrently.  No netCDF call is made
here: reads stay on the caller's thread (samehost_serial_io_v841).
``WOOF_HEX_HOST_COPY_THREADS`` sets the thread count (default: up to 32,
never more than the CPUs this process may use); ``1`` restores one thread.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, Callable

import numpy as np

ENVIRONMENT = "WOOF_HEX_HOST_COPY_THREADS"
#: Below this many elements one call is cheaper than starting threads.
MIN_PARALLEL_ELEMENTS = 1 << 22
#: Smallest column block handed to one thread.
MIN_BLOCK_COLUMNS = 8192


def threads() -> int:
    text = os.environ.get(ENVIRONMENT, "").strip()
    if text:
        try:
            value = int(text)
        except ValueError:
            value = 0
        if value < 1:
            raise ValueError(
                f"{ENVIRONMENT} must be a positive integer; prevents an ambiguous host copy schedule")
        return value
    from .cpu_budget import available_cpus

    return max(1, min(32, available_cpus()))


def column_blocks(columns: int, workers: int) -> list[slice]:
    """Contiguous ``[start, stop)`` blocks covering ``range(columns)`` in order."""

    if columns <= 0:
        return [slice(0, 0)]
    size = max(MIN_BLOCK_COLUMNS, -(-columns // (workers * 4)))
    return [slice(start, min(start + size, columns)) for start in range(0, columns, size)]


def run_blocks(function: Callable[[slice], Any], columns: int, *, elements: int,
               worker_limit: int | None = None) -> None:
    """Call ``function(block)`` for every column block, on threads when it pays."""

    workers = threads()
    if worker_limit is not None:
        if worker_limit < 1:
            raise ValueError("worker_limit must be positive")
        workers = min(workers, worker_limit)
    if workers <= 1 or elements < MIN_PARALLEL_ELEMENTS:
        function(slice(0, columns))
        return
    blocks = column_blocks(columns, workers)
    if len(blocks) == 1:
        function(blocks[0])
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(blocks)),
                            thread_name_prefix="hex-host-copy") as pool:
        for _ in pool.map(function, blocks):
            pass


def transposed_copy(value: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """``np.transpose(value, axes).copy()`` for a 2-D array, written in column blocks."""

    view = np.transpose(value, axes)
    if view.ndim != 2 or view.flags.c_contiguous:
        return view.copy()
    out = np.empty(view.shape, dtype=view.dtype, order="C")
    axis = int(np.argmax(view.shape))

    def copy(block: slice) -> None:
        index = [slice(None), slice(None)]
        index[axis] = block
        out[tuple(index)] = view[tuple(index)]

    run_blocks(copy, view.shape[axis], elements=view.size)
    return out


def contiguous_copy(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    """Preserve ``ascontiguousarray`` bytes, blocking large strided numeric copies.

    The Rust terrain bridge receives two (level, side, edge) views of the
    native (edge, side, level) arrays. At 30 km each view is 881 MB; copying
    one on four workers measured 0.057-0.076 s versus 0.373-0.469 s on one.
    Numeric float conversions also use the same NumPy assignment on bounded
    blocks, preserving subnormal and NaN payload conversion on the host.
    Small or already-contiguous same-dtype inputs retain their alias behavior.
    Split the longest axis so both level-major terrain inputs and the
    entity-major views used by vertical receipts have enough blocks to run
    in parallel. Four workers bound memory-bandwidth contention with
    background hashing; the copy-thread setting can reduce this further.
    """
    selected = np.dtype(dtype)
    if (not isinstance(value, np.ndarray) or value.ndim < 2
            or (value.flags.c_contiguous and value.dtype == selected)
            or value.size < MIN_PARALLEL_ELEMENTS
            or value.dtype.kind not in "fiu"
            or (value.dtype != selected and (value.dtype.kind != "f" or selected.kind != "f"))):
        return np.ascontiguousarray(value, dtype=dtype)
    out = np.empty(value.shape, dtype=selected, order="C")

    axis = int(np.argmax(value.shape))

    def copy(block: slice) -> None:
        index = [slice(None)] * value.ndim
        index[axis] = block
        out[tuple(index)] = value[tuple(index)]

    run_blocks(copy, value.shape[axis], elements=value.size, worker_limit=4)
    return out


def copy_into(target: np.ndarray, source: np.ndarray) -> None:
    """``np.copyto(target, source)`` for a same-shape, same-dtype pair, written in blocks on host threads.

    THE COST THIS REMOVES (RRFS hrrrv5 limited area, mv3km 206,726 cells, V6000, 2026-10-28): the ported MPAS
    framework's pool copies (``put_field`` while the physics session hydrates its pools, ``get_field`` copies in
    the TEMPO-aerosol and ozone initialisation) ran about 5 s on one core while the card waited, and they grow
    with the mesh (about 11x on NOAA's conus3km).  A same-dtype copy moves every word to one fixed place with no
    conversion, so blocks write the bytes the one call writes (NaN payloads included).
    """
    if (target.shape != source.shape or target.dtype != source.dtype or target.size < MIN_PARALLEL_ELEMENTS
            or threads() == 1):
        np.copyto(target, source)
        return
    flat = None
    if target.flags.f_contiguous and source.flags.f_contiguous:
        flat = target.reshape(-1, order="F"), source.reshape(-1, order="F")
    elif target.flags.c_contiguous and source.flags.c_contiguous:
        flat = target.reshape(-1), source.reshape(-1)
    if flat is not None and not flat[0].flags.owndata:
        out, words = flat
        run_blocks(lambda block: np.copyto(out[block], words[block]), out.size, elements=out.size)
        return
    # Mixed layouts: blocks of the target's outermost axis (contiguous target runs).
    axis = target.ndim - 1 if target.flags.f_contiguous else 0
    if target.shape[axis] < 2:
        axis = int(np.argmax(target.shape))

    def copy(block: slice) -> None:
        index = [slice(None)] * target.ndim
        index[axis] = block
        np.copyto(target[tuple(index)], source[tuple(index)])

    run_blocks(copy, target.shape[axis], elements=target.size)


def copied(value: np.ndarray, *, order: str = "K") -> np.ndarray:
    """``value.copy(order=order)``, written in blocks on host threads (``copy_into``)."""
    value = np.asarray(value)
    if value.size < MIN_PARALLEL_ELEMENTS or threads() == 1:
        return value.copy(order=order)
    out = np.empty_like(value, order=order)
    copy_into(out, value)
    return out


def all_finite(value: np.ndarray) -> bool:
    """Check every numeric word, with bounded temporary masks per worker.

    Boolean conjunction has no floating reduction or reassociation. Strided
    terrain views are sliced along their longest axis, without flattening
    them into a temporary copy. Small and nonnumeric arrays keep the original
    NumPy call.
    """
    array = np.asarray(value)
    if (array.ndim == 0 or array.size < MIN_PARALLEL_ELEMENTS
            or array.dtype.kind not in "fciu" or threads() == 1):
        return bool(np.all(np.isfinite(array)))
    view = array.reshape(-1) if array.flags.c_contiguous else array
    axis = int(np.argmax(view.shape))
    failures = []

    def check(block: slice) -> None:
        index = [slice(None)] * view.ndim
        index[axis] = block
        if not bool(np.all(np.isfinite(view[tuple(index)]))):
            failures.append(True)

    run_blocks(check, view.shape[axis], elements=view.size)
    return not failures


def active_indices_in_range(cells: np.ndarray, counts: np.ndarray, limit: int) -> bool:
    """Check only active stencil entries, without a Python loop per edge.

    Call after the existing shape/dtype/count-width checks. Inactive padding
    is deliberately not inspected. Blocks bound temporary masks to edge rows.
    """
    slots = np.arange(cells.shape[1])
    failures = []

    def check(block: slice) -> None:
        values = cells[block]
        active = slots[None, :] < counts[block, None]
        if bool(np.any(active & ((values < 0) | (values >= limit)))):
            failures.append(True)

    # Always bound temporary storage, including the one-thread fallback.
    blocks = column_blocks(len(cells), threads())
    if threads() == 1 or cells.size < MIN_PARALLEL_ELEMENTS:
        for block in blocks:
            check(block)
    else:
        with ThreadPoolExecutor(max_workers=min(threads(), len(blocks)),
                                thread_name_prefix="hex-host-check") as pool:
            for _ in pool.map(check, blocks):
                pass
    return not failures


def _caller_device():
    """The calling thread's current CUDA device when CuPy is already loaded, else None.

    CuPy keeps the current device per host thread; a pool thread starts on device 0. A task that
    launches a kernel (the moving height integrals) must run on the caller's card, or a multi-card
    executor's remap would upload its geometry to card 0.
    """
    import sys
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return None
    try:
        return int(cupy.cuda.runtime.getDevice())
    except Exception:  # noqa: BLE001 - no usable device: the tasks run without one
        return None


def run_tasks(tasks, *, workers: int | None = None) -> list:
    """Run independent zero-argument callables concurrently; results in task order.

    Exact by construction: each task is the same call it was on one thread, on its own inputs, and
    nothing is reduced across tasks here. The first failure (in task order) is raised after every
    task has finished, so no task is left running against state its caller is about to discard.
    ``WOOF_HEX_HOST_COPY_THREADS=1`` runs them in order on the caller's thread.
    """
    tasks = list(tasks)
    limit = threads() if workers is None else max(1, min(threads(), int(workers)))
    if limit <= 1 or len(tasks) <= 1:
        return [task() for task in tasks]
    device = _caller_device()

    def call(task):
        if device is None:
            return task()
        import cupy
        with cupy.cuda.Device(device):
            return task()

    with ThreadPoolExecutor(max_workers=min(limit, len(tasks)), thread_name_prefix="hex-host-task") as pool:
        futures = [pool.submit(call, task) for task in tasks]
        errors = [future.exception() for future in futures]
    for error in errors:
        if error is not None:
            raise error
    return [future.result() for future in futures]


def row_map(function: Callable[[slice], Any], rows: int, *, elements: int) -> None:
    """Call ``function(rows_block)`` over contiguous blocks of the leading axis, on threads when it pays.

    For level-major (level, cell) fields an elementwise expression on a block of whole levels reads
    and writes C-contiguous rows, so NumPy takes the same inner loop per element that the whole-array
    expression takes, and every output word is the one the single call writes.
    """
    workers = min(threads(), rows)
    if workers <= 1 or elements < MIN_PARALLEL_ELEMENTS:
        function(slice(0, rows))
        return
    bounds = np.linspace(0, rows, workers + 1, dtype=np.int64)
    blocks = [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    run_tasks([lambda block=block: function(block) for block in blocks], workers=workers)
