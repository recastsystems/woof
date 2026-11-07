# SPDX-License-Identifier: Apache-2.0
"""How many same-host ranks share each card, and one copy per card of a read-only device object.

the radiation wrappers size each call's transient from a per-call budget (shortwave: a sunlit-column
cap and half the free memory; longwave: 1 GiB and an eighth of the free memory, at most 1,024 columns
under the compact policy).  Those budgets assume one caller per card.  On the same-host route N ranks
share one card and reach the radiation step together, so each took the whole budget at once and the
step's transient was N times the one-card route's.  Measured on x1.40962 hrrrv5 (8 rank threads, RTX
5090, strict): pool used 15.6 GB between radiation steps and 20-26 GB on them, pool 27-30 GB, then out
of memory on a radiation step between +9 h and +12 h (3 of 3 120 h runs).  Dividing each budget by the
ranks on the card bounds the step's sum to one caller's budget.

Batch width is memory shape only: per-column words do not depend on it (independent per-column KISS
streams, column-independent solvers), so the division changes no output word.

Both same-host executors are counted:

* rank threads of one process (``SameHostRanks``) :func:`register` their cards;
* one process per rank (``woof.hex.mpasphys.process_ranks_v842``, the default) cannot see each other,
  so the launcher tells every rank process how many ranks share its card (:func:`rank_process_card`).

A process with neither counts 1 on every card, so the one-card route and one rank per card are
unchanged.

:func:`shared_readonly` is the second half: one copy per card of a read-only device object the ranks
would otherwise each build.  THE BREAKAGE IT PREVENTS (same evidence): every rank session built TEMPO's
native-initialized table blob (934 MiB binary64; const in every kernel), so 8 ranks on one card held 8
identical copies.  Rank threads share the object itself; rank processes share its device memory
through CUDA IPC: the first rank process on the card builds and exports it, the others map it.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import socket
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

__all__ = ["register", "unregister", "ranks_on_device", "share", "shared_readonly", "rank_process_card",
           "release_exports"]

_lock = threading.Lock()
_ranks: Counter = Counter()
_shared: dict = {}
_building: dict = {}
#: Rank processes: card -> (ranks on that card across this host's rank processes, IPC slot directory or None).
_process: dict = {}
#: IPC exports this process owns: slot path -> (cupy Memory, slot path, array).
_exported: dict = {}
#: IPC mappings this process holds: slot path -> (device pointer, marker path, device, array).
_imported: dict = {}
#: Seconds an exporter waits at release for its importers, and an importer waits for the export.
EXPORT_WAIT_SECONDS = float(os.environ.get("WOOF_HEX_CARD_SHARE_WAIT_SECONDS", "1800"))


def register(devices: Iterable[int]) -> tuple[int, ...]:
    """Count one rank thread per entry of ``devices``; returns the token for :func:`unregister`."""

    held = tuple(int(device) for device in devices)
    with _lock:
        _ranks.update(held)
    return held


def unregister(held: Iterable[int]) -> None:
    """Undo one :func:`register`."""

    with _lock:
        _ranks.subtract(tuple(held))
        for device in [device for device, count in _ranks.items() if count <= 0]:
            del _ranks[device]
            for key in [key for key in _shared if key[0] == device]:
                del _shared[key]
            for key in [key for key in _building if key[0] == device]:
                del _building[key]


@contextlib.contextmanager
def rank_process_card(device: int, ranks: int, share_dir: str | os.PathLike | None = None):
    """This rank process runs on ``device`` with ``ranks`` rank processes on that card in all (itself included).

    ``share_dir`` (this host's file system, the same path for every rank process of the run) holds the
    IPC slots of :func:`shared_readonly`.  On exit the device objects this process mapped are unmapped;
    the ones it exported stay until :func:`release_exports` (or process exit), because peers may still
    read them.
    """

    device, ranks = int(device), int(ranks)
    with _lock:
        previous = _process.get(device)
        _process[device] = (max(1, ranks), None if share_dir is None else str(share_dir))
    try:
        yield
    finally:
        _close_imports(device)
        with _lock:
            if previous is None:
                _process.pop(device, None)
            else:
                _process[device] = previous


def ranks_on_device(device: int | None = None, *, cp: Any = None) -> int:
    """Ranks on ``device`` (default: the calling thread's card), threads or processes; at least 1."""

    if device is None:
        try:
            if cp is None:
                import cupy as cp
            device = int(cp.cuda.runtime.getDevice())
        except Exception:
            return 1
    device = int(device)
    with _lock:
        threads = int(_ranks.get(device, 0))
        processes = int(_process.get(device, (0, None))[0])
    return max(1, threads, processes)


def share(budget: int, device: int | None = None, *, cp: Any = None) -> int:
    """``budget`` divided among the ranks on ``device``."""

    return int(budget) // ranks_on_device(device, cp=cp)


def _device(device, cp):
    if device is not None:
        return int(device)
    if cp is None:
        import cupy as cp
    return int(cp.cuda.runtime.getDevice())


def shared_readonly(key: Any, build, device: int | None = None, *, cp: Any = None):
    """``build()`` once per card for every rank on it that asks with the same ``key``.

    ``key`` must name everything the object's bytes depend on, and the object must never be written
    after ``build`` returns (it is read on other ranks' streams; ``build`` must leave it complete,
    e.g. by synchronizing its stream).  Rank threads share the returned object.  Rank processes share
    a contiguous CuPy array's device memory through CUDA IPC.  With at most one rank on the card
    nothing is stored and ``build()`` runs every time, as before.
    """

    device = _device(device, cp)
    slot = (device, key)
    with _lock:
        threads = _ranks.get(device, 0)
        processes, share_dir = _process.get(device, (0, None))
        if threads <= 1:
            slot = None
        elif slot in _shared:
            return _shared[slot]
        else:
            gate = _building.setdefault(slot, threading.Lock())
    if slot is None:
        if processes > 1 and share_dir is not None:
            return _process_shared(key, build, device, share_dir, cp)
        return build()
    with gate:
        with _lock:
            if slot in _shared:
                return _shared[slot]
        value = build()
        with _lock:
            if _ranks.get(device, 0) > 1:
                _shared[slot] = value
        return value


# ---------------------------------------------------------------------------------------------------------
# Rank processes: one exporter per card and key, every other rank process maps its memory (CUDA IPC).
# ---------------------------------------------------------------------------------------------------------
def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _slot_path(key, device, share_dir, cp) -> Path:
    # The physical card (PCI bus id), not the ordinal: a rank process sees only its own card, as ordinal 0.
    bus = cp.cuda.runtime.deviceGetPCIBusId(int(device))
    name = hashlib.sha256(json.dumps([socket.gethostname(), bus, repr(key)]).encode()).hexdigest()[:24]
    return Path(share_dir) / name


def _process_shared(key, build, device, share_dir, cp):
    if cp is None:
        import cupy as cp
    root = Path(share_dir)
    root.mkdir(parents=True, exist_ok=True)
    slot = _slot_path(key, device, root, cp)
    with _lock:
        held = _imported.get(str(slot)) or _exported.get(str(slot))
    if held is not None:
        return held[-1]
    return _process_shared_retry(slot, build, device, cp, 0)


def _process_shared_retry(slot, build, device, cp, attempt):
    builder = slot.with_suffix(".builder")
    try:
        fd = os.open(builder, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return _import(slot, builder, build, device, cp, attempt)
    with os.fdopen(fd, "w") as handle:
        handle.write(str(os.getpid()))
    return _export(slot, build, device, cp)


def _write_atomic(path: Path, record: dict) -> None:
    temporary = path.with_suffix(f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(record))
    os.replace(temporary, path)


def _export(slot, build, device, cp):
    meta = slot.with_suffix(".json")
    value = None
    try:
        value = build()
        if not isinstance(value, cp.ndarray) or not value.flags.c_contiguous:
            raise TypeError(f"only a contiguous CuPy array is shared between rank processes, not {type(value).__name__}")
        # Its own cudaMalloc (outside the pool): an IPC handle names the base of an allocation.
        memory = cp.cuda.Memory(int(value.nbytes))
        shared = cp.ndarray(value.shape, value.dtype, cp.cuda.MemoryPointer(memory, 0))
        shared[...] = value
        cp.cuda.get_current_stream().synchronize()
        handle = cp.cuda.runtime.ipcGetMemHandle(int(memory.ptr))
        record = dict(pid=os.getpid(), handle=bytes(handle).hex(), shape=list(shared.shape),
                      dtype=shared.dtype.str, nbytes=int(shared.nbytes))
    except BaseException as error:
        _write_atomic(meta, dict(pid=os.getpid(), error=f"{type(error).__name__}: {error}"))
        if value is None:
            raise
        print(f"woof hex: card {device}: a read-only table is not shared between rank processes, each builds "
              f"its own ({type(error).__name__}: {error})", file=sys.stderr, flush=True)
        return value
    del value
    with _lock:
        _exported[str(slot)] = (memory, slot, shared)
    _write_atomic(meta, record)
    return shared


def _import(slot, builder, build, device, cp, attempt=0):
    meta = slot.with_suffix(".json")
    deadline = time.monotonic() + EXPORT_WAIT_SECONDS
    while not meta.is_file():
        try:
            pid = int(builder.read_text() or 0)
        except FileNotFoundError:
            # The exporter released its slot before this rank looked (it had no importer yet): take the slot over.
            if attempt < 3:
                return _process_shared_retry(slot, build, device, cp, attempt + 1)
            return build()
        except (OSError, ValueError):
            pid = 0
        if pid and not _alive(pid):
            raise RuntimeError(f"card {device}: the rank process exporting a shared table (pid {pid}) died")
        if time.monotonic() > deadline:
            raise TimeoutError(f"card {device}: no shared table export after {EXPORT_WAIT_SECONDS:.0f} s")
        time.sleep(0.05)
    record = json.loads(meta.read_text())
    if "error" in record:
        return build()
    marker = slot.with_suffix(f".user{os.getpid()}")
    marker.write_text(str(os.getpid()))
    try:
        pointer = int(cp.cuda.runtime.ipcOpenMemHandle(bytes.fromhex(record["handle"])))
    except BaseException:
        marker.unlink(missing_ok=True)
        raise
    import numpy as np
    memory = cp.cuda.UnownedMemory(pointer, int(record["nbytes"]), None, int(device))
    value = cp.ndarray(tuple(record["shape"]), np.dtype(record["dtype"]), cp.cuda.MemoryPointer(memory, 0))
    with _lock:
        _imported[str(slot)] = (pointer, marker, int(device), value)
    return value


def _close_imports(device=None) -> None:
    with _lock:
        items = [(path, item) for path, item in _imported.items() if device is None or item[2] == int(device)]
        for path, _ in items:
            del _imported[path]
    if not items:
        return
    import cupy as cp
    for _, (pointer, marker, card, _value) in items:
        try:
            with cp.cuda.Device(card):
                cp.cuda.runtime.deviceSynchronize()
                cp.cuda.runtime.ipcCloseMemHandle(pointer)
        except Exception:
            pass
        marker.unlink(missing_ok=True)


def release_exports(wait: float | None = None) -> None:
    """Free this process's exports once no live importer maps them (waiting up to ``wait`` seconds)."""

    with _lock:
        items = list(_exported.items())
    deadline = time.monotonic() + (EXPORT_WAIT_SECONDS if wait is None else float(wait))
    for path, (_memory, slot, _shared_array) in items:
        while time.monotonic() < deadline:
            users = []
            for user in slot.parent.glob(slot.name + ".user*"):
                try:
                    if _alive(int(user.suffix[len(".user"):])):
                        users.append(user)
                except ValueError:
                    pass
            if not users:
                break
            time.sleep(0.05)
        with _lock:
            _exported.pop(path, None)
        for suffix in (".json", ".builder"):
            slot.with_suffix(suffix).unlink(missing_ok=True)


@atexit.register
def _at_exit():  # process teardown: unmap first, then hold exports until their importers are gone
    try:
        _close_imports()
        release_exports()
    except Exception:
        pass
