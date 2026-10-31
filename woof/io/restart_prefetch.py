"""Owned host checkpoint payloads, prefetched beside other preparation work.

Only archive I/O is cached. Every restore still runs the native complete-set,
physics, setup, inventory and forcing validation against its fresh model.
"""
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import copy
import threading

_ACTIVE = ContextVar('gpuwm_restart_payloads', default={})
#: True on a :func:`prefetch_beside` thread: its own archive read never waits for itself.
_READING_BESIDE = ContextVar('gpuwm_restart_reading_beside', default=False)


def _identity(path):
    # Device, inode, size and mtime name a content change: checkpoints are
    # published by os.replace (a new inode), never rewritten in place. ctime
    # is left out because the ensemble roster seal hard-links live checkpoint
    # sets (U-E10, as E9 in restart_roster._source_revision); a link changes
    # ctime, not a byte.
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass(frozen=True)
class RestartPayload:
    path: Path
    identity: tuple
    header: dict
    arrays: dict
    sha256: str


def prefetch_restart(path):
    """Read once on a host worker; immutable arrays stay owned by the set."""
    from woof.io.restart import _load_restart, RestartMismatchError
    path = Path(path).resolve()
    before = _identity(path)
    header, arrays = _load_restart(path, with_arrays=True)
    # The process-wide digest (woof.file_digest): the same SHA-256, and a digest the analysis door or
    # a checkpoint check already has (or is computing beside) is not read a second time.
    from woof.file_digest import sha256_file
    digest = sha256_file(path)
    if _identity(path) != before:
        raise RestartMismatchError(f'{path}: checkpoint changed while being prefetched')
    for value in arrays.values():
        value.flags.writeable = False
    return RestartPayload(path, before, header, arrays, digest)


#: Checkpoints being read beside a forecast leg's other preparation (:func:`prefetch_beside`), by
#: resolved path: ``(identity when started, Future of RestartPayload or None)``.
_BESIDE_LOCK = threading.Lock()
_BESIDE: dict = {}


def prefetch_beside(path) -> None:
    """Start reading the checkpoint ``path`` (:func:`prefetch_restart`) on a host thread.

    Breakage it prevents (single-core law, 2026-10-29: DA member legs
    on woof 2.9.9, ~26 s per leg at ~1 core before the engine starts): the leg's restore read its
    2.5-3.5 GB checkpoint archive in full twice on the main thread (``validate_manifest_checkpoint``
    reads every member to force the lazy archive, then ``_validate_restart`` loads every array again,
    each member with its zip CRC) and its header three more times, after the analysis door and the
    input reads and while the card waited.  Started when the leg starts (``woof.leg_server``), the one
    read runs beside the door and the input reads, and every read of that file in that state is then a
    lookup (:func:`cached_restart`).  The restore's checks are unchanged: only archive I/O is cached, as
    for the ensemble launcher.  A read that fails is forgotten, so the restore reads the file itself and
    meets the error as before; :func:`release_beside` drops the arrays once the restore applied them."""

    try:
        path = Path(path).resolve()
        started = _identity(path)
    except OSError:
        return
    with _BESIDE_LOCK:
        held = _BESIDE.get(path)
        if held is not None and held[0] == started:
            return
        future = Future()
        _BESIDE[path] = (started, future)

    def run():
        _READING_BESIDE.set(True)
        try:
            payload = prefetch_restart(path)
        except BaseException:  # noqa: BLE001 - the restore reads the file itself and meets the error
            payload = None
        future.set_result(payload)

    threading.Thread(target=run, name='restart-prefetch', daemon=True).start()


def _beside_payload(path, identity):
    """The :func:`prefetch_beside` payload of ``path`` in the state ``identity`` (waiting for it), or None."""

    with _BESIDE_LOCK:
        held = _BESIDE.get(path)
    if held is None or held[0] != identity:
        return None
    payload = held[1].result()
    return payload if payload is not None and payload.identity == identity else None


def release_beside(paths=None) -> None:
    """Forget the :func:`prefetch_beside` reads of ``paths`` (every one when None): a restore that applied
    them, or a leg that ended.  A later read of such a file reads the archive itself."""

    with _BESIDE_LOCK:
        if paths is None:
            _BESIDE.clear()
            return
        for path in paths:
            try:
                _BESIDE.pop(Path(path).resolve(), None)
            except OSError:
                pass


def keep_only_beside(paths) -> None:
    """Forget every :func:`prefetch_beside` read except those of ``paths``: a leg's start drops what an earlier
    leg kept (a failed or refused leg keeps its reads for its retry, ``woof.leg_server``)."""

    keep = set()
    for path in paths:
        try:
            keep.add(Path(path).resolve())
        except OSError:
            pass
    with _BESIDE_LOCK:
        for key in [key for key in _BESIDE if key not in keep]:
            del _BESIDE[key]


@contextmanager
def restart_payload_scope(payloads):
    token = _ACTIVE.set({payload.path: payload for payload in payloads})
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def cached_restart(path, *, with_arrays):
    from woof.io.restart import RestartMismatchError
    path = Path(path).resolve()
    payload = _ACTIVE.get().get(path)
    if payload is None:
        with _BESIDE_LOCK:
            beside = path in _BESIDE
        if not beside or _READING_BESIDE.get():
            return None
        try:
            payload = _beside_payload(path, _identity(path))
        except OSError:
            return None
        if payload is None:
            return None
    if _identity(path) != payload.identity:
        raise RestartMismatchError(f'{path}: checkpoint changed after prefetch; start a fresh ensemble set')
    # A validator may annotate metadata; no member may mutate another's input.
    return copy.deepcopy(payload.header), dict(payload.arrays) if with_arrays else None


@contextmanager
def restart_archive(path):
    """Manifest reader over prefetched arrays, or the normal lazy archive."""
    cached = cached_restart(path, with_arrays=True)
    if cached is None:
        import numpy as np
        with np.load(path, allow_pickle=False) as archive:
            yield archive
    else:
        class Archive:
            files = tuple(cached[1])
            def __getitem__(self, name):
                return cached[1][name]
        yield Archive()
