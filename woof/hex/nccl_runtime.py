"""NCCL/RCCL through ctypes: the communicator the cross-host halo transport and rank collectives use.

NCCL (BSD-3-Clause, https://github.com/NVIDIA/nccl) is loaded at run time and never shipped in the hex wheel.  It is
found, in order, at ``WOOF_HEX_NCCL_LIBRARY``, in the ``nvidia.nccl`` pip package (its ``lib/libnccl.so.2``; the
licence file inside that wheel is NCCL's BSD-3 text), or as ``libnccl.so.2`` on the loader path (a source build in a
baked image).  Nothing else in hex needs it: one host without ``--samehost-transport nccl`` keeps the shared-memory
process transport.

Only the calls hex makes are bound: unique id, rank init, send/recv inside groups, allreduce/allgather, abort,
destroy and the asynchronous error query.  ctypes releases the GIL around every foreign call, so a rank thread
blocked in ``ncclCommInitRank`` or ``ncclGroupEnd`` does not stop its process's other threads.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
import threading
from typing import Any

__all__ = ["NcclError", "NcclCommunicator", "library", "library_path", "unique_id", "version", "UNIQUE_ID_BYTES"]

UNIQUE_ID_BYTES = 128

#: ncclDataType_t / ncclRedOp_t values (nccl.h; stable since NCCL 2.0).
DATATYPES = {"int8": 0, "uint8": 1, "int32": 2, "uint32": 3, "int64": 4, "uint64": 5,
             "float16": 6, "float32": 7, "float64": 8}
OPS = {"sum": 0, "prod": 1, "max": 2, "min": 3}
_SUCCESS = 0
_IN_PROGRESS = 7


class NcclError(RuntimeError):
    """An NCCL call failed; the message carries NCCL's own text."""


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * UNIQUE_ID_BYTES)]


_LIBRARY = None
_LIBRARY_PATH = None
_LIBRARY_LOCK = threading.Lock()


def _candidates() -> list[str]:
    from woof.gpu_smi import is_amd
    amd = is_amd()
    named = os.environ.get("WOOF_HEX_RCCL_LIBRARY") if amd else os.environ.get("WOOF_HEX_NCCL_LIBRARY")
    if named:
        return [named]
    found = []
    if amd:
        root = Path(os.environ.get("ROCM_PATH", os.environ.get("ROCM_HOME", "/opt/rocm")))
        for directory in (root / "lib", root / "lib64"):
            found.append(str(directory / "librccl.so.1"))
            found.append(str(directory / "librccl.so"))
        system = ctypes.util.find_library("rccl")
        if system:
            found.append(system)
        found.extend(("librccl.so.1", "librccl.so"))
        return found
    try:
        import nvidia.nccl as package   # the nvidia-nccl-cu12/cu13 wheels
        for root in getattr(package, "__path__", ()):
            path = Path(root) / "lib" / "libnccl.so.2"
            if path.is_file():
                found.append(str(path))
    except ImportError:
        pass
    system = ctypes.util.find_library("nccl")
    if system:
        found.append(system)
    found.append("libnccl.so.2")
    return found


def library():
    """The loaded libnccl, with argument types bound once per process."""
    global _LIBRARY, _LIBRARY_PATH
    with _LIBRARY_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        errors = []
        lib = None
        for candidate in _candidates():
            try:
                lib = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                _LIBRARY_PATH = candidate
                break
            except OSError as error:
                errors.append(f"{candidate}: {error}")
        if lib is None:
            raise NcclError("NCCL/RCCL is not installed: the halo transport needs librccl.so on AMD, libnccl.so.2 on NVIDIA "
                            "(WOOF_HEX_RCCL_LIBRARY / WOOF_HEX_NCCL_LIBRARY may name an installed library); "
                            + "; ".join(errors))
        p, v, i, s = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(p), i, _UniqueId, i]
        lib.ncclCommDestroy.argtypes = [p]
        lib.ncclCommAbort.argtypes = [p]
        lib.ncclCommGetAsyncError.argtypes = [p, ctypes.POINTER(ctypes.c_int)]
        lib.ncclGetErrorString.argtypes = [i]
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        if hasattr(lib, "ncclGetLastError"):
            lib.ncclGetLastError.argtypes = [p]
            lib.ncclGetLastError.restype = ctypes.c_char_p
        lib.ncclSend.argtypes = [v, s, i, i, p, p]
        lib.ncclRecv.argtypes = [v, s, i, i, p, p]
        lib.ncclAllReduce.argtypes = [v, v, s, i, i, p, p]
        lib.ncclAllGather.argtypes = [v, v, s, i, p, p]
        lib.ncclGroupStart.argtypes = []
        lib.ncclGroupEnd.argtypes = []
        for name in ("ncclGetVersion", "ncclGetUniqueId", "ncclCommInitRank", "ncclCommDestroy", "ncclCommAbort",
                     "ncclCommGetAsyncError", "ncclSend", "ncclRecv", "ncclAllReduce", "ncclAllGather",
                     "ncclGroupStart", "ncclGroupEnd"):
            getattr(lib, name).restype = ctypes.c_int
        _LIBRARY = lib
        return lib


def library_path() -> str | None:
    """Where libnccl was loaded from (receipts name it)."""
    library()
    return _LIBRARY_PATH


def _check(code: int, call: str, comm: Any = None) -> None:
    if code in (_SUCCESS, _IN_PROGRESS):
        return
    lib = library()
    text = lib.ncclGetErrorString(int(code))
    detail = ""
    if comm is not None and hasattr(lib, "ncclGetLastError"):
        last = lib.ncclGetLastError(comm)
        if last:
            detail = f" ({last.decode(errors='replace')})"
    raise NcclError(f"{call} failed: NCCL error {code}: {text.decode(errors='replace') if text else '?'}{detail}")


def version() -> str:
    value = ctypes.c_int()
    _check(library().ncclGetVersion(ctypes.byref(value)), "ncclGetVersion")
    code = int(value.value)
    # NCCL_VERSION(X,Y,Z) is X*10000+Y*100+Z from 2.9 on (X*1000+Y*100+Z before).
    major, rest = divmod(code, 10000) if code >= 10000 else divmod(code, 1000)
    minor, patch = divmod(rest, 100)
    return f"{major}.{minor}.{patch}"


def unique_id() -> bytes:
    """A fresh ncclUniqueId (rank 0 makes it; every rank receives the same 128 bytes)."""
    handle = _UniqueId()
    _check(library().ncclGetUniqueId(ctypes.byref(handle)), "ncclGetUniqueId")
    # All 128 bytes: a c_char array's value stops at its first NUL, and the id has many.
    return ctypes.string_at(ctypes.addressof(handle), UNIQUE_ID_BYTES)


class NcclCommunicator:
    """One rank of an NCCL communicator on this process's current CUDA device.

    ``ncclCommInitRank`` blocks until every rank has called it.  It runs on a helper thread so a rank whose peers
    never arrive stops after ``timeout`` seconds with a named error instead of waiting forever (a remote host that
    failed during its preparation never calls it).
    """

    def __init__(self, nranks: int, uid: bytes, rank: int, *, timeout: float = 1800.0):
        if len(uid) != UNIQUE_ID_BYTES:
            raise NcclError(f"an NCCL unique id is {UNIQUE_ID_BYTES} bytes, got {len(uid)}")
        self.nranks, self.rank = int(nranks), int(rank)
        self.lib = library()
        self.comm = ctypes.c_void_p()
        self.aborted = None
        self._destroyed = False
        handle = _UniqueId()
        ctypes.memmove(ctypes.addressof(handle), bytes(uid), UNIQUE_ID_BYTES)
        result: dict[str, Any] = {}
        # The device the caller made current; the init thread must use the same one.
        import cupy as cp
        device = int(cp.cuda.Device().id)

        def init():
            try:
                cp.cuda.Device(device).use()
                result["code"] = self.lib.ncclCommInitRank(ctypes.byref(self.comm), self.nranks, handle, self.rank)
            except BaseException as error:   # reported on the calling thread
                result["error"] = error

        worker = threading.Thread(target=init, name=f"hex-nccl-init-{self.rank}", daemon=True)
        worker.start()
        worker.join(float(timeout))
        if worker.is_alive():
            raise NcclError(f"rank {self.rank}: ncclCommInitRank did not complete in {timeout:.0f} s; a rank of the "
                            f"{self.nranks} never joined (its host failed or cannot reach this one)")
        if "error" in result:
            raise result["error"]
        _check(result["code"], f"ncclCommInitRank(rank {self.rank} of {self.nranks})")

    # -- point to point inside groups -------------------------------------------------------------------------------

    def group_start(self) -> None:
        _check(self.lib.ncclGroupStart(), "ncclGroupStart", self.comm)

    def group_end(self) -> None:
        _check(self.lib.ncclGroupEnd(), "ncclGroupEnd", self.comm)

    def send(self, pointer: int, nbytes: int, peer: int, stream: int) -> None:
        _check(self.lib.ncclSend(ctypes.c_void_p(int(pointer)), ctypes.c_size_t(int(nbytes)), DATATYPES["uint8"],
                                 int(peer), self.comm, ctypes.c_void_p(int(stream))), "ncclSend", self.comm)

    def recv(self, pointer: int, nbytes: int, peer: int, stream: int) -> None:
        _check(self.lib.ncclRecv(ctypes.c_void_p(int(pointer)), ctypes.c_size_t(int(nbytes)), DATATYPES["uint8"],
                                 int(peer), self.comm, ctypes.c_void_p(int(stream))), "ncclRecv", self.comm)

    # -- collectives (order-free reductions only: max/min; sums go through allgather + rank-order adds) -------------

    def allreduce(self, send: int, recv: int, count: int, dtype: str, op: str, stream: int) -> None:
        if op not in ("max", "min"):
            raise NcclError("ncclAllReduce with a sum is not word-stable across rank counts; gather and add in "
                            "rank order instead (prevents words that change with the card count)")
        _check(self.lib.ncclAllReduce(ctypes.c_void_p(int(send)), ctypes.c_void_p(int(recv)), ctypes.c_size_t(count),
                                      DATATYPES[dtype], OPS[op], self.comm, ctypes.c_void_p(int(stream))),
               "ncclAllReduce", self.comm)

    def allgather(self, send: int, recv: int, count: int, dtype: str, stream: int) -> None:
        _check(self.lib.ncclAllGather(ctypes.c_void_p(int(send)), ctypes.c_void_p(int(recv)), ctypes.c_size_t(count),
                                      DATATYPES[dtype], self.comm, ctypes.c_void_p(int(stream))),
               "ncclAllGather", self.comm)

    # -- health and teardown -----------------------------------------------------------------------------------------

    def async_error(self) -> int:
        if self._destroyed:
            return 0
        code = ctypes.c_int()
        self.lib.ncclCommGetAsyncError(self.comm, ctypes.byref(code))
        return int(code.value)

    def error_text(self, code: int) -> str:
        text = self.lib.ncclGetErrorString(int(code))
        return text.decode(errors="replace") if text else str(code)

    def abort(self, reason: Any = None) -> None:
        """Stop every NCCL kernel of this communicator (queued sends and receives return) and free it."""
        if self._destroyed:
            return
        self.aborted = str(reason) if reason is not None else "aborted"
        self._destroyed = True
        self.lib.ncclCommAbort(self.comm)

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self.lib.ncclCommDestroy(self.comm)
