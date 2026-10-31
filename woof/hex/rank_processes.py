"""Same-host ranks as one process per card: links, an MPI-free communicator and the rank supervisor.

THE BREAKAGE THIS PREVENTS: same-host ranks ran as threads of one process, so on an ordinary CPython (with the GIL)
every rank's Python kernel-launch work shared one interpreter lock.  On 8x RTX 5090 (2026-10-28; 286,530-cell zoom
globe, convection_permitting, fast numerics, a plain ``pip install woof hex`` on Python 3.12) 2 cards ran
0.513 s/step, 4 cards 0.699 s/step and 8 cards about 7 s/step: more cards were slower, and nothing said why.  With
one process per card each rank has its own interpreter, whatever the Python build.  It needs no MPI, no mpi4py and
no second Python.

Pieces:

- :class:`Link`: one end of a socketpair carrying pickled values.  Buffers of 1 MiB and more travel out of band in
  one ``memfd`` per message (one copy into shared memory, the receiver maps it), so a gathered history frame is not
  pushed through the socket.
- :class:`ProcessComm`: the mpi4py calls the rank hooks use (``Get_rank``, ``Get_size``, ``Barrier``,
  ``allgather``, ``gather``, ``bcast``, ``scatter``, ``alltoall``, ``Abort``/``abort``), as a star through rank 0.
  Values arrive in rank order, so every reduction a hook makes from them is the thread route's.
- :class:`RankProcesses`: starts ranks 1..N-1 as children of the forecast process (rank 0 stays in it), hands each
  its links, and aborts every rank when any child dies.  Children die with their parent (``PR_SET_PDEATHSIG``).
"""

from __future__ import annotations

import copyreg
import ctypes
import io
import json
import mmap
import os
import pickle
import platform
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from types import MappingProxyType
from typing import Any

OUT_OF_BAND_BYTES = 1 << 20
_ALIGN = 64
_HEADER = struct.Struct("!QI")
#: After a rank process fails, rank 0 (the forecast process) has this long to reach an abort point of its own before
#: the supervisor ends the process with the failure named.  THE BREAKAGE THIS PREVENTS (an RTX 5090 host, x1.2562 hrrrv5, 4 rank
#: processes, 2026-10-29, MULTICARD/ckpt-shards/tip-s4-oom-hang.log): rank 2 died of an OOM and rank 0 sat 4+ min in
#: a CUDA module load until it was SIGABRTed by hand; on a paid box that holds the cards until the outer timeout.
RANK0_FAILURE_GRACE_SECONDS = 10.0
#: How long the supervisor waits for the other rank processes it killed to be reaped.
_REAP_SECONDS = 10.0


class ProcessCommAborted(RuntimeError):
    """A peer rank failed or closed its link; collectives stop instead of waiting forever."""


def memfd(name: str, size: int) -> int:
    """An anonymous shared-memory file of ``size`` bytes (``os.memfd_create``, or glibc's through ctypes).

    memfd, not ``/dev/shm``: containers often give ``/dev/shm`` 64 MB, far below one history frame.
    """
    create = getattr(os, "memfd_create", None)
    if create is not None:
        fd = create(name, getattr(os, "MFD_CLOEXEC", 1))
    else:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        libc.memfd_create.restype = ctypes.c_int
        fd = libc.memfd_create(name.encode(), 1)
        if fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, f"memfd_create({name}): {os.strerror(error)}")
    try:
        if size:
            os.ftruncate(fd, int(size))
    except BaseException:
        os.close(fd)
        raise
    return fd


def supported() -> tuple[bool, str]:
    """Whether this host can run one process per card, and if not, the reason (named in the receipt)."""
    if not sys.platform.startswith("linux"):
        return False, f"one process per card needs Linux shared memory and fd passing ({sys.platform})"
    if not hasattr(socket, "send_fds"):
        return False, "this Python has no socket.send_fds"
    try:
        release = platform.release().lower()
    except Exception:
        release = ""
    if "microsoft" in release or "wsl" in release:
        return False, "WSL2 has no pinned registration of shared host memory for CUDA"
    try:
        os.close(memfd("hex-rank-probe", 4096))
    except OSError as error:
        return False, f"memfd_create failed ({error})"
    return True, "linux"


class _SharingPickler(pickle.Pickler):
    """A pickler that writes a reference instead of any object ``shared`` names (SharedObjects.persistent_id)."""

    def __init__(self, handle, shared, **options):
        super().__init__(handle, **options)
        self._shared = shared

    def persistent_id(self, obj):
        return self._shared(obj)


class _SharingUnpickler(pickle.Unpickler):
    def __init__(self, handle, resolve, **options):
        super().__init__(handle, **options)
        self._resolve = resolve

    def persistent_load(self, pid):
        return self._resolve(pid)


def _dumps(value: Any, buffers: list, shared=None) -> bytes:
    """Protocol-5 pickle; large contiguous buffers are appended to ``buffers`` instead of copied in band.

    ``shared(obj)`` returns a reference for an object the receiver already holds (SharedObjects), or None.
    """

    def keep(buffer):
        raw = buffer.raw()
        if raw.nbytes >= OUT_OF_BAND_BYTES:
            buffers.append(raw)
            return False
        return True

    handle = io.BytesIO()
    if shared is None:
        writer = pickle.Pickler(handle, protocol=5, buffer_callback=keep)
    else:
        writer = _SharingPickler(handle, shared, protocol=5, buffer_callback=keep)
    writer.dispatch_table = dict(copyreg.dispatch_table)
    writer.dispatch_table[MappingProxyType] = _reduce_mapping
    writer.dump(value)
    return handle.getvalue()


class SharedObjects:
    """Objects every child rank receives, pickled once into one memfd that each child maps copy-on-write.

    THE COST THIS REMOVES (NOAA RRFS v2 conus3km, 8 rank processes, V8X6000, 2026-10-29): every child's request
    carried the whole global mesh its partition was cut from (the local mesh's ``partition_source_mesh``, 29.0 of
    30.5 GB per request), so rank 0 copied it into 7 memfds (213 GB; ``rank_request_seconds`` 15.5 s while all 8
    cards waited) and each child kept its own copy (103 GB max RSS per child; box peak 520-550 GB).  Now it is
    written once; a request carries a reference, and each child maps the one copy (MAP_PRIVATE: a write in one
    child stays in that child).  Every child unpickles the same bytes it unpickled before, so its objects hold the
    same words.
    """

    def __init__(self, objects: dict[str, Any], *, threads: int = 16):
        self._tags = {id(obj): tag for tag, obj in objects.items()}
        self._objects = dict(objects)   # keeps the ids above alive until close()
        buffers: list = []
        data = _dumps(self._objects, buffers)
        sizes = [int(buffer.nbytes) for buffer in buffers]
        head = _HEADER.pack(len(data), len(sizes)) + struct.pack(f"!{len(sizes)}Q", *sizes)
        start = _aligned(len(head) + len(data))
        offsets, offset = [], start
        for size in sizes:
            offsets.append(offset)
            offset += _aligned(size)
        self.nbytes = offset
        self.fd = memfd("hex-shared-objects", offset)
        try:
            os.pwrite(self.fd, head + data, 0)

            def write(item):
                buffer, at = item
                view = buffer.cast("B") if buffer.format != "B" else buffer
                done = 0
                while done < view.nbytes:
                    done += os.pwrite(self.fd, view[done:], at + done)

            from concurrent.futures import ThreadPoolExecutor
            # os.pwrite releases the GIL; the copies run side by side.
            with ThreadPoolExecutor(max_workers=max(1, min(int(threads), len(buffers) or 1)),
                                    thread_name_prefix="hex-shared-objects") as pool:
                list(pool.map(write, zip(buffers, offsets)))
        except BaseException:
            os.close(self.fd)
            raise

    def persistent_id(self, obj):
        tag = self._tags.get(id(obj))
        return None if tag is None or self._objects.get(tag) is not obj else tag

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        self._objects = {}
        self._tags = {}


class SharedObjectsView:
    """A child's side of :class:`SharedObjects`: maps the memfd once and resolves the references in its request."""

    def __init__(self, fd: int):
        self.fd = int(fd)
        self._objects = None

    def resolve(self, pid):
        if self._objects is None:
            size = os.fstat(self.fd).st_size
            shared = mmap.mmap(self.fd, size, mmap.MAP_PRIVATE, mmap.PROT_READ | mmap.PROT_WRITE)
            os.close(self.fd)
            view = memoryview(shared)
            length, count = _HEADER.unpack_from(view, 0)
            sizes = struct.unpack_from(f"!{count}Q", view, _HEADER.size)
            begin = _HEADER.size + 8 * count
            data = bytes(view[begin:begin + length])
            offset = _aligned(begin + length)
            buffers = []
            for size in sizes:
                buffers.append(view[offset:offset + size])
                offset += _aligned(size)
            self._objects = pickle.loads(data, buffers=buffers)
        return self._objects[pid]


def _mapping_proxy(values):
    return MappingProxyType(values)


def _reduce_mapping(value):
    return _mapping_proxy, (dict(value),)


def _aligned(size: int) -> int:
    return -(-int(size) // _ALIGN) * _ALIGN


class Link:
    """One end of a stream socketpair: framed pickles, out-of-band buffers in one memfd per message."""

    def __init__(self, sock: socket.socket, *, timeout: float, name: str = "link"):
        self.sock = sock
        self.name = name
        self.timeout = float(timeout)
        self.sock.settimeout(self.timeout)
        self.closed = False
        self.sent_bytes = 0
        self.received_bytes = 0

    def fileno(self) -> int:
        return self.sock.fileno()

    def send(self, value: Any, shared=None) -> None:
        buffers: list = []
        data = _dumps(value, buffers, shared)
        sizes = [int(buffer.nbytes) for buffer in buffers]
        header = _HEADER.pack(len(data), len(sizes)) + struct.pack(f"!{len(sizes)}Q", *sizes)
        try:
            if sizes:
                total = sum(_aligned(size) for size in sizes)
                fd = memfd(f"hex-{self.name}", total)
                try:
                    # os.pwrite releases the GIL for the copy (an mmap slice assignment holds it), so several
                    # links can fill their memfds at once (run_process_ranks sends every child's request at once).
                    offset = 0
                    for buffer, size in zip(buffers, sizes):
                        view = buffer.cast("B") if buffer.format != "B" else buffer
                        done = 0
                        while done < size:
                            done += os.pwrite(fd, view[done:], offset + done)
                        offset += _aligned(size)
                    socket.send_fds(self.sock, [header], [fd])
                finally:
                    os.close(fd)
            else:
                self.sock.sendall(header)
            self.sock.sendall(data)
        except (OSError, socket.timeout) as error:
            raise ProcessCommAborted(f"{self.name}: send failed ({type(error).__name__}: {error}); "
                                     "a peer rank has stopped") from error
        self.sent_bytes += len(header) + len(data) + sum(sizes)

    def _exact(self, size: int, fds: list | None = None) -> bytearray:
        out = bytearray(size)
        view = memoryview(out)
        got = 0
        while got < size:
            try:
                if fds is not None:
                    chunk, received, _flags, _address = socket.recv_fds(self.sock, size - got, 4)
                    fds.extend(received)
                    count = len(chunk)
                    view[got:got + count] = chunk
                else:
                    count = self.sock.recv_into(view[got:], size - got)
            except socket.timeout as error:
                raise ProcessCommAborted(f"{self.name}: no message from the peer rank in {self.timeout:.0f} s") from error
            except OSError as error:
                raise ProcessCommAborted(f"{self.name}: receive failed ({error}); a peer rank has stopped") from error
            if count == 0:
                raise ProcessCommAborted(f"{self.name}: the peer rank closed its link (it failed or was stopped)")
            got += count
        return out

    def recv(self, resolve=None) -> Any:
        fds: list = []
        length, count = _HEADER.unpack(self._exact(_HEADER.size, fds))
        sizes = struct.unpack(f"!{count}Q", self._exact(8 * count, fds)) if count else ()
        data = self._exact(length, fds if count and not fds else None)
        buffers = []
        if count:
            if len(fds) != 1:
                for fd in fds:
                    os.close(fd)
                raise ProcessCommAborted(f"{self.name}: a message with {count} shared buffers carried {len(fds)} fds")
            total = sum(_aligned(size) for size in sizes)
            try:
                shared = mmap.mmap(fds[0], total, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            finally:
                os.close(fds[0])
            view = memoryview(shared)
            offset = 0
            for size in sizes:
                buffers.append(view[offset:offset + size])
                offset += _aligned(size)
        else:
            for fd in fds:
                os.close(fd)
        self.received_bytes += _HEADER.size + 8 * count + length + sum(sizes)
        if resolve is not None:
            return _SharingUnpickler(io.BytesIO(data), resolve, buffers=buffers).load()
        return pickle.loads(data, buffers=buffers)

    def shutdown(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.sock.close()
            except OSError:
                pass


class ProcessComm:
    """The mpi4py collectives the rank hooks use, between rank processes, as a star through rank 0.

    Rank 0 holds a link to every other rank; every other rank holds one link to rank 0.  Each communicator is used
    by one thread of its process (the step thread, or the writer thread for the output communicator), so a frame
    gather can never be matched against a step collective (``samehost_v842.GatheredPhysicsStreams``).
    """

    def __init__(self, rank: int, size: int, links: dict[int, Link]):
        self._rank, self._size = int(rank), int(size)
        self._links = dict(links)
        self.aborted = None
        expected = set(range(1, self._size)) if self._rank == 0 else {0}
        if set(self._links) != expected:
            raise ValueError(f"rank {rank} of {size} needs links to {sorted(expected)}, got {sorted(self._links)}")

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def _check(self):
        if self.aborted is not None:
            raise ProcessCommAborted(f"rank {self._rank}: communicator aborted: {self.aborted}")

    def _gather0(self, value):
        self._check()
        if self._rank == 0:
            return [value] + [self._links[peer].recv() for peer in range(1, self._size)]
        self._links[0].send(value)
        return None

    def _bcast0(self, value):
        self._check()
        if self._rank == 0:
            for peer in range(1, self._size):
                self._links[peer].send(value)
            return value
        return self._links[0].recv()

    def allgather(self, value):
        return self._bcast0(self._gather0(value))

    def gather(self, value, root=0):
        if root == 0:
            return self._gather0(value)
        values = self.allgather(value)
        return values if self._rank == root else None

    def bcast(self, value, root=0):
        if root == 0:
            return self._bcast0(value if self._rank == 0 else None)
        return self.allgather(value if self._rank == root else None)[root]

    def scatter(self, values, root=0):
        if root == 0:
            self._check()
            if self._rank == 0:
                values = list(values)
                for peer in range(1, self._size):
                    self._links[peer].send(values[peer])
                return values[0]
            return self._links[0].recv()
        return self.allgather(values if self._rank == root else None)[root][self._rank]

    def alltoall(self, values):
        rows = self.allgather(list(values))
        return [rows[peer][self._rank] for peer in range(self._size)]

    def Barrier(self):
        self.allgather(None)

    def Abort(self, code=1):
        self.abort(f"rank {self._rank} aborted with code {code}")
        raise ProcessCommAborted(f"rank {self._rank} aborted")

    def abort(self, reason):
        if self.aborted is None:
            self.aborted = str(reason)
        for link in self._links.values():
            link.shutdown()

    def close(self):
        for link in self._links.values():
            link.close()


def set_parent_death_signal() -> None:
    """A rank process dies with the forecast process (Linux ``PR_SET_PDEATHSIG``), so no orphan holds a card."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, int(signal.SIGKILL), 0, 0, 0)   # PR_SET_PDEATHSIG
    except Exception:
        pass
    if os.getppid() == 1:   # the parent died before prctl took effect
        os._exit(1)


#: The forecast process's warm kernel store (:mod:`woof.hex.kernel_warm`), handed to every rank process it starts.
KERNEL_STORE_ENVIRONMENT = "WOOF_HEX_RANK_KERNEL_STORE"


def parent_kernel_store() -> str | None:
    """This process's warm kernel store as environment text for its rank processes, or None if it has none."""
    try:
        from . import kernel_warm
        config = kernel_warm.worker_config()
    except Exception:
        return None
    return None if config is None else json.dumps(config, sort_keys=True)


def install_parent_kernel_store() -> None:
    """Serve this rank process's NVRTC requests from the forecast process's warm kernel store.

    THE BREAKAGE THIS PREVENTS (woof hex fold/hex-0410-hb18, RTX 5090, lam7k hrrrv5 1 h fast, 2026-10-29):
    the forecast process installs the store (kernel_plan.install -> kernel_warm.install) before step 1, but a rank
    process is a fresh interpreter started with ``python -m``, and nothing installed it there.  Every direct and raw
    NVRTC unit with no disk entry of CuPy's own (RUC LSM, MYNN surface layer, TEMPO, the v8.4.1 transport, horizontal,
    acoustic and regional kernels) was therefore compiled again, inline on one core, in every rank process of every
    run, warm cache or not: py-spy on a warm 2-rank run put 18.7 s of rank 1 in ``compile_using_nvrtc`` (RUC LSM's
    boundary module alone 9.9 s) while rank 0 compiled nothing.  Rank 0 then waited for its neighbours at their first
    halo round and collective, so step 1 cost 6-8 s against a 0.13-0.37 s median step on 2 and 4 ranks (12-32 s on
    conus3km 4 and 8 cards) and the ranks' physics initialisation 9.5-12.5 s against rank 0's 1.4-2.0 s.

    The store is content-addressed (each entry's key digests its whole request), so this process reads the images
    the forecast process and earlier runs wrote and writes the ones only it compiles; its per-artifact file locks let
    a rank wait for a compile another process has started instead of repeating it.  The images are the bytes NVRTC
    returns for the same request, so no word of the forecast changes.  A store that cannot be installed is reported
    and this rank compiles as first reached, as before.
    """
    text = os.environ.get(KERNEL_STORE_ENVIRONMENT)
    if not text:
        return
    try:
        from pathlib import Path
        from . import kernel_warm
        config = json.loads(text)
        kernel_warm.install(root=Path(config["root"]), tag=config["tag"], identity=config["identity"])
    except Exception as error:
        print(f"woof hex: rank process {os.environ.get('WOOF_HEX_RANK_PROCESS', '?')}: warm kernel store not "
              f"installed ({type(error).__name__}: {error}); its kernels compile as first reached",
              file=sys.stderr, flush=True)


def _kernel_store_counts() -> dict | None:
    """This process's warm kernel store counters (hits, compiles, ...), or None if it has no store."""
    try:
        from . import kernel_warm
        store = kernel_warm._STATE
        if store is None:
            return None
        with store.lock:
            return {name: (round(value, 3) if isinstance(value, float) else value)
                    for name, value in store.counts.items()}
    except Exception:
        return None


def child_visible_devices(device: int) -> str:
    """CUDA_VISIBLE_DEVICES that shows a rank process only its own card (the parent's numbering kept)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return str(int(device))
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if not 0 <= int(device) < len(entries):
        raise ValueError(f"rank device {device} is not among the visible devices {visible!r}")
    return entries[int(device)]


class RankProcesses:
    """Start ranks 1..N-1 as child processes, give each its links, and stop everyone when one fails.

    Rank 0 is the calling process.  Links made here: a step link and an output link from rank 0 to each child, and
    one halo socket plus one 4 KiB flag file per neighbouring rank pair.  Children are killed only by their own PIDs.
    """

    def __init__(self, devices, neighbours, *, module: str, timeout: float, job):
        self.devices = tuple(int(device) for device in devices)
        self.size = len(self.devices)
        self.neighbours = {int(rank): sorted(int(peer) for peer in peers) for rank, peers in neighbours.items()}
        self.module = module
        self.timeout = float(timeout)
        self.job = str(job)
        self.processes: dict[int, subprocess.Popen] = {}
        self.exit_codes: dict[int, int | None] = {}
        self.failure = None
        self.finished = False
        self._lock = threading.Lock()
        self._on_failure = []
        self._watchers = []
        #: Seconds rank 0 has after a child's failure to reach :meth:`rank0_responded`; None never ends the process.
        self.rank0_grace: float | None = RANK0_FAILURE_GRACE_SECONDS
        self._rank0_responded = threading.Event()
        self.started_utc = None
        self.step_links: dict[int, Link] = {}
        self.output_links: dict[int, Link] = {}
        self.halo_sockets: dict[int, socket.socket] = {}   # rank 0's ends, by peer
        self.halo_flags: dict[int, int] = {}                # rank 0's flag fds, by peer
        #: A SharedObjects memfd every child inherits (set before start()), or None.
        self.shared_fd: int | None = None

    def on_failure(self, callback) -> None:
        self._on_failure.append(callback)

    def rank0_responded(self) -> None:
        """Rank 0's own thread reached an abort point or left its rank work: a child's failure need not end it."""
        self._rank0_responded.set()

    def start(self) -> None:
        child_fds: dict[int, list[int]] = {rank: [] for rank in range(1, self.size)}
        specs: dict[int, dict] = {rank: dict(halo={}) for rank in range(1, self.size)}
        closers: list = []
        for rank in range(1, self.size):
            for kind, store in (("step", self.step_links), ("output", self.output_links)):
                parent_end, child_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
                store[rank] = Link(parent_end, timeout=self.timeout, name=f"{kind}-0-{rank}")
                specs[rank][kind] = child_end.fileno()
                child_fds[rank].append(child_end.fileno())
                closers.append(child_end)
        pairs = sorted({(min(rank, peer), max(rank, peer)) for rank, peers in self.neighbours.items() for peer in peers})
        for low, high in pairs:
            low_end, high_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            flags = memfd(f"hex-halo-flags-{low}-{high}", 4096)
            for rank, end, peer in ((low, low_end, high), (high, high_end, low)):
                if rank == 0:
                    self.halo_sockets[peer] = end
                    self.halo_flags[peer] = os.dup(flags)
                else:
                    specs[rank]["halo"][str(peer)] = [end.fileno(), flags]
                    child_fds[rank].extend((end.fileno(), flags))
                    closers.append(end)
            closers.append(flags)
        if self.shared_fd is not None:
            for rank in range(1, self.size):
                specs[rank]["shared"] = int(self.shared_fd)
                child_fds[rank].append(int(self.shared_fd))
        self.started_utc = time.time()
        store = parent_kernel_store()
        try:
            for rank in range(1, self.size):
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = child_visible_devices(self.devices[rank])
                environment["WOOF_HEX_RANK_PROCESS"] = str(rank)
                if store is not None:
                    environment[KERNEL_STORE_ENVIRONMENT] = store
                command = [sys.executable, "-m", self.module, self.job, str(rank), json.dumps(specs[rank])]
                self.processes[rank] = subprocess.Popen(command, pass_fds=tuple(child_fds[rank]), env=environment,
                                                        close_fds=True)
        except BaseException:
            self.kill()
            raise
        finally:
            for item in closers:
                if isinstance(item, int):
                    os.close(item)
                else:
                    item.close()
        for rank, process in self.processes.items():
            watcher = threading.Thread(target=self._watch, args=(rank, process), name=f"hex-rank-watch-{rank}",
                                       daemon=True)
            watcher.start()
            self._watchers.append(watcher)

    def _watch(self, rank: int, process: subprocess.Popen) -> None:
        code = process.wait()
        with self._lock:
            self.exit_codes[rank] = code
            failed = code != 0 and self.failure is None
            if failed:
                self.failure = f"rank {rank} process (pid {process.pid}) exited with code {code}"
        if not failed:
            return
        # A forecast that lost a rank cannot finish, so every other rank process is stopped now (each by its own PID)
        # and reaped BEFORE the callbacks release rank 0's halo flags.  THE BREAKAGE THE ORDER PREVENTS
        # (tip-s4-oom-hang.log): with live peers, rank 0's release (flags set to their largest value) let its card
        # consume a round, which let a live peer's card write its next round number over the released flag; rank 0's
        # card then waited on a round nobody would write, and a module load (which synchronizes the context) hung
        # rank 0 until it was killed by hand.  A reaped process's card work is gone, so the release holds.
        self._stop_others(rank)
        for callback in self._on_failure:
            try:
                callback(self.failure)
            except Exception:
                pass
        # Rank 0 normally reaches an abort point within a second or two (its next collective, halo round or deferred
        # check raises).  If its thread is still held (in the CUDA driver, say), the process ends here with the
        # failure named instead of holding every card until an outer timeout.
        if self.rank0_grace is not None and not self._rank0_responded.wait(self.rank0_grace):
            self._exit_now()

    def _stop_others(self, rank: int) -> None:
        others = [process for other, process in self.processes.items() if other != rank]
        for process in others:
            if process.poll() is None:
                try:
                    process.kill()   # this child's own PID only
                except OSError:
                    pass
        deadline = time.monotonic() + _REAP_SECONDS
        for process in others:
            try:
                process.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass

    def _exit_now(self) -> None:
        message = (f"woof hex rank processes: {self.failure}; rank 0 (pid {os.getpid()}) did not stop within "
                   f"{self.rank0_grace:g} s of it, so the forecast process exits now with code 1")
        try:
            sys.stdout.flush()
            sys.stderr.write(message + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(1)

    def join(self, timeout: float) -> dict[int, int | None]:
        deadline = time.monotonic() + float(timeout)
        for rank, process in self.processes.items():
            try:
                process.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for watcher in self._watchers:
            watcher.join(max(0.0, deadline - time.monotonic()))
        self.finished = True
        return {rank: process.returncode for rank, process in self.processes.items()}

    def kill(self) -> None:
        for process in self.processes.values():
            if process.poll() is None:
                try:
                    process.kill()   # this child's own PID only
                except OSError:
                    pass

    def close(self) -> None:
        for link in (*self.step_links.values(), *self.output_links.values()):
            link.close()
        for sock in self.halo_sockets.values():
            try:
                sock.close()
            except OSError:
                pass
        for fd in self.halo_flags.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self.halo_flags.clear()


#: Named points of this process's run where its CPU use was read (:func:`cpu_mark`), in the order reached.
_CPU_MARKS: list[dict] = []
_CPU_MARK_LOCK = threading.Lock()
#: Thread groups listed per window in a rank record (the rest are summed into one "other" row).
CPU_WINDOW_THREAD_ROWS = 12


def _thread_name(comm: str, tid: int, pid: int) -> str:
    """A thread's group: its kernel name without a trailing pool index ("rank-request_3" -> "rank-request")."""
    if tid == pid:
        return "main"
    stripped = comm.rstrip("0123456789").rstrip("_-.: ")
    return stripped or comm


def _thread_counters() -> dict[int, tuple]:
    """Every live thread of this process: tid -> (group, utime s, stime s, minor faults, major faults).

    One read of /proc/self/task/<tid>/stat per thread; no device synchronization, no signal, no ptrace.
    """
    ticks = float(os.sysconf("SC_CLK_TCK"))
    pid = os.getpid()
    out: dict[int, tuple] = {}
    try:
        tids = os.listdir("/proc/self/task")
    except OSError:
        return out
    for name in tids:
        try:
            with open(f"/proc/self/task/{name}/stat", "rb") as handle:
                text = handle.read().decode("utf-8", "replace")
        except OSError:
            continue   # the thread ended between the listing and the read
        head, _, tail = text.rpartition(")")
        comm = head.partition("(")[2]
        fields = tail.split()
        try:
            # proc(5) fields 10 minflt, 12 majflt, 14 utime, 15 stime; ``fields`` starts at field 3 (state).
            out[int(name)] = (_thread_name(comm, int(name), pid), int(fields[11]) / ticks, int(fields[12]) / ticks,
                              int(fields[7]), int(fields[9]))
        except (IndexError, ValueError):
            continue
    return out


def _process_counters() -> dict:
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(wall=time.monotonic(), user=usage.ru_utime, system=usage.ru_stime, minor_faults=usage.ru_minflt,
                major_faults=usage.ru_majflt, voluntary_switches=usage.ru_nvcsw,
                involuntary_switches=usage.ru_nivcsw, block_in=usage.ru_inblock, block_out=usage.ru_oublock)


#: /proc/self/status lines a rank record carries, in MiB: the peak, the resident total and its three kinds.
_STATUS_MEMORY_KEYS = ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem")


def _own_memory_mib() -> dict | None:
    """This process's OWN resident memory (/proc/self/status, MiB): VmHWM peak, VmRSS now, and its kinds.

    THE BREAKAGE (NOAA RRFS v2 conus3km, V8XB, 2026-10-29): the rank record's max_rss_mib was getrusage's ru_maxrss,
    and Linux carries ru_maxrss across fork and exec (fs/exec.c folds the old address space's high-water mark into
    signal->maxrss).  Every rank process the forecast launches therefore reported the forecast parent's RSS at launch,
    not its own: 104.3 GiB on every rank 1-7 of an 8-card run whose children's VmHWM was 35.2-36.0 GiB, and 104.0
    GiB on 4 cards against 58.9-59.2 GiB.  VmHWM belongs to the address space, which exec replaces, so it is the
    process's own peak.  One small file read; None off Linux or when /proc cannot be read.
    """
    try:
        with open("/proc/self/status", "rb") as handle:
            text = handle.read().decode("ascii", "replace")
    except OSError:
        return None
    out = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key in _STATUS_MEMORY_KEYS:
            parts = value.split()
            try:
                out[key] = round(int(parts[0]) / 1024.0, 1)   # proc(5) reports kB
            except (IndexError, ValueError):
                continue
    return out or None


def _process_age_seconds() -> float | None:
    """Seconds since this process started (proc(5) stat field 22 against /proc/uptime), or None off Linux."""
    try:
        with open("/proc/self/stat", "rb") as handle:
            started = int(handle.read().decode().rpartition(")")[2].split()[19]) / float(os.sysconf("SC_CLK_TCK"))
        with open("/proc/uptime", "rb") as handle:
            return float(handle.read().split()[0]) - started
    except (OSError, IndexError, ValueError):
        return None


def cpu_mark(label: str) -> None:
    """Read this process's CPU use at a named point of its run (the first time ``label`` is reached).

    THE QUESTION THIS ANSWERS (NOAA RRFS v2 conus3km, V8X6000, 2026-10-29): rank processes spent more CPU time in the
    kernel than in user code (4 cards: rank 0 775.7 s system against 428 s user in a 543 s run, ranks 1-3 ~262 s
    system each; 8 cards: children 149-199 s system), and system time grew with host RSS (x1.163842: 8-9 s at 10 GB).
    The exit totals could not say whether that time is start-up (page population, file reads, pinning) or stepping
    (the thread that issues the card's work), nor which thread spent it.  A mark splits the run into windows, and
    :func:`process_cpu_record` reports each window's user and system seconds, page faults and context switches, with
    the time per thread group and the process's own resident memory at the mark.  Cost: one getrusage, one
    /proc/self/status read and one /proc stat read per live thread at each mark (a handful of marks a run, none per
    step); no device synchronization; no forecast word depends on it.
    """
    if not sys.platform.startswith("linux"):
        return
    with _CPU_MARK_LOCK:
        if any(mark["label"] == label for mark in _CPU_MARKS):
            return
    try:
        mark = dict(label=str(label), process=_process_counters(), threads=_thread_counters(),
                    memory=_own_memory_mib())
    except Exception:
        return   # a receipt instrument never fails a forecast
    with _CPU_MARK_LOCK:
        _CPU_MARKS.append(mark)


def _window(name: str, before: dict | None, after: dict, wall: float | None) -> dict:
    """One window between two marks (``before`` None: from process start)."""
    first = dict.fromkeys(after["process"], 0) if before is None else before["process"]
    last = after["process"]
    row = dict(window=name, wall_seconds=(None if wall is None else round(wall, 3)))
    for key in ("user", "system"):
        row[f"cpu_{key}_seconds"] = round(last[key] - first[key], 3)
    for key in ("minor_faults", "major_faults", "voluntary_switches", "involuntary_switches", "block_in",
                "block_out"):
        row[key] = int(last[key] - first[key])
    groups: dict[str, list] = {}
    earlier = {} if before is None else before["threads"]
    for tid, (group, user, system, minor, major) in after["threads"].items():
        prior = earlier.get(tid)
        if prior is not None and prior[0] != group:
            prior = None   # a reused tid is a new thread
        base = (0.0, 0.0, 0, 0) if prior is None else prior[1:]
        item = groups.setdefault(group, [0, 0.0, 0.0, 0, 0])
        item[0] += 1
        item[1] += user - base[0]
        item[2] += system - base[1]
        item[3] += minor - base[2]
        item[4] += major - base[3]
    ordered = sorted(groups.items(), key=lambda kv: kv[1][2] + kv[1][1], reverse=True)
    rows = []
    for group, (count, user, system, minor, major) in ordered[:CPU_WINDOW_THREAD_ROWS]:
        rows.append(dict(threads=group, count=count, cpu_user_seconds=round(user, 2),
                         cpu_system_seconds=round(system, 2), minor_faults=int(minor), major_faults=int(major)))
    rest = ordered[CPU_WINDOW_THREAD_ROWS:]
    if rest:
        rows.append(dict(threads="other", count=sum(v[0] for _, v in rest),
                         cpu_user_seconds=round(sum(v[1] for _, v in rest), 2),
                         cpu_system_seconds=round(sum(v[2] for _, v in rest), 2),
                         minor_faults=int(sum(v[3] for _, v in rest)), major_faults=int(sum(v[4] for _, v in rest))))
    seen_user = sum(v[1] for v in groups.values())
    seen_system = sum(v[2] for v in groups.values())
    # Threads that ended inside the window are not in /proc any more; their time is in the process totals only.
    row["ended_threads_cpu_user_seconds"] = round(max(0.0, row["cpu_user_seconds"] - seen_user), 2)
    row["ended_threads_cpu_system_seconds"] = round(max(0.0, row["cpu_system_seconds"] - seen_system), 2)
    # This process's own resident memory when the window closed (VmHWM = its peak so far): on rank 0, the forecast
    # parent, the "process start -> step 1 begin" row is what it still holds once the ranks are running.
    row["memory_at_end_mib"] = after.get("memory")
    row["threads"] = rows
    return row


def cpu_windows() -> dict | None:
    """This process's CPU use per window between its :func:`cpu_mark` points, up to now (see cpu_mark)."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        now = dict(label="exit", process=_process_counters(), threads=_thread_counters(), memory=_own_memory_mib())
        with _CPU_MARK_LOCK:
            marks = list(_CPU_MARKS)
        age = _process_age_seconds()
        start_wall = None if age is None else now["process"]["wall"] - age
        windows, before = [], None
        for mark in (*marks, now):
            name = f"{'process start' if before is None else before['label']} -> {mark['label']}"
            began = start_wall if before is None else before["process"]["wall"]
            wall = None if began is None else mark["process"]["wall"] - began
            windows.append(_window(name, before, mark, wall))
            before = mark
        return dict(law="getrusage(RUSAGE_SELF), /proc/self/task/*/stat and /proc/self/status at each mark; "
                        "no device sync; thread rows are threads alive at the window's end "
                        "(ended threads: the remainder)",
                    marks=[mark["label"] for mark in marks], windows=windows)
    except Exception as error:   # a receipt instrument never fails a forecast
        return dict(error=f"{type(error).__name__}: {error}")


def process_cpu_record(rank: int) -> dict:
    """What a rank process records about itself at exit: interpreter, GIL state, CPU seconds, own memory, CPU windows.

    ``max_rss_mib`` is this process's own peak (VmHWM, see :func:`_own_memory_mib`).  ``rusage_max_rss_mib`` keeps
    getrusage's figure, labelled: in a rank process it is the larger of its own peak and the forecast parent's RSS at
    launch, so it is never a size for host RAM.
    """
    import resource   # POSIX only; imported here so supported() can say "not on this platform" on Windows
    usage = resource.getrusage(resource.RUSAGE_SELF)
    gil = getattr(sys, "_is_gil_enabled", None)
    memory = _own_memory_mib()
    own_peak = None if memory is None else memory.get("VmHWM")
    return dict(rank=int(rank), pid=os.getpid(), python=sys.version.split()[0],
                gil_enabled=True if gil is None else bool(gil()),
                cpu_user_seconds=usage.ru_utime, cpu_system_seconds=usage.ru_stime,
                max_rss_mib=own_peak,
                max_rss_source=("VmHWM (/proc/self/status): this process's own peak" if own_peak is not None
                                else "unavailable (/proc/self/status not readable)"),
                rusage_max_rss_mib=usage.ru_maxrss / 1024.0,
                rusage_max_rss_note=("getrusage ru_maxrss: carried across fork and exec, so a rank process reports "
                                     "the larger of its own peak and the forecast parent's RSS at launch"),
                memory_at_exit_mib=memory,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                kernel_store=_kernel_store_counts(),
                cpu_windows=cpu_windows())
