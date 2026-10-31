"""Halo rounds between rank processes (one process per card), with no host numerics and no host wait on a GPU.

The same rounds, packets and byte kernels as :class:`~woof.hex.partition_samehost_transport_v841.SameHostTransport`
(this class reuses its schema check, packet layout, descriptor cache and pack/unpack launches), so a halo carries
the same bytes in the same round order and the forecast's words do not depend on the executor.

What differs is how a packet crosses from one process to the other.  For each neighbour and field schema the sender
owns one host buffer in shared memory (a ``memfd``, registered for DMA in both processes):

1. the sender packs on its compute stream; its staging stream waits for that, copies the packet into the shared
   buffer and then writes the round number into a flag word in shared host memory (``cuStreamWriteValue32``);
2. the receiver's copy stream waits on the GPU until that flag reaches the round (``cuStreamWaitValue32``), copies
   the buffer to its device packet, writes a "consumed" flag the sender's next copy into the same buffer waits for,
   and unpacks; its compute stream then waits for the unpack.

No host thread waits for a GPU, so each rank's host keeps running ahead of its card as on the thread route.  The
hosts meet over their sockets only in each field schema's first round with a neighbour: that header carries the round,
the digest of tag and field schema, the packet size and the buffer's fd, and the receiver checks it.  Every later
round of that schema crosses on the GPU flags alone.  The round check stays, in shared memory: each rank folds every
round's digest into a running chain and writes (round, chain) into each neighbour's flag page; whichever end of a
link reaches a round second compares the other's chain with its own for that round, so a schedule or schema
disagreement still stops the run instead of mixing halos, and no host waits for a neighbour's host to get there.
At close each rank writes its final round, compares it with each neighbour's, and releases the flags its neighbours'
cards wait on, so a rank that ran more rounds than a neighbour stops with that named instead of hanging on a flag.

Breakage the per-round header caused: conus3km on 8 x RTX PRO 6000, one process per card, rank 0's host waited
48.9 s per forecast hour (about 272 ms of each 861 ms step) for neighbours' headers it only needed for the round
check (V8X6000 lease mcs2-tmr-8c, 2026-10-29, MULTICARD/SCALING-2.md section 4).

The probe on 8x RTX 5090 with no peer access (2026-10-28): 1,000 rounds of 6 MiB each way between cards 0 and 1,
0 wrong words, 0.71 ms per round.
"""

from __future__ import annotations

from collections import deque
import ctypes
import hashlib
import mmap
import os
import socket
import struct
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .partition_samehost_transport_v841 import (SameHostTransport, SameHostTransportError,
                                                exchange_timing_requested)

__all__ = ["ProcessHaloTransport"]

_HEADER = struct.Struct("!Q16sQB")   # round, digest of tag and schema, packet bytes, carries an fd
_FLAG_SPACING = 128
_WAIT_GEQ = 0x0
# The round-check record each end of a link writes into the pair's 4 KiB flag page (the four flag words use bytes
# 0-511): round, chain digest of every round up to it, final (the writer has closed), and a check word over the
# three, so a record read while it is being rewritten is skipped instead of compared.
_ROUND_RECORD = struct.Struct("=Q16sQQ")
_ROUND_RECORD_OFFSET = (1024, 2048)   # written by the lower rank of the pair, by the higher rank
_FLAG_PAGE = 4096
_RELEASED = 0xFFFFFFFF


class RoundCheck:
    """The per-round schedule check between rank processes, through shared memory instead of a host rendezvous.

    ``pages`` maps each neighbour to ``(buffer, out_offset, in_offset)``: a writable buffer shared with that neighbour
    (the pair's flag page), where this rank writes its record and reads the neighbour's.  Every round's digest (tag
    and field schema) is folded into a chain, so two ranks that agree on round ``n`` agree on every round before it;
    the end of a link that reaches a round second compares, and the history of chains kept here reaches back only
    to the slowest neighbour's last compared round.
    """

    def __init__(self, rank: int, pages: Mapping[int, tuple[Any, int, int]]):
        self.rank = int(rank)
        self.pages = {int(peer): (buffer, int(out_offset), int(in_offset))
                      for peer, (buffer, out_offset, in_offset) in pages.items()}
        self.round = 0
        self.chain = bytes(16)
        self.history: dict[int, bytes] = {0: self.chain}
        self._oldest = 0
        self.compared = {peer: 0 for peer in self.pages}
        self.comparisons = 0

    @staticmethod
    def _check_word(round_: int, chain: bytes, final: int) -> int:
        return int.from_bytes(hashlib.blake2b(struct.pack("=Q", round_) + chain + struct.pack("=Q", final),
                                              digest_size=8).digest(), "little")

    def _write(self, final: int) -> None:
        record = _ROUND_RECORD.pack(self.round, self.chain, final, self._check_word(self.round, self.chain, final))
        for buffer, out_offset, _ in self.pages.values():
            buffer[out_offset:out_offset + _ROUND_RECORD.size] = record

    def read(self, peer: int):
        """The neighbour's (round, chain, final), or None when it has written nothing yet or is mid-write."""
        buffer, _, in_offset = self.pages[peer]
        round_, chain, final, check = _ROUND_RECORD.unpack_from(buffer, in_offset)
        if check != self._check_word(round_, chain, final):
            return None
        return round_, chain, bool(final)

    def advance(self, round_: int, digest: bytes, tag: Any = None) -> None:
        """Fold this rank's round into the chain, publish it, and compare every neighbour that is not ahead."""
        if int(round_) != self.round + 1:
            raise SameHostTransportError(f"rank {self.rank}: round {round_} follows round {self.round}")
        self.round = int(round_)
        self.chain = hashlib.blake2b(self.chain + struct.pack("=Q", self.round) + bytes(digest),
                                     digest_size=16).digest()
        self.history[self.round] = self.chain
        self._write(0)
        for peer in self.pages:
            self._compare(peer, tag)
        self._trim()

    def _trim(self) -> None:
        floor = min(self.compared.values(), default=self.round)
        while self._oldest < floor:
            self.history.pop(self._oldest, None)
            self._oldest += 1

    def _compare(self, peer: int, tag: Any = None, *, closing: bool = False) -> None:
        record = self.read(peer)
        if record is None:
            return
        theirs, chain, final = record
        if final and theirs < self.round:
            raise SameHostTransportError(
                f"rank {peer} closed after {theirs} halo rounds, but rank {self.rank} is at round {self.round}: "
                "the ranks disagree on the halo schedule")
        if closing and theirs > self.round:
            raise SameHostTransportError(
                f"rank {peer} is at halo round {theirs}, but rank {self.rank} closed after {self.round}: "
                "the ranks disagree on the halo schedule")
        if self.compared[peer] < theirs <= self.round:
            mine = self.history.get(theirs)
            if mine is not None:
                self.comparisons += 1
                if mine != chain:
                    raise SameHostTransportError(
                        f"rank {self.rank} and rank {peer} disagree on the halo schedule at or before round {theirs} "
                        f"(tag or field schema; this rank's latest tag {tag!r}); stopped instead of mixing halos")
        # A neighbour ahead of this rank never writes an older round again: nothing at or below this rank's round
        # is left for this end to compare (that neighbour compares them against this rank's record).
        self.compared[peer] = max(self.compared[peer], min(theirs, self.round))

    def finish(self) -> None:
        """Write this rank's final round, so a neighbour still running can tell it ran more rounds than this one."""
        self._write(1)

    def close(self) -> None:
        """Compare this rank's final round with every neighbour that has got that far or closed."""
        self._write(1)
        for peer in self.pages:
            self._compare(peer, closing=True)
        self._trim()


class _Driver:
    """The two CUDA driver stream memory operations this transport needs (libcuda, through ctypes)."""

    def __init__(self):
        cuda = ctypes.CDLL("libcuda.so.1")

        def symbol(*names):
            for name in names:
                function = getattr(cuda, name, None)
                if function is not None:
                    return function
            raise OSError(f"libcuda has none of {names}")

        self.wait = symbol("cuStreamWaitValue32_v2", "cuStreamWaitValue32")
        self.wait.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        self.write = symbol("cuStreamWriteValue32_v2", "cuStreamWriteValue32")
        self.write.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        self.device_pointer = symbol("cuMemHostGetDevicePointer_v2", "cuMemHostGetDevicePointer")
        self.device_pointer.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]

    def wait_geq(self, stream, address: int, value: int) -> None:
        code = self.wait(ctypes.c_void_p(int(stream.ptr)), ctypes.c_uint64(address), ctypes.c_uint32(value), _WAIT_GEQ)
        if code:
            raise SameHostTransportError(f"cuStreamWaitValue32 failed with CUDA driver error {code}")

    def write_value(self, stream, address: int, value: int) -> None:
        code = self.write(ctypes.c_void_p(int(stream.ptr)), ctypes.c_uint64(address), ctypes.c_uint32(value), 0)
        if code:
            raise SameHostTransportError(f"cuStreamWriteValue32 failed with CUDA driver error {code}")

    def mapped(self, address: int) -> int:
        pointer = ctypes.c_uint64()
        code = self.device_pointer(ctypes.byref(pointer), ctypes.c_void_p(address), 0)
        if code:
            raise SameHostTransportError(f"cuMemHostGetDevicePointer failed with CUDA driver error {code}")
        return int(pointer.value)


class _Shared:
    """A shared-memory file mapped and registered for DMA in this process."""

    def __init__(self, xp, fd: int, size: int, *, mapped: bool = False, owner_fd: bool = True):
        self.xp, self.size = xp, int(size)
        self.fd = fd if owner_fd else None
        self.map = mmap.mmap(fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.map))
        # Portable (1), plus mapped (2) for the flag words the GPU reads and writes.
        xp.cuda.runtime.hostRegister(self.address, self.size, 1 | (2 if mapped else 0))
        self.registered = True
        self.generation = 0

    def close_fd(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def release(self) -> None:
        if self.registered:
            try:
                self.xp.cuda.runtime.hostUnregister(self.address)
            except Exception:
                pass
            self.registered = False
        self.close_fd()


class ProcessHaloTransport(SameHostTransport):
    """This rank's end of the halo rounds; the other ranks run their own ends in their own processes.

    ``links`` maps each neighbouring rank to ``(socket, flag_fd)``: the socket carries round headers and buffer fds,
    the 4 KiB flag file holds the four flag words of the pair (ready and consumed, each direction).
    """

    def __init__(self, tables: Sequence[Any], devices: Sequence[int], rank: int,
                 links: Mapping[int, tuple[socket.socket, int]], *, xp: Any, timeout: float = 1800.0,
                 field_halo_indices: Mapping[tuple[int, int, str, str], Any] | None = None,
                 measure_gpu: bool | None = None) -> None:
        # Not SameHostTransport.__init__: that enables peer routes and builds streams on every card from one
        # process.  This process touches only its own card.
        self.xp = xp
        self._cpu = xp is np
        if self._cpu:
            raise SameHostTransportError("the process transport moves CUDA packets; the CPU reference is SameHostTransport")
        # GPU-event halo timing (WOOF_HEX_EXCHANGE_TIMING=1 unless given): this rank's receive (neighbour ready
        # flag, host-to-device copy, unpack) and its exposed stall (compute stream waiting for the unpack after the
        # interior work). Events are read only once complete, in _retire: no host sync is added to the step.
        self.measure_gpu = exchange_timing_requested() if measure_gpu is None else bool(measure_gpu)
        self.gpu_step_timings = {}
        self.tables = tuple(tables)
        self.devices = tuple(map(int, devices))
        self.rank = int(rank)
        self.device = self.devices[self.rank]
        self.timeout = float(timeout)
        self.descriptor_cache_size = int(os.environ.get("WOOF_HEX_HALO_DESCRIPTOR_CACHE", "4096"))
        self.stage_same_device = False
        self.field_halo_indices = dict(field_halo_indices or {})
        self.release_fields_after_fence = False
        self.release_fields_on_retire = True
        if not self.tables or len(self.tables) != len(self.devices) or not 0 <= self.rank < len(self.tables):
            raise SameHostTransportError("one table and one device are required for every rank")
        self.n_ranks = len(self.tables)
        self._peers = []
        self._validate_tables()
        self._validate_field_indices()
        neighbours = sorted(self._peers[self.rank])
        if sorted(int(peer) for peer in links) != neighbours:
            raise SameHostTransportError(f"rank {self.rank} has links to {sorted(links)}, but its halo peers are {neighbours}")
        self._stats_lock = threading.Lock()
        self._condition = threading.Condition()
        self._cache = [dict() for _ in self.tables]
        self.paths = {(src, dst): "process-shared-host" for src in set(self.devices) for dst in set(self.devices)}
        self.copy_policy = "process"
        self._error = None
        self._closed = False
        self._generation = 0
        self._retained: deque = deque()
        self._send_host: dict[tuple[int, Any], _Shared] = {}
        self._recv_host: dict[tuple[int, Any], _Shared] = {}
        self._driver = _Driver()
        self._sockets = {}
        self._flags = {}
        with xp.cuda.Device(self.device):
            self._kernel = xp.RawKernel(_byte_copy_source(), "halo_copy_bytes")
            self._word_kernel = xp.RawKernel(_word_copy_source(), "halo_copy_words_2d")
            self._kernel.compile()
            self._word_kernel.compile()
            self._copy_stream = xp.cuda.Stream(non_blocking=True)
            self._stage_streams = {peer: xp.cuda.Stream(non_blocking=True) for peer in neighbours}
            for peer in neighbours:
                sock, flag_fd = links[peer]
                sock.settimeout(self.timeout)
                self._sockets[peer] = sock
                flags = _Shared(xp, flag_fd, _FLAG_PAGE, mapped=True)
                flags.close_fd()
                base = self._driver.mapped(flags.address)
                low = self.rank < peer
                # Slots: 0 ready low->high, 1 consumed low->high, 2 ready high->low, 3 consumed high->low.
                slots = dict(ready_out=0 if low else 2, consumed_out=1 if low else 3,
                             ready_in=2 if low else 0, consumed_in=3 if low else 1)
                self._flags[peer] = dict(share=flags, slots=slots,
                                         **{name: base + _FLAG_SPACING * slot for name, slot in slots.items()})
        # Headers cross a link only in a field schema's first round (and whenever a buffer's fd must); the round
        # check is the shared-memory chain in each pair's flag page.
        self._announced: set = set()
        self._heard: set = set()
        self._round_check = RoundCheck(self.rank, {
            peer: (flags["share"].map, _ROUND_RECORD_OFFSET[0 if self.rank < peer else 1],
                   _ROUND_RECORD_OFFSET[1 if self.rank < peer else 0])
            for peer, flags in self._flags.items()})
        self.stats: dict[str, int | float] = {
            "rounds": 0, "bytes": 0, "local_bytes": 0, "peer_bytes": 0, "staged_bytes": 0, "seconds": 0.0,
            "completed_rounds": 0, "pack_launches": 0, "unpack_launches": 0, "descriptor_uploads": 0,
            "dimension_uploads": 0, "descriptor_cache_hits": 0, "descriptor_cache_misses": 0,
            "descriptor_cache_evictions": 0, "serial_unpack_batches": 0, "serial_unpack_launches": 0,
            "header_wait_seconds": 0.0, "headers_sent": 0, "headers_received": 0, "round_checks": 0,
            "shared_buffers": 0, "shared_buffer_bytes": 0,
        }

    def _header_due(self, seen: set, peer: int, schema: Any, needs_fd: bool) -> bool:
        """A header crosses this link in the schema's first round with this neighbour, and whenever an fd must.

        Both ends decide from the same facts (the round's schema and packet size, which the first header checks),
        so a sender sends exactly the headers its receiver reads.
        """
        key = (peer, schema)
        if key in seen and not needs_fd:
            return False
        seen.add(key)
        return True

    # -- headers ---------------------------------------------------------------------------------------------------

    def _send_header(self, peer: int, generation: int, digest: bytes, size: int, fd: int | None) -> None:
        message = _HEADER.pack(generation, digest, size, 0 if fd is None else 1)
        try:
            if fd is None:
                self._sockets[peer].sendall(message)
            else:
                socket.send_fds(self._sockets[peer], [message], [fd])
        except (OSError, socket.timeout) as error:
            raise SameHostTransportError(f"rank {self.rank}: halo header to rank {peer} failed ({error}); "
                                         "that rank has stopped") from error

    def _recv_header(self, peer: int):
        sock = self._sockets[peer]
        out = bytearray(_HEADER.size)
        view = memoryview(out)
        fds: list[int] = []
        got = 0
        while got < _HEADER.size:
            try:
                chunk, received, _flags, _address = socket.recv_fds(sock, _HEADER.size - got, 2)
            except socket.timeout as error:
                raise SameHostTransportError(f"rank {self.rank}: no halo header from rank {peer} in "
                                             f"{self.timeout:.0f} s") from error
            except OSError as error:
                raise SameHostTransportError(f"rank {self.rank}: halo header from rank {peer} failed ({error})") from error
            fds.extend(received)
            if not chunk:
                raise SameHostTransportError(f"rank {self.rank}: rank {peer} closed its halo link (it failed or stopped)")
            view[got:got + len(chunk)] = chunk
            got += len(chunk)
        generation, digest, size, has_fd = _HEADER.unpack(out)
        if bool(has_fd) != bool(fds) or len(fds) > 1:
            for fd in fds:
                os.close(fd)
            raise SameHostTransportError(f"rank {self.rank}: malformed halo header from rank {peer}")
        return generation, digest, size, (fds[0] if fds else None)

    # -- the round -------------------------------------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise SameHostTransportError("process halo transport is closed")
        if self._error is not None:
            raise self._error

    def _retire(self) -> None:
        while self._retained and self._retained[0][0].done and (
                self._retained[0][3] is None or self._retained[0][3][4].done):
            timing = self._retained.popleft()[3]
            if timing is not None:
                self._record_timing(timing)
        self.stats["completed_rounds"] = self.stats["rounds"] - len(self._retained)

    def _record_timing(self, timing) -> None:
        step, receive_start, unpacked, wait_start, wait_end = timing
        elapsed = self.xp.cuda.get_elapsed_time
        wait = float(elapsed(wait_start, wait_end)) / 1000.
        receive = float(elapsed(receive_start, unpacked)) / 1000.
        with self._stats_lock:
            row = self.gpu_step_timings.setdefault(step, {"rounds": 0, "wait_seconds": 0., "receive_seconds": 0.})
            row["rounds"] += 1
            row["wait_seconds"] += wait
            row["receive_seconds"] += receive

    def exchange(self, rank: int, tag: Any, fields: Sequence[tuple[str, Any, str]], *,
                 interior_work=None) -> dict[str, int | float]:
        """Queue this rank's half of one round and the consumer fence on its compute stream."""
        started = time.perf_counter()
        try:
            if int(rank) != self.rank:
                raise SameHostTransportError(f"rank {rank} used rank {self.rank}'s process transport")
            self._check_open()
            fields = tuple(fields)
            schema = self._schema(self.rank, fields)
            self._generation += 1
            generation = self._generation
            digest = hashlib.blake2b(repr((tag, schema)).encode(), digest_size=16).digest()
            # The round check, before this round queues anything: publish this round's chain to every neighbour and
            # compare every neighbour that has got here first.  No wait for a neighbour that has not.
            self._round_check.advance(generation, digest, tag)
            xp = self.xp
            sent = received_bytes = 0
            header_wait = 0.0
            headers_sent = headers_received = 0
            with xp.cuda.Device(self.device):
                stream = xp.cuda.get_current_stream()
                buffers = self._buffers(self.rank, schema, fields)
                # The next pack into these send packets waits for the copies out of them in the previous round.
                for event in getattr(buffers, "process_copied", ()):
                    stream.wait_event(event)
                self._copy_packets(fields, buffers.send.values(), buffers.send_batch, scatter=False, stream=stream)
                produced = xp.cuda.Event(disable_timing=True)
                produced.record(stream)
                owners = [buffers.send_batch.descriptor, *buffers.send_batch.owners]
                copied = []
                for peer, packet in buffers.send.items():
                    size = int(packet.buffer.nbytes)
                    fd = None
                    if size:
                        key = (peer, schema)
                        host = self._send_host.get(key)
                        if host is None:
                            host = _Shared(xp, _memfd(f"hex-halo-{self.rank}-{peer}", size), size)
                            self._send_host[key] = host
                            self.stats["shared_buffers"] += 1
                            self.stats["shared_buffer_bytes"] += size
                            fd = host.fd
                        flags = self._flags[peer]
                        staging = self._stage_streams[peer]
                        staging.wait_event(produced)
                        if host.generation:
                            # The receiver has copied the previous round out of this buffer.
                            self._driver.wait_geq(staging, flags["consumed_out"], host.generation)
                        xp.cuda.runtime.memcpyAsync(host.address, packet.buffer.data.ptr, size,
                                                    xp.cuda.runtime.memcpyDeviceToHost, staging.ptr)
                        self._driver.write_value(staging, flags["ready_out"], generation)
                        host.generation = generation
                        done = xp.cuda.Event(disable_timing=True)
                        done.record(staging)
                        copied.append(done)
                        sent += size
                    if self._header_due(self._announced, peer, schema, fd is not None):
                        self._send_header(peer, generation, digest, size, fd)
                        headers_sent += 1
                    if fd is not None:
                        self._send_host[(peer, schema)].close_fd()
                buffers.process_copied = tuple(copied)
                copy = self._copy_stream
                with copy:
                    copy.wait_event(produced)   # the unpack writes storage this rank's pack read
                    receive_start = None
                    if self.measure_gpu:
                        receive_start = xp.cuda.Event()
                        receive_start.record(copy)
                    for peer, packet in buffers.recv.items():
                        size = int(packet.buffer.nbytes)
                        key = (peer, schema)
                        if self._header_due(self._heard, peer, schema, bool(size) and key not in self._recv_host):
                            waited = time.perf_counter()
                            their_generation, their_digest, their_size, fd = self._recv_header(peer)
                            header_wait += time.perf_counter() - waited
                            headers_received += 1
                            if fd is not None:
                                if key in self._recv_host:
                                    os.close(fd)
                                    raise SameHostTransportError(f"rank {peer} sent a second buffer for one schema")
                                self._recv_host[key] = _Shared(xp, fd, their_size)
                                self._recv_host[key].close_fd()
                            if (their_generation, their_digest, their_size) != (generation, digest, size):
                                raise SameHostTransportError(
                                    f"round {generation} rank {self.rank} and rank {peer} disagree on the round, tag "
                                    f"or field schema (tag {tag!r}; {their_size} bytes against {size})")
                        if size:
                            host = self._recv_host.get(key)
                            if host is None:
                                raise SameHostTransportError(f"rank {peer} never shared its buffer for this schema")
                            flags = self._flags[peer]
                            self._driver.wait_geq(copy, flags["ready_in"], generation)
                            xp.cuda.runtime.memcpyAsync(packet.buffer.data.ptr, host.address, size,
                                                        xp.cuda.runtime.memcpyHostToDevice, copy.ptr)
                            self._driver.write_value(copy, flags["consumed_in"], generation)
                            received_bytes += size
                    self._copy_packets(fields, buffers.recv.values(), buffers.recv_batch, scatter=True, stream=copy)
                    owners.extend((buffers.recv_batch.descriptor, *buffers.recv_batch.owners))
                    unpacked = xp.cuda.Event(disable_timing=not self.measure_gpu)
                    unpacked.record(copy)
                if interior_work is not None:
                    # Pack is ordered before the caller's work; the unpack runs on the copy stream.
                    interior_work()
                timing = None
                if self.measure_gpu:
                    wait_start = xp.cuda.Event()
                    wait_start.record(stream)
                stream.wait_event(unpacked)
                if self.measure_gpu:
                    wait_end = xp.cuda.Event()
                    wait_end.record(stream)
                    timing = (str(tag).split(":", 1)[0], receive_start, unpacked, wait_start, wait_end)
                # Fields and descriptors stay referenced until the unpack has run on the card.
                self._retained.append((unpacked, owners, fields, timing))
                with self._stats_lock:
                    self.stats["rounds"] += 1
                    self.stats["bytes"] += sent
                    self.stats["staged_bytes"] += sent
                    self.stats["header_wait_seconds"] += header_wait
                    self.stats["headers_sent"] += headers_sent
                    self.stats["headers_received"] += headers_received
                    self.stats["round_checks"] = self._round_check.comparisons
                    self.stats["seconds"] += time.perf_counter() - started
                self._retire()
            return {"bytes_sent": sent, "bytes_received": received_bytes, "seconds": time.perf_counter() - started}
        except BaseException as exc:
            self.abort(f"rank {self.rank} process halo exchange failed: {exc}")
            if isinstance(exc, SameHostTransportError):
                raise
            raise self._error from exc

    # -- lifecycle -------------------------------------------------------------------------------------------------

    def abort(self, reason: Any) -> None:
        """Stop this rank's rounds and close its halo links, so neighbours waiting on a header stop too.

        The flag words this rank's card waits on are set to their largest value, so no copy queued on this card
        waits for a peer that has stopped: a later drain or context teardown cannot hang on them.
        """
        if self._error is None:
            self._error = SameHostTransportError(str(reason))
        for flags in self._flags.values():
            share = flags["share"]
            for name in ("ready_in", "consumed_out"):
                try:
                    struct.pack_into("=I", share.map, _FLAG_SPACING * flags["slots"][name], 0xFFFFFFFF)
                except (ValueError, TypeError, OSError):
                    pass
        for sock in self._sockets.values():
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _release_neighbours(self) -> None:
        for flags in self._flags.values():
            share = flags["share"]
            for name in ("ready_out", "consumed_in"):
                struct.pack_into("=I", share.map, _FLAG_SPACING * flags["slots"][name], _RELEASED)

    def synchronize(self, *, timeout: float | None = None) -> None:
        with self.xp.cuda.Device(self.device):
            self._copy_stream.synchronize()
            for staging in self._stage_streams.values():
                staging.synchronize()
            if self.measure_gpu and self._error is None:
                # Drain only: the last rounds' compute-stream wait events, so their timings are kept.
                for entry in self._retained:
                    if entry[3] is not None:
                        entry[3][4].synchronize()
                        self._record_timing(entry[3])
        self._retained.clear()

    def close(self, *, timeout: float | None = None) -> None:
        if self._closed:
            return
        try:
            if self._error is None:
                self.synchronize(timeout=timeout)
                # This rank's rounds are all on the shared buffers and its flags are final.  Publish the final
                # round, then release the flags the neighbours' cards wait on (every round this rank served is
                # already below them), so a neighbour that queued more rounds than this rank reaches its own close
                # and stops there with the disagreement named, instead of hanging its card on a flag.
                self._round_check.finish()
                self._release_neighbours()
                self._round_check.close()
                self.stats["round_checks"] = self._round_check.comparisons
        finally:
            self._closed = True
            for share in (*self._send_host.values(), *self._recv_host.values(),
                          *(flags["share"] for flags in self._flags.values())):
                share.release()
            for sock in self._sockets.values():
                try:
                    sock.close()
                except OSError:
                    pass

    def transport_report(self) -> list[dict[str, Any]]:
        return [{"source_device": self.device, "destination_rank": peer, "actual": "shared host memory between processes",
                 "host_legs": 2} for peer in sorted(self._sockets)]

    def receipt(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "transport_version": "partition-process-transport-v841/1",
                "rank": self.rank, "ranks": self.n_ranks, "device": self.device,
                "neighbours": sorted(self._sockets),
                "timeout_seconds": self.timeout,
                "descriptor_cache_size": self.descriptor_cache_size,
                "pack_unpack": "all-peer byte kernels (SameHostTransport's); overlapping field storage retains serial peer scatter",
                "routes": self.transport_report(), "stats": dict(self.stats),
                "gpu_step_timings": {step: dict(row) for step, row in self.gpu_step_timings.items()},
                "gpu_timing": _timing_summary(self.gpu_step_timings) if self.measure_gpu else None,
                "gpu_timing_law": "this rank only: receive = neighbour ready-flag wait + host-to-device copy + "
                                  "unpack on the copy stream; wait = compute stream stalled on the unpack after the "
                                  "interior work (exposed halo stall); seconds summed per step label",
                "aborted": self._error is not None,
                "completion": "GPU-side flags in shared host memory (cuStreamWriteValue32/cuStreamWaitValue32); "
                              "no host thread waits for a card",
                "headers": "socket header per link only in a field schema's first round (round, digest, size, fd)",
                "round_check": "every round: running chain digest of tag and field schema in each pair's flag page; "
                               "the end that reaches a round second compares; final rounds compared at close",
                "seconds_law": "host enqueue time including first-round header waits; GPU completion is ordered on the "
                               "card",
            }


def _timing_summary(rows: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    waits = sorted(float(row["wait_seconds"]) for row in rows.values())
    receives = sorted(float(row["receive_seconds"]) for row in rows.values())
    middle = len(waits) // 2
    return {
        "steps": len(waits), "rounds": sum(int(row["rounds"]) for row in rows.values()),
        "exposed_wait_seconds": sum(waits), "receive_seconds": sum(receives),
        "median_step_exposed_wait_seconds": waits[middle] if waits else None,
        "median_step_receive_seconds": receives[middle] if receives else None,
    }


def _memfd(name: str, size: int) -> int:
    from .rank_processes import memfd
    return memfd(name, size)


def _byte_copy_source() -> str:
    from . import partition_samehost_transport_v841 as base
    return base._BYTE_COPY


def _word_copy_source() -> str:
    from . import partition_samehost_transport_v841 as base
    return base._WORD_COPY_2D
