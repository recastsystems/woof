"""Halo rounds between rank processes on any number of hosts, through NCCL point-to-point.

The same rounds, packets and byte kernels as :class:`~woof.hex.partition_process_transport_v841.ProcessHaloTransport`
(both reuse :class:`~woof.hex.partition_samehost_transport_v841.SameHostTransport`'s schema check, packet layout,
descriptor cache and pack/unpack launches), so a halo carries the same bytes in the same round order and the
forecast's words do not depend on the transport or on how many hosts the ranks span.

How a round moves:

1. this rank packs every neighbour's packet on its compute stream (one fused launch) and records ``produced``;
2. its comm stream waits for ``produced`` and issues one NCCL group: ``ncclSend`` of each non-empty send packet and
   ``ncclRecv`` of each non-empty receive packet, neighbours in rank order.  NCCL picks the route per pair: card to
   card (P2P/NVLink) or shared host memory inside a host, InfiniBand/RoCE/EFA (GPUDirect where offered) or TCP
   sockets between hosts;
3. the comm stream unpacks every received packet (one fused launch) and records ``unpacked``;
4. the caller's ``interior_work`` (work that reads no halo word) runs on the compute stream while 2-3 run;
5. the compute stream waits for ``unpacked``.

No host thread waits for a card and no host meets another host per round: the hosts never exchange a header.  The
only links are to ranks that share at least one halo cell or edge (``active_peers``).

Checks that replace the per-round header of the shared-memory transport:

- Every ``digest_every`` rounds (and every round of the first ``digest_rounds``) each rank also sends each
  neighbour a 16-byte digest of the round number, tag and field schema in the same group; the received digests come
  back to pinned host memory on the comm stream and are compared when the round has completed (off the step path).
  A schedule or schema disagreement stops the run with both digests named, instead of mixing halos.
- A watchdog thread reads NCCL's asynchronous error and the oldest unfinished round; a failed peer or network, or no
  completion in ``timeout`` seconds, aborts the communicator (queued NCCL kernels return) so the rank stops with a
  named error instead of hanging.
"""

from __future__ import annotations

from collections import deque
import hashlib
import os
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .partition_samehost_tables_v841 import active_peers
from .partition_samehost_transport_v841 import (SameHostTransport, SameHostTransportError,
                                                exchange_timing_requested)

__all__ = ["NcclHaloTransport"]

_DIGEST_BYTES = 16


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    text = os.environ.get(name)
    if text is None or not text.strip():
        return default
    value = int(text)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


class NcclHaloTransport(SameHostTransport):
    """This rank's end of the halo rounds over an NCCL communicator (one communicator per run, every rank)."""

    collective_close = True

    def __init__(self, tables: Sequence[Any], devices: Sequence[int], rank: int, communicator: Any, *, xp: Any,
                 timeout: float = 1800.0,
                 field_halo_indices: Mapping[tuple[int, int, str, str], Any] | None = None,
                 measure_gpu: bool | None = None) -> None:
        # Not SameHostTransport.__init__: that enables peer routes and builds streams on every card from one
        # process.  This process touches only its own card; NCCL owns the routes.
        self.xp = xp
        self._cpu = xp is np
        if self._cpu:
            raise SameHostTransportError("the NCCL transport moves CUDA packets; the CPU reference is SameHostTransport")
        self.measure_gpu = exchange_timing_requested() if measure_gpu is None else bool(measure_gpu)
        self.gpu_step_timings: dict[str, dict[str, float]] = {}
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
        if int(communicator.nranks) != self.n_ranks or int(communicator.rank) != self.rank:
            raise SameHostTransportError(
                f"NCCL communicator is rank {communicator.rank} of {communicator.nranks}, but this transport is rank "
                f"{self.rank} of {self.n_ranks}; prevents halos sent to the wrong rank")
        self._peers = []
        self._validate_tables()
        self._validate_field_indices()
        self.neighbours = active_peers(self.tables[self.rank])
        self._nccl = communicator
        self._stats_lock = threading.Lock()
        self._condition = threading.Condition()
        self._cache = [dict() for _ in self.tables]
        # SameHostTransport._make_packet reads a route per device pair; none is "staged" here (NCCL owns the route).
        self.paths = {(src, dst): "nccl" for src in set(self.devices) for dst in set(self.devices)}
        self.copy_policy = "nccl"
        self._error: SameHostTransportError | None = None
        self._closed = False
        self._generation = 0
        self._retained: deque = deque()
        self._digests: deque = deque()
        # Digest rounds: every round of the first digest_rounds, then every digest_every-th round (0: never after).
        self.digest_rounds = _int_env("WOOF_HEX_NCCL_DIGEST_ROUNDS", 512)
        self.digest_every = _int_env("WOOF_HEX_NCCL_DIGEST_EVERY", 50)
        with xp.cuda.Device(self.device):
            self._kernel = xp.RawKernel(_byte_copy_source(), "halo_copy_bytes")
            self._word_kernel = xp.RawKernel(_word_copy_source(), "halo_copy_words_2d")
            self._kernel.compile()
            self._word_kernel.compile()
            # Highest priority, so a round's NCCL and unpack kernels are scheduled ahead of the interior work they
            # overlap (CUDA priority: lower number is higher).
            try:
                _, highest = xp.cuda.runtime.deviceGetStreamPriorityRange()
                self._comm_stream = xp.cuda.Stream(non_blocking=True, priority=highest)
            except (AttributeError, TypeError):
                self._comm_stream = xp.cuda.Stream(non_blocking=True)
            count = max(1, len(self.neighbours))
            self._digest_send = xp.zeros((count, _DIGEST_BYTES), dtype=xp.uint8)
            self._digest_recv = xp.zeros((count, _DIGEST_BYTES), dtype=xp.uint8)
        self.stats: dict[str, int | float] = {
            "rounds": 0, "bytes": 0, "received_bytes": 0, "local_bytes": 0, "peer_bytes": 0, "staged_bytes": 0,
            "nccl_bytes": 0, "seconds": 0.0, "completed_rounds": 0, "nccl_groups": 0, "nccl_sends": 0,
            "nccl_recvs": 0, "digest_rounds": 0, "digests_checked": 0,
            "pack_launches": 0, "unpack_launches": 0, "descriptor_uploads": 0, "dimension_uploads": 0,
            "descriptor_cache_hits": 0, "descriptor_cache_misses": 0, "descriptor_cache_evictions": 0,
            "descriptor_cache_retired": 0,
            "serial_unpack_batches": 0, "serial_unpack_launches": 0, "host_enqueue_seconds": 0.0,
        }
        self.peer_bytes = {peer: 0 for peer in self.neighbours}
        self._watchdog = threading.Thread(target=self._watch, name=f"hex-nccl-watchdog-{self.rank}", daemon=True)
        self._watchdog.start()

    # -- the round -------------------------------------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise SameHostTransportError("NCCL halo transport is closed")
        if self._error is not None:
            raise self._error

    def _digest_round(self, generation: int) -> bool:
        return generation <= self.digest_rounds or (self.digest_every and generation % self.digest_every == 0)

    def _retire(self) -> None:
        while self._retained and self._retained[0][0].done and (
                self._retained[0][3] is None or self._retained[0][3]["wait_end"].done):
            timing = self._retained.popleft()[3]
            if timing is not None:
                self._record_timing(timing)
        while self._digests and self._digests[0][0].done:
            self._check_digest(self._digests.popleft())
        self.stats["completed_rounds"] = self.stats["rounds"] - len(self._retained)

    def _check_digest(self, entry) -> None:
        _event, generation, tag, expected, host, peers = entry
        received = np.asarray(host).reshape(-1, _DIGEST_BYTES)
        for slot, peer in enumerate(peers):
            theirs = bytes(received[slot])
            if theirs != expected:
                self.abort(f"round {generation} (tag {tag!r}): rank {peer} sent digest {theirs.hex()} for this "
                           f"round, rank {self.rank} expected {expected.hex()}; the two ranks disagree on the round, "
                           "tag or field schema, so their halos would mix")
                raise self._error
        with self._stats_lock:
            self.stats["digests_checked"] += len(peers)

    def _record_timing(self, timing) -> None:
        elapsed = self.xp.cuda.get_elapsed_time
        wait = float(elapsed(timing["wait_start"], timing["wait_end"])) / 1000.
        transfer = float(elapsed(timing["start"], timing["sent"])) / 1000.
        receive = float(elapsed(timing["start"], timing["unpacked"])) / 1000.
        with self._stats_lock:
            row = self.gpu_step_timings.setdefault(timing["step"], {"rounds": 0, "wait_seconds": 0.,
                                                                     "receive_seconds": 0., "transfer_seconds": 0.})
            row["rounds"] += 1
            row["wait_seconds"] += wait
            row["receive_seconds"] += receive
            row["transfer_seconds"] += transfer

    def exchange(self, rank: int, tag: Any, fields: Sequence[tuple[str, Any, str]], *,
                 interior_work=None) -> dict[str, int | float]:
        """Queue this rank's half of one round on its comm stream and the consumer fence on its compute stream."""
        started = time.perf_counter()
        try:
            if int(rank) != self.rank:
                raise SameHostTransportError(f"rank {rank} used rank {self.rank}'s NCCL transport")
            self._check_open()
            fields = tuple(fields)
            schema = self._schema(self.rank, fields)
            self._generation += 1
            generation = self._generation
            xp = self.xp
            sent = received = 0
            with xp.cuda.Device(self.device):
                stream = xp.cuda.get_current_stream()
                buffers = self._buffers(self.rank, schema, fields)
                previous = getattr(buffers, "nccl_sent", None)
                if previous is not None:
                    # The next pack into these send packets waits for the previous round's sends out of them.
                    stream.wait_event(previous)
                self._copy_packets(fields, buffers.send.values(), buffers.send_batch, scatter=False, stream=stream)
                produced = xp.cuda.Event(disable_timing=True)
                produced.record(stream)
                owners = [buffers.send_batch.descriptor, *buffers.send_batch.owners]
                comm = self._comm_stream
                digest = None
                with comm:
                    comm.wait_event(produced)   # the unpack writes storage this rank's pack read
                    timing = None
                    if self.measure_gpu:
                        timing = {"step": str(tag).split(":", 1)[0], "start": xp.cuda.Event()}
                        timing["start"].record(comm)
                    peers = self.neighbours
                    if peers and self._digest_round(generation):
                        expected = hashlib.blake2b(repr((generation, tag, schema)).encode(),
                                                   digest_size=_DIGEST_BYTES).digest()
                        import cupyx
                        mirror = cupyx.empty_pinned((len(peers) * _DIGEST_BYTES,), dtype=np.uint8)
                        self._digest_send[:len(peers)].set(
                            np.frombuffer(expected * len(peers), dtype=np.uint8).reshape(len(peers), _DIGEST_BYTES),
                            stream=comm)
                        digest = (generation, tag, expected, mirror, tuple(peers))
                    self._nccl.group_start()
                    try:
                        for slot, peer in enumerate(peers):
                            out_packet, in_packet = buffers.send[peer], buffers.recv[peer]
                            size_out, size_in = int(out_packet.buffer.nbytes), int(in_packet.buffer.nbytes)
                            if size_out:
                                self._nccl.send(out_packet.buffer.data.ptr, size_out, peer, comm.ptr)
                                self.stats["nccl_sends"] += 1
                                self.peer_bytes[peer] += size_out
                                sent += size_out
                            if size_in:
                                self._nccl.recv(in_packet.buffer.data.ptr, size_in, peer, comm.ptr)
                                self.stats["nccl_recvs"] += 1
                                received += size_in
                            if digest is not None:
                                self._nccl.send(self._digest_send[slot].data.ptr, _DIGEST_BYTES, peer, comm.ptr)
                                self._nccl.recv(self._digest_recv[slot].data.ptr, _DIGEST_BYTES, peer, comm.ptr)
                    finally:
                        self._nccl.group_end()
                    done_sending = xp.cuda.Event(disable_timing=not self.measure_gpu)
                    done_sending.record(comm)
                    buffers.nccl_sent = done_sending
                    if digest is not None:
                        xp.cuda.runtime.memcpyAsync(digest[3].ctypes.data, self._digest_recv.data.ptr,
                                                    len(peers) * _DIGEST_BYTES,
                                                    xp.cuda.runtime.memcpyDeviceToHost, comm.ptr)
                        arrived = xp.cuda.Event(disable_timing=True)
                        arrived.record(comm)
                        self._digests.append((arrived, *digest[:3], digest[3], digest[4]))
                        self.stats["digest_rounds"] += 1
                    self._copy_packets(fields, buffers.recv.values(), buffers.recv_batch, scatter=True, stream=comm)
                    owners.extend((buffers.recv_batch.descriptor, *buffers.recv_batch.owners))
                    unpacked = xp.cuda.Event(disable_timing=not self.measure_gpu)
                    unpacked.record(comm)
                enqueued = time.perf_counter() - started
                if interior_work is not None:
                    # Pack is ordered before the caller's work; send, receive and unpack run on the comm stream.
                    interior_work()
                if timing is not None:
                    timing.update(sent=done_sending, unpacked=unpacked, wait_start=xp.cuda.Event(),
                                  wait_end=xp.cuda.Event())
                    timing["wait_start"].record(stream)
                stream.wait_event(unpacked)
                if timing is not None:
                    timing["wait_end"].record(stream)
                # Fields, packets and descriptors stay referenced until the unpack has run on the card.
                self._retained.append((unpacked, owners, fields, timing, time.monotonic()))
                with self._stats_lock:
                    self.stats["rounds"] += 1
                    self.stats["nccl_groups"] += 1
                    self.stats["bytes"] += sent
                    self.stats["nccl_bytes"] += sent
                    self.stats["received_bytes"] += received
                    self.stats["host_enqueue_seconds"] += enqueued
                    self.stats["seconds"] += time.perf_counter() - started
                self._retire()
            return {"bytes_sent": sent, "bytes_received": received, "seconds": time.perf_counter() - started}
        except BaseException as exc:
            self.abort(f"rank {self.rank} NCCL halo exchange failed: {exc}")
            if isinstance(exc, SameHostTransportError):
                raise
            raise self._error from exc

    # -- watchdog and lifecycle ------------------------------------------------------------------------------------

    def _watch(self) -> None:
        """Abort the communicator on an NCCL error or when the oldest round makes no progress for ``timeout`` s."""
        while not self._closed and self._error is None:
            time.sleep(2.0)
            if self._closed or self._error is not None:
                return
            try:
                code = self._nccl.async_error()
            except Exception:
                code = 0
            if code not in (0, 7):
                self.abort(f"rank {self.rank}: NCCL reported {self._nccl.error_text(code)} (a peer rank, its host "
                           "or the network between them failed)")
                return
            try:
                oldest = self._retained[0]
            except IndexError:
                continue
            if not oldest[0].done and time.monotonic() - oldest[4] > self.timeout:
                self.abort(f"rank {self.rank}: a halo round queued {self.timeout:.0f} s ago has not completed; a "
                           "neighbour rank stopped sending (it failed, or its host or the network did)")
                return

    def abort(self, reason: Any) -> None:
        """Stop this rank's rounds: record the reason and abort the communicator, so queued NCCL kernels return."""
        if self._error is None:
            self._error = SameHostTransportError(str(reason))
        try:
            self._nccl.abort(reason)
        except Exception:
            pass

    def synchronize(self, *, timeout: float | None = None) -> None:
        with self.xp.cuda.Device(self.device):
            self._comm_stream.synchronize()
            if self.measure_gpu and self._error is None:
                for entry in self._retained:
                    if entry[3] is not None:
                        entry[3]["wait_end"].synchronize()
                        self._record_timing(entry[3])
            if self._error is None:
                while self._digests:
                    self._digests[0][0].synchronize()
                    self._check_digest(self._digests.popleft())
        self._retained.clear()

    def close(self, *, timeout: float | None = None) -> None:
        if self._closed:
            return
        try:
            if self._error is None:
                self.synchronize(timeout=timeout)
        finally:
            self._closed = True
            if self._error is None:
                try:
                    self._nccl.destroy()
                except Exception:
                    pass

    def transport_report(self) -> list[dict[str, Any]]:
        return [{"source_device": self.device, "destination_rank": peer, "actual": "RCCL point-to-point" if self.xp.cuda.runtime.is_hip else "NCCL point-to-point",
                 "bytes_sent": int(self.peer_bytes.get(peer, 0))} for peer in self.neighbours]

    def receipt(self) -> dict[str, Any]:
        from . import nccl_runtime
        with self._stats_lock:
            try:
                library, version = nccl_runtime.library_path(), nccl_runtime.version()
            except Exception as error:   # receipts never fail a finished forecast
                library, version = None, f"unavailable ({error})"
            return {
                "transport_version": "partition-nccl-transport-v841/1",
                "rank": self.rank, "ranks": self.n_ranks, "device": self.device,
                "neighbours": list(self.neighbours),
                "timeout_seconds": self.timeout,
                "nccl_library": library, "nccl_version": version,
                "nccl_environment": {name: value for name, value in os.environ.items() if name.startswith("NCCL_")},
                "digest_rounds": self.digest_rounds, "digest_every": self.digest_every,
                "descriptor_cache_size": self.descriptor_cache_size,
                "pack_unpack": "all-peer byte kernels (SameHostTransport's); overlapping field storage retains "
                               "serial peer scatter",
                "routes": self.transport_report(), "stats": dict(self.stats),
                "gpu_step_timings": {step: dict(row) for step, row in self.gpu_step_timings.items()},
                "gpu_timing": _timing_summary(self.gpu_step_timings) if self.measure_gpu else None,
                "gpu_timing_law": "this rank only, CUDA events read once complete (no host sync added): transfer = "
                                  "comm stream from the pack's completion to the end of the NCCL group; receive = "
                                  "transfer + unpack; wait = compute stream stalled on the unpack after the interior "
                                  "work (the exposed halo stall); seconds summed per step label",
                "aborted": self._error is not None,
                "abort_reason": None if self._error is None else str(self._error),
                "completion": "NCCL group per round on a high-priority comm stream; the compute stream waits on the "
                              "unpack event; no host thread waits for a card and hosts exchange no per-round header",
                "seconds_law": "host enqueue time of the round (no network wait on the host)",
            }


def _timing_summary(rows: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    waits = sorted(float(row["wait_seconds"]) for row in rows.values())
    receives = sorted(float(row["receive_seconds"]) for row in rows.values())
    transfers = sorted(float(row.get("transfer_seconds", 0.0)) for row in rows.values())
    middle = len(waits) // 2
    return {
        "steps": len(waits), "rounds": sum(int(row["rounds"]) for row in rows.values()),
        "exposed_wait_seconds": sum(waits), "receive_seconds": sum(receives), "transfer_seconds": sum(transfers),
        "median_step_exposed_wait_seconds": waits[middle] if waits else None,
        "median_step_receive_seconds": receives[middle] if receives else None,
        "median_step_transfer_seconds": transfers[middle] if transfers else None,
    }


def _byte_copy_source() -> str:
    from . import partition_samehost_transport_v841 as base
    return base._BYTE_COPY


def _word_copy_source() -> str:
    from . import partition_samehost_transport_v841 as base
    return base._WORD_COPY_2D
