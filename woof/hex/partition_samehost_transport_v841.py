"""Resident byte-only halo transport for concurrent same-host ranks.

Every rank calls ``exchange`` at the same driver schedule position.  Fields
are gathered into one device packet per neighbour, copied once, and scattered
to that neighbour's halo indices. All peer fields share one pack launch.
Unpack uses one all-peer launch unless field storage overlaps. Integer
address calculations and byte
loads/stores are the only operations performed on field storage.  Strided
views, signed zero and NaN payloads therefore retain their representations.

The caller supplies a separate current compute stream for each rank.  Pack
events order producer work before nonblocking transfer streams; receive
events order halo writes before the caller's next kernels can run. A completed
rendezvous means every copy and reuse fence is queued. A completion worker
retains field and descriptor owners until every rank's receive event completes;
failed rounds remain owned until the caller has drained its CUDA work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict, deque
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .partition_executor_v841 import ExecutorError

__all__ = ["SameHostTransport", "SameHostTransportError", "choose_copy_paths", "exchange_timing_requested"]


def exchange_timing_requested() -> bool:
    """``WOOF_HEX_EXCHANGE_TIMING=1``: GPU-event halo timing on every same-host transport.

    One reader for the WOOF route, the MPAS-physics rank threads and the MPAS-physics rank processes. Breakage
    it prevents: the variable was read only by the ArWen-route driver, so on the MPAS-physics route (RRFS and
    MPAS-A, threads and processes) ``gpu_step_timings`` stayed empty and the exposed halo stall could not be
    measured (MULTICARD/MPAS-A-SCALING.md, 2026-10-28). Timing events add no host sync on the step path.
    """
    value = os.environ.get("WOOF_HEX_EXCHANGE_TIMING", "0")
    if value not in ("0", "1"):
        raise ValueError("WOOF_HEX_EXCHANGE_TIMING must be 0 or 1; prevents ambiguous halo timing receipts")
    return value == "1"


def _device_numa_nodes(xp: Any, devices: Sequence[int],
                       sysfs: Path = Path("/sys/bus/pci/devices")) -> dict[int, int | None]:
    """Read assigned cards' locality once; absent topology remains unknown."""
    result = {}
    for device in sorted(set(map(int, devices))):
        try:
            bus = str(xp.cuda.Device(device).pci_bus_id).lower()
            if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", bus):
                raise ValueError("noncanonical PCI address")
            node = int((sysfs / bus / "numa_node").read_text().strip())
            result[device] = node if node >= 0 else None
        except (AttributeError, OSError, ValueError):
            result[device] = None
    return result


class SameHostTransportError(ExecutorError):
    """An exchange stopped before ranks could consume inconsistent halos."""


def choose_copy_paths(
    devices: Sequence[int],
    peer_access: Mapping[tuple[int, int], bool],
    copy_policy: str = "auto",
    numa_nodes: Mapping[int, int | None] | None = None,
) -> dict[tuple[int, int], str]:
    """Resolve actual routes once, including the measured mixed-NUMA case.

    Auto keeps admitted same-socket peers and stages cross-socket traffic.
    The byte self-test still rejects peer paths that complete with wrong bytes.
    Same-card traffic always uses a device-local copy.
    """
    if copy_policy not in ("auto", "peer", "staged"):
        raise SameHostTransportError(f"unknown copy policy {copy_policy!r}")
    nodes = {device: node for device, node in (numa_nodes or {}).items()
             if node is not None and int(node) >= 0}
    paths = {}
    for src in set(map(int, devices)):
        for dst in set(map(int, devices)):
            if src == dst:
                paths[src, dst] = "local"
                continue
            both = peer_access.get((src, dst), False) and peer_access.get((dst, src), False)
            if copy_policy == "peer" and not both:
                raise SameHostTransportError(
                    f"cards {src} and {dst} lack bidirectional peer access; "
                    "a staged transfer cannot be reported as direct peer traffic"
                )
            paths[src, dst] = (
                "peer" if both and (copy_policy == "peer" or
                                     copy_policy == "auto" and
                                     (not nodes or src in nodes and dst in nodes and nodes[src] == nodes[dst]))
                else "staged"
            )
    return paths


_BYTE_COPY = r'''
extern "C" __global__ void halo_copy_bytes(
    const long long* __restrict__ table, const int scatter) {
    // One block row per field.  The descriptor carries byte strides so a
    // view is moved exactly as its ndarray indexing describes it.
    const long long* t = table + 9 * (long long)blockIdx.y;
    unsigned char* live = (unsigned char*)t[0];
    unsigned char* packet = (unsigned char*)t[1];
    const long long* indices = (const long long*)t[2];
    const long long rows = t[3];
    const long long count = t[4];
    const long long itemsize = t[5];
    const long long ndim = t[6];
    const long long* dims = (const long long*)t[7];
    const long long unit = t[8];
    const long long units_per_item = itemsize / unit;
    const long long width = count * units_per_item;
    const long long total = rows * width;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         q < total; q += stride) {
        long long row = q / width;
        const long long within = q - row * width;
        const long long column = within / units_per_item;
        const long long byte = (within - column * units_per_item) * unit;
        long long offset = indices[column] * dims[2 * ndim - 1] + byte;
        for (long long dim = ndim - 2; dim >= 0; --dim) {
            const long long coordinate = row % dims[2 * dim];
            row /= dims[2 * dim];
            offset += coordinate * dims[2 * dim + 1];
        }
        if (unit == 4) {
            unsigned int* value = (unsigned int*)(live + offset);
            unsigned int* packed = (unsigned int*)(packet + 4 * q);
            if (scatter) *value = *packed;
            else *packed = *value;
        } else {
            if (scatter) live[offset] = packet[q];
            else packet[q] = live[offset];
        }
    }
}
'''


_WORD_COPY_2D = r'''
extern "C" __global__ void halo_copy_words_2d(
    const long long* __restrict__ table, const int scatter) {
    // Cached descriptors admit only aligned 32-bit 1-D/2-D fields with
    // unit cell stride and at most INT_MAX words in each peer field.
    const long long* t = table + 9 * (long long)blockIdx.y;
    unsigned int* live = (unsigned int*)t[0];
    unsigned int* packet = (unsigned int*)t[1];
    const long long* indices = (const long long*)t[2];
    const int columns = (int)t[4];
    const long long total = t[3] * columns;
    const long long* dims = (const long long*)t[7];
    const long long row_stride = dims[1] / 4;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         q < total; q += stride) {
        const int row = (int)q / columns;
        const int column = (int)q - row * columns;
        const long long offset = (long long)row * row_stride + indices[column];
        if (scatter) live[offset] = packet[q];
        else packet[q] = live[offset];
    }
}
'''


@dataclass
class _Packet:
    buffer: Any
    specs: tuple[tuple[Any, tuple[int, ...], int, int], ...]
    indices: tuple[Any, ...]
    descriptor: Any = None
    dimensions: tuple[Any, ...] = ()
    pointers: tuple[Any, ...] = ()
    host: Any = None
    source_stream: Any = None
    host_descriptor: Any = None


@dataclass
class _Binding:
    descriptor: Any
    owners: tuple[Any, ...]
    rows: int
    overlapping_fields: bool = False
    peer_descriptors: tuple[Any, ...] = ()
    word_rows: bool = False


@dataclass
class _Batch:
    """Retained address table for one all-peer pack or unpack launch."""

    descriptor: Any = None
    signature: tuple[Any, ...] = ()
    rows: int = 0
    owners: tuple[Any, ...] = ()
    bindings: OrderedDict = field(default_factory=OrderedDict)
    serial_unpack: bool = False


@dataclass
class _Buffers:
    send: dict[int, _Packet]
    recv: dict[int, _Packet]
    send_batch: _Batch = field(default_factory=_Batch)
    recv_batch: _Batch = field(default_factory=_Batch)


@dataclass
class _Submission:
    fields: tuple[tuple[str, Any, str], ...]
    buffers: _Buffers
    stream: Any = None
    produced: Any = None
    received: Any = None
    timing: dict[str, Any] = field(default_factory=dict)
    # A queued kernel retains pointer addresses, not Python allocation owners.
    owners: list[Any] = field(default_factory=list)


@dataclass
class _Round:
    generation: int
    deadline: float
    tag: Any = None
    schema: tuple[Any, ...] | None = None
    entered: set[int] = field(default_factory=set)
    submissions: dict[int, _Submission] = field(default_factory=dict)
    completed: set[int] = field(default_factory=set)
    done: bool = False
    completion: Any = None
    producer_streams: dict[int, Any] = field(default_factory=dict)
    reserved_fields: list[Any] = field(default_factory=list)
    timing_fenced: set[int] = field(default_factory=set)


class SameHostTransport:
    """An N-rank rendezvous moving cell and edge halos without host numerics.

    ``tables`` has one table object per rank, with ``peers`` containing peer,
    cell_send/cell_recv and edge_send/edge_recv index arrays.  ``fields`` passed
    to ``exchange`` is a sequence of ``(name, array, entity_kind)`` triples.
    All ranks must use the same tag and field schema at each rendezvous.

    NumPy is a byte-layout reference for CPU tests.  Real runs use CuPy and
    source arrays never pass through ``asnumpy``.  Explicit staging allocates
    pinned byte storage and performs D2H then H2D copies without reading it.
    A bounded descriptor cache retains integer tables for rotating field
    addresses. Live fields validate each lookup and remain owned by their
    submission through DMA completion, including after cache eviction.
    """

    # Field-ownership policy (set per instance by the same-host driver; see
    # __init__ for the breakage each prevents). Class defaults keep the
    # documented contract for transports built without __init__.
    release_fields_after_fence = False
    release_fields_on_retire = True

    def __init__(
        self, tables: Sequence[Any], devices: Sequence[int], *, xp: Any = None,
        timeout: float = 60.0, copy_policy: str = "auto",
        numa_nodes: Mapping[int, int | None] | None = None,
        descriptor_cache_size: int | None = None,
        stage_same_device: bool = False,
        field_halo_indices: Mapping[tuple[int, int, str, str], Any] | None = None,
        measure_gpu: bool = False,
    ) -> None:
        if xp is None:
            import cupy as xp
        self.xp = xp
        self._cpu = xp is np
        self.measure_gpu = bool(measure_gpu) and not self._cpu
        self.gpu_step_timings = {}
        self.tables = tuple(tables)
        self.devices = tuple(map(int, devices))
        self.timeout = float(timeout)
        if descriptor_cache_size is None:
            # Breakage it prevents: at 64 bindings per batch the 15 km 8-rank
            # run missed 78% of descriptor lookups (120,684 of 155,520 in the
            # 6 h 8-card RTX PRO 6000 host baseline, 2026-10-10), rebuilding and uploading an
            # address table on the rank thread for almost every round.
            descriptor_cache_size = int(os.environ.get("WOOF_HEX_HALO_DESCRIPTOR_CACHE", "4096"))
        self.descriptor_cache_size = int(descriptor_cache_size)
        if stage_same_device and copy_policy != "staged":
            raise SameHostTransportError("same-device staging requires the explicit staged copy policy")
        self.stage_same_device = bool(stage_same_device)
        self.field_halo_indices = dict(field_halo_indices or {})
        # When each field a rank exchanges was allocated on that rank's compute
        # stream (the same-host driver builds and steps every rank on one
        # stream), the rank's own pack and fence order every later reuse of a
        # field's pool block after the halo DMA and unpack: CuPy returns a
        # freed block to the arena of the stream that allocated it. The
        # driver then sets this, and the transport drops its references to a
        # rank's fields as soon as that rank's consumer fence is queued.
        # Breakage it prevents: the schedule-slot transport kept every field
        # of every round alive until end_step, so a rank held each acoustic
        # substep's and each recovery's temporaries for the whole step
        # (+735 MiB live and +2.0 GB pool at 120 km, io-halo census 2026-10-15).
        self.release_fields_after_fence = False
        # False keeps the old schedule-slot behaviour (fields held to end_step);
        # True drops a retired round's field references, as the base transport does.
        self.release_fields_on_retire = True
        if self.descriptor_cache_size < 1 or self.descriptor_cache_size != descriptor_cache_size:
            raise SameHostTransportError("descriptor cache size must be a positive integer")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise SameHostTransportError("exchange timeout must be finite and positive")
        if not self.tables or len(self.tables) != len(self.devices):
            raise SameHostTransportError("one table and one device are required for every rank")
        if any(device < 0 for device in self.devices):
            raise SameHostTransportError("rank devices must be nonnegative ordinals")
        self.n_ranks = len(self.tables)
        self._peers = []
        self._validate_tables()
        self._validate_field_indices()
        self._condition = threading.Condition()
        self._stats_lock = threading.Lock()
        self._round = _Round(0, float("inf"))
        self._error: SameHostTransportError | None = None
        self._failed_rounds: list[_Round] = []
        self._cache: list[dict[tuple[Any, ...], _Buffers]] = [dict() for _ in self.tables]
        self._copy_streams = [None] * self.n_ranks
        self._staging_streams: dict[int, Any] = {}
        self._pending: deque[_Round] = deque()
        self._completion_worker = None
        self._watchdog_worker = None
        self._stopping = False
        self._closed = False
        self._peer_access = {}
        if not self._cpu:
            for src in set(self.devices):
                for dst in set(self.devices):
                    if src != dst:
                        self._peer_access[src, dst] = bool(xp.cuda.runtime.deviceCanAccessPeer(src, dst))
        if numa_nodes is None and not self._cpu:
            numa_nodes = _device_numa_nodes(xp, self.devices)
        self.paths = choose_copy_paths(self.devices, self._peer_access, copy_policy, numa_nodes)
        if self.stage_same_device:
            # An explicit diagnostic can exercise both pinned-host DMA legs
            # with two ranks on one physical card. Ordinary routing is intact.
            for device in set(self.devices):
                self.paths[device, device] = "staged"
        self.copy_policy = copy_policy
        self._numa_nodes = dict(numa_nodes or {})
        self._kernel = None if self._cpu else xp.RawKernel(_BYTE_COPY, "halo_copy_bytes")
        self._word_kernel = None if self._cpu else xp.RawKernel(_WORD_COPY_2D, "halo_copy_words_2d")
        if not self._cpu:
            # Compile before concurrent rank threads can enter CuPy's lazy
            # per-device module cache or spend a round's timeout in NVRTC.
            for device in set(self.devices):
                with xp.cuda.Device(device):
                    self._kernel.compile()
                    self._word_kernel.compile()
            for (src, dst), path in self.paths.items():
                if path == "peer":
                    with xp.cuda.Device(src):
                        try:
                            xp.cuda.runtime.deviceEnablePeerAccess(dst)
                        except xp.cuda.runtime.CUDARuntimeError as exc:
                            if exc.status != 704:  # cudaErrorPeerAccessAlreadyEnabled
                                raise
            self.peer_self_test = self._verify_peer_paths(copy_policy)
            for rank, device in enumerate(self.devices):
                with xp.cuda.Device(device):
                    self._copy_streams[rank] = xp.cuda.Stream(non_blocking=True)
            for device in set(self.devices):
                if any(src == device and path == "staged" for (src, _), path in self.paths.items()):
                    with xp.cuda.Device(device):
                        self._staging_streams[device] = xp.cuda.Stream(non_blocking=True)
        self.stats: dict[str, int | float] = {
            "rounds": 0, "bytes": 0, "local_bytes": 0,
            "peer_bytes": 0, "staged_bytes": 0, "seconds": 0.0,
            "completed_rounds": 0,
            "pack_launches": 0, "unpack_launches": 0,
            "descriptor_uploads": 0,
            "dimension_uploads": 0, "descriptor_cache_hits": 0,
            "descriptor_cache_misses": 0, "descriptor_cache_evictions": 0,
            "serial_unpack_batches": 0, "serial_unpack_launches": 0,
        }

    def _verify_peer_paths(self, copy_policy: str) -> list[dict[str, Any]]:
        """Run the shared peer-copy self-test over every peer route before any halo uses it.

        THE BREAKAGE THIS PREVENTS is recorded in :mod:`woof.hex.peer_self_test`: on a rented
        8x RTX PRO 6000 container peer copies completed with success codes and wrong bytes,
        and every N-rank forecast on one NUMA node went non-finite at step 1.  A route that
        does not pass is staged under ``auto`` and refused under an explicit ``peer`` policy;
        each pair's verdict is kept for the receipt.
        """
        from woof.hex.peer_self_test import check_pairs

        pairs = [pair for pair, path in sorted(self.paths.items()) if path == "peer"]
        records = check_pairs(self.xp, pairs, enable_access=False)
        for record in records:
            pair = (record["source_device"], record["destination_device"])
            if record["verdict"] == "passed":
                record["route"] = "peer"
                continue
            if copy_policy == "peer":
                raise SameHostTransportError(
                    f"peer copy from card {pair[0]} to card {pair[1]} failed the startup "
                    f"self-test ({record['verdict']}, wrong words {record['wrong_words']}); "
                    "halos moved over this route would make the forecast non-finite. "
                    "Use the staged or auto policy"
                )
            self.paths[pair] = "staged"
            record["route"] = "staged (peer self-test " + record["verdict"] + ")"
            # Leave no enabled route behind: CuPy's own cross-device copies
            # would otherwise still take it (with access disabled the same
            # memcpyPeer calls were exact on the box that failed).
            with self.xp.cuda.Device(pair[0]):
                try:
                    self.xp.cuda.runtime.deviceDisablePeerAccess(pair[1])
                except self.xp.cuda.runtime.CUDARuntimeError as exc:
                    if exc.status != 705:  # cudaErrorPeerAccessNotEnabled
                        raise
        return records

    def _validate_tables(self) -> None:
        for rank, table in enumerate(self.tables):
            if int(table.rank) != rank:
                raise SameHostTransportError("tables must be ordered by their rank ordinals")
            peers = {int(peer.peer): peer for peer in table.peers}
            if len(peers) != len(table.peers) or rank in peers or any(
                peer < 0 or peer >= self.n_ranks for peer in peers
            ):
                raise SameHostTransportError(f"rank {rank} has duplicate, self or invalid peers")
            self._peers.append(peers)
            for kind in ("cell", "edge"):
                owned = int(getattr(table, f"n_owned_{kind}s"))
                local = int(getattr(table, f"n_local_{kind}s"))
                if not 0 <= owned <= local:
                    raise SameHostTransportError(f"rank {rank} has invalid {kind} counts")
                received = []
                for peer in peers.values():
                    for direction in ("send", "recv"):
                        index = np.asarray(getattr(peer, f"{kind}_{direction}"))
                        if index.ndim != 1 or index.dtype != np.dtype(np.int64):
                            raise SameHostTransportError(f"rank {rank} {kind} {direction} indices must be int64 vectors")
                        if index.size and (
                            index.min() < (0 if direction == "send" else owned)
                            or index.max() >= (owned if direction == "send" else local)
                        ):
                            raise SameHostTransportError(f"rank {rank} {kind} {direction} index violates owned/halo storage")
                    received.extend(map(int, getattr(peer, f"{kind}_recv")))
                if sorted(received) != list(range(owned, local)):
                    raise SameHostTransportError(f"rank {rank} {kind} receive tables do not cover each halo exactly once")
        for rank, peers in enumerate(self._peers):
            for peer_rank, peer in peers.items():
                reverse = self._peers[peer_rank].get(rank)
                if reverse is None:
                    raise SameHostTransportError(f"ranks {rank} and {peer_rank} lack reciprocal halo tables")
                for kind in ("cell", "edge"):
                    if len(getattr(peer, f"{kind}_send")) != len(getattr(reverse, f"{kind}_recv")):
                        raise SameHostTransportError(f"ranks {rank} and {peer_rank} disagree on {kind} packet length")

    def _validate_field_indices(self):
        if not self.field_halo_indices:
            return
        from .partition_field_halos_v841 import FIELD_KINDS
        for (rank, peer, name, direction), index in self.field_halo_indices.items():
            if (not 0 <= rank < self.n_ranks or peer not in self._peers[rank]
                    or name not in FIELD_KINDS or direction not in ("send", "recv")):
                raise SameHostTransportError("invalid field halo identity; prevents attaching indices to the wrong rank or field")
            index = np.asarray(index)
            base = getattr(self._peers[rank][peer], f"{FIELD_KINDS[name]}_{direction}")
            if (index.ndim != 1 or index.dtype != np.dtype(np.int64) or
                    not np.array_equal(index, base[np.isin(base, index)])):
                raise SameHostTransportError("field halo indices are not an ordered subset of the owned/halo plan; prevents wrong or duplicate scatter")
            other = "recv" if direction == "send" else "send"
            reverse = self.field_halo_indices.get((peer, rank, name, other))
            if reverse is None or len(index) != len(reverse):
                raise SameHostTransportError("field halo lengths disagree across peers; prevents truncated or misaligned packets")

    def _schema(self, rank: int, fields: tuple[tuple[str, Any, str], ...]) -> tuple[Any, ...]:
        schema = []
        names = set()
        for name, array, kind in fields:
            if not isinstance(name, str) or name in names:
                raise SameHostTransportError(f"rank {rank} has an invalid or duplicate field name {name!r}")
            names.add(name)
            if kind not in ("cell", "edge"):
                raise SameHostTransportError(f"unknown entity kind {kind!r} for field {name!r}")
            local = int(getattr(self.tables[rank], f"n_local_{kind}s"))
            if not array.ndim or int(array.shape[-1]) != local:
                raise SameHostTransportError(f"rank {rank} field {name!r} last axis must equal its local {kind} count {local}")
            dtype = np.dtype(array.dtype)
            if dtype.hasobject or dtype.itemsize <= 0 or dtype.fields is not None:
                raise SameHostTransportError(f"field {name!r} requires a plain fixed-width dtype for byte transport")
            if self._cpu:
                if not isinstance(array, np.ndarray) or not array.flags.writeable:
                    raise SameHostTransportError(f"field {name!r} must be a writable NumPy array")
            elif not isinstance(array, self.xp.ndarray) or int(array.device.id) != self.devices[rank]:
                raise SameHostTransportError(f"rank {rank} field {name!r} is not resident on its assigned device")
            schema.append((name, kind, dtype.str, tuple(map(int, array.shape[:-1]))))
        return tuple(schema)

    def _make_packet(self, fields, peer, direction: str, *, rank=None) -> _Packet:
        specs, indices, offset = [], [], 0
        for name, array, kind in fields:
            index = self.field_halo_indices.get((rank, int(peer.peer), name, direction),
                                               getattr(peer, f"{kind}_{direction}"))
            shape = tuple(map(int, array.shape[:-1])) + (len(index),)
            size = math.prod(shape) * int(array.dtype.itemsize)
            specs.append((np.dtype(array.dtype), shape, offset, size))
            indices.append(self.xp.asarray(index))
            offset += size
        packet = _Packet(self.xp.empty(offset, dtype=self.xp.uint8), tuple(specs), tuple(indices))
        if not self._cpu:
            if direction == "send":
                source = int(self.xp.cuda.Device().id)
                destination = self.devices[int(peer.peer)]
                if self.paths[source, destination] == "staged" and offset:
                    packet.source_stream = self._staging_streams[source]
                    packet.host = self.xp.cuda.alloc_pinned_memory(offset)
        return packet

    def _buffers(self, rank, schema, fields) -> _Buffers:
        cached = self._cache[rank].get(schema)
        if cached is None:
            cached = _Buffers(
                {peer: self._make_packet(fields, table, "send", rank=rank) for peer, table in self._peers[rank].items()},
                {peer: self._make_packet(fields, table, "recv", rank=rank) for peer, table in self._peers[rank].items()},
            )
            self._cache[rank][schema] = cached
        return cached

    def _copy_fields(self, fields, packet: _Packet, *, scatter: bool, stream=None) -> None:
        if self._cpu:
            for (_, array, _), index, (dtype, shape, offset, size) in zip(fields, packet.indices, packet.specs):
                block = packet.buffer[offset:offset + size].view(dtype).reshape(shape)
                if scatter:
                    array[..., index] = block
                else:
                    block[...] = array[..., index]
            return
        dimensions, descriptors = self._prepare_packet(fields, packet)
        launches = 0
        if fields and int(packet.buffer.size):
            self._kernel((32, len(fields)), (256,), (packet.descriptor, np.int32(scatter)), stream=stream)
            launches = 1
        self._add_stats({"dimension_uploads": dimensions, "descriptor_uploads": descriptors,
                         "unpack_launches" if scatter else "pack_launches": launches})

    def _add_stats(self, delta: Mapping[str, int]) -> None:
        """Commit one batch of launch counters on free-threaded Python too."""
        with self._stats_lock:
            for name, amount in delta.items():
                self.stats[name] += amount

    def _prepare_packet(self, fields, packet: _Packet, *, upload: bool = True) -> tuple[int, int]:
        """Upload an address table only when field pointers or strides change."""
        dimension_uploads = descriptor_uploads = 0
        pointers = tuple((int(array.data.ptr), tuple(array.strides), tuple(array.shape),
                          np.dtype(array.dtype).str) for _, array, _ in fields)
        if pointers != packet.pointers:
            dimensions, descriptor = [], []
            for field_position, ((_, array, _), index, (_, shape, offset, _)) in enumerate(zip(fields, packet.indices, packet.specs)):
                if (packet.dimensions and tuple(array.strides) == packet.pointers[field_position][1]
                        and tuple(array.shape) == packet.pointers[field_position][2]):
                    dims = packet.dimensions[field_position]
                else:
                    dims = self.xp.asarray([value for pair in zip(array.shape, array.strides) for value in pair], dtype=self.xp.int64)
                    dimension_uploads += 1
                dimensions.append(dims)
                packed_pointer = int(packet.buffer.data.ptr) + offset
                unit = 4 if (
                    int(array.dtype.itemsize) % 4 == 0 and int(array.data.ptr) % 4 == 0
                    and packed_pointer % 4 == 0 and all(stride % 4 == 0 for stride in array.strides)
                ) else 1
                descriptor.append([
                    int(array.data.ptr), packed_pointer,
                    int(index.data.ptr), math.prod(shape[:-1]), shape[-1],
                    int(array.dtype.itemsize), int(array.ndim), int(dims.data.ptr), unit,
                ])
            packet.dimensions = tuple(dimensions)
            packet.host_descriptor = np.asarray(descriptor, dtype=np.int64).reshape((-1, 9))
            packet.descriptor = None
            packet.pointers = pointers
        if upload and packet.descriptor is None and packet.host_descriptor is not None:
            packet.descriptor = self.xp.asarray(packet.host_descriptor, dtype=self.xp.int64)
            descriptor_uploads += 1
        return dimension_uploads, descriptor_uploads

    def _copy_packets(self, fields, packets, batch: _Batch, *, scatter: bool, stream=None) -> None:
        """Fuse all peer packets into a single byte-only rank launch.

        Separate contiguous packets keep the copy interface suitable for
        replacing each peer DMA with a send/recv operation. The kernel owns
        one independent descriptor row per peer field, so field order,
        strides and payload representations are unchanged. Aliased field
        storage retains the original serialized peer scatter order.
        """
        packets = tuple(packets)
        if self._cpu:
            for packet in packets:
                self._copy_fields(fields, packet, scatter=scatter)
            return
        active = tuple(packet for packet in packets if int(packet.buffer.size))
        signature = (
            tuple(id(packet) for packet in active),
            tuple((int(array.data.ptr), tuple(array.strides), tuple(array.shape),
                   np.dtype(array.dtype).str) for _, array, _ in fields),
        )
        binding = batch.bindings.get(signature)
        delta = {"dimension_uploads": 0, "descriptor_uploads": 0,
                 "descriptor_cache_misses": 0, "descriptor_cache_hits": 0,
                 "descriptor_cache_evictions": 0,
                 "serial_unpack_batches": 0, "serial_unpack_launches": 0,
                 "unpack_launches" if scatter else "pack_launches": 0}
        if binding is None:
            overlapping = self._overlapping_fields(fields)
            for packet in active:
                dimensions, descriptors = self._prepare_packet(fields, packet, upload=overlapping)
                delta["dimension_uploads"] += dimensions
                delta["descriptor_uploads"] += descriptors
            rows = len(fields) * len(active)
            descriptor = (
                self.xp.asarray(np.concatenate([packet.host_descriptor for packet in active]), dtype=self.xp.int64)
                if rows else None
            )
            peer_descriptors = tuple(packet.descriptor for packet in active) if overlapping else ()
            owners = tuple(dim for packet in active for dim in packet.dimensions) + peer_descriptors
            # Decide once per retained address binding, never per word or field
            # in the warm exchange path. Byte/strided/high-dimensional fields
            # retain the general kernel and its exact descriptor format.
            word_rows = all(
                array.ndim in (1, 2) and array.dtype.itemsize == 4
                and array.strides[-1] == 4 and int(array.data.ptr) % 4 == 0
                and all(stride % 4 == 0 for stride in array.strides)
                for _, array, _ in fields
            ) and all(math.prod(shape) <= 2147483647
                      for packet in active for _, shape, _, _ in packet.specs)
            binding = _Binding(descriptor, owners, rows, overlapping, peer_descriptors, word_rows)
            batch.bindings[signature] = binding
            delta["descriptor_cache_misses"] += 1
            if rows:
                delta["descriptor_uploads"] += 1
            if len(batch.bindings) > self.descriptor_cache_size:
                batch.bindings.popitem(last=False)
                delta["descriptor_cache_evictions"] += 1
        else:
            batch.bindings.move_to_end(signature)
            delta["descriptor_cache_hits"] += 1
        batch.signature = signature
        batch.descriptor, batch.owners, batch.rows = binding.descriptor, binding.owners, binding.rows
        batch.serial_unpack = bool(scatter and binding.overlapping_fields)
        if batch.rows:
            if batch.serial_unpack:
                for descriptor in binding.peer_descriptors:
                    self._kernel((32, len(fields)), (256,), (descriptor, np.int32(1)), stream=stream)
                delta["unpack_launches"] += len(binding.peer_descriptors)
                delta["serial_unpack_batches"] += 1
                delta["serial_unpack_launches"] += len(binding.peer_descriptors)
            else:
                kernel = self._word_kernel if binding.word_rows else self._kernel
                kernel((32, batch.rows), (256,), (batch.descriptor, np.int32(scatter)), stream=stream)
                delta["unpack_launches" if scatter else "pack_launches"] += 1
        self._add_stats(delta)

    @staticmethod
    def _overlapping_fields(fields) -> bool:
        """Conservatively detect shared storage using integer byte extents.

        Peer scatter was serialized before fusion. Different fields may
        alias across peer destinations, so their old peer order is retained.
        Empty arrays do not own a byte span. Interleaved views can select the
        fallback even when their addressed bytes are disjoint.
        """
        spans = []
        for _, array, _ in fields:
            if not int(array.size):
                continue
            first = last = int(array.data.ptr)
            for extent, stride in zip(array.shape, array.strides):
                reach = (int(extent) - 1) * int(stride)
                first += min(0, reach)
                last += max(0, reach)
            spans.append((first, last + int(array.dtype.itemsize)))
        spans.sort()
        end = None
        for first, last in spans:
            if end is not None and first < end:
                return True
            end = last
        return False

    def _check_live(self, round_: _Round) -> None:
        if self._closed or self._stopping:
            raise SameHostTransportError("same-host transport is closed")
        if self._error is not None:
            raise self._error
        if time.monotonic() >= round_.deadline:
            self.abort(f"round {round_.generation} tag {round_.tag!r} timed out waiting for all ranks and CUDA copies")
            raise self._error

    @staticmethod
    def _retain_descriptors(submission: _Submission, packets) -> None:
        for packet in packets:
            submission.owners.extend((packet.descriptor, *packet.dimensions))

    def _start_completion_workers(self) -> None:
        # Called under the rendezvous lock before the first async round.
        if self._completion_worker is not None:
            return
        self._completion_worker = threading.Thread(
            target=self._retire_completed, name="hex-halo-completion", daemon=True,
        )
        self._watchdog_worker = threading.Thread(
            target=self._watch_completions, name="hex-halo-watchdog", daemon=True,
        )
        self._completion_worker.start()
        self._watchdog_worker.start()

    def _retire_completed(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping or self._error is not None:
                    return
                round_ = self._pending[0]
            try:
                self._wait_completion_events(round_)
            except BaseException as error:
                self.abort(f"round {round_.generation} CUDA completion failed: {error}")
                return
            with self._condition:
                if self._pending and self._pending[0] is round_:
                    self._pending.popleft()
                    self.stats["completed_rounds"] += 1
                    if self.release_fields_on_retire:
                        # Every rank's receive event of this round has
                        # completed: no queued copy or kernel reads or writes
                        # its fields. A schedule slot keeps the round object
                        # until end_step; it no longer keeps the fields.
                        self._drop_round_fields(round_)
                self._condition.notify_all()
            # Drop ndarray owners on an ordinary thread after DMA completion,
            # never from a CUDA callback that forbids CUDA allocator calls.
            round_ = None

    @staticmethod
    def _drop_round_fields(round_: _Round) -> None:
        round_.reserved_fields.clear()
        for submission in round_.submissions.values():
            submission.fields = ()

    def _release_rank_fields(self, round_: _Round, submission: _Submission, fields: Any) -> None:
        """Forget one rank's fields once its pack, unpack and fence are queued.

        Called only when ``release_fields_after_fence`` is set (see __init__):
        the rank's compute stream already waits for its own unpack, so a
        block freed now is reused only by later work on that stream.
        """
        with self._condition:
            submission.fields = ()
            round_.reserved_fields[:] = [held for held in round_.reserved_fields if held is not fields]

    def _wait_completion_events(self, round_: _Round) -> None:
        # This thread owns its current-device state. Do not repeatedly restore
        # its initial device zero or enqueue an aggregate on that device.
        # Rotating the first wait avoids assigning a fixed retirement lead.
        events = round_.completion
        first = round_.generation % len(events)
        for offset in range(len(events)):
            _rank, device, event = events[(first + offset) % len(events)]
            self.xp.cuda.Device(device).use()
            event.synchronize()
        if self.measure_gpu:
            # Consumer timing events are recorded after receive DMA is queued.
            # Do not retire a generation until every rank has installed them.
            with self._condition:
                while len(round_.timing_fenced) != self.n_ranks:
                    self._check_live(round_)
                    self._condition.wait(max(0., round_.deadline - time.monotonic()))
            waits, receives = [], []
            for rank in range(self.n_ranks):
                submission = round_.submissions[rank]
                self.xp.cuda.Device(self.devices[rank]).use()
                submission.timing["wait_end"].synchronize()
                waits.append(float(self.xp.cuda.get_elapsed_time(submission.timing["wait_start"],
                                                                submission.timing["wait_end"])) / 1000.)
                receives.append(float(self.xp.cuda.get_elapsed_time(submission.timing["receive_start"],
                                                                   submission.received)) / 1000.)
            step = str(round_.tag).split(":", 1)[0]
            with self._stats_lock:
                row = self.gpu_step_timings.setdefault(step, {
                    "rounds": 0, "wait_seconds_by_rank": [0.] * self.n_ranks,
                    "receive_seconds_by_rank": [0.] * self.n_ranks})
                row["rounds"] += 1
                for rank in range(self.n_ranks):
                    row["wait_seconds_by_rank"][rank] += waits[rank]
                    row["receive_seconds_by_rank"][rank] += receives[rank]

    def _watch_completions(self) -> None:
        with self._condition:
            while not self._stopping and self._error is None:
                if not self._pending:
                    self._condition.wait()
                    continue
                oldest = self._pending[0]
                remaining = oldest.deadline - time.monotonic()
                if remaining <= 0:
                    self.abort(
                        f"round {oldest.generation} tag {oldest.tag!r} timed out "
                        "waiting for CUDA completion; field and buffer owners are retained"
                    )
                    return
                # The completion worker wakes this condition on retirement.
                # The timeout watches wall time, without polling CUDA state.
                oldest = None
                self._condition.wait(remaining)

    def _queue_completion(self, round_: _Round) -> None:
        if self._cpu:
            self.stats["completed_rounds"] += 1
            return
        # No CUDA API runs under this rendezvous lock. Each receive event
        # already follows its producer pack, both staged legs or peer DMA,
        # and unpack. Retain all owners until every one of them completes.
        round_.completion = tuple(
            (rank, self.devices[rank], round_.submissions[rank].received)
            for rank in range(self.n_ranks)
        )
        self._start_completion_workers()
        self._pending.append(round_)
        self._condition.notify_all()

    def _fence_consumers(self, rank: int, round_: _Round, submission: _Submission, *, all_ranks=True) -> None:
        if self._cpu:
            return
        # Consumer events are unique to this generation. A fast rank may start
        # the next round before another rank issues these waits, so cached,
        # re-recorded receive events could introduce a future-round cycle.
        with self.xp.cuda.Device(self.devices[rank]):
            if self.measure_gpu:
                submission.timing["wait_start"] = self.xp.cuda.Event()
                submission.timing["wait_start"].record(submission.stream)
            for peer in (round_.submissions.values() if all_ranks else (submission,)):
                submission.stream.wait_event(peer.received)
            if self.measure_gpu:
                submission.timing["wait_end"] = self.xp.cuda.Event()
                submission.timing["wait_end"].record(submission.stream)
        if self.measure_gpu:
            with self._condition:
                round_.timing_fenced.add(rank)
                self._condition.notify_all()

    def abort(self, reason: Any) -> None:
        """Permanently stop the run and wake every rank blocked in exchange."""
        with self._condition:
            if self._error is None:
                self._error = SameHostTransportError(str(reason))
                self._failed_rounds.append(self._round)
            self._condition.notify_all()

    def _receive(self, rank: int, round_: _Round, submission: _Submission) -> None:
        if self._cpu:
            for peer, receive in submission.buffers.recv.items():
                source = round_.submissions[peer].buffers.send[rank]
                receive.buffer[...] = source.buffer
                self._copy_fields(submission.fields, receive, scatter=True)
            return
        xp = self.xp
        device = self.devices[rank]
        stream = self._copy_streams[rank]
        with xp.cuda.Device(device), stream:
            # A rank's unpack writes the same storage its local producer read.
            stream.wait_event(submission.produced)
            if self.measure_gpu:
                submission.timing["receive_start"] = xp.cuda.Event()
                submission.timing["receive_start"].record(stream)
            for peer, receive in submission.buffers.recv.items():
                self._check_live(round_)
                source = round_.submissions[peer].buffers.send[rank]
                producer = round_.submissions[peer].produced
                source_device = self.devices[peer]
                size = int(source.buffer.nbytes)
                path = self.paths[source_device, device]
                if size:
                    if path == "staged":
                        with xp.cuda.Device(source_device), source.source_stream:
                            source.source_stream.wait_event(producer)
                            xp.cuda.runtime.memcpyAsync(
                                source.host.ptr, source.buffer.data.ptr, size,
                                xp.cuda.runtime.memcpyDeviceToHost, source.source_stream.ptr,
                            )
                            staged_done = xp.cuda.Event(disable_timing=True)
                            staged_done.record(source.source_stream)
                        submission.owners.append(staged_done)
                        # Pinned storage is used only by ordered DMA. The
                        # destination DMA waits for the source DMA on-device;
                        # no CPU read or pageable staging occurs in this path.
                        stream.wait_event(staged_done)
                        xp.cuda.runtime.memcpyAsync(
                            receive.buffer.data.ptr, source.host.ptr, size,
                            xp.cuda.runtime.memcpyHostToDevice, stream.ptr,
                        )
                    else:
                        stream.wait_event(producer)
                        if path == "local":
                            xp.cuda.runtime.memcpyAsync(
                                receive.buffer.data.ptr, source.buffer.data.ptr, size,
                                xp.cuda.runtime.memcpyDeviceToDevice, stream.ptr,
                            )
                        else:
                            xp.cuda.runtime.memcpyPeerAsync(
                                receive.buffer.data.ptr, device, source.buffer.data.ptr,
                                source_device, size, stream.ptr,
                            )
            self._copy_packets(submission.fields, submission.buffers.recv.values(),
                               submission.buffers.recv_batch, scatter=True, stream=stream)
            submission.owners.extend((submission.buffers.recv_batch.descriptor,
                                      *submission.buffers.recv_batch.owners))
            submission.received = xp.cuda.Event(block=True, disable_timing=not self.measure_gpu)
            submission.received.record(stream)

    def exchange(self, rank: int, tag: Any, fields: Sequence[tuple[str, Any, str]], *, interior_work=None) -> dict[str, int | float]:
        """Queue one round and consumer fences; return byte/enqueue-time counters.

        A CUDA caller may immediately enqueue its next work on the same
        compute stream. Reading fields from another stream or the host needs
        that compute stream synchronized, or a transport ``synchronize``.
        """
        started = time.perf_counter()
        round_ = None
        try:
            rank = int(rank)
            if not 0 <= rank < self.n_ranks:
                raise SameHostTransportError(f"invalid rank {rank}")
            fields = tuple(fields)
            schema = self._schema(rank, fields)
            with self._condition:
                round_ = self._round
                self._check_live(round_)
                if not round_.entered:
                    round_.deadline = time.monotonic() + self.timeout
                    round_.tag, round_.schema = tag, schema
                elif round_.tag != tag or round_.schema != schema:
                    raise SameHostTransportError(f"round {round_.generation} rank {rank} disagrees on tag or field schema")
                if rank in round_.entered:
                    raise SameHostTransportError(f"round {round_.generation} rank {rank} entered twice")
                round_.entered.add(rank)
                # Retain before the first launch. A record/copy error can
                # abort this rank before its ready submission is published.
                round_.reserved_fields.append(fields)
            if self._cpu:
                buffers = self._buffers(rank, schema, fields)
                stream = None
                self._copy_packets(fields, buffers.send.values(), buffers.send_batch, scatter=False)
            else:
                with self.xp.cuda.Device(self.devices[rank]):
                    stream = self.xp.cuda.get_current_stream()
                    # A pack or descriptor upload can fail before its ready
                    # submission is published. Public drain must still fence
                    # that producer before any of its owners can be released.
                    with self._condition:
                        round_.producer_streams[rank] = stream
                    buffers = self._buffers(rank, schema, fields)
                    self._copy_packets(fields, buffers.send.values(), buffers.send_batch, scatter=False, stream=stream)
            submission = _Submission(fields, buffers, stream)
            if not self._cpu:
                with self.xp.cuda.Device(self.devices[rank]):
                    submission.owners.extend((buffers.send_batch.descriptor, *buffers.send_batch.owners))
                    submission.produced = self.xp.cuda.Event(disable_timing=True)
                    submission.produced.record(stream)
            with self._condition:
                self._check_live(round_)
                round_.submissions[rank] = submission
                self._condition.notify_all()
                while len(round_.submissions) != self.n_ranks:
                    self._check_live(round_)
                    self._condition.wait(max(0.0, round_.deadline - time.monotonic()))
            self._receive(rank, round_, submission)
            if interior_work is not None:
                # Pack is ordered before this rank's compute stream callback.
                # The callback may touch only owned storage and owned inputs.
                # Receive DMA and halo scatter run on a separate stream.
                interior_work()
            sent = sum(int(packet.buffer.nbytes) for packet in buffers.send.values())
            received = sum(int(packet.buffer.nbytes) for packet in buffers.recv.values())
            with self._condition:
                self._check_live(round_)
                round_.completed.add(rank)
                if len(round_.completed) == self.n_ranks:
                    moved = 0
                    for source_rank, source_submission in round_.submissions.items():
                        for peer, packet in source_submission.buffers.send.items():
                            size = int(packet.buffer.nbytes)
                            moved += size
                            path = self.paths[self.devices[source_rank], self.devices[peer]]
                            self.stats[f"{path}_bytes"] += size
                    self.stats["rounds"] += 1
                    self.stats["bytes"] += moved
                    self.stats["seconds"] += max(0.0, time.monotonic() - (round_.deadline - self.timeout))
                    self._queue_completion(round_)
                    round_.done = True
                    self._round = _Round(round_.generation + 1, float("inf"))
                    self._condition.notify_all()
                while not round_.done:
                    self._check_live(round_)
                    self._condition.wait(max(0.0, round_.deadline - time.monotonic()))
            self._fence_consumers(rank, round_, submission)
            if self.release_fields_after_fence and not self._cpu:
                self._release_rank_fields(round_, submission, fields)
            return {"bytes_sent": sent, "bytes_received": received, "seconds": time.perf_counter() - started}
        except BaseException as exc:
            self.abort(f"same-host exchange failed: {exc}")
            if isinstance(exc, SameHostTransportError):
                raise
            raise self._error from exc

    def transport_report(self) -> list[dict[str, Any]]:
        """Report requested and actual card routes separately."""
        return [
            {
                "source_device": src, "destination_device": dst,
                "requested": self.copy_policy,
                "can_access_peer": bool(self._peer_access.get((src, dst), False)),
                "actual": "host-reference" if self._cpu else path,
                "host_legs": 0 if self._cpu or path != "staged" else 2,
            }
            for (src, dst), path in sorted(self.paths.items())
        ]

    def receipt(self) -> dict[str, Any]:
        """Small transport evidence, with field bytes counted once per direction."""
        with self._condition, self._stats_lock:
            return {
                "transport_version": "partition-samehost-transport-v841/1",
                "ranks": self.n_ranks, "devices": list(self.devices),
                "timeout_seconds": self.timeout,
                "descriptor_cache_size": self.descriptor_cache_size,
                "stage_same_device": self.stage_same_device,
                "pack_unpack": "all-peer byte kernels; overlapping field storage retains serial peer scatter",
                "routes": self.transport_report(), "stats": dict(self.stats),
                "peer_self_test": list(getattr(self, "peer_self_test", [])),
                "field_halo_index_vectors": len(self.field_halo_indices),
                "gpu_step_timings": {step: {key: list(value) if isinstance(value, list) else value
                                            for key, value in row.items()}
                                     for step, row in self.gpu_step_timings.items()},
                "gpu_timing_law": "receive includes peer producer waits and DMA/unpack; compute-stream wait after interior work is exposed halo stall, per rank; overlapping rank durations are never added across cards",
                "aborted": self._error is not None,
                "pending_rounds": len(self._pending),
                "completion": "host-reference" if self._cpu else "per-rank receive fences with a blocking completion worker",
                "completion_aggregate_device": None,
                "seconds_law": "rank rendezvous and enqueue time; GPU completion is included only by the caller's stream or transport synchronization",
            }

    def synchronize(self, *, timeout: float | None = None) -> None:
        """Drain transport streams, including after abort, before freeing storage.

        Call after rank workers have stopped issuing exchanges.  Failure keeps
        every buffer and failed submission owned; a timeout never frees pinned
        memory while a transfer could still reference it.
        """
        if self._cpu:
            return
        seconds = self.timeout if timeout is None else float(timeout)
        if not math.isfinite(seconds) or seconds <= 0:
            raise SameHostTransportError("drain timeout must be finite and positive")
        deadline = time.monotonic() + seconds
        with self._condition:
            if self._round.entered and self._error is None:
                raise SameHostTransportError("cannot drain while rank exchanges are still being issued")
        fences = []
        streams_by_device: dict[int, list[Any]] = {}
        for rank, device in enumerate(self.devices):
            streams_by_device.setdefault(device, []).append(self._copy_streams[rank])
        for device, stream in self._staging_streams.items():
            streams_by_device.setdefault(device, []).append(stream)
        with self._condition:
            retained_rounds = (self._round, *self._pending, *self._failed_rounds)
            for round_ in retained_rounds:
                for rank, stream in round_.producer_streams.items():
                    streams_by_device.setdefault(self.devices[rank], []).append(stream)
        for device, streams in streams_by_device.items():
            with self.xp.cuda.Device(device):
                seen = set()
                for stream in streams:
                    if stream is None or int(stream.ptr) in seen:
                        continue
                    seen.add(int(stream.ptr))
                    event = self.xp.cuda.Event(disable_timing=True)
                    event.record(stream)
                    fences.append((device, event))
        for device, event in fences:
            while True:
                with self.xp.cuda.Device(device):
                    if event.done:
                        break
                if time.monotonic() >= deadline:
                    raise SameHostTransportError("transport drain timed out; buffer ownership is retained")
                time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))
        with self._condition:
            # Tail fences prove that every completed round's pack, DMA and
            # unpack has ended even if a previous failure stopped retirement.
            self.stats["completed_rounds"] += len(self._pending)
            self._pending.clear()
            self._condition.notify_all()

    def close(self, *, timeout: float | None = None) -> None:
        """Drain and stop completion workers within a bounded shutdown wait.

        Call after rank workers have stopped. If draining or joining fails,
        ownership remains retained and the transport must stay alive.
        """
        if self._closed:
            return
        seconds = self.timeout if timeout is None else float(timeout)
        if not math.isfinite(seconds) or seconds <= 0:
            raise SameHostTransportError("shutdown timeout must be finite and positive")
        deadline = time.monotonic() + seconds
        self.synchronize(timeout=seconds)
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        for worker in (self._completion_worker, self._watchdog_worker):
            if worker is not None:
                worker.join(max(0.0, deadline - time.monotonic()))
                if worker.is_alive():
                    raise SameHostTransportError("completion worker shutdown timed out; ownership is retained")
        self._closed = True
