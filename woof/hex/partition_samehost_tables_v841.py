"""Owner-to-halo index plans for any number of same-host partitions.

The partition layout remains authoritative.  Each pair's send indexes follow
the receiver's local halo order, and each receive index addresses only the
halo tail.  Planning moves integer indexes only; no field values are read.
"""

from __future__ import annotations

import hashlib
import operator
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .partition_assets_v841 import PartitionLayout
from .partition_executor_v841 import ExecutorError

TABLES_VERSION = "partition-samehost-tables-v841/1"

__all__ = [
    "TABLES_VERSION",
    "PeerHaloTables",
    "SameHostHaloExchangeTables",
    "active_peers",
    "build_all",
]


def _index_vector(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.int64)
    result.setflags(write=False)
    return result


def _index_receipt(values: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(values.size),
        "sha256_int64_le": hashlib.sha256(
            values.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }


@dataclass(frozen=True)
class PeerHaloTables:
    """Local send and receive indexes for one peer, in receiver order."""

    peer: int
    cell_send: np.ndarray
    cell_recv: np.ndarray
    edge_send: np.ndarray
    edge_recv: np.ndarray

    def receipt(self) -> dict[str, Any]:
        return {
            "peer": int(self.peer),
            "cell_send": _index_receipt(self.cell_send),
            "cell_recv": _index_receipt(self.cell_recv),
            "edge_send": _index_receipt(self.edge_send),
            "edge_recv": _index_receipt(self.edge_recv),
        }


@dataclass(frozen=True)
class SameHostHaloExchangeTables:
    """One rank's halo plan, with peers ordered by rank, including empty links."""

    rank: int
    n_ranks: int
    n_owned_cells: int
    n_local_cells: int
    n_owned_edges: int
    n_local_edges: int
    n_owned_vertices: int
    n_local_vertices: int
    peers: tuple[PeerHaloTables, ...]

    @property
    def cell_recv_count(self) -> int:
        return self.n_local_cells - self.n_owned_cells

    @property
    def edge_recv_count(self) -> int:
        return self.n_local_edges - self.n_owned_edges

    @classmethod
    def build(
        cls, layouts: Sequence[PartitionLayout], rank: int
    ) -> "SameHostHaloExchangeTables":
        try:
            rank = operator.index(rank)
        except TypeError as exc:
            raise ExecutorError("rank must be an integer") from exc
        if not 0 <= rank < len(layouts):
            raise ExecutorError(f"rank {rank} is outside {len(layouts)} layouts")
        return build_all(layouts)[rank]

    def receipt(self) -> dict[str, Any]:
        return {
            "tables_version": TABLES_VERSION,
            "rank": self.rank,
            "n_ranks": self.n_ranks,
            "n_owned_cells": self.n_owned_cells,
            "n_local_cells": self.n_local_cells,
            "n_owned_edges": self.n_owned_edges,
            "n_local_edges": self.n_local_edges,
            "n_owned_vertices": self.n_owned_vertices,
            "n_local_vertices": self.n_local_vertices,
            "cells_received_per_field": self.cell_recv_count,
            "edges_received_per_field": self.edge_recv_count,
            "peers": [peer.receipt() for peer in self.peers],
        }


def active_peers(table: SameHostHaloExchangeTables) -> tuple[int, ...]:
    """Return rank-ordered peers with at least one cell or edge halo packet."""
    return tuple(
        int(peer.peer) for peer in table.peers
        if any(indices.size for indices in (
            peer.cell_send, peer.cell_recv, peer.edge_send, peer.edge_recv
        ))
    )


def _entity_ownership(
    layouts: Sequence[PartitionLayout], kind: str, plural: str
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[int, ...]]:
    """Validate the global exact cover and retain the owner-local index."""

    extents = []
    for rank, layout in enumerate(layouts):
        try:
            extent = operator.index(getattr(layout, f"_global_{plural}"))
        except TypeError as exc:
            raise ExecutorError(f"rank {rank} {plural}: global extent is not an integer") from exc
        if extent < 0:
            raise ExecutorError(f"rank {rank} {plural}: negative global extent {extent}")
        extents.append(extent)
    if len(set(extents)) != 1:
        raise ExecutorError(f"{plural}: layouts disagree on the global extent {extents}")

    n_global = extents[0]
    owner = np.full(n_global, -1, dtype=np.int32)
    owner_local = np.full(n_global, -1, dtype=np.int64)
    local_vectors = []
    owned_counts = []
    for rank, layout in enumerate(layouts):
        vector = np.asarray(getattr(layout, f"{kind}_l2g"))
        if vector.ndim != 1 or vector.dtype.kind not in "iu":
            raise ExecutorError(f"rank {rank} {plural}: local indexes must be an integer vector")
        try:
            n_owned = operator.index(getattr(layout, f"n_owned_{plural}"))
        except TypeError as exc:
            raise ExecutorError(f"rank {rank} {plural}: owned count is not an integer") from exc
        if not 0 <= n_owned <= vector.size:
            raise ExecutorError(
                f"rank {rank} {plural}: owned count {n_owned} is outside local size {vector.size}"
            )
        if vector.size and (vector.min() < 0 or vector.max() >= n_global):
            raise ExecutorError(
                f"rank {rank} {plural}: local index is outside global extent {n_global}"
            )
        vector = np.asarray(vector, dtype=np.int64)
        # The refusal np.unique(vector).size != vector.size made, in one pass over a mark array (the indexes are
        # known to lie in [0, n_global) above): the sort held the main thread ~4 s of a 30 km 4-rank start.
        marks = np.zeros(n_global, dtype=np.bool_)
        marks[vector] = True
        if int(np.count_nonzero(marks)) != vector.size:
            raise ExecutorError(f"rank {rank} {plural}: duplicate local global index")
        owned = vector[:n_owned]
        duplicates = owned[owner[owned] >= 0]
        if duplicates.size:
            raise ExecutorError(
                f"{plural}: global index {int(duplicates[0])} has more than one owner"
            )
        owner[owned] = rank
        owner_local[owned] = np.arange(n_owned, dtype=np.int64)
        local_vectors.append(vector)
        owned_counts.append(n_owned)

    missing = np.flatnonzero(owner < 0)
    if missing.size:
        raise ExecutorError(
            f"{plural}: global index {int(missing[0])} has no owner; layouts are not an exact cover"
        )
    for rank, (vector, n_owned) in enumerate(zip(local_vectors, owned_counts)):
        if np.any(owner[vector[n_owned:]] == rank):
            raise ExecutorError(f"rank {rank} {plural}: halo tail contains this rank's owned index")
    return owner, owner_local, tuple(local_vectors), tuple(owned_counts)


def build_all(
    layouts: Sequence[PartitionLayout],
) -> tuple[SameHostHaloExchangeTables, ...]:
    """Validate all ownership classes once and plan every rank's cell/edge links.

    Layout position is the execution rank.  Requiring matching partition ids
    prevents accidentally attaching a partition's halo plan to another rank.
    Vertices carry no mutable exchanged state, but their owned union must also
    be an exact cover for global state reconstruction.
    """

    layouts = tuple(layouts)
    if not layouts:
        raise ExecutorError("same-host halo planning needs at least one layout")
    for rank, layout in enumerate(layouts):
        if layout.partition != rank:
            raise ExecutorError(
                f"layout position {rank} has partition {layout.partition}; rank order must match"
            )

    cells = _entity_ownership(layouts, "cell", "cells")
    edges = _entity_ownership(layouts, "edge", "edges")
    _entity_ownership(layouts, "vertex", "vertices")

    def pair_indexes(
        entity: tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[int, ...]],
        rank: int,
        peer: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        owner, owner_local, vectors, counts = entity
        peer_halo = vectors[peer][counts[peer]:]
        sent_globals = peer_halo[owner[peer_halo] == rank]
        send = owner_local[sent_globals]
        mine_halo = vectors[rank][counts[rank]:]
        recv = np.flatnonzero(owner[mine_halo] == peer) + counts[rank]
        return _index_vector(send), _index_vector(recv)

    planned = []
    for rank, layout in enumerate(layouts):
        peers = []
        for peer in range(len(layouts)):
            if peer == rank:
                continue
            cell_send, cell_recv = pair_indexes(cells, rank, peer)
            edge_send, edge_recv = pair_indexes(edges, rank, peer)
            peers.append(PeerHaloTables(peer, cell_send, cell_recv, edge_send, edge_recv))
        planned.append(
            SameHostHaloExchangeTables(
                rank=rank,
                n_ranks=len(layouts),
                n_owned_cells=int(layout.n_owned_cells),
                n_local_cells=layout.n_local_cells,
                n_owned_edges=int(layout.n_owned_edges),
                n_local_edges=layout.n_local_edges,
                n_owned_vertices=int(layout.n_owned_vertices),
                n_local_vertices=layout.n_local_vertices,
                peers=tuple(peers),
            )
        )
    return tuple(planned)
