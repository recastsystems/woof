"""The same-host halo descriptor cache keeps live bindings and drops a step's dead ones.

THE BREAKAGE THIS CHECKS: each binding is keyed by its fields' device addresses, and a field the
step allocates afresh gets a new binding every step. Those were kept until a 4,096-entry LRU filled
(it never did): the 8-rank same-host x1.40962 run kept 41,194 in 120 steps and its pool floor rose
~0.7 MiB per step. CPU doubles stand in for device arrays; only the cache bookkeeping runs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woof.hex.partition_samehost_transport_v841 import SameHostTransport, _Batch, _Packet  # noqa: E402
from woof.hex.partition_samehost_v841 import SameHostHaloExchanger  # noqa: E402


class _Ptr:
    def __init__(self, ptr):
        self.ptr = ptr


class _Device:
    """A 1-D float32 device array: an address and a layout, no storage."""

    def __init__(self, ptr, n=4):
        self.data = _Ptr(ptr)
        self.shape, self.strides, self.dtype, self.ndim, self.size = (n,), (4,), np.dtype(np.float32), 1, n
        self.nbytes = 4 * n


def _transport():
    tables = [SimpleNamespace(rank=rank, peers=(SimpleNamespace(
        peer=1 - rank, cell_send=np.array([0]), cell_recv=np.array([3]),
        edge_send=np.array([0]), edge_recv=np.array([3])),),
        n_owned_cells=3, n_local_cells=4, n_owned_edges=3, n_local_edges=4) for rank in range(2)]
    transport = SameHostTransport(tables, [0, 0], xp=np)
    transport._cpu = False   # the device path's cache; kernels and uploads are doubles
    transport.xp = SimpleNamespace(asarray=lambda value, dtype=None: _Device(id(object())), int64=np.int64)
    transport._kernel = transport._word_kernel = lambda *args, **kwargs: None
    return transport


def _packet():
    return _Packet(_Device(7000, 4), ((np.dtype(np.float32), (1,), 0, 4),), (_Device(9000, 1),))


def _step(transport, batch, packet, live, temporary):
    fields = (("live", live, "cell"),)
    transport._copy_packets(fields, (packet,), batch, scatter=False)
    transport._copy_packets((("temporary", temporary, "cell"),), (packet,), batch, scatter=False)


def test_a_temporarys_bindings_are_retired_and_live_bindings_stay_hit():
    transport = _transport()
    batch, packet = _Batch(), _packet()
    transport._cache[0][("schema",)] = SimpleNamespace(send_batch=batch, recv_batch=_Batch())
    live = _Device(1000)
    for step in range(50):
        _step(transport, batch, packet, live, _Device(2000 + 64 * step))   # a fresh address every step
        transport.retire_stale_bindings(0)
    # Old code: 51 bindings, one per step's temporary plus the live field, none ever dropped.
    assert len(batch.bindings) <= 1 + transport.stale_binding_steps
    assert transport.stats["descriptor_cache_hits"] == 49            # the live field after step 0
    assert transport.stats["descriptor_cache_misses"] == 51
    assert transport.stats["descriptor_cache_retired"] >= 50 - transport.stale_binding_steps


def test_two_level_rotation_and_idle_batches_keep_their_bindings():
    transport = _transport()
    batch, packet = _Batch(), _packet()
    idle, idle_packet = _Batch(), _packet()
    transport._cache[0][("a",)] = SimpleNamespace(send_batch=batch, recv_batch=idle)
    levels = (_Device(1000), _Device(1100))
    transport._copy_packets((("x", levels[1], "cell"),), (idle_packet,), idle, scatter=True)
    for step in range(20):
        transport._copy_packets((("x", levels[step % 2], "cell"),), (packet,), batch, scatter=False)
        transport.retire_stale_bindings(0)
    # A two-level rotation is hit every other step; a batch that ran once keeps its binding while idle.
    assert transport.stats["descriptor_cache_misses"] == 3
    assert len(batch.bindings) == 2 and len(idle.bindings) == 1


def test_the_exchanger_retires_at_every_step_end():
    calls = []
    link = SimpleNamespace(retire_stale_bindings=calls.append)
    exchanger = SameHostHaloExchanger(SimpleNamespace(rank=3), link, expected_sequence=())
    for step in range(3):
        exchanger.begin_step(step)
        exchanger.end_step()
    assert calls == [3, 3, 3]
