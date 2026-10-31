"""The shared boundary-file reader: same-host rank threads admit one decoded copy.

Breakage it guards (VRFS 2026-10-28): 8 rank threads each read and kept a whole
7.1 GB boundary file before slicing their rows, 369 GB host RSS on a 1.25 M-cell
conus3km cut. One read per file per process, identical values, freed with its
last holder.
"""
from __future__ import annotations

import gc
import threading

import numpy as np

import woof.hex.lbc as lbc
from woof.hex.lbc import LbcInventory, LbcPool, read_lbc_file, read_lbc_file_shared

from test_lbc import write_lbc, at


def test_rank_threads_share_one_read_with_the_readers_values(tmp_path, monkeypatch):
    path = write_lbc(tmp_path / "lbc.2023-03-31_12.00.00.nc", "2023-03-31_12:00:00", 1.5)
    calls = []
    real = lbc.read_lbc_file

    def counted(p, **kw):
        calls.append(str(p))
        return real(p, **kw)

    monkeypatch.setattr(lbc, "read_lbc_file", counted)
    barrier = threading.Barrier(8)
    got = [None] * 8

    def rank(i):
        barrier.wait()
        got[i] = read_lbc_file_shared(path)

    threads = [threading.Thread(target=rank, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1
    assert all(g is got[0] for g in got)
    alone = real(path)
    assert alone.valid_time == got[0].valid_time
    assert set(alone.fields) == set(got[0].fields)
    for name, value in alone.fields.items():
        assert value.dtype == got[0].fields[name].dtype
        assert np.array_equal(value.view(np.uint32), got[0].fields[name].view(np.uint32))
        assert not got[0].fields[name].flags.writeable


def test_the_shared_copy_dies_with_its_last_holder(tmp_path):
    path = write_lbc(tmp_path / "lbc.2023-03-31_12.00.00.nc", "2023-03-31_12:00:00", 2.0)
    held = read_lbc_file_shared(path)
    assert len(lbc._SHARED_LBC) >= 1
    del held
    gc.collect()
    assert all(str(path.resolve()) != key[0] for key in list(lbc._SHARED_LBC.keys()))


def test_pools_on_every_rank_admit_the_same_object(tmp_path):
    a = write_lbc(tmp_path / "lbc.2023-03-31_12.00.00.nc", "2023-03-31_12:00:00", 1.0)
    b = write_lbc(tmp_path / "lbc.2023-03-31_13.00.00.nc", "2023-03-31_13:00:00", 2.0)
    pools = [LbcPool(LbcInventory([a, b]), compute_tendencies=False) for _ in range(4)]
    starts = [pool.start(at("2023-03-31_12:00:00")) for pool in pools]
    assert all(s is starts[0] for s in starts)
    nexts = [pool.advance() for pool in pools]
    assert all(n is nexts[0] for n in nexts)
    assert nexts[0].valid_time == at("2023-03-31_13:00:00")
