from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.lbc import LbcFile
from woof.hex.partition_regional_v841 import PartitionLbcPoolV841, project_lbc_file


def test_lbc_projection_preserves_payload_and_entity_order():
    layout = SimpleNamespace(cell_l2g=np.array([3, 0, 2]), edge_l2g=np.array([1, 4]),
                             _global_cells=4, _global_edges=5)
    bits = np.array([0x80000000, 0x7fc12345, 0, 0x3f800000], dtype=np.uint32)
    cells = np.repeat(bits.view(np.float32)[:, None], 2, axis=1)
    edges = np.arange(10, dtype=np.float32).reshape(5, 2)
    source = LbcFile(Path("pool.nc"), datetime(2026, 1, 1), {"lbc_theta": cells, "lbc_u": edges})
    result = project_lbc_file(source, layout)
    assert result.path == source.path and result.valid_time == source.valid_time
    assert result.fields["lbc_theta"].tobytes() == cells[[3, 0, 2]].tobytes()
    assert result.fields["lbc_u"].tobytes() == edges[[1, 4]].tobytes()
    assert not result.fields["lbc_theta"].flags.writeable
    with pytest.raises(ValueError, match="expected 4"):
        project_lbc_file(LbcFile(source.path, source.valid_time, {"lbc_theta": cells[:3]}), layout)


def test_projected_pool_delegates_admission_and_clock():
    stamp = datetime(2026, 1, 1)
    source = LbcFile(Path("pool.nc"), stamp, {"lbc_theta": np.zeros((2, 1), np.float32)})
    class Pool:
        interval_end = stamp
        def start(self, when):
            assert when is stamp
            return source
        def advance(self, when):
            assert when is stamp
            return source
    layout = SimpleNamespace(cell_l2g=np.array([1]), _global_cells=2)
    view = PartitionLbcPoolV841(Pool(), layout)
    assert view.interval_end is stamp
    assert view.start(stamp).n_cells == 1
    assert view.advance(stamp).n_cells == 1


def _series(tmp_path, hours=4):
    from datetime import timedelta
    stamp = datetime(2023, 3, 31, 12)
    files = {}
    for hour in range(hours):
        path = tmp_path / f"lbc.{hour:02d}.nc"
        cells = (np.arange(8, dtype=np.float32)[:, None] + 100 * hour) * np.ones((1, 3), np.float32)
        edges = (np.arange(10, dtype=np.float32)[:, None] - 100 * hour) * np.ones((1, 3), np.float32)
        files[str(path)] = LbcFile(path, stamp + timedelta(hours=hour),
                                   {"lbc_theta": cells, "lbc_qv": cells * 2, "lbc_u": edges})
    return files


class _Inventory:
    def __init__(self, files):
        self.files = files
        self.identity = ("series",) + tuple(files)

    def latest_before(self, when):
        return [p for p, f in self.files.items() if f.valid_time <= when][-1]

    def earliest_strictly_after(self, when):
        later = [p for p, f in self.files.items() if f.valid_time > when]
        if not later:
            from woof.hex.lbc import LbcAdmissionError
            raise LbcAdmissionError("no later file")
        return later[0]


def test_shared_lbc_reads_read_each_file_once_for_every_rank(tmp_path):
    """The conus3km breakage: N ranks each read every full boundary file.  Now one read per file, the same rows."""
    import threading
    from woof.hex.lbc import LbcPool
    from woof.hex.partition_regional_v841 import RankLbcReader, SharedLbcReads

    files = _series(tmp_path)
    calls = []
    lock = threading.Lock()

    def read(path, extra_scalars=()):
        with lock:
            calls.append(str(path))
        return files[str(path)]

    shared = SharedLbcReads(read=read)
    inventory = _Inventory(files)
    layouts = [SimpleNamespace(cell_l2g=np.array(c), edge_l2g=np.array(e), _global_cells=8, _global_edges=10)
               for c, e in (([0, 1, 2, 5], [0, 1, 2, 3, 9]), ([3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9]))]
    pools = []
    for layout in layouts:
        pool = LbcPool(inventory, scalar_names=("lbc_qv",), compute_tendencies=False)
        pool.reader = RankLbcReader(shared, inventory, layout)
        pools.append(pool)
    stamp = datetime(2023, 3, 31, 12)
    got = [[] for _ in pools]

    def rank(i):
        got[i].append(pools[i].start(stamp))
        got[i].append(pools[i].advance(stamp))
        for hour in (1, 2):
            got[i].append(pools[i].advance(pools[i].interval_end))

    threads = [threading.Thread(target=rank, args=(i,)) for i in range(len(pools))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(calls) == sorted(files)          # every file read exactly once, for both ranks
    for i, layout in enumerate(layouts):
        assert [f.path for f in got[i]] == [files[p].path for p in files]
        for admitted in got[i]:
            expect = project_lbc_file(files[str(admitted.path)], layout)
            for name, array in expect.fields.items():
                assert admitted.fields[name].tobytes() == array.tobytes()
    assert shared._entries == {}                  # no full file held once every rank has its rows


def test_failed_read_ahead_reads_on_admission(tmp_path):
    """A read-ahead that failed (a live series' next file not written yet) never decides what admission returns."""
    from woof.hex.lbc import LbcPool
    from woof.hex.partition_regional_v841 import RankLbcReader, SharedLbcReads

    files = _series(tmp_path, hours=3)
    second = list(files)[1]
    state = {"fail": True, "calls": 0}

    def read(path, extra_scalars=()):
        state["calls"] += 1
        if str(path) == second and state["fail"]:
            raise OSError("not written yet")
        return files[str(path)]

    inventory = _Inventory(files)
    pool = LbcPool(inventory, scalar_names=("lbc_qv",), compute_tendencies=False)
    shared = SharedLbcReads(read=read)
    pool.reader = RankLbcReader(shared, inventory)
    stamp = datetime(2023, 3, 31, 12)
    pool.start(stamp)
    pool.reader._ahead.exception()                # the read-ahead of the second file failed
    state["fail"] = False
    admitted = pool.advance(stamp)
    assert str(admitted.path) == second and admitted.fields["lbc_theta"].tobytes() == files[second].fields["lbc_theta"].tobytes()
