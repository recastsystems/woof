"""The GFS preparation's bulk host passes run on host threads with byte-identical results.

Breakage it prevents: ``woof.gfs_direct`` at PREPARING_INITIALIZE_ALL_TIMES held one core at
98-101 % for 70-190 s per 3 km nest while 200+ cores sat idle (ml-probs T1 on V8X, 2026-10-27).
Profiled: the decoded-lead reads, the snapshot's defensive field copy, the decoded-array and GRIB
hashes and the CPU mixed-phase RH conversion were serial passes (woof/ingest/host_parallel.py).
Each test holds the threaded result to the serial one bit for bit.
"""

from datetime import datetime

import numpy as np
import pytest

from woof.ingest import grib, host_parallel


def _fields(rng, nlev=6, ny=40, nx=50):
    return {
        "T": rng.normal(270.0, 20.0, (nlev, ny, nx)),
        "RH": rng.uniform(0.0, 100.0, (nlev, ny, nx)),
        "PSFC": rng.normal(1.0e5, 900.0, (ny, nx)),
        "LANDSEA": (rng.uniform(size=(ny, nx)) > 0.5).astype(np.float64),
    }


def _snapshot(fields, nlev=6, ny=40, nx=50):
    return grib.Era5Snapshot(
        valid_time=datetime(2024, 5, 14),
        levels_hpa=np.linspace(1000.0, 100.0, nlev),
        latitude=np.linspace(20.0, 50.0, ny),
        longitude=np.linspace(-120.0, -70.0, nx),
        fields=fields)


def test_map_ordered_keeps_order_and_raises_the_first_error():
    assert host_parallel.map_ordered(lambda x: x * x, range(50), workers=8) == [
        x * x for x in range(50)]

    def boom(x):
        if x in (3, 7):
            raise ValueError(f"item {x}")
        return x
    with pytest.raises(ValueError, match="item 3"):
        host_parallel.map_ordered(boom, range(10), workers=4)


def test_a_call_from_a_host_thread_runs_inline_instead_of_waiting_on_its_own_pool():
    inner = lambda x: host_parallel.map_ordered(lambda y: x * 100 + y, range(4), workers=4)  # noqa: E731
    assert host_parallel.map_ordered(inner, range(6), workers=3) == [
        [x * 100 + y for y in range(4)] for x in range(6)]


@pytest.mark.parametrize("threshold", [0, 1 << 60])
def test_snapshot_copy_is_identical_threaded_or_not(monkeypatch, threshold):
    monkeypatch.setattr(grib, "_PARALLEL_FIELD_COPY_BYTES", threshold)
    raw = _fields(np.random.default_rng(7))
    snapshot = _snapshot(raw)
    assert list(snapshot.fields) == list(raw)
    for name, value in raw.items():
        held = snapshot.fields[name]
        assert held.tobytes() == value.tobytes()
        assert not held.flags.writeable
        assert not np.shares_memory(held, value)


def test_the_threaded_copy_refuses_the_first_bad_field_in_order(monkeypatch):
    monkeypatch.setattr(grib, "_PARALLEL_FIELD_COPY_BYTES", 0)
    raw = _fields(np.random.default_rng(1))
    raw["RH"] = raw["RH"].astype(np.float32)
    raw["PSFC"] = raw["PSFC"][:, :-1]
    with pytest.raises(TypeError, match="field RH must have dtype float64"):
        _snapshot(raw)


def test_a_streamed_snapshot_still_copies_one_field_at_a_time():
    raw = _fields(np.random.default_rng(3))
    streamed = grib.Era5Snapshot.from_field_items(
        valid_time=datetime(2024, 5, 14), levels_hpa=np.linspace(1000.0, 100.0, 6),
        latitude=np.linspace(20.0, 50.0, 40), longitude=np.linspace(-120.0, -70.0, 50),
        field_items=iter(raw.items()))
    for name, value in raw.items():
        assert streamed.fields[name].tobytes() == value.tobytes()


def test_rh_to_water_threaded_is_bitwise_the_one_pass(monkeypatch):
    from woof.ingest import preprocess_backend as pb

    rng = np.random.default_rng(11)
    t = rng.uniform(200.0, 310.0, (5, 30, 40)).astype(np.float32)
    rh = rng.uniform(0.0, 110.0, (5, 30, 40)).astype(np.float32)
    one_pass = pb._rh_to_water_block(
        np.asarray(rh, dtype=np.float64), np.asarray(t, dtype=np.float64), workers=None)
    monkeypatch.setattr(pb, "_RH_SLAB_MIN_VALUES", 256)
    threaded = pb._era5_rh_to_water_cpu(rh, t)
    assert threaded.dtype == np.float32 and threaded.shape == rh.shape
    assert threaded.tobytes() == one_pass.tobytes()


def test_bridge_snapshot_reads_every_field_as_the_serial_loop_did(tmp_path):
    from woof import gfs_direct

    rng = np.random.default_rng(5)
    nlev, ny, nx = 4, 9, 11
    root = tmp_path / "f003"
    root.mkdir()
    expected = {}
    for name in gfs_direct._THREE_D:
        values = rng.normal(size=(nlev, ny, nx)).astype("<f4")
        values.tofile(root / f"{name}.f32le")
        expected[name] = values.astype(np.float64)
    for name in gfs_direct._TWO_D:
        values = rng.normal(size=(ny, nx)).astype("<f4")
        values.tofile(root / f"{name}.f32le")
        expected[name] = values.astype(np.float64)
    snapshot = gfs_direct._bridge_snapshot(
        tmp_path, datetime(2024, 5, 14), 3, np.linspace(1000.0, 500.0, nlev),
        np.linspace(20.0, 30.0, ny), np.linspace(-100.0, -90.0, nx), ny, nx)
    assert list(snapshot.fields) == [*gfs_direct._THREE_D, *gfs_direct._TWO_D]
    for name, value in expected.items():
        assert snapshot.fields[name].tobytes() == value.tobytes()


def test_a_column_helper_block_runs_on_the_shared_threads_and_waits_for_its_tasks():
    """real._column_pool no longer starts and joins a pool per helper call (thread churn)."""
    import threading
    import time

    from woof.ingest import real

    seen, done = set(), []

    def task(i):
        time.sleep(0.01)
        seen.add(threading.current_thread().name)
        done.append(i)
        return i * i

    with real._column_pool(4) as executor:
        futures = [executor.submit(task, i) for i in range(8)]
        mapped = list(executor.map(task, range(8, 12)))
    assert len(done) == 12  # leaving the block waited for everything submitted
    assert [f.result() for f in futures] == [i * i for i in range(8)]
    assert mapped == [i * i for i in range(8, 12)]
    assert all(name.startswith("gpuwm-host") for name in seen)
    with real._column_pool(4) as executor:
        executor.submit(task, 0).result()
    assert all(name.startswith("gpuwm-host") for name in seen)



def test_a_running_block_survives_a_larger_pool_being_made():
    """OK-WOFS 2026-11-02 20:40Z: a shared_executor block held the shared pool while another caller asked for more
    threads; the old pool was shut and the block's next submit raised 'cannot schedule new futures after shutdown'."""
    from woof.ingest import host_parallel as H
    with H.shared_executor(2) as ex:
        first = ex.submit(lambda: 1)
        H._shared_pool(H._pool_size + 8)          # another caller grows the pool mid-block
        second = ex.submit(lambda: 2)             # the block's pool must still take work
    assert (first.result(), second.result()) == (1, 2)
