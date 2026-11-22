"""Unit tests for Step 1 multi-node features (external/pool launch, cross-host shards, per-rank output/start, 64-bit)."""
from __future__ import annotations

import time
from types import SimpleNamespace

import netCDF4
import numpy as np

from woof.hex import multihost
from woof.hex.output import HistoryStreamOptions
from woof.hex.partition_samehost_transport_v841 import _WORD_COPY_2D
from woof.hex.mpasphys import checkpoint_shards_v842 as shards
from woof.hex.mpasphys import per_rank_output, per_rank_start


def test_external_launch_and_env_join(monkeypatch, tmp_path):
    monkeypatch.setenv(multihost.ENV_EXTERNAL, "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "0")
    monkeypatch.setenv(multihost.ENV_TOKEN, "step1token")
    hosts = multihost.parse_hosts("host0:0,host1:0")
    args = SimpleNamespace(out=tmp_path)
    session = multihost.LaunchSession(hosts, args)
    try:
        assert session.external is True
        assert session.address == "127.0.0.1"
        assert session.rendezvous.token == "step1token"
        session.start()
        assert session.processes == {}
        assert session.receipt()["external"] is True

        monkeypatch.delenv(multihost.ENV_JOIN, raising=False)
        monkeypatch.setenv(multihost.ENV_HOST_INDEX, "1")
        monkeypatch.setenv("MASTER_PORT", str(session.rendezvous.port))
        join = multihost._join_from_env()
        assert join["address"] == "127.0.0.1"
        assert join["port"] == session.rendezvous.port
        assert join["token"] == "step1token"
        assert join["host"] == 1

        joined = multihost.JoinedHost(join, hosts)
        try:
            rec = dict(token="seal42", rank=1, name="rank0001.pkl", bytes=1234, sha256="ab" * 32)
            multihost.send_shard_receipt(joined.connect_spec(), "seal42", 1, rec)
            got = None
            for _ in range(50):
                got = multihost.recv_shard_receipt(session.rendezvous, "seal42", 1)
                if got is not None:
                    break
                time.sleep(0.02)
            assert got == rec
        finally:
            joined.finish()
    finally:
        session.close()


def test_crosshost_shard_writer_seals_without_shared_fs(tmp_path):
    rv = multihost.Rendezvous(host="127.0.0.1", port=0, token="tok99")
    spec = dict(address="127.0.0.1", port=rv.port, token=rv.token, timeout=10.0)
    try:
        host0_target = tmp_path / "host0" / "checkpoint.f003.pkl"
        host1_target = tmp_path / "host1" / "checkpoint.f003.pkl"
        host0_target.parent.mkdir()
        host1_target.parent.mkdir()
        w0 = shards.ShardWriter(0, 2, rendezvous=rv, timeout=10.0)
        w1 = shards.ShardWriter(1, 2, multihost_connect=spec, timeout=10.0)
        token = "crosshost-token-1"
        w1.submit(host1_target, token, {"rank": 1, "vals": np.arange(4, dtype=np.float32)}, step_seconds=0.01)
        w0.submit(host0_target, token, {"rank": 0, "vals": np.arange(4, dtype=np.float32)}, step_seconds=0.01)
        w0.wait()
        w1.wait()
        peers = w0._peers(shards.shard_directory(host0_target), token)
        assert [p["rank"] for p in peers] == [0, 1]
        assert (shards.shard_directory(host1_target) / "rank0001.pkl").is_file()
        assert not (shards.shard_directory(host0_target) / "rank0001.pkl").exists()
    finally:
        rv.close()


def test_per_rank_output_stitch_and_0ulp_compare(tmp_path):
    n_cells, nz = 10, 4
    p0 = SimpleNamespace(
        n_owned_cells=6, n_owned_edges=4, n_owned_vertices=3,
        cell_l2g=np.array([0, 2, 4, 6, 8, 9, 1, 3], dtype=np.int64),
        edge_l2g=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        vertex_l2g=np.array([0, 1, 2, 3], dtype=np.int64),
    )
    p1 = SimpleNamespace(
        n_owned_cells=4, n_owned_edges=3, n_owned_vertices=2,
        cell_l2g=np.array([1, 3, 5, 7, 0, 2], dtype=np.int64),
        edge_l2g=np.array([4, 5, 6, 0], dtype=np.int64),
        vertex_l2g=np.array([3, 4, 0], dtype=np.int64),
    )
    glayout = SimpleNamespace(dimensions=dict(nCells=n_cells, nEdges=7, nVertices=5, nVertLevels=nz))
    m0 = per_rank_output.write_rank_map(tmp_path, 0, p0, glayout, n_ranks=2)
    m1 = per_rank_output.write_rank_map(tmp_path, 1, p1, glayout, n_ranks=2)

    rng_words = np.arange(n_cells * nz, dtype=np.uint32).reshape(1, n_cells, nz) + 0x3F800000
    rng_words[0, 1, 0] = 0x80000000  # -0.0
    rng_words[0, 5, 2] = 0x7FC00123  # specific NaN payload
    gtheta = rng_words.view(np.float32)

    gpath = tmp_path / "history.global.nc"
    with netCDF4.Dataset(gpath, "w", format="NETCDF3_64BIT_DATA") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("nCells", n_cells)
        ds.createDimension("nVertLevels", nz)
        v = ds.createVariable("theta", "f4", ("Time", "nCells", "nVertLevels"))
        v[...] = gtheta

    rpaths = []
    for r, part in enumerate((p0, p1)):
        rp = per_rank_output.rank_target(tmp_path / "history.nc", r)
        rpaths.append(rp)
        with netCDF4.Dataset(rp, "w", format="NETCDF3_64BIT_DATA") as ds:
            ds.createDimension("Time", 1)
            ds.createDimension("nCells", part.n_owned_cells)
            ds.createDimension("nVertLevels", nz)
            v = ds.createVariable("theta", "f4", ("Time", "nCells", "nVertLevels"))
            v[...] = gtheta[:, part.cell_l2g[: part.n_owned_cells], :]

    cmp_res = per_rank_output.compare_sharded_vs_global(rpaths, [m0, m1], gpath)
    assert cmp_res["bit_exact"] is True
    assert cmp_res["differing_words"] == 0
    assert cmp_res["total_words"] == n_cells * nz

    stitched = per_rank_output.stitch_rank_files(rpaths, [m0, m1], tmp_path / "stitched.nc")
    with netCDF4.Dataset(stitched, "r") as sds, netCDF4.Dataset(gpath, "r") as gds:
        assert np.asarray(sds.variables["theta"][...]).tobytes() == np.asarray(gds.variables["theta"][...]).tobytes()


def test_per_rank_start_shards_roundtrip(tmp_path):
    p0 = SimpleNamespace(
        n_owned_cells=3, n_local_cells=4, n_owned_edges=2, n_owned_vertices=2,
        cell_l2g=np.array([0, 1, 2, 3], dtype=np.int64),
        edge_l2g=np.array([0, 1, 2], dtype=np.int64),
        vertex_l2g=np.array([0, 1, 2], dtype=np.int64),
    )
    pool0 = tmp_path / "src_pool0"
    pool0.mkdir()
    (pool0 / "dummy.bin").write_bytes(b"1234")
    mesh0 = SimpleNamespace(partition_source_mesh="HUGE_GLOBAL", n_cells=4)
    layout0 = SimpleNamespace(dimensions=dict(nCells=4), configs=dict(dt=600.0))
    sdir = per_rank_start.save_rank_start_shards(
        tmp_path / "shards",
        [p0],
        [(layout0, pool0, (mesh0, "state0"), 0.1)],
        global_dimensions=dict(nCells=6),
        field_indices={"k": "v"},
    )
    assert mesh0.partition_source_mesh is None
    assert per_rank_start.load_field_indices(sdir) == {"k": "v"}
    loaded_layout, loaded_pool, loaded_inputs, _ = per_rank_start.load_rank_start_shard(sdir, 0, p0)
    assert loaded_layout.samehost_partition is True
    assert (loaded_pool / "dummy.bin").read_bytes() == b"1234"
    assert loaded_inputs[1] == "state0"


def test_64bit_hardening(monkeypatch):
    opts = HistoryStreamOptions(io_type="pnetcdf")
    assert opts.netcdf_format == "NETCDF3_64BIT_OFFSET"
    assert opts.netcdf_format_for_max_var_bytes(1 << 30) == "NETCDF3_64BIT_OFFSET"
    assert opts.netcdf_format_for_max_var_bytes(1 << 31) == "NETCDF3_64BIT_DATA"
    monkeypatch.setenv("WOOF_HEX_NETCDF_64BIT", "1")
    assert opts.netcdf_format == "NETCDF3_64BIT_DATA"

    assert "const long long columns = t[4];" in _WORD_COPY_2D
    assert "const long long row = q / columns;" in _WORD_COPY_2D
    assert "(int)q" not in _WORD_COPY_2D


def test_external_finish_graceful_lifeline(monkeypatch, tmp_path):
    monkeypatch.setenv(multihost.ENV_EXTERNAL, "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "0")
    monkeypatch.setenv(multihost.ENV_TOKEN, "gracetok")
    hosts = multihost.parse_hosts("host0:0,host1:0")
    session = multihost.LaunchSession(hosts, SimpleNamespace(out=tmp_path))
    session.start()
    joined = multihost.JoinedHost(
        dict(address="127.0.0.1", port=session.rendezvous.port, token="gracetok", host=1, timeout=10.0),
        hosts,
    )
    # Wait until lifeline accepted on rank 0
    for _ in range(50):
        if 1 in session._lifelines:
            break
        time.sleep(0.02)
    assert 1 in session._lifelines
    # Rank 0 finishes BEFORE joined.finish() is called: joined must NOT os._exit(3)!
    session.finish()
    time.sleep(0.1)
    joined.finish()


def test_multihost_memory_admission_slices_to_local_host(monkeypatch):
    from types import SimpleNamespace
    from woof.hex.mpasphys.memory_admission import _slice_multihost_admission

    local_cells = [370_000 + i for i in range(16)]
    global_devices = tuple(list(range(8)) * 2)

    # Single-host: completely untouched
    lc0, dv0 = _slice_multihost_admission(SimpleNamespace(hosts=None), local_cells[:8], tuple(range(8)))
    assert lc0 == local_cells[:8]
    assert dv0 == tuple(range(8))

    # Multi-host host 0 (ranks 0..7)
    monkeypatch.delenv("WOOF_HEX_MULTIHOST_JOIN", raising=False)
    monkeypatch.setenv("WOOF_HEX_MULTIHOST_HOST_INDEX", "0")
    lc_h0, dv_h0 = _slice_multihost_admission(
        SimpleNamespace(hosts="boxC:0-7,boxD:0-7"), local_cells, global_devices
    )
    assert lc_h0 == local_cells[:8]
    assert dv_h0 == tuple(range(8))

    # Multi-host host 1 (ranks 8..15)
    monkeypatch.setenv("WOOF_HEX_MULTIHOST_HOST_INDEX", "1")
    lc_h1, dv_h1 = _slice_multihost_admission(
        SimpleNamespace(hosts="boxC:0-7,boxD:0-7"), local_cells, global_devices
    )
    assert lc_h1 == local_cells[8:]
    assert dv_h1 == tuple(range(8))
