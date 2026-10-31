"""A rank process reads only its own rows of each boundary file, word for word what the whole-file cut gives.

THE COST (NOAA RRFS v2 conus3km, one process per card, V8X6000, 2026-10-29): every rank process read each 13.3 GB
boundary file whole to keep its own rows; rank processes spent more CPU time in the kernel than in user code.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import classic_netcdf
from woof.hex.lbc import LBC_OPTIONAL_SCALAR_VARIABLES, LBC_REQUIRED_VARIABLES, LbcPool, read_lbc_file
from woof.hex.partition_regional_v841 import (RankLbcReader, SharedLbcReads, layout_rows, project_lbc_file,
                                             read_lbc_rows)

netCDF4 = pytest.importorskip("netCDF4")

N_CELLS, N_EDGES, NZ = 37, 101, 5


def write_lbc(path: Path, xtime: str, seed: int, *, extra=("lbc_qc", "lbc_qr", "lbc_nr")) -> Path:
    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_DATA") as d:
        d.createDimension("nVertLevels", NZ)
        d.createDimension("nCells", N_CELLS)
        d.createDimension("Time", None)
        d.createDimension("nEdges", N_EDGES)
        d.createDimension("nVertLevelsP1", NZ + 1)
        d.createDimension("StrLen", 64)
        schema = {**LBC_REQUIRED_VARIABLES, **{n: LBC_OPTIONAL_SCALAR_VARIABLES[n] for n in extra}}
        for name, dims in schema.items():
            v = d.createVariable(name, "f4", dims)
            shape = tuple(d.dimensions[n].size for n in dims[1:])
            values = rng.standard_normal(shape).astype(np.float32)
            values.reshape(-1)[0] = np.float32(-0.0)
            values.reshape(-1)[-1] = np.float32(1.0e-40)   # a float32 subnormal
            v[0] = values
        xt = d.createVariable("xtime", "S1", ("Time", "StrLen"))
        xt[0] = np.frombuffer(xtime.ljust(64).encode("ascii"), dtype="S1")
    return path


def layout(cells, edges):
    return SimpleNamespace(cell_l2g=np.asarray(cells), edge_l2g=np.asarray(edges),
                           _global_cells=N_CELLS, _global_edges=N_EDGES)


LAYOUTS = [
    layout([4, 0, 36, 17, 18, 19, 2], [100, 0, 50, 51, 7, 3]),          # owned first, halo after: not sorted
    layout(list(range(N_CELLS)), list(range(N_EDGES))[::-1]),           # everything, edges reversed
    layout([], []),                                                     # a rank with no rows of this kind
]


@pytest.mark.parametrize("cut", LAYOUTS)
def test_rank_rows_are_the_whole_file_cut(tmp_path, cut):
    path = write_lbc(tmp_path / "lbc.nc", "2023-03-31_12:00:00", 3)
    whole = project_lbc_file(read_lbc_file(path), cut)
    rows = read_lbc_rows(path, (), cut)
    assert rows.valid_time == whole.valid_time and str(rows.path) == str(whole.path)
    assert sorted(rows.fields) == sorted(whole.fields)
    for name, array in whole.fields.items():
        got = rows.fields[name]
        assert got.dtype == array.dtype == np.float32
        assert got.shape == array.shape
        assert got.tobytes() == array.tobytes(), name
        assert not got.flags.writeable


def test_variables_the_classic_reader_does_not_take_are_cut_after_decoding(tmp_path, monkeypatch):
    """WOOF_HEX_CLASSIC_READ=off sends every variable through the bridge; the rows are still the cut words."""
    path = write_lbc(tmp_path / "lbc.nc", "2023-03-31_12:00:00", 5)
    cut = LAYOUTS[0]
    whole = project_lbc_file(read_lbc_file(path), cut)
    monkeypatch.setenv("WOOF_HEX_CLASSIC_READ", "off")
    rows = read_lbc_rows(path, (), cut)
    for name, array in whole.fields.items():
        assert rows.fields[name].tobytes() == array.tobytes(), name


def test_a_layout_for_another_mesh_is_refused_before_reading(tmp_path):
    path = write_lbc(tmp_path / "lbc.nc", "2023-03-31_12:00:00", 7)
    wrong = SimpleNamespace(cell_l2g=np.array([0]), edge_l2g=np.array([0]), _global_cells=N_CELLS + 1,
                            _global_edges=N_EDGES)
    with pytest.raises(ValueError, match="entities, expected"):
        read_lbc_rows(path, (), wrong)


def test_read_rows_matches_read_variables_on_many_threads(tmp_path):
    path = write_lbc(tmp_path / "lbc.nc", "2023-03-31_12:00:00", 9)
    table, numrecs = classic_netcdf.extents(path)
    names = list(LBC_REQUIRED_VARIABLES)
    whole = classic_netcdf.read_variables(path, names, table, workers=4)
    rng = np.random.default_rng(1)
    rows = {name: rng.integers(0, table[name].shape[1], size=500) for name in names}
    old = classic_netcdf._ROW_BLOCK
    classic_netcdf._ROW_BLOCK = 7   # many blocks on the pool
    try:
        got = classic_netcdf.read_rows(path, names, table, rows, workers=4)
    finally:
        classic_netcdf._ROW_BLOCK = old
    for name in names:
        assert got[name].dtype == whole[name].dtype
        assert got[name].tobytes() == np.ascontiguousarray(whole[name][0][rows[name]]).tobytes()
    with pytest.raises(IndexError):
        classic_netcdf.read_rows(path, ["lbc_qv"], table, {"lbc_qv": np.array([N_CELLS])}, workers=1)


class _Inventory:
    def __init__(self, files):
        self.files = files
        self.identity = ("test", id(self))

    def latest_before(self, when):
        return [p for p, t in self.files if t <= when][-1]

    def earliest_strictly_after(self, when):
        later = [p for p, t in self.files if t > when]
        if not later:
            from woof.hex.lbc import LbcAdmissionError
            raise LbcAdmissionError("no later file")
        return later[0]


def test_one_rank_process_reads_rows_and_threads_still_share_one_read(tmp_path):
    """A process's only consumer of a series takes the rows path; two consumers (rank threads) keep the shared read."""
    paths = [write_lbc(tmp_path / f"lbc.{h}.nc", f"2023-03-31_{12 + h:02d}:00:00", 20 + h) for h in range(3)]
    files = [(p, datetime(2023, 3, 31, 12 + h)) for h, p in enumerate(paths)]
    whole_reads, row_reads = [], []
    lock = threading.Lock()

    def read(path, extra_scalars=()):
        with lock:
            whole_reads.append(str(path))
        return read_lbc_file(path, extra_scalars=extra_scalars)

    def rows(path, extra_scalars, cut):
        with lock:
            row_reads.append(str(path))
        return read_lbc_rows(path, extra_scalars, cut)

    # One consumer in this "process": rows only, the whole file never read.
    inventory = _Inventory(files)
    shared = SharedLbcReads(read=read, read_rows=rows)
    pool = LbcPool(inventory, scalar_names=("lbc_qv",), compute_tendencies=False)
    pool.reader = RankLbcReader(shared, inventory, LAYOUTS[0])
    stamp = datetime(2023, 3, 31, 12)
    got = [pool.start(stamp), pool.advance(stamp), pool.advance(pool.interval_end)]
    assert whole_reads == []
    assert sorted(set(row_reads)) == sorted(str(p) for p in paths)
    for admitted in got:
        expect = project_lbc_file(read_lbc_file(admitted.path), LAYOUTS[0])
        for name, array in expect.fields.items():
            assert admitted.fields[name].tobytes() == array.tobytes()

    # Two consumers (rank threads of one process): one whole read per file, as before.
    whole_reads.clear(); row_reads.clear()
    inventory = _Inventory(files)
    shared = SharedLbcReads(read=read, read_rows=rows)
    pools = []
    for cut in LAYOUTS[:2]:
        p = LbcPool(inventory, scalar_names=("lbc_qv",), compute_tendencies=False)
        p.reader = RankLbcReader(shared, inventory, cut)
        pools.append(p)
    for p in pools:
        p.start(stamp)
    assert row_reads == []
    assert whole_reads.count(str(paths[0])) == 1


def test_layout_rows_names_edges_for_lbc_u_only():
    rows = layout_rows(LAYOUTS[0])
    edges, n_edges = rows("lbc_u")
    cells, n_cells = rows("lbc_theta")
    assert n_edges == N_EDGES and n_cells == N_CELLS
    assert edges.dtype == cells.dtype == np.int64
    assert list(edges) == list(LAYOUTS[0].edge_l2g) and list(cells) == list(LAYOUTS[0].cell_l2g)
