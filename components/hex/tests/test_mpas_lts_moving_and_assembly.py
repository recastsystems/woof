"""MPAS-suite moving segments ride the LTS class executor.

THE BREAKAGES THESE PREVENT (MPAS-ROUTE-SPEED-2026-10-24):
- moving segments on the MPAS suite route ran the fixed step (2.4x the WOOF route's LTS moving segments on the
  same storm-following mesh and card); a resumed segment is a carrier restart and must not be refused as a
  native-file restart by the class executor;
"""
from argparse import Namespace
from types import SimpleNamespace

from woof.hex.mpasphys.lts_v842 import admission_refusal, moving_admitted
from woof.hex.mpasphys.moving_v842 import MovingRoute


def _args(**values):
    base = dict(fullstep_local_timestep=True, fullstep_lts_mode="active", fullstep_lts_max_rate=None,
                samehost_devices=None)
    base.update(values)
    return Namespace(**base)


def test_moving_and_checkpoint_routes_ride_the_classes_and_a_da_analysis_does_not() -> None:
    from woof.hex.mpasphys.checkpoint_v842 import CheckpointRoute
    assert moving_admitted(None)
    assert moving_admitted(MovingRoute())
    # --checkpoint-hours / --restart-from seal and restore every class (lts_v842.ClassMoving); they ran the fixed
    # step without a word before.
    assert moving_admitted(CheckpointRoute())
    assert not moving_admitted(CheckpointRoute(da_analysis=True))


def test_a_carrier_resume_is_not_refused_as_a_native_file_restart() -> None:
    layout = SimpleNamespace(dimensions=dict(nCells=10, nCellsSolve=10), configs=dict(config_do_restart=True))
    config = SimpleNamespace(table_key="mpasphys_v842")
    assert "native-file restart" in admission_refusal(_args(), layout, config)
    assert "native-file restart" in admission_refusal(_args(), layout, config, MovingRoute())
    assert admission_refusal(_args(), layout, config, MovingRoute(carrier={"state": {}})) is None


def test_a_spent_resume_carrier_keeps_its_times_and_drops_its_bulk() -> None:
    """A resume from sealed class state drops each carrier's bulk once restored; slimming to the rank route's words
    alone dropped time_seconds and the first LTS resume stopped in its own class-time check (KeyError)."""
    from woof.hex.mpasphys.lts_v842 import spent_carrier
    carrier = dict(schema="s", route="mpas-suite", time_seconds=3600.0, origin_time="o", valid_time="v",
                   state={"rho": [0.0]}, mesh=object(), vertical=object(), scalar_names=["qv"], stochastic=None,
                   chemistry=None,
                   physics=dict(suite="convection_permitting", pools={"a": [0.0]}, fields={"f": 1}, manager={},
                                manager_elapsed="3600", continuation=dict(flags=1, output_accumulators={"x": 1})))
    spent = spent_carrier(carrier)
    assert spent["time_seconds"] == 3600.0 and spent["origin_time"] == "o" and spent["valid_time"] == "v"
    assert not {"state", "mesh", "vertical"} & set(spent) and "pools" not in spent["physics"]
    assert spent["physics"]["suite"] == "convection_permitting" and spent["physics"]["fields"] == {"f": 1}
    assert "output_accumulators" not in spent["physics"]["continuation"]



def test_an_lts_class_edge_with_one_real_cell_takes_the_dummy_column_geometry() -> None:
    """An LTS class solves its outermost halo ring, whose edges name the native dummy as their second cell.

    THE BREAKAGE THIS PREVENTS: the closed-sphere physics geometry indexed that endpoint one column past the
    tendency arrays, so the PBL edge projection read stray device words and some class's first step was refused
    by the dycore validation flag (storm-following moving segments, MPAS-ROUTE-SPEED-2026-10-24).
    """
    import numpy as np
    from woof.hex.mpasphys.resident_v842 import _solved_edges_reference_dummy

    class Native:
        def __init__(self, cells):
            self.cells = np.asarray(cells, dtype=np.int32)

        def get_field(self, pool, name):
            assert (pool, name) == ('mesh', 'cellsOnEdge')
            return self.cells

    closed = Native([[1, 2, 3, 1], [2, 3, 1, 4]])         # 3 cells, 3 edges, then the dummy edge column
    assert not _solved_edges_reference_dummy(closed, 3, 3)
    ring = Native([[1, 2, 3, 1], [2, 4, 1, 4]])           # edge 2's second cell is the dummy (nCells+1)
    assert _solved_edges_reference_dummy(ring, 3, 3)
    cull = Native([[1, 2, 3, 1], [2, 0, 1, 4]])           # a stored 0: no second cell
    assert _solved_edges_reference_dummy(cull, 3, 3)


def test_one_collective_pool_gather_writes_the_per_field_words() -> None:
    """A moving frame's pool words, gathered in one collective and assembled on all cores, equal the old
    field-by-field, level-by-level gather (which cost about 11 s a frame on the storm-following 3 km region)."""
    import threading
    import numpy as np
    from woof.hex.mpasphys.moving_mpi_v842 import gather_pool_words
    from woof.hex.mpasphys.partition_v842 import gather_owned_field
    from woof.hex.mpasphys.samehost_v842 import ThreadComm

    ids = [np.array([0, 3, 6, 1]), np.array([1, 4, 7, 2, 0]), np.array([2, 5, 8, 4])]
    parts = [SimpleNamespace(cell_l2g=i, n_owned_cells=3, edge_l2g=np.arange(1), n_owned_edges=1,
                             vertex_l2g=np.arange(1), n_owned_vertices=1) for i in ids]
    rows = [dict(pool='diag', name='theta', pool_name='theta', kind='real', nlevels=1,
                 dimensions=['nVertLevels', 'nCells'], shape=[4, 10]),
            dict(pool='diag_physics', name='rainnc', pool_name='rainnc', kind='real', nlevels=2,
                 dimensions=['nCells'], shape=[10]),
            dict(pool='sfc_input', name='ivgtyp', pool_name='ivgtyp', kind='integer', nlevels=1,
                 dimensions=['nCells'], shape=[10])]
    layout = SimpleNamespace(dimensions=dict(nCells=9, nEdges=1, nVertices=1, nVertLevels=4), fields=rows)
    rng = np.random.default_rng(3)
    local = {}
    for rank, part in enumerate(parts):
        n = len(part.cell_l2g) + 1
        for row in rows:
            for level in range(1, row['nlevels'] + 1):
                shape = (4, n) if row['name'] == 'theta' else (n,)
                value = (rng.integers(0, 20, shape).astype('<i4') if row['kind'] == 'integer'
                         else rng.standard_normal(shape).astype('<f4'))
                local[rank, row['name'], level] = np.asfortranarray(value)
    wanted = {(row['pool'], row['name']) for row in rows}
    comms = ThreadComm.world(len(parts))
    results = [None] * len(parts)

    def rank_thread(rank):
        results[rank] = gather_pool_words(comms[rank], layout, parts, wanted,
                                          lambda row, level: local[rank, row['name'], level])

    threads = [threading.Thread(target=rank_thread, args=(rank,)) for rank in range(len(parts))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert results[1] is None and results[2] is None
    for row in rows:
        for level in range(1, row['nlevels'] + 1):
            serial = gather_owned_field(row, layout, parts, [local[r, row['name'], level] for r in range(3)])
            serial = np.concatenate((serial, np.zeros((*serial.shape[:-1], 1), serial.dtype)), axis=-1)
            assembled = results[0].get_field(row['pool'], row['pool_name'], level=level)
            assert assembled.dtype == serial.dtype and assembled.tobytes(order='F') == serial.tobytes(order='F')
