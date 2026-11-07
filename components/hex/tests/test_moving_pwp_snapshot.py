"""The distributed carrier keeps PWP beside exact-restart continuation words."""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys import moving_mpi_v842 as moving
from woof.hex.mpasphys import moving_v842


@pytest.mark.parametrize('parallel', [False, True])
def test_prepare_moving_keeps_ocean_when_physics_restore_runs_beside_remap(monkeypatch, parallel):
    """The resident-move merge must create the ocean proxy before either restore path."""
    from woof.hex.mpasphys.ocean_pwp_v842 import OceanWords
    monkeypatch.setattr(moving, 'NativePoolHydration', lambda *args: None)
    layout = SimpleNamespace(fields=[], configs={'config_start_time': 'start'})
    ocean = dict(identity='pwp-test', levels=2, layer_m=5.0,
                 columns=np.arange(16.).reshape(2, 4, 2), diagnostics=np.arange(8.).reshape(2, 4))
    marker = object()

    def restore(session, *_):
        assert isinstance(session.ocean_pwp, OceanWords)
        session.ocean_pwp.restore_state(ocean)
        return {'ocean_restored': True}

    route = SimpleNamespace(carrier={'physics': {'ocean_pwp': ocean}}, control=None,
        begin=lambda *_: None, move_weights=lambda **kw: marker if parallel else None,
        resume_dycore=lambda **kw: ('state', 'saved', marker), restore_physics=restore)
    inputs = ('mesh', SimpleNamespace(vertical_grid=object()), 'profiles', 'state', 'reference', 'saved', ('qv',))
    _, prepared = moving.prepare_moving(route, inputs, None, layout, None,
                                        SimpleNamespace(grid='grid.nc', init='init.nc'))
    assert prepared['restore_receipt']['ocean_restored']
    np.testing.assert_array_equal(prepared['ocean_pwp']['columns'], ocean['columns'])


def test_rank_snapshot_keeps_ocean_and_continuation_without_halo_values(monkeypatch):
    from woof.hex.mpasphys import checkpoint_v842

    monkeypatch.setattr(moving_v842, 'STATE_FIELDS', ('rho',))
    monkeypatch.setattr(moving_v842, 'SAVED_FIELDS', ('theta_m',))
    monkeypatch.setattr(checkpoint_v842, 'rank_continuation', lambda session: session.continuation)
    parts = [SimpleNamespace(cell_l2g=np.array(ids), n_owned_cells=2,
                             edge_l2g=np.array([], dtype=int), n_owned_edges=0,
                             vertex_l2g=np.array([], dtype=int), n_owned_vertices=0,
                             _global_cells=4, _global_edges=0)
             for ids in ([0, 2, 1], [1, 3, 0])]
    layout = SimpleNamespace(fields=[], dimensions=dict(nCells=4, nEdges=0, nVertices=0))
    expected = np.arange(32, dtype=np.float64).reshape(4, 4, 2)
    expected_diag = np.arange(16, dtype=np.float64).reshape(4, 4)
    pieces = []
    for part in parts:
        ids = part.cell_l2g
        ocean = dict(identity='pwp-test', levels=2, layer_m=5.0,
                     columns=expected[ids].copy(), diagnostics=expected_diag[ids].copy())
        ocean['columns'][2:] = -999
        ocean['diagnostics'][2:] = -999
        sums = np.array([ids[0]+1, ids[1]+1, -999, 0], dtype=np.float32)
        session = SimpleNamespace(precipitation_window=None, surface_service=None,
            ocean_pwp=SimpleNamespace(restart_state=lambda value=ocean: value),
            continuation=dict(scheme_flags={'first': False}, clock_start={'hour': 6},
                              output_accumulators={'rain': sums}))
        field = np.array([ids[0]+10, ids[1]+10, -999, 0], dtype=np.float32)[None, :]
        dycore = SimpleNamespace(cp=SimpleNamespace(asnumpy=np.copy), atmosphere=SimpleNamespace(
            state=SimpleNamespace(rho=field), saved=SimpleNamespace(theta_m=field)))
        pieces.append(moving.rank_words(SimpleNamespace(carried_rows=lambda layout: []),
                                       layout, session, dycore, continuation=True))
    proxy, _, _ = moving.assemble_snapshot(pieces, layout, parts, config=None, manager=None)
    restored = proxy.ocean_pwp.restart_state()
    np.testing.assert_array_equal(restored['columns'], expected)
    np.testing.assert_array_equal(restored['diagnostics'], expected_diag)
    np.testing.assert_array_equal(proxy.continuation['output_accumulators']['rain'], [1, 2, 3, 4, 0])
    assert proxy.continuation['scheme_flags'] == {'first': False}
    assert proxy.continuation['clock_start'] == {'hour': 6}
