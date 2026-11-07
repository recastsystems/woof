"""A blend refuses a cut whose producer changed the shared coordinate surfaces."""
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset
from woof.hex import blend_start as blend


@pytest.mark.parametrize('changed', [False, True])
def test_cut_must_keep_whole_mesh_zgrid(tmp_path, monkeypatch, changed):
    geometry = {'Cell': np.array([[1., 0, 0], [0, 1., 0]]),
                'Edge': np.array([[.5, .5, 0]]), 'Vertex': np.array([[.5, .5, .5]]),
                'bdyMaskCell': np.array([0, 0]), 'cellsOnEdge': np.array([[1, 2]]),
                'cellsOnVertex': np.array([[1, 2, 0]])}
    monkeypatch.setattr(blend, '_grid_geometry', lambda p: geometry)
    monkeypatch.setattr(blend, 'cell_weights', lambda *a: np.array([.25, .75]))
    for name in ('outside', 'inside'):
        with Dataset(tmp_path / (name + '.nc'), 'w') as f:
            for dim, count in [('nCells', 2), ('nEdges', 1), ('nVertices', 1),
                               ('nVertLevels', 2), ('nVertLevelsP1', 3)]:
                f.createDimension(dim, count)
            f.createVariable('zgrid', 'f8', ('nCells', 'nVertLevelsP1'))[:] = [[0, 10, 30], [1, 11, 31]]
            f.createVariable('theta', 'f8', ('nCells', 'nVertLevels'))[:] = [[300.1, 310.2], [301.2, 311.3]]
            if name == 'inside':
                f['theta'][:] += 4
                if changed:
                    f['zgrid'][0, 1] += .01
    args = [tmp_path / (n + '.nc') for n in ('outside', 'inside', 'result')]
    kwargs = dict(taper_km=400, parent_grid=Path('grid'), cut_grid=Path('cut'))
    if changed:
        with pytest.raises(blend.BlendRefusal, match='changed.*vertical grid'):
            blend.blend_inits(*args, **kwargs)
        assert not args[-1].exists()
    else:
        blend.blend_inits(*args, **kwargs)
        with Dataset(args[-1]) as f:
            np.testing.assert_array_equal(f['zgrid'][:], [[0, 10, 30], [1, 11, 31]])
            np.testing.assert_allclose(f['theta'][:], [[301.1, 311.2], [304.2, 314.3]])
