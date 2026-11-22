"""The hrrr-gfs blend row, and a blend over a limited-area parent (NOAA's conus3km): its own rings stay GFS's."""
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset
from woof.hex import blend_start as blend, init_sources


def test_hrrr_gfs_row_is_a_two_source_blend():
    table = init_sources.load_table()
    row = init_sources.resolve_source(table, 'hrrr-gfs')
    block = blend.blend_block(row)
    assert (block['inside'], block['outside']) == ('hrrr', 'gfs-rrfs')
    assert block['taper_km'] > 0 and 0 < block['lid_taper_m'][0] < block['lid_taper_m'][1]
    outside, inside = blend.blend_rows(table, row)
    assert outside['id'] == 'gfs-rrfs' and inside['id'] == 'hrrr'
    assert inside['intermediate']['row'] == 'hrrr'
    assert blend.effective_row(table, row)['id'] == 'gfs-rrfs'
    assert init_sources.resolve_source(table, 'hrrr-blend')['id'] == 'hrrr-gfs'
    # The zoom door's automatic choice still meets the RRFS row first.
    blends = [r['id'] for r in table['sources'] if r.get('blend')]
    assert blends.index('rrfs-gfs') < blends.index('hrrr-gfs')


#: The research kit's Vtable.GFS (kit 1960569 small.tar.zst), which the conus3km deck decodes its start and its 61
#: boundary files with (box-c60.sh fetch60/init60); the deck's start MET, sha256 078d0a78..., is decoded with it.
DECK_VTABLE_SHA256 = 'f414c8847cca5898cb4e245d0af43382ab1d822de48145d613bf3910d4997aba'


def test_blend_outside_half_is_the_decks_58_level_gfs_start():
    """THE BREAKAGE (hrrr-gfs on conus3km, 2026-11-12): the outside half was the gfs row (pgrb2 alone, 34 levels,
    PRMSL), so its rings sat up to 3.6 K from the 58-level pgrb2 + pgrb2b boundary files at t=0."""
    import hashlib

    table = init_sources.load_table()
    outside, _ = blend.blend_rows(table, init_sources.resolve_source(table, 'hrrr-gfs'))
    assert init_sources.resolve_source(table, 'gfs-pgrb2b')['id'] == outside['id'] == 'gfs-rrfs'
    plans = init_sources.plan_products(outside, init_sources.Cycle.parse('2026-11-12T00'), None)
    primary = [plan for plan in plans if plan.role == 'primary']
    # Both halves whole (the deck's own bytes), in the deck's order: pgrb2 then pgrb2b.
    assert [plan.filename for plan in primary] == ['gfs.t00z.pgrb2.0p25.f000', 'gfs.t00z.pgrb2b.0p25.f000']
    assert all(plan.index_url is None for plan in primary)
    assert outside['first_guess'] == {'nfglevels': 58, 'nfgsoillevels': 4, 'use_spechumd': 'no'}
    vtable = init_sources.resolve_vtable(outside)
    assert hashlib.sha256(vtable.read_bytes()).hexdigest() == DECK_VTABLE_SHA256
    assert outside['decode']['map_source'] == 'ncep-gfs'
    # The plain gfs row keeps its own start; only the blend's outside half moved.
    assert init_sources.resolve_source(table, 'gfs')['first_guess']['nfglevels'] == 34


def _lattice(path: Path, n: int, rings: int) -> np.ndarray:
    """An n x n lattice of cells 0.1 degree apart (4 neighbours), with ``rings`` boundary rings if > 0."""
    lat, lon = np.meshgrid(30 + 0.1 * np.arange(n), -100 + 0.1 * np.arange(n), indexing='ij')
    index = np.arange(n * n).reshape(n, n)
    neighbours = np.zeros((n * n, 4), dtype=np.int32)
    for k, (di, dj) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        ok = (i + di >= 0) & (i + di < n) & (j + dj >= 0) & (j + dj < n)
        neighbours[:, k] = np.where(ok, index[np.clip(i + di, 0, n - 1), np.clip(j + dj, 0, n - 1)] + 1, 0).ravel()
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    edge = np.minimum(np.minimum(i, j), np.minimum(n - 1 - i, n - 1 - j))
    mask = np.where(edge < rings, edge + 1, 0).ravel() if rings else np.zeros(n * n, dtype=np.int32)
    with Dataset(path, 'w') as f:
        f.createDimension('nCells', n * n)
        f.createDimension('maxEdges', 4)
        f.createVariable('latCell', 'f8', ('nCells',))[:] = np.radians(lat.ravel())
        f.createVariable('lonCell', 'f8', ('nCells',))[:] = np.radians(lon.ravel())
        f.createVariable('cellsOnCell', 'i4', ('nCells', 'maxEdges'))[:] = neighbours
        f.createVariable('nEdgesOnCell', 'i4', ('nCells',))[:] = (neighbours > 0).sum(axis=1)
        if rings:
            f.createVariable('bdyMaskCell', 'i4', ('nCells',))[:] = mask
    return mask


def test_cut_keeps_off_a_limited_area_parents_rings(tmp_path):
    n, rings = 60, 7
    mask = _lattice(tmp_path / 'regional.grid.nc', n, rings)
    _lattice(tmp_path / 'global-like.grid.nc', n, 0)
    # The inside source's box holds the whole limited-area mesh, as HRRR's box holds parts of conus3km's edge.
    box = {'south': 29.0, 'north': 40.0, 'west': -101.0, 'east': -90.0}
    regional = blend.plan_cut(tmp_path / 'regional.grid.nc', box)['region_box']
    unbounded = blend.plan_cut(tmp_path / 'global-like.grid.nc', box)['region_box']
    # Without rings nothing lies outside the box, so the cut (and the seven rings a cull adds) would run off
    # the parent's edge; with them the cut keeps CULL_RINGS + 1 cells clear of the parent's rings.
    assert unbounded == {k: round(v, 6) for k, v in box.items()}
    with Dataset(tmp_path / 'regional.grid.nc') as f:
        lat, lon = np.degrees(f['latCell'][:]), np.degrees(f['lonCell'][:])
    kept = blend._in_box(lat, lon, regional)
    assert kept.any() and not (kept & (mask > 0)).any()
    # Every kept cell is at least CULL_RINGS + 1 lattice steps from the parent's innermost ring.
    margin = (rings + blend.CULL_RINGS) * 0.1 - 1e-6
    assert lat[kept].min() >= 30 + margin and lat[kept].max() <= 30 + 0.1 * (n - 1) - margin
    assert lon[kept].min() >= -100 + margin and lon[kept].max() <= -100 + 0.1 * (n - 1) - margin


def test_blend_weights_leave_parent_rings_to_the_outside_source(tmp_path, monkeypatch):
    # Cell 2 lies inside the cut's interior but is a ring of the limited-area parent: it keeps the outside value.
    parent = {'Cell': np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]]),
              'Edge': np.array([[.5, .5, 0]]), 'Vertex': np.array([[.5, .5, .5]]),
              'bdyMaskCell': np.array([0, 0, 1]), 'cellsOnEdge': np.array([[1, 2]]),
              'cellsOnVertex': np.array([[1, 2, 3]])}
    cut = {**parent, 'bdyMaskCell': np.array([0, 0, 0])}
    monkeypatch.setattr(blend, '_grid_geometry', lambda p: parent if p.name == 'grid' else cut)
    seen = {}

    def weights(xyz, interior, taper_km):
        seen['interior'] = interior.copy()
        return np.where(interior, 1.0, 0.0)

    monkeypatch.setattr(blend, 'cell_weights', weights)
    for name in ('outside', 'inside'):
        with Dataset(tmp_path / (name + '.nc'), 'w') as f:
            for dim, count in [('nCells', 3), ('nEdges', 1), ('nVertices', 1), ('nVertLevels', 1),
                               ('nVertLevelsP1', 2)]:
                f.createDimension(dim, count)
            f.createVariable('zgrid', 'f8', ('nCells', 'nVertLevelsP1'))[:] = [[0, 10], [0, 10], [0, 10]]
            f.createVariable('theta', 'f8', ('nCells', 'nVertLevels'))[:] = [[300.5], [301.5], [302.5]]
            if name == 'inside':
                f['theta'][:] += 2
    result = blend.blend_inits(tmp_path / 'outside.nc', tmp_path / 'inside.nc', tmp_path / 'init.nc',
                               taper_km=100, parent_grid=Path('grid'), cut_grid=Path('cut'))
    assert seen['interior'].tolist() == [True, True, False]
    assert result['cells']['parent_rings'] == 1
    with Dataset(tmp_path / 'init.nc') as f:
        np.testing.assert_allclose(f['theta'][:, 0], [302.5, 303.5, 302.5])


def test_native_inside_source_gets_the_two_hours_its_decoder_needs():
    from woof.hex.hrrr_intermediate import parse_hours
    # The native HRRR decoder series refuses one hour; the packaged RRFS road keeps its single hour.
    assert parse_hours(blend.inside_hours('hrrr', 0)) == (0, 1)
    assert parse_hours(blend.inside_hours('hrrr', 6)) == (6, 7)
    assert blend.inside_hours('rrfs', 0) == '0'


@pytest.mark.parametrize('spec', [None, Path('v.json')])
def test_init_argv_names_a_vertical_spec_only_when_given(spec):
    table = init_sources.load_table()
    row = init_sources.resolve_source(table, 'gfs')
    argv = init_sources.init_argv(table, row, met=Path('m'), grid=Path('g'), static=Path('s'),
                                  vertical_spec=spec, out=Path('o'),
                                  cycle=init_sources.Cycle.parse('2026-11-12T00'))
    assert ('--vertical-spec' in argv) == (spec is not None)
    if spec is not None:
        assert argv[argv.index('--vertical-spec') + 1] == 'v.json'
