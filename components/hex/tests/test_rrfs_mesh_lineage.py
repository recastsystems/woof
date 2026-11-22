"""Native fork lineage admission retains grid identity and dimension guards."""
import pytest
import woof.hex.mesh as module


@pytest.mark.parametrize('grid_id,selected,accepted', [
    ('ancestor', True, True), ('immediate', True, True),
    ('ancestor', False, False), ('other', True, False),
    ('anc', True, False),
])
def test_native_parent_chain_requires_an_exact_ancestor(tmp_path, monkeypatch,
                                                       grid_id, selected, accepted):
    grid = tmp_path / 'grid.nc'
    static = tmp_path / 'init.nc'
    grid.touch()
    static.touch()
    def read(path):
        attrs = ({'file_id': grid_id} if path == grid else
                 {'parent_id': 'immediate\nancestor'})
        return {}, attrs, {}, {}, {}
    monkeypatch.setattr(module, '_read_netcdf', read)
    if accepted:
        mesh = module.Mesh.from_netcdf(grid, static, validate=False,
                                      native_parent_chain=selected)
        assert mesh.provenance['static_attrs']['parent_id'] == 'immediate\nancestor'
    else:
        with pytest.raises(module.MeshValidationError, match='parent_id'):
            module.Mesh.from_netcdf(grid, static, validate=False,
                                   native_parent_chain=selected)


def test_native_parent_chain_keeps_dimension_guard(tmp_path, monkeypatch):
    grid = tmp_path / 'grid.nc'
    static = tmp_path / 'init.nc'
    grid.touch()
    static.touch()
    def read(path):
        if path == grid:
            return {'nCells': 2}, {'file_id': 'ancestor'}, {}, {}, {}
        return {'nCells': 3}, {'parent_id': 'immediate\nancestor'}, {}, {}, {}
    monkeypatch.setattr(module, '_read_netcdf', read)
    with pytest.raises(module.MeshValidationError, match='dimensions disagree'):
        module.Mesh.from_netcdf(grid, static, validate=False, native_parent_chain=True)


def _static(path, file_id, parent_id):
    from netCDF4 import Dataset
    with Dataset(path, 'w') as d:
        d.file_id = file_id
        d.parent_id = parent_id
    return path


@pytest.mark.parametrize('static_id,static_parent,accepted', [
    ('stat', 'grid\n', True),          # init -> static -> grid: the gpuwm 3.0 init's one-parent lineage
    ('stat', 'other\n', False),        # the static is not of this grid
    ('nope', 'grid\n', False),         # the init was not built on this static
])
def test_one_parent_init_descends_through_the_run_static(tmp_path, monkeypatch,
                                                         static_id, static_parent, accepted):
    # rehearsal HX7 (2026-11-16): the manual's quickstart was refused on the published x1.40962 pair.
    grid = tmp_path / 'grid.nc'
    init = tmp_path / 'init.nc'
    grid.touch()
    init.touch()
    static = _static(tmp_path / 'static.nc', static_id, static_parent)
    def read(path):
        attrs = {'file_id': 'grid'} if path == grid else {'parent_id': 'stat\n'}
        return {}, attrs, {}, {}, {}
    monkeypatch.setattr(module, '_read_netcdf', read)
    if accepted:
        module.Mesh.from_netcdf(grid, init, validate=False, native_parent_chain=True, lineage_static=static)
    else:
        with pytest.raises(module.MeshValidationError, match='parent_id'):
            module.Mesh.from_netcdf(grid, init, validate=False, native_parent_chain=True,
                                   lineage_static=static)
    with pytest.raises(module.MeshValidationError, match='parent_id'):
        module.Mesh.from_netcdf(grid, init, validate=False, native_parent_chain=True)
