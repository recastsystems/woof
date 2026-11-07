from types import SimpleNamespace
import sys
import numpy as np
import pytest
from woof.hex.tc_flux import attach_lake_mask


def test_off_has_no_geography_or_cuda_dependency():
    assert attach_lake_mask(SimpleNamespace(), None) is None


@pytest.mark.parametrize('padding', [0, 1])
def test_tc_column_mask_leaves_stock_lake_words_unchanged(monkeypatch, padding):
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(asarray=np.asarray))
    stock = np.zeros((1, 3 + padding), np.float32)
    driver = SimpleNamespace(tc_surface=SimpleNamespace(), fields={'lakemask': stock})
    backend = SimpleNamespace(_seam=SimpleNamespace(_driver=driver))
    mesh = SimpleNamespace(n_cells=3, arrays={'tc_lakemask': np.array([0, 1, 0])})
    receipt = attach_lake_mask(backend, mesh)
    assert driver.fields['lakemask'] is stock
    assert not stock.any()
    np.testing.assert_array_equal(driver.tc_surface.lake_mask_override, [[0, 1, 0] + [1] * padding])
    assert receipt['lake_cells'] == 1
    # A newly moved member owns its own geography, not a remapped old mask.
    mesh.arrays['tc_lakemask'] = np.array([1, 0, 0])
    attach_lake_mask(backend, mesh)
    np.testing.assert_array_equal(driver.tc_surface.lake_mask_override, [[1, 0, 0] + [1] * padding])


def test_partitioned_masks_follow_target_cell_gathers(tmp_path, monkeypatch):
    from test_partition_assets_v841 import regional_case
    from test_partition_local_mesh_v841 import _mesh
    from woof.hex.partition_local_mesh_v841 import build_local_mesh
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(asarray=np.asarray))
    layouts, coe, cov, _, _ = regional_case(tmp_path)
    mesh = _mesh(coe, cov)
    mask = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0], np.int32)
    mesh.arrays['tc_lakemask'] = mask
    mesh.variable_dimensions['tc_lakemask'] = ('nCells',)
    for layout in layouts:
        local = build_local_mesh(mesh, layout)
        stock = np.zeros((1, local.n_cells + 1), np.float32)
        driver = SimpleNamespace(tc_surface=SimpleNamespace(), fields={'lakemask': stock})
        backend = SimpleNamespace(_seam=SimpleNamespace(_driver=driver))
        attach_lake_mask(backend, local)
        np.testing.assert_array_equal(driver.tc_surface.lake_mask_override[0, :-1],
                                      mask[layout.cell_l2g])
        assert driver.tc_surface.lake_mask_override[0, -1] == 1
        assert not stock.any()
