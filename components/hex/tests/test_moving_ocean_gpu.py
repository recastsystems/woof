"""Cold-wake and inventory regression for the actual CUDA overlap operator."""
from copy import deepcopy
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from woof.hex.moving_ocean import remap_surface, restore_surface_backend


def test_prescribed_ocean_never_imports_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    payload = {'backends': [{'surface_updates': None}]}
    fresh = {'surface_updates': {'ocean': None}}
    before = deepcopy(fresh)
    assert remap_surface(payload, fresh, None, None, None, None, {}) is None
    assert fresh == before
    assert restore_surface_backend(None, payload, None, None, None, {}) is None


def test_cold_wake_heat_salt_and_halos_survive_mesh_change():
    cp = pytest.importorskip('cupy')
    from scipy.sparse import csr_matrix
    # Two ownership classes; poisoned halo values must never contribute.
    arrays = dict(ocean_mask=np.ones(2, np.float32), ocean_area=np.zeros(2, np.float32),
                  ocean_temperature=np.array([294., 300.], np.float32),
                  ocean_depth=np.full(2, 5., np.float32),
                  sst=np.array([294., 300.], np.float32),
                  landmask=np.zeros(2, np.float32),
                  ocean_pwp_columns=np.array([[[294.,290.],[34.,35.],[1.,1.],[0.,0.]],
                                              [[300.,296.],[36.,37.],[3.,3.],[0.,0.]]]),
                  ocean_pwp_diagnostics=np.array([[5.,0.,0.,1.],[5.,0.,2.,1.]]))
    p = dict(density=1025., heat_capacity=4000., layer_m=5.,
             thermal_expansion=2e-4, haline_contraction=8e-4, density_jump=.01)
    state = dict(arrays=arrays, ocean=dict(options=dict(pwp=p), steps=48),
                 identity=dict(cells=2, start='2021-01-01', times=[]), alarms={},
                 february=28, counts={}, last_record=0)
    layouts=[]; backends=[]
    for owner in (0,1):
        order=np.array([owner,1-owner]); local=deepcopy(state)
        for key,a in local['arrays'].items():
            local['arrays'][key]=a[order].copy()
            local['arrays'][key][1]=999.
        layouts.append(SimpleNamespace(cell_l2g=order,n_owned_cells=1))
        backends.append(dict(surface_updates=local))
    mesh=SimpleNamespace(arrays=dict(latCell=np.zeros(2),lonCell=np.zeros(2)))
    payload=dict(backends=backends,layouts=layouts,mesh=mesh)
    weights=SimpleNamespace(overlap=csr_matrix([[.75,.25],[.25,.75]]),
                            source_area=np.ones(2),target_area=np.ones(2))
    fresh=dict(surface_updates=deepcopy(state))
    fresh['surface_updates']['arrays']['ocean_pwp_columns'][:,0,:]=310.
    target=SimpleNamespace(cell_l2g=np.array([0,1,0]),n_owned_cells=2)
    for key,a in fresh['surface_updates']['arrays'].items():
        fresh['surface_updates']['arrays'][key]=a[[0,1,0]].copy()
    fresh['surface_updates']['identity']['cells']=3
    receipt=remap_surface(payload,fresh,weights,target,mesh,np.array([0,1,0]),{})
    result=fresh['surface_updates']
    columns=cp.asnumpy(result['arrays']['ocean_pwp_columns'])
    np.testing.assert_allclose(columns[:2,0],[[295.5,291.5],[298.5,294.5]])
    np.testing.assert_allclose(columns[:2,1],[[34.5,35.5],[35.5,36.5]])
    np.testing.assert_allclose(columns[:2,2],[[1.5,1.5],[2.5,2.5]])
    np.testing.assert_array_equal(columns[2],columns[0])
    np.testing.assert_array_equal(cp.asnumpy(result['arrays']['ocean_pwp_diagnostics'])[:,2],[2,2,2])
    assert result['ocean']['steps']==48
    assert receipt['target_owned_sensible_heat_above_273_15_j']==pytest.approx(
        receipt['source_global_sensible_heat_above_273_15_j'],rel=1e-14)
    assert receipt['target_owned_salt_mass_kg']==pytest.approx(receipt['source_global_salt_mass_kg'],rel=1e-14)
    assert receipt['newly_initialized_owned_pwp_cells']==0
