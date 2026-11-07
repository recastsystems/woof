from types import SimpleNamespace
import numpy as np
import pytest
from woof.hex.moving_seaice import SEAICE_FIELDS, complete_source_ice, prepare_restart_ice


def source(ice):
    return dict(seam=dict(arrays={'fields/xice':np.array([ice],np.float32),
        'fields/landmask':np.zeros((1,len(ice)),np.float32),
        'fields/lakemask':np.zeros((1,len(ice)),np.float32),
        'fields/tsk':np.full((1,len(ice)),280,np.float32)}))


def test_optional_classes_keep_real_stores_and_fill_only_ice_free_columns():
    polar,tropical=source([.5,0]),source([0,0])
    for key in SEAICE_FIELDS:
        polar['seam']['arrays'][key]=np.full((1,2),2,np.float32)
    payload=dict(backends=[polar,tropical])
    result=complete_source_ice(payload)
    assert result['backends'][0]['seam']['arrays'] is polar['seam']['arrays']
    out=result['backends'][1]['seam']['arrays']
    for key in SEAICE_FIELDS-{'fields/ice_skin'}:
        np.testing.assert_array_equal(out[key],0)
    np.testing.assert_array_equal(out['fields/ice_skin'],np.float32(273.15))
    assert not SEAICE_FIELDS & tropical['seam']['arrays'].keys()
    tropical['seam']['arrays']['fields/xice'][0,0]=.1
    with pytest.raises(ValueError,match='no carried'):
        complete_source_ice(payload)


def test_every_carrier_the_engine_creates_moves():
    """run_84cabe12ce57b858: the engine's two newer carriers were missing from a copied list, so a roster member stopped."""
    from woof.ocean.seaice import STATE
    assert {'fields/'+name for name in STATE}<=SEAICE_FIELDS
    assert {'fields/ice_ocean_owed','fields/ice_ocean_coupled'}<=SEAICE_FIELDS


def test_unknown_and_partial_carriers_are_not_generalized():
    s=source([0])
    p=dict(backends=[s])
    assert complete_source_ice(p) is p
    s['seam']['arrays']['fields/ice_volume']=np.zeros((1,1),np.float32)
    with pytest.raises(ValueError,match='incomplete'):
        complete_source_ice(p)


def test_restore_allocates_exact_optional_manifest_without_overwriting_live_state():
    fields=dict(tsk=np.ones((1,3),np.float32),ice_volume=np.ones((1,3),np.float32)*7)
    seam=SimpleNamespace(_driver=SimpleNamespace(fields=fields),_cp=np)
    arrays={k:np.ones((1,3),np.float32) for k in SEAICE_FIELDS}
    prepare_restart_ice(seam,arrays)
    assert {'fields/'+k for k in fields if k!='tsk'}==SEAICE_FIELDS
    np.testing.assert_array_equal(fields['ice_volume'],7)
    arrays['fields/ice_skin']=np.zeros((1,2),np.float32)
    with pytest.raises(ValueError,match='invalid'):
        prepare_restart_ice(seam,arrays)


def test_remap_combines_ice_and_ice_free_classes_without_losing_stores():
    from copy import deepcopy
    from woof.hex.moving_physics import restore_remapped_backend
    from woof.hex.moving_remap import ConservativeMap
    from test_moving_patch import octants
    cells=octants();weights=ConservativeMap.build(cells,cells)
    mesh=SimpleNamespace(attrs={'sphere_radius':6371229.},
        arrays={k:cells.centers[:,i]*6371229. for i,k in enumerate(('xCell','yCell','zCell'))})
    sources=[source([.5,0,0,0]),source([0,0,0,0])]
    for s in sources:
        a=s['seam']['arrays']
        a.update({'fields/xland':np.full((1,4),2,np.float32),
                  'fields/isltyp':np.ones((1,4),np.int32),
                  'fields/ivgtyp':np.ones((1,4),np.int32)})
        s.update(identity={},adapter={})
        s['seam'].update(identity={'xice_threshold':.02},scalars=dict(
            elapsed_seconds=150.,carriers={},kf_history_time=None))
    for key in SEAICE_FIELDS:
        sources[0]['seam']['arrays'][key]=np.array([[1,0,0,0]],np.float32)
    layouts=[SimpleNamespace(cell_l2g=np.arange(i,i+4),n_owned_cells=4) for i in (0,4)]
    target=SimpleNamespace(cell_l2g=np.arange(8),n_owned_cells=8)
    fresh=deepcopy(sources[0])
    fresh['seam']['arrays']={k:np.concatenate([s['seam']['arrays'][k] for s in sources],axis=-1)
        for k in sources[1]['seam']['arrays']}
    saved=[]
    backend=SimpleNamespace(restart_state=lambda:deepcopy(fresh),restore_restart_state=saved.append)
    restore_remapped_backend(backend,dict(backends=sources,layouts=layouts,mesh=mesh,time_seconds=150.),
        weights,target,base_dt=18.75,rate=8,target_mesh=mesh)
    got=saved[0]['seam']['arrays']
    np.testing.assert_array_equal(got['fields/ice_volume'],[[1,0,0,0,0,0,0,0]])
    assert SEAICE_FIELDS <= got.keys()
    assert not SEAICE_FIELDS & sources[1]['seam']['arrays'].keys()


def test_coastline_transfer_conserves_ice_and_never_enables_glaciers():
    from scipy.sparse import csr_matrix
    from woof.hex.moving_seaice import remap_ocean_ice
    # Donor 0 overlaps only new glacier cell 0: nearest wet fallback is 1.
    # Donor 1 overlaps wet cells 1 and 2, and must retain its whole inventory.
    w=SimpleNamespace(overlap=csr_matrix([[1.,0,0],[0,.3,0],[0,.7,1.]]),
                      source_area=np.ones(3),target_area=np.array([1.,.3,1.7]))
    mesh=SimpleNamespace(attrs={'sphere_radius':6371229.},arrays={
        'xCell':np.array([1.,.99,0]),'yCell':np.array([0,.01,1.]),'zCell':np.zeros(3)})
    s=source([.8,.5,0])
    for key in SEAICE_FIELDS:s['seam']['arrays'][key]=np.zeros((1,3),np.float32)
    s['seam']['arrays']['fields/ice_volume']=np.array([[2.,1.,0]],np.float32)
    s['seam']['arrays']['fields/ice_snow_mass']=np.array([[30.,10.,0]],np.float32)
    s['seam']['arrays']['fields/ice_skin']=np.array([[260.,270.,273.15]],np.float32)
    s['seam']['arrays']['fields/ice_ocean_coupled']=np.array([[1.,1.,0]],np.float32)
    p=dict(backends=[s],layouts=[SimpleNamespace(cell_l2g=np.arange(3),n_owned_cells=3)],mesh=mesh)
    out,r=remap_ocean_ice(p,w,mesh,np.array([1,0,0]))
    np.testing.assert_array_equal(out['fields/ice_ocean_mask'],[[0,1,1]])
    np.testing.assert_array_equal(out['fields/ice_ocean_coupled'],[[0,1,1]])   # a 0/1 flag, not an overlap average
    assert out['fields/ice_volume'][0,0]==out['fields/ice_snow_mass'][0,0]==0
    for key in ('fields/ice_volume','fields/ice_snow_mass'):
        np.testing.assert_allclose((out[key]*w.target_area).sum(),s['seam']['arrays'][key].sum(),rtol=1e-7)
    np.testing.assert_allclose(out['fields/ice_thickness']*out['fields/xice'],out['fields/ice_volume'])
    np.testing.assert_allclose(out['fields/ice_snow_depth']*330*out['fields/xice'],out['fields/ice_snow_mass'],rtol=1e-7)
    assert r['nearest_wet_donors']==1 and r['concentration_capped_cells']==1
    assert r['nearest_wet_donors_with_ice']==1
    assert r['nearest_wet_inventory_fractions']['fields/ice_volume']==pytest.approx(2/3)
    assert 0 < r['nearest_wet_distance_max_km'] < 100
    assert np.all(out['fields/xice']<=1)
    assert np.all((out['fields/ice_skin']>=260)&(out['fields/ice_skin']<=273.15))


def test_partitioned_restore_keeps_glacier_snow_separate_from_ocean_inventory():
    from copy import deepcopy
    from woof.hex.moving_physics import restore_remapped_backend
    from woof.hex.moving_remap import ConservativeMap
    from test_moving_patch import octants
    cells=octants();weights=ConservativeMap.build(cells,cells)
    mesh=SimpleNamespace(attrs={'sphere_radius':6371229.},
        arrays={k:cells.centers[:,i]*6371229. for i,k in enumerate(('xCell','yCell','zCell'))})
    old=source([0,.5,0,0,0,0,0,0]);a=old['seam']['arrays']
    a['fields/landmask'][0,0]=1
    a.update({'fields/xland':np.array([[1,1,2,2,2,2,2,2]],np.float32),
              'fields/isltyp':np.ones((1,8),np.int32),
              'fields/ivgtyp':np.array([[15,15,17,17,17,17,17,17]],np.int32),
              'fields/snow':np.array([[100,20,0,0,0,0,0,0]],np.float32),
              'fields/snowh':np.array([[1,.1,0,0,0,0,0,0]],np.float32)})
    for key in SEAICE_FIELDS:a[key]=np.zeros((1,8),np.float32)
    a['fields/ice_volume'][0,1]=1
    a['fields/ice_snow_mass'][0,1]=10
    a['fields/ice_skin'][:]=260
    old.update(identity={},adapter={});old['seam'].update(identity={'xice_threshold':.02},
        scalars=dict(elapsed_seconds=150.,carriers={},kf_history_time=None))
    p=dict(backends=[old],layouts=[SimpleNamespace(cell_l2g=np.arange(8),n_owned_cells=8)],
           mesh=mesh,time_seconds=150.)
    target_land=np.array([0,1,0,0,0,0,0,0],np.float32)
    cache={'target_landmask':target_land};outputs=[]
    for ids in (np.arange(4),np.arange(4,8)):
        fresh=deepcopy(old)
        fresh['seam']['arrays']={k:v[...,ids].copy() for k,v in a.items() if k not in SEAICE_FIELDS}
        f=fresh['seam']['arrays'];f['fields/landmask']=target_land[None,ids]
        f['fields/xland']=np.where(f['fields/landmask']>0,1,2).astype(np.float32)
        f['fields/xice'][:]=0
        f['fields/ivgtyp']=np.where(f['fields/landmask']>0,15,17).astype(np.int32)
        backend=SimpleNamespace(restart_state=lambda:deepcopy(fresh),restore_restart_state=outputs.append)
        restore_remapped_backend(backend,p,weights,SimpleNamespace(cell_l2g=ids,n_owned_cells=len(ids)),
            base_dt=18.75,rate=8,target_mesh=mesh,mapped_cache=cache)
    result={k:np.concatenate([o['seam']['arrays'][k] for o in outputs],axis=-1) for k in a}
    assert result['fields/snow'][0,1]==100 and result['fields/snowh'][0,1]==1
    assert result['fields/ice_ocean_mask'][0,1]==0
    assert result['fields/ice_volume'][0,1]==result['fields/ice_snow_mass'][0,1]==0
    np.testing.assert_allclose(np.sum(result['fields/ice_volume']*weights.target_area),
                               np.sum(a['fields/ice_volume']*weights.source_area),rtol=1e-7)
    wet=target_land==0
    np.testing.assert_allclose(result['fields/snow'][...,wet]*result['fields/xice'][...,wet],
                               result['fields/ice_snow_mass'][...,wet])
