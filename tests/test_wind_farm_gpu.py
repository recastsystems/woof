"""Fitch conservation and sparse GPU equivalence; run inside a GPU lease."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import cupy as cp
from woof.core.wind_farm import WindFarm

def demo_table():
    return dict(schema="gpuwm-wind-farm.v1", types={"example":dict(curve=[[0,.8,.4],[30,.8,.4]])},
        turbines=[dict(id="first",type="example",grid_x=8.,grid_y=8.,hub_height_m=100.,rotor_diameter_m=120.),
                  dict(id="second",type="example",grid_x=14.,grid_y=8.,hub_height_m=100.,rotor_diameter_m=120.)])

def setup(tmp_path, scheme='fitch', *, turbines=1, dz=10, km=2):
    data=demo_table();data['turbines']=data['turbines'][:turbines]
    # Constant coefficients make the integrated energy budget independent of interpolation.
    data['types']['example']['curve']=[[0,.8,.4,12,0],[30,.8,.4,12,0]]
    path=tmp_path/'turbines.json';path.write_text(json.dumps(data))
    cfg=SimpleNamespace(wind_farm_table=path,wind_farm_scheme=scheme,dx=100.,dy=100.,dt=.1)
    shape=(40,24,24);farm=WindFarm(cfg,shape)
    zi=cp.broadcast_to(cp.arange(41,dtype=cp.float32)[:,None,None]*dz,(41,24,24)).copy()
    atmosphere=dict(z_interface=zi,u=cp.full(shape,8,cp.float32),v=cp.zeros(shape,cp.float32),rho=cp.full(shape,1.2,cp.float32))
    fields=dict(qke=cp.ones(shape,cp.float32),qke_adv=cp.ones(shape,cp.float32),exch_m=cp.full(shape,km,cp.float32))
    rates=dict(du=cp.zeros(shape,cp.float32),dv=cp.zeros(shape,cp.float32))
    return farm,cfg,atmosphere,fields,rates

def test_integrated_fitch_energy_and_retained_tke(tmp_path):
    farm,cfg,a,f,r=setup(tmp_path)
    farm.apply_mynn(a,f,r)
    force=float(cp.sum(r['du']).get())*100**2*10
    expected=-.5*.8*8**2*np.pi*60**2
    assert force==pytest.approx(expected,rel=2e-6)
    assert float(cp.sum(farm.tke_rate).get())*100**2*10==pytest.approx(.5*.25*(.8-.4)*8**3*np.pi*60**2,rel=2e-6)
    cp.testing.assert_array_equal(f['qke'],f['qke_adv'])
    cp.testing.assert_allclose(f['qke']-1,2*cfg.dt*farm.tke_rate,atol=6e-8,rtol=1e-5)
    cp.testing.assert_array_equal(r['dv'],0)

def test_parked_turbine_inert(tmp_path):
    farm,cfg,a,f,r=setup(tmp_path)
    farm.turbines[:,4]=0
    farm.apply_mynn(a,f,r)
    cp.testing.assert_array_equal(r['du'],0)
    cp.testing.assert_array_equal(f['qke'],1)


def test_changed_step_integrates_tke_for_actual_elapsed_time(tmp_path):
    farm,cfg,a,f,r=setup(tmp_path)
    farm.apply_mynn(a,f,r,step_seconds=.1)
    farm.apply_mynn(a,f,r,step_seconds=.2)
    cp.testing.assert_allclose(f['qke']-1, 2*.3*farm.tke_rate, rtol=1e-5, atol=1e-7)
    cp.testing.assert_array_equal(f['qke'],f['qke_adv'])


def test_voronoi_area_sets_turbine_density(tmp_path):
    from woof.core.wind_farm import attach_voronoi_wind_farm
    data=demo_table()
    for row,lon in zip(data['turbines'],(-.01,.01)):
        row.pop('grid_x');row.pop('grid_y')
        row.update(latitude=0.,longitude=lon)
    path=tmp_path/'cells.json';path.write_text(json.dumps(data))
    cfg=SimpleNamespace(wind_farm_table=str(path),dt=.1)
    zi=cp.broadcast_to(cp.arange(41,dtype=cp.float32)[:,None]*10,(41,3)).copy()
    farm=attach_voronoi_wind_farm(cfg,latitude=cp.zeros(3,cp.float32),
        longitude=cp.deg2rad(cp.asarray([-.01,.01,120.],dtype=cp.float32)),
        area=cp.asarray([10000.,20000.,10000.],dtype=cp.float32),z_interface=zi)
    atmosphere=dict(z_interface=zi,u=cp.full((40,3),8,cp.float32),
                    v=cp.zeros((40,3),cp.float32),rho=cp.full((40,3),1.2,cp.float32))
    fields=dict(qke=cp.ones((40,3),cp.float32),qke_adv=cp.ones((40,3),cp.float32))
    rates=dict(du=cp.zeros((40,3),cp.float32),dv=cp.zeros((40,3),cp.float32))
    farm.apply_mynn(atmosphere,fields,rates)
    cp.testing.assert_allclose(rates['du'][:,0],2*rates['du'][:,1],rtol=2e-6)
    cp.testing.assert_array_equal(rates['du'][:,2],0)
    outputs=farm.history_fields()
    cp.testing.assert_allclose(outputs['FARM_POWER'].sum(),outputs['TURBINE_POWER'].sum())

def test_sparse_matches_original_with_shear_and_shared_column(tmp_path):
    farm,cfg,a,f,r=setup(tmp_path,turbines=2)
    # Two turbines share a grid cell: exercise block accumulation, with veer.
    data=farm.data
    data['turbines'][1].update(grid_x=8.,grid_y=8.)
    cfg.wind_farm_table.write_text(json.dumps(data))
    farm=WindFarm(cfg,farm.shape)
    z=cp.arange(40,dtype=cp.float32)[:,None,None]
    a['u'][:]=3+.2*z
    a['v'][:]=1+.04*z
    legacy=WindFarm(cfg,farm.shape);legacy.sparse=False
    f2={k:v.copy() for k,v in f.items()};r2={k:v.copy() for k,v in r.items()}
    farm.apply_mynn(a,f,r);legacy.apply_mynn(a,f2,r2)
    for name in r:cp.testing.assert_allclose(r[name],r2[name],rtol=3e-6,atol=1e-9)
    for name in ('qke','qke_adv'):cp.testing.assert_allclose(f[name],f2[name],rtol=3e-6)
    cp.testing.assert_allclose(farm.tke_rate,legacy.tke_rate,rtol=3e-6,atol=1e-9)

def test_geographic_inventory_accepts_host_grid_arrays(tmp_path):
    from woof.core.wind_farm import attach_wind_farm
    data=demo_table();data['turbines']=data['turbines'][:1]
    row=data['turbines'][0]
    row.pop('grid_x');row.pop('grid_y')
    row.update(latitude=35.08,longitude=-97.92)
    path=tmp_path/'geographic.json';path.write_text(json.dumps(data))
    cfg=SimpleNamespace(wind_farm_table=str(path),wind_farm_scheme='auto',
        wind_farm_max_dx_m=3000.,dx=1000.,dy=1000.,dt=5.,
        bl_pbl_physics=5,bl_mynn_tkeadvect=True,bldt=0)
    lat=np.broadcast_to((35.+np.arange(24)*.01)[:,None],(24,24)).copy()
    lon=np.broadcast_to((-98.+np.arange(24)*.01)[None,:],(24,24)).copy()
    farm,receipt=attach_wind_farm(cfg,(40,24,24),latitude=lat,longitude=lon,
        msf=np.ones((24,24),np.float32))
    assert receipt['enabled'] and receipt['turbine_count']==1
    assert int(farm.col[0].get())==8*24+8


@pytest.mark.parametrize("advect", [False, True])
def test_whole_mynn_sparse_coupling(tmp_path, monkeypatch, advect):
    from test_mynn_mixscalars_inert_gpu import _config, _run
    import woof.core.physics as physics
    from woof.netcdf_bridge import open_dataset
    data=demo_table();data['turbines']=data['turbines'][:1]
    data['turbines'][0].update(grid_x=3.,grid_y=3.)
    path=tmp_path/'table.json';path.write_text(json.dumps(data))
    captured=[]
    original=physics.initialize_physics
    def initialize(*a,**kw):
        driver=original(*a,**kw)
        assert bool(cp.all(driver.wind_farm.hub[:,2] > 0))
        assert float(driver.wind_farm.hub[:,7].sum().get()) > 0
        cp.testing.assert_array_equal(driver.wind_farm.tke_rate, 0)
        captured.append(driver)
        return driver
    monkeypatch.setattr(physics,'initialize_physics',initialize)
    arrays,_=_run(_config(wind_farm_scheme='fitch',wind_farm_table=str(path),bl_mynn_tkeadvect=advect))
    farm=captured[0].wind_farm
    assert farm.cost.calls==0
    assert farm.sparse
    assert farm.receipt()['cost'] is None
    assert np.isfinite(arrays['u']).all()
    if advect:
        assert np.isfinite(arrays['qke_adv']).all()
    else:
        assert 'qke_adv' not in arrays
        assert bool(cp.all(cp.isfinite(captured[0].fields['qke'])))
    assert float(farm.tke_rate.sum().get())>0
    assert farm.replay is None
