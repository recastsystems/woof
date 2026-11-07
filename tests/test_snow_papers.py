"""Physical and lifecycle checks for the independently written snow solver."""
import numpy as np

def test_packaged_optics_provenance_matches_runtime():
    import hashlib,json
    from pathlib import Path
    from woof.core.snow_papers import TABLE_SHA256,TABLE_WORDS,STATE_WORDS,identity
    data=Path(__file__).resolve().parents[1]/'woof/data/snow_papers'
    receipt=json.loads((data/'receipt.json').read_text())
    pin=json.loads((data/'source-pin.json').read_text())
    assert receipt['algorithm']==identity()['algorithm']
    assert receipt['table_words']==TABLE_WORDS and receipt['state_words']==STATE_WORDS
    assert receipt['table_sha256']==pin['generated_sha256']==TABLE_SHA256
    assert hashlib.sha256((data/'optics.bin').read_bytes()).hexdigest()==TABLE_SHA256
import pytest
from conftest import requires_gpu

@pytest.mark.parametrize("surface,albedo", [(0, 2), (0, 3), (4, 2)])
def test_unselected_snow_controls_preserve_checkpoint_identity(surface, albedo):
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.io.restart import configuration_echo, _require_config_match
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=3000, dy=3000, ztop=12000,
                    dt=10, run_seconds=60, sf_surface_physics=surface,
                    opt_alb=albedo)
    old = configuration_echo(cfg)
    assert not any(key.startswith("snicar_") for key in old)
    # Even an explicit unused control cannot change this trajectory.
    changed = replace(cfg, snicar_bc_ppb=100)
    assert configuration_echo(changed) == old
    _require_config_match(old, changed, "older-checkpoint.npz")


@pytest.mark.parametrize("name,value", [
    ("snicar_snowshape_opt", 1), ("snicar_snowbc_intmix", False),
    ("snicar_snowdust_intmix", True), ("snicar_bc_ppb", 100.0),
    ("snicar_dust_ppm", 10.0), ("snicar_ice_density", 850.0),
    ("snicar_bubble_radius_um", 500.0),
])
def test_selected_snow_controls_bind_checkpoint_identity(name, value):
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.io.restart import configuration_echo, _require_config_match
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=3000, dy=3000, ztop=12000,
                    dt=10, run_seconds=60, sf_surface_physics=4, opt_alb=3)
    old = configuration_echo(cfg)
    assert name in old
    with pytest.raises(ValueError, match=name):
        _require_config_match(old, replace(cfg, **{name: value}), "snow.npz")

@requires_gpu
def test_snow_grain_impurity_and_depth_response():
    import cupy as cp
    from woof.core.snow_papers import evaluate
    n=7
    x=cp.zeros((n,56),dtype=cp.float32)
    x[:,24]=60; x[:,25]=.7; x[:,28]=263.15; x[:,30]=.4; x[:,31]=1
    x[:,34]=100; x[:,42]=-55
    x[:,5:9]=.1
    x[:,52:56]=100
    x[1,42]=-1000
    x[3,34]=.1
    x[4,25]=.1
    x[5,25]=-1
    x[6,34]=0
    carbon=cp.zeros(n,dtype=cp.float32);carbon[2]=1e-6
    y=cp.asnumpy(evaluate(x,cp.ones(n,dtype=cp.int32),carbon=carbon))
    assert np.isfinite(y).all()
    assert ((y[:,:4]>=0)&(y[:,:4]<=1)).all()
    assert y[0,0]>.9 and .3<y[0,1]<.9
    assert y[1,1]<y[0,1] # larger grains absorb more NIR
    assert y[2,0]<y[0,0] # carbon darkens visible snow
    assert y[3,0]<y[0,0] # thin snow reveals dark ground
    assert y[4,0]>y[0,0] # grazing beam reflects more
    assert np.array_equal(y[5,:4],np.zeros(4))
    assert y[5,4]<-55 # night grain evolution is carried
    assert y[6,4]==0
    print('snow responses',y.tolist())

@requires_gpu
def test_snow_papers_restart_trajectory(tmp_path):
    import cupy as cp
    from test_noahmp_runtime import _build,_CARRIED
    from woof.core.dycore import step
    from woof.io.restart import write_restart,restore_restart
    options=dict(nx=6,ny=4,snow_mm=40,snow_depth_m=.2,opt_alb=3)
    a,cfg,da=_build(**options)
    for _ in range(3):step(a,cfg)
    assert float(cp.min(da.fields['taussxy']).item())<0
    path=tmp_path/'snow.npz';write_restart(path,a,cfg)
    b,cfgb,db=_build(**options);restore_restart(path,b,cfgb)
    step(a,cfg);step(b,cfgb)
    for key in (*_CARRIED, "snicar_statexy"):
        np.testing.assert_array_equal(cp.asnumpy(da.fields[key]),cp.asnumpy(db.fields[key]),err_msg=key)
    assert da.noahmp_params.restart_identity()['snow_papers']['status']=='written from papers, not bitwise to WRF'


def test_stock_namelist_preserves_papers_option(tmp_path):
    from test_namelist_import import _import_with
    text,report=_import_with(tmp_path,extra_input="&noah_mp opt_alb=3 /\n")
    assert 'opt_alb = 3' in text


def test_snow_memory_prices_are_conditional():
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.core.preflight import noahmp_lsm_transient_shapes, noahmp_lsm_cache_shapes
    from woof.core.noahmp_selected_frames import selected_frames
    cfg = RunConfig(nx=73, ny=60, nz=34, dx=30000, dy=30000,
                    ztop=18000, dt=60, run_seconds=10800,
                    sf_surface_physics=4, opt_alb=3)
    assert noahmp_lsm_transient_shapes(cfg)["noahmp_lsm/snow_papers_transients"] == (4380, 1024)
    assert noahmp_lsm_cache_shapes(cfg)["noahmp_lsm/snow_papers_optics"] == (283703, 4)
    assert selected_frames(snow_papers=True)[0]["snow_papers"] > 0
    for option in (1, 2):
        other = replace(cfg, opt_alb=option)
        assert not any("snow_papers" in key for key in noahmp_lsm_transient_shapes(other))
        assert not any("snow_papers" in key for key in noahmp_lsm_cache_shapes(other))


@requires_gpu
def test_rust_rejects_changed_asset_without_touching_output(tmp_path):
    import ctypes
    from pathlib import Path
    from woof.core.snow_papers import _host_table, TABLE_WORDS, TABLE_SHA256
    from woof.static.rust_bridge import load
    assert _host_table().shape == (TABLE_WORDS,)
    reader = load().gpuwm_snow_optics_load_v1
    source = Path(__file__).resolve().parents[1]/"woof/data/snow_papers/optics.bin"
    bad = bytearray(source.read_bytes()); bad[-1] ^= 1
    changed = tmp_path/"changed.bin"; changed.write_bytes(bad)
    output = np.full(TABLE_WORDS, -123, dtype=np.float32)
    assert reader(str(changed).encode(), TABLE_SHA256.encode(),
                  output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size) != 0
    np.testing.assert_array_equal(output, np.full_like(output, -123))

@requires_gpu
@pytest.mark.parametrize('feedback',[0,1])
def test_nested_snow_checkpoint(feedback,tmp_path):
    from test_noahmp_nested_lifecycle import test_live_selected_nested_checkpoint_continuation
    test_live_selected_nested_checkpoint_continuation(feedback,dict(opt_alb=3,snow_mm=40,snow_depth_m=.2),tmp_path)

@requires_gpu
def test_selected_glacier_snow_captures(monkeypatch):
    from test_noahmp_glacier_slab import test_selected_glacier_step_captures_without_host_transfers
    test_selected_glacier_step_captures_without_host_transfers(dict(opt_alb=3),monkeypatch)

@requires_gpu
def test_layered_energy_habits_mixing_and_ice():
    import cupy as cp
    from woof.core.snow_papers import evaluate, refreeze
    n=4
    x=cp.zeros((n,56),dtype=cp.float32)
    x[:,24]=60; x[:,25]=.6; x[:,28]=263.15; x[:,30]=.4
    x[:,31]=1; x[:,34]=100; x[:,42]=-100; x[:,5:9]=.15; x[:,52:56]=100
    layers=cp.zeros((n,3,4),dtype=cp.float32)
    layers[:,:,0]=cp.asarray([10,30,60],dtype=cp.float32)
    layers[:,:,2]=cp.asarray([.05,.12,.23],dtype=cp.float32)
    layers[:,:,3]=263.15
    values=[]
    for shape in range(1,5):
        y,s,a=evaluate(x,cp.ones(n,dtype=cp.int32),layers=layers,
            options=dict(snicar_snowshape_opt=shape),return_fluxes=True)
        np.testing.assert_allclose(cp.asnumpy(cp.sum(a,axis=1)+y[:,:4]),1,atol=3e-6)
        assert bool(cp.all(a>=0))
        values.append(cp.asnumpy(y[0,:4]))
    assert len({tuple(v) for v in values})==4
    clean=values[2]
    for species,concentration in [('bc',100.),('dust',10.)]:
        for internal in (False,True):
            options=dict(snicar_snowbc_intmix=False,snicar_snowdust_intmix=False)
            options['snicar_'+species+('_ppb' if species=='bc' else '_ppm')]=concentration
            options['snicar_snow'+species+'_intmix']=internal
            y,s,a=evaluate(x,cp.ones(n,dtype=cp.int32),layers=layers,options=options,return_fluxes=True)
            assert float(y[0,0])<clean[0]
            np.testing.assert_allclose(cp.asnumpy(cp.sum(a,axis=1)+y[:,:4]),1,atol=3e-6)
    ice=cp.empty((n,4,3),dtype=cp.float32)
    ice[:,:,0]=cp.asarray([.1,.3,.6,1],dtype=cp.float32)
    ice[:,:,1]=cp.asarray([700,800,900,916],dtype=cp.float32)[:,None]
    ice[:,:,2]=cp.asarray([100,500,1000,10000],dtype=cp.float32)[:,None]
    x[:,34]=0; layers[:]=0
    y,s,a=evaluate(x,cp.ones(n,dtype=cp.int32),layers=layers,glacier=cp.ones(n),ice_layers=ice,return_fluxes=True)
    np.testing.assert_allclose(cp.asnumpy(cp.sum(a,axis=1)+y[:,:4]),1,atol=3e-6)
    assert bool(cp.all(cp.isfinite(y))) and bool(cp.any(a[:,4:,:]>0))
    state=cp.zeros((n,21),dtype=cp.float32); state.reshape(n,3,7)[:,:,1]=100
    before=cp.full((n,3),10,dtype=cp.float32);after=cp.full((n,3),12,dtype=cp.float32)
    refreeze(state,before,after,cp.zeros_like(before))
    np.testing.assert_allclose(cp.asnumpy(state.reshape(n,3,7)[:,:,1]),250,atol=1e-5)
    print('four-habit albedo',np.asarray(values).tolist(),'ice albedo',cp.asnumpy(y[:,:4]).tolist())

@requires_gpu
def test_ad_conservative_and_zero_depth_limits():
    import cupy as cp
    from woof.core.snow_papers import _module
    x=cp.asarray([[0.,.1,.8,.6],[10.,0.,.8,.6],[10000.,.05,.8,.6],[1.,.5,0.,(1/1.5)**.5]],dtype=cp.float64)
    y=cp.empty((4,10),dtype=cp.float64)
    _module(cp.cuda.Device().id).get_function('snow_ad_probe')((1,),(32,),(x,y,np.int32(4)))
    a=cp.asnumpy(y)
    np.testing.assert_allclose(a[0,:4],[0,1,0,1],atol=1e-12)
    np.testing.assert_allclose(a[1,:2].sum(),1,atol=1e-12)
    np.testing.assert_allclose(a[1,2:4].sum(),1,atol=1e-12)
    assert np.isfinite(a).all() and (a>=0).all() and (a<=1+1e-12).all()
    assert a[2,1]<1e-12
