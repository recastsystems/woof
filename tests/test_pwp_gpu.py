"""Actual CUDA PWP driver, finite radiation, unequal depths, and restart."""
import dataclasses
import json
import numpy as np
import pytest
from conftest import requires_gpu
from test_oml_gpu import make_state
from woof.ocean.mixed_layer import initialize, step, check, STATE_FIELDS
from test_oml_contract import run_config


def columns(cp,levels=30):
    fields={name:cp.zeros((8,10),cp.float32) for name in
            ('hfx','lh','gsw','glw','qfx','ust','uair','vair','f')}
    fields['rhoair']=cp.full((8,10),1.2,cp.float32)
    fields['xland']=cp.full((8,10),2.,cp.float32)
    fields['xland'][:,-1]=1.
    fields['tsk']=cp.full((8,10),302.,cp.float32)
    cfg=run_config(sf_ocean_physics=2,ocean_levels=levels,ocean_mld_climatology='configured_uniform_mld')
    with pytest.warns(RuntimeWarning,match='configured_uniform_mld'):
        initialize(fields,cfg)
    return fields


@requires_gpu
@pytest.mark.gpu
def test_zero_radiation_is_finite_and_heat_loss_has_correct_sign():
    import cupy as cp
    a=columns(cp);a['hfx'].fill(80.);a['lh'].fill(300.)
    before=a['om_tmp'].copy()
    step(a,dt=30.,dx=10000.,dy=10000.);check(a)
    assert bool(cp.all(cp.isfinite(a['om_tmp'])))
    assert bool(cp.all(a['om_tmp'][0,:,:-1]<before[0,:,:-1]))
    assert bool(cp.all(a['tsk'][:,-1]==302.))
    assert bool(cp.all(a['om_u'][:,:,-1]==0.))


@requires_gpu
@pytest.mark.gpu
def test_unequal_layers_mix_without_truncated_weight_and_conserve_tracers():
    import cupy as cp
    a=columns(cp,4)
    z=cp.asarray([5.,20.,50.,100.],dtype=cp.float32)[:,None,None]
    a['om_depth'][...]=z
    a['om_tmp'][...]=cp.asarray([300.,299.99,299.,297.],dtype=cp.float32)[:,None,None]
    a['om_tini'][...]=a['om_tmp']
    a['om_u'][0]=3.
    a['glw'][...]=cp.float32(5.67051e-8)*a['om_tmp'][0]**4
    # Surface longwave balances emission; no longwave penetrates.
    dz=cp.diff(cp.concatenate((cp.zeros_like(z[:1]),z)),axis=0)
    before=cp.sum(a['om_tmp']*dz,axis=0)
    step(a,dt=30.,dx=10000.,dy=10000.);check(a)
    assert bool(cp.all(cp.isfinite(a['om_u'])))
    assert float(a['om_u'][0,3,3])<3.
    expected=before
    cp.testing.assert_allclose(cp.sum(a['om_tmp']*dz,axis=0)[:,:-1],expected[:,:-1],rtol=2e-6,atol=0.01)


@requires_gpu
@pytest.mark.gpu
def test_pwp_surface_driver_cadence():
    import cupy as cp
    from woof.core.dycore import run_steps
    with pytest.warns(RuntimeWarning,match='configured_uniform_mld'):
        state,cfg,driver=make_state(cp,ocean=2)
    run_steps(state,cfg,1);first=cp.asnumpy(driver.fields['om_tmp'])
    run_steps(state,cfg,4)
    assert cp.asnumpy(driver.fields['om_tmp']).tobytes()==first.tobytes()
    run_steps(state,cfg,1)
    assert cp.asnumpy(driver.fields['om_tmp']).tobytes()!=first.tobytes()
    check(driver.fields)


@requires_gpu
@pytest.mark.gpu
def test_actual_pwp_restart_roundtrip_and_continuation(tmp_path):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    def make():
        with pytest.warns(RuntimeWarning,match='configured_uniform_mld'):
            return make_state(cp,ocean=2)
    a,cfg,da=make();run_steps(a,cfg,16)
    ref=restart.write_restart(tmp_path/'reference.npz',a,cfg)
    b,cfg,db=make();run_steps(b,cfg,8)
    mid=restart.write_restart(tmp_path/'mid.npz',b,cfg)
    c,cfg,dc=make();restart.restore_restart(mid,c,cfg)
    for name in STATE_FIELDS:
        cp.testing.assert_array_equal(dc.fields[name],db.fields[name])
    run_steps(c,cfg,8)
    resumed=restart.write_restart(tmp_path/'resumed.npz',c,cfg)
    with np.load(ref,allow_pickle=False) as expected,np.load(resumed,allow_pickle=False) as got:
        assert set(expected.files)==set(got.files)
        for name in expected.files:
            if name=='__gpuwm_restart_header__':
                left=json.loads(expected[name].tobytes());right=json.loads(got[name].tobytes())
                for key in ('elapsed_seconds','setup_fingerprint','physics_setup_fingerprint','physics_setup','driver','array_manifest','config'):
                    assert left[key]==right[key],key
            else:
                assert expected[name].tobytes()==got[name].tobytes(),name


@requires_gpu
@pytest.mark.gpu
def test_nearest_bins_match_spherical_search_at_dateline_and_poles():
    import cupy as cp
    from woof.ocean.mixed_layer import kernels
    rng=np.random.default_rng(1020)
    slat=np.concatenate((rng.uniform(-90,90,512),[89.9,-89.9,0,0])).astype('f4')
    slon=np.concatenate((rng.uniform(-180,180,512),[-170,170,179.9,-179.9])).astype('f4')
    tlat=np.array([89.99,-89.99,0,0,35,-35],dtype='f4')
    tlon=np.array([50,-50,-179.95,179.95,80,-80],dtype='f4')
    surface=cp.full(slat.size,300,dtype='f4');surface[2]=cp.nan
    lat,lon=cp.asarray(slat),cp.asarray(slon)
    counts=cp.zeros(64800,cp.int32);offset=cp.zeros(64801,cp.int32)
    cursor=cp.zeros_like(counts);members=cp.empty(slat.size,cp.int32)
    mod=kernels(2);ns=np.int32(slat.size);nt=np.int32(tlat.size)
    mod.get_function('ocean_bin_count')((3,),(256,),(lat,lon,surface,counts,ns))
    cp.cumsum(counts,dtype=cp.int32,out=offset[1:])
    mod.get_function('ocean_bin_scatter')((3,),(256,),(lat,lon,surface,offset,cursor,members,ns))
    out=cp.empty(tlat.size,cp.int32)
    mod.get_function('ocean_nearest')((int(nt),),(256,),
        (lat,lon,offset,members,cp.asarray(tlat),cp.asarray(tlon),cp.full(int(nt),2.,cp.float32),out,nt))
    expected=[]
    for a,b in zip(tlat,tlon):
        dy=np.sin(np.deg2rad(a-slat)*.5);dx=np.sin(np.deg2rad(b-slon)*.5)
        d=dy*dy+np.cos(np.deg2rad(a))*np.cos(np.deg2rad(slat))*dx*dx
        d[2]=np.inf;j=int(d.argmin());expected.append(j if d[j]<=0.00007615242 else -1)
    cp.testing.assert_array_equal(out,cp.asarray(expected,dtype=cp.int32))


@requires_gpu
@pytest.mark.gpu
def test_lakes_do_not_receive_ocean_cooling_or_currents():
    import cupy as cp
    a=columns(cp,4);a['lakemask']=cp.zeros_like(a['xland']);a['lakemask'][:,4]=1
    cfg=run_config(sf_ocean_physics=2,ocean_levels=4,ocean_mld_climatology='configured_uniform_mld')
    with pytest.warns(RuntimeWarning,match='configured_uniform_mld'):
        initialize(a,cfg)
    a['lh'].fill(300);a['ust'].fill(2);a['uair'].fill(40)
    step(a,dt=30,dx=10000,dy=10000);check(a)
    cp.testing.assert_array_equal(a['tsk'][:,4],cp.full(8,302,cp.float32))
    cp.testing.assert_array_equal(a['om_u'][:,:,4],cp.zeros((4,8),cp.float32))
    assert bool(cp.all(a['tsk'][:,3]<302))


@requires_gpu
@pytest.mark.gpu
def test_surface_interval_is_fully_advanced_by_bounded_ocean_substeps():
    import cupy as cp
    from woof.core.surface_ocean import launch_ocean_step
    a=columns(cp,4);b=columns(cp,4)
    for fields in (a,b):
        fields['lh'].fill(300);fields['ust'].fill(1);fields['uair'].fill(30)
    launch_ocean_step(a,dt=150,dx=10000,dy=10000,gamma=.14,relaxation_time=0,ocean_timestep_minutes=1)
    for _ in range(3):step(b,dt=50,dx=10000,dy=10000)
    check(a);check(b)
    for name in STATE_FIELDS:cp.testing.assert_array_equal(a[name],b[name])


@requires_gpu
@pytest.mark.gpu
def test_actual_history_rust_roundtrip_keeps_independent_ocean_axis(tmp_path):
    import cupy as cp
    from woof.io.wrfout import WrfoutWriter, state_frame
    from woof.netcdf_bridge import open_dataset
    with pytest.warns(RuntimeWarning,match='configured_uniform_mld'):
        state,cfg,driver=make_state(cp,ocean=2)
    frame=state_frame(state,include_diagnostic_pressure=True)
    path=tmp_path/'ocean-history.nc'
    with WrfoutWriter(path,nx=cfg.nx,ny=cfg.ny,nz=cfg.nz,dx=cfg.dx,dy=cfg.dy,engine='rust') as writer:
        writer.write_frame('2022-09-26_00:00:00',frame)
    with open_dataset(path) as ds:
        assert ds.variables['OM_TMP'].dimensions==('Time','ocean_levels','south_north','west_east')
        assert ds.variables['OM_TMP'].shape==(1,30,cfg.ny,cfg.nx)
        np.testing.assert_array_equal(ds.variables['OM_TMP'][0],cp.asnumpy(driver.fields['om_tmp']))
        assert ds.variables['T'].shape==(1,cfg.nz,cfg.ny,cfg.nx)
        delta=ds.variables['OCEAN_SST_CHANGE'][0]
        np.testing.assert_array_equal(delta[:,:-1],0.)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize('option',[1,2])
def test_ocean_sst_change_tracks_coupled_initial_state_and_masks_land(option):
    import cupy as cp
    from woof.core.surface_ocean import update_ocean_diagnostics
    from woof.core.dycore import run_steps
    if option==2:
        with pytest.warns(RuntimeWarning,match='configured_uniform_mld'):
            state,cfg,driver=make_state(cp,ocean=option)
    else:
        state,cfg,driver=make_state(cp,ocean=option)
    f=driver.fields
    f['lakemask']=cp.zeros_like(f['xland']);f['lakemask'][:,4]=1
    before=f['tsk'].copy()
    run_steps(state,cfg,7)
    update_ocean_diagnostics(f)
    wet=(f['xland']>1.5)&(f['lakemask']<.5)
    cp.testing.assert_array_equal(f['ocean_sst_change'][wet],(f['tsk']-before)[wet])
    assert bool(cp.all(cp.isfinite(f['ocean_sst_change'])))
    assert bool(cp.all(cp.isfinite(f['ocean_sst'])))
    from woof.io.wrfout import state_frame
    frame = state_frame(state)
    # state_frame hands history writers host arrays; index them with a host mask.
    land = ~cp.asnumpy(wet)
    assert bool(np.all(np.isnan(np.asarray(frame['OCEAN_SST_CHANGE'])[land])))
    assert bool(np.all(np.isnan(np.asarray(frame['OCEAN_SST'])[land])))


@requires_gpu
@pytest.mark.gpu
def test_water_beyond_one_degree_of_woa18_borrows_the_nearest_mld_instead_of_refusing():
    # PWP is the 48 h+ default; inland water far from the WOA18 ocean refused
    # every continental forecast (full-CONUS HRRR, test host 2026-10-21).
    import cupy as cp
    from woof.ocean.mixed_layer.climatology import _borrow_beyond_one_degree
    lon, lat = cp.meshgrid(cp.arange(0., 4., 0.25, dtype=cp.float32),
                           cp.arange(0., 2., 0.25, dtype=cp.float32))
    valid = cp.zeros(lat.shape, dtype=cp.bool_)
    valid[0, 0] = True          # (0, 0)
    valid[0, 12] = True         # (0, 3)
    target_lat = cp.asarray([[0.0, 0.0, 1.0]], dtype=cp.float32)
    target_lon = cp.asarray([[0.1, 8.0, 4.4]], dtype=cp.float32)
    indices = cp.asarray([[0, -1, -1]], dtype=cp.int32)   # first covered within one degree
    farthest, count = _borrow_beyond_one_degree(indices, target_lat, target_lon, lat, lon, valid)
    assert count == 2
    assert indices.get().tolist() == [[0, 12, 12]]
    assert 550 < farthest < 560   # (0, 8) -> (0, 3): 5 degrees of the equator
    untouched = cp.asarray([[0, -2]], dtype=cp.int32)
    assert _borrow_beyond_one_degree(untouched, target_lat[:, :2], target_lon[:, :2],
                                     lat, lon, valid) is None


@requires_gpu
@pytest.mark.gpu
def test_the_woa18_profile_stops_its_lapse_rate_at_sea_water_freezing():
    # Cold (275 K) water under a 10 m WOA18 MLD fell below 260 K at 150 m with
    # the 0.14 K/m default lapse rate and the forecast refused its ocean
    # (full-CONUS HRRR 48 h on the fold, test host 2026-10-21).
    import cupy as cp
    import numpy as np
    from woof.ocean.mixed_layer import kernels
    levels, n = 30, 3
    mld = cp.asarray([10.0], dtype=cp.float32)
    index = cp.zeros(n, dtype=cp.int32)
    sst = cp.asarray([275.0, 300.0, 270.0], dtype=cp.float32)
    out = {name: cp.empty((levels, n), dtype=cp.float32) for name in ('t', 's', 'u', 'v', 'z')}
    kernels(levels).get_function('ocean_mld_profiles')(((n*levels+127)//128,), (128,),
        (mld, index, sst, out['t'], out['s'], out['u'], out['v'], out['z'], np.float32(0.14), np.int32(n)))
    t = out['t'].get()
    assert t[:, 0].min() == np.float32(271.25)          # clamped at freezing
    assert abs(float(t[-1, 1]) - (300.0 - 0.14 * 140.0)) < 1e-3   # warm column: lapse rate as before
    assert (t[:, 2] == np.float32(270.0)).all()        # already below freezing: holds its SST


def _floor_tendency(cp, sign):
    """Temperature tendency of one interior column under a uniform divergent (sign=1) or convergent (-1) current."""
    from woof.ocean.mixed_layer import kernels
    levels, ny, nx = 4, 8, 8
    n = nx*ny
    shape = (levels, ny, nx)
    z = cp.broadcast_to(cp.asarray([5., 10., 15., 20.], cp.float32)[:, None, None], shape).copy()
    t = cp.broadcast_to(cp.asarray([290., 289., 288., 287.], cp.float32)[:, None, None], shape).copy()
    s = cp.full(shape, 35., cp.float32)
    u = cp.broadcast_to(sign*0.05*(cp.arange(nx, dtype=cp.float32)-3.5)[None, None, :], shape).copy()
    v = cp.zeros(shape, cp.float32)
    land = cp.full((ny, nx), 2., cp.float32)
    ones = cp.ones((ny, nx), cp.float32)
    w, p = cp.zeros(shape, cp.float32), cp.zeros(shape, cp.float32)
    tend = cp.zeros((4, *shape), cp.float32)
    module = kernels(levels)
    rd = np.float32(1/250.)
    module.get_function('pwp_geometry')(((n+127)//128,), (128,),
        (t, s, u, v, z, t, s, land, ones, ones, w, p, rd, rd, np.int32(nx), np.int32(ny)))
    module.get_function('pwp_transport')(((n*levels+127)//128,), (128,),
        (t, s, u, v, z, w, p, land, ones, ones, tend, rd, rd, np.int32(nx), np.int32(ny), np.int32(1)))
    return cp.asnumpy(w[:, 4, 4]), cp.asnumpy(tend[0, :, 4, 4])


@requires_gpu
@pytest.mark.gpu
def test_upwelling_through_the_floor_does_not_cool_the_bottom_layer_without_bound():
    # SF Bay 250 m nest (HRRR 2026-10-21T12): WRF's one-sided floor difference is downwind for
    # upwelling, so bottom water ran away to 246-308 K and convection carried it to TSK.
    import cupy as cp
    w, tend = _floor_tendency(cp, 1.)
    assert w[-1] > 0                      # divergent surface current: water rises through the floor
    assert tend[-1] == 0.                 # it carries the bottom layer's own temperature
    assert tend[-2] < 0.                  # interior upwelling still cools (centred, unchanged)
    w, tend = _floor_tendency(cp, -1.)
    assert w[-1] < 0                      # convergence: water sinks onto the floor from above
    assert tend[-1] > 0.                  # upwind from the warmer layer above, as before
@requires_gpu
@pytest.mark.gpu
def test_column_momentum_matches_atmospheric_stress_and_true_layer_thickness():
    import cupy as cp
    a=columns(cp,4)
    a['om_tmp'].fill(300);a['om_tini'].fill(300)
    a['glw'].fill(5.67051e-8*300**4)
    a['ust'].fill(2);a['uair'].fill(40);a['rhoair'].fill(1.2)
    step(a,dt=30,dx=10000,dy=10000);check(a)
    momentum=cp.sum(a['om_u']*5,axis=0)
    cp.testing.assert_allclose(momentum[:,:-1],1.2*2**2*30/1024,rtol=2e-5)


@requires_gpu
@pytest.mark.gpu
def test_analysis_layer_integrals_across_source_interfaces():
    import cupy as cp
    from woof.ocean.mixed_layer import kernels
    # A linear profile has analytic layer means even where source intervals
    # straddle multiple target layers. The surface extrapolation is constant.
    depth=cp.asarray([0,3,8,20],dtype='f4')
    values=2*depth+10
    out=cp.empty((4,1),dtype='f4');z=cp.empty_like(out)
    kernels(4).get_function('ocean_profiles')((1,),(128,),
        (values,depth,cp.asarray([0],dtype='i4'),out,z,
         np.int32(1),np.int32(1),np.int32(4),np.int32(0),np.float32(5)))
    cp.testing.assert_allclose(out[:,0],cp.asarray([15,25,35,45],dtype='f4'))
    cp.testing.assert_array_equal(z[:,0],cp.asarray([5,10,15,20],dtype='f4'))


@requires_gpu
@pytest.mark.gpu
def test_uncovered_analysis_cells_keep_prescribed_sst_and_exclude_transport():
    import cupy as cp
    from woof.ocean.mixed_layer import ocean_land
    a=columns(cp,4)
    initial={name:a[name].copy() for name in STATE_FIELDS}
    initial['om_active'][:,3:5]=0
    initial['om_tmp'][:,:,3:5]=0
    cfg=run_config(sf_ocean_physics=2,ocean_levels=4)
    initialize(a,cfg,initial)
    a['hfx'].fill(200.);a['lh'].fill(500.);a['ust'].fill(1.)
    before=a['tsk'].copy()
    step(a,dt=30.,dx=10000.,dy=10000.);check(a)
    cp.testing.assert_array_equal(a['tsk'][:,3:5],before[:,3:5])
    assert bool(cp.all(a['_pwp_land'][:,3:5]==1))
    assert bool(cp.all(a['tsk'][:,:3]<before[:,:3]))

@requires_gpu
@pytest.mark.gpu
def test_dated_analysis_rust_decode_to_layer_means(tmp_path):
    """Exercise the complete decoder path, including scalar depth searches."""
    import cupy as cp
    from netCDF4 import Dataset
    from woof.ocean.mixed_layer.analysis import load
    p = tmp_path / 'analysis.nc'
    with Dataset(p, 'w') as d:
        for name, values, units in [('time', [0], 'hours since 2022-09-26 00:00:00'),
                                    ('depth', [0, 50, 100, 200], 'm'),
                                    ('lat', [0, .1], 'degrees_north'),
                                    ('lon', [0, .1], 'degrees_east')]:
            d.createDimension(name, len(values))
            v = d.createVariable(name, 'f4', (name,)); v[:] = values; v.units = units
        for name, units, value in [('water_temp', 'degC', 30), ('salinity', 'psu', 35),
                                   ('water_u', 'm/s', 0), ('water_v', 'm/s', 0)]:
            v = d.createVariable(name, 'f4', ('time', 'depth', 'lat', 'lon'))
            v.units = units
            v[:] = (np.array([30, 25, 20, 10])[None, :, None, None]
                    if name == 'water_temp' else value)
    land = cp.full((1, 2), 2, cp.float32)
    a = load(p, latitude=cp.asarray([[0, 80]], cp.float32),
             longitude=cp.zeros_like(land), land=land, levels=2, layer_m=100,
             source='hycom', rotation=(cp.zeros_like(land), cp.ones_like(land)),
             expected_time='2022-09-26T00:00:00', uncovered='prescribed')
    np.testing.assert_array_equal(cp.asnumpy(a['om_active']), [[1, 0]])
    cp.testing.assert_allclose(a['om_tmp'][:, 0, 0], cp.asarray([298.15, 288.15]), atol=4e-5)
