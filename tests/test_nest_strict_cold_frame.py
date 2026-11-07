"""Cold strict mass and history presentation at bounded FORCE boundaries."""
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import requires_gpu


def _state():
    import cupy as cp
    ny, nx, nz = 5, 6, 3
    def field(shape, offset):
        return cp.arange(np.prod(shape), dtype=cp.float32).reshape(shape) * cp.float32(.03125) + cp.float32(offset)
    state = SimpleNamespace(
        mup=field((ny,nx),123.456), mub2d=field((ny,nx),90000.125),
        u=field((nz,ny,nx+1),4.1), v=field((nz,ny+1,nx),-3.3),
        w=field((nz+1,ny,nx),.003), php=field((nz+1,ny,nx),1.2),
        thp=field((nz,ny,nx),-5.3), has_msf=True,
        msft=field((ny,nx),1.1), msfu=field((ny,nx+1),1.2),
        msfv=field((ny+1,nx),1.3),
        c1h=cp.asarray([.9,.5,.1],dtype=cp.float32),
        c2h=cp.asarray([1000.,30000.,60000.],dtype=cp.float32),
        c1f=cp.asarray([1.,.7,.3,0.],dtype=cp.float32),
        c2f=cp.asarray([0.,20000.,40000.,80000.],dtype=cp.float32))
    halo=cp.full_like(state.mup,-9999.)
    state.existing_scratch=lambda _slot:halo
    state._wrf_mass_carriers_ready=False
    return state


def _words_equal(a,b):
    import cupy as cp
    np.testing.assert_array_equal(cp.asnumpy(a).view(np.uint32),cp.asnumpy(b).view(np.uint32))


@requires_gpu
@pytest.mark.gpu
def test_cold_primed_halo_cannot_change_force_round_trip():
    from woof.core.nest_round_trip import couple_round_trip,round_trip_targets,couple_halo_faces
    run=SimpleNamespace(specified=True,nested=False,moist=False)
    state=_state()
    expected={k:v.copy() for k,v in round_trip_targets(state,run).items()}
    actual={k:v.copy() for k,v in expected.items()}
    halo=state.existing_scratch
    state.existing_scratch=lambda _slot:None
    couple_round_trip(state,run,expected)
    state.existing_scratch=halo
    couple_round_trip(state,run,actual)
    for k in expected:_words_equal(actual[k],expected[k])
    for kind in ('u','v'):
        out=getattr(state,kind).copy();before=out.copy()
        couple_halo_faces(state,kind,out)
        _words_equal(out,before)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize('ready',[False,True])
def test_bounded_frame_matches_resident_round_trip_without_mutating_store(ready):
    import cupy as cp
    from woof.core.nest_round_trip import couple_round_trip,round_trip_targets
    from woof.core.nest_operands import NestWindowSource,round_tripped_canonical_frame
    state=_state();state._wrf_mass_carriers_ready=ready
    run=SimpleNamespace(specified=True,nested=False,moist=False)
    targets=round_trip_targets(state,run)
    expected={k:v.copy() for k,v in targets.items()};couple_round_trip(state,run,expected)
    geography={f'setup/{k}':cp.asnumpy(v) for k,v in vars(state).items() if isinstance(v,cp.ndarray) and k not in targets and k!='mup'}
    store={f'state/{k}':cp.asnumpy(v) for k,v in targets.items()}
    store['state/mup']=cp.asnumpy(state.mup)
    store['scratch/wrf_exact_mu_halo']=cp.asnumpy(state.existing_scratch(None))
    before={k:v.copy() for k,v in store.items()}
    metadata=SimpleNamespace(c1h=state.c1h,c2h=state.c2h,c1f=state.c1f,c2f=state.c2f)
    owner=SimpleNamespace(store=store,_geography=geography,template_state=metadata,scalars={'wrf_mass_carriers_ready':ready})
    source=NestWindowSource(SimpleNamespace(_streamed_domain=owner))
    names={'U':'u','V':'v','W':'w','PH':'php','THM':'thp'}
    fields={name:store[f'state/{attr}'] for name,attr in names.items()}
    fields['T']=fields['THM'];fields['MU']=store['state/mup']
    plan=SimpleNamespace(origin={name:f'state/{attr}' for name,attr in names.items()})
    result=round_tripped_canonical_frame(source,run,fields,plan,chunk_shape=(2,3),destinations={})
    for name,attr in names.items():_words_equal(result[name],expected[attr])
    assert result['T'] is fields['T'] and result['MU'] is fields['MU']
    for k in store:np.testing.assert_array_equal(store[k].view(np.uint32),before[k].view(np.uint32))


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize('ready', [False, True])
@pytest.mark.parametrize('coupled', [False, True])
def test_bounded_heating_frame_matches_resident_and_preserves_carriers(ready, coupled):
    import cupy as cp
    from woof.core.physics import PhysicsDriver
    from woof.core.nest_operands import NestWindowSource, radiation_canonical_frame
    state = _state()
    state._wrf_mass_carriers_ready = ready
    muts = state.mub2d + state.mup + cp.float32(8.125)
    state.existing_scratch = lambda _slot: muts
    state.total_mu = lambda: state.mub2d + state.mup
    heating = cp.arange(state.thp.size, dtype=cp.float32).reshape(state.thp.shape) * cp.float32(.00013)
    driver = SimpleNamespace(state=state, _wrf_rthraten=heating, _wrf_rthraten_coupled=coupled)
    state.physics = driver
    expected = PhysicsDriver.radiation_theta_forcing(driver)
    store = {'state/mup': cp.asnumpy(state.mup),
             'diag/_wrf_rthraten': cp.asnumpy(heating),
             'diag/rthraten': np.full(heating.shape, -9.5, dtype=np.float32),
             'scratch/wrf_exact_muts': cp.asnumpy(muts)}
    geography = {'setup/mub2d': cp.asnumpy(state.mub2d)}
    owner = SimpleNamespace(store=store, _geography=geography, template_state=state,
                            scalars={'wrf_mass_carriers_ready': ready,
                                     'wrf_rthraten_coupled': coupled})
    source = NestWindowSource(SimpleNamespace(_streamed_domain=owner))
    before = {k: v.copy() for k, v in store.items()}
    fields = {'RTHRATEN': store['diag/rthraten'], 'MU': store['state/mup']}
    result = radiation_canonical_frame(source, fields, chunk_shape=(2, 3), destinations={})
    _words_equal(result['RTHRATEN'], expected)
    assert result['MU'] is fields['MU']
    for key in store:
        np.testing.assert_array_equal(store[key].view(np.uint32), before[key].view(np.uint32))


@pytest.mark.parametrize('coupled', [False, True])
def test_heating_coupling_flag_survives_scalar_checkpoint(coupled):
    from tilestream.checkpoint import driver_restart_header, restart_scalars
    from tilestream.physics_inventory import set_carrier_scalars
    values = {'elapsed_seconds': 18., 'call_counts': {}, 'ysu_nan_guard_fires': 0,
              'microphysics_updates': 1, 'wrf_rthraten_coupled': coupled}
    header = {'elapsed_seconds': 18., 'driver': driver_restart_header(values)}
    driver = SimpleNamespace(call_counts={}, _wrf_rthraten_coupled=not coupled)
    state = SimpleNamespace(physics=driver)
    set_carrier_scalars(state, restart_scalars(header))
    assert driver._wrf_rthraten_coupled is coupled


@requires_gpu
@pytest.mark.gpu
def test_tile_history_t_keeps_solve_end_before_feedback():
    import cupy as cp
    from woof.core.streaming import refl_handoff_hook, diagnostic_inventory
    from tilestream.physics_inventory import checkpointed_carriers
    state = _state()
    saved = cp.empty_like(state.thp)
    state.existing_scratch = lambda slot: saved if slot == 'wrf_history_t' else None
    expected = state.thp.copy()
    refl_handoff_hook()(state, None, 0, cp.cuda.Stream.null)
    # Parent feedback changes live t_2 after the solve-end snapshot.
    state.thp += cp.float32(.125)
    captured = diagnostic_inventory(lambda *_: {})(state)
    _words_equal(captured['scratch/wrf_history_t'], expected)
    assert not cp.array_equal(saved, state.thp)
    assert 'scratch/wrf_history_t' not in checkpointed_carriers(captured)


def test_canonical_digest_keeps_checkpointed_mass_and_omits_history_snapshot():
    from woof.state_digest import canonical_store_digest
    clock = SimpleNamespace(dtbc_fp32=np.float32(18.))
    arrays = {'scratch/wrf_exact_muts': np.arange(6, dtype=np.float32).reshape(2, 3),
              'scratch/wrf_exact_mu_halo': np.ones((2, 3), dtype=np.float32),
              'scratch/wrf_history_t': np.zeros((4, 2, 3), dtype=np.float32)}
    first = canonical_store_digest(arrays, {'elapsed_seconds': 18.}, clock)
    assert set(first['field_order']) == {'scratch/wrf_exact_muts', 'scratch/wrf_exact_mu_halo'}
    arrays['scratch/wrf_history_t'].fill(2.)
    assert canonical_store_digest(arrays, {'elapsed_seconds': 18.}, clock) == first
    arrays['scratch/wrf_exact_muts'][0, 0] = np.float32(.125)
    assert canonical_store_digest(arrays, {'elapsed_seconds': 18.}, clock)['sha256'] != first['sha256']
