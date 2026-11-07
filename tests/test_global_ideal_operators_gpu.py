"""Global ideal building blocks: geometry, isolated streams and capture."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu


def _initial(nx=32, ny=18, nz=12, hybrid=0):
    from woof.core.global_ideal_init import initialize
    # A deterministic device input exercises the explicit native RNG contract.
    samples = cp.full((nz, ny, nx), .625, cp.float32)
    return initialize(nx, ny, nz, hybrid_opt=hybrid, random_samples=samples)


@pytest.mark.parametrize("hybrid", (0, 1, 2, 3))
def test_initializer_geometry_and_defined_skin(hybrid):
    fields = _initial(hybrid=hybrid)
    for name, value in fields.items():
        assert bool(cp.isfinite(value).all()), name
    assert bool((fields['xlat'][1:] > fields['xlat'][:-1]).all())
    assert float(fields['xlat'][0, 0]) == -85
    assert float(fields['xlat'][-1, 0]) == 85
    assert bool((fields['msfvx_inv'][0] == 0).all())
    assert bool((fields['msfvx_inv'][-1] == 0).all())
    assert bool((fields['msfty'] == 1).all())
    assert bool((fields['msfvy'] == 1).all())
    assert bool((fields['pb'][:-1] > fields['pb'][1:]).all())
    assert bool((fields['phb'][1:] > fields['phb'][:-1]).all())
    assert bool((fields['alb'] > 0).all())
    # Use the defined grid weights; native local cf1/2/3 are uninitialized.
    assert bool(((fields['tsk'] > 299) & (fields['tsk'] < 301)).all())
    assert bool((fields['tmn'] == fields['tsk']-np.float32(.5)).all())


def _hydro(fields):
    from woof.core.global_hydrostatic import HydrostaticDiagnostics
    nz, ny, nx = fields['t_2'].shape
    profiles = cp.stack([fields[n] for n in
                         ('c1h', 'c2h', 'c3h', 'c4h', 'c3f', 'c4f', 'dnw', 'rdnw', 'rdn')])
    constants = cp.asarray((100000, 300, .11537721008062363, 287,
                            1.6083624362945557, -.7142857313156128), cp.float32)
    operator = HydrostaticDiagnostics(nx, ny, nz, constants=constants, profiles=profiles)
    data = dict(mu=fields['mu_2'], muts=fields['mu_2']+fields['mub'],
                alb=fields['alb'], pb=fields['pb'], t=fields['t_2'],
                phb=fields['phb'], moist=cp.empty((0, nz, ny, nx), cp.float32),
                p=cp.empty_like(fields['pb']), al=cp.empty_like(fields['pb']),
                ph=fields['ph_2'].copy(), full_pressure=cp.empty_like(fields['pb']),
                alt=cp.empty_like(fields['pb']))
    return operator, data


def test_hydrostatic_window_preserves_other_columns():
    fields = _initial()
    operator, full = _hydro(fields)
    operator.apply(**full)
    operator, window = _hydro(fields)
    for n in ('p', 'al', 'full_pressure', 'alt'):
        window[n].fill(np.float32(-12345))
    operator.apply(**window, window=(4, 5, 7, 8))
    for n in ('p', 'al', 'full_pressure', 'alt'):
        assert bool((window[n][:, 4:11, 5:13].view(cp.uint32)
                     == full[n][:, 4:11, 5:13].view(cp.uint32)).all())
        assert bool((window[n][:, :4] == -12345).all())
    assert bool((window['ph'][:, :4].view(cp.uint32)
                 == fields['ph_2'][:, :4].view(cp.uint32)).all())


def test_hydrostatic_and_fft_are_graph_capturable_and_stream_private():
    from woof.core.global_polar import PolarFFT
    fields = _initial()
    source = fields['t_2']
    serial = source.copy()
    reference = PolarFFT(32, 18, 12, latitude=fields['xlat'])
    reference.apply(serial)
    expected_hydro, expected = _hydro(fields)
    expected_hydro.apply(**expected)
    cp.cuda.get_current_stream().synchronize()
    plans = []
    for _ in range(2):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            operator = PolarFFT(32, 18, 12, latitude=fields['xlat'])
            field = source.copy()
            hydro, data = _hydro(fields)
            # Resolve lazy CUDA-library initialization before graph capture.
            operator.apply(field)
            field[...] = source
            hydro.apply(**data)
            stream.synchronize()
            stream.begin_capture()
            operator.apply(field)
            hydro.apply(**data)
            graph = stream.end_capture()
        plans.append((stream, operator, field, hydro, data, graph))
    assert plans[0][1].work.data.ptr != plans[1][1].work.data.ptr
    for stream, _, _, _, _, graph in plans:
        graph.launch(stream)
    for stream, _, field, _, data, _ in plans:
        stream.synchronize()
        assert bool((field.view(cp.uint32) == serial.view(cp.uint32)).all())
        for name in ('p', 'al', 'ph', 'full_pressure', 'alt'):
            assert bool((data[name].view(cp.uint32) == expected[name].view(cp.uint32)).all()), name


def test_fft_retains_unfiltered_rows_and_constant_modes():
    from woof.core.global_polar import PolarFFT
    fields = _initial()
    # All longitudes are identical; filtering cannot change this exact DC mode.
    value = cp.full((12, 18, 32), np.float32(17.5))
    operator = PolarFFT(32, 18, 12, latitude=fields['xlat'], last_row=15)
    operator.apply(value)
    assert bool((value == np.float32(17.5)).all())
    source = cp.arange(value.size, dtype=cp.float32).reshape(value.shape)
    value[...] = source
    operator.apply(value)
    unfiltered = (cp.abs(fields['xlat'][:, 0]) < 45) | (cp.arange(18) > 15)
    assert bool((value[:, unfiltered].view(cp.uint32)
                 == source[:, unfiltered].view(cp.uint32)).all())


def test_pole_faces_copy_and_zero_preserve_the_interior():
    from woof.core.global_polar import PoleFaces
    source = cp.arange(12*19*32, dtype=cp.float32).reshape((12, 19, 32))
    field = source.copy()
    operator = PoleFaces(32, 18, 12)
    operator.apply(field, copy_neighbor=True)
    assert bool((field[:, 0] == source[:, 1]).all())
    assert bool((field[:, -1] == source[:, -2]).all())
    assert bool((field[:, 1:-1] == source[:, 1:-1]).all())
    operator.apply(field)
    assert bool((field[:, 0] == 0).all())
    assert bool((field[:, -1] == 0).all())
    assert bool((field[:, 1:-1] == source[:, 1:-1]).all())


def test_hydrostatic_acoustic_and_w_capture_without_shared_workspaces():
    from woof.core.global_hydrostatic import HydrostaticAcoustic, HydrostaticVerticalVelocity
    nz, ny, nx = 12, 18, 32
    fields = _initial(nx, ny, nz)
    acoustic = HydrostaticAcoustic(nx, ny, nz, profiles=cp.stack([
        fields[n][:nz] for n in ('c1h', 'c2h', 'c3h', 'dnw')]))
    arrays = dict(mu=cp.full((ny, nx), .125, cp.float32), mut=fields['mub'],
                  alt=fields['alb'], theta_coupled=cp.full((nz, ny, nx), .25, cp.float32),
                  theta_ref=fields['t_2'], c2a=cp.full((nz, ny, nx), 100, cp.float32),
                  p=cp.zeros((nz, ny, nx), cp.float32), al=cp.zeros((nz, ny, nx), cp.float32),
                  ph=cp.zeros((nz+1, ny, nx), cp.float32), pm1=cp.zeros((nz, ny, nx), cp.float32))
    constants = cp.asarray((9.81, 300, 1.5, -.5, 0, 1/30000, 1/30000), cp.float32)
    w_operator = HydrostaticVerticalVelocity(nx, ny, nz, constants=constants,
        profiles=cp.stack([fields[n] for n in ('c1f', 'c2f')]))
    w_arrays = dict(mut=fields['mub'], msftx=fields['msftx'], msfty=fields['msfty'],
                    height_halo=cp.zeros((ny+2, nx+2), cp.float32),
                    u=cp.zeros((nz, ny, nx+1), cp.float32),
                    v=cp.zeros((nz, ny+1, nx), cp.float32),
                    ph_tend=cp.zeros((nz+1, ny, nx), cp.float32),
                    ph_new=arrays['ph'], ph_old=cp.zeros_like(arrays['ph']),
                    w=cp.zeros_like(arrays['ph']))
    acoustic.apply(**arrays, first=True)
    w_operator.apply(**w_arrays)
    expected = {n: arrays[n].copy() for n in ('p', 'al', 'ph', 'pm1')}
    expected_w = w_arrays['w'].copy()
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        acoustic.apply(**arrays, first=True)
        w_operator.apply(**w_arrays)
        graph = stream.end_capture()
    graph.launch(stream)
    stream.synchronize()
    for n, value in expected.items():
        assert bool((arrays[n].view(cp.uint32) == value.view(cp.uint32)).all()), n
    assert bool((w_arrays['w'].view(cp.uint32) == expected_w.view(cp.uint32)).all())


def _domain(member=0):
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.core.state import DomainState
    from woof.core.global_state import bind,build_stepper
    raw=_initial()
    raw['t_2']+=cp.float32(member*.125)
    scalars=cp.asnumpy(raw['scalars'])
    cfg=RunConfig(nx=32,ny=18,nz=12,dx=float(scalars[1]),dy=float(scalars[2]),
                  ztop=120000,dt=300,run_seconds=1200,polar=True,
                  ra_lw_physics=31,ra_sw_physics=0,diff_opt=1)
    state=bind(DomainState(cfg),cfg,raw)
    build_stepper(state,cfg)
    return state,cfg


def test_global_graph_checkpoint_continuation_and_geometry_identity(tmp_path):
    from woof.core.dycore import step
    from woof.io.restart import (write_restart,restore_restart,RestartMismatchError,
                                   state_manifest)
    state,cfg=_domain()
    for _ in range(2):step(state,cfg)
    checkpoint=write_restart(tmp_path/'checkpoint.npz',state,cfg)
    resumed,resumed_cfg=_domain()
    restore_restart(checkpoint,resumed,resumed_cfg)
    # Captured graph owns stable pointers; restore updates them in place.
    for _ in range(2):step(state,cfg);step(resumed,resumed_cfg)
    assert state.elapsed_seconds==resumed.elapsed_seconds==1200
    expected,actual=state_manifest(state),state_manifest(resumed)
    assert expected.keys()==actual.keys()
    for key in expected:
        assert bool((expected[key].view(cp.uint32)==actual[key].view(cp.uint32)).all()),key
    changed,changed_cfg=_domain()
    changed.global_tx[2,3]+=cp.float32(.001)
    with pytest.raises(RestartMismatchError,match='fingerprint'):
        restore_restart(checkpoint,changed,changed_cfg)
    # The inactive regional defaults may be omitted, but active polar
    # settings must still bind the trajectory when a checkpoint resumes.
    from dataclasses import replace
    with pytest.raises(RestartMismatchError,match='fft_filter_lat'):
        restore_restart(checkpoint,resumed,replace(resumed_cfg,fft_filter_lat=46.))


def test_full_global_graphs_are_private_on_concurrent_member_streams():
    from woof.core.dycore import step
    from woof.io.restart import state_manifest
    references=[]
    for member in range(2):
        state,cfg=_domain(member)
        for _ in range(3):step(state,cfg)
        references.append({k:v.copy() for k,v in state_manifest(state).items()})
    members=[(*_domain(member),cp.cuda.Stream(non_blocking=True))for member in range(2)]
    ready=cp.cuda.Event();ready.record()
    for state,cfg,stream in members:
        stream.wait_event(ready)
        with stream:
            for _ in range(3):step(state,cfg)
    assert members[0][0]._global_stepper.filters['scalar'].work.data.ptr!=members[1][0]._global_stepper.filters['scalar'].work.data.ptr
    for member,(state,cfg,stream)in enumerate(members):
        stream.synchronize()
        for key,value in state_manifest(state).items():
            assert bool((value.view(cp.uint32)==references[member][key].view(cp.uint32)).all()),key
