"""CUDA wiring fixtures; these tests are not native-run proof."""
from datetime import datetime
from types import SimpleNamespace
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('WOOF_DOWNSCALE_GPU_TEST') != '1', reason='explicit GPU test opt-in')


def fixture(monkeypatch, tmp_path, *, zero=0, upper=False, dimensions=(5, 7, 9), fasdas=False):
    import cupy as cp
    from woof.spectral_nudging.config import NudgingConfig
    from woof.spectral_nudging.surface_forecast import SurfaceForecast
    import woof.netcdf_bridge as bridge
    import woof.spectral_nudging.surface_inputs as inputs
    nz, ny, nx = dimensions
    def full(shape, value): return cp.full(shape, value, cp.float32)
    def state(elapsed=0.):
        mass = (nz, ny, nx)
        return SimpleNamespace(u=full((nz, ny, nx+1), 2), v=full((nz, ny+1, nx), 1),
            thp=full(mass, 0), thb=full((nz,), 300), p=full(mass, 90000), qv=full(mass, .005),
            php=full((nz+1, ny, nx), 0), phb=cp.arange(nz+1, dtype=cp.float32)*981,
            ht=full((ny, nx), 0), mup=full((ny, nx), 0), mub2d=full((ny, nx), 90000),
            c1h=full((nz,), 1), c2h=full((nz,), 0), msfu=full((ny, nx+1), 1),
            msfv=full((ny+1, nx), 1), msft=full((ny, nx), 1), has_msf=True,
            physics=SimpleNamespace(fields={'pblh': full((ny, nx), 120),
                'znt': full((ny, nx), .1), 'regime': full((ny, nx), 4),
                'xland':full((ny,nx),1), 'hfx':full((ny,nx),30),
                'qfx':full((ny,nx),.0001), 'cpm':full((ny,nx),1004.5)}), elapsed_seconds=elapsed)
    data = {}
    for end in ('OLD', 'NEW'):
        for name, value in (('U10', 4), ('V10', 3), ('TH2', 302), ('Q2', .006), ('ODIS', 20)):
            shape = (ny, nx+1) if name == 'U10' else ((ny+1, nx) if name == 'V10' else (ny, nx))
            data[f'{name}_NDG_{end}'] = np.full(shape, value, np.float32)
    monkeypatch.setattr(inputs, 'read_native_surface', lambda *a, **k: SimpleNamespace(fields=data))
    class Dataset:
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def stored_words(self, names, record=None):
            if names == ('Times',):
                return {'Times': np.array([list('2026-10-16_09:00:00')], dtype='S1')}
            rows = {}
            for name in names:
                shape = (nz, ny, nx+1) if name.startswith('U_') else ((nz, ny+1, nx) if name.startswith('V_') else (nz, ny, nx))
                rows[name] = np.full((1, *shape), {'U': 4., 'V': 3., 'T': 2., 'Q': .006}[name[0]], np.float32)
            return rows
    monkeypatch.setattr(bridge, 'Dataset', Dataset)
    path = tmp_path/'forcing'; path.write_bytes(b'constructed CUDA wiring fixture')
    cfg = SimpleNamespace(nx=nx, ny=ny, nz=nz, dx=3000., dt=15., specified=True,
        nested=False, sf_sfclay_physics=91, use_adaptive_time_step=False)
    controls = NudgingConfig(enabled=True, mode='grid', convention='wrf', grid_sfdda=2 if fasdas else 1,
        sgfdda_interval_m=180, sgfdda_end_h=6, rinblw=250,
        guv_sfc=.0003, gt_sfc=.0003, gq_sfc=.0003, guv=.0003 if upper else 0.,
        gt=.0003 if upper else 0., gq=.0003 if upper else 0.,
        gfdda_interval_m=180, gfdda_end_h=6, fgdt=.75, fgdtzero=zero)
    def owner(s):
        return SurfaceForecast(s, cfg, controls, path, datetime(2026, 10, 16, 9), 180,
                               upper_path=path if upper else None)
    return state, owner


@pytest.mark.parametrize('zero', [0, 1])
@pytest.mark.parametrize('fasdas', [False, True])
def test_restart_between_cadence_calls_and_member_isolation(monkeypatch, tmp_path, zero, fasdas):
    import cupy as cp
    from woof.spectral_nudging.surface_forecast import HELD, FASDAS_HELD
    carried=(*HELD, "hfx_fdda", *(FASDAS_HELD if fasdas else ()))
    create, attach = fixture(monkeypatch, tmp_path, zero=zero, fasdas=fasdas)
    state = create(); live = attach(state)
    member = create(); independent = attach(member)
    for _ in range(2): live.fold(None)
    cp.cuda.get_current_stream().synchronize()
    assert np.any(state.surface_fdda_theta.get() != 0.)
    resumed = create(30.)
    for name in carried: setattr(resumed, name, getattr(state, name).copy())
    continued = attach(resumed)
    before = member.surface_fdda_theta.get()
    for _ in range(6):
        a = live.fold(None); b = continued.fold(None)
        cp.cuda.get_current_stream().synchronize()
        for name in ('ru', 'rv', 'rtheta', 'rqv'):
            np.testing.assert_array_equal(getattr(a, name).view(cp.uint32).get(), getattr(b, name).view(cp.uint32).get())
        for name in carried:
            np.testing.assert_array_equal(getattr(state, name).view(cp.uint32).get(), getattr(resumed, name).view(cp.uint32).get())
    np.testing.assert_array_equal(member.surface_fdda_theta.get(), before)
    assert live.clock.get()[0] == continued.clock.get()[0] == 120.
    assert independent.clock.get()[0] == 0.
    if zero:
        np.testing.assert_array_equal(state.surface_fdda_theta.get(), 0.)
    else:
        assert np.any(state.surface_fdda_theta.get() != 0.)


def test_stock_upper_and_surface_share_native_mass_coupling(monkeypatch, tmp_path):
    import cupy as cp
    create, attach = fixture(monkeypatch, tmp_path, upper=True)
    state = create(); owner = attach(state); rates = owner.fold(None)
    cp.cuda.get_current_stream().synchronize()
    # Above the PBL surface weight is zero. The upper target gives
    # 0.0003*(302-300)*90000 = 54 in mass-coupled units.
    np.testing.assert_allclose(rates.rtheta.get()[4, 3, 3], 54., rtol=2.e-6)
    assert owner.receipt()['native_upper']['records'] == 1
    assert np.isfinite(rates.rtheta.get()).all()


@pytest.mark.parametrize('feedback', [0, 1])
@pytest.mark.parametrize('fasdas', [False, True])
def test_real_nested_steps_keep_domain_clocks_and_surface_carriers(monkeypatch, tmp_path, feedback, fasdas):
    import cupy as cp
    from dataclasses import replace
    from tests.test_feedback import _feedback_experiment, _feedback_model
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    from woof.core.model import execute_experiment
    from woof.spectral_nudging.surface_forecast import SurfaceForecast
    exp = _feedback_experiment(12., feedback=feedback)
    domains = tuple(replace(d, run=replace(d.run, nz=4, dt=3. if d.parent_id == 0 else 1.,
        ztop=8000., spec_bdy_width=3, spec_zone=1, relax_zone=2, sf_sfclay_physics=91, sf_surface_physics=2 if fasdas else 0, bl_pbl_physics=0),
        time_step=3 if d.parent_id == 0 else None) for d in exp.domains)
    exp = replace(exp, domains=domains, restart_interval_s=0.)
    model = _feedback_model(exp)
    owners = []
    for node in model.walk_parent_first():
        cfg = node.cfg.run
        coord = make_vertical_coord(cfg.nz)
        base = make_base_state(coord, lambda z: 300.+.003*np.asarray(z), p_surf=cfg.p_surf, ztop=cfg.ztop)
        state = init_moist_balanced(cfg, coord, base, lambda z: .006*np.exp(-np.asarray(z)/2000.))
        state._nest_restart_classification = 'REBUILT'
        initialize_physics(state, cfg, pblh=700., glw=300., swdown=600.,
            landmask=np.ones((cfg.ny,cfg.nx),np.float32), tsk=290.,
            ivgtyp=7, isltyp=6, vegfra=55., tmn=287.)
        node.state = state
        _, attach = fixture(monkeypatch, tmp_path, dimensions=(cfg.nz, cfg.ny, cfg.nx))
        # Bind the common fixture analyses to the actual domain geometry,
        # time step and original experiment epoch.
        from woof.spectral_nudging.config import NudgingConfig
        options = NudgingConfig(enabled=True, mode='grid', convention='wrf', grid_sfdda=2 if fasdas else 1,
            sgfdda_interval_m=180, sgfdda_end_h=6, rinblw=250., guv=0., gt=0., gq=0.,
            guv_sfc=.0003, gt_sfc=.0003, gq_sfc=.0003)
        owner = SurfaceForecast(state, replace(cfg, specified=cfg.specified or not cfg.nested), options,
            tmp_path/'forcing', datetime(2026,10,16,9), exp.run_seconds)
        owners.append(owner)
    report = execute_experiment(model, experiment=exp, pool_trim_per_period=False)
    cp.cuda.get_current_stream().synchronize()
    assert report.steps == 16
    assert [o.steps for o in owners] == [4, 12]
    assert [o.clock.get()[0] for o in owners] == [12., 12.]
    for node in model.walk_parent_first():
        assert np.isfinite(node.state.thp.get()).all()
        assert np.any(node.state.surface_fdda_theta.get() != 0.)
    assert owners[0].state.surface_fdda_theta.data.ptr != owners[1].state.surface_fdda_theta.data.ptr


def test_fasdas_physical_noah_inputs_and_qnorm(monkeypatch,tmp_path):
    import cupy as cp
    create,attach=fixture(monkeypatch,tmp_path,fasdas=True)
    state=create();owner=attach(state)
    owner.fold(None)
    owner.update_qnorm(cp.full_like(state.mup,1.e-6),cp.full_like(state.mup,.005))
    want=np.float32(np.float32(np.float32(np.float32(1.e-6)*np.float32(30.))/np.float32(.005))*np.float32(100.))/np.float32(100.)
    np.testing.assert_array_equal(state.qnorm.get(),want)
    atmosphere={'rho':cp.full_like(state.qv,1.2),'dz':cp.full_like(state.qv,80.)}
    physical=owner.noah_inputs(atmosphere)
    np.testing.assert_allclose(physical['heat'].get(),state.sda_hfx.get()*np.float32(1.2)*np.float32(1004.5)*np.float32(80.),rtol=2.e-7)
    np.testing.assert_array_equal(physical['qnorm'].get(),state.qnorm.get())
    assert np.isfinite(state.hfx_both.get()).all()
