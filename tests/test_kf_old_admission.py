"""Old KF enters through the public namelist and publishes its diagnostics."""
from dataclasses import replace
import pytest
from test_namelist_import import INPUT_TEXT, _pair
from woof.namelist_import import import_namelists
from woof.experiment import load_experiment

@pytest.mark.parametrize('edrates', [0, 1])
def test_old_kf_namelist_roundtrip(tmp_path, edrates):
    inp = INPUT_TEXT.replace('cu_physics = 1, 0,', 'cu_physics = 99, 0,')
    inp = inp.replace('cudt = 5, 0,', f'cudt = 5, 0,\n kf_edrates = {edrates}, 0,')
    text, report = import_namelists(*_pair(tmp_path, inp=inp))
    path = tmp_path / 'old-kf.toml'
    path.write_text(text)
    exp = load_experiment(path)
    assert exp.root.run.cu_physics == 99
    assert exp.root.run.cudt_minutes == 5.0
    assert exp.root.run.kf_edrates == edrates
    assert not any(s.key == 'cu_physics' for s in report.substitutions)
    assert all(d.run.kf_edrates == 0 for d in exp.domains[1:])

def test_old_kf_diagnostic_history_is_budgeted(tmp_path):
    from woof.io.history_layout import produced_history_shapes
    from woof.core.preflight import physics_array_shapes
    inp = INPUT_TEXT.replace('cu_physics = 1, 0,', 'cu_physics = 99, 0,')
    text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    path = tmp_path / 'old-kf.toml'
    path.write_text(text)
    cfg = replace(load_experiment(path).root.run, kf_edrates=1)
    budget = physics_array_shapes(cfg)
    fields = produced_history_shapes(cfg)
    for name in ('udr_kf', 'ddr_kf', 'uer_kf', 'der_kf'):
        assert budget[f'cumulus/{name}'] == (cfg.nz, cfg.ny, cfg.nx)
        assert fields[name.upper()] == (cfg.nz, cfg.ny, cfg.nx)
    assert fields['TIMEC_KF'] == (cfg.ny, cfg.nx)
    assert budget['cumulus/w0avg'] == (cfg.nz, cfg.ny, cfg.nx)
    assert not any('kf_lut' in key for key in budget)
    from woof.core.preflight import scratch_slot_registry, _CUMULUS_KERNEL_MODULES
    assert scratch_slot_registry(cfg)['physics_validation_status'] == (1,)
    assert _CUMULUS_KERNEL_MODULES[99] == ('kf_old', 'kf_validation')


def test_old_kf_authored_clock_keeps_the_requested_cumulus_period():
    from fractions import Fraction
    from woof.domain_wizard import _root_physics_periods, snap_cadences_to_clock
    selected = {'cu_physics':99, 'cudt_minutes':5., 'radt':12.}
    assert Fraction(300) in _root_physics_periods(selected, {'ra_physics':4})
    snapped, notes = snap_cadences_to_clock(Fraction(720, 11), selected)
    assert snapped['cudt_minutes'] == 12.
    assert any('cudt_minutes' in note for note in notes)

def test_diagnostics_require_their_producer(tmp_path):
    inp = INPUT_TEXT.replace('cudt = 5, 0,', 'cudt = 5, 0,\n kf_edrates = 1, 0,')
    with pytest.raises(ValueError, match='output profiles'):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_omitted_old_kf_diagnostics_do_not_change_inactive_import_receipt(tmp_path):
    _, report = import_namelists(*_pair(tmp_path))
    assert not any(item.key == 'kf_edrates' for item in report.fixed)

from conftest import requires_gpu
@requires_gpu
def test_old_kf_real_kernel_diagnostics_survive_history_and_restart(tmp_path):
    import cupy as cp
    import numpy as np
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    from woof.io import restart
    from test_kf_old_wrf461_parity import FIXTURE, _groups

    with np.load(FIXTURE) as data:
        chosen = None
        for g in _groups(data):
            for row in range(data[f'{g}/t'].shape[0]):
                if (not data[f'{g}/adapt'][row] and data[f'{g}/kf_edrates'][row] and
                    data[f'{g}/f_qi'][row] and data[f'{g}/f_qs'][row] and
                    data[f'{g}/f_qr'][row] and data[f'{g}/pratec'][row] > 0):
                    chosen = g, row
                    break
            if chosen: break
        assert chosen
        g, row = chosen
        nz = data[f'{g}/t'].shape[1]
        cfg = RunConfig(nx=3, ny=2, nz=nz, dx=float(data[f'{g}/dx'][row]),
                        dy=float(data[f'{g}/dx'][row]), ztop=16000.,
                        dt=float(data[f'{g}/dt'][row]), run_seconds=0.,
                        moist=True, mp_physics=6, cu_physics=99, kf_edrates=1,
                        cudt_minutes=float(data[f'{g}/cudt'][row]),
                        use_adaptive_time_step=bool(data[f'{g}/adapt'][row]))
        # Override cadence to match the oracle's fixed-step driver input.
        if not cfg.use_adaptive_time_step:
            cfg = replace(cfg, cudt_minutes=cfg.dt * int(data[f'{g}/stepcu'][row]) / 60.)
        def state_for():
            coord = make_vertical_coord(cfg.nz)
            base = make_base_state(coord, lambda z: 300.+.003*np.asarray(z),
                                   p_surf=cfg.p_surf, ztop=cfg.ztop)
            return init_moist_balanced(cfg, coord, base,
                                      lambda z: .01*np.exp(-np.asarray(z)/2200.))
        state = state_for()
        driver = initialize_physics(state, cfg)
        def columns(key):
            a=data[f'{g}/{key}'][row].astype(np.float32)
            return cp.asarray(np.ascontiguousarray(np.broadcast_to(a[:,None,None],(a.size,2,3))))
        state.w[...] = columns('w')
        adapter = driver.cumulus_callable
        adapter.w0avg[...] = columns('w0avg')
        driver.cu_nca[...] = data[f'{g}/nca'][row]
        atmosphere = {key: columns(src) for key,src in {
            'u':'u','v':'v','temperature':'t','qv':'qv','pressure':'p',
            'rho':'rho','dz':'dz8w','exner':'pi'}.items()}
        driver._run_cumulus(atmosphere, state, cfg)
        output = driver.output_fields()
        for name in ('udr_kf','ddr_kf','uer_kf','der_kf','timec_kf'):
            got = cp.asnumpy(output[name.upper()])
            want = data[f'{g}/{name}'][row].astype(np.float32)
            if got.ndim == 3:
                want = np.broadcast_to(want[:,None,None],got.shape)
            else:
                want = np.full(got.shape,want,dtype=np.float32)
            np.testing.assert_array_equal(got.view(np.uint32),want.copy().view(np.uint32))
        assert bool(cp.any(adapter.udr_kf != 0))
        before = {name: cp.asnumpy(getattr(adapter,name)).view(np.uint32).copy()
                  for name in ('udr_kf','ddr_kf','uer_kf','der_kf','timec_kf')}
        driver.cu_nca[...] = np.float32(cfg.dt*10)
        driver._run_cumulus(atmosphere,state,cfg)
        for name,bits in before.items():
            np.testing.assert_array_equal(cp.asnumpy(getattr(adapter,name)).view(np.uint32),bits)
        path = restart.write_restart(tmp_path/'old-kf.npz', state, cfg)
        restored = state_for()
        restored_driver = initialize_physics(restored,cfg)
        restart.restore_restart(path,restored,cfg)
        for name in ('w0avg','udr_kf','ddr_kf','uer_kf','der_kf','timec_kf'):
            np.testing.assert_array_equal(cp.asnumpy(getattr(adapter,name)).view(np.uint32),
                                          cp.asnumpy(getattr(restored_driver.cumulus_callable,name)).view(np.uint32))


def test_diagnostic_registry_row_is_implemented():
    from woof.physics_registry import physics_registry
    spec = physics_registry()['parameters']['kf_edrates']
    assert spec.get('implemented') is not False
    assert spec['enum'] == [0, 1]
    assert spec['read_when'] == {'cu_physics': [11, 99]}


@requires_gpu
def test_old_kf_passive_vapour_drops_native_cloud_water_tendency():
    """WRF retains RQCCUTEN but has no P_QC to receive it with MP off."""
    import cupy as cp
    import numpy as np
    from test_kf_old_wrf461_parity import FIXTURE, _groups
    from test_restart import _physics_state

    with np.load(FIXTURE) as data:
        chosen = None
        for group in _groups(data):
            mask = (~data[f'{group}/adapt'].astype(bool)
                    & ~data[f'{group}/warm_rain'].astype(bool)
                    & ~data[f'{group}/f_qi'].astype(bool)
                    & ~data[f'{group}/f_qs'].astype(bool)
                    & ~data[f'{group}/f_qr'].astype(bool)
                    & (data[f'{group}/pratec'] > 0)
                    & (data[f'{group}/rqccuten'].max(axis=1) > 0))
            rows = np.flatnonzero(mask)
            if rows.size:
                chosen = group, int(rows[0])
                break
        assert chosen is not None, 'oracle must include a detraining passive-qv case'
        group, row = chosen
        nz = data[f'{group}/t'].shape[1]
        dt = float(data[f'{group}/dt'][row])
        state, cfg, driver = _physics_state(cp, nz=nz, mp_physics=0,
            cu_physics=99, kf_edrates=int(data[f'{group}/kf_edrates'][row]),
            dx=float(data[f'{group}/dx'][row]), dt=dt,
            cudt_minutes=dt*int(data[f'{group}/stepcu'][row])/60.)

        def columns(key):
            column = data[f'{group}/{key}'][row].astype(np.float32)
            return cp.asarray(np.ascontiguousarray(np.broadcast_to(
                column[:, None, None], (column.size, cfg.ny, cfg.nx))))

        state.w[...] = columns('w')
        driver.cumulus_callable.w0avg[...] = columns('w0avg')
        driver.cu_nca[...] = data[f'{group}/nca'][row]
        atmosphere = {key: columns(source) for key, source in {
            'u': 'u', 'v': 'v', 'temperature': 't', 'qv': 'qv',
            'pressure': 'p', 'rho': 'rho', 'dz': 'dz8w', 'exner': 'pi'}.items()}
        driver._run_cumulus(atmosphere, state, cfg)
        raw = cp.asnumpy(driver.cu_rates['rqccuten'])
        expected = cp.asnumpy(columns('rqccuten'))
        np.testing.assert_array_equal(raw.view(np.uint32), expected.view(np.uint32))
        assert raw.max() > 0
        assert bool(cp.all(driver.cumulus_tendencies.rqc == 0))
        assert bool(cp.any(driver.cumulus_tendencies.rqv != 0))
        # The relocation/checkpoint rebuild must apply the same scalar gate.
        driver.recouple_cumulus_tendencies(state, cfg)
        assert bool(cp.all(driver.cumulus_tendencies.rqc == 0))
        np.testing.assert_array_equal(cp.asnumpy(driver.cu_rates['rqccuten']).view(np.uint32),
                                      expected.view(np.uint32))


def test_public_old_kf_preflight_prices_its_kernel_and_workspace(tmp_path):
    from woof.experiment import load_experiment
    from woof.core import preflight as pf
    text, _ = import_namelists(*_pair(tmp_path, inp=INPUT_TEXT.replace('cu_physics = 1, 0,', 'cu_physics = 99, 0,')))
    path = tmp_path / 'experiment.toml'
    path.write_text(text)
    exp = load_experiment(path)
    modules = pf.physics_kernel_modules(exp)
    assert 'kf_old' in modules
    assert 'kf' not in modules and 'kf_validation' in modules
    assert pf.kf_column_workspace_bytes(exp, profile=pf.MEASURED_LOCAL_MEMORY_PROFILE) > 0
    frames = pf.kernel_local_frame_bytes(exp, profile=pf.MEASURED_LOCAL_MEMORY_PROFILE)
    assert frames['kf_old'] == 240
    assert 'kf_old' not in pf.assumed_bound_modules(modules)
