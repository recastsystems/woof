"""CAM-specific native residue and member ownership contracts."""
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu


@pytest.mark.parametrize('strict', [False, True])
def test_cam_inactive_radar_stays_zero_without_a_standalone_operator(monkeypatch, strict):
    from woof import wrf_exact
    from woof.core import refl
    monkeypatch.setattr(wrf_exact, 'ENABLED', strict)
    shape = (3, 2, 2)
    field = np.full(shape, 17., np.float32)
    state = SimpleNamespace(qv=np.zeros(shape, np.float32),
                            p=np.ones(shape, np.float32),
                            physics=SimpleNamespace(refl_10cm=None),
                            scratch=lambda size, name: field)
    cfg = SimpleNamespace(mp_physics=11)
    initial = refl.analysis_refl_10cm(state, cfg)
    assert initial.shape == shape and not initial.any()
    monkeypatch.setattr(refl, 'compute_refl_10cm', lambda *a, **k: pytest.fail(
        'CAMMGMP has no standalone reflectivity operator'))
    refl.compute_and_stash_refl_10cm(state, cfg, None, None)
    assert refl.consume_refl_10cm(state) is field
    assert not field.any()


def test_cam_native_signed_moments_keep_finiteness_and_ceiling_checks():
    from woof.core.health import collect_state_fields, validate_fields_cpu
    state = SimpleNamespace(cldfra_old_mp=np.zeros((1, 1, 1), np.float32),
                            qc=np.array([[[-1.124140447e-9]]], np.float32),
                            nc=np.array([[[-69242.4765625]]], np.float32))
    check = lambda: validate_fields_cpu(collect_state_fields(state))
    assert check().ok
    state.nc[...] = np.nan
    assert check().first_bad_field == 'nc'
    state.nc[...] = 1.e16
    assert check().first_bad_field == 'nc'
    state.nc[...] = -69242.4765625
    state.cldfra_old_mp = None
    assert not check().ok


def test_cu_native_undefined_witness_never_exempts_active_or_float_words():
    from tools.cam5_wrf471_oracle.replay_capture import native_undefined_words
    witness = dict(kind='native-undefined-storage-witness', passed=True,
        source_commit='f52c197ed39d12e087d02c50f412d90d418f6186')
    expected = dict(lengath2d=np.array([0, 1], np.int32),
                    maxg2d=np.array([1, 1], np.int32),
                    ideep2d=np.array([111111111, 3], np.int32),
                    jt2d=np.array([111111111, 4], np.int32),
                    cape=np.array([0., 2.], np.float32))
    actual = {n: a.copy() for n, a in expected.items()}
    actual['ideep2d'][...] = -1
    actual['jt2d'][...] = -1
    actual['cape'][0] = 1.
    assert native_undefined_words('cu7', actual, expected, witness) == 2
    actual['lengath2d'][0] = 1
    assert native_undefined_words('cu7', actual, expected, witness) == 0
    assert native_undefined_words('mp11', actual, expected, witness) == 0
    with pytest.raises(ValueError, match='pinned native'):
        native_undefined_words('cu7', actual, expected, witness | {'passed': False})


def test_cam_nested_uw_pools_are_all_priced_as_resident():
    from woof.config import RunConfig
    from woof.core.preflight import uwpbl_column_workspace_bytes
    from woof.core.physics_inventory import uwpbl_workspace_bytes
    common = dict(nz=49, dx=12000., dy=12000., ztop=18000., dt=60.,
                  run_seconds=0., bl_pbl_physics=9, mp_physics=11)
    a = RunConfig(nx=6, ny=5, **common)
    b = RunConfig(nx=8, ny=7, **common)
    exp = SimpleNamespace(domains=[SimpleNamespace(run=c) for c in (a, b)])
    assert uwpbl_column_workspace_bytes(exp) == sum(
        uwpbl_workspace_bytes(c.nz, c.nx*c.ny) for c in (a, b))


@pytest.mark.gpu
@requires_gpu
def test_cam_suite_member_drivers_match_independent_calls_without_aliasing():
    import cupy as cp
    from test_physics_driver import _full_state
    from test_ensemble_physics_execution_gpu import _snapshot
    from woof.core.dycore import update_diagnostics
    from woof.core.microphysics import apply
    from woof.ensemble.physics_execution import MemberPhysicsBinding, MemberPhysicsExecutor

    originals, bindings = [], []
    for member in range(2):
        pair = []
        for _ in range(2):
            state, cfg, driver = _full_state(nx=6, ny=5, nz=49, ztop=18000.,
                dx=12000., dy=12000., dt=60., mp_physics=11,
                bl_pbl_physics=9, sf_sfclay_physics=1,
                cu_physics=7, shcu_physics=2, bldt=0., radt=0.)
            state.u[...] = cp.float32(6. + member * .25)
            state.elapsed_seconds = float(member * 60)
            pair.append((state, cfg, driver))
        originals.append(pair[0])
        state, cfg, driver = pair[1]
        bindings.append(MemberPhysicsBinding(state, cfg, driver, member))
    executor = MemberPhysicsExecutor(bindings)
    for _ in range(2):
        executor.compute()
        for state, cfg, driver in originals:
            driver.compute(state, cfg)
        executor.apply_microphysics()
        for member, (state, cfg, driver) in enumerate(originals):
            driver.accept_microphysics(apply(state, cfg, cfg.dt), dt=cfg.dt)
            update_diagnostics(state, cfg.hypsometric_opt)
            update_diagnostics(bindings[member].state, cfg.hypsometric_opt)
            assert _snapshot(bindings[member].state, cfg, bindings[member].driver) == _snapshot(state, cfg, driver)
            state.elapsed_seconds += cfg.dt
            bindings[member].state.elapsed_seconds += cfg.dt
        executor.finish_step()
        for _, _, driver in originals:
            driver.finish_step()
    assert executor.call_counts == dict(compute=[2, 2], microphysics=[2, 2], finish_step=[2, 2])
    assert bindings[0].state.cldfra_old_mp.data.ptr != bindings[1].state.cldfra_old_mp.data.ptr
    assert bindings[0].driver.fields['cush'].data.ptr != bindings[1].driver.fields['cush'].data.ptr
