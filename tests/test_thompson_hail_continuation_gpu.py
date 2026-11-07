"""MP38 continuation must retain the prognostic hail moments as words."""
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def hail_state(cp):
    from test_restart import _physics_state

    state, cfg, driver = _physics_state(
        cp, mp_physics=38, nz=24, ztop=14000.0, dt=2.0,
        sf_sfclay_physics=91)
    state.nc[2:6] = cp.float32(1.e8)
    state.qi[12:18] = cp.float32(4.e-4)
    state.ni[12:18] = cp.float32(5.e4)
    state.qs[12:18] = cp.float32(2.e-4)
    state.qg[12:18] = cp.float32(2.e-4)
    state.ng[12:18] = cp.float32(1.e4)
    state.qvolg[12:18] = state.qg[12:18] / cp.float32(500.)
    return state, cfg, driver


def test_hail_full_physics_restart_retains_live_number_and_volume(tmp_path):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _assert_restart_equal

    control, cfg, _ = hail_state(cp)
    run_steps(control, cfg, 40)
    reference = restart.write_restart(tmp_path / 'reference.npz', control, cfg)
    split, split_cfg, _ = hail_state(cp)
    run_steps(split, split_cfg, 20)
    for name in ('qg', 'ng', 'qvolg'):
        assert bool(cp.any(getattr(split, name) > 0)), name
    mid = restart.write_restart(tmp_path / 'mid.npz', split, split_cfg)
    resumed, resumed_cfg, _ = hail_state(cp)
    info = restart.restore_restart(mid, resumed, resumed_cfg)
    assert info.elapsed_seconds == 40.
    run_steps(resumed, resumed_cfg, 20)
    final = restart.write_restart(tmp_path / 'resumed.npz', resumed, resumed_cfg)
    _assert_restart_equal(final, reference)
    manifest = restart.read_restart_header(mid)['array_manifest']
    for name in ('qg', 'ng', 'qvolg'):
        assert 'state/' + name in manifest
    assert resumed.physics.microphysics_updates == 40
