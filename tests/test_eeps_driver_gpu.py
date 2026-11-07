"""EEPS carried scalars through real model steps and restart publication."""
import numpy as np
import pytest

@pytest.mark.gpu
@pytest.mark.parametrize('mp', [1, 10])
def test_eeps_live_step_and_restart_preserve_all_carried_words(tmp_path, mp):
    import cupy as cp
    from test_restart import _physics_state, _assert_restart_equal
    from woof.core.dycore import run_steps
    from woof.io import restart
    from woof.core.moist import extra_moist_species

    def make():
        return _physics_state(cp, bl_pbl_physics=16, mp_physics=mp, cu_physics=0)

    reference, cfg, driver = make()
    assert extra_moist_species(reference)[-2:] == ('pek_adv', 'pep_adv')
    # A nonuniform carried profile prevents a vacuous scalar carry check.
    run_steps(reference, cfg, 1)
    assert float(cp.max(reference.pek_adv)) > 0.0
    assert float(cp.max(reference.pep_adv)) > 0.0
    profile = cp.asarray(np.linspace(1., 1.4, cfg.nx, dtype=np.float32))[None,None,:]
    reference.pek_adv *= profile
    reference.pep_adv *= profile
    mid = restart.write_restart(tmp_path/'mid.npz', reference, cfg)
    carried = {name: cp.asnumpy(getattr(reference,name)).tobytes()
               for name in ('pek_adv','pep_adv')}
    fresh, cfg2, driver2 = make()
    restart.restore_restart(mid, fresh, cfg2)
    for name, expected in carried.items():
        assert cp.asnumpy(getattr(fresh,name)).tobytes() == expected, name
    for name in ('pek_pbl','pep_pbl'):
        assert cp.asnumpy(driver.fields[name]).tobytes() == cp.asnumpy(driver2.fields[name]).tobytes()
    run_steps(reference, cfg, 3)
    run_steps(fresh, cfg2, 3)
    a = restart.write_restart(tmp_path/'reference.npz',reference,cfg)
    b = restart.write_restart(tmp_path/'resumed.npz',fresh,cfg2)
    _assert_restart_equal(a,b)
    for name in ('pek_adv','pep_adv','thp'):
        assert np.isfinite(cp.asnumpy(getattr(fresh,name))).all(), name
    assert driver2.call_counts['ysu'] > 0
