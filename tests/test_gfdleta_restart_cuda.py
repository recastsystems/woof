"""Live GFDL memory and held radiation survive an off-calendar checkpoint."""
import numpy as np
import pytest

from conftest import requires_gpu
from woof.gfdleta_contract import GFDLETA_FIELDS, GFDLETA_INTEGER_FIELDS
from woof.io import restart
from test_restart import _physics_state, _assert_restart_equal


@requires_gpu
@pytest.mark.gpu
def test_gfdl_restart_rebuilds_tables_and_resumes_bitwise(tmp_path):
    import cupy as cp
    from woof.core.dycore import run_steps

    def initial():
        return _physics_state(cp, mp_physics=6, cu_physics=0, ra_physics=0,
                              ra_lw_physics=99, ra_sw_physics=99,
                              radt=1., radt_minutes=1.)

    reference, cfg, reference_driver = initial()
    run_steps(reference, cfg, 12)
    full = restart.write_restart(tmp_path/'full.npz', reference, cfg)

    split, _, split_driver = initial()
    run_steps(split, cfg, 5)  # Between the first and seventh-step calls.
    mid = restart.write_restart(tmp_path/'mid.npz', split, cfg)
    header = restart.read_restart_header(mid)
    assert {f'fields/{name}' for name in GFDLETA_FIELDS} <= set(header['array_manifest'])
    assert bool(cp.any(split_driver.fields['ncfrst'] > 0))

    resumed, _, target = initial()
    assert target.radiation_callable.tables is None
    restart.restore_restart(mid, resumed, cfg)
    for name in GFDLETA_FIELDS:
        assert cp.asnumpy(target.fields[name]).tobytes() == cp.asnumpy(split_driver.fields[name]).tobytes(), name
    for name in GFDLETA_INTEGER_FIELDS:
        assert target.fields[name].dtype == np.int32
    run_steps(resumed, cfg, 7)
    continued = restart.write_restart(tmp_path/'continued.npz', resumed, cfg)
    _assert_restart_equal(full, continued)
    assert cp.asnumpy(target.radiation_callable.tables.words).tobytes() == cp.asnumpy(reference_driver.radiation_callable.tables.words).tobytes()
    assert target.call_counts['radiation'] == reference_driver.call_counts['radiation'] == 2


@requires_gpu
@pytest.mark.gpu
def test_gfdl_member_executor_keeps_private_memory_and_scalar_words():
    import cupy as cp
    from woof.core.dycore import update_diagnostics
    from woof.ensemble.physics_execution import MemberPhysicsBinding, MemberPhysicsExecutor

    originals, bindings = [], []
    for member in range(2):
        pair = []
        for _ in range(2):
            state, cfg, driver = _physics_state(cp, mp_physics=6, cu_physics=0,
                ra_physics=0, ra_lw_physics=99, ra_sw_physics=99,
                radt=1., radt_minutes=1.)
            state.thp.fill(cp.float32(.25*member))
            update_diagnostics(state, cfg.hypsometric_opt)
            pair.append((state, cfg, driver))
        originals.append(pair[0])
        bindings.append(MemberPhysicsBinding(*pair[1], member))
    executor = MemberPhysicsExecutor(bindings)
    for elapsed in (0., 10., 60.):
        for state, _, _ in originals:
            state.elapsed_seconds = elapsed
        for binding in bindings:
            binding.state.elapsed_seconds = elapsed
        held = executor.compute()
        for member, (state, cfg, driver) in enumerate(originals):
            driver.compute(state, cfg)
            target = bindings[member].driver
            assert held[member] is target.tendencies
            actual, expected = restart._driver_manifest(target), restart._driver_manifest(driver)
            assert actual.keys() == expected.keys()
            for name, value in expected.items():
                assert cp.asnumpy(actual[name]).tobytes() == cp.asnumpy(value).tobytes(), (member, elapsed, name)
            assert target.call_counts == driver.call_counts
    left, right = (binding.driver for binding in bindings)
    assert left.radiation_callable.tables is not right.radiation_callable.tables
    for name in GFDLETA_FIELDS:
        assert left.fields[name].data.ptr != right.fields[name].data.ptr
    assert cp.asnumpy(left.fields['glw']).tobytes() != cp.asnumpy(right.fields['glw']).tobytes()
