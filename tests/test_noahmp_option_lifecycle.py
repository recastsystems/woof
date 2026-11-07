"""Selected options keep checkpoint identity and member-owned trajectories."""
from dataclasses import replace

import pytest

from conftest import requires_gpu
from test_noahmp_no_consumer_knobs import _noahmp_cfg


def test_selected_checkpoint_identity_is_checked_per_option():
    from woof.io.restart import configuration_echo, _require_config_match, RestartMismatchError

    selected = _noahmp_cfg(opt_btr=2, opt_rsf=2)
    echo = configuration_echo(selected)
    assert echo["opt_btr"] == echo["opt_rsf"] == 2
    _require_config_match(echo, selected, "selected.npz")
    for name in ("opt_btr", "opt_rsf"):
        with pytest.raises(RestartMismatchError, match=name):
            _require_config_match(echo, replace(selected, **{name: 1}), "selected.npz")


@requires_gpu
def test_selected_noahmp_members_reproduce_independent_original_operations():
    from woof.ensemble.physics_execution import MemberPhysicsBinding, MemberPhysicsExecutor
    from woof.io.restart import _driver_manifest
    from test_noahmp_runtime import _build

    choices = ({"opt_btr": 2, "opt_rsf": 2}, {})
    actual = [_build(nx=6, ny=4, **options) for options in choices]
    reference = [_build(nx=6, ny=4, **options) for options in choices]
    executor = MemberPhysicsExecutor([
        MemberPhysicsBinding(state, cfg, driver, index)
        for index, (state, cfg, driver) in enumerate(actual)])
    for elapsed in (0.0, 12.0, 24.0):
        for state, _cfg, _driver in (*actual, *reference):
            state.elapsed_seconds = elapsed
        executor.compute()
        for index, (state, cfg, driver) in enumerate(reference):
            driver.compute(state, cfg)
            expected = _driver_manifest(driver)
            got = _driver_manifest(actual[index][2])
            assert got.keys() == expected.keys()
            for name, array in expected.items():
                assert got[name].get().tobytes() == array.get().tobytes(), (elapsed, index, name)
    assert executor.call_counts["compute"] == [3, 3]
    assert actual[0][2].noahmp_params is not actual[1][2].noahmp_params
    assert dict(actual[0][2].noahmp_params.land_options)["opt_btr"] == 2
