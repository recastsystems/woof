"""Immutable schedule reuse retains all stage words and responds to config changes."""
from types import SimpleNamespace
from woof.hex.cuda_driver_rrfs import CudaRRFSDycoreDriver
from woof.hex.integration import RKSchedule


def test_schedule_reuse_and_invalidation():
    driver = SimpleNamespace(config=SimpleNamespace(config_time_integration_order=3,
        config_number_of_sub_steps=6), _rrfs_schedule_cache=None)
    get = lambda dt, splits: CudaRRFSDycoreDriver._rrfs_schedules(driver, dt, splits)
    a, b = get(720.0, 3)
    assert get(720.0, 3)[0] is a and get(720.0, 3)[1] is b
    for dt, splits, order, acoustic in ((600., 3, 3, 6), (600., 1, 3, 6),
                                       (600., 1, 2, 6), (600., 1, 2, 4)):
        driver.config.config_time_integration_order = order
        driver.config.config_number_of_sub_steps = acoustic
        dynamics, scalars = get(dt, splits)
        assert dynamics == RKSchedule.from_mpas(dt, order=order, acoustic_substeps=acoustic,
                                                dynamics_splits=splits)
        assert scalars == RKSchedule.from_mpas(dt, order=order, acoustic_substeps=acoustic,
                                              dynamics_splits=1)
        assert dynamics is not a
        a = dynamics
