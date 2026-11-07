"""Exercise the production Rust cadence ABI against the WRF driver contract."""
import math
import os

import pytest

from woof.core.cu99_launch import plan_step


@pytest.fixture(scope="module", autouse=True)
def native_runtime():
    from woof.core.cu99_launch import resolve, BRIDGE_ENV
    try:
        resolve()
    except FileNotFoundError:
        if os.environ.get(BRIDGE_ENV):
            raise
        pytest.skip("CU99 native artifact is not built on this host")


def test_native_clock_does_not_load_the_cuda_driver(monkeypatch):
    from woof.core import cu99_launch
    original = cu99_launch.ctypes.CDLL
    def cpu_library(name, *args, **kwargs):
        assert name is not None
        assert not os.path.basename(os.fspath(name)).lower().startswith(("libcuda", "nvcuda"))
        return original(name, *args, **kwargs)
    monkeypatch.setattr(cu99_launch, "_LIBRARY", None)
    monkeypatch.setattr(cu99_launch, "_DRIVER", None)
    monkeypatch.setattr(cu99_launch.ctypes, "CDLL", cpu_library)
    assert plan_step(elapsed=0., dt=15., cudt_minutes=5.).due == 1
    assert cu99_launch._DRIVER is None


@pytest.mark.parametrize("dt,clock,cudt", [
    (15., 0., 5.), (45., 0., 5.), (5., 15., 5.),
    (7.5, 0., 0.), (12., 0., .4), (11., 0., 5.),
])
def test_native_plan_matches_wrf_and_restored_domain_clock(dt, clock, cudt):
    effective = clock or dt
    for epoch in (0., 3600.):
        for step in range(721):
            elapsed = epoch + step * dt
            for override in (None, False, True):
                plan = plan_step(elapsed=elapsed, epoch=epoch, dt=dt,
                    clock_dt=clock, cudt_minutes=cudt, due_override=override)
                interval = max(math.floor(cudt * 60 / dt + .5), 1)
                timestep = math.floor((elapsed - epoch) / dt + .5) + 1
                due = (timestep == 1 or cudt == 0 or interval == 1
                       or timestep % interval == 0)
                remainder = abs(math.fmod(elapsed + dt, effective))
                assert bool(plan.due) == (due if override is None else override)
                assert plan.clock_dt == effective
                assert plan.stepcu == interval
                assert plan.column_stepcu == max(math.floor(cudt * 60 / effective + .5), 1)
                assert bool(plan.advance) == (min(remainder, effective - remainder)
                                              <= 1.e-6 * effective)


@pytest.mark.parametrize("kwargs", [dict(dt=0), dict(dt=-1),
    dict(elapsed=float("nan")), dict(clock_dt=-1), dict(cudt_minutes=float("inf"))])
def test_invalid_native_clock_is_rejected(kwargs):
    values = dict(elapsed=0., dt=15., cudt_minutes=5.)
    values.update(kwargs)
    with pytest.raises(ValueError, match="Rust scheduler rejected"):
        plan_step(**values)
