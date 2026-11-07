"""Original atmospheric solves with an actual packed-physics entry group."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace

import pytest

from conftest import requires_gpu
from woof.ensemble.scheduled_production_physics import ScheduledProductionPhysics
from test_ensemble_packed_production_physics_gpu import _driver, _assert_driver_words

pytestmark = [pytest.mark.gpu, requires_gpu]


def _cfl_context():
    from woof.core.cfl_member import member_cfl_scope
    with member_cfl_scope() as owner:
        context = copy_context()
    owner.enabled = True
    return context, owner


def _state_words(state):
    from woof.io.restart import state_manifest
    return {name: array.get().tobytes() for name, array in state_manifest(state).items()}


@pytest.mark.parametrize("cu_physics", [0, 99])
def test_real_original_dycore_steps_with_packed_physics_match_standalone_members(cu_physics):
    import cupy as cp
    from woof.core.dycore import step as ordinary_step
    from woof.core.mynn_pbl_runtime import release_mynn_stream_scratch
    members = 2
    # This isolated atmospheric solve has no LBC input bundle. Specified
    # rings belong to the prepared nested forecast gate with its real LBCs.
    together = [_driver(member, specified=False, cu_physics=cu_physics, kf_edrates=int(cu_physics == 99)) for member in range(members)]
    alone = [_driver(member, specified=False, cu_physics=cu_physics, kf_edrates=int(cu_physics == 99)) for member in range(members)]
    drivers, cfgs = zip(*together)
    ordinary_drivers, ordinary_cfgs = zip(*alone)
    contexts = [_cfl_context() for _ in range(members)]
    ordinary_contexts = [_cfl_context() for _ in range(members)]
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(members)]
    # A production session already has resident coefficient tables. Cold
    # upload reservations would hide missing packed return allocations.
    adapter = drivers[0].radiation_callable
    for name in ("lw_tables", "sw_tables", "lw_cloud_tables", "sw_cloud_tables"):
        getattr(adapter, name).to_device()
    bound = ScheduledProductionPhysics(drivers, cfgs, available_bytes=2 << 30)
    assert bound.owner._priced == 0
    try:
        with ThreadPoolExecutor(max_workers=members) as pool:
            for number, (elapsed, dt, sound) in enumerate(((0.0, 12.0, 4), (12.0, 10.0, 6)), 1):
                cfgs = tuple(replace(cfg, dt=dt, time_step_sound=sound) for cfg in cfgs)
                ordinary_cfgs = tuple(replace(cfg, dt=dt, time_step_sound=sound) for cfg in ordinary_cfgs)
                for driver in (*drivers, *ordinary_drivers):
                    driver.state.elapsed_seconds = elapsed
                    driver.bldt_seconds = dt
                    driver.radiation_due_override = number == 1
                    driver.surface_pbl_due_override = True
                for member, (driver, cfg) in enumerate(zip(ordinary_drivers, ordinary_cfgs)):
                    ordinary_contexts[member][0].run(ordinary_step, driver.state, cfg, refl_10cm_due=False)
                cp.cuda.get_current_stream().synchronize()
                ready = cp.cuda.Event()
                ready.record(cp.cuda.get_current_stream())
                bound.arm()
                def work(member):
                    with cp.cuda.Device(bound.owner.device), streams[member]:
                        streams[member].wait_event(ready)
                        try:
                            contexts[member][0].run(bound.wrap_step(member, ordinary_step),
                                drivers[member].state, cfgs[member], refl_10cm_due=False)
                        except BaseException as error:
                            bound.abort(error)
                            raise
                        finally:
                            streams[member].synchronize()
                futures = [pool.submit(work, member) for member in range(members)]
                errors = []
                for future in futures:
                    try:
                        future.result()
                    except BaseException as error:
                        errors.append(error)
                bound.finish()
                assert not errors
                for member in range(members):
                    assert _state_words(drivers[member].state) == _state_words(ordinary_drivers[member].state), (member, number)
                    _assert_driver_words(drivers[member], ordinary_drivers[member], member, number)
                    assert drivers[member].state.physics is drivers[member]
                assert bound.owner._priced <= bound.plan.required_bytes(members)
        assert bound.receipt["entry_groups"] == 2
        expected = {"radiation": 1, "surface": 4, "land": 2, "pbl": 2}
        if cu_physics:
            expected["cumulus"] = 1  # native CUDT holds the second call
        assert bound.owner.receipt["leaf_calls"] == expected
        assert bound.receipt["complete_forecast_qualified"] is False
    finally:
        bound.close()
        for stream in streams:
            with stream:
                release_mynn_stream_scratch(device_id=bound.owner.device, stream=stream)
        for _context, owner in (*contexts, *ordinary_contexts):
            for bank in owner.banks.values():
                bank.clear()
