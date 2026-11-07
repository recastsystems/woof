"""CU99 held carriers survive real stepping, checkpoints and nest coupling."""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("edrates,missing", [(0, None), (1, None),
                                             (1, "DER_KF"), (0, "W0AVG")])
def test_native_restart_overlay_restores_old_kf_hold(monkeypatch, edrates, missing):
    """Exercise the native overlay on a real driver, including its rain clock.

    The manifest is restricted to CU99 here; other physics' source maps are
    independent of the native KF carriers this regression exercises.
    """
    import cupy as cp
    from types import SimpleNamespace
    from woof.ingest import wrfrst
    from woof.io import restart
    from test_restart import _physics_state

    state, cfg, driver = _physics_state(cp, cu_physics=99, kf_edrates=edrates)
    adapter = driver.cumulus_callable
    # A fresh adapter must also materialise the complete required inventory.
    adapter.w0avg = None
    adapter._history_state = None
    for name in ("udr_kf", "ddr_kf", "uer_kf", "der_kf", "timec_kf"):
        setattr(adapter, name, None)
    native = {"W0AVG": np.full(state.p.shape, .125, np.float32)}
    scratch = {key: value for key, value in restart._scratch_manifest(state).items()
               if key.startswith("scratch/cu_")
               and key.partition("/")[2] in wrfrst.SCRATCH_SOURCES}
    for slot, (key, target) in enumerate(sorted(scratch.items()), 1):
        native[wrfrst.SCRATCH_SOURCES[key.partition("/")[2]]] = np.full(
            target.shape, slot * .25, np.float32)
    native["NCA"].fill(1000.)
    native["PRATEC"].fill(.03125)
    if edrates:
        for slot, name in enumerate(("UDR_KF", "DDR_KF", "UER_KF", "DER_KF",
                                     "TIMEC_KF"), 1):
            shape = state.p.shape[1:] if name == "TIMEC_KF" else state.p.shape
            native[name] = np.full(shape, slot * .125, np.float32)
    if missing:
        del native[missing]
    real_driver_manifest = restart._driver_manifest
    monkeypatch.setattr(restart, "state_manifest", lambda state: {})
    monkeypatch.setattr(restart, "_scratch_manifest", lambda state: scratch)
    monkeypatch.setattr(restart, "_driver_manifest", lambda driver: {
        k: v for k, v in real_driver_manifest(driver).items()
        if k.startswith("cumulus/")})
    restored = SimpleNamespace(raw=native, warm_start=SimpleNamespace(
        clock=SimpleNamespace(elapsed_seconds=120., itimestep=12)))
    if missing:
        with pytest.raises(ValueError, match=missing):
            wrfrst.apply_warm_start_physics(state, driver, restored, cfg)
        return
    report = wrfrst.apply_warm_start_physics(state, driver, restored, cfg)
    for key, target in scratch.items():
        np.testing.assert_array_equal(cp.asnumpy(target), native[report.filled[key]])
    for name in (("w0avg", "udr_kf", "ddr_kf", "uer_kf", "der_kf", "timec_kf")
                 if edrates else ("w0avg",)):
        np.testing.assert_array_equal(cp.asnumpy(getattr(adapter, name)),
                                      native[name.upper()])
    assert adapter._history_state is state
    assert adapter._history_time == 120.
    assert bool(cp.any(driver.cumulus_tendencies.rtheta != 0))
    before = cp.asnumpy(driver.rainc)
    driver._advance_cumulus_clock(state, cfg)
    np.testing.assert_array_equal(cp.asnumpy(driver.cu_nca), native["NCA"]-cfg.dt)
    np.testing.assert_array_equal(cp.asnumpy(driver.rainc),
                                  before+native["PRATEC"]*cfg.dt)


def _seed_hold(cp, state):
    """A nonzero hold crosses the checkpoint without a new convection call.

    Active convection is checked against native output in the oracle tests.
    Here distinct diagnostic values make an omitted carrier observable even
    in a stable synthetic atmosphere.
    """
    driver = state.physics
    adapter = driver.cumulus_callable
    driver.cu_nca.fill(cp.float32(1000.0))
    driver.cu_rates['rthcuten'].fill(cp.float32(1.e-5))
    driver.cu_pratec.fill(cp.float32(.001))
    for slot, name in enumerate(('w0avg', 'udr_kf', 'ddr_kf', 'uer_kf',
                                 'der_kf', 'timec_kf'), 1):
        array = getattr(adapter, name)
        shape = state.p.shape[1:] if name == 'timec_kf' else state.p.shape
        if array is None:
            array = cp.empty(shape, dtype=cp.float32)
            setattr(adapter, name, array)
        array.fill(cp.float32(slot * .125))


def test_old_kf_held_full_physics_restart_is_bit_identical(tmp_path, monkeypatch):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _assert_restart_equal, _physics_state

    # This production checkpoint path must obtain CU99 cadence from Rust.
    # Poison the former Python predicate so a scheduling fallback fails.
    from woof.core import physics
    def legacy_predicate(*args, **kwargs):
        pytest.fail("CU99 reached the Python cumulus scheduling predicate")
    monkeypatch.setattr(physics, "_cumulus_step_due", legacy_predicate)

    def build():
        state, cfg, driver = _physics_state(cp, cu_physics=99, kf_edrates=1)
        _seed_hold(cp, state)
        driver.recouple_cumulus_tendencies(state, cfg)
        return state, cfg, driver

    control, cfg, _ = build()
    run_steps(control, cfg, 40)
    reference = restart.write_restart(tmp_path/'reference.npz', control, cfg)
    split, cfg, _ = build()
    run_steps(split, cfg, 20)
    mid = restart.write_restart(tmp_path/'mid.npz', split, cfg)
    resumed, cfg, _ = build()
    restart.restore_restart(mid, resumed, cfg)
    run_steps(resumed, cfg, 20)
    result = restart.write_restart(tmp_path/'resumed.npz', resumed, cfg)
    _assert_restart_equal(reference, result)
    assert resumed.physics.call_counts['cumulus'] == 7
    # Microphysics can exhaust its condensate before step 40. Exercise the
    # nonzero CU99 hold itself rather than that independent heating source.
    assert bool(cp.all(resumed.physics.cu_rates['rthcuten'] == cp.float32(1.e-5)))
    assert bool(cp.any(resumed.physics.cumulus_tendencies.rtheta != 0))
    assert bool(cp.all(resumed.physics.rainc > 0))
    assert bool(cp.all(resumed.physics.cumulus_callable.udr_kf == .25))
    manifest = restart.read_restart_header(result)['array_manifest']
    for name in ('w0avg', 'udr_kf', 'ddr_kf', 'uer_kf', 'der_kf', 'timec_kf'):
        assert 'cumulus/'+name in manifest


def test_old_kf_tree_checkpoint_preserves_both_feedback_modes(tmp_path, monkeypatch):
    import cupy as cp
    from datetime import timedelta
    from woof.core.model import execute_experiment
    from woof.io.restart import restore_tree_restart, write_tree_restart
    from tilestream import test_moving_nest as moving
    from tilestream import test_nest_executor as executor
    from test_ensemble_nested_concurrent_gpu import _carried_words

    monkeypatch.setattr(executor, 'I_PARENT_START', 41)
    monkeypatch.setattr(executor, 'J_PARENT_START', 41)
    monkeypatch.setattr(executor, 'PARENT_DT', 6.)

    def build(steps, feedback):
        parent = moving.parent_cfg(nx=96, ny=96, nz=24, dt=6.,
                                   run_seconds=steps*6., nwp_diagnostics=0,
                                   cu_physics=99, kf_edrates=1,
                                   radt_minutes=.5, cudt_minutes=.5)
        child = moving.child_cfg(parent, nx=48, ny=48)
        boundaries = moving.domain_boundaries(parent, seconds=21600.)
        state, geography = moving.build_parent(parent, boundaries=boundaries)
        experiment = replace(executor.experiment(parent, child, steps=steps),
                             feedback=feedback)
        model = executor.assemble(experiment, state, geography.grid, feedback=feedback)
        if feedback:
            executor.initial_feedback(model)
        for node in model.walk_parent_first():
            _seed_hold(cp, node.state)
        return model, experiment

    parent_digests = {}
    for feedback in (0, 1):
        control, experiment = build(6, feedback)
        execute_experiment(control, validate_state=True, pool_trim_per_period=False)
        expected = _carried_words(control)
        parent_digests[feedback] = expected[1]['arrays']
        del control
        split, experiment = build(3, feedback)
        execute_experiment(split, validate_state=True, pool_trim_per_period=False)
        path = write_tree_restart(tmp_path/str(feedback), split,
                                  experiment.start_time+timedelta(seconds=18))
        del split
        resumed, _ = build(6, feedback)
        restore_tree_restart(path, resumed)
        execute_experiment(resumed, validate_state=True, pool_trim_per_period=False)
        assert _carried_words(resumed) == expected
        assert resumed.node(2).coupler.feedback_count > 0 if feedback else \
            resumed.node(2).coupler.feedback_count == 0
        del resumed
        cp.get_default_memory_pool().free_all_blocks()
    assert parent_digests[0] != parent_digests[1]


def test_old_kf_concurrent_members_keep_independent_held_carriers(tmp_path):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.ensemble.admission import EnsembleMemoryModel, MemoryComponent
    from woof.ensemble.execution import execute_concurrent_members
    from woof.ensemble.member_stream import member_cuda_scope
    from woof.ensemble.packing import CardBudget, pack_members
    from woof.io import restart
    from test_restart import _assert_restart_equal, _physics_state

    def forecast(member, tag):
        state, cfg, _ = _physics_state(cp, cu_physics=99, kf_edrates=1)
        _seed_hold(cp, state)
        # Distinct members make sharing a diagnostic plane observable.
        state.physics.cumulus_callable.udr_kf.fill(cp.float32(member+.25))
        run_steps(state, cfg, 8)
        assert bool(cp.all(state.physics.cumulus_callable.udr_kf == member+.25))
        return str(restart.write_restart(tmp_path/f'{tag}-{member}.npz', state, cfg))

    before = int(cp.get_default_memory_pool().total_bytes())
    references = [forecast(member, 'ordinary') for member in range(2)]
    cp.cuda.get_current_stream().synchronize()
    # Charge each member the whole measured pool growth, plus 64 MiB for
    # stream-local working storage. The actual card's free bytes own admission.
    envelope = max(0, int(cp.get_default_memory_pool().total_bytes())-before)+64*1024**2
    free, _ = cp.cuda.runtime.memGetInfo()
    model = EnsembleMemoryModel((MemoryComponent('CU99 complete member', 'ordinary',
                                                 fixed_bytes=envelope),))
    plan = pack_members(2, (CardBudget(0, int(free)),), model,
                        batched=False, concurrent_ordinary=True)
    assert len(plan.batches) == 1 and plan.batches[0].members == 2
    receipt = execute_concurrent_members(plan.batches[0],
        lambda single: forecast(single.member_indices[0], 'concurrent'),
        member_scope=lambda **kw: member_cuda_scope(array_module=cp, **kw))
    for member, row in enumerate(receipt['members']):
        _assert_restart_equal(references[member], row['result'])
    assert receipt['members'][0]['cuda_scope']['stream_ptr'] != \
        receipt['members'][1]['cuda_scope']['stream_ptr']
