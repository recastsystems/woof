"""The exposed-strip failure, including A -> B -> A and checkpoint carry."""
from types import SimpleNamespace
import numpy as np

from woof.core.nest_accumulation import advance, capture, install, META, PREFIX
from woof.core.physics_continuation import shift_continuation
from woof.core.nest_relocation import Placement, plan_relocation


def move(start, end):
    return plan_relocation(placement_from=Placement(2, start, 10),
        placement_to=Placement(2, end, 10), parent_grid_ratio=3,
        child_nx=12, child_ny=12)


def test_first_real_relocation_archive_passes_health_inventory():
    # Exact first relocation from the 12/3 km, 160x160, 48 h case: the
    # forecast stopped after 600 domain steps on this int64 metadata field.
    import pytest
    from woof.core.health import gpu_integer_policy
    plan = plan_relocation(placement_from=Placement(2, 63, 74),
        placement_to=Placement(2, 63, 72), parent_grid_ratio=4,
        child_nx=160, child_ny=160)
    values = {'mp_rainnc': np.ones((160, 160), np.float32)}
    archive, seed, _ = advance({}, values, plan, ring=1)
    np.testing.assert_array_equal(archive[META], [0, -8, 0, -8])
    assert archive[PREFIX+'mp_rainnc'].shape == (168, 160)
    assert gpu_integer_policy('nest.scratch.'+META, archive[META].dtype) is None
    from woof.io.restart import _validate_scratch_target, RestartMismatchError
    restored_state = SimpleNamespace(_scratch={}, u=np.empty(0, np.float32))
    _validate_scratch_target(restored_state, META, archive[META], 'scratch/'+META)
    with pytest.raises(RestartMismatchError, match='four int64'):
        _validate_scratch_target(restored_state, META, archive[META].astype(np.float32), 'scratch/'+META)
    assert gpu_integer_policy('nest.scratch.'+PREFIX+'mp_rainnc', np.float32) == 0
    with pytest.raises(TypeError, match='unsupported dtype int64'):
        gpu_integer_policy('nest.scratch.unknown', np.int64)
    # Restored geometry must still contain the current footprint.
    archive[META][0] = 1
    with pytest.raises(ValueError, match='outside its archive'):
        advance(archive, seed, plan, ring=1)


def test_exposed_strips_zero_and_return_retains_nest_totals(tmp_path):
    # Parent rain was 12 mm; the nest has integrated only 3 mm. Boundary
    # cells were not integrated and may not overwrite archived interior.
    original = np.zeros((12, 12), np.float32)
    original[1:-1, 1:-1] = 3.
    ab = move(10, 12)
    archive, seed, receipt = advance({}, {'mp_rainnc': original}, ab, ring=1)
    b = shift_continuation({'mp_rainnc': original}, ab, strip=seed, ring=1)
    np.testing.assert_array_equal(b['mp_rainnc'][1:-1, :5], 3.)
    np.testing.assert_array_equal(b['mp_rainnc'][:, 5:], 0.)
    assert not receipt['parent_precipitation']
    # The restart registry must retain exactly the full archive including
    # the native-frame origin, without treating it as a relocatable tile.
    state = SimpleNamespace(_scratch={})
    install(state, archive)
    from woof.io.restart import _scratch_manifest
    payload = _scratch_manifest(state)
    path = tmp_path/'archive.npz'
    np.savez(path, **payload)
    with np.load(path) as saved:
        restored = {k[8:]: saved[k] for k in saved.files}
    np.testing.assert_array_equal(restored[META], archive[META])
    b['mp_rainnc'][1:-1, 1:-1] += 2.
    ba = move(12, 10)
    archive2, seed2, _ = advance(restored, b, ba, ring=1)
    returned = shift_continuation(b, ba, strip=seed2, ring=1)['mp_rainnc']
    # West of B retains 3 mm. B's old specified ring also restores its
    # archived 3 mm, rather than replacing it with the ring's cold zero.
    np.testing.assert_array_equal(returned[1:-1, 1:7], 3.)
    np.testing.assert_array_equal(returned[1:-1, 7:11], 5.)
    assert archive2[PREFIX+'mp_rainnc'].shape == (12, 18)
    install(state, archive2)
    assert capture(state)[META] is archive2[META]


def test_legacy_off_policy_keeps_parent_seed_and_old_restart_echo():
    from woof.core.nest_accumulation import relocate
    from woof.core.physics_continuation import parent_strip_accumulations
    from woof.config import RunConfig
    from woof.io.restart import configuration_echo, _require_config_match, RestartMismatchError
    import pytest
    run = RunConfig(nx=12, ny=12, nz=4, dx=3000., dy=3000.,
                    ztop=20000., dt=10., run_seconds=3600.,
                    nested=True, nest_accumulation_archive=False)
    cfg = SimpleNamespace(run=run, parent_grid_ratio=3,
                          i_parent_start=4, j_parent_start=4)
    parent = SimpleNamespace(state=SimpleNamespace(
        existing_scratch=lambda name: np.full((run.ny, run.nx), 12., np.float32)),
        cfg=SimpleNamespace(run=run))
    values = {'mp_rainnc': np.zeros((run.ny, run.nx), np.float32)}
    legacy, _ = parent_strip_accumulations(parent, cfg, values)
    archive, seed, _ = relocate({}, values, move(10, 12), 1, parent, cfg)
    assert not archive
    np.testing.assert_array_equal(seed['mp_rainnc'], legacy['mp_rainnc'])
    echo = configuration_echo(run)
    assert 'nest_accumulation_archive' not in echo
    _require_config_match(echo, run, 'legacy')
    from dataclasses import replace
    run = replace(run, nest_accumulation_archive=True)
    with pytest.raises(RestartMismatchError, match='nest_accumulation_archive'):
        _require_config_match(echo, run, 'legacy')


def test_archive_policy_reuses_prepared_initial_state_but_not_restart():
    from woof.ingest.prepared_cache import (
        compare_prepared_domain_config, effective_prepared_domain_config)
    old = {'run': {'nx': 160, 'nested': True}}
    off = {'run': {**old['run'], 'nest_accumulation_archive': False}}
    on = {'run': {**old['run'], 'nest_accumulation_archive': True}}
    for cached in (old, off, on):
        for live in (old, off, on):
            assert compare_prepared_domain_config(cached, live) == ([], [])
            assert effective_prepared_domain_config(cached) == effective_prepared_domain_config(live)


def test_old_checkpoint_physics_fingerprint_rebuilds_legacy_archive_policy():
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.io.restart import (
        configuration_echo, _run_config_from_header, _configuration_fingerprint)
    run = RunConfig(nx=12, ny=12, nz=4, dx=3000., dy=3000.,
                    ztop=20000., dt=10., run_seconds=3600.,
                    nested=True, nest_accumulation_archive=False)
    old_echo = configuration_echo(run)
    assert 'nest_accumulation_archive' not in old_echo
    rebuilt = _run_config_from_header(old_echo)
    assert rebuilt is not None and rebuilt.nest_accumulation_archive is False
    assert _configuration_fingerprint(rebuilt) == _configuration_fingerprint(run)
    assert _configuration_fingerprint(rebuilt) != _configuration_fingerprint(
        replace(run, nest_accumulation_archive=True))
