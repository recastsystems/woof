"""Tree checkpoints stage canonical stores before mutating any sibling."""
from dataclasses import replace
import numpy as np
import pytest

from woof.io import restart
from tilestream import restart_stream, physics_inventory
from test_restart import _cfg, _fill_setup, _rewrite_restart_archive
from test_held_pbl_restart import _shim_driver_state


def _member(tmp_path, monkeypatch, gid):
    cfg = replace(_cfg(moist=True, cu_physics=3, bldt=2.), grid_id=gid)
    state, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    state.thp.fill(gid * 7)
    state.elapsed_seconds = 20.
    path = restart.write_restart(tmp_path/f'd{gid}.npz', state, cfg)
    store = {k: np.zeros_like(v) for k, v in
             physics_inventory.carrier_manifest(state).items()}
    return cfg, state, path, store


def test_staged_siblings_do_not_mutate_until_complete_validation(tmp_path, monkeypatch):
    members = [_member(tmp_path, monkeypatch, gid) for gid in (1, 2)]
    staged = []
    scalars = []
    for cfg, state, path, store in members:
        clock = {'elapsed_seconds': 0.}
        staged.append(restart_stream.validate_streamed_restart(
            path, store, cfg, setup=restart_stream.capture_domain_setup(state),
            template_state=state, scalars=clock))
        scalars.append(clock)
        assert not store['state/thp'].any()
        assert clock['elapsed_seconds'] == 0.
    cfg, state, path, store = members[1]
    def remove(payload, header):
        payload.pop('state/thp')
    bad = _rewrite_restart_archive(path, tmp_path/'bad.npz', remove)
    with pytest.raises(restart_stream.RestartRefused, match='missing'):
        restart_stream.validate_streamed_restart(
            bad, store, cfg, setup=restart_stream.capture_domain_setup(state),
            template_state=state, scalars=scalars[1])
    assert all(not member[3]['state/thp'].any() for member in members)
    # Application uses the already validated payload, never a second file read.
    for _, _, path, _ in members:
        path.write_bytes(b'the file changed after staging')
    for validation, (cfg, state, _, store), clock in zip(staged, members, scalars):
        result = validation.apply()
        assert result.device_copies == 0
        assert clock['elapsed_seconds'] == 20.
        np.testing.assert_array_equal(store['state/thp'], state.thp)


def test_store_writer_preserves_tree_header_and_protects_base_keys(tmp_path, monkeypatch):
    cfg, state, _, _ = _member(tmp_path, monkeypatch, 1)
    store = {k: v.copy() for k, v in physics_inventory.carrier_manifest(state).items()}
    args = dict(setup=restart_stream.capture_domain_setup(state),
                template_state=state, scalars=physics_inventory.carrier_scalars(state),
                check_pinned=False)
    header = {'checkpoint_set_id': 'one-generation', 'elapsed_ticks': 20000,
              'tick_den': 1000, 'adaptive_clock': {'step_count': 3}}
    result = restart_stream.write_streamed_restart(
        tmp_path/'tree.npz', store, cfg, tree_header=header, **args)
    written, arrays = restart._load_restart(result.path, with_arrays=True)
    assert all(written[key] == value for key, value in header.items())
    np.testing.assert_array_equal(arrays['state/thp'], state.thp)
    with pytest.raises(ValueError, match='may not replace base keys'):
        restart_stream.write_streamed_restart(
            tmp_path/'collision.npz', store, cfg,
            tree_header={'elapsed_seconds': 999.}, **args)
    assert not (tmp_path/'collision.npz').exists()


def test_streamed_feedback_checkpoint_preserves_original_identity_and_evolved_setup(tmp_path, monkeypatch):
    from woof import wrf_exact
    from woof.state_serialization_contract import (
        FEEDBACK_BASE_ATTRS, FEEDBACK_GEOGRAPHY_ATTRS,
        retain_feedback_setup, retain_feedback_geography)
    monkeypatch.setattr(wrf_exact, 'ENABLED', True)
    cfg = _cfg(moist=True, terrain_opt=1)
    state, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    setup = restart_stream.capture_domain_setup(state)
    # Snapshot before feedback, as the real canonical owner does.
    setup.arrays = {key: value.copy() for key, value in setup.arrays.items()}
    original = restart.setup_fingerprint(state)
    store = {key: value.copy() for key, value in physics_inventory.carrier_manifest(state).items()}
    retain_feedback_setup(state)
    retain_feedback_geography(state)
    for index, name in enumerate(FEEDBACK_BASE_ATTRS + FEEDBACK_GEOGRAPHY_ATTRS):
        getattr(state, name)[...] += np.float32(index+1)
    state._feedback_base_children = {'2': [4, 4]}
    state.set_base_geopotential(state.phb)
    info = restart_stream.write_streamed_restart(tmp_path/'feedback-stream.npz', store, cfg,
        scalars=physics_inventory.carrier_scalars(state), setup=setup,
        template_state=state, feedback_state=state, check_pinned=False)
    assert info.header['setup_fingerprint'] == original
    for target_streams in (False, True):
        target, _ = _shim_driver_state(cfg, monkeypatch)
        _fill_setup(target)
        if target_streams:
            target_store = {key: value.copy() for key, value in physics_inventory.carrier_manifest(target).items()}
            staged = restart_stream.validate_streamed_restart(info.path, target_store, cfg,
                setup=restart_stream.capture_domain_setup(target), template_state=target,
                feedback_state=target)
            assert not hasattr(target, '_setup_feedback_base')
            staged.apply()
        else:
            restart.restore_restart(info.path, target, cfg)
        assert restart.setup_fingerprint(target) == original
        assert target._feedback_base_children == {'2': [4, 4]}
        for name in FEEDBACK_BASE_ATTRS + FEEDBACK_GEOGRAPHY_ATTRS:
            np.testing.assert_array_equal(getattr(target, name), getattr(state, name))
    bad = _rewrite_restart_archive(info.path, tmp_path/'partial.npz',
        lambda payload, header: payload.pop('feedback_base/pb'))
    with pytest.raises(restart_stream.RestartRefused, match='complete strict feedback base'):
        restart_stream.validate_streamed_restart(bad, target_store, cfg,
            setup=setup, template_state=target, feedback_state=target)


def test_actual_tree_door_uses_both_stores_and_reseeds_after_validation(tmp_path, monkeypatch):
    from datetime import timedelta
    from types import SimpleNamespace
    from woof.core.streaming import StreamedDomain
    from test_restart import _lifecycle_free_model

    class Endpoint:
        write_restart = StreamedDomain.write_restart
        validate_restart = StreamedDomain.validate_restart
        apply_restart = StreamedDomain.apply_restart
        impose_clock = StreamedDomain.impose_clock
        _nest_archive_state = StreamedDomain._nest_archive_state
        _template = None

    def attach(node, *, clear):
        endpoint = Endpoint()
        endpoint.store = {k: v.copy() for k, v in
                          physics_inventory.carrier_manifest(node.state).items()}
        if clear:
            for array in endpoint.store.values():
                array.fill(0)
        endpoint.template_state = node.state
        setup = restart_stream.capture_domain_setup(node.state)
        endpoint.restart_setup = lambda: setup
        endpoint.scalars = physics_inventory.carrier_scalars(node.state)
        endpoint.host_store = False  # CPU storage witness; no pinning or GPU.
        endpoint.reseeds = []
        endpoint._run = SimpleNamespace(reseed_clock=lambda value:
                                        endpoint.reseeds.append(dict(value)))
        node.state._streamed_domain = endpoint
        return endpoint

    source, start = _lifecycle_free_model(monkeypatch)
    endpoints = {node.cfg.grid_id: attach(node, clear=False)
                 for node in source.walk_parent_first()}
    for node in source.walk_parent_first():
        endpoints[node.cfg.grid_id].store['state/thp'].fill(100 + node.cfg.grid_id)
        node.state.thp.fill(-99)  # stale resident copy must never be published.
    def resident_forbidden(*args, **kwargs):
        raise AssertionError('tree checkpoint reached the resident payload route')
    monkeypatch.setattr(restart, 'write_restart', resident_forbidden)
    path = restart.write_tree_restart(tmp_path/'source', source,
                                      start+timedelta(seconds=3600))
    resumed, _ = _lifecycle_free_model(monkeypatch)
    targets = {node.cfg.grid_id: attach(node, clear=True)
               for node in resumed.walk_parent_first()}
    monkeypatch.setattr(restart, '_validate_restart', resident_forbidden)
    result = restart.restore_tree_restart(path, resumed)
    for gid, endpoint in targets.items():
        np.testing.assert_array_equal(endpoint.store['state/thp'],
                                      endpoints[gid].store['state/thp'])
        assert endpoint.reseeds[-1]['elapsed_seconds'] == 3600.
    assert result.already_complete
