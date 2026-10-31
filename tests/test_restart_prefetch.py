import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from woof.io.restart import _load_restart, RestartMismatchError
from woof.io.restart_prefetch import prefetch_restart, restart_payload_scope
from woof.supervisor import validate_manifest_checkpoint, CheckpointValidationError


def checkpoint(path, *, shape=(2, 2)):
    header = {'array_manifest': {'state/u': {'shape': list(shape), 'dtype': 'float32'}}}
    np.savez(path, __gpuwm_restart_header__=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8),
             **{'state/u': np.ones((2, 2), dtype=np.float32)})


def test_prefetched_manifest_is_still_validated_and_does_not_read_again(tmp_path, monkeypatch):
    path = tmp_path/'parent.npz'
    checkpoint(path)
    payload = prefetch_restart(path)
    path.chmod(0o444)
    # A read-only checkpoint is valid on POSIX. Scope notices metadata changes,
    # including chmod; prefetch the now archived input before using it.
    payload = prefetch_restart(path)
    with restart_payload_scope((payload,)):
        monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('archive read on forecast path'))
        assert validate_manifest_checkpoint(path) == path.resolve()
        header, arrays = _load_restart(path, with_arrays=True)
        assert not arrays['state/u'].flags.writeable
        header['array_manifest'].clear()
        assert _load_restart(path, with_arrays=False)[0]['array_manifest']


def test_prefetched_payload_does_not_bypass_manifest_shape_refusal(tmp_path):
    path = tmp_path/'parent.npz'
    checkpoint(path, shape=(3, 3))
    with restart_payload_scope((prefetch_restart(path),)):
        with pytest.raises(CheckpointValidationError, match='shape/dtype'):
            validate_manifest_checkpoint(path)


def test_replaced_parent_cannot_silently_change_ensemble(tmp_path):
    path = tmp_path/'parent.npz'
    checkpoint(path)
    payload = prefetch_restart(path)
    checkpoint(path)
    with restart_payload_scope((payload,)):
        with pytest.raises(RestartMismatchError, match='changed after prefetch'):
            _load_restart(path, with_arrays=True)


def test_payload_scope_is_not_shared_with_unrelated_thread(tmp_path, monkeypatch):
    path = tmp_path/'parent.npz'
    checkpoint(path)
    payload = prefetch_restart(path)
    original = np.load
    opened = []
    def read(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(np, 'load', read)
    with restart_payload_scope((payload,)):
        _load_restart(path, with_arrays=True)
        with ThreadPoolExecutor(1) as pool:
            pool.submit(_load_restart, path, with_arrays=True).result()
    assert opened == [path]


def _beside_done(path):
    from woof.io import restart_prefetch
    held = restart_prefetch._BESIDE[path.resolve()]
    return held[1].result()


def test_beside_read_serves_the_restore_reads_and_is_released(tmp_path, monkeypatch):
    # SP3 (2026-10-29): a leg's checkpoint is read once on a host thread from the leg's start; the
    # restore's manifest check and array load are lookups, and the arrays go once the restore applied them.
    from woof.io.restart_prefetch import prefetch_beside, release_beside
    path = tmp_path/'member.npz'
    checkpoint(path)
    prefetch_beside(path)
    assert _beside_done(path) is not None
    original = np.load
    opened = []
    monkeypatch.setattr(np, 'load', lambda *a, **k: opened.append(a[0]) or original(*a, **k))
    assert validate_manifest_checkpoint(path) == path
    header, arrays = _load_restart(path, with_arrays=True)
    assert opened == []
    np.testing.assert_array_equal(arrays['state/u'], np.ones((2, 2), dtype=np.float32))
    release_beside([path])
    _load_restart(path, with_arrays=True)
    assert opened == [path]


def test_beside_read_of_a_replaced_checkpoint_is_not_used(tmp_path):
    from woof.io.restart_prefetch import prefetch_beside, release_beside
    path = tmp_path/'member.npz'
    checkpoint(path)
    prefetch_beside(path)
    _beside_done(path)
    header = {'array_manifest': {'state/u': {'shape': [2, 2], 'dtype': 'float32'}}}
    other = tmp_path/'other.npz'
    np.savez(other, __gpuwm_restart_header__=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8),
             **{'state/u': np.full((2, 2), 7, dtype=np.float32)})
    other.replace(path)
    try:
        assert _load_restart(path, with_arrays=True)[1]['state/u'][0, 0] == 7
    finally:
        release_beside()


def test_a_legs_start_keeps_only_its_own_beside_reads(tmp_path):
    # a refused leg's read is kept for its retry; the next leg (other files) drops it
    from woof.io import restart_prefetch
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    checkpoint(a)
    checkpoint(b)
    try:
        restart_prefetch.prefetch_beside(a)
        restart_prefetch.prefetch_beside(b)
        restart_prefetch.keep_only_beside([a])
        assert set(restart_prefetch._BESIDE) == {a.resolve()}
        restart_prefetch.keep_only_beside([])
        assert not restart_prefetch._BESIDE
    finally:
        restart_prefetch.release_beside()


def test_leg_server_reads_the_legs_restart_flag():
    from woof.leg_server import _flag_value
    argv = ['--wrfinput', 'run', '--restart', '/x/a.npz', '--outdir', 'o']
    assert _flag_value(argv, '--restart') == '/x/a.npz'
    assert _flag_value(['--restart=/y.npz'], '--restart') == '/y.npz'
    assert _flag_value(['--wrfinput', 'run'], '--restart') is None
