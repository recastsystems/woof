"""A dead resident process must not erase or masquerade as a recovery checkpoint."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import pytest

from woof.hex import moving_recovery as recovery, moving_run


def carrier(folder, seconds, *, resident=False, seal=False):
    path = folder / ('control/seal-%08d.pkl' % seconds if seal else 'boundary.pkl')
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b'closed test payload'
    if not resident:
        path.write_bytes(data)
    path.with_suffix('.pkl.json').write_text(json.dumps(dict(time_seconds=seconds, resident=resident,
        bytes=None if resident else len(data), sha256=None if resident else hashlib.sha256(data).hexdigest())))
    return path


def controller(tmp_path):
    c = object.__new__(moving_run.Controller)
    c.out = tmp_path
    folders = [tmp_path / f'segment-{i:03d}' for i in range(4)]
    c.state = dict(t=8., current='b', boundary=str(folders[3] / 'boundary.pkl'),
        operator='a--b.npz', source_grid='a.grid.nc', pending_move={'target': 'b'},
        frames=[dict(model_seconds=t) for t in (2, 4, 6, 8, 9)],
        moves=[dict(at_seconds=5., source='start', target='a'), dict(at_seconds=8., source='a', target='b')],
        deleted=[], members={n: dict(status='stale') for n in ('start', 'a', 'b')},
        segments=[dict(index=i, path=str(p), member='a' if i >= 2 else 'start',
                       begin_seconds=float(i*2), end_seconds=float(i*2+2)) for i, p in enumerate(folders)])
    c.session = SimpleNamespace(fixes=[SimpleNamespace(seconds=t) for t in (2, 4, 6, 8, 9)],
                                decisions=[SimpleNamespace(seconds=t) for t in (2, 4, 6, 8, 9)])
    c.events = []
    c.event = lambda kind, **row: c.events.append(dict(kind=kind, **row))
    c.save = lambda: None
    return c, folders


def test_resident_close_retains_two_real_clocks(tmp_path):
    c, folders = controller(tmp_path)
    old = carrier(folders[0], 2)
    previous = carrier(folders[1], 4, seal=True)
    latest = carrier(folders[2], 6, seal=True)
    duplicate = carrier(folders[2], 6)
    carrier(folders[3], 8, resident=True)
    recovery.retain_latest_two(c)
    assert previous.is_file() and duplicate.is_file()
    assert not old.exists() and not latest.exists()
    assert [r['time_seconds'] for r in c.state['recovery_carriers']] == [6, 4]


def test_process_loss_recovers_previous_member_and_replays_its_frame(tmp_path):
    c, folders = controller(tmp_path)
    carrier(folders[1], 4, seal=True)
    latest = carrier(folders[2], 6, seal=True)
    carrier(folders[3], 8, resident=True)
    assert recovery.recover_lost_resident(c)
    assert c.state['boundary'] == str(latest) and c.state['current'] == 'a'
    assert c.state['t'] == c.state['t_seen'] == 6
    assert c.state['operator'] is None and c.state['source_grid'] is None
    assert [f['model_seconds'] for f in c.state['frames']] == [2, 4]
    assert [f.seconds for f in c.session.fixes] == [2, 4]
    assert [m['at_seconds'] for m in c.state['moves']] == [5]
    assert c.state['pending_move'] is None
    assert c.state['segments'][3]['superseded_after_seconds'] == 6


def test_same_clock_seal_preserves_the_original_move_mapping(tmp_path):
    c, folders = controller(tmp_path)
    latest = carrier(folders[3], 8, seal=True)
    carrier(folders[3], 8, resident=True)
    assert recovery.recover_lost_resident(c)
    assert c.state['boundary'] == str(latest) and c.state['current'] == 'b'
    assert c.state['operator'] == 'a--b.npz' and c.state['source_grid'] == 'a.grid.nc'
    assert [f['model_seconds'] for f in c.state['frames']] == [2, 4, 6, 8]


def test_absent_or_preanalysis_recovery_fails_clearly(tmp_path):
    c, folders = controller(tmp_path)
    carrier(folders[3], 8, resident=True)
    with pytest.raises(RuntimeError, match='no durable recovery seal'):
        recovery.recover_lost_resident(c)
    carrier(folders[2], 6, seal=True)
    c.state['segments'][-1]['cycle_analysis'] = dict(time_seconds=7)
    with pytest.raises(RuntimeError, match='predates a cycle analysis'):
        recovery.recover_lost_resident(c)


def test_resident_receipt_is_not_read_as_an_on_disk_payload(tmp_path, monkeypatch):
    from woof.hex.mpasphys import moving_v842, moving_resident_v842
    path = carrier(tmp_path, 8, resident=True)
    monkeypatch.setattr(moving_resident_v842, 'take_carrier', lambda _: None)
    with pytest.raises(RuntimeError, match='exited forecast process'):
        moving_v842.load_carrier(path)


def test_resume_marks_the_lost_controller_before_launch(tmp_path):
    c, folders = controller(tmp_path)
    carrier(folders[2], 6, seal=True)
    carrier(folders[3], 8, resident=True)
    c.a = SimpleNamespace(resume=True, member_fallback=False)
    c.duration = 20
    c.request_startup_member = lambda: None
    class ReachedLaunch(Exception):
        pass
    def launch(kind):
        assert kind == 'recover'
        assert c.state['segments'][-1]['rc'] == 'controller-lost'
        assert c.state['t'] == 6 and c.state['current'] == 'a'
        raise ReachedLaunch
    c.launch = launch
    with pytest.raises(ReachedLaunch):
        c.run()
