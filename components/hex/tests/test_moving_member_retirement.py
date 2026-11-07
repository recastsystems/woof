from pathlib import Path
from types import SimpleNamespace

import pytest
from woof.hex.moving_run import Controller


def controller(tmp_path):
    c = object.__new__(Controller)
    c.out = tmp_path
    c.members_dir = tmp_path / 'members'
    c.members_dir.mkdir()
    c.a = SimpleNamespace(member_clean_script='/cleanup', start_member='start')
    members = {}
    for name in ('start', 'current', 'carrier', 'worker', 'old'):
        row = dict(status='stale')
        for kind in ('static', 'init', 'grid'):
            p = c.members_dir / f'{name}.{kind}.nc'
            p.write_bytes(b'fixture')
            row[kind] = str(p)
        (c.members_dir / f'{name}-case.json').write_text('{}')
        members[name] = row
    c.state = dict(current='current', members=members)
    c.procs = {'op:worker--current': SimpleNamespace(poll=lambda: None)}
    c.carrier_member = lambda: 'carrier'
    c.save = lambda: None
    c.event = lambda *args, **kwargs: None
    return c


def test_retirement_protects_live_members_and_preserves_geometry(tmp_path, monkeypatch):
    c = controller(tmp_path)
    def clean(argv, check):
        assert argv[:2] == ['bash', '/cleanup']
        for value in argv[2:]:
            Path(value).unlink()
    monkeypatch.setattr('woof.hex.moving_run.subprocess.run', clean)
    c.retire_stale_members()
    for name, row in c.state['members'].items():
        assert Path(row['grid']).exists()
        assert Path(row['init']).exists() == (name != 'old')
        assert (c.members_dir / f'{name}-case.json').exists() == (name != 'old')
    assert c.state['members']['old']['status'] == 'retired'


def test_retirement_refuses_outside_inputs(tmp_path):
    c = controller(tmp_path)
    c.state['members']['old']['init'] = str(tmp_path / 'outside.nc')
    with pytest.raises(ValueError, match='outside owned run'):
        c.retire_stale_members()
    assert c.state['members']['old']['status'] == 'stale'


def test_default_does_not_retire(tmp_path):
    c = controller(tmp_path)
    c.a.member_clean_script = None
    c.retire_stale_members()
    assert all(Path(m['init']).exists() for m in c.state['members'].values())


def test_uncollected_frame_protects_member_inputs(tmp_path, monkeypatch):
    c = controller(tmp_path)
    history = tmp_path / 'tracker.nc'
    history.write_bytes(b'closed tracker')
    c.state['frames'] = [dict(member='old', history=str(history))]
    def clean(argv, check):
        for value in argv[2:]:
            Path(value).unlink()
    monkeypatch.setattr('woof.hex.moving_run.subprocess.run', clean)
    c.retire_stale_members()
    assert c.state['members']['old']['status'] == 'stale'
    assert Path(c.state['members']['old']['init']).exists()
    history.unlink()  # collector acknowledgment follows its final render
    c.retire_stale_members()
    assert c.state['members']['old']['status'] == 'retired'
