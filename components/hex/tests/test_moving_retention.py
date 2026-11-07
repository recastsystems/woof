import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex.moving_retention import artifacts_present, retire_members, speculative_admission


def test_ready_member_keeps_its_speculative_slot_after_builder_exits():
    state = dict(current='start', members={'start': dict(status='ready')})
    assert speculative_admission(state, 1)
    for status in ('preparing', 'prepared', 'ready'):
        state['members']['next'] = dict(status=status)
        assert not speculative_admission(state, 1)
        assert speculative_admission(state, 0)
    state['current'] = 'next'
    state['members']['start']['status'] = 'stale'
    assert speculative_admission(state, 1)


def fixture(tmp_path):
    root = tmp_path / 'run'
    members = root / 'members'
    members.mkdir(parents=True)
    row = dict(status='stale', center_deg=[0, 0])
    for key in ('grid', 'init', 'static'):
        path = members / f'old.{key}.nc'
        path.write_bytes(b'owned fixture')
        row[key] = str(path)
    operators = members / 'operators'
    operators.mkdir()
    for name in ('old--next.npz', 'start--old.zone.npz', 'start--next.npz'):
        (operators / name).write_bytes(b'operator fixture')
    frame = dict(member='old', segment=0, model_seconds=3600,
                 history=str(root / 'closed-history.nc'))
    evidence = tmp_path / 'evidence'
    receipt = evidence / 's000-t0003600/receipt.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(dict(frame=frame, history=frame['history'],
                                      raw_retirement_permitted=True)))
    state = dict(current='next', source_grid='another.grid.nc', pending_move=None,
                 members={'old': row}, segments=[dict(member='old', rc=0, end_seconds=3600)],
                 frames=[frame], deleted=[])
    controller = SimpleNamespace(state=state, out=root, members_dir=members,
                                 a=SimpleNamespace(start_member='start'), prelaunched=None,
                                 procs={}, event=lambda *a, **k: None)
    controller.save = lambda: (root / 'controller.json').write_text(json.dumps(state))
    return controller, evidence, receipt


def test_retire_only_reconstructible_files_after_evidence(tmp_path):
    c, evidence, _ = fixture(tmp_path)
    assert artifacts_present(c.state['members']['old'])
    retire_members(c, evidence)
    row = c.state['members']['old']
    assert row['status'] == 'retired'
    assert Path(row['static']).is_file()
    assert not artifacts_present(row)
    assert not Path(row['init']).exists()
    assert not Path(row['grid']).exists()
    assert sorted(p.name for p in (c.members_dir/'operators').iterdir()) == ['start--next.npz']
    deleted = [json.loads(line) for line in (c.out/'retired-members.jsonl').read_text().splitlines()]
    assert len(deleted) == 4
    assert all(r['path'].endswith('.npz') for r in deleted[:2])
    assert json.loads((c.out/'controller.json').read_text())['members']['old']['status'] == 'retired'


@pytest.mark.parametrize('guard', ['disabled', 'current', 'source', 'start', 'pending', 'prelaunch',
                                  'worker', 'segment', 'history', 'receipt', 'wrong_frame', 'symlink'])
def test_live_or_uncollected_members_are_never_retired(tmp_path, guard):
    c, evidence, receipt = fixture(tmp_path)
    row = c.state['members']['old']
    if guard == 'disabled': evidence = None
    elif guard == 'current': c.state['current'] = 'old'
    elif guard == 'source': c.state['source_grid'] = row['grid']
    elif guard == 'start': c.a.start_member = 'old'
    elif guard == 'pending': c.state['pending_move'] = dict(target='old')
    elif guard == 'prelaunch': c.prelaunched = dict(target='old')
    elif guard == 'worker': c.procs['op:old--next'] = object()
    elif guard == 'segment': c.state['segments'][0].pop('end_seconds')
    elif guard == 'history': Path(c.state['frames'][0]['history']).write_bytes(b'not collected')
    elif guard == 'receipt': receipt.unlink()
    elif guard == 'wrong_frame':
        record = json.loads(receipt.read_text());record['frame']['member'] = 'other'
        receipt.write_text(json.dumps(record))
    elif guard == 'symlink':
        path = Path(row['grid']);path.unlink();path.symlink_to(row['static'])
    retire_members(c, evidence)
    assert row['status'] == 'stale'
    assert artifacts_present(row)
    assert not (c.out/'retired-members.jsonl').exists()


def test_interrupted_unlink_keeps_rebuild_status_and_removes_operators_first(tmp_path, monkeypatch):
    c, evidence, _ = fixture(tmp_path)
    original = Path.unlink
    def interrupted(path, *args, **kwargs):
        if str(path) == c.state['members']['old']['grid']:
            raise OSError('fixture interruption')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', interrupted)
    with pytest.raises(OSError, match='interruption'):
        retire_members(c, evidence)
    saved = json.loads((c.out/'controller.json').read_text())
    assert saved['members']['old']['status'] == 'retired'
    assert Path(saved['members']['old']['grid']).is_file()
    assert sorted(p.name for p in (c.members_dir/'operators').iterdir()) == ['start--next.npz']


def test_external_member_cache_is_never_retired(tmp_path):
    c, evidence, _ = fixture(tmp_path)
    c.out = tmp_path/'different-run'
    with pytest.raises(ValueError, match='inside this run'):
        retire_members(c, evidence)
    assert artifacts_present(c.state['members']['old'])


@pytest.mark.parametrize('used', [False, True])
def test_cancelled_speculative_files_retired_only_before_forecast_use(tmp_path, used):
    c, evidence, _ = fixture(tmp_path)
    c.state['members']['old'] = dict(status='cancelled', center_deg=[0, 0])
    c.state['source_grid'] = None
    if not used:
        c.state['frames'] = []
        c.state['segments'] = []
    partial = c.members_dir/'old.static.nc.partial'
    partial.write_bytes(b'incomplete stage')
    retire_members(c, evidence)
    assert partial.exists() == used
    assert (c.members_dir/'old.grid.nc').exists() == used
    assert (c.members_dir/'old.static.nc').exists() == used
    assert c.state['members']['old']['status'] == ('cancelled' if used else 'retired')


@pytest.mark.parametrize('guard', [None, 'disabled', 'unfinished', 'short', 'worker',
                                  'prelaunch', 'pending', 'used', 'source', 'current', 'failed'])
def test_finished_run_retires_only_unused_speculation(tmp_path, guard):
    from woof.hex.moving_retention import retire_finished_preparations
    c, evidence, _ = fixture(tmp_path)
    row = c.state['members']['old']
    row['status'] = 'ready'
    c.state.update(finished_utc='2026-10-24T00:00:00Z', hours=1, t=3600,
                   frames=[], segments=[dict(member='next', rc=0, end_seconds=3600)])
    if guard == 'disabled': evidence = None
    elif guard == 'unfinished': c.state.pop('finished_utc')
    elif guard == 'short': c.state['t'] = 3599
    elif guard == 'worker': c.procs['prep:old'] = object()
    elif guard == 'prelaunch': c.prelaunched = dict(target='old')
    elif guard == 'pending': c.state['pending_move'] = dict(target='old')
    elif guard == 'used': c.state['segments'][0]['member'] = 'old'
    elif guard == 'source': c.state['source_grid'] = row['grid']
    elif guard == 'current': c.state['current'] = 'old'
    elif guard == 'failed': c.state['segments'][0]['rc'] = 1
    retire_finished_preparations(c, evidence)
    assert row['status'] == ('retired' if guard is None else 'ready')
    assert Path(row['static']).exists() == (guard is not None)
    assert Path(row['grid']).exists() == (guard is not None)
