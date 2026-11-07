"""Hold the first model fix, and release owned children before the card grant."""
from types import SimpleNamespace
import pytest
from woof.hex import moving_run


@pytest.mark.parametrize('status,newer,artifacts,expected', [
    ('failed', True, True, 'prepared'), ('failed', False, True, 'failed'),
    ('failed', True, False, 'failed'), ('preparing', False, True, 'cancelled'),
    ('preparing', True, True, 'prepared'), ('unstable', True, True, 'unstable'),
])
def test_resume_adopts_only_a_completed_newer_preparation(tmp_path, monkeypatch, status, newer, artifacts, expected):
    import json
    import os
    from woof.hex import moving_retention
    receipt = tmp_path / 'target-case.json'
    receipt.write_text(json.dumps({'name': 'target'}))
    os.utime(receipt, (200 if newer else 50,) * 2)
    monkeypatch.setattr(moving_retention, 'artifacts_present', lambda row: artifacts)
    controller = object.__new__(moving_run.Controller)
    controller.members_dir = tmp_path
    controller.state = {'members': {'target': dict(status=status, requested_wall=100, order=1)}}
    controller.read_member = lambda name, status: dict(name=name, status=status)
    events = []
    controller.event = lambda kind, **row: events.append(kind)
    controller.recover_preparations()
    assert controller.state['members']['target']['status'] == expected
    assert events == (['member-prep-recovered'] if expected == 'prepared' else [])

@pytest.mark.parametrize('lost_controller', [False, True])
def test_resume_rewinds_future_tracking_before_launch(tmp_path, lost_controller):
    controller = object.__new__(moving_run.Controller)
    record = dict(path=str(tmp_path))
    if not lost_controller:
        record['rc'] = -15
    controller.a = SimpleNamespace(resume=True, member_fallback=False)
    controller.duration = 21600
    controller.state = dict(t=19800, t_seen=21600, segments=[record],
        frames=[dict(model_seconds=t) for t in (0, 19800, 20700, 21600)],
        pending_move=dict(target='discarded-future'))
    controller.session = SimpleNamespace(
        fixes=[SimpleNamespace(seconds=t) for t in (0, 19800, 20700, 21600)],
        decisions=[SimpleNamespace(seconds=t) for t in (0, 19800, 20700, 21600)])
    controller.newest_seal = lambda _: None
    events = []
    controller.event = lambda kind, **data: events.append((kind, data))
    controller.save = lambda: None
    class LaunchReached(Exception): pass
    def launch(kind):
        assert [f['model_seconds'] for f in controller.state['frames']] == [0, 19800]
        assert [f.seconds for f in controller.session.fixes] == [0, 19800]
        assert [d.seconds for d in controller.session.decisions] == [0, 19800]
        assert controller.state['t_seen'] == 19800
        assert controller.state['pending_move'] is None
        # The replayed +5.75 h frame can now pass on_frame's duplicate gate.
        assert 20700 not in {f['model_seconds'] for f in controller.state['frames']}
        raise LaunchReached
    controller.launch = launch
    with pytest.raises(LaunchReached): controller.run()
    assert events == [('rewind-tracking-to-carrier', dict(t=19800, frames_discarded=2))]

@pytest.mark.parametrize('offsets,passed', [([9.9, 86.469241315], False),
                                        ([9.9, None], False), ([9.9, 40], True)])
def test_completed_forecast_requires_every_fix_inside_fine_interior(offsets, passed):
    state = dict(t=21600, hours=6, wall_seconds=1500, moves=[],
                 fixes_source='run-time tracker on model state',
                 frames=[dict(offset_km=offset, fix=dict(ok=True)) for offset in offsets],
                 segments=[dict(rc=0, integration_seconds=1000, wall_seconds=1400)])
    summary = moving_run.summarize(state, 82)
    assert (summary['status'] == 'passed') == passed
    assert summary['a2_storm_inside_interior_every_frame'] == passed

@pytest.mark.parametrize('offsets,passed', [([None, None, None], True), ([9.9, None, 40], True),
                                            ([9.9, None, 86.5], False)])
def test_setup_target_holds_its_region_on_frames_with_no_fix(offsets, passed):
    """Owner canary run_6fe920b4414c6b7c: GFS 18Z f354 had no CAPE x shear area all 24 h; the run must finish."""
    state = dict(t=86400, hours=24, wall_seconds=1500, moves=[], fixes_source='run-time tracker on model state',
                 frames=[dict(offset_km=o, fix=dict(ok=o is not None)) for o in offsets],
                 segments=[dict(rc=0, integration_seconds=1000, wall_seconds=1400)])
    summary = moving_run.summarize(state, 82, mode='cape_shear')
    assert (summary['status'] == 'passed') == passed
    assert summary['held_frames'] == offsets.count(None)
    if passed and offsets.count(None):
        assert 'cape_shear' in summary['note']
    assert moving_run.summarize(state, 82, mode='tc')['status'] == 'incomplete'


@pytest.mark.parametrize('error',[KeyboardInterrupt,RuntimeError,SystemExit])
def test_failed_controller_stops_all_owned_child_groups(monkeypatch,error):
    prep,forecast=object(),object()
    def run():raise error('interrupted')
    controller=SimpleNamespace(run=run,procs={'prep:x':prep},forecast_children=[forecast])
    monkeypatch.setattr(moving_run,'Controller',lambda _:controller)
    stopped=[]
    monkeypatch.setattr(moving_run,'stop_process_tree',stopped.append)
    with pytest.raises(error):moving_run.run_controller(None)
    assert stopped==[prep,forecast]


@pytest.mark.parametrize('previous', [[], [{'decision': {'move': False, 'dwell_minutes': 600}}]])
def test_fix_holds_even_when_previous_fix_predicts_no_near_move(tmp_path, previous):
    control=tmp_path/'control';control.mkdir()
    (control/'progress.jsonl').write_text('{"model_seconds":0,"history":"initial.nc"}\n')
    observed=[]
    def on_frame(*args):observed.append((control/'await-verdict.json').exists())
    fake=SimpleNamespace(poll_background=lambda:None,ready_members=lambda:{},
        a=SimpleNamespace(hold_for_member_seconds=10,verdict_timeout=1,await_within_minutes=5,segment_timeout=60),
        state={'frames':previous,'members':{'start':{'status':'ready'}},'current':'start'},on_frame=on_frame)
    proc=SimpleNamespace(poll=lambda:0,returncode=0)
    assert moving_run.Controller.monitor(fake,proc,tmp_path,control)==0
    assert observed==[True]


def test_missing_controller_reply_cannot_advance_forecast(tmp_path):
    from woof.hex.moving_control import MovingControl
    (tmp_path/'await-verdict.json').write_text('{"timeout_seconds":0}')
    with pytest.raises(TimeoutError, match='did not answer frame 21600'):
        MovingControl(tmp_path).at_frame(model_seconds=21600, final=False, seal=lambda _: {})


def test_control_without_hold_request_does_not_wait(tmp_path):
    from woof.hex.moving_control import MovingControl
    assert not MovingControl(tmp_path).at_frame(model_seconds=21600, final=False, seal=lambda _: {})


@pytest.mark.parametrize('hold', [0, 10])
def test_initial_hold_exists_before_forecast_can_publish(monkeypatch, tmp_path, hold):
    member = {'name': 'start', 'center_deg': [0, 0]}
    controller = SimpleNamespace(
        out=tmp_path, duration=60, case={}, init_source='fixture', forecast_children=[],
        state={'segments': [], 'members': {'start': member}, 'current': 'start',
               't': 0, 'boundary': None, 'operator': None, 'source_grid': None},
        a=SimpleNamespace(hold_for_member_seconds=hold, verdict_timeout=1,
                          check_minutes=1, engine='arwen', forecast_arg=[]),
        segment_env=lambda *args: {}, save=lambda: None, log=lambda message: None)
    proc = SimpleNamespace()

    def spawn(command, *, stdout, env, cwd):
        assert (tmp_path / 'segment-000/control/await-verdict.json').exists() == bool(hold)
        stdout.close()
        return proc

    monkeypatch.setattr(moving_run, 'segment_command', lambda *args, **kwargs: ['forecast'])
    monkeypatch.setattr(moving_run, 'start_child', spawn)
    controller.allocate_segment = lambda: moving_run.Controller.allocate_segment(controller)
    moving_run.Controller.launch(controller, kind='start')
    assert controller.forecast_children == [proc]


def _anticipation_controller():
    from woof.hex import storm_tracker as st
    controller = object.__new__(moving_run.Controller)
    controller.st = st
    controller.duration = 1800
    controller.stitch = False
    controller.a = SimpleNamespace(anticipate_moves=2, zone_carry=False, anticipate_hours=24)
    controller.session = st.TrackSession(st.TrackerConfig(initial_center_deg=[0, 0], placement=st.PlacementConfig(
        interior_radius_km=81, core_margin_km=10, min_move_km=30,
        follow='continuous', check_minutes=5, snap_deg=0.05,
        max_move_km=60, continuous_lead_minutes=120)))
    controller.session.fixes = [st.Fix(seconds=0, ok=True, lat=0, lon=0),
                                st.Fix(seconds=300, ok=True, lat=0, lon=-0.02)]
    return controller


@pytest.mark.parametrize('zone', [False, True])
def test_return_to_retired_location_preserves_earlier_geometry(tmp_path, monkeypatch, zone):
    import json
    controller = _anticipation_controller()
    controller.a = SimpleNamespace(zone_carry=zone, reuse_km=25, max_preparing=1,
        case=tmp_path/'case.json', casedir=tmp_path, prep_threads=2,
        prep_tile_workers=2, geog=None, rotate_template=False)
    controller.members_dir = tmp_path
    controller.state = dict(current='start', members={}, frames=[], t_seen=7200)
    name = controller.member_key(0, 1)
    old = dict(status='retired', center_deg=[0, 1])
    # A crash after persisting retirement may leave all artifacts present.
    # They must not be reused under the retired identity or overwritten later.
    for key in ('grid', 'static', 'init'):
        path = tmp_path/f'{name}.{key}.nc'; path.write_bytes(b'earlier geometry')
        old[key] = str(path)
    (tmp_path/f'{name}-case.json').write_text(json.dumps(old))
    controller.state['members'][name] = old
    controller.state['frames'].append(dict(member=name))
    controller.procs = {}
    controller.save = lambda: None
    controller.event = lambda *a, **k: None
    launched = []
    def spawn(argv, *, stdout):
        stdout.close(); launched.append(argv)
        return SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(moving_run, 'start_child', spawn)
    replacement = controller.request_member([0, 1], 'returned after retirement')
    assert replacement != name
    assert controller.state['members'][name] is old
    assert controller.state['frames'][0]['member'] == name
    assert (tmp_path/f'{name}.static.nc').read_bytes() == b'earlier geometry'
    assert launched[0][launched[0].index('--name')+1] == replacement
    if zone:
        assert replacement.endswith('xstart')
    assert controller.request_member([0, 1], 'same pending preparation') == replacement
    assert len(launched) == 1
    controller.state['members'][replacement]['status'] = 'failed'
    controller.procs.clear()
    neighbour = controller.request_member([0, 1], 'retry location after preparation refusal')
    assert neighbour not in (None, name, replacement)
    assert controller.state['members'][name] is old
    assert len(launched) == 2


@pytest.mark.parametrize('anticipated_seconds', [1800, 2700])
def test_no_preparation_for_a_move_after_forecast_ends(anticipated_seconds):
    controller = _anticipation_controller()
    controller.session.policy.anticipate = lambda *a, **kw: {'at_seconds': anticipated_seconds}
    decision = controller.st.Decision(300, False, [0, 0], 600, 0, 'test')
    assert controller.anticipate_chain(decision, {'center_deg': [0, 0]}) == []


def test_anticipation_chain_stops_at_forecast_end():
    controller = _anticipation_controller()
    controller.session.policy.anticipate = lambda *a, **kw: {'at_seconds': 900, 'centre': [0, 0]}
    controller.session.policy.dwell = lambda *a: 20
    decision = controller.st.Decision(300, False, [0, 0], 600, 0, 'test')
    chain = controller.anticipate_chain(decision, {'center_deg': [0, 0]})
    assert [row['at_seconds'] for row in chain] == [900]


def test_zero_anticipated_moves_does_not_prepare_next_member():
    controller = SimpleNamespace(a=SimpleNamespace(anticipate_moves=0))
    assert moving_run.Controller.anticipate_chain(controller, None, None) == []


def test_decided_move_waits_for_current_centre_not_next_anticipated_move(tmp_path):
    from woof.hex.storm_tracker import Fix, Decision, PlacementPolicy, PlacementConfig, MotionModel
    controller = object.__new__(moving_run.Controller)
    current = [24.4, -83.0]
    chosen = [24.75, -83.0]  # emergency move39km is below the usual41km minimum
    future = [25.25, -83.0]  # ready but outside the fix's safe core radius
    fix = Fix(seconds=7200, ok=True, lat=24.733039782250614, lon=-83.00713393179386)
    decision = Decision(7200, True, chosen, 10800, 60, 'current region unsafe')
    controller.state = dict(current='start', members={
        'start': dict(status='ready', center_deg=current),
        'future': dict(status='ready', center_deg=future)}, frames=[], segments=[{}])
    controller.interior_km = 82
    controller.session = SimpleNamespace(
        policy=PlacementPolicy(PlacementConfig(interior_radius_km=82, core_margin_km=27.333333333333332)),
        motion=MotionModel(), fixes=[fix], decide=lambda *args, **kwargs: decision)
    controller.a = SimpleNamespace(hold_for_member_seconds=1800)
    controller.observe = lambda *args: (fix, 0.1)
    controller.anticipate_chain = lambda *args: [dict(centre=future, at_seconds=9000)]
    requested = []
    def request(centre, why, admit=None):
        assert admit is not None and admit(chosen)     # the due move asks with pick_target's admission
        requested.append(centre)
        controller.state['members']['chosen'] = dict(status='prepared', center_deg=centre)
        return 'chosen'
    controller.request_member = request
    def poll():
        if 'chosen' in controller.state['members']:
            controller.state['members']['chosen']['status'] = 'ready'
    controller.poll_background = poll
    controller.event = lambda *args, **kwargs: None
    controller.save = lambda: None
    controller.log = lambda message: None
    (tmp_path/'await-verdict.json').write_text('{}')
    verdict = controller.on_frame(7200, 'history.nc', tmp_path)
    assert requested == [chosen]
    assert verdict == dict(stop=True, target='chosen', reason=decision.reason)
    assert controller.state['pending_move']['target'] == 'chosen'


def test_resume_redraws_products_that_died_with_the_controller(tmp_path):
    def segment(name, products_tail, rc=0):
        root = tmp_path / name
        fc = root / 'fc'
        (fc / 'products.products-scratch' / 'render-global').mkdir(parents=True)
        (fc / 'products.log').write_text('PRODUCTS view=global window=global\n' + products_tail)
        (root / 'forecast.log').write_text(
            'x\nPRODUCTS $ woof hex products --forecast ' + str(fc) + ' --out ' + str(fc / 'products') + '\n')
        return dict(path=str(root), rc=rc)
    killed = segment('segment-000', 'DOOR rendered=84 skipped=2 failed=0 out=x\n')
    finished = segment('segment-001', 'PRODUCTS done maps=307 point_pictures=1 failed_views=none manifest=m\n')
    failed = segment('segment-002', '', rc=2)
    started, logged = [], []

    def popen(argv, **kw):
        started.append((argv, kw['env']['CUDA_VISIBLE_DEVICES']))
        return SimpleNamespace(pid=7)
    redrawn = moving_run.restart_dead_products([killed, finished, failed], logged.append,
                                               popen=popen, alive=lambda fc: False)
    fc = tmp_path / 'segment-000' / 'fc'
    assert redrawn == [str(fc)]
    assert started == [([moving_run.sys.executable, '-m', 'woof.hex.cli', 'products', '--forecast', str(fc),
                         '--out', str(fc / 'products')], '')]
    assert not (fc / 'products.products-scratch').exists()
    assert (tmp_path / 'segment-001' / 'fc' / 'products.products-scratch').exists()
    assert moving_run.restart_dead_products([killed], logged.append, popen=popen,
                                            alive=lambda fc: True) == []


def test_moving_products_waits_for_and_redraws_segment_maps(tmp_path):
    """Box L 2026-10-24: the last segment's detached maps were stopped with the leased tree and never drawn."""
    import json
    from woof.hex.moving_products import finish_segment_maps
    segments = []
    for name, text in (('done', 'PRODUCTS done maps=9\n'), ('drawing', 'CONVERT\n'), ('dead', 'CONVERT\n')):
        fc = tmp_path / name / 'fc'
        fc.mkdir(parents=True)
        (fc / 'products.log').write_text(text)
        segments.append(dict(path=str(tmp_path / name), rc=0))
    (tmp_path / 'controller.json').write_text(json.dumps(dict(segments=segments)))
    calls = {'drawing': 0}

    def alive(fc):
        if fc.parent.name != 'drawing':
            return fc.parent.name == 'dead' and calls.get('redrawn')
        calls['drawing'] += 1
        if calls['drawing'] > 2:                       # finishes after a couple of polls
            (fc / 'products.log').write_text('PRODUCTS done maps=9\n')
            return False
        return True

    def restart(records, log):
        calls['redrawn'] = True
        (tmp_path / 'dead' / 'fc' / 'products.log').write_text('PRODUCTS done maps=9\n')
        return [str(tmp_path / 'dead' / 'fc')]

    result = finish_segment_maps(tmp_path, poll_seconds=0.0, alive=alive, restart=restart, log=lambda m: None)
    assert result['waited'] == [str(tmp_path / 'drawing' / 'fc')]
    assert result['redrawn'] == [str(tmp_path / 'dead' / 'fc')]
    assert result['unfinished'] == []


def test_resume_preserves_unadopted_segment_without_skipping_index(tmp_path):
    controller = object.__new__(moving_run.Controller)
    controller.out = tmp_path
    controller.state = {'segments': [{'path': str(tmp_path / 'segment-000')}]}
    orphan = tmp_path / 'segment-001'
    orphan.mkdir()
    (orphan / 'command.json').write_text('interrupted preparation')
    events = []
    controller.event = lambda kind, **row: events.append((kind, row))
    index, segment = controller.allocate_segment()
    assert index == 1
    assert segment == orphan and list(segment.iterdir()) == []
    preserved = list((tmp_path / 'orphaned-segments').glob('*/command.json'))
    assert len(preserved) == 1
    assert preserved[0].read_text() == 'interrupted preparation'
    assert events[0][0] == 'preserve-unadopted-segment'
    # A second interrupted preparation is retained separately, without overwriting evidence.
    (segment / 'command.json').write_text('second interruption')
    assert controller.allocate_segment()[0] == 1
    assert len(list((tmp_path / 'orphaned-segments').iterdir())) == 2


def test_segment_allocation_refuses_to_move_recorded_directory(tmp_path):
    controller = object.__new__(moving_run.Controller)
    controller.out = tmp_path
    recorded = tmp_path / 'segment-001'
    recorded.mkdir()
    controller.state = {'segments': [{'path': str(recorded)}]}
    with pytest.raises(RuntimeError, match='already recorded'):
        controller.allocate_segment()
    assert recorded.is_dir()
    assert not (tmp_path / 'orphaned-segments').exists()
