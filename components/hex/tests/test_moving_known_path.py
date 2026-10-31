"""A moving region on a path known before the forecast runs: members along it are prepared from launch, by
rotating the premade template, and the receipt names every decided move, hold and safe-edge crossing.

THE BREAKAGE THIS PREVENTS (HEX-SHAPES-1CARD 2026-10-29, the 48 -> 3 km storm-following region on a user's
path, 306,507 cells, one RTX PRO 6000, 29x real time): the move decided at the 1800 s frame found no member, so
its preparation started then (168 s, 116 s of it regenerating the ring mesh although the start member had been
rotated from the template in 4.6 s); the forecast reached its 1 h end first, the controller held itself 170 s at
that final frame for a move it could never make, and the summary said passed with moves 0, integration_seconds 0
and overhead_fraction 1.0, with the storm 12 km past the safe edge.
"""
import json
from types import SimpleNamespace

import pytest

from woof.hex import moving_run
from woof.hex import storm_tracker as st
from woof.hex.moving_author import mesh_block
from woof.hex import moving_members as members_module


PATH = [[0.0, 36.5, -88.0], [3600.0, 37.7, -86.2]]
PLACEMENT = dict(interior_radius_km=246.0, core_margin_km=50.0, check_minutes=30.0, min_move_km=60.0,
                 max_move_km=492.0, max_dwell_minutes=180.0, follow='continuous')


def _controller(tmp_path, *, hold=1800.0):
    controller = object.__new__(moving_run.Controller)
    controller.st = st
    controller.duration = 3600.0
    controller.check = 1800.0
    controller.stitch = False
    controller.zone_transfer = False
    controller.interior_km = 246.0
    controller.track = None
    controller.prelaunched = None
    controller.case = {'mesh': mesh_block(3.0, 48.0, 300.0)}
    controller.session = st.TrackSession(st.TrackerConfig.from_dict(dict(
        mode='path', initial_center_deg=[36.5, -88.0], path=dict(rows=PATH), placement=PLACEMENT)))
    controller.a = SimpleNamespace(anticipate_moves=3, anticipate_hours=36.0, reuse_km=40.0, max_preparing=2,
                                   hold_for_member_seconds=hold, zone_carry=False, start_member='start',
                                   rotate_template=None, casedir=str(tmp_path / 'casedir'),
                                   case=str(tmp_path / 'case.json'), prep_threads=2, prep_tile_workers=2,
                                   geog=None, member_retirement_evidence=None)
    controller.members_dir = tmp_path / 'members'
    controller.members_dir.mkdir()
    controller.state = dict(current='start', t=0.0, members={'start': dict(status='ready', center_deg=[36.5, -88.0])},
                            frames=[], segments=[{}], moves=[], events=[], pending_move=None)
    controller.procs = {}
    controller.save = lambda: None
    controller.log = lambda message: None
    controller.event = lambda kind, **data: controller.state['events'].append(dict(kind=kind, **data))
    return controller


def test_the_plan_replays_the_policy_along_the_path(tmp_path):
    controller = _controller(tmp_path)
    plan = controller.plan_known_path()
    # the shapes run decided exactly this move at 1800 s ('continuous follow: move 277 km ...', mn03800e27425)
    assert [(p['at_seconds'], p['centre']) for p in plan] == [(1800.0, [38.0, -85.75])]
    assert moving_run.member_name(*plan[0]['centre']) == 'mn03800e27425'
    # the run's last frame plans nothing: no forecast is left to use a member there
    assert all(p['at_seconds'] < controller.duration for p in plan)


def test_a_planned_member_is_requested_at_launch_and_the_move_lands_on_its_frame(tmp_path):
    controller = _controller(tmp_path)
    controller.plan = controller.plan_known_path()
    requested = []

    def request(centre, why, order=1, admit=None):
        name = moving_run.member_name(*st.snap(centre[0], centre[1], 0.25))
        requested.append((name, why))
        controller.state['members'].setdefault(name, dict(status='preparing', center_deg=list(centre), order=order))
        return name

    controller.request_member = request
    controller.request_planned()            # what run() does before segment 0 starts
    assert [n for n, _ in requested] == ['mn03800e27425'] and 'planned along the known path' in requested[0][1]
    assert controller.state['members']['mn03800e27425']['planned_source'] == 'start'
    control = tmp_path / 'control'
    control.mkdir()
    (control / 'await-verdict.json').write_text('{}')
    controller.poll_background = lambda: None
    assert controller.on_frame(0.0, 'h0.nc', control) == {'stop': False}
    controller.state['members']['mn03800e27425']['status'] = 'ready'    # prepared while segment 0 ran
    verdict = controller.on_frame(1800.0, 'h1.nc', control)
    assert verdict['stop'] and verdict['target'] == 'mn03800e27425'
    assert not [e for e in controller.state['events'] if e['kind'] in ('held-for-member', 'move-deferred')]
    assert all(f['storm_to_edge_km'] > 0 for f in controller.state['frames'])


def test_the_last_frame_never_holds_or_moves(tmp_path):
    controller = _controller(tmp_path, hold=1e6)
    controller.plan = None
    controller.anticipate_chain = lambda *a: []
    controller.request_member = lambda *a, **k: pytest.fail('no member is requested at the last frame')
    controller.poll_background = lambda: None
    control = tmp_path / 'control'
    control.mkdir()
    (control / 'await-verdict.json').write_text('{}')
    controller.on_frame(0.0, 'h0.nc', control)
    controller.state['frames'][-1]['decision']['move'] = False
    verdict = controller.on_frame(3600.0, 'h2.nc', control)     # region never moved: the storm is 208 km off
    assert verdict == {'stop': False}
    frame = controller.state['frames'][-1]
    assert frame['move_wanted']['skipped'] == 'the run ends at this frame'
    assert frame['storm_to_edge_km'] < 0
    summary = moving_run.summarize(dict(controller.state, t=3600.0, hours=1.0, fixes_source='path',
                                        segments=[dict(rc=0, integration_seconds=100.0, wall_seconds=160.0)]), 246.0,
                                   mode='path')
    assert summary['frames_core_past_safe_edge'] == [3600.0] and summary['held_for_member_seconds'] == 0
    assert any('past the safe edge' in w for w in summary['warnings'])


def _family(tmp_path, case):
    casedir = tmp_path / 'casedir'
    casedir.mkdir(exist_ok=True)
    name = members_module.TEMPLATE_NAME
    (casedir / f'{name}.spec.json').write_text(json.dumps(members_module.template_spec(case)))
    (casedir / f'{name}.grid.nc').write_bytes(b'template grid')
    (casedir / f'{name}-mesh.json').write_text('{}')
    return casedir


@pytest.mark.parametrize('choice, corridor, expected', [
    (None, False, members_module.TEMPLATE_NAME),     # the default: rotate the premade template
    (True, False, members_module.TEMPLATE_NAME),
    (False, False, None),                            # --no-rotate-template generates
    (None, True, None),                              # a corridor family stitches
])
def test_runtime_members_rotate_the_premade_template_by_default(tmp_path, monkeypatch, choice, corridor, expected):
    controller = _controller(tmp_path)
    if corridor:
        controller.case['mesh']['corridor'] = {'cross_track_km': 450.0}
        controller.stitch = True
    controller.a.rotate_template = choice
    _family(tmp_path, controller.case)
    controller.rotation_source = controller.member_rotation_source()
    assert controller.rotation_source == expected
    if expected:
        assert (controller.members_dir / f'{expected}.grid.nc').read_bytes() == b'template grid'
    launched = []

    def spawn(argv, *, stdout):
        stdout.close()
        launched.append([str(a) for a in argv])
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(moving_run, 'start_child', spawn)
    controller.stitch = False
    controller.a.zone_carry = False
    controller.request_member([38.0, -85.75], 'a move')
    argv = launched[0]
    assert (argv[argv.index('--rotate-from') + 1] if '--rotate-from' in argv else None) == expected


def test_a_member_near_the_template_antipode_rotates_the_start_member(tmp_path):
    controller = _controller(tmp_path)
    _family(tmp_path, controller.case)
    controller.rotation_source = controller.member_rotation_source()
    (controller.members_dir / 'start.spec.json').write_text('{}')
    assert controller.rotation_source_for(0.5, 179.5) == 'start'
    assert controller.rotation_source_for(38.0, -85.75) == members_module.TEMPLATE_NAME


def test_the_operator_starts_once_both_meshes_exist(tmp_path, monkeypatch):
    controller = _controller(tmp_path)
    controller.state['members']['start'].update(grid=str(tmp_path / 'start.grid.nc'))
    (tmp_path / 'start.grid.nc').write_bytes(b'g')
    controller.state['members']['next'] = dict(status='preparing', center_deg=[38.0, -85.75], order=1,
                                               planned_source='start')
    started = []
    monkeypatch.setattr(moving_run, 'start_child', lambda argv, *, stdout: started.append(argv) or SimpleNamespace(
        poll=lambda: None))
    controller.start_operators_early()
    assert started == []                                    # its mesh stage is still open
    (controller.members_dir / 'next.grid.nc').write_bytes(b'g')
    (controller.members_dir / 'next-mesh.json').write_text('{}')
    controller.start_operators_early()
    assert len(started) == 1 and 'op:start--next' in controller.procs
    assert str(controller.members_dir / 'next.grid.nc') in [str(a) for a in started[0]]


def test_the_mpas_route_segment_reports_its_stepping(tmp_path):
    (tmp_path / 'fc').mkdir()
    (tmp_path / 'control').mkdir()
    (tmp_path / 'fc' / 'native-physics-receipt.json').write_text(json.dumps({'moving_control': {}}))
    rows = [{'model_seconds': 0.0, 'history': 'a', 'wall_utc': 1000.0},
            {'verdict': {'stop': False}, 'model_seconds': 0.0, 'waited_seconds': 0.5},
            {'model_seconds': 1800.0, 'history': 'b', 'wall_utc': 1046.0},
            {'verdict': {'stop': False}, 'model_seconds': 1800.0, 'waited_seconds': 1.5},
            {'model_seconds': 3600.0, 'history': 'c', 'wall_utc': 1109.0},
            {'verdict': {'stop': False}, 'model_seconds': 3600.0, 'waited_seconds': 170.0}]
    (tmp_path / 'control' / 'progress.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out = moving_run.segment_receipt(tmp_path)
    assert out['route'] == 'mpas-suite' and out['integration_seconds'] == pytest.approx(107.0)
    assert out['verdict_wait_seconds'] == pytest.approx(2.0)


def test_the_summary_of_the_shapes_run_names_its_lost_move(tmp_path):
    """The receipt of the 1 h path run as it was written (controller.json fields)."""
    state = dict(t=3600.0, hours=1.0, wall_seconds=312.5, fixes_source="the user's path (not a tracker)", moves=[],
                 pending_move=dict(target='mn03800e27425', decided_at_seconds=3600.0, reason='continuous follow'),
                 events=[dict(kind='move-deferred', name='mn03800e27425', at_seconds=1800.0),
                         dict(kind='held-for-member', name='mn03800e27425', seconds=170.29)],
                 frames=[dict(model_seconds=s, offset_km=o, fix=dict(ok=True)) for s, o in
                         ((0.0, 0.0), (1800.0, 104.0), (3600.0, 208.06))],
                 segments=[dict(rc=0, wall_seconds=312.48, route='mpas-suite')])
    summary = moving_run.summarize(state, 246.0, mode='path')
    assert summary['integration_seconds'] is None and summary['overhead_fraction'] is None   # unknown, not 100 %
    assert summary['held_for_member_seconds'] == pytest.approx(170.29)
    assert [(m['at_seconds'], m['member'], m['landed_at_seconds']) for m in summary['moves_decided']] == [
        (1800.0, 'mn03800e27425', None), (3600.0, 'mn03800e27425', None)]
    assert any('held 170 s' in w for w in summary['warnings'])


@pytest.mark.parametrize('ready_after_polls, verdict_stop', [(2, True), (None, False)])
def test_a_move_due_before_the_last_check_waits_instead_of_deferring(tmp_path, monkeypatch, ready_after_polls,
                                                                     verdict_stop):
    """Rerun of the shapes path job on fix/hex-movpath-prep 1c1bdefb: the member prepared from launch was ready
    12 s after the 1800 s frame; deferring to the 3600 s end frame lost the move (storm 12.1 km past the edge)."""
    controller = _controller(tmp_path, hold=1800.0 if ready_after_polls else 1e-9)
    controller.plan = None
    controller.anticipate_chain = lambda *a: []
    monkeypatch.setattr(moving_run.time, 'sleep', lambda seconds: None)
    polls = []

    def request(centre, why, order=1, admit=None):
        controller.state['members']['next'] = dict(status='preparing', center_deg=list(centre))
        return 'next'

    def poll():
        polls.append(1)
        if ready_after_polls and len(polls) >= ready_after_polls:
            controller.state['members']['next']['status'] = 'ready'

    controller.request_member = request
    controller.poll_background = poll
    control = tmp_path / 'control'
    control.mkdir()
    (control / 'await-verdict.json').write_text('{}')
    controller.on_frame(0.0, 'h0.nc', control)
    verdict = controller.on_frame(1800.0, 'h1.nc', control)
    assert verdict['stop'] is verdict_stop
    kinds = [e['kind'] for e in controller.state['events']]
    assert 'move-deferred' not in kinds
    if not verdict_stop:     # never ready: the storm is safe to the end, so the run finishes on this region
        assert [e for e in controller.state['events'] if e['kind'] == 'move-member-unavailable'][0]['continued']


def test_a_move_segment_steps_from_its_construction_end(tmp_path):
    """Its begin frame (1800 s) was decided on the previous member, so its progress has the 3600 s frame only
    (the second rerun: integration_seconds came out unknown)."""
    (tmp_path / 'fc').mkdir()
    (tmp_path / 'control').mkdir()
    (tmp_path / 'fc' / 'native-physics-receipt.json').write_text(json.dumps({'moving_control': {}}))
    (tmp_path / 'command.json').write_text(json.dumps({'begin_seconds': 1800.0}))
    (tmp_path / 'fc' / 'timeline.jsonl').write_text(
        json.dumps({'t_utc': '2026-10-29T09:11:44.337057Z', 'stage': 'forecast.construct', 'phase': 'begin'}) + '\n'
        + json.dumps({'t_utc': '2026-10-29T09:15:12.892000Z', 'stage': 'forecast.construct', 'phase': 'end'}) + '\n')
    from datetime import datetime, timezone
    frame = datetime(2026, 10, 29, 9, 16, 11, 352466, tzinfo=timezone.utc).timestamp()
    (tmp_path / 'control' / 'progress.jsonl').write_text(json.dumps(
        {'model_seconds': 3600.0, 'history': 'c', 'wall_utc': frame}) + '\n')
    out = moving_run.segment_receipt(tmp_path)
    assert out['integration_seconds'] == pytest.approx(58.46, abs=0.01)
    assert 'construction' in out['integration_basis']
