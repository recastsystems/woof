"""The first member can prepare before any model frame without seeding the real tracker."""
from types import SimpleNamespace
import copy

import pytest

from woof.hex import moving_run, storm_tracker


def controller():
    c = object.__new__(moving_run.Controller)
    c.a = SimpleNamespace(anticipate_moves=1)
    c.st = storm_tracker
    c.session = storm_tracker.TrackSession(storm_tracker.TrackerConfig.from_dict(dict(
        mode='tc', initial_center_deg=[20., -85.], placement=dict(interior_radius_km=146.,
        core_margin_km=70., min_move_km=40., max_move_km=200., follow='continuous'))))
    c.case = dict(storm=dict(initial_center_deg=[20., -85.],
                            initial_motion=dict(heading_deg=90., speed_kmh=20.)))
    c.state = dict(t=0., current='start', members=dict(start=dict(center_deg=[20., -85.6])))
    c.duration = 48 * 3600.
    c.plan = None
    c.known_future = lambda: False
    c.requests = []
    def request(centre, why, **options):
        c.requests.append(dict(centre=centre, why=why, **options))
        c.state['members']['predicted'] = dict(center_deg=centre)
        return 'predicted'
    c.request_member = request
    return c


def test_case_motion_requests_due_member_without_a_forecast_fix():
    c = controller()
    before = copy.deepcopy(c.session.to_json())
    c.request_startup_member()
    assert len(c.requests) == 1
    row = c.requests[0]
    assert row['speculative'] and row['order'] == 1
    assert 'for 0 s' in row['why']
    assert row['centre'][1] > -85.6
    assert 'subject to the next field fix' in row['why']
    assert c.state['members']['predicted']['planned_source'] == 'start'
    assert c.session.to_json() == before


def test_resumed_field_fix_takes_precedence_over_case_motion():
    c = controller()
    c.state['t'] = 1800.
    c.session.fixes.append(storm_tracker.Fix(seconds=1800., ok=True, lat=20., lon=-86.,
                                           steering_ms=[-5., 0.]))
    before = copy.deepcopy(c.session.to_json())
    c.request_startup_member()
    assert c.requests[0]['centre'][1] < -85.6
    assert 'restored field fixes' in c.requests[0]['why']
    assert c.session.to_json() == before


@pytest.mark.parametrize('case', ['disabled', 'known_path', 'missing_motion', 'invalid_motion', 'no_time'])
def test_no_speculation_without_a_valid_useful_prediction(case):
    c = controller()
    if case == 'disabled': c.a.anticipate_moves = 0
    if case == 'known_path': c.known_future = lambda: True
    if case == 'missing_motion': c.case['storm'].pop('initial_motion')
    if case == 'invalid_motion': c.case['storm']['initial_motion']['speed_kmh'] = float('nan')
    if case == 'no_time': c.duration = 0.
    c.request_startup_member()
    assert not c.requests


def preparation_controller(tmp_path, monkeypatch):
    c = controller()
    del c.request_member
    c.a = SimpleNamespace(anticipate_moves=1, max_preparing=1, reuse_km=20.,
        zone_carry=False, case='case.json', casedir='case', prep_threads=2,
        prep_tile_workers=2, geog=None)
    c.stitch = False
    c.state['members']['start']['status'] = 'ready'
    c.members_dir = tmp_path
    c.procs = {}
    c.events, c.started, c.stopped = [], [], []
    c.event = lambda kind, **row: c.events.append(dict(kind=kind, **row))
    c.save = lambda: None
    c.rotation_source_for = lambda *args: None
    c.stop_child = lambda proc: c.stopped.append(proc)
    def start(argv, *, stdout):
        stdout.close()
        proc = SimpleNamespace(poll=lambda: None)
        c.started.append(proc)
        return proc
    monkeypatch.setattr(moving_run, 'start_child', start)
    return c


@pytest.mark.parametrize('legacy', [False, True])
def test_future_anticipation_cannot_cancel_a_deferred_field_move(tmp_path, monkeypatch, legacy):
    c = preparation_controller(tmp_path, monkeypatch)
    needed = c.request_member([20., -85.], 'field-decided move, deferred while safe')
    if legacy:
        c.state['members'][needed].pop('speculative')
    assert c.request_member([20., -84.25], 'following anticipated move', speculative=True) is None
    assert c.state['members'][needed]['status'] == 'preparing'
    assert len(c.started) == 1 and not c.stopped
    assert c.events[-1]['kind'] == 'member-speculation-deferred'


@pytest.mark.parametrize('offset', [0., .25])
def test_a_field_decision_promotes_exact_or_nearby_speculation(tmp_path, monkeypatch, offset):
    c = preparation_controller(tmp_path, monkeypatch)
    needed = c.request_member([20., -85.], 'startup prediction', speculative=True)
    assert c.state['members'][needed]['speculative']
    assert c.request_member([20., -85.+offset], 'field-decided move', admit=lambda centre: True) == needed
    assert not c.state['members'][needed]['speculative']
    assert c.request_member([20., -84.25], 'following anticipated move', speculative=True) is None
    assert not c.stopped


def test_unneeded_speculation_can_still_be_replaced(tmp_path, monkeypatch):
    c = preparation_controller(tmp_path, monkeypatch)
    old = c.request_member([20., -85.], 'old prediction', speculative=True)
    new = c.request_member([20., -84.25], 'updated prediction', speculative=True)
    assert old != new and c.state['members'][old]['status'] == 'cancelled'
    assert c.state['members'][new]['speculative']
    assert len(c.stopped) == 1 and len(c.started) == 2
