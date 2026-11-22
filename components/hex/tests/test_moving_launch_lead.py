"""A forced region move prepares early and waits for a member that keeps the storm safe.

Replay the controller with opposed steering and observed storm motion, a simulated
wall clock, and measured preparation durations. The earlier last-check cap refused
after 30 seconds while the required member was still initializing.
"""
import inspect
from types import SimpleNamespace

import pytest

from woof.hex import moving_run
from woof.hex import storm_tracker as st

START = [38.25, -98.9]
PLACEMENT = dict(interior_radius_km=198.0, core_margin_km=150.0, check_minutes=3.75, min_move_km=20,
                 max_move_km=90, continuous_lead_minutes=30, follow='continuous')
TRACK_FIXES = {900.0: (37.310015330548296, -98.87208427031194), 1125.0: (37.35, -98.89), 1350.0: (37.39, -99.16),
             1575.0: (37.34, -99.12)}
STEERING = {900.0: [15.809768286429055, 15.893254787252559]}
FRAME_WALL = {1125.0: 26.0, 1350.0: 30.0, 1575.0: 34.0}      # resume 09:46:10, frames 09:46:36/40/45
ROTATE, STATIC, INIT, OPERATOR = 0.6, 25.0, 40.0, 4.0
# The cycle IAU gate (no move before the analysis increment's window closes) may be carried by a cycling controller; a
# tree without it lands the forced move at the first frame instead of the first frame after the window.
LANDING = 1575.0 if 'cycle_iau_until' in inspect.getsource(moving_run.Controller.on_frame) else 1125.0


class Clock:
    def __init__(self):
        self.now = 0.0

    def sleep(self, seconds):
        self.now += seconds


class FakeProc:
    """A member preparation or operator build that closes its stages at simulated wall times."""

    def __init__(self, clock, stages):
        self.clock, self.stages, self.killed = clock, sorted(stages, key=lambda s: s[0]), False

    def poll(self):
        if self.killed:
            return -15
        while self.stages and self.clock.now >= self.stages[0][0]:
            self.stages.pop(0)[1]()
        return None if self.stages else 0


def _controller(tmp_path, monkeypatch, clock, fixes):
    c = object.__new__(moving_run.Controller)
    c.st = st
    c.duration, c.check = 1800.0, 225.0
    c.stitch = c.zone_transfer = False
    c.interior_km = PLACEMENT['interior_radius_km']
    c.track, c.prelaunched, c.plan, c.rotation_source = None, None, None, None
    if LANDING > 1125.0:
        c.cycle_iau_until = 900.0 + 600.0     # the analysis at 900 s, IAU window 600 s
    c.session = st.TrackSession(st.TrackerConfig.from_dict(dict(
        mode='object', initial_center_deg=START, placement=PLACEMENT)))
    c.a = SimpleNamespace(anticipate_moves=3, anticipate_hours=36.0, reuse_km=40.0, max_preparing=2,
                          hold_for_member_seconds=1800.0, zone_carry=False, start_member='start',
                          rotate_template=None, casedir=str(tmp_path), case=str(tmp_path / 'case.json'),
                          prep_threads='32', prep_tile_workers='8', geog=None, member_retirement_evidence=None)
    c.members_dir = tmp_path / 'members'
    c.members_dir.mkdir()
    (tmp_path / 'start.grid.nc').write_text('grid')
    c.state = dict(current='start', t=900.0, frames=[], segments=[{}], moves=[], events=[], pending_move=None,
                   members={'start': dict(status='ready', center_deg=START, grid=str(tmp_path / 'start.grid.nc'))})
    c.procs = {}
    c.save = lambda: None
    c.log = lambda text: None
    c.event = lambda kind, **data: c.state['events'].append(dict(kind=kind, wall=clock.now, **data))
    centres = {}
    c.read_member = lambda name, status: dict(name=name, center_deg=centres[name], status=status,
                                              grid=str(c.members_dir / f'{name}.grid.nc'))

    def stop_child(proc):
        proc.killed = True
    c.stop_child = stop_child

    def start_child(argv, *, stdout, env=None, cwd=None):
        argv = [str(a) for a in argv]
        t0 = clock.now
        if 'prepare-member' in argv:
            name = argv[argv.index('--name') + 1]
            centres[name] = [float(argv[argv.index('--lat') + 1]), float(argv[argv.index('--lon') + 1])]
            mesh = lambda: [(c.members_dir / f'{name}{s}').write_text('x') for s in ('.grid.nc', '-mesh.json')]
            case = lambda: (c.members_dir / f'{name}-case.json').write_text('{}')
            return FakeProc(clock, [(t0 + ROTATE, mesh), (t0 + ROTATE + STATIC + INIT, case)])
        path = moving_run.Path(argv[-1])
        return FakeProc(clock, [(t0 + OPERATOR, lambda: path.write_text('op'))])

    monkeypatch.setattr(moving_run, 'start_child', start_child)
    monkeypatch.setattr(moving_run.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(moving_run.time, 'time', lambda: 1.8e9 + clock.now)
    monkeypatch.setattr(moving_run.time, 'sleep', clock.sleep)

    def observe(seconds, history):
        lat, lon = fixes[seconds]
        fix = st.Fix(seconds=seconds, ok=True, lat=lat, lon=lon, steering_ms=STEERING.get(seconds, [0.0, 0.0]))
        c.session.fixes.append(fix)
        return fix, 0.0
    c.observe = observe
    observe(900.0, None)                     # the bootstrap fix the resumed session carries
    return c


def _replay(tmp_path, monkeypatch, fixes=TRACK_FIXES):
    clock = Clock()
    c = _controller(tmp_path, monkeypatch, clock, fixes)
    launch = getattr(c, 'prepare_at_launch', lambda: None)      # what run() does before the resumed segment
    launch()
    control = tmp_path / 'control'
    control.mkdir()
    (control / 'await-verdict.json').write_text('{}')
    verdicts = {}
    for seconds, wall in FRAME_WALL.items():
        while clock.now < wall:                                  # monitor polls every 2 s between frames
            clock.sleep(2.0)
            c.poll_background()
        clock.now = wall
        verdicts[seconds] = c.on_frame(seconds, f'history-{seconds:.0f}.nc', control)
        if verdicts[seconds]['stop']:
            break                                                # the forecast stops here for the move
    return c, clock, verdicts


def test_g03_replay_reproduces_the_box_geometry(tmp_path, monkeypatch):
    clock = Clock()
    c = _controller(tmp_path, monkeypatch, clock, TRACK_FIXES)
    edge = c.session.policy.c.safe_km()
    for seconds, logged in ((1125.0, -51.6), (1350.0, -50.8), (1575.0, -54.6)):
        offset = moving_run.great_circle_km(TRACK_FIXES[seconds], START)
        assert edge - offset == pytest.approx(logged, abs=1.5)


def test_forced_move_lands_on_the_member_started_at_launch(tmp_path, monkeypatch):
    c, clock, verdicts = _replay(tmp_path, monkeypatch)
    events = c.state['events']
    kinds = [e['kind'] for e in events]
    # Before the fix this raised at 1575 s: move-member-unavailable after a 30 s hold (member in its init).
    assert 'move-member-unavailable' not in kinds
    lead = [e for e in events if e['kind'] == 'launch-lead']
    assert lead and lead[0]['launch_seconds'] == 900.0 and lead[0]['at_seconds'] == LANDING
    assert lead[0]['storm_to_edge_km'] < 0 and lead[0]['wall'] == 0.0      # started beside the launch
    assert all(verdicts[s] == {'stop': False} for s in verdicts if s < LANDING)     # inside the IAU window
    move = verdicts[LANDING]
    assert move['stop'] is True
    target = c.state['members'][move['target']]
    # the member of the move started before the move's own frame (on the box: at it), and was never cancelled
    started = [e['wall'] for e in events if e['kind'] == 'member-prep-start' and e['name'] == move['target']]
    assert started and started[0] < FRAME_WALL[LANDING]
    assert not [e for e in events if e['kind'] == 'member-prep-cancelled' and e['name'] == move['target']]
    # the storm is inside the new region's safe radius by more than the fix error
    inside = c.session.policy.c.safe_km() - moving_run.great_circle_km(TRACK_FIXES[LANDING], target['center_deg'])
    assert inside > c.session.policy.c.fix_error_km
    held = sum(e['seconds'] for e in events if e['kind'] == 'held-for-member')
    assert held == pytest.approx(started[0] + ROTATE + STATIC + INIT - FRAME_WALL[LANDING], abs=2.5)


def test_a_storm_well_inside_starts_nothing_at_launch(tmp_path, monkeypatch):
    inside = {s: (START[0] - 0.05, START[1]) for s in (900.0, 1125.0, 1350.0, 1575.0)}
    c, clock, verdicts = _replay(tmp_path, monkeypatch, fixes=inside)
    kinds = [e['kind'] for e in c.state['events']]
    assert 'launch-lead' not in kinds and 'member-prep-start' not in kinds
    assert all(v == {'stop': False} for v in verdicts.values())
