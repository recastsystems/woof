"""Closed-loop controller for one moving fine region (one card): ``woof hex moving run``.

  woof hex moving run --case CASE.json --casedir DIR --out RUN --hours H
                       [--members DIR] [--device N] [--check-minutes 60]
                       [--tracker-config JSON] [--scripted-track JSON]
                       [--seal-every-hours 12] [--resume]

One forecast process runs on the current mesh member until a move is due.  At every
history frame (``--check-minutes``) the controller reads the model's own state from
the frame, fixes the storm with the run-time tracker (``woof.hex.storm_tracker``: the
predictive TrackSession -- observe the frame, decide, anticipate), and

* starts preparing the member the tracker anticipates (mesh, static, init on the CPU,
  in the background, while the card keeps integrating), then builds the conservative
  overlap operator from the current member to it;
* while a prepared member is ready, holds the forecast at each frame for a verdict
  (``woof.hex.moving_control``), so a move the tracker decides on a frame happens on
  that frame;
* on a move, the forecast seals its state at that frame and the next forecast process
  continues on the new member through the conservative remap.

A crash restarts the member from its newest recovery seal (``--seal-every-hours``) or
from the segment's own starting carrier.  ``--resume`` continues a run whose controller
died, from its saved state.  ``--scripted-track`` replaces the fixes (never the policy)
with an externally given track, for machinery tests only; the receipt says so.

Cases, storms and positions come only from the case file and the tracker config.

Every child is this interpreter running this package (``python -m woof.hex.cli``):
member preparation (``moving prepare-member``), operators (``moving operator`` /
``moving zone``) and forecast segments (``forecast``).  Children start in their
own process group, so stopping one stops its whole tree by its own PID.  The
card is the one CUDA makes visible (``--device N`` picks the Nth card already visible, so a card
queue's lease holds, :func:`visible_devices_for`); no queue or host layout is assumed.  Moved here from the source
tree's ``tools/moving48/longrun.py``, which ran only under one rented host's
card queue.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


from .moving_case import great_circle_km, interior_km, load_case, member_name

SCHEMA = 'gpuwm-hex.moving-longrun/v1'
RUN_STATE_VARIABLES = ('WOOF_HEX_MOVING_CHECKPOINT_IN', 'WOOF_HEX_MOVING_CHECKPOINT_OUT',
                       'WOOF_HEX_MOVING_CHECKPOINT_WAIT', 'WOOF_HEX_MOVING_CONTROL', 'WOOF_HEX_MOVING_OPERATOR',
                       'WOOF_HEX_MOVING_SOURCE_GRID', 'WOOF_HEX_MOVING_ZONE', 'WOOF_HEX_MOVING_SHOCK_ZONE')


from .moving_control import (RESIDENT_CLOSED, RESIDENT_ENV, RESIDENT_NEXT, RESIDENT_READY,
                             RESIDENT_SEGMENT_VARIABLES, RESIDENT_SPEC_SCHEMA)


def carrier_sealed(boundary) -> bool:
    """A segment's end carrier exists: its file, or (a resident move) its receipt for the in-memory hand-off."""
    boundary = Path(boundary)
    if boundary.is_file():
        return True
    receipt = boundary.with_suffix(boundary.suffix + '.json')
    if not receipt.is_file():
        return False
    try:
        return bool(json.loads(receipt.read_text()).get('resident'))
    except ValueError:
        return False


def visible_devices_for(device, environ=None) -> str:
    """The CUDA_VISIBLE_DEVICES a segment gets for ``--device N``: the Nth card the caller can see.

    THE BREAKAGE THIS PREVENTS (V8X6000, 2026-10-29, one-card shapes lane): under a box card queue the lease sets
    CUDA_VISIBLE_DEVICES to the leased card (a UUID).  ``moving run --device 0``, the docs' own example, replaced it
    with the literal ``0``, so the segment ran on PHYSICAL card 0, which another lane's exclusive lease held (44.7 GB
    there for 2 min while the leased card sat idle).  ``--device`` now indexes the cards already visible, as CUDA
    ordinals do inside the process; with nothing inherited it is the physical index, as before.
    """
    environ = os.environ if environ is None else environ
    inherited = []
    for c in str(environ.get('CUDA_VISIBLE_DEVICES', '')).split(','):
        # CUDA reads the list up to its first invalid entry: '-1' (the CPU battery's and many hosts' "no card")
        # leaves nothing visible, so it is nothing inherited, not one card to index.
        if c.strip().startswith('-'):
            break
        if c.strip():
            inherited.append(c.strip())
    wanted = [c.strip() for c in str(device).split(',') if c.strip()]
    if not inherited:
        return ','.join(wanted)
    chosen = []
    for item in wanted:
        if not item.isdigit() or int(item) >= len(inherited):
            raise SystemExit(f'--device {device}: this process sees {len(inherited)} card(s) '
                             f'(CUDA_VISIBLE_DEVICES={environ.get("CUDA_VISIBLE_DEVICES")}); --device indexes those '
                             'cards, 0 to {}'.format(len(inherited) - 1))
        chosen.append(inherited[int(item)])
    return ','.join(chosen)


def restart_dead_products(segments, log, popen=None, alive=None) -> list[str]:
    """Redraw the products of closed segments whose detached product stage died with the controller.

    THE BREAKAGE THIS PREVENTS: each segment's forecast starts its product door detached
    (forecast_door.start_detached_products), so a box or container restart kills it with
    the controller.  ``--resume`` then carried on from the saved state and never redrew it:
    on a rented box (2026-10-21) member-004 segment-000 of a moving ensemble was left with no global
    maps and 60 GB of render-global scratch frames that nothing would ever clear.  A closed
    segment (rc 0) whose products.log has no ``PRODUCTS done`` line and no live product
    process gets its stale scratch cleared and the same product command started again,
    detached, with no CUDA device (as the forecast door starts it).
    """
    import shlex
    popen = popen or subprocess.Popen
    alive = alive or product_stage_alive
    restarted = []
    for record in segments:
        fc = Path(record.get('path', '')) / 'fc'
        products_log = fc / 'products.log'
        if record.get('rc') != 0 or not products_log.is_file():
            continue
        if any(line.startswith('PRODUCTS done ')
               for line in products_log.read_text(errors='replace').splitlines()):
            continue
        if alive(fc):
            continue
        logged = Path(record['path']) / 'forecast.log'
        lines = logged.read_text(errors='replace').splitlines() if logged.is_file() else []
        command = next((shlex.split(line[len('PRODUCTS $ '):]) for line in reversed(lines)
                        if line.startswith('PRODUCTS $ ')), None)
        if not command:
            log(f'products of {fc} died unfinished and their command is not in {logged}; not redrawn')
            continue
        if command[:2] == ['woof', 'hex']:
            # the door logs its program as "woof hex" (one argv entry, two words), and command[1:] below
            # drops exactly one entry; split, it relaunched `woof.hex.cli hex products`, which refuses
            command = ['woof hex', *command[2:]]
        if '--scratch' not in command and '--keep-scratch' not in command:
            shutil.rmtree(fc / 'products.products-scratch', ignore_errors=True)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', GPUWM_NO_LOCAL_GPU='1')
        with products_log.open('ab') as stream:
            child = popen([sys.executable, '-m', 'woof.hex.cli', *command[1:]], stdin=subprocess.DEVNULL,
                          stdout=stream, stderr=subprocess.STDOUT, env=environment,
                          start_new_session=True)
        log(f'products of {fc} died unfinished with the controller; redrawing detached pid={child.pid}')
        restarted.append(str(fc))
    return restarted


def product_stage_alive(fc: Path) -> bool:
    """True when a live process is a product door for this forecast folder."""
    target = str(fc)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if 'products' in argv and any(a == target and i and argv[i - 1] == '--forecast'
                                      for i, a in enumerate(argv)):
            return True
    return False

def card_arguments(cards: int, case, extra=()) -> list[str]:
    """The forecast-door flags that split every segment across ``cards`` visible cards of this host.

    With more than one card each segment runs the multi-card class blocks of active LTS
    (``--samehost-devices 0,...,K-1``; the door builds each member's partition from its own grid).
    """
    cards = int(cards or 1)
    if cards < 1:
        raise SystemExit('--cards must be at least 1')
    if cards == 1:
        return []
    from .moving_case import resolve_physics
    given = resolve_physics(case, extra)
    if any(str(arg).startswith('--samehost-') for arg in given):
        # Breakage prevented: two --samehost-devices on one door command, the later silently winning,
        # or explicit layouts built for one member's grid refused on every other member.
        raise SystemExit('--cards and --samehost-* in the case physics_args or --forecast-arg both pick the '
                         'cards; use --cards alone (each member is partitioned from its own grid)')
    return ['--samehost-devices', ','.join(str(card) for card in range(cards))]


def label_argv(arguments) -> list[str]:
    """The run's ``--model-label``/``--brand``, for every segment's product stage."""

    from .products_door import label_argv as product_labels

    return product_labels(arguments)


def segment_command(case, member, seg: Path, hours: float, history_minutes: float, engine=None, *,
                    init_source: str, extra=(), cards: int = 1, source_id: str | None = None) -> list[str]:
    """One forecast segment on one member: the forecast door with the moving run's fixed flags.

    The fixed-region control (``woof hex moving fixed``) uses the same command, so the
    two differ only in the moves.  ``cards`` above one splits the region across that many
    visible cards of this host (:func:`card_arguments`).
    """
    from .moving_case import resolve_physics
    extra = [*extra, *card_arguments(cards, case, extra)]
    return hexcore_command(
        'forecast', '--mesh', member['mesh'], '--grid', member['grid'], '--static', member['static'],
        '--init', member['init'], '--start-time', case['start_time'], '--init-source', init_source,
        '--hours', repr(float(hours)), '--history-every-minutes', repr(float(history_minutes)),
        '--dt', member['dt_seconds'], '--out', seg / 'fc', '--scratch', seg / 'scratch',
        *(['--gpuwm-checkout', engine] if engine else []),
        '--fullstep-local-timestep', '--fullstep-lts-mode', 'active', '--fullstep-lts-max-rate', '8',
        *resolve_physics(case, extra, source=source_id))


def route_refusal(seg) -> str | None:
    """The door's moving-route refusal in a segment's log, or None (woof.hex.fullstep_lts_admission)."""
    from .fullstep_lts_admission import MOVING_ROUTE_REFUSAL

    log = Path(seg) / 'forecast.log'
    if not log.is_file():
        return None
    for line in reversed(log.read_text(errors='replace').splitlines()[-50:]):
        if MOVING_ROUTE_REFUSAL in line:
            return line.strip().removeprefix('woof hex: ')
    return None


def door_refusal(seg, rc) -> str | None:
    """The forecast door's refusal of a segment before its first step, or None.

    The door exits 2 with one ``woof hex: ...`` line when it refuses a command
    (woof.hex.cli.main).  Refused before the first step (the timeline has no
    ``forecast.first_step`` and no recovery seal exists), the same command is
    refused the same way every time.  THE BREAKAGE THIS PREVENTS (HEX-PHYSICS-PARITY
    run group 2 #6): every member retried a namelist refusal three times and the
    ensemble then reported a track-plume failure instead of the refusal.
    """
    if rc != 2:
        return None
    seg = Path(seg)
    if any((seg / 'control').glob('seal-*.pkl')):
        return None
    timeline = seg / 'fc' / 'timeline.jsonl'
    if timeline.is_file() and '"forecast.first_step"' in timeline.read_text(errors='replace'):
        return None
    log = seg / 'forecast.log'
    if not log.is_file():
        return None
    for line in reversed(log.read_text(errors='replace').splitlines()[-50:]):
        if line.startswith('woof hex: '):
            return line.strip().removeprefix('woof hex: ')
    return None


def now_utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def hexcore_command(*args) -> list[str]:
    """A child of this controller: the same interpreter running this package's console door."""
    return [sys.executable, '-m', 'woof.hex.cli', *[str(a) for a in args]]


def start_child(argv, *, stdout, env=None, cwd=None):
    """Start a child in its own process group (POSIX), so its whole tree stops with it."""
    return subprocess.Popen(argv, stdout=stdout, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                            start_new_session=(os.name == 'posix'))


def stop_process_tree(proc, grace_seconds: float = 60.0):
    """Terminate a child started by start_child and everything it started, by its own PID."""
    if proc.poll() is not None:
        return
    def signal_group(sig):
        try:
            if os.name == 'posix':
                os.killpg(proc.pid, sig)
            else:  # pragma: no cover - not a POSIX host
                proc.send_signal(sig)
        except ProcessLookupError:
            pass
    # Give the coordinator a chance to close its pool and retire scratch.
    # Killing workers while they write a multiprocessing result pipe can
    # strand its reader thread and prevent the coordinator's finally blocks.
    try:
        proc.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        signal_group(signal.SIGKILL if os.name == 'posix' else signal.SIGTERM)
        proc.wait()
    else:
        # A coordinator without a handler exits immediately. Its detached
        # descendants still belong to this owned group and must not survive.
        signal_group(signal.SIGTERM)


def load_tracker_module(path: str | None):
    """P2's ``woof.hex.storm_tracker``; ``path`` loads a staged copy of that one file."""
    if path:
        spec = importlib.util.spec_from_file_location('woof.hex.storm_tracker', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['woof.hex.storm_tracker'] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module('woof.hex.storm_tracker')


# --------------------------------------------------------------------------- model state

def history_fields(st, path: Path, guess, radius_km: float, object_field: str | None = None, *,
                   columns: bool = False, surface: bool = False, storm_field: str | None = None):
    """Tracker fields from one closed history frame (MPAS cell arrays).

    ``columns`` (setup target cape_shear) also reads the model columns (pressure, theta, qv, u, v) and ``surface``
    (setup target boundary) the 2 m fields (t2, q2) with the surface pressure, for the cells near the guess.

    ``object_field`` (object mode: the tracker's object.field_name) is read too, the column maximum of a 3-D
    field. Breakage it prevents: the object field was never read, so an object-mode tracker answered 'no object
    field in the state' at every frame and the moving region never moved (derecho 1 km run, box 2026-10-17
    01:41Z, refl10cm >= 40 dBZ on 6,256 cells inside the search radius)."""
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        lat = np.asarray(d['latCell'][:], np.float64)
        lon = np.asarray(d['lonCell'][:], np.float64)
        if np.nanmax(np.abs(lat)) <= math.pi / 2 + 1e-6:
            lat, lon = np.degrees(lat), np.degrees(lon)
        g = st.xyz(guess[0], guess[1])
        p = np.stack([np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                      np.cos(np.radians(lat)) * np.sin(np.radians(lon)), np.sin(np.radians(lat))], 1)
        reach = (radius_km + 700.0) / st.EARTH_RADIUS_KM
        subset = np.flatnonzero(p @ np.asarray(g, float).reshape(3) >= math.cos(min(reach, math.pi)))
        psfc = np.asarray(d['surface_pressure'][0], np.float64)
        terrain = np.asarray(d['ter'][:], np.float64)
        pressure = np.asarray(d['pressure'][0], np.float64)
        theta = np.asarray(d['theta'][0, :, 0], np.float64)
        t_low = theta * (pressure[:, 0] / 1.0e5) ** (287.0 / 1004.0)
        u = np.asarray(d['u_zonal'][0], np.float64)
        v = np.asarray(d['v_meridional'][0], np.float64)
        u10 = np.asarray(d['u10'][0], np.float64)
        v10 = np.asarray(d['v10'][0], np.float64)
        objective = None
        if object_field:
            if object_field not in d.variables:
                raise KeyError(f'object tracker field {object_field!r} is not in history {path}')
            objective = np.asarray(d[object_field][0], np.float64)
            if objective.ndim > 1:
                objective = objective.max(axis=tuple(range(1, objective.ndim)))
        elif storm_field and storm_field in d.variables:
            # cape_shear's storms (EnvConfig.storm_field): optional, a frame without the field fixes on the index
            objective = np.asarray(d[storm_field][0], np.float64)
            if objective.ndim > 1:
                objective = objective.max(axis=tuple(range(1, objective.ndim)))
        column_block = surface_block = None
        if columns:
            column_block = dict(pressure=pressure[subset], theta=np.asarray(d['theta'][0], np.float64)[subset],
                                qv=np.asarray(d['qv'][0], np.float64)[subset], u=u[subset], v=v[subset],
                                terrain=terrain[subset], psfc=psfc[subset])
        if surface:
            # 2 m fields under either route's history name (hex history t2; the MPAS stream names it t2m).
            t2_name = 't2' if 't2' in d.variables else 't2m'
            surface_block = dict(t2=np.asarray(d[t2_name][0], np.float64)[subset],
                                 q2=np.asarray(d['q2'][0], np.float64)[subset], psfc=psfc[subset])
    keep = subset
    fields = st.fields_from_columns(lat[keep], lon[keep], psfc=psfc[keep], terrain=terrain[keep],
                                    t_low=t_low[keep], pressure=pressure[keep], u=u[keep], v=v[keep],
                                    u10=u10[keep], v10=v10[keep],
                                    object_field=None if objective is None else objective[keep],
                                    **({} if column_block is None else dict(columns=column_block)),
                                    **({} if surface_block is None else dict(surface=surface_block)))
    return fields


def scripted_position(track, seconds):
    import numpy as np
    t = np.array([row[0] for row in track], float)
    lat = np.array([row[1] for row in track], float)
    lon = np.array([row[2] for row in track], float)
    return float(np.interp(seconds, t, lat)), float(np.interp(seconds, t, lon))


def scripted_fix(st, track, seconds):
    """The fix a scripted track gives at ``seconds``; its own motion stands in for the steering flow a model
    fix diagnoses, so the policy's short-history blend sees the same motion."""
    lat, lon = scripted_position(track, seconds)
    ahead = scripted_position(track, seconds + 1800.0)
    north = (ahead[0] - lat) * math.pi * 6371229.0 / 180.0 / 1800.0
    east = (ahead[1] - lon) * math.pi * 6371229.0 / 180.0 * math.cos(math.radians(lat)) / 1800.0
    return st.Fix(seconds=float(seconds), ok=True, lat=lat, lon=lon, reason='scripted track',
                  steering_ms=[east, north])


# --------------------------------------------------------------------------- controller

class Controller:
    def __init__(self, a):
        self.a = a
        a.case = str(Path(a.case).expanduser().absolute())
        a.casedir = str(Path(a.casedir).expanduser().absolute())
        a.out = str(Path(a.out).expanduser().absolute())
        a.members = str(Path(a.members or Path(a.out) / 'members').expanduser().absolute())
        self.case = load_case(a.case)
        # M2 appended max-rate=1 after the controller's max-rate=8. That executor
        # has no progress or carrier hooks, so refuse before any member preparation.
        from .moving_case import resolve_physics
        flags = resolve_physics(self.case, getattr(a, 'forecast_arg', []))
        rate = '8'
        for index, flag in enumerate(flags):
            if flag == '--fullstep-lts-max-rate' and index + 1 < len(flags):
                rate = flags[index + 1]
            elif str(flag).startswith('--fullstep-lts-max-rate='):
                rate = str(flag).split('=', 1)[1]
        if str(rate) == '1':
            raise SystemExit('moving run requires the active multi-rate executor: max-rate=1 selects '
                             'an executor without moving progress or carrier hooks; omit the override')
        if getattr(a, 'rotate_template', False) and (a.zone_carry or self.case['mesh'].get('corridor')):
            raise SystemExit('--rotate-template requires a concentric global family without corridor or --zone-carry')
        if a.check_minutes is None:
            # The case's own tracker cadence. Breakage it prevents: the door default (60 min) set the history and
            # tracker interval although the case said placement.check_minutes 5, so a 1 h moving run read one
            # frame and never moved (derecho 1 km case, box 2026-10-17 01:39Z).
            placement = ((self.case.get('tracker') or {}).get('placement') or {})
            a.check_minutes = float(placement.get('check_minutes', 60.0))
        self.st = load_tracker_module(a.tracker_module)
        self.out = Path(a.out)
        self.members_dir = Path(a.members)
        self.duration = float(a.hours) * 3600.0
        self.check = float(a.check_minutes) * 60.0
        self.dt = float(self.case['dt_seconds'])
        if self.check % (8 * self.dt) or self.duration % self.check:
            raise SystemExit('check interval must be whole macro steps and divide the run length')
        self.state_path = self.out / 'controller.json'
        self.session_path = self.out / 'track-session.json'
        self.procs = {}          # name -> Popen (member prep or operator build)
        self.forecast_children = []
        self.prelaunched = None  # the next member's forecast, started before the move
        # The running forecast process and its segment (run): a resident process continues there after a move.
        self.current_proc = self.current_seg = None
        mesh = self.case['mesh']
        self.interior_km = interior_km(self.case)
        # A corridor case stitches every member from the same frame and template
        # (woof.hex.moving_zone.build_insert_member): a member depends on its centre alone,
        # any two members share the far field, and moves use the zone carry.
        self.stitch = bool(mesh.get('corridor'))
        if self.stitch:
            a.zone_carry = True
        self.zone_transfer = a.zone_carry and not getattr(a, 'whole_domain_remap', False)
        transfer = 'zone' if self.zone_transfer else 'whole-domain'
        if a.resume:
            self.state = json.loads(self.state_path.read_text())
            previous_transfer = self.state.get('transfer', 'zone' if a.zone_carry else 'whole-domain')
            if previous_transfer != transfer:
                raise SystemExit('resume must retain the original transfer route: ' + previous_transfer)
            self.session = self.st.TrackSession.restore(json.loads(self.session_path.read_text()))
            from .moving_retention import retire_finished_preparations
            retire_finished_preparations(self, getattr(a, 'member_retirement_evidence', None))
            self.state['hours'] = float(a.hours)
            self.state.pop('finished_utc', None)
            # Preparations of the dead controller have no process any more: a finished one is
            # prepared, an unfinished one is cancelled (its closed stages are kept and skipped).
            for name, member in self.state['members'].items():
                if member['status'] == 'preparing':
                    done = (self.members_dir / f'{name}-case.json').is_file()
                    self.state['members'][name] = (dict(self.read_member(name, 'prepared'), why=member.get('why'))
                                                   if done else dict(member, status='cancelled'))
            self.log(f'resume at model {self.state["t"]:.0f} s, {len(self.state["segments"])} segments')
            restart_dead_products(self.state['segments'], self.log)
        else:
            self.out.mkdir(parents=True, exist_ok=False)
            # The case's own tracker block is the base; --tracker-config overrides it
            # (its placement block merges key by key).
            config = json.loads(json.dumps(self.case.get('tracker') or {}))
            if a.tracker_config:
                override = json.loads(Path(a.tracker_config).read_text())
                placement = dict(config.get('placement') or {}, **(override.pop('placement', None) or {}))
                config.update(override)
                config['placement'] = placement
            config.setdefault('mode', 'tc')
            config.setdefault('initial_center_deg', self.case['storm']['initial_center_deg'])
            placement = dict(config.get('placement') or {})
            placement.setdefault('interior_radius_km', self.interior_km)
            placement.setdefault('check_minutes', float(a.check_minutes))
            config['placement'] = placement
            self.session = self.st.TrackSession(self.st.TrackerConfig.from_dict(config))
            start = self.adopt_start_member()
            self.state = dict(schema=SCHEMA, started_utc=now_utc(),
                              wall_started=time.time(), case=str(Path(a.case).resolve()),
                              hours=float(a.hours), check_minutes=float(a.check_minutes), transfer=transfer,
                              fixes_source=('scripted track (machinery test; not a tracker)'
                                            if a.scripted_track else
                                            "the user's path (not a tracker)" if config.get('mode') == 'path'
                                            else f"run-time tracker on model state ({config.get('mode')} target)"),
                              tracker_module=self.st.__file__, t=0.0, current=start,
                              boundary=None, operator=None, source_grid=None,
                              members={start: self.read_member(start, 'ready')},
                              segments=[], moves=[], frames=[], decisions=[], events=[],
                              deleted=[], pending_move=None)
            self.save()
        self.track = json.loads(Path(a.scripted_track).read_text()) if a.scripted_track else None
        from .moving_members import read_stage
        self.init_source = read_stage(Path(a.casedir)).get('source', 'staged')
        self.rotation_source = self.member_rotation_source()
        self.plan = self.plan_known_path()
        if self.plan:
            self.event('path-plan', moves=[dict(at_seconds=p['at_seconds'], centre=p['centre']) for p in self.plan],
                       why='every frame of this target is known before the forecast runs; its members are '
                           'prepared from launch, while segment 0 runs')

    # ---------------------------------------------------------------- bookkeeping
    def log(self, text):
        line = f'{now_utc()} {text}'
        print(line, flush=True)
        with (self.out / 'controller.log').open('a') as stream:
            stream.write(line + '\n')

    def save(self):
        tmp = self.state_path.with_suffix('.json.partial')
        tmp.write_text(json.dumps(self.state, indent=1, default=float) + '\n')
        tmp.replace(self.state_path)
        self.session.save(self.session_path)

    def event(self, kind, **data):
        row = dict(kind=kind, wall_utc=now_utc(), wall=time.time(), **data)
        self.state['events'].append(row)
        self.log(f'{kind} {json.dumps(data, default=float)}')

    # ---------------------------------------------------------------- members
    def adopt_start_member(self):
        family = json.loads((Path(self.a.casedir) / 'family.json').read_text())
        row = next(m for m in family if m['name'] == self.a.start_member)
        self.members_dir.mkdir(parents=True, exist_ok=True)
        (self.members_dir / f"{row['name']}-case.json").write_text(json.dumps(row, indent=2) + '\n')
        names = [row['name'] + suffix
                 for suffix in ('.grid.nc', '.static.nc', '.init.nc', '.spec.json', '-mesh.json', '-rows.json',
                                '-static.json', '-init.json')]
        if self.stitch:
            names += ['frame.grid.nc', 'frame.spec.json', 'template.grid.nc', 'template.spec.json']
        for name in names:
            source = Path(self.a.casedir) / name
            link = self.members_dir / name
            if source.exists() and not link.exists():
                link.symlink_to(source)
        return row['name']

    def read_member(self, name, status):
        row = json.loads((self.members_dir / f'{name}-case.json').read_text())
        # Admission is the same for the start and runtime members. The old
        # absolute 2.2 km cutoff described 48/3 km zone experiments, not
        # validity of a case-requested 1 km (or finer) mesh.
        return dict(row, status=status, min_dc_edge_m=shortest_edge_m(row['static']))

    def member_centres(self, statuses):
        return {n: m['center_deg'] for n, m in self.state['members'].items() if m['status'] in statuses}

    def member_key(self, lat, lon):
        base = member_name(lat, lon)
        # A frozen-far-field member depends on the member it is built from (P1, woof.hex.moving_zone).
        return f'{base}x{self.state["current"]}' if self.a.zone_carry and not self.stitch else base

    def member_rotation_source(self):
        """The mesh every member prepared during the run is rotated from, or None when they are generated.

        THE BREAKAGE THIS PREVENTS (HEX-SHAPES-1CARD 2026-10-29, the 48 -> 3 km storm-following region on a path,
        306,507 cells, one RTX PRO 6000): ``moving prepare`` rotates the start member from the premade family
        template (4.6 s), but members prepared during the run were rotated only under the opt-in
        --rotate-template, so the first move's member regenerated the whole ring mesh (116 s of a 168 s
        preparation) and the forecast reached its end first.  A concentric global family now rotates the
        premade template (one rotation, exactly as the start member was made), else the start member's own mesh;
        --no-rotate-template generates every member."""
        choice = getattr(self.a, 'rotate_template', None)
        if choice is False or self.stitch or self.a.zone_carry or self.case['mesh'].get('corridor'):
            return None
        from .moving_case import spec as member_spec
        from .moving_members import TEMPLATE_CENTER, TEMPLATE_NAME

        def family(path, name, centre=None):
            try:
                body = json.loads(Path(path).read_text())
                centre = centre or body['regions'][-1]['shape']['center_deg']
                return body == member_spec(self.case, name, *centre)
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                return False

        casedir = Path(self.a.casedir)
        suffixes = ('.spec.json', '.grid.nc', '-mesh.json')
        if all((casedir / f'{TEMPLATE_NAME}{s}').is_file() for s in suffixes) \
                and family(casedir / f'{TEMPLATE_NAME}.spec.json', TEMPLATE_NAME, TEMPLATE_CENTER):
            self.members_dir.mkdir(parents=True, exist_ok=True)
            for s in suffixes:
                link = self.members_dir / f'{TEMPLATE_NAME}{s}'
                if not link.exists() and casedir.resolve() != self.members_dir.resolve():
                    try:
                        link.symlink_to(casedir / f'{TEMPLATE_NAME}{s}')
                    except OSError:     # a host without symlink rights (Windows): a hard link, same bytes
                        os.link(casedir / f'{TEMPLATE_NAME}{s}', link)
            return TEMPLATE_NAME
        start = self.a.start_member
        if all((self.members_dir / f'{start}{s}').is_file() for s in suffixes) \
                and family(self.members_dir / f'{start}.spec.json', start):
            return start
        return None

    def rotation_source_for(self, lat, lon):
        """``--rotate-from`` for a member at (lat, lon), or None to generate it."""
        if not hasattr(self, 'rotation_source'):
            # a controller built without __init__ (tests): the old explicit flag only
            return self.a.start_member if getattr(self.a, 'rotate_template', False) else None
        source = self.rotation_source
        if source is None:
            return None
        from .moving_members import TEMPLATE_CENTER, TEMPLATE_NAME, _near_antipode
        if source == TEMPLATE_NAME and _near_antipode((lat, lon), TEMPLATE_CENTER):
            # the minimal rotation from 0N 0E is ill posed there; the start member is a few hundred km away
            start = self.a.start_member
            return start if (self.members_dir / f'{start}.spec.json').is_file() else None
        return source

    # ---------------------------------------------------------------- a path known in advance
    def known_future(self):
        """True when every future fix is known before the forecast runs: the user's path or a scripted track.
        (A cyclone, object or setup target is fixed on model state, so its future is predicted, not known.)"""
        return getattr(self, 'track', None) is not None or getattr(self.session.config, 'mode', None) == 'path'

    def plan_known_path(self):
        """The moves the placement policy will decide along a known path, from the current state.

        THE BREAKAGE THIS PREVENTS (HEX-SHAPES-1CARD 2026-10-29, storm-following on a user's path, 29x real
        time): the move decided at the 1800 s frame found no member; its preparation started only then and the
        forecast reached its 1 h end first, with the storm 12 km past the safe edge and the controller held
        170 s.  A path fixes every frame, so the policy's own decisions are replayed here (the same session code,
        the same frame times, each planned member counted as available once planned) and their members are
        prepared from launch on host cores while segment 0 runs.  Re-planned at every frame, so a deferred or
        different move re-plans from what actually happened."""
        if (not self.known_future() or self.session.policy is None
                or getattr(self.a, 'anticipate_moves', 1) <= 0):
            return None     # no known future: the anticipation chain predicts from the fixes instead
        sim = self.st.TrackSession.restore(self.session.to_json())
        decided = {float(f['model_seconds']) for f in self.state['frames']}
        pending = self.state.get('pending_move')
        region = list(self.state['members'][pending['target'] if pending else self.state['current']]['center_deg'])
        step = sim.config.placement.snap_deg
        horizon = min(self.duration, float(self.state['t']) + float(getattr(self.a, 'anticipate_hours', 36.0)) * 3600.0)
        seconds = math.ceil(float(self.state['t']) / self.check - 1e-9) * self.check
        planned, available = [], []
        while seconds < horizon - 1e-6:
            if seconds not in decided:
                if self.track is not None:
                    sim.fixes.append(scripted_fix(self.st, self.track, seconds))
                    sim.misses = 0
                else:
                    sim.observe(seconds, None)
                decision = sim.decide(seconds, region,
                                      available=[c for c in available if great_circle_km(c, region) > 1e-6])
                if decision.move:
                    centre = [float(v) for v in self.st.snap(decision.centre[0], decision.centre[1], step)]
                    planned.append(dict(centre=centre, at_seconds=float(seconds), reason=decision.reason))
                    available.append(centre)
                    region = centre
            seconds += self.check
        return planned

    def planned_chain(self, after_seconds):
        """The next planned moves after ``after_seconds`` in the anticipation-chain shape (order 1 = next)."""
        upcoming = [p for p in getattr(self, 'plan', None) or [] if p['at_seconds'] > after_seconds]
        return [dict(p, order=i + 1) for i, p in enumerate(upcoming[:max(1, int(self.a.anticipate_moves))])]

    def prepare_chain(self, chain, current, seconds):
        """Request the members of an anticipation (or planned) chain that no member serves yet.

        A planned member (a chain row with a ``reason``) remembers the member its move starts from
        (``planned_source``), so its overlap operator is built as soon as both meshes exist, beside its static
        and init (:meth:`poll_background`)."""
        source = (self.state.get('pending_move') or {}).get('target') or self.state['current']
        limit = min(self.a.reuse_km, self.session.policy.c.min_move_km)
        for anticipated in chain:
            planned = 'reason' in anticipated
            near = [n for n, c in self.member_centres(('preparing', 'prepared', 'ready')).items()
                    if n != self.state['current'] and great_circle_km(c, anticipated['centre']) <= limit]
            name = None
            if near:
                name = min(near, key=lambda n: great_circle_km(self.state['members'][n]['center_deg'],
                                                                anticipated['centre']))
            elif great_circle_km(anticipated['centre'], current['center_deg']) >= self.session.policy.c.min_move_km:
                why = (f'planned along the known path at {seconds:.0f} s for {anticipated["at_seconds"]:.0f} s '
                       f'(move {anticipated["order"]})' if planned else
                       f'anticipated at {seconds:.0f} s for {anticipated["at_seconds"]:.0f} s '
                       f'(move {anticipated["order"]})')
                name = self.request_member(anticipated['centre'], why, order=anticipated['order'])
            if planned and name in self.state['members'] and source is not None and name != source:
                self.state['members'][name]['planned_source'] = source
            source = name if name in self.state['members'] else None

    def request_planned(self):
        """Keep the members of the next planned moves preparing (at launch and at every background poll)."""
        if not getattr(self, 'plan', None):
            return
        frames = self.state['frames']
        after = max((float(f['model_seconds']) for f in frames), default=float(self.state['t']) - 1.0)
        chain = self.planned_chain(after)
        if chain:
            self.prepare_chain(chain, self.state['members'][self.state['current']], after)

    def request_member(self, centre, why, order=1, admit=None):
        """Name of a member at (or serving) ``centre``, starting its preparation when none exists.

        ``admit`` (a centre -> bool rule, the same one :meth:`pick_target` applies) filters the two
        shortcuts that serve the request with a member at ANOTHER centre: an on-disk member within
        reuse distance and a member already preparing within min_move_km.  THE BREAKAGE THIS
        PREVENTS (hex-day2 storm-following runs, 2026-10-27, twice): a due move reused an on-disk
        member up to min(--reuse-km, min_move_km) off the policy's centre whose predicted core was
        not safe there for one check, so pick_target refused it once it was READY and the door
        stopped the run ('moving member ... is ready after holding the forecast')."""
        step = self.session.config.placement.snap_deg
        lat, lon = self.st.snap(centre[0], centre[1], step)
        name = self.member_key(lat, lon)
        members = self.state['members']
        from .moving_retention import rebuild_name
        name = rebuild_name(name, members)
        if name in members and members[name]['status'] in ('failed', 'unstable') and step > 0:
            # The generator cannot settle every centre (measured 2026-10-12: "relaxation is in
            # a limit cycle" for one 0.25 deg lattice point); the nearest lattice neighbour
            # that has not failed serves the same move.
            around = sorted(((great_circle_km((lat + i * step, lon + j * step), centre), lat + i * step, lon + j * step)
                             for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j))
            for _, la, lo in around:
                alternative = rebuild_name(self.member_key(la, lo), members)
                if alternative not in members or members[alternative]['status'] not in ('failed', 'unstable'):
                    self.event('member-neighbour', failed=name, instead=alternative, why=why)
                    lat, lon, name = la, lo, alternative
                    break
        if name in members and members[name]['status'] == 'cancelled':
            del members[name]
        if name in members:
            # The order is also what poll_background reads to build a planned member's operator from the
            # current region once its move is the next one (not only while it is preparing).
            members[name]['order'] = min(members[name].get('order', order), order)
            if members[name]['status'] == 'stale':
                from .moving_retention import artifacts_present
                if not artifacts_present(members[name]):
                    raise RuntimeError(f'stale member {name} lost an artifact outside managed retirement')
                members[name]['status'] = 'prepared'
                self.event('member-revived', name=name, why=why)
            # a failed preparation is not retried: it would fail the same way every check
            return None if members[name]['status'] in ('failed', 'unstable') else name
        # A member already on disk (an earlier run of this case) within reuse distance
        # is used as is: its geometry is fixed by its centre, never by the run.
        on_disk = []
        for path in self.members_dir.glob('*-case.json'):
            key = path.name[:-len('-case.json')]
            if key == self.state['current'] or (key in members and members[key]['status'] in ('failed', 'unstable', 'retired')):
                continue
            if key in ('frame', 'template'):
                continue
            if self.a.zone_carry and not self.stitch and not key.endswith(f'x{self.state["current"]}'):
                continue    # a frozen-far-field member is built from the region it moves FROM
            try:
                row = json.loads(path.read_text())
            except ValueError:
                continue
            distance = great_circle_km(row['center_deg'], (lat, lon))
            from .moving_retention import artifacts_present
            if (distance <= min(self.a.reuse_km, self.session.policy.c.min_move_km) and artifacts_present(row)
                    and (admit is None or distance <= 1e-6 or admit(row['center_deg']))):
                on_disk.append((distance, key))
        if on_disk:
            key = min(on_disk)[1]
            if key not in members or members[key]['status'] == 'stale':
                members[key] = self.read_member(key, 'prepared')
                self.event('member-reused', name=key, center=members[key]['center_deg'], wanted=[lat, lon], why=why)
            return key
        busy = [n for n, m in members.items() if m['status'] == 'preparing']
        near = [(great_circle_km(members[n]['center_deg'], (lat, lon)), n) for n in busy]
        near = [(d, n) for d, n in near if d <= self.session.policy.c.min_move_km
                and (admit is None or d <= 1e-6 or admit(members[n]['center_deg']))]
        if near:
            name = min(near)[1]
            members[name]['order'] = min(members[name].get('order', order), order)
            return name
        if len(busy) >= min(2, max(1, self.a.max_preparing)):
            # Refresh the next target without restarting every unfinished member.
            # A speculative later move must not evict work for the next one.
            if order > 1:
                return None
            victim = max(busy, key=lambda n: (
                great_circle_km(members[n]['center_deg'], (lat, lon)),
                members[n].get('requested_wall', 0.0)))
            proc = self.procs.pop(f'prep:{victim}', None)
            if proc is not None and proc.poll() is None:
                self.stop_child(proc)
            members[victim]['status'] = 'cancelled'
            self.event('member-prep-cancelled', name=victim, for_order=order,
                       distance_km=great_circle_km(members[victim]['center_deg'], (lat, lon)),
                       wanted=[lat, lon], why=why)
        log = self.members_dir / 'logs' / f'{name}-prep.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        # The static's tile decode is the long pole of a member (590 s at 2 tile workers,
        # 123 s at 12, every field byte-identical: rw_mpas_static --compare).
        argv = ['moving', 'prepare-member', '--case', self.a.case, '--casedir', self.a.casedir,
                '--folder', self.members_dir, '--name', name, '--lat', f'{lat}', '--lon', f'{lon}',
                '--threads', self.a.prep_threads, '--tile-workers', self.a.prep_tile_workers]
        if self.a.geog:
            argv += ['--geog', self.a.geog]
        if self.stitch:
            argv += ['--stitch']                         # frame + rigidly moved template (no chain)
        elif self.a.zone_carry:
            argv += ['--prev', self.state['current']]   # frozen-far-field member builder
        elif self.rotation_source_for(lat, lon) is not None:
            argv += ['--rotate-from', self.rotation_source_for(lat, lon)]
        proc = start_child(hexcore_command(*argv), stdout=log.open('w'))
        self.procs[f'prep:{name}'] = proc
        members[name] = dict(name=name, center_deg=[lat, lon], status='preparing', order=order,
                             requested_wall=time.time(), requested_model_seconds=self.state.get('t_seen'), why=why)
        self.event('member-prep-start', name=name, center=[lat, lon], why=why)
        self.save()
        return name

    def operator_path(self, source, target):
        return self.members_dir / 'operators' / f'{source}--{target}.zone.npz' if self.zone_transfer \
            else self.members_dir / 'operators' / f'{source}--{target}.npz'

    def ensure_operator(self, target, source=None):
        source = source or self.state['current']
        path = self.operator_path(source, target)
        key = f'op:{source}--{target}'
        if path.is_file():
            return True
        if key in self.procs or key in self.state.setdefault('operator_failures', []):
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        src = self.state['members'][source]
        tgt = self.state['members'][target]
        log = self.members_dir / 'logs' / f'{source}--{target}-operator.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        if self.zone_transfer:
            command = hexcore_command('moving', 'zone', src['grid'], tgt['grid'], src['static'], tgt['static'],
                                      src['init'], tgt['init'], path)
        else:
            command = hexcore_command('moving', 'operator', self.member_grid(source), self.member_grid(target), path)
        self.procs[key] = start_child(command, stdout=log.open('a'))
        self.event('operator-start', source=source, target=target,
                   **({} if tgt.get('status') != 'preparing' else {'while': 'target static and init still preparing'}))
        return False

    def member_grid(self, name):
        """A member's grid path, also while it is still preparing (its mesh stage closes first)."""
        member = self.state['members'].get(name) or {}
        return member.get('grid') or str(self.members_dir / f'{name}.grid.nc')

    def mesh_closed(self, name):
        """Whether a member's mesh stage is closed: the grid and its receipt (written after the grid) exist."""
        member = self.state['members'].get(name) or {}
        if member.get('grid') and member.get('status') != 'preparing':
            return Path(member['grid']).is_file()
        return (self.members_dir / f'{name}-mesh.json').is_file() and Path(self.member_grid(name)).is_file()

    def start_operators_early(self):
        """Overlap operators whose two meshes exist, started before the target's static and init close.

        A whole-domain operator reads only the two grids (``moving operator``), so the 44 s build of the 306,507
        cell region (HEX-SHAPES-1CARD 2026-10-29) runs beside the member's static and init instead of after
        them.  The source is the member the planned move starts from, else the current one.  At most two
        operator builds run at once, next move first."""
        if self.zone_transfer:
            return      # the zone carry reads both statics and inits too
        members = self.state['members']
        current = self.state['current']
        building = sum(1 for key in self.procs if key.startswith('op:'))
        # A stale member (prepared for an earlier region) is built for only when its planned move starts here.
        waiting = sorted(((m.get('order', 1), n) for n, m in members.items()
                          if n != current and (m['status'] in ('preparing', 'prepared')
                                               or (m['status'] == 'stale' and m.get('planned_source') == current))),
                         key=lambda row: row[0])
        for _, name in waiting:
            if building >= 2:
                return
            source = members[name].get('planned_source') or current
            if source not in members or members[source].get('status') in ('failed', 'unstable', 'cancelled'):
                continue
            key = f'op:{source}--{name}'
            if key in self.procs or self.operator_path(source, name).is_file():
                continue
            if self.mesh_closed(source) and self.mesh_closed(name):
                self.ensure_operator(name, source=source)
                building += 1

    def poll_background(self):
        for key, proc in list(self.procs.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del self.procs[key]
            kind, name = key.split(':', 1)
            if kind == 'prep':
                member = self.state['members'][name]
                ok = rc == 0 and (self.members_dir / f'{name}-case.json').is_file()
                if ok:
                    self.state['members'][name] = dict(self.read_member(name, 'prepared'),
                                                       prep_wall_seconds=time.time() - member['requested_wall'],
                                                       why=member.get('why'),
                                                       **{k: member[k] for k in ('order', 'planned_source')
                                                          if k in member})
                else:
                    member['status'] = 'failed'
                self.event('member-prep-done', name=name, rc=rc,
                           wall_seconds=time.time() - member['requested_wall'])
            else:
                self.event('operator-done', pair=name, rc=rc)
                if rc:
                    # never relaunched: a failing pair fails the same way every poll
                    self.state.setdefault('operator_failures', []).append(f'op:{name}')
        current = self.state['current']
        # Prepared members need an operator from the CURRENT member before they are ready.  A member planned
        # for a LATER move along a known path (it starts from another planned member) waits for that move,
        # unless a move to it is due now (request_member lowers its order to 1).
        for name, member in self.state['members'].items():
            if name == current or member['status'] not in ('prepared', 'ready'):
                continue
            if member.get('planned_source') not in (None, current) and member.get('order', 1) > 1:
                continue
            if self.ensure_operator(name):
                member['status'] = 'ready'
            else:
                member['status'] = 'prepared'
        if hasattr(self, 'plan'):
            self.request_planned()
            self.start_operators_early()
        self.save()

    def ready_members(self):
        return {n: m['center_deg'] for n, m in self.state['members'].items()
                if m['status'] == 'ready' and n != self.state['current']}

    # ---------------------------------------------------------------- tracking
    def observe(self, seconds, history):
        began = time.monotonic()
        if self.track is not None:
            fix = scripted_fix(self.st, self.track, seconds)
            self.session.fixes.append(fix)
            self.session.misses = 0
        else:
            guess, since = self.session.predicted(seconds)
            config = self.session.config
            needs = (self.st.history_needs(config) if hasattr(self.st, 'history_needs')
                     else dict(object_field=config.object.field_name if config.mode == 'object' else None,
                               columns=False, surface=False, state=True))
            # A user's path reads nothing from the state; every other target reads the frame.
            fields = (None if not needs['state'] else
                      history_fields(self.st, Path(history), guess, self.session.search_radius(since),
                                     object_field=needs['object_field'],
                                     columns=needs['columns'], surface=needs['surface'],
                                     storm_field=needs.get('storm_field')))
            fix = self.session.observe(seconds, fields)
        return fix, time.monotonic() - began

    def fix_velocity(self):
        """Motion of the last two distinct valid model fixes, with no steering blend."""
        good = [f for f in self.session.fixes if f.ok]
        if len(good) < 2:
            return None
        last = good[-1]
        previous = next((f for f in reversed(good[:-1]) if f.seconds < last.seconds), None)
        if previous is None:
            return None
        x, y = self.st.project(last.lat, last.lon, previous.lat, previous.lon)
        hours = (last.seconds - previous.seconds) / 3600.0
        velocity = self.st.np.array([-float(x) / hours, -float(y) / hours])
        speed = float(self.st.np.hypot(*velocity))
        limit = self.session.motion.c.max_speed_kmh
        if speed > limit:
            velocity *= limit / speed
        return velocity

    def anticipate_chain(self, decision, current):
        """Prepare along the model's latest displacement, from the actual member.

        An unexecuted decision is not a new mesh centre. A single initial fix
        cannot measure displacement, so it must not launch a case-motion guess.
        """
        if self.a.anticipate_moves <= 0:
            return []
        velocity = self.fix_velocity()
        if velocity is None:
            return []
        policy = self.session.policy
        first = policy.anticipate(replace(decision, centre=current['center_deg']),
                                  self.session.fixes, current['center_deg'], velocity=velocity)
        if first is None or first['at_seconds'] >= self.duration:
            return []
        chain = [dict(first, order=1)]
        if self.a.anticipate_moves <= 1 or (self.a.zone_carry and not self.stitch):
            return chain
        last = next(f for f in reversed(self.session.fixes) if f.ok)
        origin = (last.lat, last.lon)
        while len(chain) < self.a.anticipate_moves:
            previous = chain[-1]
            start = (previous['at_seconds'] - last.seconds) / 60.0
            stay = policy.dwell(previous['centre'], origin, velocity, start, policy.c.max_lead_minutes)
            if stay <= 0:
                break
            begin = start + stay
            if (begin * 60.0 > self.a.anticipate_hours * 3600.0
                    or last.seconds + begin * 60.0 >= self.duration):
                break
            centre, dwell, _, _ = policy.choose(origin, velocity, previous['centre'], (), begin)
            chain.append(dict(centre=list(centre), at_seconds=last.seconds + begin * 60.0, dwell_minutes=dwell,
                              order=len(chain) + 1))
        return chain

    def storm_room(self, current, fix):
        """(minutes, km): how long the predicted storm core stays safe in the current region, and
        how far the fix is from that region's safe edge (interior radius minus core margin).

        The minutes are the placement policy's own dwell (fix and motion uncertainty included),
        -1 when the core is already unsafe."""
        policy = self.session.policy
        if not fix.ok:
            return -1.0, 0.0
        velocity, _ = policy.motion.velocity(self.session.fixes)
        stay = policy.dwell(current['center_deg'], (fix.lat, fix.lon), velocity, 0.0)
        edge = policy.c.safe_km() - great_circle_km((fix.lat, fix.lon), current['center_deg'])
        return float(stay), float(edge)

    def pick_target(self, decision, current, seconds, ready, fix):
        """Nearest ready member whose predicted core remains safe for a check.

        Admission uses the placement policy's fix and motion uncertainty too;
        mere current-position containment can immediately force a return move.
        """
        if not fix.ok:
            return None
        candidates = [(great_circle_km(centre, (fix.lat, fix.lon)), name)
                      for name, centre in ready.items() if name != self.state['current']
                      and self.admits(centre, seconds, fix)]
        return min(candidates)[1] if candidates else None

    def admits(self, centre, seconds, fix):
        """True when a member at ``centre`` keeps the predicted core (fix and motion uncertainty
        included) safe for at least one check: the admission rule of :meth:`pick_target`."""
        if not fix.ok:
            return False
        policy = self.session.policy
        fixes = [f for f in self.session.fixes if f.seconds < fix.seconds] + [fix]
        velocity, _ = self.session.motion.velocity(fixes)
        return policy.dwell(centre, (fix.lat, fix.lon), velocity,
                            max(0.0, (seconds - fix.seconds) / 60.0),
                            max(policy.c.check_minutes, policy.SUBSTEP_MIN)) >= policy.c.check_minutes

    def on_frame(self, seconds, history, control: Path):
        if seconds in {f['model_seconds'] for f in self.state['frames']}:
            # A frame this controller already decided.  A segment that is holding at it still
            # gets an answer: THE BREAKAGE THIS PREVENTS (HEX-PHYSICS-PARITY run group 2 #6) is a
            # recover segment waiting forever for verdict-00005400.json, a frame answered for the
            # failed segment only, holding its card at 0 % until killed.
            answer = control / f'verdict-{int(seconds):08d}.json'
            if (control / 'await-verdict.json').is_file() and not answer.exists():
                tmp = answer.with_suffix('.partial')
                tmp.write_text(json.dumps({'stop': False}) + '\n')
                tmp.replace(answer)
            return None
        # At most once per tracking frame, never on every step/background poll.
        from .moving_retention import retire_members
        retire_members(self, getattr(self.a, 'member_retirement_evidence', None))
        current = self.state['members'][self.state['current']]
        from .stage_timing import stage
        timings = []
        with stage(timings, 'model-state-tracker', parallelism={'device': 'CPU', 'threads': 1}):
            fix, fix_seconds = self.observe(seconds, history)
        ready = self.ready_members()
        decision = self.session.decide(seconds, current['center_deg'], available=list(ready.values()))
        offset = None if not fix.ok else great_circle_km((fix.lat, fix.lon), current['center_deg'])
        # The storm's room in the region it is in at this frame, on every frame (the receipt's safe-edge record).
        room_minutes, edge_km_now = self.storm_room(current, fix)
        frame = dict(model_seconds=seconds, history=str(history), member=self.state['current'],
                     member_center=current['center_deg'], segment=len(self.state['segments']) - 1,
                     fix=fix.as_dict(), offset_km=offset,
                     inside_interior=(None if offset is None else offset <= self.interior_km),
                     storm_room_minutes=room_minutes, storm_to_edge_km=edge_km_now if fix.ok else None,
                     decision=decision.as_dict(), tracker_seconds=fix_seconds, stage_timings=timings,
                     wall=time.time())
        self.state['frames'].append(frame)
        if decision.move:
            frame['move_wanted'] = dict(centre=decision.centre, reason=decision.reason)
        if decision.move and seconds >= getattr(self, 'duration', math.inf) - 1e-6:
            # THE BREAKAGE THIS PREVENTS (HEX-SHAPES-1CARD 2026-10-29, the 1 h path run): the forecast never
            # waits for a verdict at its final frame (MovingControl.at_frame returns at once there) and no
            # forecast is left to run on a new member, yet the controller held itself 170 s at that frame for
            # one, and the run ended 170 s late with no move.
            decision = replace(decision, move=False, reason=decision.reason + '; the run ends at this frame: no move')
            self.session.decisions[-1] = decision
            frame['decision'] = decision.as_dict()
            frame['move_wanted']['skipped'] = 'the run ends at this frame'
        chain = self.anticipate_chain(decision, current) if getattr(self, 'plan', None) is None else []
        frame['anticipated'] = chain
        verdict = {'stop': False}
        if decision.move:
            target = self.pick_target(decision, current, seconds, ready, fix)
            if target is None:
                # The decision is for this frame. The anticipation chain starts at the
                # NEXT move and can already lie outside this fix's safe radius.
                centre = decision.centre
                wanted = (self.request_member(centre, f'move decided at {seconds:.0f} s, no ready member',
                                              admit=lambda c: self.admits(c, seconds, fix))
                          if fix.ok and great_circle_km(centre, current['center_deg']) > 1e-6
                          else None)
                # a member reused from disk with a cached operator is ready at once
                self.poll_background()
                ready = self.ready_members()
                target = self.pick_target(decision, current, seconds, ready, fix)
                # The honest bound is the storm's own room: how long the predicted core (with its
                # fix and motion uncertainty) stays inside the current region's safe radius.  A
                # continuous-follow re-centre while the storm still has a check of room is not
                # due yet, so the forecast keeps integrating on the current member and the move
                # lands at a later check once the member is ready (a Champi storm-following run:
                # the first re-centre came 19 s after its member started preparing, with the
                # storm 13 km off centre and ~140 km of room; hex 0.4.4 refused it after 0 s).
                stay, edge_km = self.storm_room(current, fix) if target is None else (0.0, 0.0)
                check = self.session.policy.c.check_minutes if target is None else 0.0
                frame['move_wanted']['name'] = wanted
                # A deferral lands at the next check.  When that check is the run's last frame no move can land
                # there, so deferring loses the move: the decided member is waited for here instead.  THE
                # BREAKAGE THIS PREVENTS (HEX-SHAPES-1CARD 2026-10-29, the 1 h path run at 29x real time): the
                # 1800 s move was deferred to the 3600 s end frame, never made, and the storm ended 12 km past the
                # safe edge; its member, prepared from launch, was ready 12 s after the 1800 s frame.
                last_chance = seconds + check * 60.0 >= getattr(self, 'duration', math.inf) - 1e-6
                hold_limit = self.a.hold_for_member_seconds
                if last_chance:
                    # waiting longer than the forecast takes for the one check it serves buys nothing
                    previous = [f for f in self.state['frames'][:-1] if f.get('segment') == frame['segment']
                                and f.get('wall') is not None and f['model_seconds'] < seconds]
                    if previous:
                        per_check = ((frame['wall'] - previous[-1]['wall'])
                                     / max(1.0, (seconds - previous[-1]['model_seconds']) / (check * 60.0)))
                        hold_limit = min(hold_limit, max(30.0, per_check))
                if wanted and target is None and stay >= check and not last_chance:
                    frame['move_wanted']['deferred'] = True
                    self.event('move-deferred', name=wanted, at_seconds=seconds,
                               status=self.state['members'].get(wanted, {}).get('status'),
                               storm_room_minutes=stay, storm_to_edge_km=edge_km)
                    # retried at the next check, where the policy sees the member once it is ready
                    decision = replace(decision, move=False, dwell_minutes=check,
                                       next_check_seconds=seconds + check * 60.0,
                                       reason=decision.reason + f'; deferred: member {wanted} not ready, '
                                       f'storm safe for {stay:.0f} min ({edge_km:.0f} km to the safe edge)')
                    self.session.decisions[-1] = decision
                    frame['decision'] = decision.as_dict()
                    wanted = None
                held = time.monotonic()
                # The move is due now: the storm would leave the safe radius before the next check.
                # The forecast is held at this frame (it rendezvous at every check, see monitor), so
                # the storm does not move while we wait for the member to finish preparing.
                while (target is None and wanted and self.a.hold_for_member_seconds > 0
                       and (control / 'await-verdict.json').is_file()
                       and self.state['members'].get(wanted, {}).get('status') in ('preparing', 'prepared')
                       and f'op:{self.state["current"]}--{wanted}' not in self.state.get('operator_failures', [])
                       and time.monotonic() - held < hold_limit):
                    time.sleep(2.0)
                    self.poll_background()
                    target = self.pick_target(decision, current, seconds, self.ready_members(), fix)
                if time.monotonic() - held > 5.0:
                    frame['move_wanted']['held_seconds'] = time.monotonic() - held
                    self.event('held-for-member', name=wanted, seconds=time.monotonic() - held,
                               status=self.state['members'].get(wanted, {}).get('status'),
                               storm_room_minutes=stay, storm_to_edge_km=edge_km)
                if wanted and self.a.hold_for_member_seconds > 0 and target is None:
                    status = self.state['members'].get(wanted, {}).get('status', 'missing')
                    if f'op:{self.state["current"]}--{wanted}' in self.state.get('operator_failures', []):
                        status += ' (its overlap operator failed)'
                    waited = time.monotonic() - held
                    if last_chance and stay >= check:
                        # The storm stays safe until the run's end without the move: finish there, do not fail.
                        self.event('move-member-unavailable', name=wanted, status=status, at_seconds=seconds,
                                   waited_seconds=waited, storm_room_minutes=stay, storm_to_edge_km=edge_km,
                                   continued='the predicted core stays safe in this region until the run ends')
                        frame['move_wanted']['deferred'] = True
                        decision = replace(decision, move=False, reason=decision.reason + f'; member {wanted} '
                                           f'{status} after {waited:.0f} s, the storm safe to the run end')
                        self.session.decisions[-1] = decision
                        frame['decision'] = decision.as_dict()
                        wanted = None
                if wanted and self.a.hold_for_member_seconds > 0 and target is None:
                    self.event('move-member-unavailable', name=wanted, status=status, at_seconds=seconds,
                               waited_seconds=waited, storm_room_minutes=stay, storm_to_edge_km=edge_km)
                    self.save()
                    raise RuntimeError(f'moving member {wanted} is {status} after holding the forecast '
                                       f'{waited:.0f} s (--hold-for-member-seconds '
                                       f'{self.a.hold_for_member_seconds:.0f}); the storm is {edge_km:.0f} km from '
                                       f'the safe edge of the current region and its predicted core is safe there '
                                       f'for {max(stay, 0.0):.0f} min, less than one {check:.0f}-min check')
            if target is not None:
                frame['move_wanted']['name'] = target
                verdict = {'stop': True, 'target': target, 'reason': decision.reason}
                self.name_resident_move(target, seconds, control)
                self.state['pending_move'] = dict(target=target, decided_at_seconds=seconds,
                                                  reason=decision.reason)
        if getattr(self, 'plan', None) is not None:
            # A known path re-plans from what actually happened at this frame (a deferral, a different
            # member), then keeps the next planned members preparing, the pending move's included.
            self.plan = self.plan_known_path()
            chain = self.planned_chain(seconds)
            frame['anticipated'] = chain
            self.prepare_chain(chain, current, seconds)
        elif not verdict['stop']:
            self.prepare_chain(chain, current, seconds)
        if (control / 'await-verdict.json').is_file() or verdict['stop']:
            path = control / f'verdict-{int(seconds):08d}.json'
            tmp = path.with_suffix('.partial')
            tmp.write_text(json.dumps(verdict) + '\n')
            tmp.replace(path)
        if verdict['stop']:
            # If the forecast passed this frame before the await request existed, it
            # stops at the next frame instead (the move is then one check late).
            path = control / 'stop-request.json'
            tmp = path.with_suffix('.partial')
            tmp.write_text(json.dumps({'stop_at_seconds': seconds, 'target': verdict['target']}) + '\n')
            tmp.replace(path)
        self.log(f'frame t={seconds/3600:.2f} h fix={None if not fix.ok else (round(fix.lat, 2), round(fix.lon, 2))} '
                 f'offset={offset if offset is None else round(offset)} km '
                 f'storm_to_edge_km={None if not fix.ok else round(edge_km_now, 1)} move={decision.move} verdict={verdict}')
        self.save()
        return verdict

    # ---------------------------------------------------------------- resident continuation
    def resident_ready(self) -> bool:
        """Whether the running forecast process can continue the next segment itself (moving_resident_v842).

        It says so in its control folder once its route has built the moving hooks a continuation needs (the MPAS
        suite route on one card); any other route, or a process that has not got that far, takes a pre-launched
        process as before.
        """
        if (not getattr(self.a, 'resident', False) or getattr(self, 'current_seg', None) is None
                or getattr(self, 'current_proc', None) is None):
            return False
        if self.current_proc.poll() is not None:
            return False
        return (Path(self.current_seg) / 'control' / RESIDENT_READY).is_file()

    def name_resident_move(self, target, seconds, control):
        """Before a stop verdict: the running process's launch record names the move's target member.

        THE BREAKAGE THIS PREVENTS: a resident process continues only the member its record names; a move to a
        member the record did not name (the policy chose another ready member) would seal its carrier to the
        file and exit, and the next segment would start from a cold process after all.
        """
        if not getattr(self.a, 'resident', False):
            return   # one process per segment (--no-resident): nothing to name
        pre = getattr(self, 'prelaunched', None)
        if pre is not None and pre['target'] == target:
            return
        if not self.resident_ready():
            return   # the run adopts or drops a pre-launched process at the move, as before
        if pre is not None:
            self.drop_prelaunch(f'the move goes to {target}')
            return
        source = self.state['current']
        proc, pseg, pcontrol, record = self.launch(
            'move', member_name=target, begin=seconds, checkpoint_in=str(Path(control).parent / 'boundary.pkl'),
            operator=str(self.operator_path(source, target)), source_grid=self.state['members'][source]['grid'],
            prelaunch=True)
        self.prelaunched = dict(proc=proc, seg=pseg, control=pcontrol, record=record, target=target,
                                resident=bool(record.get('resident')))
        self.event('prelaunch', target=target, segment=record['index'], resident=bool(record.get('resident')),
                   at_move=True)

    # ---------------------------------------------------------------- segments
    def segment_env(self, member, control, out_boundary, checkpoint_in, operator, source_grid):
        # The run-state switches (carrier, operator, zone) are set here and only here: a
        # segment never inherits them from the caller's environment.  Knobs such as
        # WOOF_HEX_MOVING_BALANCE (the balanced remap initialization, default on) pass through.
        env = {k: v for k, v in os.environ.items() if k not in RUN_STATE_VARIABLES}
        env.update(WOOF_HEX_MOVING_SHOCK='1', GPUWM_WRF_EXACT='0', OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='1', WOOF_HEX_MESH_ROWS=member['mesh_rows'],
                   WOOF_HEX_MOVING_CHECKPOINT_OUT=str(out_boundary),
                   WOOF_HEX_MOVING_CONTROL=str(control),
                   WOOF_HEX_MOVING_SEAL_EVERY_SECONDS=str(int(self.a.seal_every_hours * 3600)),
                   WOOF_HEX_MOVING_HEIGHT_REMAP='absolute')
        if getattr(self.a, 'resident', False):
            # The forecast process may continue the next segment after a move (moving_resident_v842); it does
            # only when its route builds the moving hooks for it (resident_ready) and the controller names the move.
            env[RESIDENT_ENV] = '1'
        if self.a.device is not None:
            env['CUDA_VISIBLE_DEVICES'] = visible_devices_for(self.a.device)
        if checkpoint_in:
            env['WOOF_HEX_MOVING_CHECKPOINT_IN'] = str(checkpoint_in)
        if operator and str(operator).endswith('.zone.npz'):
            env.update(WOOF_HEX_MOVING_ZONE=str(operator), WOOF_HEX_MOVING_SHOCK_ZONE=str(operator))
        elif operator:
            env.update(WOOF_HEX_MOVING_OPERATOR=str(operator), WOOF_HEX_MOVING_SOURCE_GRID=str(source_grid))
        return env

    def launch(self, kind, *, member_name=None, begin=None, checkpoint_in='state', operator='state',
               source_grid='state', prelaunch=False):
        index = len(self.state['segments'])
        seg = self.out / f'segment-{index:03d}'
        if prelaunch and seg.exists():
            shutil.rmtree(seg)
        seg.mkdir()
        control = seg / 'control'
        control.mkdir()
        if self.a.hold_for_member_seconds > 0:
            (control / 'await-verdict.json').write_text(json.dumps({
                'from_seconds': 0,
                'timeout_seconds': self.a.verdict_timeout + self.a.hold_for_member_seconds}) + '\n')
        member = self.state['members'][member_name or self.state['current']]
        begin = float(self.state['t'] if begin is None else begin)
        checkpoint_in = self.state['boundary'] if checkpoint_in == 'state' else checkpoint_in
        operator = self.state['operator'] if operator == 'state' else operator
        source_grid = self.state['source_grid'] if source_grid == 'state' else source_grid
        hours = (self.duration - begin) / 3600.0
        from woof.output_formats import argument_values
        cmd = segment_command(self.case, member, seg, hours, self.a.check_minutes, self.a.engine,
                              init_source=f'{self.init_source} analysis; moving run member {member["name"]}',
                              extra=[*self.a.forecast_arg, *label_argv(self.a), *argument_values(self.a, relative=seg.name)],
                              cards=getattr(self.a, 'cards', 1), source_id=self.init_source)
        env = self.segment_env(member, control, seg / 'boundary.pkl', checkpoint_in, operator, source_grid)
        resident = prelaunch and self.resident_ready()
        if prelaunch and not resident:
            # Started before the move: the real begin is the (later) stop frame, so the
            # schedule runs past the end; a standing stop request ends it exactly there.
            env['WOOF_HEX_MOVING_CHECKPOINT_WAIT'] = str(int(self.a.segment_timeout))
        if prelaunch:
            (control / 'stop-request.json').write_text(json.dumps({'stop_at_seconds': self.duration,
                                                                   'reason': 'run end'}) + '\n')
        record = dict(index=index, path=str(seg), kind=kind, member=member['name'], center=member['center_deg'],
                      begin_seconds=begin, requested_hours=hours, checkpoint_in=checkpoint_in,
                      operator=operator, source_grid=source_grid, prelaunched=prelaunch,
                      command=cmd, launched_wall=time.time(), launched_utc=now_utc())
        if resident:
            # The running forecast process continues this segment after the move (moving_resident_v842): its
            # launch record goes into the running segment's control folder, where that process prepares it.
            record['resident'] = True
            (seg / 'command.json').write_text(json.dumps(record, indent=2) + '\n')
            spec = dict(schema=RESIDENT_SPEC_SCHEMA, member=member['name'], index=index, segment=str(seg),
                        argv=[str(a) for a in cmd[3:]], begin_seconds=begin,
                        env={k: env[k] for k in RESIDENT_SEGMENT_VARIABLES if k in env})
            running = Path(self.current_seg) / 'control' / RESIDENT_NEXT
            temporary = running.with_suffix('.partial')
            temporary.write_text(json.dumps(spec, indent=2) + '\n')
            temporary.replace(running)
            self.log(f'resident next segment {index} ({kind}) member={member["name"]} begin={begin/3600:.2f} h '
                     f'(prepared in the running forecast process)')
            return self.current_proc, seg, control, record
        (seg / 'command.json').write_text(json.dumps(record, indent=2) + '\n')
        if not prelaunch:
            self.state['segments'].append(record)
            self.save()
        self.log(f'{"pre-launch" if prelaunch else "launch"} segment {index} ({kind}) member={member["name"]} '
                 f'begin={begin/3600:.2f} h hours={hours:g}')
        stream = (seg / 'forecast.log').open('w')
        proc = start_child(cmd, stdout=stream, env=env, cwd=str(self.out))
        self.forecast_children.append(proc)
        proc.hex_started = time.monotonic()
        return proc, seg, control, record

    def maybe_prelaunch(self, seg):
        """Start the next member's forecast process now, so it is waiting at the carrier."""
        if not self.a.prelaunch or self.prelaunched is not None:
            return
        ready = self.ready_members()
        if not ready:
            return
        frames = [f for f in self.state['frames'] if f['segment'] == len(self.state['segments']) - 1]
        if not frames and self.state['frames'] and self.resident_ready():
            # A move segment writes no frame at its begin (the controller decided that frame on the previous
            # member).  A resident process prepares the next segment beside this one, so its record is named from
            # the move's frame now rather than at the next frame.  THE COST THIS REMOVES (RTX 5090, 2026-10-29): every
            # move after the first named its member only at the move, so the next region's pool decode, inputs and
            # remap map were built after the stop (4-6 s per move) instead of beside the running segment.
            frames = [self.state['frames'][-1]]
        if not frames:
            return
        # the ready member the policy is most likely to choose: longest predicted dwell
        good = [f for f in self.session.fixes if f.ok]
        target = sorted(ready)[0]
        if good:
            # the member the policy itself would pick for a move at the predicted time
            velocity, _ = self.session.motion.velocity(self.session.fixes)
            origin = (good[-1].lat, good[-1].lon)
            policy = self.session.policy
            last = frames[-1]['decision']
            start = 0.0 if last['move'] else float(last['dwell_minutes'])
            current = self.state['members'][self.state['current']]['center_deg']
            centre, _, _, _ = policy.choose(origin, velocity, current, list(ready.values()), start)
            target = min(ready, key=lambda n: great_circle_km(ready[n], centre))
        upcoming = self.planned_chain(max(f['model_seconds'] for f in frames)) if getattr(self, 'plan', None) else []
        if upcoming:
            # a known path: the next planned move's member, when it is the ready one
            planned = [n for n, c in ready.items() if great_circle_km(c, upcoming[0]['centre']) <= 1.0]
            target = planned[0] if planned else target
        source = self.state['current']
        begin = max(f['model_seconds'] for f in frames)
        if begin >= self.duration:
            return  # the run ends at this frame; a move segment of zero hours is never adopted
        proc, pseg, control, record = self.launch(
            'move', member_name=target, begin=begin, checkpoint_in=str(seg / 'boundary.pkl'),
            operator=str(self.operator_path(source, target)), source_grid=self.state['members'][source]['grid'],
            prelaunch=True)
        self.prelaunched = dict(proc=proc, seg=pseg, control=control, record=record, target=target,
                                resident=bool(record.get('resident')))
        self.event('prelaunch', target=target, segment=record['index'], resident=bool(record.get('resident')))

    def stop_child(self, proc):
        """End one of this controller's own children with its whole process tree.

        Each child leads its own process group (start_child), so the group is
        addressed by the child's own PID and nothing outside it is touched; a
        plain terminate would orphan the stage a preparation runs.
        """
        stop_process_tree(proc)

    def drop_prelaunch(self, why):
        if self.prelaunched is None:
            return
        proc = self.prelaunched['proc']
        if self.prelaunched.get('resident'):
            # The running forecast process itself: only its launch record goes (it prepares nothing for it then).
            record = Path(self.current_seg) / 'control' / RESIDENT_NEXT if self.current_seg else None
            if record is not None and record.is_file():
                try:
                    named = json.loads(record.read_text()).get('member')
                except ValueError:
                    named = None
                if named == self.prelaunched['target']:
                    record.unlink()
        elif proc.poll() is None:
            self.stop_child(proc)
        seg = self.prelaunched['seg']
        self.event('prelaunch-dropped', target=self.prelaunched['target'], why=why)
        shutil.rmtree(seg, ignore_errors=True)
        self.prelaunched = None

    def monitor(self, proc, seg, control):
        seen = 0
        progress = control / 'progress.jsonl'
        while True:
            self.poll_background()
            ready = self.ready_members()
            awaiting = control / 'await-verdict.json'
            # With member waiting enabled, every check must rendezvous with the
            # controller. A new fix can demand a move even when the PREVIOUS fix
            # predicted a long dwell (observed in the full global run at +6 h).
            # Removing this request between checks races the new decision: the
            # forecast advances and on_frame cannot safely wait for preparation.
            # The asynchronous opt-out retains its imminent-ready optimization.
            last = self.state['frames'][-1]['decision'] if self.state['frames'] else None
            imminent = last is None or last['move'] or last['dwell_minutes'] <= self.a.await_within_minutes
            ready = ready if imminent else {}
            # A model-state fix can request the very first new member. Hold
            # before observing that frame, even when no preparation exists yet.
            coming = self.a.hold_for_member_seconds > 0
            timeout = self.a.verdict_timeout + self.a.hold_for_member_seconds
            if (ready or coming) and not awaiting.exists():
                temporary = awaiting.with_suffix('.partial')
                temporary.write_text(json.dumps({'from_seconds': 0, 'timeout_seconds': timeout}) + '\n')
                temporary.replace(awaiting)
            elif not (ready or coming) and awaiting.exists():
                awaiting.unlink()
            if ready and proc.poll() is None:
                self.maybe_prelaunch(seg)
            rows = progress.read_text().splitlines() if progress.is_file() else []
            for line in rows[seen:]:
                row = json.loads(line)
                if 'history' in row and 'model_seconds' in row:
                    self.state['t_seen'] = row['model_seconds']
                    self.on_frame(float(row['model_seconds']), row['history'], control)
            seen = len(rows)
            started = getattr(proc, 'hex_started', None)
            closed = seg / RESIDENT_CLOSED
            if proc.poll() is None and closed.is_file():
                # A resident process closed this segment and continues the next one (moving_resident_v842).
                rows = progress.read_text().splitlines() if progress.is_file() else []
                for line in rows[seen:]:
                    row = json.loads(line)
                    if 'history' in row and 'model_seconds' in row:
                        self.on_frame(float(row['model_seconds']), row['history'], control)
                wall = json.loads(closed.read_text()).get('segment_wall_seconds')
                if wall is not None:
                    (seg / 'forecast-process-wall.txt').write_text(f'{float(wall):.2f}\n')
                return 0
            if proc.poll() is None and started is not None \
                    and time.monotonic() - started > self.a.segment_timeout:
                self.event('segment-timeout', segment=str(seg), seconds=time.monotonic() - started)
                self.stop_child(proc)
            if proc.poll() is not None:
                if started is not None:
                    (seg / 'forecast-process-wall.txt').write_text(f'{time.monotonic() - started:.2f}\n')
                # drain frames written just before exit
                rows = progress.read_text().splitlines() if progress.is_file() else []
                for line in rows[seen:]:
                    row = json.loads(line)
                    if 'history' in row and 'model_seconds' in row:
                        self.on_frame(float(row['model_seconds']), row['history'], control)
                return proc.returncode
            time.sleep(2.0)

    def segment_receipt(self, seg):
        return segment_receipt(seg)

    def delete_owned(self, path, reason):
        path = Path(path)
        if not path.exists() or path.is_symlink() or not path.resolve().is_relative_to(self.out.resolve()):
            return
        for item in (path, path.with_suffix(path.suffix + '.json')):
            if item.exists():
                self.state['deleted'].append(dict(path=str(item), bytes=item.stat().st_size, reason=reason))
                item.unlink()

    def failure_kind(self, seg):
        """'numerical' for a model refusal (non-finite or invalid state), else 'other'."""
        path = Path(seg) / 'fc' / 'fullstep-lts-first-failure.json'
        if path.is_file():
            row = json.loads(path.read_text())
            if row.get('exception') == 'FloatingPointError' or 'validation flag' in str(row.get('message', '')):
                return 'numerical'
        log = Path(seg) / 'forecast.log'
        tail = log.read_text(errors='replace')[-4000:] if log.is_file() else ''
        if 'FloatingPointError' in tail:
            return 'numerical'
        if 'mesh bind refused' in tail or 'unstable mesh/timestep' in tail:
            # the member itself is unusable (r48s: a zone member with a 1.92 km edge); retrying it
            # fails identically, so it takes the same member fallback as a numerical failure
            return 'numerical'
        return 'other'

    def forget_after(self, seconds):
        """Frames, fixes and decisions after a carrier's time belong to a trajectory that was discarded."""
        before = len(self.state['frames'])
        self.state['frames'] = [f for f in self.state['frames'] if f['model_seconds'] <= seconds]
        self.session.fixes = [f for f in self.session.fixes if f.seconds <= seconds]
        self.session.decisions = [d for d in self.session.decisions if d.seconds <= seconds]
        return before - len(self.state['frames'])

    def carrier_member(self):
        """The member whose mesh the current carrier (state['boundary']) was sealed on."""
        boundary = self.state.get('boundary')
        if not boundary:
            return self.state['current']
        path = Path(boundary)
        folder = path.parent if path.name == 'boundary.pkl' else path.parent.parent
        for record in self.state['segments']:
            if Path(record['path']) == folder:
                return record['member']
        return self.state['current']

    def fallback(self, record):
        """Continue from the recovery carrier on another member after a numerical failure.

        The carrier is the newest recovery seal of the failing member, or -- when the
        failure came before any seal of a move segment -- the move's own starting carrier,
        sealed on the member it moved FROM; that member is then also a candidate (stay).
        """
        failing = self.state['current']
        source = self.carrier_member()
        self.state['members'][failing]['status'] = 'unstable'
        self.state['members'][failing]['unstable_at_seconds'] = self.state['t']
        t = float(self.state['t'])
        dropped = self.forget_after(t)
        for move in self.state['moves']:
            if move['target'] == failing and float(move['at_seconds']) >= t and source != failing:
                move['abandoned'] = 'numerical failure before the first recovery seal of the new member'
        candidates = {n: m for n, m in self.state['members'].items()
                      if n != failing and m['status'] in ('prepared', 'ready', 'stale')
                      and Path(m.get('init', '')).is_file()}
        if source != failing:
            candidates[source] = self.state['members'][source]
        if self.a.zone_carry and not self.stitch:
            # a zone member must share the far field: built from the same chain
            candidates = {n: m for n, m in candidates.items() if 'x' in n or n == 'start'}
        good = [f for f in self.session.fixes if f.ok]
        if not candidates or not good:
            self.event('fallback-none', source=source, t=t)
            return None
        velocity, _ = self.session.motion.velocity(self.session.fixes)
        origin = (good[-1].lat, good[-1].lon)
        policy = self.session.policy
        lag = (t - good[-1].seconds) / 60.0
        scored = sorted(((policy.dwell(m['center_deg'], origin, velocity, lag, policy.c.max_lead_minutes),
                          -great_circle_km(m['center_deg'], origin), n) for n, m in candidates.items()), reverse=True)
        target = scored[0][2]
        if target == source:
            self.state.update(current=source, operator=None, source_grid=None)
            self.state['members'][source]['status'] = 'ready'
            self.event('fallback', source=failing, target=source, t=t, dwell_minutes=scored[0][0],
                       frames_discarded=dropped, note='stay on the carrier member')
            return 'continue'
        path = self.operator_path(source, target)
        if not path.is_file():
            # build it now, in the foreground: the card has nothing else to do
            # (ensure_operator builds from the current member: that is the carrier's here)
            self.state['current'] = source
            self.ensure_operator(target)
            proc = self.procs.get(f'op:{source}--{target}')
            if proc is not None:
                proc.wait()
                del self.procs[f'op:{source}--{target}']
        if not path.is_file():
            self.event('fallback-operator-failed', source=source, target=target)
            return None
        self.state['moves'].append(dict(target=target, source=source, at_seconds=t, decided_at_seconds=t,
                                        reason='numerical failure fallback (same carrier, other member)',
                                        source_center=self.state['members'][source]['center_deg'],
                                        target_center=self.state['members'][target]['center_deg'],
                                        distance_km=great_circle_km(self.state['members'][source]['center_deg'],
                                                                    self.state['members'][target]['center_deg']),
                                        fallback=True))
        self.state.update(operator=str(path), source_grid=self.state['members'][source]['grid'], current=target)
        self.state['members'][target]['status'] = 'ready'
        self.event('fallback', source=source, failing=failing, target=target, t=t, dwell_minutes=scored[0][0],
                   frames_discarded=dropped)
        return 'move'

    def newest_seal(self, control):
        seals = sorted(control.glob('seal-*.pkl'))
        seals = [s for s in seals if s.with_suffix('.pkl.json').is_file()]
        return seals[-1] if seals else None

    def run(self):
        retries = 0
        kind = 'resume' if self.a.resume else 'start'
        trailing = []
        for record in reversed(self.state['segments']):
            if record.get('rc') in (None, 0):
                break
            trailing.append(record)
        if self.a.resume and trailing and self.a.member_fallback \
                and any(self.failure_kind(r['path']) == 'numerical' for r in trailing):
            # The run stopped after repeated numerical failures: move off the failing member.
            kind = self.fallback(self.state['segments'][-1]) or 'resume'
        elif self.a.resume and self.state['segments'] and 'rc' not in self.state['segments'][-1]:
            # The controller died mid-segment: recover like a crash.
            last = self.state['segments'][-1]
            last['rc'] = 'controller-lost'
            seal = self.newest_seal(Path(last['path']) / 'control')
            if seal is not None:
                self.state.update(boundary=str(seal), operator=None, source_grid=None,
                                  t=float(json.loads(seal.with_suffix('.pkl.json').read_text())['time_seconds']))
            # The recovered segment re-reaches the frames after its carrier: they are decided again.
            self.forget_after(float(self.state['t']))
            kind = 'recover'
        adopted = None
        if getattr(self, 'plan', None):
            # A known path: its first members start preparing now, on host cores, while segment 0 starts.
            self.poll_background()
        while self.state['t'] < self.duration:
            if adopted is not None:
                proc, seg, control, record = adopted
                record.update(begin_seconds=self.state['t'], adopted_wall=time.time(), adopted_utc=now_utc())
                (seg / 'command.json').write_text(json.dumps(record, indent=2) + '\n')
                self.state['segments'].append(record)
                self.save()
                self.log(f'adopt pre-launched segment {record["index"]} member={record["member"]} '
                         f'begin={self.state["t"]/3600:.2f} h')
                adopted = None
            else:
                proc, seg, control, record = self.launch(kind)
            if record.get('resident'):
                # Timed from the move, not from the first segment's launch (the segment timeout reads this).
                proc.hex_started = time.monotonic()
            self.current_proc, self.current_seg = proc, seg
            rc = self.monitor(proc, seg, control)
            record.update(rc=rc, ended_wall=time.time(), wall_seconds=time.time() - record['launched_wall'],
                          **self.segment_receipt(seg))
            write_result(seg, record)
            stopped = (record.get('control') or {}).get('stopped_on_request')
            boundary = seg / 'boundary.pkl'
            if rc == 0 and carrier_sealed(boundary):
                retries = 0
                previous_in = self.state['boundary']
                end = float(json.loads(boundary.with_suffix('.pkl.json').read_text())['time_seconds'])
                if not math.isfinite(end) or not record['begin_seconds'] < end <= self.duration:
                    self.event('segment-contract-failed', index=record['index'], end_seconds=end,
                               begin_seconds=record['begin_seconds'], duration_seconds=self.duration)
                    self.save()
                    raise SystemExit(f'segment {record["index"]}: carrier time {end} must advance '
                                     f'beyond {record["begin_seconds"]} and not beyond {self.duration}')
                record['end_seconds'] = end
                # recovery seals of a closed segment are superseded by its boundary
                for seal in control.glob('seal-*.pkl'):
                    self.delete_owned(seal, 'segment closed; its end carrier supersedes recovery seals')
                if previous_in:
                    self.delete_owned(previous_in, 'consumed by a closed segment')
                if stopped and self.state.get('pending_move'):
                    move = self.state['pending_move']
                    source = self.state['current']
                    target = move['target']
                    self.state['moves'].append(dict(move, at_seconds=end, source=source, target=target,
                                                    source_center=self.state['members'][source]['center_deg'],
                                                    target_center=self.state['members'][target]['center_deg'],
                                                    distance_km=great_circle_km(
                                                        self.state['members'][source]['center_deg'],
                                                        self.state['members'][target]['center_deg'])))
                    self.state.update(boundary=str(boundary), operator=str(self.operator_path(source, target)),
                                      source_grid=self.state['members'][source]['grid'], current=target, t=end,
                                      pending_move=None)
                    # members prepared for the old region are stale until the tracker asks
                    # for them again; only then is an operator from the new region built
                    for name, member in self.state['members'].items():
                        if name != target and member['status'] in ('ready', 'prepared'):
                            member['status'] = 'stale'
                    kind = 'move'
                    self.event('move', source=source, target=target, at_seconds=end)
                    if self.prelaunched is not None and self.prelaunched['target'] == target \
                            and self.prelaunched['proc'].poll() is None:
                        pre = self.prelaunched
                        adopted = (pre['proc'], pre['seg'], pre['control'], pre['record'])
                        self.prelaunched = None
                    else:
                        self.drop_prelaunch(f'the move went to {target}')
                else:
                    self.drop_prelaunch('the segment ended without a move')
                    self.state.update(t=end if end >= self.duration else end, boundary=str(boundary),
                                      operator=None, source_grid=None)
                    if end < self.duration:
                        kind = 'continue'
                    else:
                        self.state['t'] = self.duration
            else:
                self.drop_prelaunch('the segment failed')
                if rc == 0:
                    # M2 reran a successful fixed-mesh forecast three times without hooks.
                    self.event('segment-contract-failed', index=record['index'],
                               reason='successful forecast did not seal its moving carrier',
                               expected_carrier=str(boundary))
                    self.save()
                    raise SystemExit(f'segment {record["index"]}: forecast returned 0 without '
                                     f'moving carrier {boundary}; inspect {seg / "forecast.log"}. '
                                     'The selected executor must publish moving progress and a carrier; '
                                     'retrying the same interval cannot repair missing hooks.')
                refusal = route_refusal(seg) or door_refusal(seg, rc)
                if refusal is not None:
                    # The route refuses this command the same way every time: retrying spends nothing useful.
                    self.event('segment-refused', index=record['index'], rc=rc)
                    self.save()
                    raise SystemExit(f'segment {record["index"]}: {refusal}')
                retries += 1
                failure = self.failure_kind(seg)
                self.event('segment-failed', index=record['index'], rc=rc, retries=retries, failure=failure)
                if retries > self.a.max_retries:
                    self.save()
                    raise SystemExit(f'segment {record["index"]} failed {retries} times; stopping')
                seal = self.newest_seal(control)
                if seal is not None:
                    t = float(json.loads(seal.with_suffix('.pkl.json').read_text())['time_seconds'])
                    self.state.update(boundary=str(seal), operator=None, source_grid=None, t=t)
                    self.event('recover-from-seal', seal=str(seal), t=t,
                               frames_discarded=self.forget_after(t))
                else:
                    self.event('recover-from-segment-start', t=self.state['t'],
                               frames_discarded=self.forget_after(float(self.state['t'])))
                kind = 'recover'
                self.state['pending_move'] = None
                if failure == 'numerical' and self.a.member_fallback:
                    # Restart is bit-exact, so the same member fails again at the same step
                    # (measured 3x on r48c and r48z).  Continue from the same carrier on another
                    # member instead: the same 22 h carrier moved to the neighbouring member ran
                    # through the failure time (r48z diag A), the failing member did not.
                    kind = self.fallback(record) or kind
            self.save()
        for key, proc in self.procs.items():
            stop_process_tree(proc)
        self.procs.clear()
        self.state['wall_seconds'] = time.time() - self.state['wall_started']
        self.state['finished_utc'] = now_utc()
        from .moving_retention import retire_finished_preparations
        retire_finished_preparations(self, getattr(self.a, 'member_retirement_evidence', None))
        self.state['summary'] = summarize(self.state, self.interior_km,
                                          mode=getattr(self.session.config, 'mode', None) or 'tc')
        self.save()
        print(json.dumps(self.state['summary'], indent=1, default=float))


def frame_stepping_seconds(progress, begin_seconds=None, timeline=None):
    """The wall seconds a segment stepped: from its start (its history frame at ``begin_seconds``, else the end of
    its forecast construction in ``timeline``, the stage move_costs also reads) to its last history frame (the
    forecast's own ``frame_written`` rows), less the time it waited at frames for the controller's verdict.
    A move segment writes no frame at its begin (the controller decided that frame on the previous member).
    Start-up before the first step and the export after the last frame are not stepping.  None when unknown."""
    from datetime import datetime, timezone
    progress = Path(progress)
    if not progress.is_file():
        return None
    rows = []
    for line in progress.read_text(errors='replace').splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    frames = [r for r in rows if 'history' in r and r.get('wall_utc') is not None]
    if not frames:
        return None
    first = frames[0]
    start, basis = None, None
    if begin_seconds is None or abs(float(first.get('model_seconds', -1.0)) - float(begin_seconds)) < 1e-6:
        start, basis = float(first['wall_utc']), 'its first history frame'
    elif timeline is not None and Path(timeline).is_file():
        for line in Path(timeline).read_text(errors='replace').splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('stage') == 'forecast.construct' and row.get('phase') == 'end':
                start = datetime.strptime(row['t_utc'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(
                    tzinfo=timezone.utc).timestamp()
                basis = 'the end of its forecast construction'
    last = float(frames[-1]['wall_utc'])
    if start is None or last <= start:
        return None
    waits, waits_at_last, open_ = 0.0, 0.0, False
    for row in rows:
        if 'history' in row and row.get('wall_utc') is not None:
            open_ = open_ or float(row['wall_utc']) >= start
            waits_at_last = waits
        elif 'verdict' in row and open_:
            waits += float(row.get('waited_seconds') or 0.0)
    return dict(seconds=last - start - waits_at_last, verdict_wait_seconds=waits_at_last,
                basis=f'{basis} to its last history frame (forecast wall clock), less verdict waits at frames')


def segment_receipt(seg):
    """A closed segment's own figures from its forecast receipts (either physics route) and control files."""
    seg = Path(seg)
    out = {}
    path = seg / 'fc' / 'fullstep-lts-receipt.json'
    if path.is_file():
        r = json.loads(path.read_text())
        macros = r.get('macros') or []
        out.update(construct_seconds=r.get('construct_seconds'),
                   integration_seconds=sum(m['seconds'] for m in macros),
                   first_macro_seconds=(macros[0]['seconds'] if macros else None),
                   warm_seconds_per_forecast_hour=r.get('warm_seconds_per_forecast_hour'),
                   export=r.get('moving_boundary_export'), control=r.get('moving_control'),
                   rate_assignment=(r.get('rate_assignment') or {}).get('geometric_cells'),
                   dry_mass_relative_drift=r.get('dry_mass_relative_drift'))
        remap = r.get('moving_remap')
        if remap:
            out['transfer_seconds'] = remap.get('total_atmospheric_transfer_seconds')
            out['conservation'] = remap.get('conservation') or remap.get('relative_change')
    native = seg / 'fc' / 'native-physics-receipt.json'
    if not path.is_file() and native.is_file():
        # The MPAS suite route (woof.hex.mpasphys.moving_v842) writes its moving fields into its own receipt.
        r = json.loads(native.read_text())
        remap = r.get('moving_remap') or {}
        out.update(route='mpas-suite', export=r.get('moving_boundary_export'), control=r.get('moving_control'),
                   steps_completed=r.get('steps_completed'), transfer_seconds=remap.get('atmosphere_remap_seconds'),
                   conservation=remap.get('conservation'),
                   physics_carry=(remap.get('physics') or {}).get('surface_donors'))
        # THE BREAKAGE THIS PREVENTS (HEX-SHAPES-1CARD 2026-10-29): this route's receipt has no macro table, so
        # every MPAS-suite moving run reported integration_seconds 0 and overhead_fraction 1.0.  The stepping is
        # read from the control's own frame rows instead (no added synchronization).
        begin = None
        if (seg / 'command.json').is_file():
            try:
                begin = json.loads((seg / 'command.json').read_text()).get('begin_seconds')
            except ValueError:
                begin = None
        stepping = frame_stepping_seconds(seg / 'control' / 'progress.jsonl', begin, seg / 'fc' / 'timeline.jsonl')
        if stepping is not None:
            out.update(integration_seconds=stepping['seconds'], integration_basis=stepping['basis'],
                       verdict_wait_seconds=stepping['verdict_wait_seconds'])
    wall = seg / 'forecast-process-wall.txt'
    if wall.is_file() and wall.read_text().strip():
        try:
            out['process_wall_seconds'] = float(wall.read_text().strip().splitlines()[-1])
        except ValueError:
            pass
    return out


def shortest_edge_m(static_path):
    """Minimum dcEdge of a member's static (earth-radius metres), or None when unreadable."""
    try:
        import netCDF4
        import numpy as np
        with netCDF4.Dataset(static_path) as d:
            return float(np.min(np.asarray(d['dcEdge'][:])))
    except Exception:
        return None


def write_result(seg, record):
    """Per-segment result.json in the layout summarize_run.py (the shared A3 metric) reads."""
    end = None
    boundary = Path(seg) / 'boundary.pkl.json'
    if boundary.is_file():
        end = json.loads(boundary.read_text())['time_seconds']
    result = dict(rc=record.get('rc'), begin_seconds=record['begin_seconds'], end_seconds=end,
                  remesh=record['kind'] == 'move', selected_center_deg=record['center'],
                  forecast_process_wall_seconds=record.get('process_wall_seconds'),
                  wall_including_lease_wait_seconds=record.get('wall_seconds'), kind=record['kind'],
                  member=record['member'])
    (Path(seg) / 'result.json').write_text(json.dumps(result, indent=2, default=float) + '\n')


def move_costs(state):
    """Critical path of each move: the old forecast's stop at a frame -> the new forecast's first macro.

    Stop time is the old control's 'stopped' row; the first macro starts when the new
    process closes its construction ledger stage (timeline.jsonl).  Includes the carrier
    export, process start and host preparation (unless pre-launched), carrier load,
    atmospheric transfer and construction.
    """
    from datetime import datetime, timezone
    out = []
    segments = state['segments']
    for previous, record in zip(segments, segments[1:]):
        if record.get('kind') != 'move':
            continue
        stopped = None
        progress = Path(previous['path']) / 'control' / 'progress.jsonl'
        if progress.is_file():
            for line in progress.read_text().splitlines():
                row = json.loads(line)
                if 'stopped' in row:
                    stopped = float(row['stopped']['wall_utc'])
        started = None
        timeline = Path(record['path']) / 'fc' / 'timeline.jsonl'
        if timeline.is_file():
            for line in timeline.read_text().splitlines():
                row = json.loads(line)
                if row.get('stage') == 'forecast.construct' and row.get('phase') == 'end':
                    started = datetime.strptime(row['t_utc'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(
                        tzinfo=timezone.utc).timestamp()
        out.append(dict(segment=record['index'], at_seconds=record['begin_seconds'], prelaunched=record.get('prelaunched'),
                        stop_to_first_macro_seconds=(None if stopped is None or started is None else started - stopped),
                        export_seconds=(previous.get('export') or {}).get('wall_seconds'),
                        transfer_seconds=record.get('transfer_seconds'), construct_seconds=record.get('construct_seconds')))
    return out


def moves_decided(state):
    """Every move the policy decided, and what became of it: executed (the segment stopped there and the next
    one started on the member), deferred to a later check, skipped at the run's last frame, or never made.

    THE BREAKAGE THIS PREVENTS (HEX-SHAPES-1CARD 2026-10-29, the 1 h path run): the receipt said moves 0 and
    passed although a move was decided at 1800 s, deferred for a member that was not ready and never made,
    with the storm 12 km past the safe edge at the end; nothing in the summary named that move."""
    executed = {float(m['decided_at_seconds']) for m in state.get('moves') or []
                if m.get('decided_at_seconds') is not None and not m.get('abandoned') and not m.get('fallback')}
    rows = []
    for frame in state.get('frames') or []:
        wanted = frame.get('move_wanted')
        if wanted:
            rows.append(dict(at_seconds=float(frame['model_seconds']), member=wanted.get('name'),
                             centre=wanted.get('centre'), executed=float(frame['model_seconds']) in executed,
                             deferred=bool(wanted.get('deferred')), skipped=wanted.get('skipped'),
                             held_seconds=float(wanted.get('held_seconds') or 0.0), reason=wanted.get('reason')))
    if not rows:
        # a run written before frames recorded their wanted move: its deferral events name them
        for event in state.get('events') or []:
            if event.get('kind') == 'move-deferred':
                rows.append(dict(at_seconds=float(event['at_seconds']), member=event.get('name'), centre=None,
                                 executed=False, deferred=True, skipped=None, held_seconds=0.0, reason=None))
        rows += [dict(at_seconds=float(m['decided_at_seconds']), member=m.get('target'), centre=m.get('target_center'),
                      executed=True, deferred=False, skipped=None, held_seconds=0.0, reason=m.get('reason'))
                 for m in state.get('moves') or [] if m.get('decided_at_seconds') is not None]
        pending = state.get('pending_move')
        if pending:
            rows.append(dict(at_seconds=float(pending['decided_at_seconds']), member=pending.get('target'),
                             centre=None, executed=False, deferred=False, skipped=None, held_seconds=0.0,
                             reason=pending.get('reason')))
    rows = sorted(rows, key=lambda row: row['at_seconds'])
    for row in rows:
        # a deferred move that a later check made with the same member landed late, it was not lost
        later = [r['at_seconds'] for r in rows if r['executed'] and r['at_seconds'] > row['at_seconds']
                 and r['member'] is not None and r['member'] == row['member']]
        row['landed_at_seconds'] = row['at_seconds'] if row['executed'] else (min(later) if later else None)
    return rows


def summarize(state, interior_km, mode=None):
    """A cyclone run passes only with the storm fixed inside the fine interior on every frame.

    A setup target (cape_shear, low, boundary, object, path) holds its region on a frame with
    no fix: the weather there may simply have no such area (owner canary run_6fe920b4414c6b7c,
    GFS 2026-10-23 18Z f354, no CAPE x shear area all 24 h). It passes when the forecast ran
    to its end and every frame that DID fix stayed inside the interior; the held frames are
    counted and named in the note."""
    segments = state['segments']
    closed = [s for s in segments if 'rc' in s]
    # Every closed segment must carry its stepping figure; a missing one makes the overhead unknown, never 100 %.
    measured = bool(closed) and all(s.get('integration_seconds') is not None for s in closed)
    integration = sum(s.get('integration_seconds') or 0 for s in segments)
    walls = sum(s.get('wall_seconds') or 0 for s in segments)
    frames = state['frames']
    offsets = [f['offset_km'] for f in frames if f['offset_km'] is not None]
    mode = mode or state.get('tracker_mode') or 'tc'
    held = len(frames) - len(offsets)
    if mode == 'tc':
        inside = bool(offsets) and len(offsets) == len(frames) and max(offsets) <= interior_km
    else:
        inside = bool(frames) and (not offsets or max(offsets) <= interior_km)
    ran = state['t'] >= state['hours'] * 3600 and all(s.get('rc') == 0 for s in segments[-1:])
    note = None
    if mode != 'tc' and held:
        note = (f'no {mode} area found on {held} of {len(frames)} frames; the region held where it was'
                if offsets else f'no {mode} area found on any frame; held at the start point')
    total = state.get('wall_seconds') or walls
    decided = moves_decided(state)
    held_seconds = sum(float(e.get('seconds') or 0.0) for e in state.get('events') or []
                       if e.get('kind') == 'held-for-member')
    edges = [f['storm_to_edge_km'] for f in frames if f.get('storm_to_edge_km') is not None]
    past_edge = [f['model_seconds'] for f in frames
                 if f.get('storm_to_edge_km') is not None and f['storm_to_edge_km'] < 0]
    late = [m for m in decided if m['landed_at_seconds'] is None and not m.get('skipped')]
    warnings = []
    if held_seconds > 0:
        warnings.append(f'the controller held {held_seconds:.0f} s for members that were not ready')
    if late:
        warnings.append('decided move(s) never made: ' + ', '.join(
            f"{m['member'] or m['centre']} at {m['at_seconds']:.0f} s" for m in late))
    if past_edge:
        warnings.append(f'the storm core was past the safe edge (interior minus core margin) at '
                        + ', '.join(f'{t:.0f} s' for t in past_edge))
    return dict(
        status='passed' if inside and ran else 'incomplete',
        tracker_mode=mode, held_frames=held, note=note, warnings=warnings,
        forecast_hours=state['t'] / 3600, wall_seconds=state.get('wall_seconds'),
        segment_wall_seconds=walls, integration_seconds=integration if measured else None,
        integration_basis=sorted({s.get('integration_basis') or 'forecast macro table' for s in closed}) or None,
        overhead_seconds=total - integration if measured else None,
        overhead_fraction=((total - integration) / total if measured and total else None),
        held_for_member_seconds=held_seconds,
        moves=len(state['moves']), moves_decided=decided, pending_move=state.get('pending_move'),
        storm_to_edge_km_min=min(edges) if edges else None, frames_core_past_safe_edge=past_edge,
        segments=len(segments),
        failed_segments=sum(1 for s in segments if s.get('rc') not in (0, None)),
        frames=len(frames), fixes_ok=sum(1 for f in frames if f['fix'].get('ok')),
        max_offset_km=max(offsets) if offsets else None, interior_km=interior_km,
        a2_storm_inside_interior_every_frame=inside,
        fixes_source=state['fixes_source'],
        move_costs=move_costs(state),
        seals=[row for record in segments for row in ((record.get('control') or {}).get('recovery_seals') or [])])


def summarize_folder(run) -> dict:
    """Rewrite per-segment result.json and the run summary from a run's controller.json."""
    run = Path(run)
    state = json.loads((run / 'controller.json').read_text())
    case = load_case(state['case'])
    for record in state['segments']:
        if 'rc' in record:
            if Path(record['path']).is_dir():
                record.update(segment_receipt(record['path']))   # figures a newer reader takes from the receipts
            write_result(record['path'], record)
    summary = summarize(state, interior_km(case), mode=(case.get('tracker') or {}).get('mode') or 'tc')
    (run / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    return summary


def run_controller(a) -> int:
    controller = Controller(a)
    try:
        controller.run()
        return 0 if controller.state.get('summary', {}).get('status') == 'passed' else 1
    finally:
        # Children lead their own process groups; clean up only this run's
        # children before releasing its outer GPU grant, including interrupts.
        for proc in [*controller.procs.values(), *controller.forecast_children]:
            stop_process_tree(proc)
