"""Moving cases from track metadata. Runtime following uses model state only."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from .moving_case import MovingCaseError, great_circle_km, load_case, resolve_vertical_spec


def timestamp(value):
    try:
        result = dt.datetime.fromisoformat(str(value).replace('_', 'T').replace('Z', '+00:00'))
        if result.tzinfo is None:
            result = result.replace(tzinfo=dt.timezone.utc)
        return result.astimezone(dt.timezone.utc)
    except (ValueError, TypeError):
        raise MovingCaseError(f'invalid UTC track time: {value!r}') from None


def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise MovingCaseError(f'{label} must be a finite number') from None
    if not math.isfinite(result):
        raise MovingCaseError(f'{label} must be a finite number')
    return result


def read_track(path, *, storm=None, best_track=False):
    """Read tc-tracks tracks/fixes, moving best-track rows, or IBTrACS CSV."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == '.csv':
        if not best_track or not storm:
            raise MovingCaseError('IBTrACS CSV requires --best-track and --storm SID')
        from .moving_case import best_track_from_ibtracs
        document = best_track_from_ibtracs(path, storm)
    else:
        try:
            document = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise MovingCaseError(f'cannot read track {path}: {error}') from None
    if not isinstance(document, dict):
        raise MovingCaseError('track JSON must be an object containing tracks, fixes or rows')
    if 'tracks' in document:
        candidates = document['tracks']
        if not isinstance(candidates, list) or any(not isinstance(t, dict) for t in candidates):
            raise MovingCaseError('tracks must be an array of track objects')
        if storm is not None:
            candidates = [t for t in candidates if str(t.get('track_id')) == storm]
        if len(candidates) != 1:
            raise MovingCaseError('select exactly one track with --storm TRACK_ID; '
                                  f'found {len(candidates)} matching tracks')
        selected = candidates[0]
        identifier = selected.get('track_id')
        rows = selected.get('fixes', [])
    else:
        identifier = document.get('sid', document.get('track_id'))
        if storm and identifier and str(identifier) != storm:
            raise MovingCaseError(f'track identifier {identifier!r} does not match --storm {storm!r}')
        rows = document.get('rows', document.get('fixes'))
        if rows is None and ('time' in document or 'valid' in document):
            rows = [document]
    if not isinstance(rows, list) or not rows:
        raise MovingCaseError('selected track has no rows or fixes')
    fixes = {}
    for index, row in enumerate(rows):
        if row is None:
            continue
        if not isinstance(row, dict):
            raise MovingCaseError(f'row {index} must be a track-fix object or null')
        if row.get('lat') is None or row.get('lon') is None:
            continue
        time = timestamp(row.get('valid', row.get('time')))
        lat = _number(row['lat'], f'row {index} latitude')
        lon = _number(row['lon'], f'row {index} longitude')
        if abs(lat) > 90 or not -180 <= lon <= 360:
            raise MovingCaseError(f'row {index} coordinates outside latitude/longitude bounds')
        point = [lat, (lon + 180) % 360 - 180]
        if time in fixes and fixes[time] != point:
            raise MovingCaseError(f'conflicting positions at {time.isoformat()}')
        fixes[time] = point
    if not fixes:
        raise MovingCaseError('selected track contains no valid positions')
    return document, identifier, sorted(fixes.items()), path


SETUP_TARGETS = ('low', 'cape_shear', 'boundary', 'path', 'object')
TARGETS = ('tc',) + SETUP_TARGETS


def last_lead_hours(row, cycle) -> int:
    """The source's last posted forecast hour for this cycle (its table row's ladders or max_hours)."""
    from .source_horizon import SourceHorizonError, last_posted_lead
    try:
        return last_posted_lead(row, cycle.hour)
    except SourceHorizonError as error:
        raise MovingCaseError(str(error)) from None


def resolve_lead(row, cycle, lead_hours=0, from_last_hour=False) -> int:
    from .source_horizon import SourceHorizonError, resolve_lead as resolve
    try:
        return resolve(row, cycle, lead_hours, from_last_hour)
    except SourceHorizonError as error:
        raise MovingCaseError(str(error)) from None


def _path_rows(path, start):
    """A user's path file -> [[seconds since start, lat, lon], ...] (times UTC, ascending)."""
    _, _, fixes, path_file = read_track(path)
    rows = [[(time - start).total_seconds(), p[0], p[1]] for time, p in fixes]
    if len(rows) < 2:
        raise MovingCaseError('a path needs at least two timed points')
    return rows, path_file


def _interp_path(rows, seconds):
    from .storm_tracker import path_position
    return path_position(rows, seconds)


def mesh_block(fine_km=3.0, background_km=None, fine_radius_km=200.0):
    """The concentric cap family a case authored from these three numbers carries (``moving case``).

    One function, so the premade rotation template (``moving template``, :func:`woof.hex.moving_members.
    rotation_template`) and every authored case name the same rings and therefore the same generator spec."""
    fine = _number(fine_km, '--fine-km')
    background = (_number(background_km, '--background-km') if background_km is not None
                  else fine * 2**max(1, math.ceil(math.log2(48/fine))) if fine > 0 else 48)
    radius = _number(fine_radius_km, '--fine-radius-km')
    if not 0 < fine < background or radius <= 18*fine:
        raise MovingCaseError('require 0 < fine spacing < background and fine radius > 18 fine cells')
    ratio = math.log2(background/fine)
    if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
        raise MovingCaseError('background/fine spacing must be a power of two for the mesh generator')
    rings = [dict(spacing_km=fine*2**i, extra_radius_km=54*fine*(2**i-1),
                  transition_km=18*fine*2**i) for i in range(1, round(ratio))]
    return dict(background_km=background, fine_km=fine, fine_radius_km=radius,
                fine_transition_km=18*fine, rings=rings)


def author_case(path, *, track=None, cycle, storm=None, best_track=False, template=None,
                fine_km=3.0, background_km=None, fine_radius_km=200.0,
                hours=12.0, source='gfs', dt_seconds=None, physics_args=None,
                target='tc', point=None, user_path=None, segment=None, boundary_field='dewpoint',
                object_field=None, object_threshold=None, check_minutes=None, lead_hours=0,
                from_last_hour=False, target_config=None):
    """Seed from an exact analysis position and earlier fixes, never future observations.

    ``target`` picks what the fine region follows: ``tc`` (a tropical cyclone from a track position), or a
    severe-weather setup started at ``point`` (or the midpoint of ``segment``, or the start of ``user_path``):
    ``low``, ``cape_shear``, ``boundary`` (with ``segment``), ``path`` (``user_path``: not a tracker) or
    ``object`` (a thresholded field such as reflectivity). ``lead_hours`` / ``from_last_hour`` start from the
    source's forecast at that lead (the run begins at cycle + lead); beyond the source's last hour the run is a
    free run, and the case says so."""
    from .init_sources import load_table, resolve_source
    target = str(target).replace('-', '_')
    if target not in TARGETS:
        raise MovingCaseError(f'--target {target!r} is not one of {", ".join(TARGETS)}')
    source_row = resolve_source(load_table(), source)
    source = source_row['id']
    source_cycle = timestamp(cycle)
    if source_cycle.minute or source_cycle.second or source_cycle.microsecond:
        raise MovingCaseError('--cycle must be a whole UTC analysis hour')
    lead = resolve_lead(source_row, source_cycle, lead_hours, from_last_hour)
    analysis = source_cycle + dt.timedelta(hours=lead)
    identifier, fixes, track_path, path_rows = None, [], None, None
    if target == 'path':
        if user_path is None:
            raise MovingCaseError('--target path needs --path (the timed points the region follows)')
        path_rows, track_path = _path_rows(user_path, analysis)
        center = list(_interp_path(path_rows, 0.0))
    elif track is not None:
        _, identifier, fixes, track_path = read_track(track, storm=storm, best_track=best_track)
        at_analysis = [p for time, p in fixes if time == analysis]
        if not at_analysis:
            raise MovingCaseError('track has no valid position exactly at the start time; '
                                  'an interpolated future best-track position cannot seed an analysis')
        center = at_analysis[0]
    elif point is not None:
        center = [_number(point[0], '--point latitude'), (_number(point[1], '--point longitude') + 180) % 360 - 180]
    elif segment is not None:
        a, b = (segment[0], segment[1]), (segment[2], segment[3])
        from .storm_tracker import project, unproject
        x, y = project(a[0], a[1], b[0], b[1])
        mid = unproject(a[0], a[1], float(x) / 2, float(y) / 2)
        center = [float(mid[0]), float(mid[1])]
    else:
        raise MovingCaseError('give a track (--tracks/--best-track), a --point, a --segment or a --path')
    if target == 'tc' and track is None and point is None:
        raise MovingCaseError('--target tc needs a track position or a --point')
    if target == 'boundary' and segment is None and point is None:
        raise MovingCaseError('--target boundary needs --segment (the dryline or front as drawn) or a --point')
    past = [(time, p) for time, p in fixes if dt.timedelta(0) < analysis-time <= dt.timedelta(hours=6)]
    heading, speed = 0.0, 0.0
    motion_times = []
    if past:
        previous_time, previous = past[-1]
        elapsed = (analysis-previous_time).total_seconds()/3600
        speed = great_circle_km(previous, center)/elapsed
        lat1, lat2 = map(math.radians, (previous[0], center[0]))
        delta = math.radians((center[1]-previous[1]+180)%360-180)
        heading = math.degrees(math.atan2(math.sin(delta)*math.cos(lat2),
            math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(delta)))%360
        motion_times = [previous_time.isoformat(), analysis.isoformat()]
    if template:
        case = load_case(template)
        vertical = resolve_vertical_spec(case)
        packaged = Path(__file__).with_name('data') / 'countries' / vertical.name
        case['vertical_spec'] = vertical.name if packaged.is_file() and packaged.read_bytes() == vertical.read_bytes() else str(vertical)
        case.pop('_path', None)
        case['physics_args'] = case.pop('_requested_physics_args', case.get('physics_args', []))
        case['tracker'] = dict(case.get('tracker') or {})
        case['tracker'].pop('initial_center_deg', None)
        case.pop('gfs_cycle', None)
    else:
        mesh = mesh_block(fine_km, background_km, fine_radius_km)
        fine, radius = mesh['fine_km'], mesh['fine_radius_km']
        case = dict(n_levels=55, vertical_spec='tc55-v1.json', dt_seconds=6.25*fine,
                    mesh=mesh,
                    tracker=dict(mode='tc', placement=dict(
                        core_margin_km=min(50.0, (radius-18*fine)/3), check_minutes=30.0,
                        min_move_km=min(60.0, (radius-18*fine)/2), max_move_km=2*(radius-18*fine),
                        max_dwell_minutes=180.0, follow='continuous')), physics_args=[])
    tracker = case.setdefault('tracker', {})
    tracker['mode'] = target
    if check_minutes is not None:
        tracker.setdefault('placement', {})['check_minutes'] = _number(check_minutes, '--check-minutes')
    if target == 'path':
        tracker['path'] = dict(rows=path_rows)
    elif target == 'boundary':
        block = dict(field=boundary_field)
        if segment is not None:
            block['segment'] = [[float(segment[0]), float(segment[1])], [float(segment[2]), float(segment[3])]]
        tracker['boundary'] = block
    elif target == 'object':
        block = {}
        if object_field:
            block['field_name'] = object_field
        if object_threshold is not None:
            block['threshold'] = _number(object_threshold, '--object-threshold')
        tracker['object'] = block
    if target_config:
        extra = json.loads(Path(target_config).read_text(encoding='utf-8'))
        for key, value in extra.items():
            tracker[key] = dict(tracker.get(key) or {}, **value) if isinstance(value, dict) else value
    if dt_seconds is not None:
        case['dt_seconds'] = _number(dt_seconds, '--dt-seconds')
    step = _number(case['dt_seconds'], 'dt_seconds')
    duration = _number(hours, '--hours')
    check_minutes = float((case.get('tracker', {}).get('placement') or {}).get('check_minutes', 60))
    macro = 8*step
    for label, seconds in [('radiation interval', 1800), ('tracker interval', 60*check_minutes),
                           ('duration', 3600*duration)]:
        if step <= 0 or seconds <= 0 or not math.isclose(seconds/macro, round(seconds/macro), abs_tol=1e-8):
            raise MovingCaseError(f'{label} must be a positive whole multiple of 8 * dt_seconds ({macro:g} s)')
    if not math.isclose(duration*60/check_minutes, round(duration*60/check_minutes), abs_tol=1e-8):
        raise MovingCaseError('--hours must contain a whole number of tracker intervals')
    if physics_args is not None:
        case['physics_args'] = list(physics_args)
    describe = {'tc': 'Track-authored moving fine region; model-state following',
                'low': 'Setup-following fine region: a surface low, followed on model state',
                'cape_shear': 'Setup-following fine region: the dominant CAPE x shear area, followed on model state',
                'boundary': f'Setup-following fine region: a {boundary_field} boundary, followed on model state',
                'object': 'Setup-following fine region: a thresholded feature, followed on model state',
                'path': "Fine region on the user's path (not a tracker)"}[target]
    case.update(case=Path(path).stem, description=describe,
                start_time=analysis.strftime('%Y-%m-%d_%H:%M:%S'), hours=duration, init_source=source)
    if lead:
        case['init_cycle'] = source_cycle.strftime('%Y-%m-%d_%H:%M:%S')
        case['init_lead_hours'] = lead
    from .source_horizon import beyond_source
    extension = beyond_source(source_row, source_cycle, lead, duration)
    if extension:
        # Breakage it prevents: a run past the source's last hour read as if the source covered it. The moving
        # region's globe needs no boundaries, so the extra hours are a free run of our own model.
        case['beyond_source'] = extension
    origin = ({'tc': 'selected track position at the analysis time' if track is not None else 'the given point at the start time',
               'path': "the user's path at the start time"}.get(target)
              or ('the midpoint of the drawn segment' if point is None and segment is not None
                  else 'the given point at the start time'))
    case['storm'] = dict(initial_center_deg=center, initial_center_source=origin, target=target,
        initial_motion=dict(heading_deg=heading, speed_kmh=speed,
                            source='positions at and before analysis' if past else 'no prior fix within 6 h; stationary seed'))
    if best_track and identifier:
        case['storm']['ibtracs_sid'] = identifier
    case['authoring'] = dict(schema='gpuwm-hex.moving-author/v1',
        track=None if track_path is None else str(track_path),
        track_sha256=None if track_path is None else hashlib.sha256(track_path.read_bytes()).hexdigest(),
        track_id=identifier, target=target, point=None if point is None else list(map(float, point)),
        segment=None if segment is None else list(map(float, segment)),
        kind=('best-track analysis seed' if best_track else 'tracks analysis seed' if track is not None
              else "user's path" if target == 'path' else 'point seed'),
        analysis_utc=analysis.isoformat(), source_cycle_utc=source_cycle.isoformat(), init_lead_hours=lead,
        motion_times_utc=motion_times,
        future_positions_used=target == 'path', runtime_fixes=("the user's path" if target == 'path' else 'model state'),
        template=str(Path(template).resolve()) if template else None)
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(case, indent=2, allow_nan=False)+'\n')
    return case
