"""Table-driven lake state: Rust CF reads/remapping, CUDA column binding.

Automatic sources are public and keyless. Licensed sources require a local
file and exact licence acceptance. Credentials are never discovered.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import contextvars
import ctypes
import functools
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import tomllib
from urllib.error import HTTPError
from urllib.request import urlopen
import numpy as np

TABLE = Path(__file__).resolve().parents[1] / 'data/surface_obs/lake-state-sources.v1.toml'
FIELDS = ('water_temperature', 'bottom_temperature', 'mixed_layer_depth',
          'temperature_shape_factor', 'ice_thickness', 'ice_fraction', 'ice_temperature')


def source_rows(path=TABLE):
    doc = tomllib.loads(Path(path).read_text())
    if doc.get('schema') != 'gpuwm-lake-state-sources-v1':
        raise ValueError('unrecognized lake-state source schema')
    rows = {r['id']: r for r in doc['source']}
    if len(rows) != len(doc['source']):
        raise ValueError('duplicate lake-state source ids')
    return rows


def lake_state_choices(path=TABLE):
    """What ``--lake-state`` accepts: auto, forcing, then every table row.

    A row added to the source table is selectable at the input doors with
    no code change; the choices are read from the table, not listed here.
    """
    return ('auto', 'forcing', *source_rows(path))


def add_lake_state_arguments(parser, *, prefix=''):
    """``--lake-state`` and ``--lake-state-path``, spelled once for every input door.

    ``woof run`` and ``woof go`` and both worker parsers register these,
    so the flag a user types is the flag the worker child receives.
    Registered only when absent, as the composed go parser needs.
    """
    if '--lake-state' not in parser._option_string_actions:
        parser.add_argument(
            '--lake-state', dest='lake_state', choices=lake_state_choices(), default=None,
            help=prefix + 'where the lakes start from at a cold start: auto (default) requests '
                 'the analysis of an automatic source that covers the domain\'s lakes (the '
                 'NOAA GLERL Great Lakes analysis), forcing keeps the input files\' own lake '
                 'surface and requests nothing, or a source-table row by id')
    if '--lake-state-path' not in parser._option_string_actions:
        parser.add_argument(
            '--lake-state-path', dest='lake_state_path', default=None, metavar='FILE',
            help=prefix + 'the acquired CF NetCDF file a --lake-state row reads, for a row '
                 'with no download address')


def select_lake_state(toml_text, source=None, path=None):
    """Write an input door's ``--lake-state`` choice into its resolved TOML.

    The met_em and wrfinput doors build their configuration from the
    producing namelist, which has no lake-state key, so before this every
    such run took the RunConfig default ``auto`` and asked for the Great
    Lakes analysis whenever the domain held lake cells (item 4 of the
    2026-11-12 runner defects).  Both doors call this one function on the
    imported TOML before it is built, so the choice reaches every domain's
    RunConfig, the prepared ``experiment.toml`` and the restart identity
    through the same keys a CONFIG run sets by hand.

    ``source=None`` writes nothing: the TOML stays byte-identical to the
    base, and the run keeps the ``auto`` default.  Returns
    ``(toml_text, sentence)``; the sentence says what was chosen, for the
    door to print at plan review, and is ``None`` when nothing was.
    """
    if source is None:
        if path is not None:
            # Breakage it prevents: the file would be read by nothing.
            raise ValueError('--lake-state-path names the file a --lake-state row reads; '
                             'give --lake-state ROW with it')
        return toml_text, None
    rows = source_rows()
    choices = ('auto', 'forcing', *rows)
    if source not in choices:
        raise ValueError(f'--lake-state {source!r} is not a lake-state source; '
                         f'choices: {", ".join(choices)}')
    from woof.verify_exact.replay import set_toml_key
    text = set_toml_key(toml_text, 'lake_state_source', source)
    if source == 'forcing':
        if path is not None:
            raise ValueError('--lake-state forcing reads no lake file; remove --lake-state-path')
        return text, ('lake state: forcing (--lake-state); lakes start from the input '
                      "files' own surface temperature and ice, and no lake analysis is requested")
    if source == 'auto':
        if path is not None:
            # Breakage it prevents: auto picks whichever automatic row covers
            # the domain, and would decode this file as that row's product.
            raise ValueError('--lake-state auto chooses its row by coverage and downloads it; '
                             'name the row the file belongs to with --lake-state ROW')
        return text, ('lake state: auto (--lake-state); where an automatic source covers '
                      'the domain\'s lakes its analysis is requested at the start')
    row = rows[source]
    if path is None and any('url' not in field for field in row['field']):
        # Breakage it prevents: the run would take a card and then refuse at
        # surface initialization, where load_state finds no file to read.
        raise ValueError(f'--lake-state {source} has no download address in the source table; '
                         'give its acquired CF NetCDF file with --lake-state-path FILE')
    if path is not None:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'--lake-state-path {path} is not a file')
        text = set_toml_key(text, 'lake_state_path', str(path))
    where = f'from {path}' if path is not None else 'downloaded at the start'
    return text, f'lake state: {source} (--lake-state); {row["citation"]}, {where}'


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def sample_field(latitude, longitude, values, target_latitude, target_longitude, *, scale=1., offset=0., zero_roundoff_tolerance=0.):
    """Marshal arrays only; interpolation and SI conversion run in Rust."""
    from woof.ingest.cpu_backend import masked_fields_cpu_backend, host_step_workers
    backend = masked_fields_cpu_backend()
    try:
        entry = backend._library.gpuwm_lake_state_sample_f64
    except AttributeError as e:
        raise RuntimeError(f'{backend.path} lacks gpuwm_lake_state_sample_f64; rebuild tools/grib1_bridge') from e
    la, lo, data, tla, tlo = [np.ascontiguousarray(v, dtype=np.float64) for v in
              (latitude, longitude, values, target_latitude, target_longitude)]
    if la.ndim != 1 or lo.ndim != 1 or data.shape != (la.size, lo.size) or tla.shape != tlo.shape:
        raise ValueError('lake source axes/field or target coordinates have incompatible dimensions')
    out = np.empty(tla.shape, np.float64)
    ptr = ctypes.c_void_p
    entry.argtypes = [ptr, ctypes.c_size_t, ptr, ctypes.c_size_t, ptr, ctypes.c_double,
                      ctypes.c_double, ptr, ptr, ctypes.c_size_t, ptr, ctypes.c_size_t]
    entry.restype = ctypes.c_int32
    rc = entry(ptr(la.ctypes.data), la.size, ptr(lo.ctypes.data), lo.size, ptr(data.ctypes.data),
               scale, offset, ptr(tla.ctypes.data), ptr(tlo.ctypes.data), tla.size, ptr(out.ctypes.data), host_step_workers(backend))
    if rc:
        raise ValueError(f'lake-state sampler refused malformed coordinates (native status {rc})')
    if zero_roundoff_tolerance:
        normalize = backend._library.gpuwm_lake_state_zero_roundoff_f64
        normalize.argtypes = [ptr, ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t]
        normalize.restype = ctypes.c_int32
        if normalize(ptr(out.ctypes.data), out.size, zero_roundoff_tolerance, host_step_workers(backend)):
            raise ValueError('lake source zero-roundoff tolerance must be finite and nonnegative')
    return out


def decode_field(path, row, field, valid_time, lat, lon):
    from woof.netcdf_bridge import Dataset
    with Dataset(path) as ds:
        times = tuple(utc(v) for v in ds.variables[row['time']].times())
        admitted = [(t, i) for i, t in enumerate(times) if t <= utc(valid_time)]
        if not admitted:
            raise ValueError(f'{path}: no lake analysis at or before {valid_time}')
        selected, index = max(admitted)
        age = (utc(valid_time)-selected).total_seconds()/3600
        if age > row['max_age_hours']:
            raise ValueError(f'{path}: lake analysis is {age:g} h old, row limit {row["max_age_hours"]} h')
        variable = ds.variables[field['variable']]
        axes = (row['latitude'], row['longitude'])
        expected = (row['time'], *axes)
        if variable.dimensions != expected:
            raise ValueError(f'{path}: {variable.name} axes {variable.dimensions} != {expected}')
        if variable.units not in (field['units'], *field.get('unit_aliases', ())):
            raise ValueError(f'{path}: {variable.name} units {variable.units!r} != {field["units"]!r}')
        result = sample_field(ds.variables[axes[0]][:], ds.variables[axes[1]][:],
            variable[index], lat, lon, scale=field.get('scale', 1.), offset=field.get('offset', 0.),
            zero_roundoff_tolerance=field.get('zero_roundoff_tolerance', 0.))
    return result, dict(field=field['name'], variable=field['variable'], file=str(path), sha256=sha256(path),
                        valid_time=selected.isoformat(), age_hours=age, zero_roundoff_tolerance=field.get('zero_roundoff_tolerance', 0.), covered_targets=int(verification_scores(result, result)['count']))


class LakeAnalysisUnavailable(ValueError):
    """No analysis day inside the row's age limit was served; names each day asked."""


def analysis_stamps(row, valid_time):
    """Analysis times to ask for, newest first, back to the row's ``max_age_hours``.

    A daily analysis is posted the day after it is valid, so the newest day
    is missing for any start near or after the wall clock (measured: GLSEA
    for 2026-11-11T12Z answered HTTP 404 "greater than the axis maximum"
    while 2026-11-10T12Z was served).  The row's own age limit, which
    ``decode_field`` enforces, bounds how far back a day is still usable.
    The newest stamp is always asked, as before.
    """
    valid = utc(valid_time)
    stamp = valid.replace(hour=row.get('analysis_hour', 12), minute=0, second=0, microsecond=0)
    if stamp > valid:
        stamp -= timedelta(days=1)
    stamps = [stamp]
    while (valid-(stamps[-1]-timedelta(days=1))).total_seconds()/3600 <= row['max_age_hours']:
        stamps.append(stamps[-1]-timedelta(days=1))
    return stamps


def _fetch(url, path):
    partial = path.with_suffix(f'.{os.getpid()}.part')
    try:
        with urlopen(url, timeout=30) as response, partial.open('wb') as out:
            for block in iter(lambda: response.read(1024*1024), b''):
                out.write(block)
        with partial.open('rb') as f:
            if f.read(3) not in (b'CDF', b'\x89HD'):
                raise ValueError('response is not NetCDF')
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def acquire_field(row, field, valid_time, bounds, cache, attempts=None):
    """Fetch the newest served analysis day inside the row's age limit.

    ``attempts`` (a list) receives one entry per day asked, so the receipt
    and the run's lake line can say which days were missing and why.  A
    day the server answers for (an HTTP status, or a body that is not
    NetCDF) moves to the day before; a request that never reached the
    server (no network, timeout) stops at once, since an earlier day
    would meet the same wall.
    """
    attempts = [] if attempts is None else attempts
    label = field.get('dataset', field['name'])
    for stamp in analysis_stamps(row, valid_time):
        when = stamp.strftime('%Y-%m-%dT%H:%M:%SZ')
        url = field['url'].format(**field, time=when, **bounds)
        path = cache / (hashlib.sha256(url.encode()).hexdigest()+'.nc')
        if path.exists():
            attempts.append(dict(time=when, outcome='cached'))
            return path, url
        try:
            _fetch(url, path)
        except HTTPError as e:
            attempts.append(dict(time=when, outcome=f'HTTP {e.code}'))
            continue
        except ValueError as e:
            attempts.append(dict(time=when, outcome=str(e)))
            continue
        except OSError as e:
            attempts.append(dict(time=when, outcome=f'{type(e).__name__}: {getattr(e, "reason", e)}'))
            break
        attempts.append(dict(time=when, outcome='fetched'))
        return path, url
    asked = ', '.join(f'{a["time"]} {a["outcome"]}' for a in attempts)
    raise LakeAnalysisUnavailable(
        f'{label} served no analysis within {row["max_age_hours"]:g} h before '
        f'{utc(valid_time).strftime("%Y-%m-%dT%H:%MZ")} ({asked})')



def coordinate_coverage(latitude, longitude, row):
    """Rust request envelope and in-source count; Python only marshals arrays."""
    from woof.ingest.cpu_backend import masked_fields_cpu_backend, host_step_workers
    backend = masked_fields_cpu_backend()
    arrays = [np.ascontiguousarray(v, np.float64) for v in (latitude, longitude)]
    if arrays[0].shape != arrays[1].shape:
        raise ValueError('lake request coordinates must have identical shapes')
    entry = backend._library.gpuwm_lake_state_coverage_f64
    ptr = ctypes.c_void_p
    entry.argtypes = [ptr, ptr, ctypes.c_size_t] + [ctypes.c_double]*4 + [ptr, ctypes.c_size_t]
    entry.restype = ctypes.c_int32
    out = np.empty(5, np.float64)
    rc = entry(*(ptr(a.ctypes.data) for a in arrays), arrays[0].size,
        row.get('south', -90.), row.get('north', 90.), row.get('west', -180.), row.get('east', 180.),
        ptr(out.ctypes.data), host_step_workers(backend))
    if rc:
        raise ValueError(f'lake request has no finite geographic coordinates (native status {rc})')
    return dict(zip(('south', 'north', 'west', 'east', 'covered'), map(float, out)))


def _extent(row):
    """A row's declared extent in words, or None for a row with none (it covers everywhere)."""
    if not all(k in row for k in ('south', 'north', 'west', 'east')):
        return None
    lon = lambda v: f'{abs(v):.1f}{"W" if v < 0 else "E"}'
    lat = lambda v: f'{abs(v):.1f}{"S" if v < 0 else "N"}'
    return f'{lat(row["south"])} to {lat(row["north"])}, {lon(row["west"])} to {lon(row["east"])}'


def automatic_coverage(latitude, longitude, path=TABLE):
    """Each automatic row, in table order, with how many of these points its extent holds.

    THE ``auto`` decision, made in one place: the run asks the first row
    that holds any of the domain's lake cells (:func:`load_state`), and the
    plan asks the same question of the domain's own cells before the lake
    mask exists (:func:`plan_sentence`).  Lake cells are domain cells, so a
    domain no row reaches gets no request at the start and none in the
    plan.  Breakage it prevents (defect 3 of the 2026-11-12 card runs): the plan
    decided from the padded fetch area, or from nothing on the met_em door,
    and promised a GLSEA request for open Atlantic at 31N, for the Beaufort
    Sea and for Oklahoma, none of which the run made.  The count is the
    Rust coverage entry the run already uses.
    """
    return [(row, int(coordinate_coverage(latitude, longitude, row)['covered']))
            for row in source_rows(path).values() if row.get('automatic', False)]


def load_state(source, valid_time, lat, lon, *, path='', accept_licence='', cache=None):
    """Sample canonical fields and return provenance, ages, coverage and errors."""
    started = time.monotonic()
    rows = source_rows()
    if source == 'auto':
        checked = automatic_coverage(lat, lon)
        row = next((r for r, inside in checked if inside > 0), None)
        if row is None:
            # Which rows were asked, and how many lake cells each holds, so
            # the run record says which source it would have used, and why
            # none was.
            return {}, dict(schema='gpuwm.lake-init.v1', status='outside_source_coverage', requested=source,
                            source='forcing', target_cells=int(np.size(lat)), analysis_fetch='none',
                            checked=[dict(source=r['id'], extent=_extent(r), lake_cells_inside=inside)
                                     for r, inside in checked])
    else:
        if source not in rows:
            raise ValueError(f'unknown lake-state source {source!r}; choices: auto, forcing, {", ".join(rows)}')
        row = rows[source]
    if row['research_only'] and accept_licence != row['licence']:
        raise ValueError(f'lake source {row["id"]} requires lake_state_accept_licence={row["licence"]!r}; its data cannot be a default')
    if not path and any('url' not in f for f in row['field']):
        raise ValueError(f'lake source {row["id"]} requires lake_state_path to an acquired CF NetCDF file')
    cache = Path(cache or os.environ.get('WOOF_LAKE_STATE_CACHE', str(Path.home()/'.woof/cache/lake-state')))
    cache.mkdir(parents=True, exist_ok=True)
    # THE REQUEST IS THE SOURCE'S WHOLE DECLARED EXTENT, not the targets'
    # envelope.  Cropped to the targets, the same lake cell was sampled from
    # a different grid window depending on which other cells were asked for
    # in the same call, and the bilinear weights moved by a last-place bit:
    # the resident 5 km CONUS run and its slab-built streamed twin
    # disagreed by up to 3e-8 in 13 Great Lakes ice-fraction cells (X12).
    # A row with no declared extent keeps the envelope crop.
    if all(key in row for key in ('south', 'north', 'west', 'east')):
        bounds = {key: row[key] for key in ('south', 'north', 'west', 'east')}
    else:
        envelope = coordinate_coverage(lat, lon, row)
        bounds = dict(south=max(row.get('south', -90.), envelope['south']-.04),
                      north=min(row.get('north', 90.), envelope['north']+.04),
                      west=max(row.get('west', -180.), envelope['west']-.04),
                      east=min(row.get('east', 180.), envelope['east']+.04))
    date = utc(valid_time).date().isoformat()
    fields = [f for f in row['field'] if f.get('start', '0000') <= date <= f.get('end', '9999')]
    asked = {}
    def read(field):
        attempts = asked.setdefault(field['name'], [])
        p, url = (Path(path), None) if path else acquire_field(row, field, valid_time, bounds, cache, attempts)
        data, evidence = decode_field(p, row, field, valid_time, lat, lon)
        evidence['url'] = url
        evidence['attempts'] = attempts
        return field['name'], data, evidence
    values, records, errors = {}, [], []
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(fields)))) as executor:
        futures = [(f, executor.submit(read, f)) for f in fields]
        for f, future in futures:
            try:
                name, data, record = future.result()
                values[name] = data
                records.append(record)
            except LakeAnalysisUnavailable as e:
                errors.append(f'{f["name"]}: {e}')
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                errors.append(f'{f["name"]}: {type(e).__name__}: {e}')
    uncovered = {r['field']: int(lat.size)-r['covered_targets'] for r in records if r['covered_targets'] < lat.size}
    receipt = dict(schema='gpuwm.lake-init.v1', source=row['id'], requested=source,
        fetch_attempts={name: list(a) for name, a in asked.items()}, licence=row['licence'],
        licence_url=row['licence_url'], citation=row['citation'], records=records, errors=errors,
        status=('partial_forcing_fallback' if errors or uncovered else 'bound') if values else 'forcing_fallback', uncovered_targets=uncovered, wall_seconds=time.monotonic()-started,
        target_cells=int(lat.size), interpolation='Rust local bilinear, finite donors only, no extrapolation',
        profile_rule='CLM lakeini surface-constrained profile; supplied bottom/mixed-layer/shape fields use an integral-shape power profile',
        ice_rule='thickness times horizontal concentration distributed top-down as equivalent ice volume; concentration alone retains the lakeini thickness prior, reported as unobserved')
    if errors and source != 'auto':
        raise ValueError('explicit lake-state source incomplete: '+'; '.join(errors))
    return values, receipt


def _hour(text):
    return utc(text).strftime('%Y-%m-%dT%H:%MZ')


def describe(receipt):
    """The run's one lake-state line: what the lakes started from, and why.

    Every status gets a line, so a run never leaves the reader to infer
    from silence that its lakes came from the forcing.  A fallback starts
    with WARNING and names each day the source was asked for and what it
    answered.
    """
    status = receipt.get('status')
    requested = receipt.get('requested', receipt.get('source', 'auto'))
    tail = f'; receipt {receipt["receipt_path"]}' if receipt.get('receipt_path') else ''
    if status == 'retained_forcing':
        return f'lake state: forcing, because {receipt.get("reason", "lake_state_source = " + repr(requested))}' + tail
    if status == 'store_buffer':
        return 'lake state: gathered from the prepared store; no lake analysis requested' + tail
    if status == 'restored_from_checkpoint':
        before = receipt.get('start_up_bind') or {}
        overwritten = (f'; the start-up bind ({before.get("status")}, {before.get("source", "?")}) '
                       'was overwritten by it' if before.get('status') else
                       '; no lake analysis was fetched' if receipt.get('analysis_fetch') == 'skipped'
                       else '')
        where = f' {receipt["checkpoint"]}' if receipt.get('checkpoint') else ''
        return f'lake state: restored from {receipt.get("source")}{where}{overwritten}'
    if status == 'restore_pending':
        return f'lake state: to be restored from {receipt.get("source")}; no lake analysis is fetched'
    if status == 'no_lake_cells':
        return 'lake state: no lake cells in this domain; no lake analysis requested' + tail
    if status == 'outside_source_coverage':
        cells = receipt.get('target_cells')
        checked = '; '.join(
            f'{c["source"]} covers {c["extent"] or "its extent"} and holds {c["lake_cells_inside"]}'
            + (f' of the {cells}' if cells is not None else '')
            for c in receipt.get('checked', ()))
        return ('lake state: forcing, because no automatic lake analysis covers this domain\'s lake cells '
                f'(lake_state_source = {requested!r}' + (f'; {checked}' if checked else '')
                + '); no lake analysis requested' + tail)
    if status == 'missing_time_or_coordinates':
        return ('WARNING lake state: forcing, because this route gave the lake binding no start time '
                'or surface coordinates' + tail)
    source = receipt.get('source', '?')
    bound = {}
    for record in receipt.get('records', ()):
        bound.setdefault((record.get('valid_time'), record.get('age_hours')), []).append(record['field'])
    used = '; '.join(
        f'{source} {", ".join(names)} from the analysis of {_hour(when)} ({age:g} h before the start)'
        for (when, age), names in bound.items() if when is not None)
    missed = ''.join(dict.fromkeys(f' (the newer {a["time"][:16]}Z: {a["outcome"]})'
                     for record in receipt.get('records', ())
                     for a in record.get('attempts', ())[:-1]))
    errors = '; '.join(receipt.get('errors', ()))
    uncovered = receipt.get('uncovered_targets') or {}
    cells = receipt.get('temperature_bound_cells', receipt.get('ice_fraction_bound_cells'))
    where = f' in {cells} of {receipt.get("target_cells")} lake cells' if cells is not None else ''
    if status == 'bound':
        return f'lake state: {used}{missed}{where}' + tail
    if status == 'forcing_fallback':
        return f'WARNING lake state: forcing, because {source} gave nothing usable: {errors}' + tail
    # Inland lakes the analysis grid does not reach are expected and said
    # plainly; a failed field or a refused value is a WARNING.
    parts = [f'{"WARNING " if errors else ""}lake state: {used}{missed}{where}']
    if errors:
        parts.append(f'forcing for the rest, because {errors}')
    if uncovered:
        parts.append('forcing for cells outside the analysis: ' + ', '.join(f'{k} {v}' for k, v in uncovered.items()))
    return '; '.join(parts) + tail


def _area_bounds(area):
    """``south,west,north,east`` from a fetch area string, or None when it is not one."""
    try:
        south, west, north, east = (float(v) for v in str(area).split(','))
    except (TypeError, ValueError):
        return None
    return dict(south=south, west=west, north=north, east=east)


def _overlaps(row, box):
    if box is None or not all(k in row for k in ('south', 'north', 'west', 'east')):
        return None
    return (box['south'] <= row['north'] and box['north'] >= row['south']
            and box['west'] <= row['east'] and box['east'] >= row['west'])


def plan_sentence(cfg, *, start_time=None, area=None, domain=None, lakes=None):
    """The plan's and dry run's lake-state line, from the config and the source table.

    It names what the run will ask for before anything is spent; the run's
    own line (:func:`describe`) then says what it got.  ``domain`` is the
    root domain's cell latitudes and longitudes: with it, ``auto`` is
    decided by :func:`automatic_coverage`, the run's own test, so a domain
    no automatic row reaches is told that nothing is requested.  Without
    it (a configuration with no projection) the fetch area stands in.

    A row the domain does reach is named WITH ITS CONDITION, because the
    plan is written before the lake mask exists: the run asks the row only
    if a lake cell is inside its extent (:func:`load_state`).  Breakage it
    prevents: a domain whose corner enters GLSEA's box while its lakes lie
    south of it (the Ohio valley) was told "the run asks glsea", and the
    run, counting lake cells, asked nothing.

    ``lakes`` is the latitudes and longitudes of the domain's LAKE cells,
    for a door whose input files already hold the lake mask
    (:func:`input_lake_cells`).  With them the plan makes the run's
    decision outright, on the cells and through the count
    :func:`load_state` uses: which row is asked and how many lake cells
    it holds, or that none is asked, with no condition left to read.
    """
    source = getattr(cfg, 'lake_state_source', 'auto')
    if source == 'forcing':
        return 'lake state: forcing (lake_state_source = "forcing"); no lake analysis is requested'
    rows = source_rows()
    if source != 'auto':
        row = rows.get(source)
        if row is None:
            return f'lake state: {source!r} is not a lake-state source; choices: auto, forcing, {", ".join(rows)}'
        where = getattr(cfg, 'lake_state_path', '') or 'its acquired file (lake_state_path)'
        return (f'lake state: {row["id"]} ({row["citation"]}) from {where}; a missing or incomplete '
                'field stops the run')
    when = '' if start_time is None else f' before {_hour(start_time.isoformat() if hasattr(start_time, "isoformat") else start_time)}'
    if lakes is not None:
        latitude, longitude = lakes
        count = int(np.size(latitude))
        if count == 0:
            return 'lake state: no lake cells in this domain; no lake analysis is requested'
        checked = automatic_coverage(latitude, longitude)
        # load_state takes the first row that holds a lake cell, in table order.
        asked = next(((row, inside) for row, inside in checked if inside > 0), None)
        if asked is None:
            held = '; '.join(f'{row["id"]} covers only {_extent(row)} and holds none of them'
                             for row, _ in checked if _extent(row) is not None)
            return ('lake state: forcing (lake_state_source = "auto", and no automatic lake analysis covers '
                    f'this domain\'s {count} lake cells' + (f': {held}' if held else '') + '); the lakes '
                    'keep the forcing\'s own state and no lake analysis is requested')
        row, inside = asked
        extent = _extent(row)
        holds = (f'which holds all {count} of the domain\'s lake cells' if extent is None else
                 f'whose extent {extent} holds {inside} of the domain\'s {count} lake cells')
        return (f'lake state: auto -- at the start the run asks {row["id"]} ({row["citation"]}; the newest '
                f'analysis posted up to {row["max_age_hours"]:g} h{when}), {holds}; lake cells the analysis '
                'does not cover keep the forcing\'s lake state, as all of them do, with a WARNING, when it '
                'is not served; lake_state_source = "forcing" skips the request')
    if domain is not None:
        latitude, longitude = domain
        checked = automatic_coverage(latitude, longitude)
        candidates = [(row, f', where {inside} of the domain\'s {int(np.size(latitude))} cells are')
                      for row, inside in checked if inside > 0]
        if not candidates:
            reach = '; '.join(f'{row["id"]} covers only {_extent(row)}' for row, _ in checked
                              if _extent(row) is not None)
            return ('lake state: forcing (lake_state_source = "auto", and no automatic lake analysis reaches '
                    'this domain' + (f': {reach}' if reach else '') + '); the lakes keep the forcing\'s own '
                    'state and no lake analysis is requested')
    else:
        box = _area_bounds(area)
        candidates = [(row, '') for row in rows.values()
                      if row.get('automatic', False) and _overlaps(row, box) is not False]
        if not candidates:
            return ('lake state: forcing (lake_state_source = "auto", and no automatic lake analysis covers '
                    'the fetch area); no lake analysis is requested')
    def offer(row, held):
        extent = _extent(row)
        condition = ('if the domain has a lake cell' if extent is None else
                     f'if any of the domain\'s lake cells is inside {extent}{held}')
        return (f'{row["id"]} ({row["citation"]}; the newest analysis posted up to '
                f'{row["max_age_hours"]:g} h{when}) {condition}')
    offers = [offer(row, held) for row, held in candidates]
    # load_state takes the first row that holds a lake cell, in table order.
    asks = offers[0] if len(offers) == 1 else ('the first of these that applies: '
                                              + '; '.join(offers))
    return ('lake state: auto -- at the start the run asks ' + asks
            + '; otherwise no lake analysis is requested; lake cells the analysis does not cover keep '
            'the forcing\'s lake state, as all of them do, with a WARNING, when it is not served; '
            'lake_state_source = "forcing" skips the request')


def experiment_plan_sentence(experiment, *, area=None, restart=None):
    """:func:`plan_sentence` for an experiment's root domain (the lake keys are [shared]).

    A resume (``restart``, a checkpoint) takes its lakes from the
    checkpoint and fetches no analysis (:func:`skips_analysis_on_resume`),
    so its plan says that instead of naming a request the run will not make.
    """
    if restart is not None:
        return (f'lake state: a resume takes its lakes from the checkpoint {restart}; '
                'no lake analysis is fetched')
    run = experiment.domains[0].run
    if getattr(run, 'lake_state_source', 'auto') != 'auto':
        return plan_sentence(run, start_time=experiment.start_time, area=area)
    domain = root_cells(experiment)
    try:
        return plan_sentence(run, start_time=experiment.start_time, area=area, domain=domain)
    except (OSError, AttributeError, RuntimeError) as error:
        if domain is None:
            raise
        # The CPU library that counts the cells is missing or predates the
        # coverage entry; the run's own count would stop the same way, so
        # the plan says which area it fell back to and why.
        return (plan_sentence(run, start_time=experiment.start_time, area=area)
                + f' (decided from the fetch area: the domain\'s cells could not be counted: {error})')


def input_lake_cells(path, *, latitude, longitude, lake_mask_record=None):
    """The latitudes and longitudes of the cells an input file starts the run's lake mask on.

    :func:`initialize_surface` binds lakes where the driver's ``lakemask``
    is 1, and on both input doors that mask is read from the file a
    domain starts from, so the same test is made here, on the same file,
    at plan review:

    * ``LU_INDEX == ISLAKE``, the land-use rule every door's
      ``initialize_landuse`` applies
      (``woof.core.landuse._derive_categories``).  A land-use table with
      no lake category writes ``ISLAKE = -1``, which no cell equals.
    * ``lake_mask_record`` names a record the run takes in place of that
      rule when the file carries it: the wrfinput door hands the lake
      model the file's own ``LAKEMASK`` when ``sf_lake_physics = 1``
      (``woof.ingest.lake_physics.wrfinput_lake_physics_inputs``).
      real.exe has by then folded the lake category into water in
      ``LU_INDEX``, so the two rules name different cells in a wrfinput
      file, and the plan has to apply the one the run will.

    ``latitude`` and ``longitude`` name the file's mass-point coordinate
    variables (``XLAT_M``/``XLONG_M`` in met_em, ``XLAT``/``XLONG`` in
    wrfinput).
    """
    from woof.netcdf_bridge import open_dataset

    def plane(dataset, name):
        values = np.asarray(dataset.variables[name][...], dtype=np.float64)
        return values[0] if values.ndim == 3 else values

    with open_dataset(path) as dataset:
        lat, lon = plane(dataset, latitude), plane(dataset, longitude)
        if lake_mask_record is not None and lake_mask_record in dataset.variables:
            lake = plane(dataset, lake_mask_record) == 1
        else:
            lake = np.rint(plane(dataset, 'LU_INDEX')) == int(dataset.global_attributes['ISLAKE'])
    return lat[lake], lon[lake]


def input_record_cells(path, record):
    """How many cells of an input file's ``record`` are 1, or 0 when the file has no such record."""
    from woof.netcdf_bridge import open_dataset
    with open_dataset(path) as dataset:
        if record not in dataset.variables:
            return 0
        return int((np.asarray(dataset.variables[record][...]) == 1).sum())


def _unread_lakes_reason(count):
    """Why a domain whose input file holds lake cells binds no lake analysis: one wording, plan and run."""
    return ("the lake binding reads a domain's lake cells from its land use (LU_INDEX) while the lake "
            f"model is off, and LU_INDEX names none of the {count} lake cells the input files' LAKEMASK holds")


def input_plan_sentences(experiment, files, *, latitude, longitude, lake_mask_record=None,
                         restart=None, resume_source='the checkpoint'):
    """The plan-review lake lines of a met_em or wrfinput run: what each domain's lakes will start from.

    ``files`` maps a grid id to the file that domain starts from.  These
    doors hold the lake mask before anything is spent, so with
    ``lake_state_source = "auto"`` each domain's line is the run's own
    decision on its own lake cells (:func:`input_lake_cells`,
    :func:`plan_sentence` with ``lakes``), one line per domain because
    each domain binds its own lakes.  Breakage it prevents: a met_em
    domain whose corner is inside GLSEA's box and whose lakes are outside
    it was told only the condition under which GLSEA is asked, and a
    wrfinput run was told nothing at all, although both doors could read
    the answer from the files in hand.

    ``lake_mask_record`` is read only for a domain that runs the lake
    model (``sf_lake_physics = 1``), as the run reads it.  With the lake
    model off the run starts its mask from the land use, and real.exe
    writes wrfinput's ``LU_INDEX`` with the lake category folded into
    water: such a domain binds nothing although its ``LAKEMASK`` holds
    lake cells.  The plan then says exactly that, in the words the run's
    own line uses (:func:`knowing_input_lake_cells`), instead of "no lake
    cells in this domain", which is false of it.

    A resume, or a choice other than ``auto``, is one line: neither
    depends on the lake cells.  A file whose lake cells cannot be read
    keeps the conditional line and says why; the run stops on that file
    by name when it prepares it.
    """
    if restart is not None:
        where = resume_source if restart is True else f'{resume_source} {restart}'
        return [f'lake state: a resume takes its lakes from {where}; no lake analysis is fetched']
    domains = tuple(experiment.domains)
    root = domains[0].run
    if getattr(root, 'lake_state_source', 'auto') != 'auto':
        return [plan_sentence(root, start_time=experiment.start_time)]
    lines = []
    for domain in domains:
        start = getattr(domain, 'start_time', None) or experiment.start_time
        record = (lake_mask_record
                  if int(getattr(domain.run, 'sf_lake_physics', 0) or 0) == 1 else None)
        try:
            lakes = input_lake_cells(files[domain.grid_id], latitude=latitude, longitude=longitude,
                                     lake_mask_record=record)
            unread = (input_record_cells(files[domain.grid_id], lake_mask_record)
                      if record is None and lake_mask_record is not None and np.size(lakes[0]) == 0
                      else 0)
            line = (f'lake state: forcing (lake_state_source = "auto"; {_unread_lakes_reason(unread)}); '
                    "the lakes keep the input files' own state and no lake analysis is requested"
                    if unread else plan_sentence(domain.run, start_time=start, lakes=lakes))
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            # The file is absent, lacks LU_INDEX, ISLAKE or its coordinates,
            # or the NetCDF decoder refused it (its errors are RuntimeError).
            line = ((experiment_plan_sentence(experiment) if domain is domains[0]
                     else plan_sentence(domain.run, start_time=start))
                    + ' (the lake cells could not be read from '
                    f'{Path(str(files.get(domain.grid_id, "its input file"))).name}: '
                    f'{type(error).__name__}: {(str(error).splitlines() or [""])[0][:160]})')
        lines.append(line if len(domains) == 1 else
                     line.replace('lake state: ', f'lake state: d{domain.grid_id:02d}: ', 1))
    return lines


def root_cells(experiment):
    """The root domain's mass-point latitudes and longitudes, from the experiment alone.

    The grid every preparation builds for the root (``_root_grid``, as the
    download budget and the wizard's coverage checks place it), so the
    plan sees the cells the run binds lakes on, not the padded fetch area.
    None for an experiment with no projection (a wrapped legacy config).
    """
    projection = getattr(experiment, 'projection', None)
    if projection is None:
        return None
    run = experiment.domains[0].run
    from woof.source_coverage import _root_grid
    table = {key: getattr(projection, key) for key in
             ('map_proj', 'ref_lat', 'ref_lon', 'truelat1', 'truelat2', 'stand_lon')}
    return _root_grid(table, int(run.nx), int(run.ny), float(run.dx)).latlon_mass()


#: Set while a resumed run builds its physics drivers, to what the lakes
#: will be restored from; :func:`mark_restored` clears it once they are.
_RESUME_SOURCE = contextvars.ContextVar('gpuwm_lake_resume_source', default=None)


def skips_analysis_on_resume(runner):
    """Wrap a forecast runner so a resume fetches no lake analysis.

    Both prepared runners build their physics drivers (where
    :func:`initialize_surface` binds the lake analysis) and only then
    restore a checkpoint, or a WRF warm start, over them, lake fields
    included.  On a resume the bind was a download whose every value was
    overwritten: a wasted fetch at best, and at worst a wait or a WARNING
    about an analysis day the run never used (item 5 follow-up).  The
    runner's own keywords say whether it resumes (``restart``, or an
    ``initialization`` carrying ``warm_start_clocks``), so the drivers it
    builds before the restore skip the bind; :func:`mark_restored` then
    records the restore and clears the skip, so a nest spawned later in the
    run binds as an unbroken run's would.
    """
    @functools.wraps(runner)
    def wrapped(*args, **kwargs):
        source = None
        if kwargs.get('restart') is not None:
            source = 'the checkpoint'
        elif getattr(kwargs.get('initialization'), 'warm_start_clocks', None):
            source = 'the WRF restart files (wrfrst)'
        if source is None:
            return runner(*args, **kwargs)
        token = _RESUME_SOURCE.set(source)
        try:
            return runner(*args, **kwargs)
        finally:
            _RESUME_SOURCE.reset(token)
    return wrapped


#: Set while an input door builds a domain's physics driver, to how many
#: lake cells that domain's input file holds in its own lake mask.
_INPUT_LAKE_CELLS = contextvars.ContextVar('gpuwm_lake_input_cells', default=None)


def knowing_input_lake_cells(lake_mask, build):
    """Wrap a driver build so the lake bind knows how many lake cells the input file holds.

    The wrfinput door builds its driver from the land use and only then
    restores the file's own fields over it, ``LAKEMASK`` included
    (``initialize_wrfinput_physics``).  With the lake model off the bind
    therefore sees the land use's lake cells, and real.exe writes
    ``LU_INDEX`` with the lake category folded into water, so the bind
    sees none.  Breakage it prevents (the 2026-11-12 OKC wrfinput card
    run): the run line and report.json said "no lake cells in this
    domain" of a domain whose frames carry 197.  With the count in hand
    :func:`initialize_surface` says why nothing is bound.  What is bound,
    and every field, is unchanged.
    """
    count = 0 if lake_mask is None else int((np.asarray(lake_mask) == 1).sum())

    @functools.wraps(build)
    def wrapped(*args, **kwargs):
        token = _INPUT_LAKE_CELLS.set(count)
        try:
            return build(*args, **kwargs)
        finally:
            _INPUT_LAKE_CELLS.reset(token)
    return wrapped


def publish_receipt(receipt):
    """Write the receipt beside the lake cache, record its path, print the one line."""
    receipt_dir = Path(os.environ.get('WOOF_LAKE_RECEIPT_DIR', os.environ.get('WOOF_LAKE_STATE_CACHE', str(Path.home()/'.woof/cache/lake-state'))))
    receipt_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()[:16]
    receipt_path = receipt_dir/f'lake-init-{key}.json'
    receipt['receipt_path'] = str(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(describe(receipt), file=sys.stderr, flush=True)
    return receipt


def mark_restored(drivers, *, checkpoint, source='the checkpoint'):
    """Record that a resumed run's lakes came from what it restored.

    Both runners build the driver, so :func:`initialize_surface` binds the
    lake analysis afresh, and only then restore a checkpoint (or a WRF warm
    start) over it, lake fields included (``io/restart.py`` serializes
    them).  Breakage it prevents (item 5 review): report.json
    ``lake_initialization`` and ``lake_initialization_by_domain`` named that
    overwritten bind -- possibly a newer analysis day than the first segment
    started from, or one where the first segment kept the forcing -- as what
    the lakes started from.  The start-up bind is kept under
    ``start_up_bind`` so nothing it said is lost.  A driver built under
    :func:`skips_analysis_on_resume` made no bind, and its receipt says no
    analysis was fetched.  The skip ends here: the lakes are restored.
    """
    _RESUME_SOURCE.set(None)
    marked = []
    for driver in drivers:
        previous = getattr(driver, 'lake_initialization_receipt', None)
        if previous is None:
            continue
        receipt = dict(schema='gpuwm.lake-init.v1', status='restored_from_checkpoint',
                       source=source, checkpoint=None if checkpoint is None else str(checkpoint),
                       requested=previous.get('requested', previous.get('source')))
        if previous.get('status') == 'restore_pending':
            receipt['analysis_fetch'] = 'skipped'
        else:
            receipt['start_up_bind'] = {k: previous[k] for k in ('status', 'source', 'receipt_path')
                                        if k in previous}
        driver.lake_initialization_receipt = receipt
        marked.append(receipt)
    if marked:
        print(describe(marked[0]), file=sys.stderr, flush=True)
    return marked


def initialize_surface(fields, cfg, *, valid_time, latitude, longitude, cold_start=True):
    """Bind observations before either the LSM or lake cold start."""
    source = getattr(cfg, 'lake_state_source', 'auto')
    if not cold_start:
        # A tile buffer about to be gathered from a prepared store
        # (initialize_physics lsm_cold_start=False): its lake columns come
        # from the store, so nothing is bound and, one per tile, nothing is
        # printed; the receipt says why.
        return None, dict(schema='gpuwm.lake-init.v1', status='store_buffer', source='prepared store',
                          requested=source, reason='lake columns are gathered from the prepared store')
    resume = _RESUME_SOURCE.get()
    if resume is not None:
        # A resume restores the lake fields over whatever is bound here, so
        # nothing is fetched; mark_restored prints the run's lake line.
        return None, dict(schema='gpuwm.lake-init.v1', status='restore_pending', source=resume,
                          requested=source, reason=f'a resumed run restores its lakes from {resume}')
    if source == 'forcing':
        receipt = dict(schema='gpuwm.lake-init.v1', status='retained_forcing', source='forcing',
                       requested=source, reason='lake_state_source = "forcing"')
        print(describe(receipt), file=sys.stderr, flush=True)
        return None, receipt
    if valid_time is None or latitude is None or longitude is None:
        if source != 'auto':
            raise ValueError('explicit lake-state binding requires initialization time and surface coordinates')
        receipt = dict(schema='gpuwm.lake-init.v1', status='missing_time_or_coordinates', source='forcing', requested=source)
        print(describe(receipt), file=sys.stderr, flush=True)
        return None, receipt
    import cupy as cp
    shape = fields['tsk'].shape
    mask = fields['lakemask'] == 1
    if not bool(cp.any(mask).item()):
        held = _INPUT_LAKE_CELLS.get()
        if held:
            # The input file holds lake cells this mask does not (see
            # knowing_input_lake_cells): nothing is bound, and the reason
            # is not that the domain has no lakes.
            receipt = dict(schema='gpuwm.lake-init.v1', status='retained_forcing', source='forcing',
                           requested=source, analysis_fetch='none', input_lake_cells=int(held),
                           reason=_unread_lakes_reason(held) + ', so no lake analysis is requested')
        else:
            receipt = dict(schema='gpuwm.lake-init.v1', status='no_lake_cells', requested=source)
        print(describe(receipt), file=sys.stderr, flush=True)
        return None, receipt
    la, lo = cp.broadcast_arrays(cp.asarray(latitude), cp.asarray(longitude))
    if la.shape != shape:
        raise ValueError('lake initialization coordinates must match the surface grid')
    values, receipt = load_state(source, valid_time, cp.asnumpy(la[mask]), cp.asnumpy(lo[mask]),
        path=getattr(cfg, 'lake_state_path', ''), accept_licence=getattr(cfg, 'lake_state_accept_licence', ''))
    mapped = {}
    ranges = {'water_temperature': (270., 330.), 'bottom_temperature': (270., 330.),
              'mixed_layer_depth': (0., 2000.), 'temperature_shape_factor': (0.001, .999),
              'ice_thickness': (0., 20.), 'ice_fraction': (0., 1.), 'ice_temperature': (170., 273.16)}
    for name, data in values.items():
        target = cp.full(shape, cp.nan, cp.float32)
        donor = cp.asarray(data, cp.float32)
        lower, upper = ranges[name]
        bad = cp.isfinite(donor) & ((donor < lower) | (donor > upper))
        count = int(bad.sum().item())
        if count:
            message = f'{name}: {count} source values outside [{lower}, {upper}]'
            if source != 'auto':
                raise ValueError(message)
            receipt.setdefault('errors', []).append(message)
            receipt['status'] = 'partial_forcing_fallback'
        target[mask] = cp.where(bad, cp.nan, donor)
        mapped[name] = target
    water, fraction, depth = (mapped.get(k) for k in ('water_temperature', 'ice_fraction', 'ice_thickness'))
    if fraction is None and depth is not None:
        fraction = cp.where(cp.isfinite(depth), (depth > 0).astype(cp.float32), cp.nan)
        mapped['ice_fraction'] = fraction
        receipt['fraction_rule'] = 'source lake column with positive ice depth is ice-covered'
    if water is not None:
        valid = mask & cp.isfinite(water) & (water >= 270.) & (water <= 330.)
        fields['sst'][valid] = water[valid]
        fields['tsk'][valid] = water[valid]
        receipt['temperature_bound_cells'] = int(valid.sum().item())
    if fraction is not None:
        valid = mask & cp.isfinite(fraction) & (fraction >= 0) & (fraction <= 1)
        for name in ('xice', 'xicem'):
            fields[name][valid] = fraction[valid]
        icy = valid & (fraction > 0)
        ice_temp = mapped.get('ice_temperature')
        if ice_temp is not None:
            usable = icy & cp.isfinite(ice_temp) & (ice_temp <= 273.16) & (ice_temp >= 170)
            fields['tsk'][usable] = ice_temp[usable]
        fields['tsk'][icy] = cp.minimum(fields['tsk'][icy], cp.float32(273.15))
        receipt['ice_fraction_bound_cells'] = int(valid.sum().item())
        receipt['ice_thickness_unobserved_cells'] = int((icy & (~cp.isfinite(depth) if depth is not None else True)).sum().item())
    return mapped or None, publish_receipt(receipt)


def decode_buoy_temperature(text: bytes, valid_time, *, tolerance_seconds=3600):
    """Rust decodes WTMP, sentinel values and nearest report time, never ATMP."""
    from woof.ingest.cpu_backend import masked_fields_cpu_backend
    entry = masked_fields_cpu_backend()._library.gpuwm_lake_ndbc_temperature_f64
    entry.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64,
                      ctypes.c_int64, ctypes.c_void_p]
    entry.restype = ctypes.c_int32
    raw = ctypes.create_string_buffer(text)
    result = np.empty(2, np.float64)
    rc = entry(ctypes.cast(raw, ctypes.c_void_p), len(text), int(utc(valid_time).timestamp()),
               tolerance_seconds, ctypes.c_void_p(result.ctypes.data))
    if rc:
        raise ValueError(f'NDBC WTMP unavailable or malformed in requested window (native status {rc})')
    return dict(valid_time=datetime.fromtimestamp(result[0], timezone.utc).isoformat(),
                water_temperature_k=float(result[1]))


def nearest_lake_value(latitude, longitude, values, mask, target_latitude, target_longitude, *, max_distance_km):
    """Match a buoy only to a nearby wet model cell, in Rust."""
    from woof.ingest.cpu_backend import masked_fields_cpu_backend
    arrays = [np.ascontiguousarray(v, np.float64) for v in (latitude, longitude, values, mask)]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('lake verification coordinates, mask and values must have identical shapes')
    entry = masked_fields_cpu_backend()._library.gpuwm_lake_state_nearest_f64
    ptr = ctypes.c_void_p
    entry.argtypes = [ptr]*4 + [ctypes.c_size_t] + [ctypes.c_double]*3 + [ptr]
    entry.restype = ctypes.c_int32
    out = np.empty(3, np.float64)
    rc = entry(*(ptr(a.ctypes.data) for a in arrays), arrays[0].size, target_latitude,
               target_longitude, max_distance_km, ptr(out.ctypes.data))
    if rc:
        raise ValueError(f'no finite lake value within {max_distance_km:g} km (native status {rc})')
    return dict(flat_index=int(out[0]), distance_km=float(out[1]), value=float(out[2]))


def diagnostic_values(values, mask, *, scale=1., offset=0.):
    """Rust wet-cell masking and affine display conversion, without a second decoder."""
    from woof.ingest.cpu_backend import masked_fields_cpu_backend, host_step_workers
    backend = masked_fields_cpu_backend()
    values, mask = [np.ascontiguousarray(v, np.float64) for v in (values, mask)]
    if values.shape != mask.shape:
        raise ValueError('lake diagnostic values and mask must have identical shapes')
    output = np.empty_like(values)
    entry = backend._library.gpuwm_lake_state_diagnostic_f64
    ptr = ctypes.c_void_p
    entry.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ptr, ctypes.c_size_t]
    entry.restype = ctypes.c_int32
    rc = entry(ptr(values.ctypes.data), ptr(mask.ctypes.data), values.size, scale, offset,
               ptr(output.ctypes.data), host_step_workers(backend))
    if rc:
        raise ValueError(f'lake diagnostic conversion failed (native status {rc})')
    return output


def verification_scores(prediction, reference):
    """Rust parallel paired finite-value reductions; missing pairs are excluded."""
    from woof.ingest.cpu_backend import masked_fields_cpu_backend, host_step_workers
    backend = masked_fields_cpu_backend()
    arrays = [np.ascontiguousarray(v, np.float64) for v in (prediction, reference)]
    if arrays[0].shape != arrays[1].shape:
        raise ValueError('lake verification arrays must have identical shapes')
    output = np.empty(5, np.float64)
    entry = backend._library.gpuwm_lake_state_scores_f64
    ptr = ctypes.c_void_p
    entry.argtypes = [ptr, ptr, ctypes.c_size_t, ptr, ctypes.c_size_t]
    entry.restype = ctypes.c_int32
    rc = entry(*(ptr(a.ctypes.data) for a in arrays), arrays[0].size, ptr(output.ctypes.data), host_step_workers(backend))
    if rc:
        raise ValueError(f'lake verification failed (native status {rc})')
    scores = dict(zip(('count', 'bias', 'mae', 'rmse', 'max_absolute_error'), map(float, output)))
    if scores['count'] == 0:
        scores.update({name: None for name in ('bias', 'mae', 'rmse', 'max_absolute_error')})
    return scores


def write_diagnostic_cf(path, *, temperature, ice_fraction, lake_mask, latitude, longitude, valid_time, source):
    """Rust-written CF product for the production renderer, wet cells only.

    Fahrenheit uses the renderer's production temperature ladder. This file
    is a diagnostic product, never an initialization input or a restart.
    """
    from woof.io.classic_product import ClassicProduct
    shape = np.shape(temperature)
    if len(shape) != 2 or any(np.shape(v) != shape for v in (ice_fraction, lake_mask, latitude, longitude)):
        raise ValueError('lake diagnostic fields must share a two-dimensional surface grid')
    temperature_f = diagnostic_values(temperature, lake_mask, scale=1.8, offset=-459.67)
    fraction = diagnostic_values(ice_fraction, lake_mask)
    with ClassicProduct(path) as ds:
        for name, size in (('time', 1), ('y', shape[0]), ('x', shape[1])):
            ds.createDimension(name, size)
        ds.setncatts({'Conventions': 'CF-1.8', 'source': source})
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.units = 'seconds since 1970-01-01 00:00:00'
        time_var.standard_name = 'time'
        time_var[:] = np.asarray([utc(valid_time).timestamp()], np.float64)
        for name, values, units, standard in (
                ('latitude', latitude, 'degrees_north', 'latitude'),
                ('longitude', longitude, 'degrees_east', 'longitude'),
                ('lake_temperature_f', temperature_f, 'degF', ''),
                ('lake_ice_fraction', fraction, '1', 'lake_ice_area_fraction')):
            var = ds.createVariable(name, 'f8', ('y', 'x'))
            var.units = units
            if standard:
                var.standard_name = standard
            var[:] = values
    return Path(path)
