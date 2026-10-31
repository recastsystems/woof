"""RRFS fork stream contracts and native deck orchestration.

Pinned at 1d6e370e from MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:498-1332,1336-4676 and its
included registries; no runtime dependency on an upstream checkout.
Packages are OR within a list and AND between allocation/member gates:
MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:4709-4791.
"""
from __future__ import annotations

import json
import calendar
from fractions import Fraction
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Iterable
import re
import xml.etree.ElementTree as ET

_REGISTRY = json.loads(Path(__file__).with_name('streams_rrfs_registry.json').read_text())


SOURCE_SHA256 = _REGISTRY['source_sha256']
STREAM_NAMES = tuple(_REGISTRY['streams'])


def package_active(packages: str, active: Iterable[str]) -> bool:
    """RRFS MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:4709-4791: OR packages."""
    return not packages or bool(set(packages.split(';')) & set(active))


def field_definition(name: str, *, physics: bool = True, registry=None) -> dict:
    """RRFS MPAS-Model-rrfs/src/tools/registry/gen_inc.c:40-58; MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:1336-4676."""
    registry = _REGISTRY if registry is None else registry
    if not physics and name not in registry['dry_fields']:
        raise ValueError(f'{name}: field absent in a DO_PHYSICS=off build')
    try:
        return registry['fields'][name]
    except KeyError:
        raise ValueError(f'{name}: unknown fork Registry field') from None


@dataclass(frozen=True)
class Stream:
    """A runtime stream; members remain package-qualified until activation."""
    name: str
    options: Mapping[str, str]
    members: tuple[tuple[str, str, str], ...]
    immutable: bool = False
    registry: Mapping | None = field(default=None, repr=False, compare=False)
    timelevel_options: Mapping = field(default_factory=dict, repr=False)

    def timelevel_query(self, seconds: int) -> dict:
        return timelevel_query(self.options["output_timelevels"], seconds, **self.timelevel_options)

    def field_definition(self, name: str, *, physics: bool = True) -> dict:
        """Keep field attributes bound to the stream's own source pin."""
        return field_definition(name, physics=physics, registry=self.registry)

    def fields(self, active_packages: Iterable[str] = (), *, physics: bool = True,
               streams: Mapping[str, Stream] | None = None,
               _visiting: tuple[str, ...] = ()) -> tuple[str, ...]:
        """RRFS MPAS-Model-rrfs/src/framework/xml_stream_parser.c:1818-1915, expand members."""
        if self.name in _visiting:
            raise ValueError(f'cyclic stream membership: {_visiting + (self.name,)}')
        active = frozenset(active_packages)
        if not package_active(self.options.get('packages', ''), active):
            return ()
        data = _REGISTRY if self.registry is None else self.registry
        registry = data['streams' if physics else 'dry_streams']
        groups = data['groups' if physics else 'dry_groups']
        result: dict[str, None] = {}
        for kind, name, packages in self.members:
            if not package_active(packages, active):
                continue
            if kind == 'stream':
                # Native immutable membership comes from the compiled Registry.
                if name in registry and registry[name]['options'].get('immutable') == 'true':
                    sub = registry_stream(name, physics=physics, registry=self.registry)
                elif streams is not None and name in streams:
                    sub = streams[name]
                elif name in registry:
                    sub = registry_stream(name, physics=physics, registry=self.registry)
                else:
                    raise ValueError(f'{self.name}: unknown included stream {name}')
                names = sub.fields(active, physics=physics, streams=streams,
                                   _visiting=(*_visiting, self.name))
            elif kind in ('var_array', 'var_struct'):
                if name not in groups:
                    raise ValueError(f'{self.name}: unknown {kind} {name}')
                names = groups[name]
            elif kind == 'var':
                # Native stream-list files name var arrays as ordinary field
                # tokens. add_field registers their constituent fields at I/O.
                names = groups[name] if name in data['arrays' if physics else 'dry_arrays'] else (name,)
            else:
                raise ValueError(f'{self.name}: unknown stream member {kind}')
            for variable in names:
                definition = self.field_definition(variable, physics=physics)
                if all(package_active(gate, active) for gate in definition['gates']):
                    result[variable] = None
        return tuple(result)

    def interval(self, direction: str) -> str:
        """MPAS-Model-rrfs/src/framework/xml_stream_parser.c:1100-1915."""
        return self.options.get(direction + '_interval', 'none')

    def filename(self, valid_time: datetime, initial_time: datetime) -> str:
        """RRFS MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:4242-4317, time buckets."""
        from .output import format_mpas_filename
        raw = self.options.get('filename_interval')
        if raw is None:
            choices = [self.interval(d) for d in ('input', 'output') if d in self.options['type']]
            raw = next((i for i in choices if i not in _SPECIAL), 'none')
        if raw == 'output':
            return format_mpas_filename(self.options['filename_template'], valid_time)
        if raw in ('input_interval', 'output_interval'):
            raw = self.options.get(raw, 'none')
        reference = self.options.get('reference_time', 'initial_time')
        origin = initial_time if reference == 'initial_time' else _datetime(reference)
        if raw == 'none':
            # mpas_stream_manager.F:4206-4208 uses the current time.
            stamp = valid_time
        elif raw not in _SPECIAL:
            stamp = _clock_datetime(_filename_clock(origin, valid_time, _interval(raw)))
        else:
            raise ValueError(f'{self.name}: filename_interval={raw} cannot define a time bucket')
        return format_mpas_filename(self.options['filename_template'], stamp)


def registry_stream(name: str, *, physics: bool = True, registry=None) -> Stream:
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:498-1332, compiled defaults."""
    data = _REGISTRY if registry is None else registry
    raw = data['streams' if physics else 'dry_streams'][name]
    return Stream(name, dict(raw['options']), tuple(tuple(m) for m in raw['members']),
                  raw['options'].get('immutable') == 'true', registry)


_SPECIAL = {'none', 'initial_only', 'final_only'}


def _datetime(text: str) -> datetime:
    """MPAS-Model-rrfs/src/framework/xml_stream_parser.c:1100-1915."""
    return datetime.fromisoformat(text.replace('_', 'T', 1))


@dataclass(frozen=True)
class _Interval:
    """RRFS MPAS-Model-rrfs/src/external/esmf_time_f90/ESMF_TimeIntervalMod.F90:59-81."""
    months: int
    seconds: Fraction


def _interval(text: str) -> _Interval:
    """RRFS MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:1347-1591, native syntax.

    Native truncates the fractional substring to eight digits before reading it.
    Calendar components stay separate from absolute seconds.
    """
    match = re.fullmatch(r'(?:(\d+(?:-\d+){0,2})_)?(\d+(?::\d+){0,2})(?:\.(\d+))?', text)
    if match is None:
        raise ValueError(f'invalid MPAS interval {text!r}')
    date, clock, decimal = match.groups()
    dates = [int(x) for x in (date or '0').split('-')]
    dates = [0] * (3 - len(dates)) + dates
    times = [int(x) for x in clock.split(':')]
    times = [0] * (3 - len(times)) + times
    decimal = (decimal or '')[:8]
    fraction = Fraction(int(decimal), 10**len(decimal)) if decimal else Fraction(0)
    return _Interval(dates[0]*12 + dates[1],
                     Fraction(dates[2]*86400 + times[0]*3600 + times[1]*60 + times[2]) + fraction)


def interval_seconds(text: str) -> float | None:
    """RRFS MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:1594-1764, absolute intervals.

    Calendar intervals have no origin-independent second count; scheduling uses
    their native year/month components instead.
    """
    if text in _SPECIAL:
        return None
    interval = _interval(text)
    return None if interval.months else float(interval.seconds)


def _clock(when: datetime) -> Fraction:
    """RRFS MPAS-Model-rrfs/src/external/esmf_time_f90/ESMF_BaseTimeMod.F90:45-73, exact clock."""
    return Fraction((when.toordinal()-1)*86400 + when.hour*3600 + when.minute*60 + when.second) + Fraction(when.microsecond, 1000000)


def _clock_datetime(clock: Fraction) -> datetime:
    """RRFS MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:1301-1344, timestamp components."""
    whole = clock.numerator // clock.denominator
    days, seconds = divmod(whole, 86400)
    return datetime.fromordinal(days+1) + timedelta(seconds=seconds,
        microseconds=int((clock-whole)*1000000))


def _advance(clock: Fraction, interval: _Interval, count: int = 1) -> Fraction:
    """RRFS MPAS-Model-rrfs/src/external/esmf_time_f90/ESMF_TimeMod.F90:882-1020.

    A native calendar shift clamps the day and reconstructs integer seconds,
    dropping the starting fractional seconds. Absolute intervals preserve them.
    """
    months = interval.months*count
    if months:
        stamp = _clock_datetime(clock)
        year, month = divmod(stamp.year*12 + stamp.month-1 + months, 12)
        month += 1
        stamp = stamp.replace(year=year, month=month,
            day=min(stamp.day, calendar.monthrange(year, month)[1]), microsecond=0)
        clock = _clock(stamp)
    return clock + interval.seconds*count


def _divide(origin: Fraction, numerator: Fraction, interval: _Interval) -> tuple[int, Fraction]:
    """RRFS MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:1849-1932, interval division.

    Preserve native's doubling/halving addition order: calendar clamping makes
    repeated additions different from multiplying an interval once.
    """
    if not interval.months and not interval.seconds:
        return 0, Fraction(0)
    if not interval.months:
        n = numerator // interval.seconds
        return n, numerator - n*interval.seconds
    target = origin + numerator
    updated = _advance(origin, interval)
    if target == updated:
        return 1, Fraction(0)
    if target < updated:
        return 0, numerator
    n = 1
    while target > updated:
        n *= 2
        updated = _advance(origin, interval, n)
    n //= 2
    m = n
    updated = _advance(origin, interval, n)
    while m:
        candidate = _advance(updated, interval, m)
        if candidate <= target:
            updated = candidate
            n += m
        m //= 2
    return n, target-updated


def _filename_clock(origin: datetime, when: datetime, interval: _Interval) -> Fraction:
    """RRFS MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:4242-4317, signed buckets."""
    reference, now = _clock(origin), _clock(when)
    _, remainder = _divide(reference, abs(now-reference), interval)
    if not remainder:
        return now
    if now >= reference:
        return now-remainder
    return _advance(now+remainder, interval, -1)


def parse_streams(path: str | Path, *, physics: bool = True, registry=None, timelevel_options=None) -> dict[str, Stream]:
    """RRFS MPAS-Model-rrfs/src/framework/xml_stream_parser.c:520-603,1100-1915.
    MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:6710-6734;
    MPAS-Model-rrfs/src/framework/mpas_stream_list.F:100-134.

    Mutable streams use the deck's membership, not Registry defaults. An
    immutable stream may change its I/O settings but cannot change membership.
    """
    data = _REGISTRY if registry is None else registry
    path = Path(path)
    root = ET.fromstring(stream_xml(path))
    if root.tag != 'streams':
        raise ValueError('native stream deck must have a <streams> root')
    result: dict[str, Stream] = {}
    for el in root:
        if el.tag not in ('stream', 'immutable_stream'):
            raise ValueError(f'unknown stream element {el.tag}')
        options = dict(el.attrib)
        if 'output_timelevels' in options:
            if 'output_interval' in options or 'output' not in options.get('type', ''):
                raise ValueError('output_timelevels requires output and excludes output_interval')
            timelevel_query(options['output_timelevels'], 0, **(timelevel_options or {}))
        for required in ('name', 'type', 'filename_template'):
            if required not in options:
                raise ValueError(f'stream lacks required {required}')
        name = options['name']
        if name in result:
            raise ValueError(f'duplicate stream {name}')
        immutable = el.tag == 'immutable_stream'
        if immutable:
            base = registry_stream(name, physics=physics, registry=registry)
            if not base.immutable:
                raise ValueError(f'{name}: not an immutable Registry stream')
            if len(el):
                raise ValueError(f'{name}: immutable stream membership cannot be modified')
            members = base.members
            # Registry package restriction persists; deck package is additional.
            if 'packages' not in options and 'packages' in base.options:
                options['packages'] = base.options['packages']
        else:
            if name in data['streams'] and data['streams'][name]['options'].get('immutable') == 'true':
                raise ValueError(f'{name}: must be declared as immutable_stream')
            members_list = []
            for child in el:
                if set(child.attrib) - {'name', 'packages'} or 'name' not in child.attrib or len(child):
                    raise ValueError(f'{name}: invalid member {child.tag}')
                packages = child.attrib.get('packages', '')
                if child.tag == 'file':
                    # fscanf("%s") treats all whitespace-delimited tokens as fields.
                    for variable in (path.parent / child.attrib['name']).read_text().split():
                        members_list.append(('var', variable, packages))
                elif child.tag in ('var', 'var_array', 'var_struct', 'stream'):
                    members_list.append((child.tag, child.attrib['name'], packages))
                else:
                    raise ValueError(f'{name}: unknown member {child.tag}')
            members = tuple(members_list)
        # The native parser tests substrings and falls back for unknown values.
        raw_direction = options['type']
        direction = {d for d in ('input','output') if d in raw_direction} or {'none'}
        options['type'] = ';'.join(d for d in ('input','output','none') if d in direction)
        for d in ('input','output'):
            if d in direction and d + '_interval' not in options and not (d == 'output' and 'output_timelevels' in options):
                raise ValueError(f'{name}: {d} stream requires {d}_interval')
        for knob, choices, default in (
            ('precision', ('single','double'), 'native'),
            ('clobber_mode', ('never_modify','append','truncate','replace_files','overwrite'), 'never_modify'),
            ('io_type', ('pnetcdf,cdf5','pnetcdf','netcdf4','netcdf'), 'pnetcdf')):
            if knob in options:
                raw = options[knob]
                options[knob] = next((value for value in choices if value in raw), default)
        result[name] = Stream(name, options, tuple(members), immutable, registry, timelevel_options or {})
    # Resolve interval aliases; reject cycles rather than changing schedules.
    def resolve(name, knob, visiting=()):
        """MPAS-Model-rrfs/src/framework/xml_stream_parser.c:1100-1915."""
        if (name, knob) in visiting:
            raise ValueError('cyclic stream interval reference')
        raw = result[name].options.get(knob, 'none')
        if raw.startswith('stream:'):
            pieces = raw.split(':')
            if len(pieces)!=3 or pieces[2] not in ('input_interval','output_interval'):
                raise ValueError(f'{name}: invalid interval reference {raw}')
            ref, ref_knob = pieces[1:]
            if ref not in result or ref_knob not in result[ref].options:
                raise ValueError(f'{name}: missing interval reference stream {ref}')
            if result[ref].options[ref_knob].startswith('stream:'):
                raise ValueError(f'{name}: nested native interval references are unexpandable')
            return resolve(ref, ref_knob, (*visiting, (name,knob)))
        if raw not in _SPECIAL:
            _interval(raw)
        return raw
    for name, stream in tuple(result.items()):
        opts = dict(stream.options)
        for knob in ('input_interval','output_interval','record_interval'):
            if knob in opts:
                opts[knob] = resolve(name,knob)
        raw = opts.get('filename_interval')
        def recurring(knob):
            """MPAS-Model-rrfs/src/framework/xml_stream_parser.c:1100-1915."""
            value = opts.get(knob)
            return value if value and not any(x in value for x in ('initial_only','final_only','none')) else None
        if raw is None:
            direction = opts['type']
            raw = recurring('input_interval') if 'input' in direction else None
            if raw is None and 'output' in direction:
                raw = recurring('output_interval')
                if raw is None and 'output_timelevels' in opts: raw = 'output'
            raw = raw or 'none'
        elif 'input_interval' in raw:
            raw = recurring('input_interval') or 'none'
        elif 'output_interval' in raw:
            raw = recurring('output_interval') or 'none'
        if raw not in _SPECIAL | {'output'}: _interval(raw)
        opts['filename_interval'] = raw
        result[name] = replace(stream, options=opts)
    # Validate named fields even when no physics package currently activates them.
    for stream in result.values():
        stream.fields(_all_packages(registry), physics=physics, streams=result)
    # Immutable streams already occupy the Registry's manager list; the C
    # parser updates them before appending mutable streams. This order also
    # determines which native random file ID belongs to each output file.
    compiled = data['streams' if physics else 'dry_streams']
    immutable = [name for name in compiled if name in result and result[name].immutable]
    mutable = [name for name, stream in result.items() if not stream.immutable]
    return {name: result[name] for name in (*immutable, *mutable)}


def _all_packages(registry=None) -> frozenset[str]:
    """MPAS-Model-rrfs/src/framework/xml_stream_parser.c:1100-1915."""
    data = _REGISTRY if registry is None else registry
    return frozenset(p for f in data['fields'].values() for gate in f['gates']
                     for p in gate.split(';')) | frozenset(
                     p for s in data['streams'].values()
                     for p in s['options'].get('packages','').split(';')) | {'no_invariant_stream'}


@dataclass
class StreamAlarms:
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:662-678, startup resets."""
    streams: Mapping[str, Stream]
    start_time: datetime
    initial_time: datetime
    final_time: datetime
    do_restart: bool = False
    do_DAcycling: bool = False
    active_packages: frozenset[str] = frozenset()
    _variable_next: dict[str, Fraction | None] = field(default_factory=dict)
    _last: dict[tuple[str,str], datetime] = field(default_factory=dict)
    _previous: dict[tuple[str,str], Fraction] = field(default_factory=dict)

    def __post_init__(self):
        """RRFS MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:993-1062, reference alignment."""
        start = _clock(self.start_time)
        for name, stream in self.streams.items():
            for direction in ('input', 'output'):
                if direction == 'output' and 'output_timelevels' in stream.options:
                    q = stream.timelevel_query(0)
                    delay = 0 if q['matched'] and q['interval_seconds'] > 0 else q['future_seconds']
                    if delay is None:
                        raise ValueError(f'{name}: no valid initial variable output alarm')
                    self._variable_next[name] = start + Fraction(str(delay))
                    continue
                raw = stream.interval(direction)
                if direction not in stream.options['type'].split(';') or raw in _SPECIAL:
                    continue
                interval = _interval(raw)
                reference = stream.options.get('reference_time', 'initial_time')
                origin = _clock(self.initial_time if reference == 'initial_time' else _datetime(reference))
                _, rem = _divide(origin, abs(start-origin), interval)
                remainder = _Interval(0, rem)
                if start <= origin and rem:
                    remainder = _Interval(interval.months, interval.seconds-rem)
                self._previous[name,direction] = _advance(_advance(start, interval, -1), remainder, -1)
        # Input streams already consumed by initialization must not ring again.
        inputs = ('invariant', 'input', 'restart')
        if self.do_DAcycling and not self.do_restart: inputs += ('da_state',)
        for name in inputs:
            if name in self.streams: self.reset(name, self.start_time, 'input')
        # Native resets these initial alarms before entering the step loop.
        suppressed = ('restart','output','diagnostics') if self.do_restart else ('restart',)
        if self.do_DAcycling and not self.do_restart:
            suppressed += ('da_state',)
        for name in suppressed:
            if name in self.streams:
                self.reset(name, self.start_time)

    def due(self, name: str, when: datetime, direction: str = 'output') -> bool:
        """RRFS MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:876-922, ringing envelope."""
        if direction not in ('input','output'):
            raise ValueError('direction must be input or output')
        stream = self.streams[name]
        if direction not in stream.options['type'].split(';') or not package_active(stream.options.get('packages',''),self.active_packages):
            return False
        if when < self.start_time or when > self.final_time:
            return False
        if direction == 'output' and when == self.start_time:
            if name == 'restart' or (self.do_restart and name in ('output','diagnostics')) or (self.do_DAcycling and not self.do_restart and name == 'da_state'):
                return False
        if direction == 'output' and name in self._variable_next:
            next_time = self._variable_next[name]
            if next_time is None or next_time > _clock(when):
                return False
            query = stream.timelevel_query(int((when-self.start_time).total_seconds()))
            if query['matched']:
                return True
            delay = query['future_seconds']
            self._variable_next[name] = None if delay is None else _clock(self.start_time)+Fraction(str(delay))
            return False
        interval = stream.interval(direction)
        if interval == 'none': return False
        if interval in ('initial_only', 'final_only'):
            threshold = self.start_time if interval == 'initial_only' else self.final_time
            return (name,direction) not in self._last and when >= threshold
        # Keep exact rational fractions and native calendar addition/clamping.
        return _advance(self._previous[name,direction], _interval(interval)) <= _clock(when)

    def reset(self, name: str, when: datetime, direction: str = 'output') -> None:
        """RRFS MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:925-985, recurring reset."""
        key = name,direction
        interval_text = self.streams[name].interval(direction)
        if interval_text in ('initial_only', 'final_only'):
            threshold = self.start_time if interval_text == 'initial_only' else self.final_time
            if when < threshold: return
        if direction == 'output' and name in self._variable_next:
            next_time = self._variable_next[name]
            if next_time is None or next_time > _clock(when):
                return
            query = self.streams[name].timelevel_query(int((when-self.start_time).total_seconds()))
            if query['matched'] and query['interval_seconds'] > 0:
                absolute = query['next_seconds']
                self._variable_next[name] = (_clock(self.start_time)+Fraction(str(absolute)) if absolute is not None
                    else _clock(when)+Fraction(str(query['interval_seconds'])))
            else:
                future = query['future_seconds']
                self._variable_next[name] = None if future is None else _clock(self.start_time)+Fraction(str(future))
            self._last[key] = when
            return
        if key in self._previous:
            interval = _interval(self.streams[name].interval(direction))
            previous = self._previous[key]
            now = _clock(when)
            if _advance(previous, interval) > now:
                return
            _, remainder = _divide(previous, now-previous, interval)
            self._previous[key] = now-remainder
        self._last[name,direction] = when


def timelevel_query(specification: str, seconds: int, **options) -> dict:
    """RRFS MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:6059-6606.

    RKIND=real32 parsing and arithmetic in Python (mpasphys.rrfs.stream_xml_rrfs); the result
    carries already rounded MPAS dt seconds and the binary32 words.
    """
    from .mpasphys.rrfs.stream_xml_rrfs import timelevel_query as query
    return query(specification, seconds, **options)


def stream_xml(path: str | Path) -> str:
    """MPAS-Model-rrfs/src/external/ezxml/ezxml.c:83-96,490-550.

    Keeps native's first duplicate attribute before strict XML parsing
    (mpasphys.rrfs.stream_xml_rrfs.first_value_xml). The original on-disk
    workflow deck and field lists remain untouched.
    """
    from .mpasphys.rrfs.stream_xml_rrfs import first_value_xml
    return first_value_xml(Path(path).read_text())


class RegistryContract:
    """One immutable source pin's metadata with the shared RRFS stream logic.

    A contract belongs to a writer, so old and new pin forecasts may coexist
    without changing module globals or each other's stream definitions.
    """
    StreamAlarms = StreamAlarms
    stream_xml = staticmethod(stream_xml)
    _clock = staticmethod(_clock)
    _clock_datetime = staticmethod(_clock_datetime)
    _advance = staticmethod(_advance)
    _interval = staticmethod(_interval)

    def __init__(self, path, *, timelevel_options=None):
        from functools import partial
        self.data = json.loads(Path(path).read_text())
        self.SOURCE_SHA256 = self.data['source_sha256']
        self.STREAM_NAMES = tuple(self.data['streams'])
        options = dict(timelevel_options or {})
        self.timelevel_query = partial(timelevel_query, **options)
        self.Stream = partial(Stream, registry=self.data, timelevel_options=options)
        self.field_definition = partial(field_definition, registry=self.data)
        self.registry_stream = partial(registry_stream, registry=self.data)
        self.parse_streams = partial(parse_streams, registry=self.data, timelevel_options=options)
