"""Table-driven byte acquisition from a Copernicus-compatible data store."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Mapping
import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import shlex
import shutil

from woof import chem_table
from woof.chem_table import SourceRow, _freeze as freeze, _thaw as thaw
from woof import cds_credentials
from woof.source_credentials import (SourceCredential, CredentialLocation,
    credential_present, credential_facts, credential_location_display)
from woof.filesystem_paths import publish_new, replace_file_with_retry
from woof.fetch import Area, parse_area


class DataStoreCredentialMissing(ValueError): pass
class DataStoreCredentialRejected(ValueError): pass
class DataStorePolicyNotAccepted(ValueError): pass
class DataStoreLicenceNotAccepted(ValueError): pass
class DataStoreHorizonExceeded(ValueError): pass
class DataStoreFilesMissing(ValueError): pass


def canonical(value):
    return json.dumps(thaw(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class AcquisitionRequest:
    dataset: str
    cycle: str
    group: str
    request: Mapping
    sha256: str
    acquisition: Mapping
    sources: tuple[SourceRow, ...]
    variables_by_source: Mapping
    source_hashes: Mapping


@dataclass(frozen=True)
class AcquisitionPlan:
    requests: tuple[AcquisitionRequest, ...]
    area: Area
    start: str
    end: str
    now: str
    selections: Mapping

    def fetch_command(self, *, cache_root=None):
        args = ['python', '-m', 'woof.data_store_fetch', 'fetch']
        for name in sorted(self.selections):
            args += ['--source', name]
        bounds = (self.area.lat_south, self.area.lon_west, self.area.lat_north, self.area.lon_east)
        area_text = ','.join(format(v, '.12g') for v in bounds)
        args += (['--area='+area_text] if area_text.startswith('-') else ['--area', area_text])
        args += ['--start', self.start, '--end', self.end, '--now', self.now,
                 '--variables-json', canonical(self.selections)]
        licences = {r.acquisition.get('research_only_licence') for r in self.requests} - {None}
        if len(licences) == 1:
            args += ['--licence', next(iter(licences))]
        if cache_root is not None:
            args += ['--cache-root', str(cache_root)]
        # Single quotes also preserve JSON arguments in PowerShell.
        return shlex.join(args)

    def as_dict(self, *, cache_root=None):
        return {'requests': [{'dataset': r.dataset, 'cycle': r.cycle, 'group': r.group,
            'request': thaw(r.request), 'sha256': r.sha256,
            'cache_path': str(_cache_root(cache_root) / (r.sha256+'.grib2')),
            'variables_by_source': thaw(r.variables_by_source),
            'sources': _source_receipts(r)}
            for r in self.requests]}


@dataclass(frozen=True)
class FetchResult:
    files: tuple[Path, ...]
    receipts: tuple[Path, ...]
    reused: tuple[bool, ...]


def _validate_acquisition(data):
    for key in ('name', 'variables', 'grid', 'acquisition'):
        if key not in data:
            raise ValueError(f"Source row needs {key}")
    if not isinstance(data['variables'], Mapping) or not isinstance(data['grid'], Mapping):
        raise ValueError(f"{data['name']}: variables and grid must be objects")
    a = data['acquisition']
    if a.get('kind') == 'published_files':
        raise ValueError(f"{data['name']}: import published files with python -m woof.composition_source --source {data['name']} --files GROUP=FILE...; a queued data-store request cannot describe these files")
    for key in ('api_url', 'site', 'dataset', 'dataset_page', 'cycle_hours',
                'cadence_s', 'horizon_h', 'latency_s', 'margin_cells', 'keys', 'requests'):
        if key not in a:
            raise ValueError(f"{data['name']}: acquisition needs {key}")
    if (not a['cycle_hours'] or any(type(h) is not int or not 0 <= h < 24 for h in a['cycle_hours'])
            or a['cadence_s'] <= 0 or a['cadence_s'] % 3600
            or a['latency_s'] < 0 or a['horizon_h'] < 0 or a['margin_cells'] < 0
            or data['grid']['spacing_deg'] <= 0):
        raise ValueError(f"{data['name']}: invalid acquisition timing or grid")
    for key in ('variable', 'date', 'cycle', 'lead', 'area', 'area_order'):
        if key not in a['keys']:
            raise ValueError(f"{data['name']}: acquisition keys need {key}")
    horizons = a.get('cycle_horizon_h', {})
    if (not isinstance(horizons, Mapping) or any(
            str(hour) not in {str(h) for h in a['cycle_hours']}
            or type(value) is not int or value < 0
            for hour, value in horizons.items())):
        raise ValueError(f"{data['name']}: invalid cycle_horizon_h")
    if set(a['keys']['area_order']) != {'north', 'west', 'south', 'east'} or len(a['keys']['area_order']) != 4:
        raise ValueError(f"{data['name']}: invalid area order")
    for group, spec in a['requests'].items():
        if not isinstance(spec.get('fixed'), Mapping) or 'levels_key' not in spec or 'levels' not in spec:
            raise ValueError(f"{data['name']}: invalid request group {group}")
        if spec['levels_key'] and not spec['levels']:
            raise ValueError(f"{data['name']}: empty levels in {group}")
    for value in data['variables'].values():
        if not value.get('request') or value.get('group') not in a['requests']:
            raise ValueError(f"{data['name']}: invalid variable request group")


def resolve_acquisition(source_row, *, area, start, end, variables=None, now=None):
    """Resolve one row or several; variables may be a list or a per-row mapping."""
    rows = (source_row,) if isinstance(source_row, SourceRow) else tuple(source_row)
    if any(not isinstance(row, SourceRow) for row in rows):
        raise TypeError('Acquisition needs shared chem_table.SourceRow objects')
    catalog = chem_table.catalog()
    start, end = _utc(start), _utc(end)
    now = _utc(now) if now is not None else datetime.now(timezone.utc)
    if end < start:
        raise ValueError('End precedes start')
    area = parse_area(area) if isinstance(area, str) else area
    if not isinstance(area, Area) or not all(math.isfinite(v) for v in area.as_manifest().values()):
        raise ValueError('Area needs finite geographic bounds')
    if not -90 <= area.lat_south <= area.lat_north <= 90 or area.longitude_span_degrees <= 0:
        raise ValueError('Area needs ordered latitudes in [-90, 90] and positive longitude width')
    merged = {}
    selections = {}
    for row in rows:
        d = {key: getattr(row, key) for key in ('name', 'grid', 'variables', 'vertical', 'acquisition', 'credential')}
        _validate_acquisition(d)
        a = row.acquisition
        # Past starts are bounded by start, rather than start minus latency.
        cutoff = min(start, now - timedelta(seconds=a['latency_s']))
        day = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)
        cycles = [day + timedelta(days=offset, hours=h)
                  for offset in (-1, 0) for h in a['cycle_hours']]
        step = a['cadence_s']
        available = sorted((c for c in cycles if c <= cutoff), reverse=True)
        # Keep the newest cycle whenever it covers the request. Otherwise
        # use an already available extended cycle, never a future analysis.
        cycle = next((c for c in available
                      if math.ceil((end-c).total_seconds()/step) * step <= 3600 * a.get(
                          'cycle_horizon_h', {}).get(str(c.hour), a['horizon_h'])),
                     available[0])
        first = math.floor((start-cycle).total_seconds()/step)
        last = math.ceil((end-cycle).total_seconds()/step)
        leads = [i * step // 3600 for i in range(first, last+1)]
        horizon = a.get('cycle_horizon_h', {}).get(str(cycle.hour), a['horizon_h'])
        if leads[-1] > horizon:
            raise DataStoreHorizonExceeded(f"{row.name}: lead {leads[-1]} h exceeds horizon {horizon} h; shorten the interval or choose a newer available cycle")
        spacing = d['grid']['spacing_deg']
        margin = a['margin_cells']
        down = lambda v: round((math.floor(v/spacing + 1e-10)-margin)*spacing, 10)
        up = lambda v: round((math.ceil(v/spacing - 1e-10)+margin)*spacing, 10)
        west = down(area.lon_west)
        east = up(area.lon_west + area.longitude_span_degrees)
        if east-west >= 360:
            west, east = -180, 180
        else:
            west = (west+180) % 360-180
            east = (east+180) % 360-180
            if east == -180:
                east = 180
        bounds = dict(north=min(90, up(area.lat_north)), south=max(-90, down(area.lat_south)),
                      west=round(west, 10), east=round(east, 10))
        selected = variables.get(row.name) if isinstance(variables, Mapping) else variables
        selected = list(d['variables']) if selected is None else list(selected)
        selections[row.name] = list(selected)
        vertical = row.vertical or {}
        for key in ('surface_pressure_variable', 'humidity_variable',
                    'pressure_variable', 'temperature_variable'):
            if vertical.get(key) and vertical[key] not in selected:
                selected.append(vertical[key])
        groups = {}
        for field in selected:
            if field not in d['variables']:
                raise ValueError(f'{row.name}: unknown variable {field}')
            v = d['variables'][field]
            groups.setdefault(v['group'], set()).add(v['request'])
        keys = a['keys']
        for group, names in groups.items():
            spec = a['requests'][group]
            request = thaw(spec['fixed'])
            request.update({keys['variable']: sorted(names), keys['date']: cycle.strftime('%Y-%m-%d/%Y-%m-%d'),
                keys['cycle']: [cycle.strftime('%H:%M')], keys['lead']: [str(h) for h in leads],
                keys['area']: [bounds[k] for k in keys['area_order']]})
            if spec['levels_key']:
                request[spec['levels_key']] = [str(v) for v in spec['levels']]
            identity = (a['api_url'], a['dataset'], cycle.isoformat(), group)
            base = {k: v for k, v in request.items() if k != keys['variable']}
            if identity in merged:
                item = merged[identity]
                if item['base'] != base or item['a'] != a or item['credential'] != d.get('credential'):
                    raise ValueError('Cannot merge incompatible acquisition rows')
                item['names'].update(names)
                item['rows'].append(row)
                item['asked'][row.name] = sorted(names)
            else:
                merged[identity] = dict(base=base, names=set(names), rows=[row], asked={row.name: sorted(names)},
                                        a=a, credential=d.get('credential'))
    requests = []
    for (api_url, dataset, cycle, group), item in sorted(merged.items()):
        request = {**item['base'], item['a']['keys']['variable']: sorted(item['names'])}
        digest = hashlib.sha256(canonical(request).encode()).hexdigest()
        requests.append(AcquisitionRequest(dataset, cycle, group, freeze(request), digest,
            item['a'], tuple(sorted(item['rows'], key=lambda r: r.name)), freeze(item['asked']),
            freeze({r.source_file: catalog.file_hashes[r.source_file] for r in item['rows']})))
    return AcquisitionPlan(tuple(requests), area, start.isoformat(), end.isoformat(), now.isoformat(), freeze(selections))


@dataclass(frozen=True)
class DataStoreCredential(SourceCredential):
    remedy: str = ''


def load_credentials(path=None):
    """Read public credential declarations once, retaining their remedy text."""
    path = Path(path) if path else chem_table.CHEM_DATA_ROOT / 'credentials.v1.json'
    declarations = json.loads(path.read_text(encoding='utf-8'))['credentials']
    return {name: DataStoreCredential(credential_id=name,
        **{k: (CredentialLocation(v) if k == 'location_kind' else v)
           for k, v in row.items() if k in DataStoreCredential.__dataclass_fields__})
        for name, row in declarations.items()}


def _refusal(error, request, key, location):
    text = cds_credentials.redact(error, key)
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    if status == 401 or re.search(r'\b401\b', text):
        return DataStoreCredentialRejected(f'Credential from {location} is present but refused (HTTP 401); replace its key with a current personal access token')
    if status == 403 or re.search(r'\b403\b', text):
        if 'missing policies' in text.lower():
            missing = re.split('missing policies are:', text, flags=re.I)[-1].strip()
            cls = DataStoreLicenceNotAccepted if 'licence' in missing.lower() and 'terms of use' not in missing.lower() else DataStorePolicyNotAccepted
            if cls is DataStoreLicenceNotAccepted:
                return cls(f"{missing}; open {request.acquisition['dataset_page']} with the same account and accept the dataset licence")
            return cls(f"{missing}; log in to {request.acquisition['site']} with the same account and accept them; nothing else changes")
        if 'licen' in text.lower():
            return DataStoreLicenceNotAccepted(f"{text}; open {request.acquisition['dataset_page']} with the same account and accept the dataset licence")
    return ValueError(text)


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _silent(*args, **kwargs): pass


def _http_get(url, headers=None, attempts=4):
    """Anonymous HTTPS GET with a short retry; returns the body bytes."""
    import time
    import urllib.request
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=dict(headers or {}))
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except OSError as error:
            last = error
            time.sleep(1.5 * (attempt + 1))
    raise ValueError(f'{url}: {last}')


def index_selection(index_text, wanted):
    """``[(record line, first byte, last byte or None)]`` for the wanted records.

    ``index_text`` is an NCEP .idx inventory (``n:offset:d=...:VAR:LEVEL:...``);
    ``wanted`` is a set of ``VAR:LEVEL`` strings.  Every wanted record must
    appear exactly once: a missing or repeated one is refused by name, since
    a stack built from the wrong record count would mix levels or times.
    """
    rows = [line.split(':') for line in index_text.strip().splitlines() if line.strip()]
    offsets = [int(row[1]) for row in rows]
    chosen, seen = [], {}
    for i, row in enumerate(rows):
        key = f'{row[3]}:{row[4]}'
        if key in wanted:
            seen[key] = seen.get(key, 0) + 1
            chosen.append((key, offsets[i], offsets[i + 1] - 1 if i + 1 < len(rows) else None))
    missing = sorted(wanted - set(seen))
    repeated = sorted(k for k, n in seen.items() if n > 1)
    if missing or repeated:
        raise ValueError(f'index inventory: missing {missing[:6]}{"..." if len(missing) > 6 else ""}, '
                         f'repeated {repeated[:6]}; refusing a partial or ambiguous record set')
    return chosen


def index_range_download(r, part, *, progress=print, opener=None):
    """Write one request's records, every lead, into ``part`` (a ``noaa_index`` row).

    The acquisition row names the bucket (``api_url``), the file path template
    (``{cycle}``, ``{lead}``), the inventory suffix and, per request group, how
    a level is spelled in the inventory (``index_level``).  Each wanted record
    is fetched by its byte range and the GRIB2 messages are concatenated in
    inventory order, lead by lead.  Nothing is decoded here.
    """
    a = r.acquisition
    request = thaw(r.request)
    keys = a['keys']
    spec = a['requests'][r.group]
    cycle = datetime.strptime(str(request[keys['date']]).split('/')[0] + ' '
                              + request[keys['cycle']][0], '%Y-%m-%d %H:%M')
    names = list(request[keys['variable']])
    levels = (request.get(spec['levels_key']) or []) if spec['levels_key'] else [None]
    wanted = {f"{name}:{spec['index_level'].format(level=level)}" for name in names for level in levels}
    get = opener or _http_get
    total = 0
    with open(part, 'wb') as out:
        for lead in request[keys['lead']]:
            url = a['api_url'].rstrip('/') + '/' + a['path_template'].format(cycle=cycle, lead=int(lead))
            chosen = index_selection(get(url + a.get('index_suffix', '.idx')).decode('utf-8'), wanted)
            def one(item):
                _key, first, last = item
                span = f'bytes={first}-' + ('' if last is None else str(last))
                return get(url, headers={'Range': span})
            with ThreadPoolExecutor(max_workers=8) as pool:
                for body in pool.map(one, chosen):
                    if body[:4] != b'GRIB':
                        raise ValueError(f'{url}: a byte range did not start a GRIB2 message')
                    out.write(body)
                    total += len(body)
            progress(f'data-store: {r.group} f{int(lead):02d} {len(chosen)} records')
    if total == 0:
        raise ValueError('index acquisition wrote no bytes')


#: Environment variable naming the data-store cache.  The fetch stage and
#: model initialization (woof.chem_source_init) read the one root this
#: names, so a cache the fetch filled is the cache initialization searches.
CACHE_ENV = 'WOOF_DATA_STORE_CACHE'


def default_cache_root():
    """The data-store cache: ``$WOOF_DATA_STORE_CACHE`` or ~/.woof/cache/data-store."""
    value = os.environ.get(CACHE_ENV, '').strip()
    return Path(value) if value else Path.home() / '.woof/cache/data-store'


def _cache_root(cache_root):
    return Path(cache_root) if cache_root is not None else default_cache_root()


def _source_receipts(request):
    return [dict(name=row.name, file_sha256=request.source_hashes[row.source_file])
            for row in request.sources]


def _verified(request, root):
    target = root / (request.sha256+'.grib2')
    receipt = root / (request.sha256+'.json')
    try:
        prior = json.loads(receipt.read_text(encoding='utf-8'))
        required = request.acquisition.get('research_only_licence')
        if required and prior.get('accepted_licence') != required:
            return None
        if (prior['request_sha256'] == request.sha256 and prior['request'] == thaw(request.request)
                and prior['dataset'] == request.dataset and target.stat().st_size == prior['bytes'] > 0
                and _hash(target) == prior['file_sha256']):
            return target
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def cached_files(plan, *, cache_root=None):
    """The only model-input reader: verify all receipts, never acquire bytes.

    The cache is addressed by canonical request sha256, with <sha256>.grib2
    and <sha256>.json alongside it. User files need fetch(files=...) to stage
    bytes and their receipts before this reader can return them.
    """
    root = _cache_root(cache_root)
    found = tuple(_verified(r, root) for r in plan.requests)
    missing = [f"{r.group} ({r.acquisition['api_url']}, {r.dataset}, {r.cycle}, {r.sha256})"
               for r, path in zip(plan.requests, found) if path is None]
    if missing:
        raise DataStoreFilesMissing('Missing verified data-store files: ' + '; '.join(missing)
            + '. Fetch with: ' + plan.fetch_command(cache_root=cache_root))
    return found


def fetch(plan, *, cache_root=None, progress=print, client_factory=None, files=None,
          files_note=None, licence=None):
    """Publish fetched or user-supplied opaque bytes into the request-hash cache.

    files is a mapping from request sha256 to a local file, or a sequence in
    plan.requests order. A mapping can supply some requests; absent ones use
    verified cache entries or the client. Supplied files get normal receipts.
    Only cached_files reads inputs for model initialization; it never downloads.
    """
    root = _cache_root(cache_root)
    supplied = {} if files is None else dict(files) if isinstance(files, Mapping) else {
        r.sha256: path for r, path in zip(plan.requests, files, strict=True)}
    if set(supplied) - {r.sha256 for r in plan.requests}:
        raise ValueError('User files name an unknown request sha256')
    root.mkdir(parents=True, exist_ok=True)
    for request in plan.requests:
        required = request.acquisition.get('research_only_licence')
        if required and licence != required and _verified(request, root) is None:
            raise DataStoreLicenceNotAccepted(
                f"{request.dataset} is research-only under {required}; "
                f"explicit opt-in requires --licence {required}. Terms: "
                f"{request.acquisition.get('licence_url', request.acquisition['dataset_page'])}")
    # Public declarations only. No private profile is read until submission.
    credentials = load_credentials() if any(r.sources[0].credential for r in plan.requests
        if r.sha256 not in supplied and _verified(r, root) is None) else {}
    def acquire(r):
        target, receipt = root / (r.sha256+'.grib2'), root / (r.sha256+'.json')
        if r.sha256 not in supplied:
            try:
                cached_files(AcquisitionPlan((r,), plan.area, plan.start, plan.end, plan.now, plan.selections), cache_root=root)
            except DataStoreFilesMissing:
                pass
            else:
                progress(f'data-store: reused {r.group}')
                return target, receipt, True
        credential_id = r.sources[0].credential
        key, location = '', 'no declared credential'
        if credential_id and r.sha256 not in supplied:
            credential = credentials[credential_id]
            location = credential_location_display(credential)
            if not credential_present(credential):
                raise DataStoreCredentialMissing(credential_facts(credential)['absent_message'] + ' ' + credential.remedy)
            profile = cds_credentials._profile(Path.home() / credential.location)
            key = profile.get('key', '')
            if not key:
                raise DataStoreCredentialRejected(f'Credential from {location} has no usable key; write a personal access token in its key line')
        started = datetime.now(timezone.utc).isoformat()
        with tempfile.TemporaryDirectory(prefix='.data-store-', dir=root) as temporary:
            part = Path(temporary) / 'download.grib2'
            if r.sha256 in supplied:
                shutil.copyfile(supplied[r.sha256], part)
            elif r.acquisition.get('kind') == 'noaa_index':
                index_range_download(r, part, progress=progress, opener=client_factory)
            else:
                try:
                    initializing = True
                    factory = client_factory
                    if factory is None:
                        import cdsapi
                        factory = cdsapi.Client
                    def information(message, *args, **kwargs):
                        safe = cds_credentials.redact(message, key).lower()
                        for state in ('queued', 'running', 'successful', 'completed'):
                            if state in safe:
                                progress(f'data-store: {r.group} {state}')
                                break
                    initializing = True
                    client = factory(url=r.acquisition['api_url'], key=key, quiet=True, debug=False,
                        progress=False, info_callback=information, warning_callback=_silent,
                        error_callback=_silent, debug_callback=_silent)
                    initializing = False
                    client.retrieve(r.dataset, thaw(r.request)).download(str(part))
                except Exception as error:
                    refusal = _refusal(error, r, key, location)
                    if initializing and type(refusal) is ValueError:
                        refusal = ValueError(cds_credentials.client_refusal(error,
                            credential_context=location, secrets=(key,)))
                    raise refusal from None
            if not part.is_file() or part.stat().st_size == 0:
                raise ValueError('Data store returned no bytes; nothing was published')
            record = dict(request_sha256=r.sha256, request=thaw(r.request), dataset=r.dataset,
                file_sha256=_hash(part), bytes=part.stat().st_size, started_utc=started,
                completed_utc=datetime.now(timezone.utc).isoformat(),
                accepted_licence=licence, sources=_source_receipts(r))
            # Where the bytes came from, said in the receipt every reader of
            # the cache sees: the data store, or a file a person supplied (a
            # download made elsewhere, or a test fixture) with their note.
            if r.sha256 in supplied:
                record.update(origin='supplied', supplied_from=str(supplied[r.sha256]),
                              supplied_note=str(files_note or ''))
            else:
                record.update(origin='data-store', api_url=r.acquisition['api_url'])
            staged = Path(temporary) / 'receipt.json'
            staged.write_text(canonical(record), encoding='utf-8')
            for src, dst in ((part, target), (staged, receipt)):
                try:
                    publish_new(src, dst)
                except FileExistsError:
                    replace_file_with_retry(src, dst)
            progress(f'data-store: published {r.group}')
        return target, receipt, False
    if not plan.requests:
        return FetchResult((), (), ())
    with ThreadPoolExecutor(max_workers=len(plan.requests)) as pool:
        results = tuple(pool.map(acquire, plan.requests))
    return FetchResult(*(tuple(item[i] for item in results) for i in range(3)))


def is_listed_emission(row) -> bool:
    """A published-listing emission source (RAVE): hourly files named by the
    publisher's monthly index, fetched by :mod:`woof.chem_source_netcdf`,
    not by a data-store request."""
    return (getattr(row, "acquisition", None) is None
            and isinstance(getattr(row, "grid", None), Mapping)
            and isinstance(row.grid.get("listing"), Mapping))


def fetch_listed_emission(row, *, start, end, cache=None, progress=print):
    """Download every posted hour a run from ``start`` to ``end`` can read.

    The fire process's default timing reads the trailing 24 h of posted
    hours ending ``latency_s`` before the start (woof.core.chem_fire
    trailing_window); an observed-hours run reads the hours through the
    end.  Both are covered: every posted hour from 23 h before the newest
    hour available at the start through ``end``.  An hour the publisher
    never posted is a gap, as the fire process treats it; a window with no
    posted hour at all is refused by name here, at the front door, rather
    than at the forecast's first fire step after a whole preparation.
    """
    from datetime import datetime, timedelta, timezone
    from woof import chem_source_netcdf
    from woof.chem_emission_frames import source_cache_dir

    def utc(text):
        value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)

    latency = float((row.time or {}).get("latency_s", 0.0) or 0.0)
    newest = (utc(start) - timedelta(seconds=latency)).replace(minute=0, second=0, microsecond=0)
    first = newest - timedelta(hours=23)
    last = utc(end).replace(minute=0, second=0, microsecond=0)
    cache = Path(cache) if cache is not None else source_cache_dir(row.name)
    months, month = [], first.replace(day=1, hour=0)
    while month <= last:
        months.append(month)
        month = (month + timedelta(days=32)).replace(day=1)
    posted = {}
    for month in months:
        _url, entries = chem_source_netcdf.listing(row, month, cache)
        posted.update(entries)
    wanted = sorted(hour for hour in posted if first <= hour <= last)
    trailing = [hour for hour in wanted if hour <= newest]
    if not trailing:
        # Said, not refused: the forecast's fire process carries the same
        # finding (woof.core.chem_fire, "fire emissions absent") and runs
        # the smoke already in the air with no new fire smoke.
        import sys
        print(f"warning: {row.name}: the publisher lists no hour between "
              f"{first.isoformat()} and {newest.isoformat()} (the trailing 24 h a "
              f"fire run started at {utc(start).isoformat()} reads); its archive "
              f"({row.grid['listing'].get('url')}) does not reach this date, so "
              "this run emits no new fire smoke (HRRR's own smoke and RAP's edge "
              "smoke still run)", file=sys.stderr, flush=True)
    files = []
    for hour in wanted:
        path, _receipt = chem_source_netcdf.fetch_hour(row, hour, cache)
        files.append(path)
    if files:
        progress(f"emission: {row.name} {len(files)} posted hour(s) "
                 f"{wanted[0].isoformat()}..{wanted[-1].isoformat()}")
    return files


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('plan', 'fetch', 'import'))
    parser.add_argument('--source', action='append', required=True)
    for name in ('area', 'start', 'end'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--variable', action='append')
    parser.add_argument('--cache-root')
    parser.add_argument('--now', help='reproduce cycle availability at this UTC time')
    parser.add_argument('--variables-json', help='JSON mapping of source names to selected fields')
    parser.add_argument('--file', action='append', default=[],
                        help='import: GROUP=PATH, a file you hold for one request group of the plan')
    parser.add_argument('--licence', help='explicit research opt-in naming the source licence')
    parser.add_argument('--note', help='import: why these bytes are in the cache (recorded in the receipt)')
    args = parser.parse_args(argv)
    try:
        rows = chem_table.catalog().sources
        listed = [name for name in args.source if is_listed_emission(rows[name])]
        emission_files = []
        if listed and args.command != 'fetch':
            raise ValueError(f"{', '.join(listed)}: a published-listing emission source "
                             "is fetched by hour ('fetch'); it has no data-store plan or import")
        for name in listed:
            emission_files.extend(str(p) for p in fetch_listed_emission(
                rows[name], start=args.start, end=args.end))
        args.source = [name for name in args.source if name not in listed]
        if not args.source:
            print(json.dumps({'files': emission_files, 'reused': []}))
            return 0
        plan = resolve_acquisition([rows[name] for name in args.source], area=args.area,
            start=args.start, end=args.end, now=args.now,
            variables=json.loads(args.variables_json) if args.variables_json else args.variable)
        if args.command == 'plan':
            print(json.dumps(plan.as_dict(cache_root=args.cache_root), sort_keys=True))
        elif args.command == 'import':
            by_group = {}
            for item in args.file:
                group, sep, path = item.partition('=')
                if not sep or not Path(path).is_file():
                    raise ValueError(f'--file {item!r}: expected GROUP=PATH of an existing file')
                by_group[group] = Path(path)
            groups = {r.group: r.sha256 for r in plan.requests}
            unknown = sorted(set(by_group) - set(groups))
            missing = sorted(set(groups) - set(by_group))
            if unknown or missing:
                raise ValueError(f'import needs one file per request group {sorted(groups)}; '
                                 f'unknown {unknown}, missing {missing}')
            result = fetch(plan, cache_root=args.cache_root,
                           files={groups[g]: p for g, p in by_group.items()},
                           files_note=args.note or '', licence=args.licence)
            print(json.dumps({'files': [str(p) for p in result.files], 'reused': result.reused}))
        else:
            result = fetch(plan, cache_root=args.cache_root, licence=args.licence)
            print(json.dumps({'files': emission_files + [str(p) for p in result.files],
                              'reused': result.reused}))
        return 0
    except (ValueError, KeyError, OSError) as error:
        print(f'{type(error).__name__}: {cds_credentials.redact(error)}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
