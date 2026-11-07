"""Source-neutral tropical cyclone tracks from completed fields.

Atmospheric reads, unit conversion and interpolation run in rw_tc_fields.
Earth2Studio runs only in its own interpreter, never in the CLI process; woof.earth2_env finds it
(WOOF_EARTH2_PYTHON, then the managed environment) and the TempestExtremes tools, and builds them
from pins when they are missing.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from woof import earth2_env

TABLE = Path(__file__).with_name('authorities') / 'tc-track-fields.v1.json'
SCHEMA = 'gpuwm.tc-tracks.v1'
MANIFEST = 'gpuwm.tc-tracks-manifest.v1'
COLUMNS = 'member_id,track_id,weight,valid,lead_hours,lat,lon,mslp_pa,max_wind10_ms,flags'.split(',')


def utc(value):
    d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError(f'time needs an explicit UTC offset: {value}')
    return d.astimezone(timezone.utc)


def stamp(d):
    return d.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    for p in files:
        if path.is_dir():
            h.update(str(p.relative_to(path)).encode() + b'\0')
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def binary(name):
    """Resolve native tracker tools through the engine's normal artifact ladder."""
    from woof import bridges, rustwx
    env = 'GPUWM_' + name.upper()
    candidate = (rustwx.find_renderer() if name == 'rw_wrfbatch' else
                 bridges.find_artifact(env, bridges.executable_name(name),
                                       crate_relative='tools/rustwx'))
    if candidate is None:
        path = shutil.which(name)
        candidate = bridges.accept_resolved(Path(path)) if path else None
    if candidate is None and bridges.fetch_on_demand(name):
        # rw_tc_fields is on demand since 2.9.4 (PyPI per-file limit): pinned
        # in this release's bundle, staged by the first run that needs it.
        candidate = bridges.find_artifact(env, bridges.executable_name(name),
                                          crate_relative='tools/rustwx')
    if candidate is None:
        failure = bridges.on_demand_failure(name)
        raise ValueError(f'{name} is required; build tools/rustwx or install engine bridges '
                         f'({env} remains an optional override)'
                         + (f'; {failure}' if failure else ''))
    return str(candidate)


def run(cmd, **kw):
    return subprocess.run([str(x) for x in cmd], check=True, text=True, **kw)


def inventory(path):
    return json.loads(run([binary('rw_tc_fields'), 'inspect', path], capture_output=True).stdout)


def cf_seconds(values, units, absolute):
    unit, sep, origin = units.partition(' since ')
    scale = {'seconds': 1, 'second': 1, 's': 1, 'hours': 3600, 'hour': 3600,
             'days': 86400, 'minutes': 60, 'nanoseconds': 1e-9}.get(unit)
    if scale is None:
        raise ValueError(f'unsupported CF time units: {units}')
    base = 0
    if absolute:
        if not sep:
            raise ValueError(f'absolute time has no CF epoch: {units}')
        text = origin.strip().replace(' UTC', '+00:00')
        d = datetime.fromisoformat(text.replace('Z', '+00:00'))
        base = d.replace(tzinfo=d.tzinfo or timezone.utc).timestamp()
    return [base + float(v) * scale for v in values]


def producer_init(metadata):
    """A producer's init_time attribute. CF and our exporters write UTC; a value with no offset is UTC
    (ml-export of WOOF history writes 2024-09-24T12:00:00), never local time."""
    value = metadata.get('attributes', {}).get('init_time')
    if not value:
        return None
    d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return stamp(d if d.tzinfo else d.replace(tzinfo=timezone.utc))


def expand_file(member, path):
    """Expand explicit initialization and ensemble axes without collapsing identity."""
    metadata = inventory(path)
    inv = metadata['variables']
    t = inv['time']
    times = cf_seconds(t['values'], t['attrs']['units'], True)
    leads_name = next((n for n in ('lead_time', 'prediction_timedelta')
                       if n in inv and inv[n]['dims'] != t['dims']), None)
    leads = cf_seconds(inv[leads_name]['values'], inv[leads_name]['attrs']['units'], False) if leads_name else [0]
    lead_dim = inv[leads_name]['dims'][0] if leads_name else None
    ensemble_arrays = [v for v in inv.values() if 'ensemble' in v['dims']]
    nens = ensemble_arrays[0]['shape'][ensemble_arrays[0]['dims'].index('ensemble')] if ensemble_arrays else 1
    has_ensemble = bool(ensemble_arrays)
    out = []
    for ti, init in enumerate(times if leads_name else times[:1]):
        for ei in range(nens):
            m = dict(member)
            suffix = (f'-init{ti}' if leads_name and len(times) > 1 else '') + (f'-ens{ei:03d}' if has_ensemble else '')
            m['member_id'] = str(member['member_id']) + suffix
            m['init'] = member.get('init') or producer_init(metadata) or stamp(datetime.fromtimestamp(init, timezone.utc))
            m['ancestry'] = member.get('ancestry', {})
            frames = []
            for fi, offset in enumerate(leads if leads_name else times):
                select = {t['dims'][0]: ti if leads_name else fi}
                if lead_dim:
                    select[lead_dim] = fi
                if has_ensemble:
                    select['ensemble'] = ei
                valid = init + offset if leads_name else offset
                frames.append({'path': str(path), 'valid': stamp(datetime.fromtimestamp(valid, timezone.utc)), 'select': select})
            m['frames'] = frames
            if 'weight' in m and suffix:
                m['weight'] /= nens * (len(times) if leads_name else 1)
            m.pop('file', None)
            out.append(m)
    return out


def normalize_manifest(doc, root=Path('.')):
    if doc.get('schema') != MANIFEST:
        raise ValueError(f'expected {MANIFEST}')
    members = []
    for original in doc.get('members', []):
        m = dict(original)
        for key in ('member_id', 'family', 'source'):
            if not m.get(key):
                raise ValueError(f'member needs {key}')
        m['member_id'] = str(m['member_id'])
        if m['family'] not in ('ai', 'arwen', 'hex', 'analysis', 'nwp'):
            raise ValueError(f'unknown family {m["family"]}')
        m.setdefault('ancestry', {})
        if 'file' in m:
            p = (root / m['file']).resolve()
            members.extend(expand_file(m, p))
        else:
            if not m.get('frames') or not m.get('init'):
                raise ValueError('member needs file or frames plus init')
            m['frames'] = [dict(f, path=str((root / f['path']).resolve()), valid=stamp(utc(f['valid']))) for f in m['frames']]
            members.append(m)
    if not members or len({m['member_id'] for m in members}) != len(members):
        raise ValueError('manifest needs nonempty, unique member IDs')
    families = {f: [m for m in members if m['family'] == f] for f in {m['family'] for m in members}}
    for family, roster in families.items():
        present = [m for m in roster if 'weight' in m]
        if present and len(present) != len(roster):
            raise ValueError(f'{family}: supply all weights or none within a family')
        for m in roster:
            m.setdefault('weight', 1 / len(roster))
            if not math.isfinite(m['weight']) or m['weight'] < 0:
                raise ValueError('weights must be finite and nonnegative')
            m['init'] = stamp(utc(m['init']))
            m['frames'].sort(key=lambda f: f['valid'])
            valid = [f['valid'] for f in m['frames']]
            if len(valid) != len(set(valid)):
                raise ValueError(f'{m["member_id"]}: duplicate valid times')
            if any(utc(v) < utc(m['init']) for v in valid):
                raise ValueError('valid time precedes initialization')
    return dict(schema=MANIFEST, members=members,
                weight_policy='Defaults sum to 1 within each family. No combined cross-family probability is implied.')


def history_export_vocabulary():
    """The field vocabulary produced by the declared history export."""
    return json.loads(TABLE.read_text())['history_export']['vocabulary']


def history_export_arguments(domain, threads):
    """Shared WOOF history export vocabulary for standalone and roster tracks."""
    rules = json.loads(TABLE.read_text())['history_export']
    return ['--domains', domain, '--levels', ','.join(map(str, rules['levels_hpa'])),
            '--grid', rules['grid'], '--names', rules['names'],
            '--variables', ','.join(rules['variables']), '--threads', str(threads)]


def earth2_receipt(path):
    """The Earth2 fetch receipt beside a per-lead forecast file, or in a fetch output directory."""
    folder = path if path.is_dir() else path.parent
    for name in ('earth2-run.json', 'fetch-manifest.json'):
        receipt = folder / name
        if receipt.is_file():
            doc = json.loads(receipt.read_text())
            if str(doc.get('schema', '')).startswith('gpuwm-earth2-fetch') and doc.get('runner', {}).get('files'):
                return receipt, doc
    return None, None


def earth2_frames(path, receipt, doc):
    """Earth2 writes one file per member and lead; one file alone is one frame, so a track can never
    have a fix at the cycle (cyclone-setup --storm auto then rejects every track as
    track-missing-start-fix). Build every lead of the named member (or of all members, for a fetch
    directory) from the receipt's runner.files, the way earth2_job builds its track manifest."""
    folder = receipt.parent

    def local(entry):
        p = Path(entry['path'])
        return p if p.is_file() else folder / p.name
    files = [dict(entry, path=local(entry)) for entry in doc['runner']['files']]
    if not path.is_dir():
        wanted = [e['member'] for e in files if e['path'].resolve() == path]
        if not wanted:
            return None
        files = [e for e in files if e['member'] == wanted[0]]
    missing = [str(e['path']) for e in files if not e['path'].is_file()]
    if missing:
        raise ValueError(f'{receipt} lists forecast files that are not present: {missing}')
    source = doc.get('source') or path.name.split('-20')[0]
    ancestry = dict(model=doc.get('model'), parent_receipt=str(receipt))
    cycle = datetime.fromisoformat(str(doc['cycle']).replace('Z', '+00:00'))
    cycle = cycle if cycle.tzinfo else cycle.replace(tzinfo=timezone.utc)  # receipts write naive UTC
    with ThreadPoolExecutor(max_workers=min(len(files), os.cpu_count() or 1)) as pool:
        expanded = list(pool.map(lambda e: expand_file(dict(member_id=f'member-{int(e["member"]):03d}', family='ai',
                                 source=source, init=stamp(cycle), vocabulary='earth2studio', ancestry=ancestry), e['path'].resolve()), files))
    members = {}
    for parts in expanded:
        for m in parts:
            members.setdefault(m['member_id'], dict(m, frames=[]))['frames'].extend(m['frames'])
    return normalize_manifest(dict(schema=MANIFEST, members=list(members.values())))


def hex_input(path, mesh, work, *, initial=None, window="global", workers=4):
    """Native MPAS history -> Rust remapped CF frames -> common exporter.

    Coordinates and state always come from the declared mesh and initialization.
    The converter verifies mesh agreement and records interpolation in each frame.
    """
    path, mesh = Path(path).resolve(), Path(mesh).resolve()
    history = sorted(path.glob("history*.nc")) if path.is_dir() else [path]
    if not history:
        raise ValueError("native hex input has no history*.nc files")
    frames = work/"hex-cf"
    receipt = work/"hex-conversion.json"
    cmd = [binary("rw_mpas_convert"), "--history", *history, "--mesh", mesh,
           "--out-dir", frames, "--window", window, "--remap", "product",
           "--jobs", str(workers), "--json", receipt]
    if initial:
        cmd += ["--init", Path(initial).resolve()]
    run(cmd)
    provenance = dict(format_bridge="rw_mpas_convert -> rw_mlexport -> rw_tc_fields",
        mesh=dict(path=str(mesh), sha256=sha(mesh)),
        history=[dict(path=str(p), sha256=sha(p)) for p in history],
        window=window, remap="product", conversion=json.loads(receipt.read_text()))
    if initial:
        provenance["initial"] = dict(path=str(Path(initial).resolve()), sha256=sha(initial))
    manifest = shortcut(frames, work)
    for member in manifest["members"]:
        member["family"] = "hex"
        member["ancestry"]["native_hex"] = provenance
    return manifest


def shortcut(path, work):
    path = Path(path).resolve()
    receipt, doc = earth2_receipt(path)
    if receipt is not None:
        manifest = earth2_frames(path, receipt, doc)
        if manifest is not None:
            return manifest
    if path.is_dir() and not (path / '.zgroup').exists():
        stores = sorted(path.glob('*.zarr'))
        if not stores:
            export = work / 'ml-export'
            run([sys.executable, '-m', 'woof', 'ml-export', path, '--out', export,
                 *history_export_arguments('d01', os.cpu_count() or 1)])
            stores = sorted(export.glob('*.zarr'))
        if len(stores) != 1:
            raise ValueError('--input run needs one domain; use a manifest to name domains explicitly')
        path = stores[0]
    meta = inventory(path)
    if meta.get('format') == 'grib2':
        return normalize_manifest(dict(schema=MANIFEST, members=[dict(member_id='member-000', family='arwen',
            source='arwen-grib2', init=meta['init'], vocabulary='arwen-grib2', ancestry={},
            frames=[dict(path=str(path), valid=meta['valid'])])]))
    vocab = history_export_vocabulary() if path.is_dir() else 'earth2studio'
    family = 'arwen' if path.is_dir() else 'ai'
    source = meta.get('attributes', {}).get('source') or path.name.split('-20')[0]
    return normalize_manifest({'schema': MANIFEST, 'members': [dict(member_id='member-000', family=family,
                               source=source, file=str(path), vocabulary=vocab, ancestry=meta.get('attributes', {}))]})


def refine_centers(tracks, memberframes, rule):
    """Move each grid-node fix to the sub-cell minimum of rule['field'].

    The Earth2Studio Vitart tracker returns the lattice node holding the pressure
    minimum, so on the 0.25 degree export every fix sat on a node and tracks drew as
    0.25 degree staircases (improve-55). A separable three-point parabola through the
    node and its finite neighbours on the frame's own regional grid places the
    minimum inside the cell; it never moves a fix more than half a cell."""
    import numpy as np
    dirs = {(m['member_id'], f['valid']): Path(f['directory']) for m, frames, _ in memberframes for f in frames}
    planes = {}
    count = 0

    def offset(a, b, c):
        curve = a - 2 * b + c
        if not np.isfinite(curve) or curve <= 0:
            return 0.0
        return float(np.clip(0.5 * (a - c) / curve, -0.5, 0.5))

    for track in tracks:
        for fix in track['fixes']:
            d = dirs.get((track['member_id'], fix['valid']))
            if fix.get('lat') is None or d is None:
                continue
            if d not in planes:
                grid = json.loads((d / 'grid.json').read_text())
                plane = np.fromfile(d / f"{rule['field']}.f32", dtype='<f4').reshape(grid['shape']).astype(np.float64)
                mask = np.fromfile(d / 'mask.u8', dtype=np.uint8).reshape(grid['shape']).astype(bool)
                planes[d] = (np.asarray(grid['lat'], float), np.asarray(grid['lon'], float), np.where(mask, plane, np.nan))
            ys, xs, plane = planes[d]
            i = int(np.argmin(np.abs(ys - fix['lat'])))
            j = int(np.argmin(np.abs((xs - fix['lon'] + 180) % 360 - 180)))
            if not (0 < i < len(ys) - 1 and 0 < j < len(xs) - 1):
                continue
            dy = offset(plane[i - 1, j], plane[i, j], plane[i + 1, j])
            dx = offset(plane[i, j - 1], plane[i, j], plane[i, j + 1])
            step = ((xs[j + 1] - xs[j - 1]) + 180) % 360 - 180
            fix['lat'] = float(ys[i] + dy * (ys[i + 1] - ys[i - 1]) / 2)
            fix['lon'] = float((xs[j] + dx * step / 2 + 180) % 360 - 180)
            count += 1
    return count


def coordinate(s, scale):
    return float(s[:-1]) * scale * (-1 if s[-1] in 'SW' else 1)


def best_track(path, storm=None, *, start=None):
    """Normalize observed files in Rust, retaining one event identity."""
    # Reuse a saved reader result (gpuwm.tc-tracks.v1) without reinterpreting ATCF/HURDAT units.
    text = ''
    if Path(path).is_file():
        with Path(path).open('rb') as stream:
            compressed = stream.read(2) == b'\x1f\x8b'
        # The native ATCF reader already owns gzip decoding. Only probe plain
        # text for saved JSON; decoding a compressed b-deck here aborts first.
        if not compressed:
            text = Path(path).read_text()
    if text.lstrip().startswith('{'):
        saved = json.loads(text)
        # A scored result carries best_track; a tracker document (cyclone-tracks output, same schema) carries
        # tracks and is read by load_tracks below (hex Milton waves run 2026-10-30: the acquired NHC best track was refused here).
        if saved.get('schema') == 'gpuwm.tc-tracks.v1' and 'best_track' in saved:
            if not saved.get('best_track', {}).get('fixes'):
                raise ValueError('saved tracks need gpuwm.tc-tracks.v1 and a nonempty best_track')
            result = dict(saved['best_track'])
            if storm and result.get('storm_id') != storm:
                raise ValueError('saved best track belongs to another storm')
            return dict(result, saved_tracks=dict(path=str(Path(path).resolve()), sha256=sha(path)))
    import tempfile
    from woof.cyclone_tracks import load_tracks, native_track_binary
    path = Path(path).resolve()
    if path.suffix.lower() == ".json":
        doc, _, digest = load_tracks(path)
    else:
        with tempfile.TemporaryDirectory(prefix="best-track-") as tmp:
            root = Path(tmp)
            output = root/"tracks.json"
            request = dict(input=str(path), storm=storm or "auto",
                start=stamp(start) if start else "1970-01-01T00:00:00Z",
                parse_window=start is not None, cache=str(root/"cache"), output=str(output))
            req = root/"request.json"
            dump(req, request)
            result = subprocess.run([native_track_binary("cyclone-tracks"), "cyclone-tracks", "--request", str(req)],
                                    capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise ValueError("best-track-parser: "+result.stderr.strip())
            doc, _, _ = load_tracks(output)
        digest = sha(path)
    candidates = [t for t in doc["tracks"] if not storm or
                  storm.casefold() in {str(x).casefold() for x in
                    [t["track_id"], t.get("name", ""), *t.get("aliases", [])]}]
    if len(candidates) != 1:
        raise ValueError("best-track identity: select one event with --storm-id; matches: " +
                         ", ".join(t["track_id"] for t in candidates))
    chosen = candidates[0]
    return dict(path=str(path), sha256=digest, format=doc["tracker"].get("source", "normalized"),
                storm_id=chosen["track_id"],
                wind_definition="USA/NHC/JTWC maximum 1-minute sustained wind; differs from model instantaneous 10 m wind",
                acquisition=doc["tracker"], fixes=chosen["fixes"])


def distance(a, b):
    la, lb = math.radians(a['lat']), math.radians(b['lat'])
    dl = math.radians(b['lon'] - a['lon'])
    h = math.sin((lb-la)/2)**2 + math.cos(la)*math.cos(lb)*math.sin(dl/2)**2
    return 12742 * math.asin(min(1, math.sqrt(h)))


def scores(members, tracks, best):
    """Score event-matched tracks in Rust with fixed trajectory identity."""
    import tempfile
    from woof.cyclone_tracks import native_track_binary
    limit = json.loads(TABLE.read_text())['verification']['association_max_distance_km']
    with tempfile.TemporaryDirectory(prefix="track-score-") as temporary:
        request = Path(temporary)/"request.json"
        dump(request, dict(members=members, tracks=tracks, best=best,
                           association_max_distance_km=limit))
        result = subprocess.run([native_track_binary("cyclone-score"), "cyclone-score", "--request", str(request)],
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise ValueError("track-score: "+result.stderr.strip())
        return json.loads(result.stdout)


def missing(valid, init):
    return dict(valid=valid, lead_hours=(utc(valid)-utc(init)).total_seconds()/3600,
                lat=None, lon=None, mslp_pa=None, max_wind10_ms=None, flags=['missing'])


def write_csv(path, tracks):
    with Path(path).open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=COLUMNS)
        w.writeheader()
        for t in tracks:
            for f in t['fixes']:
                w.writerow(dict(member_id=t['member_id'], track_id=t['track_id'], weight=t['weight'],
                                **{k: json.dumps(f[k]) if k == 'flags' else f[k] for k in COLUMNS[3:]}))


def tempest_duration(frames, config):
    """Short forecasts can show provisional paths, not 24-hour persistence."""
    span = (utc(frames[-1]['valid']) - utc(frames[0]['valid'])).total_seconds()
    required = config['minimum_duration_seconds']
    return dict(record_seconds=span, configured_minimum_seconds=required,
        effective_minimum_seconds=min(span, required), provisional=span < required,
        policy=config['short_record_policy'])


def tempest(member, frames, config, work, workers=1, tools=None):
    nc = work / 'tempest.nc'
    req = dict(frames=[dict(directory=f['directory'], unix_hours=utc(f['valid']).timestamp()/3600) for f in frames], fields=config['fields'], out=str(nc))
    request = work / 'tempest-request.json'
    dump(request,req)
    stages=[]
    start=time.monotonic();run([binary('rw_tc_fields'),'tempest',request]);stages.append(dict(stage='tempest_write',seconds=time.monotonic()-start,parallelism='1 thread; bounded native NetCDF staging'))
    nodes=work/'nodes.txt';stitched=work/'stitched.txt'
    grid=json.loads((Path(frames[0]['directory'])/'grid.json').read_text())
    regional=grid['route']!='global_exact_lattice'
    detect_exe,stitch_exe=tools or earth2_env.tempest_tools('--backend tempestextremes')
    detect=[detect_exe,'--in_data',nc,'--out',nodes,*config['detect']]
    if regional:detect+=['--regional']
    # StitchNodes prepends the two structured-grid index columns itself.
    duration = tempest_duration(frames, config)
    stitch=[stitch_exe,'--in',nodes,'--out',stitched,'--in_fmt','lon,lat,slp,wind','--caltype','standard',*config['stitch'],
        '--mintime', f"{duration['effective_minimum_seconds']:g}s"]
    # Detect each disjoint time group concurrently, then stitch the complete
    # chronologically ordered node stream once. A single-member file benefits
    # from the worker budget just as a many-member manifest does.
    slots=min(workers,len(frames))
    groups=[frames[i*len(frames)//slots:(i+1)*len(frames)//slots] for i in range(slots)]
    def detect_group(item):
        i,group=item
        part=work/f'nodes-{i:04d}.txt'
        cmd=[str(x) for x in detect]
        cmd[cmd.index('--out')+1]=str(part)
        pattern='^('+ '|'.join(utc(f['valid']).strftime('%Y-%m-%d %H:%M:%S') for f in group) +')$'
        cmd+=['--timefilter',pattern]
        start=time.monotonic()
        logpath=work/f'detect-{i:04d}.log'
        with logpath.open('w') as log:run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if 'EXCEPTION (' in logpath.read_text():
            raise ValueError(f'TempestExtremes detect failed; see {logpath}')
        return part,dict(stage='detect_time_group',seconds=time.monotonic()-start,parallelism=f'1 process in {slots} concurrent member workers',command=cmd)
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=slots) as pool:parts=list(pool.map(detect_group,enumerate(groups)))
    with nodes.open('w') as stream:
        for part,stage in parts:
            stream.write(part.read_text());stages.append(stage)
    stages.append(dict(stage='detect_all_times',seconds=time.monotonic()-start,parallelism=f'{slots} concurrent processes; disjoint valid times'))
    for name,cmd in [('stitch',stitch)]:
        start=time.monotonic()
        with (work/f'{name}.log').open('w') as log:run(cmd,stdout=log,stderr=subprocess.STDOUT)
        # These upstream tools catch their own exceptions and can exit zero.
        backend_log=(work/f'{name}.log').read_text()
        if 'EXCEPTION (' in backend_log or 'do not have matching --in_fmt entries' in backend_log:
            raise ValueError(f'TempestExtremes {name} failed; see {work/name}.log')
        stages.append(dict(stage=name,seconds=time.monotonic()-start,parallelism='1 process per member; short complete-path association',command=[str(x) for x in cmd]))
    bytime={f['valid']:i for i,f in enumerate(frames)}; tracks=[];fixes=None
    for line in stitched.read_text().splitlines():
        f=line.split()
        if not f:continue
        if f[0]=='start':
            fixes=[missing(fr['valid'],member['init']) for fr in frames]
            tracks.append(dict(track_id=f'{member["member_id"]}-{len(tracks):03d}',member_id=member['member_id'],weight=member['weight'],fixes=fixes))
        elif fixes is not None:
            # StitchNodes output: i, j, lon, lat, slp, wind, year, month, day, hour.
            valid=stamp(datetime(int(f[-4]),int(f[-3]),int(f[-2]),tzinfo=timezone.utc)+timedelta(hours=float(f[-1])))
            if valid not in bytime:raise ValueError(f'TempestExtremes produced unexpected valid {valid}')
            fix=fixes[bytime[valid]];fix.update(lat=float(f[3]),lon=(float(f[2])+180)%360-180,mslp_pa=float(f[4]),max_wind10_ms=float(f[5]),flags=[])
            if duration['provisional']:fix['flags'].append('provisional-short-record')
            edge=min(abs(fix['lat']-min(grid['lat'])),abs(max(grid['lat'])-fix['lat']))*111.195
            edge=min(edge,min(abs(((fix['lon']-x+180)%360)-180) for x in (grid['lon'][0],grid['lon'][-1]))*111.195*math.cos(math.radians(fix['lat'])))
            if regional and edge<=config['edge_radius_km']:fix['flags'].append('edge')
    if not tracks:tracks=[dict(track_id=member['member_id']+'-none',member_id=member['member_id'],weight=member['weight'],fixes=[missing(fr['valid'],member['init']) for fr in frames])]
    return dict(tracks=tracks,stages=stages,duration=duration,route='regional TempestExtremes --regional' if regional else 'global TempestExtremes',peak_vram_mib=0)


def execute(args):
    import tempfile
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.tc-work-', dir=out))
    try:
        return execute_work(args, out, work)
    except Exception as error:
        dump(out / 'failure-receipt.json', dict(error=str(error), work=str(work)))
        raise
    finally:
        # Remove owned atmospheric staging on both success and failure.
        deleted = []
        for p in sorted(work.rglob('*')):
            if p.is_file() and (p.suffix in ('.f32', '.u8', '.nc') or any(a.endswith('.zarr') for a in p.parts)):
                deleted.append(dict(path=str(p), bytes=p.stat().st_size))
                p.unlink()
        dump(out / 'deleted-files.json', deleted)


def backend_runtime(backend):
    """Resolve the backend's runtime BEFORE any extraction work, building it on first use.

    Breakage (2.9.3 door 6, G12, 2026-10-20): a fresh install refused here only after extraction had
    run, naming an env var (GPU backends) or printing a raw ENOENT for DetectNodes (tempestextremes).
    """
    door=f'--backend {backend}'
    if backend=='tempestextremes':
        return dict(tools=earth2_env.tempest_tools(door,provision=True))
    return dict(interpreter=earth2_env.earth2_python(door,extras=('cyclone',),provision=True))


def execute_work(args, out, work):
    begin=time.monotonic();stages=[]
    table=json.loads(TABLE.read_text());config=table['backends'][args.backend]
    start=time.monotonic();resolved=backend_runtime(args.backend)
    stages.append(dict(stage='backend_runtime',seconds=time.monotonic()-start,parallelism='1 thread; interpreter probe (a first-use build runs its downloads concurrently)',**resolved))
    bridge_start=time.monotonic()
    if args.manifest:
        manifest=normalize_manifest(json.loads(Path(args.manifest).read_text()),Path(args.manifest).resolve().parent)
    elif getattr(args, 'mesh', None):
        manifest=hex_input(args.input,args.mesh,work,initial=getattr(args,'hex_init',None),
                           window=getattr(args,'hex_window','global'),workers=args.workers)
    else:
        manifest=shortcut(args.input,work)
    stages.append(dict(stage='admit_export',seconds=time.monotonic()-bridge_start,
                       parallelism=f'Rust native converter/exporter; up to {args.workers} frames'))
    dump(out/'manifest.json',manifest)
    members=manifest['members'];jobs=[];memberframes=[]
    slots=min(args.workers, max(1,sum(len(m['frames']) for m in members)))
    threads=max(1,(os.cpu_count() or 1)//slots)
    for mi,m in enumerate(members):
        mdir=work/f'member-{mi:04d}';mdir.mkdir();frames=[]
        for fi,frame in enumerate(m['frames']):
            d=mdir/f'frame-{fi:04d}';r=dict(path=frame['path'],select=frame.get('select',{}),out=str(d),vocabulary=table['vocabularies'][frame.get('vocabulary',m.get('vocabulary','earth2studio'))],fields=config['fields'])
            req=mdir/f'extract-{fi:04d}.json';dump(req,r)
            jobs.append(req);frames.append(dict(frame,directory=str(d)))
        memberframes.append((m,frames,mdir))
    start=time.monotonic()
    def extract(req):
        env=dict(os.environ,RAYON_NUM_THREADS=str(threads))
        with req.with_suffix('.log').open('w') as log:run([binary('rw_tc_fields'),'extract',req],env=env,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(max_workers=slots) as pool:list(pool.map(extract,jobs))
    stages.append(dict(stage='extract_regrid_write',seconds=time.monotonic()-start,parallelism=f'{slots} processes x {threads} Rust threads'))
    backend_results=[]
    if args.backend=='tempestextremes':
        member_slots=min(args.workers,len(memberframes))
        time_workers=max(1,args.workers//member_slots)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            backend_results=list(pool.map(lambda item:tempest(*item[:2],config,item[2],time_workers,resolved['tools']),memberframes))
    else:
        interpreter=resolved['interpreter']
        for m,frames,mdir in memberframes:
            req=mdir/'gpu-request.json';result=mdir/'gpu-result.json'
            dump(req,dict(member=m,frames=frames,backend=args.backend,config=config,out=str(result)))
            env=dict(os.environ)
            extra=env.get('WOOF_EARTH2_EXTRA_SITE')
            if extra:env['PYTHONPATH']=extra+os.pathsep+env.get('PYTHONPATH','')
            gpu_log = mdir/'gpu.log'
            try:
                with gpu_log.open('w') as log:run([interpreter,Path(__file__).with_name('tc_tracks_gpu.py'),req],env=env,stdout=log,stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as error:
                raise ValueError(f'{args.backend} tracker exited {error.returncode}; log {gpu_log}\n'
                                 + gpu_log.read_text(errors='replace')[-4000:]) from error
            backend_results.append(json.loads(result.read_text()))
    tracks=[t for r in backend_results for t in r['tracks']]
    for r in backend_results:stages.extend(r['stages'])
    if config.get('center_refinement'):
        start=time.monotonic();count=refine_centers(tracks,memberframes,config['center_refinement'])
        stages.append(dict(stage='center_refinement',seconds=time.monotonic()-start,fixes=count,parallelism='1 thread; 3x3 numpy fit per fix (milliseconds)'))
    paths=sorted({f['path'] for m in members for f in m['frames']})
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:hashes=dict(zip(paths,pool.map(sha,paths)))
    stages.append(dict(stage='input_hash',seconds=time.monotonic()-start,parallelism=f'{min(args.workers,len(paths))} active workers'))
    code_hashes = {name: sha(path) for name, path in {
        'orchestrator': Path(__file__), 'gpu_driver': Path(__file__).with_name('tc_tracks_gpu.py'),
        'native_reader': Path(binary('rw_tc_fields')), 'renderer': Path(binary('rw_wrfbatch'))}.items()}
    from woof.build_provenance import package_build_identity
    identity = package_build_identity()
    revision = identity['commit']
    tracker_config=dict(config,code_sha256=code_hashes,field_authority_sha256=sha(TABLE),cadence_hours={m['member_id']:[(utc(b['valid'])-utc(a['valid'])).total_seconds()/3600 for a,b in zip(m['frames'],m['frames'][1:])] for m in members})
    document=dict(schema=SCHEMA,tracker=dict(backend=args.backend,config=tracker_config,config_sha256=hashlib.sha256(json.dumps(tracker_config,sort_keys=True).encode()).hexdigest(),code_revision=revision),
                  members=[dict(**{k:m[k] for k in ('member_id','family','source','init','weight','ancestry')},inputs=[dict(path=p,sha256=hashes[p]) for p in sorted({f['path'] for f in m['frames']})]) for m in members],tracks=tracks)
    renderer_parallelism = f"Rayon pool; RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'unset (native default)')}"
    for member in document['members']:
        if 'stratum' in member['ancestry']:
            member['stratum'] = member['ancestry']['stratum']
    if args.best_track:
        start=time.monotonic()
        if args.best_track == 'auto':
            if not args.storm_id:
                raise ValueError('--best-track auto needs --storm-id to bind event-matched observations')
            from woof.cyclone_tracks import acquire_tracks
            observed = acquire_tracks(storm=args.storm_id,
                start=utc(members[0]['init']).replace(tzinfo=None),
                source=getattr(args, 'track_source', 'auto'),
                cache=getattr(args, 'track_cache', None))
        else:
            observed = args.best_track
        document['best_track']=best_track(observed,args.storm_id,start=utc(members[0]['init']))
        document['scores']=scores(members,tracks,document['best_track'])
        stages.append(dict(stage='best_track_acquire_score',seconds=time.monotonic()-start,
                           parallelism='Rust source parsing and trajectory scoring'))
    dump(out/'tracks.json',document);write_csv(out/'tracks.csv',tracks)
    receipt=dict(schema='gpuwm.tc-tracks-receipt.v1',command=[sys.executable,'-m','woof',*sys.argv[1:]],code_revision=revision,package_provenance=identity,backend=args.backend,config_sha256=document['tracker']['config_sha256'],input_sha256=hashes,
                 routes=[r['route'] for r in backend_results],route_reason='GPU trackers require exact 721x1440 0.25 degree global lattice; regional fields use an explicit no-signal mask. TempestExtremes uses regional topology when cropped.',
                 stages=stages,total_seconds=time.monotonic()-begin,peak_vram_mib=max(r['peak_vram_mib'] for r in backend_results),number_of_fixes=sum(f['lat'] is not None for t in tracks for f in t['fixes']),
                 backend_receipts=[{k:v for k,v in r.items() if k not in ('tracks','stages')} for r in backend_results],scores=document.get('scores'),weight_policy=manifest['weight_policy'],wind_definition='Model instantaneous maximum 10 m wind; observed USA agency 1-minute sustained wind is a different quantity.',landfall_error=None)
    if args.best_track:
        from woof.rustwx import renderer_env
        start=time.monotonic()
        run([binary('rw_wrfbatch'),'--tc-track-landfalls',out/'tracks.json',out/'landfalls.json'],env=renderer_env())
        landfalls=json.loads((out/'landfalls.json').read_text())
        comparisons=[]
        for b in document['best_track']['fixes']:
            if 'landfall' not in b['flags']:continue
            for m in members:
                choices=[f for f in landfalls['crossings'] if f['member_id']==m['member_id'] and distance(f,b)<1000]
                f=min(choices,key=lambda f:abs((utc(f['valid'])-utc(b['valid'])).total_seconds())) if choices else None
                comparisons.append(dict(member_id=m['member_id'],best=b,model=f,distance_km=distance(f,b) if f else None,time_error_hours=(utc(f['valid'])-utc(b['valid'])).total_seconds()/3600 if f else None))
        receipt['landfall_error']=dict(method=landfalls['method'],comparisons=comparisons)
        stages.append(dict(stage='landfall',seconds=time.monotonic()-start,parallelism=renderer_parallelism))
    if args.render:
        from woof.rustwx import renderer_env
        png=out/'d01'/'tc-track-plume'/members[0]['init'][:10]/'track-plume.png';png.parent.mkdir(parents=True,exist_ok=True)
        start=time.monotonic();run([binary('rw_wrfbatch'),'--tc-track-plume',out/'tracks.json',png],env=renderer_env())
        stages.append(dict(stage='render',seconds=time.monotonic()-start,parallelism=renderer_parallelism))
        chart=png.with_name('intensity.png')
        start=time.monotonic();intensity_chart(document,chart)
        stages.append(dict(stage='intensity_chart',seconds=time.monotonic()-start,parallelism='1 thread; small plotted track table'))
    receipt['total_seconds']=time.monotonic()-begin
    dump(out/'receipt.json',receipt)
    print(out/'tracks.json')
    return 0


def intensity_chart(document, path):
    """Intensity series only. Weather maps always use rw_wrfbatch."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for track in document['tracks']:
        fixes = track['fixes']
        times = [utc(f['valid']) for f in fixes]
        label = f"{track['member_id']} {track['track_id']} w={track['weight']:.4g}"
        for ax, key, scale in ((axes[0], 'mslp_pa', .01), (axes[1], 'max_wind10_ms', 1)):
            ax.plot(times, [f[key]*scale if f[key] is not None else float('nan') for f in fixes], marker='.', label=label)
    valids = [f['valid'] for t in document['tracks'] for f in t['fixes'] if f]
    best = [f for f in document.get('best_track', {}).get('fixes', [])
            if valids and min(valids) <= f['valid'] <= max(valids)]
    if best:
        for ax, key, scale in ((axes[0], 'mslp_pa', .01), (axes[1], 'max_wind10_ms', 1)):
            ax.plot([utc(f['valid']) for f in best], [f[key]*scale if f[key] is not None else float('nan') for f in best], color='black', label='Observed best track')
    axes[0].set_ylabel('Central pressure (hPa)')
    axes[1].set_ylabel('Wind (m/s)')
    axes[1].set_xlabel('Valid time (UTC)')
    axes[0].set_title('Completed-output cyclone intensity')
    # Observed-agency wording only when an observed series is drawn (improve-55: the title named NHC with no NHC line).
    axes[1].set_title('Model instantaneous 10 m maximum; observed USA agency 1-minute sustained maximum' if best
                      else 'Model instantaneous 10 m maximum')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def register_cli(subparsers):
    p=subparsers.add_parser('tc-tracks',help='Track tropical cyclones in completed member output',
        description='Track tropical cyclones in completed member output.',
        epilog=('Backends and what they run: wuduan (default) and vitart run the Earth2Studio GPU trackers '
                'in an Earth2Studio interpreter (earth2studio==0.19.0 with the [cyclone] extra); '
                'tempestextremes runs the TempestExtremes executables DetectNodes and StitchNodes. '
                'The interpreter is WOOF_EARTH2_PYTHON when set, else the managed environment; the '
                'tools are WOOF_DETECTNODES/WOOF_STITCHNODES, else PATH, else the managed environment. '
                '`woof setup --with-earth2` builds the managed environment ahead of time; otherwise '
                'the first run that needs it builds it (WOOF_EARTH2_FETCH=0 refuses instead). '
                'Location: WOOF_EARTH2_ENV, default ~/.local/share/woof/earth2-env.'))
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--manifest');group.add_argument('--input')
    p.add_argument('--out',required=True);p.add_argument('--backend',choices=('wuduan','vitart','tempestextremes'),default='wuduan')
    p.add_argument('--render',action='store_true',help='Draw the plume with rw_wrfbatch');p.add_argument('--best-track');p.add_argument('--storm-id');p.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1))
    p.add_argument('--mesh', type=Path, help='mesh/static file for native MPAS hex --input history')
    p.add_argument('--hex-init', type=Path, help='MPAS initialization supplying static fields absent from history')
    p.add_argument('--hex-window', default='global', help='rw_mpas_convert window, default global 0.25 degree')
    p.add_argument('--track-source', choices=('auto','nhc','hurdat2','ibtracs','jtwc'), default='auto')
    p.add_argument('--track-cache', type=Path)
    p.set_defaults(func=main)


def main(args):
    try:
        if args.workers<1:raise ValueError('--workers must be positive')
        return execute(args)
    except (ValueError,OSError,subprocess.CalledProcessError) as e:
        print(f'woof tc-tracks: {e}',file=sys.stderr)
        return 2
