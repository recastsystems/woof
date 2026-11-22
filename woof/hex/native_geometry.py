"""Source-bound fixed native mesh publication, without manufacturing global IDs.

Registry mesh arrays use the original native iterator order. Native transport is
little-endian RKIND/i4, Fortran storage, with one trailing dummy entity; output
captures only the declared owned extent. This contract never remaps that order.
"""
from dataclasses import replace
from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np

SCHEMA = 'hex-native-frame-identity/v1'
SOURCE_SCHEMA = 'hex-native-source-identity/v1'
FIELDS = {'indexToCellID': ('nCells', 'integer'), 'latCell': ('nCells', 'real'),
          'lonCell': ('nCells', 'real'), 'indexToEdgeID': ('nEdges', 'integer'),
          'latEdge': ('nEdges', 'real'), 'lonEdge': ('nEdges', 'real')}
CLOCK = '%Y-%m-%d_%H:%M:%S'


def fingerprint(array):
    a = np.ascontiguousarray(array)
    return dict(dtype=a.dtype.str, shape=list(a.shape),
                sha256=hashlib.sha256(a.tobytes()).hexdigest())


def _mesh_array(name, value, *, native=False):
    """The pool reader accepts NC_FLOAT/DOUBLE coordinates and NC_INT IDs."""
    a = np.asarray(value)
    kind = FIELDS[name][1]
    dtype = np.dtype('<i4' if kind == 'integer' else '<f4')
    typed = (a.dtype.kind == 'i' and a.dtype.itemsize == 4 if kind == 'integer'
             else a.dtype.kind == 'f' and a.dtype.itemsize in (4, 8))
    if (not typed or (native and a.dtype != dtype) or a.ndim != 1 or not a.size
            or np.ma.is_masked(value) or not np.isfinite(a).all()):
        raise ValueError(f'{name}: original typed mesh dtype/owned extent is invalid')
    if kind == 'integer' and (np.any(a <= 0) or len(np.unique(a)) != len(a)):
        raise ValueError(f'{name}: original typed mesh IDs must be positive and unique')
    return a, dtype


def source_identity(arrays):
    """Pin original typed mesh words and their native RKIND/i4 conversion.

    Call on the actual owned source arrays before integration, with NetCDF mask
    and scale conversion disabled. The Rust pool reader uses nc_get_vara_float
    for NC_FLOAT/DOUBLE and nc_get_vara_int for NC_INT. Fingerprints preserve
    dtype, order and signed zero; neither IDs nor original precision are rebuilt
    from the rounded native frame. No caller array is modified.
    """
    fields = {}
    for name in FIELDS:
        if name not in arrays:
            raise ValueError(f'{name}: original typed mesh field is missing')
        a, dtype = _mesh_array(name, arrays[name])
        with np.errstate(over='ignore', invalid='ignore'):
            converted = a.astype(dtype, copy=False)
        if not np.isfinite(converted).all():
            raise ValueError(f'{name}: original typed mesh exceeds native precision')
        fields[name] = dict(original=fingerprint(a), native=fingerprint(converted))
    for name, (dim, _) in FIELDS.items():
        ids = 'indexToCellID' if dim == 'nCells' else 'indexToEdgeID'
        if fields[name]['original']['shape'] != fields[ids]['original']['shape']:
            raise ValueError(f'{name}: original typed mesh identity extents differ')
    return dict(schema=SOURCE_SCHEMA, fields=fields)


def _validate_source_identity(arrays, record):
    if (not isinstance(record, dict) or set(record) != {'schema', 'fields'}
            or record['schema'] != SOURCE_SCHEMA or not isinstance(record['fields'], dict)
            or set(record['fields']) != set(FIELDS)):
        raise ValueError('original typed mesh source identity schema/fields differ')
    for name, row in record['fields'].items():
        if name not in arrays:
            raise ValueError(f'{name}: original typed mesh native field is missing')
        a, _ = _mesh_array(name, arrays[name], native=True)
        expected = fingerprint(a)
        if not isinstance(row, dict) or set(row) != {'original', 'native'} or row['native'] != expected:
            raise ValueError(f'{name}: original typed mesh native conversion differs')
        original = row['original']
        if not isinstance(original, dict) or set(original) != {'dtype', 'shape', 'sha256'}:
            raise ValueError(f'{name}: original typed mesh fingerprint is invalid')
        try:
            dtype = np.dtype(original['dtype'])
        except (TypeError, ValueError):
            raise ValueError(f'{name}: original typed mesh dtype is invalid') from None
        kind = FIELDS[name][1]
        typed = (dtype.kind == 'i' and dtype.itemsize == 4 if kind == 'integer'
                 else dtype.kind == 'f' and dtype.itemsize in (4, 8))
        digest = original['sha256']
        if (not typed or original['dtype'] != dtype.str or original['shape'] != expected['shape']
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError(f'{name}: original typed mesh fingerprint is invalid')


def validate_template(arrays, evidence, template_arrays):
    """Require exact source dtype/bytes and exact native conversion, without casts of IDs.

    Use after validate() has admitted the producer's request and transport
    binding. This check does not authorize an unsealed frame or template.
    """
    source = evidence.get('source_identity')
    if source != source_identity(template_arrays):
        raise ValueError('original typed mesh differs from the recorded source')
    _validate_source_identity(arrays, source)
    return source


def bound(path):
    p = Path(path).resolve()
    b = p.read_bytes()
    return json.loads(b), dict(path=str(p), sha256=hashlib.sha256(b).hexdigest())


def publish_native_identity(streams, layout):
    """Add actual Registry mesh slots to native history, never a synthetic range."""
    rows = {r['name']: r for r in layout.fields}
    for name, (dim, kind) in FIELDS.items():
        row = rows.get(name)
        if row is None or row['pool'] != 'mesh' or row['kind'] != kind or row['dimensions'] != [dim]:
            raise ValueError(f'{name}: native history requires the actual Registry mesh authority')
    result = []
    for stream in streams:
        runtime = callable(stream.fields)
        directions = stream.options['type'].split(';') if runtime else stream.directions
        if stream.name in ('output', 'history') and 'output' in directions:
            if runtime:
                members = tuple(dict.fromkeys((*stream.members, *(('var', n, '') for n in FIELDS))))
                stream = replace(stream, members=members)
            else:
                stream = replace(stream, fields=tuple(dict.fromkeys((*stream.fields, *FIELDS))))
        result.append(stream)
    return result


def scientific_identity(request, manifest, request_path, original_request=None):
    """Bind native settings and transported words independently of locators.

    The complete pool manifest pins transported init/grid words. Other stream
    inputs and executable readers are bound by their own file content.
    """
    def file_pin(value):
        path = Path(value)
        if (path.name == '.env' or path.name.endswith('.env') or
                path.name in ('auth.json', '.cdsapirc') or
                any(part in ('secrets', '.aws') for part in path.parts)):
            raise ValueError('native identity cannot bind credential material')
        if original_request:
            try:
                path = Path(request_path).resolve().parent / path.relative_to(Path(original_request).parent)
            except ValueError:
                pass
        with path.open('rb') as stream:
            return {'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}

    def walk(value, keys=()):
        if isinstance(value, dict):
            return {k: walk(v, (*keys, k)) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, keys) for v in value]
        if not isinstance(value, str):
            return value
        key = keys[-1] if keys else ''
        if key == 'origins':
            # Native row ordering and field roles remain bound by the manifest.
            return value.rsplit(':', 1)[-1]
        if keys in (('init',), ('grid',), ('pool_manifest',)) or key == 'source_init':
            return {'role': 'transported-native-input' if key != 'pool_manifest' else 'native-pool-manifest'}
        if key == 'input_file' and value == request.get('init'):
            return {'role': 'transported-native-input'}
        if value.startswith('/'):
            if key in ('filename', 'filename_template'):
                return {'output': list(keys)}
            return file_pin(value)
        return value
    return {'request': walk(request), 'manifest': walk(manifest)}


def seal(arrays, request_path, valid, step, *, original_request=None, bind_settings=True):
    """Qualify captured fixed mesh against original hashed Rust transport words.

    A moving/current-grid transport needs its own current qualified source; it
    cannot reuse this original fixed-grid evidence after coordinates change.
    """
    request, request_pin = bound(request_path)
    if request.get('schema') != 'mpas-native-door-v1':
        raise ValueError('native identity requires the actual native door request')
    start = datetime.strptime(request['start'], CLOCK)
    when = datetime.strptime(valid, CLOCK)
    if (not isinstance(step, int) or not 0 <= step <= request['steps'] or
            Fraction(str((when-start).total_seconds())) != step * Fraction(request['dt']) or
            when > datetime.strptime(request['stop'], CLOCK)):
        raise ValueError('native identity clock differs from the exact native request step')
    manifest_path = Path(request['pool_manifest'])
    if original_request:
        # A byte-identical retained request still records the original transport
        # directory. Locate its copied sibling without requiring that directory.
        try:
            relative = manifest_path.relative_to(Path(original_request).parent)
        except ValueError:
            pass
        else:
            manifest_path = Path(request_path).resolve().parent / relative
    manifest_path = manifest_path.resolve()
    manifest, manifest_pin = bound(manifest_path)
    if manifest.get('schema') != 'mpas-native-pool-payload-v1':
        raise ValueError('native identity lacks original native pool transport authority')
    fields = {}
    for name, (dim, kind) in FIELDS.items():
        if name not in arrays:
            raise ValueError(f'{name}: native frame lacks actual mesh identity')
        a = np.asarray(arrays[name])
        dtype = np.dtype('<i4' if kind == 'integer' else '<f4')
        if a.ndim != 1 or a.dtype != dtype or not len(a) or not np.isfinite(a).all():
            raise ValueError(f'{name}: native mesh dtype/owned extent is invalid')
        if kind == 'integer' and (np.any(a <= 0) or len(np.unique(a)) != len(a)):
            raise ValueError(f'{name}: native IDs must be actual positive unique IDs')
        rows = [r for r in manifest['fields'] if r['name'] == name and r['pool'] == 'mesh']
        if len(rows) != 1:
            raise ValueError(f'{name}: missing or ambiguous original mesh source')
        row = rows[0]
        origins = row.get('origins', [])
        if (row['kind'] != kind or row['dimensions'] != [dim] or
                len(origins) != 1 or origins[0].rsplit(':', 1)[-1] != name or row.get('fill') is not None or
                row['shape'] not in ([len(a)], [len(a)+1])):
            raise ValueError(f'{name}: original native mesh source/shape is not qualified')
        payload = (manifest_path.parent / row['payload']).resolve()
        if payload.parent != manifest_path.parent:
            raise ValueError('native mesh payload escapes original transport')
        b = payload.read_bytes()
        if hashlib.sha256(b).hexdigest() != row['sha256'] or len(b) != 4 * row['shape'][0]:
            raise ValueError(f'{name}: original native mesh payload hash/size differs')
        source = np.frombuffer(b, dtype=dtype).reshape(row['shape'], order='F')[:len(a)]
        if fingerprint(a) != fingerprint(source):
            raise ValueError(f'{name}: captured native mesh order/geometry differs from actual source')
        fields[name] = dict(array=fingerprint(a), source=dict(payload=row['payload'],
            sha256=row['sha256'], shape=row['shape'], origins=row['origins'], storage='Fortran', owned_axis=0))
    if len(arrays['latCell']) != len(arrays['indexToCellID']) or len(arrays['lonCell']) != len(arrays['indexToCellID']):
        raise ValueError('native cell identity extents differ')
    if len(arrays['latEdge']) != len(arrays['indexToEdgeID']) or len(arrays['lonEdge']) != len(arrays['indexToEdgeID']):
        raise ValueError('native edge identity extents differ')
    record = dict(schema=SCHEMA, valid=valid, step=step, source_clock=request['start'],
                source_init=request['init'], request=request_pin, manifest=manifest_pin,
                fields=fields, role='static-identity-not-analysis-control')
    if 'source_identity' in request:
        _validate_source_identity(arrays, request['source_identity'])
        record['source_identity'] = request['source_identity']
    if bind_settings:
        record['scientific_identity'] = scientific_identity(request, manifest, request_path, original_request)
    return record


def validate(arrays, evidence, request_path, valid):
    if evidence.get('schema') != SCHEMA or evidence.get('valid') != valid:
        raise ValueError('native identity evidence schema/clock differs')
    expected = seal(arrays, request_path, valid, evidence['step'],
                    original_request=evidence.get('request', {}).get('path'),
                    bind_settings='scientific_identity' in evidence)
    def identity(record):
        record = json.loads(json.dumps(record))
        for name in ('request', 'manifest'):
            record[name].pop('path', None)
        record.pop('source_init', None)
        for row in record['fields'].values():
            row['source']['origins'] = [s.rsplit(':', 1)[-1] for s in row['source']['origins']]
        if 'scientific_identity' in record:
            # Raw JSON hashes remain useful paper evidence. Rewriting locators
            # changes them without changing settings or transported content.
            record.pop('request')
            record.pop('manifest')
        return record
    if identity(evidence) != identity(expected):
        raise ValueError('native identity source/geometry binding differs; legacy records without '
                         'scientific settings identity require unchanged metadata bytes')
    return expected


def sidecar(path):
    return Path(str(path) + '.native-identity.json')


def publish(path, record):
    """Atomic small metadata publication; no forecast state words are rewritten."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, prefix='.'+target.name,
                                     delete=False) as f:
        temporary = Path(f.name)
        json.dump(record, f, sort_keys=True, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
