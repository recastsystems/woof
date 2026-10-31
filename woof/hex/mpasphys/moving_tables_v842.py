"""Certified immutable physics tables retained by a moving forecast's CUDA device.

Only the named read-only table loaders participate. Each session gets its own
containers and shortwave workspaces; decoded coefficient arrays and modules
stay resident. File identity changes replay the original authenticating loader.
"""
from copy import copy
import json
from pathlib import Path
import threading

from .. import numerics

_LOCK = threading.RLock()
_TABLES = {}
_PACKETS = {}
_LOADERS = {
    ('table_resources_v842', 'load_thompson_resources'),
    ('thompson_aero_v842', 'load_resources'),
    ('noah_v842', 'load_native_resources'),
    ('rrtmg_lw_v842', 'load_native_lw_resources'),
    ('rrtmg_sw_v842', 'load_shortwave_resources'),
}


def _files(value, key=''):
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _files(v, k)]
    if isinstance(value, (list, tuple)):
        return [p for v in value for p in _files(v, key)]
    # Cache/output directories contain derived scratch, not authenticated input.
    if key not in ('path', 'table_directory', 'auxiliary') or not isinstance(value, (str, Path)):
        return []
    path = Path(value).resolve()
    return sorted(p for p in path.iterdir() if p.is_file()) if path.is_dir() else [path]


def _identity(packet):
    result = []
    for p in sorted(set(_files(packet))):
        s = p.stat()
        result.append((str(p), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))
    return tuple(result)


def packet(path, reader):
    """Resolve an unchanged packet once; keep each session's metadata private."""
    path = Path(path).resolve()
    def identity(paths):
        rows = []
        for p in paths:
            try:
                s = p.stat()
                rows.append((str(p),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
            except FileNotFoundError:
                rows.append((str(p),None))
        return tuple(rows)
    with _LOCK:
        hit = _PACKETS.get(str(path))
        if hit is not None and identity(hit[0]) == hit[1]:
            return _session_copy(hit[2])
        value = reader(path)
        paths = [path, path.with_suffix('.receipt.json'), *_files(value)]
        paths = tuple(dict.fromkeys(paths))
        _PACKETS[str(path)] = (paths, identity(paths), value)
        return _session_copy(value)


def _session_copy(value):
    if isinstance(value, dict):
        return {k:_session_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_session_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_session_copy(v) for v in value)
    if (type(value).__module__ == 'woof.hex.mpasphys.rrtmg_sw_v842'
            and type(value).__name__ == 'MPASShortwave'):
        engine = copy(value)
        engine._shape_modules = dict(value._shape_modules)
        engine._table_streams = {}
        engine._scratch_by_stream = {}
        engine._chunk_by_nlayers = {}
        engine.max_nlay = None
        return engine
    return value  # Read-only arrays, table-ready events, modules and scalar metadata.


def load(module, name, packet, device, *, arguments=None):
    loader = getattr(module, name)
    arguments = {} if arguments is None else arguments
    if (module.__name__.rsplit('.', 1)[-1], name) not in _LOADERS:
        return loader(packet, device, **arguments)
    key = (module.__name__, name, json.dumps(packet, sort_keys=True, default=str),
           json.dumps(arguments, sort_keys=True, default=str),
           int(device.cuda.runtime.getDevice()), numerics.mode())
    with _LOCK:
        identity = _identity(packet)
        hit = _TABLES.get(key)
        if hit is None or hit[0] != identity:
            # Authenticate/decode through the unchanged loader. Publish only
            # once all initialization uploads/kernels on this stream finish.
            value = loader(packet, device, **arguments)
            device.cuda.get_current_stream().synchronize()
            after = _identity(packet)
            if after != identity:
                raise ValueError('physics table inputs changed while initializing resident tables')
            # A changed input replaces the old table instead of retaining it.
            _TABLES[key] = (identity, value)
        else:
            value = hit[1]
        return _session_copy(value)
