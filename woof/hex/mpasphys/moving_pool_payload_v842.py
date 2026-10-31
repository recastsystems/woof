"""Resident cold-pool projection for admitted classes of a moving segment.

The global transport is validated normally. Each local field is projected by
the original word-copy routine into private native storage, on demand. No
class writes, hashes and reads a temporary binary just to hydrate itself.
"""
from pathlib import Path
import json
import threading

_LOCK = threading.RLock()
_PAYLOADS = {}


def lookup(path):
    with _LOCK:
        return _PAYLOADS.get(str(Path(path).resolve()))


def clear():
    with _LOCK:
        _PAYLOADS.clear()


class _Source:
    def __init__(self, hydration):
        self.hydration = hydration
        from .moving_class_gpu_v842 import private_pool_mapping
        self.hydration.copy_on_write = private_pool_mapping()
        self.buffers = {}
        self.lock = threading.Lock()

    def field(self, row):
        key = row['pool'], row['name']
        with self.lock:
            value = self.buffers.get(key)
            if value is None:
                value = self.hydration.field_buffer(*key)
                value.flags.writeable = False
                self.buffers[key] = value
            return value


class _Payload:
    def __init__(self, source, global_layout, maps, local):
        self.source, self.global_layout, self.maps = source, global_layout, maps
        self.rows = {(r['pool'], r['name']): r for r in global_layout.fields}
        records = []
        for row in local.fields:
            original = source.hydration.records[row['pool'], row['name']]
            record = {key: row[key] for key in ('pool', 'name', 'kind', 'dimensions', 'shape')}
            record['dimensions'], record['shape'] = list(row['dimensions']), list(row['shape'])
            for key in ('fill', 'text', 'origins'):
                if key in original:
                    record[key] = original[key]
            # These are trusted in-process projections, not a file transport.
            record.update(payload=None, sha256=None)
            records.append(record)
        self.manifest = dict(schema='mpas-native-pool-payload-v1', fields=records)

    def field(self, pool, name):
        from .partition_v842 import _project_buffer
        row = self.rows[pool, name]
        return _project_buffer(self.source.field(row), row, self.global_layout, self.maps)


def project(layout, payload, partitions, outputs):
    """Return ordinary local layouts backed by process-owned projected words."""
    from .pool_v842 import NativePoolHydration
    from .partition_v842 import _maps, _partition_layout
    source = _Source(NativePoolHydration(None, layout, payload))
    result = []
    for partition, output in zip(partitions, outputs):
        maps = _maps(layout, partition)
        local = _partition_layout(layout, maps)
        memory = _Payload(source, layout, maps, local)
        output.mkdir(parents=True, exist_ok=False)
        # Small provenance stays on disk. Binary fields exist in this process
        # only, and are never claimed as a portable cold-start transport.
        (output/'resident-payload.json').write_text(json.dumps(dict(
            schema='hex-moving-resident-pool-v1', source=str(Path(payload).resolve()),
            dimensions=local.dimensions, fields=len(local.fields)), indent=2)+'\n')
        with _LOCK:
            _PAYLOADS[str(output.resolve())] = memory
        result.append(local)
    return result
