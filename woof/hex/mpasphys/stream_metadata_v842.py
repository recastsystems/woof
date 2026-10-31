"""MPAS output metadata, independent of native storage and Fortran libraries.

MPAS v8.4.2 mpas_subdriver.F:500-558 defines global attribute order/types;
mpas_bootstrapping.F:287-378 defines mesh lineage and periodicity.
"""
from collections import deque
from time import time

import numpy as np

from ..streams_v842 import field_definition


# The pinned strict build's Registry preprocessing input, also recorded in
# tools/native_dry_reference/evidence/match/logs-g1/build-trace-final.log:48.
PINNED_GIT_VERSION = 'v8.4.2-dirty'


def global_attributes(layout, input_header, *, configs=None, git_version=PINNED_GIT_VERSION):
    """Port mpas_subdriver.F:513-558 without importing forecast output headers.

    Numeric attributes have explicit native RKIND/int32 types. Configs follow
    Registry allocation order, including options absent from the namelist.
    Only the input mesh supplies geometry/lineage; arbitrary input globals are
    not forwarded to output. Single-process route, as in the pool backend.
    """
    source = input_header['global_attrs']
    configs = layout.configs if configs is None else configs

    def value(name, default):
        item = source.get(name, default)
        if isinstance(item, dict):
            item = item['values']
            if isinstance(item, list):
                item, = item
        return item

    def real(item):
        return dict(dtype='f4', values=float(np.float32(item)))

    def text(item):
        # MPAS StrKIND=512; Fortran trim removes blanks, not line breaks.
        return str(item)[:512].rstrip(' ')

    root = layout.root.attrib
    on_sphere = 'YES' in value('on_a_sphere', 'YES')
    periodic = not on_sphere and 'YES' in value('is_periodic', 'NO')
    parent = text(value('parent_id', ''))
    file_id = text(value('file_id', ''))
    if file_id:
        parent = text(file_id + '\n' + parent)
    attrs = dict(model_name=root['model'], core_name=root['core'],
        version=root['version'], source='MPAS', Conventions='MPAS',
        git_version=git_version, on_a_sphere='YES' if on_sphere else 'NO',
        sphere_radius=real(value('sphere_radius', 1)),
        is_periodic='YES' if periodic else 'NO',
        x_period=real(value('x_period', 0) if periodic else 0),
        y_period=real(value('y_period', 0) if periodic else 0),
        history=f'mpirun -n 1 ./{root["core"]}_model', parent_id=parent,
        mesh_spec=text(value('mesh_spec', '0.0')))
    for name, kind in layout.config_kinds.items():
        item = configs[name]
        attrs[name] = (real(item) if kind == 'real' else
                       dict(dtype='i4', values=int(item)) if kind == 'integer' else
                       ('YES' if item else 'NO') if kind == 'logical' else text(item))
    return attrs


def input_field_attributes(layout, header, *, field_definition=field_definition):
    """mpas_io_streams.F:4315-4363 updates only allocated Registry attributes.

    Retain Registry order. Input files may override units, descriptions and
    missing values; unrelated input attributes do not become allocated attrs.
    Time's epoch is updated from the run clock by the writer. ``field_definition``
    is the route's Registry (route_table.StreamContract); the default is v8.4.2's.
    """
    names = [name for row in layout.fields for name in (row['components'] or [row['name']])]
    result = {}
    for name in names:
        if name not in header['variables']:
            continue
        known = field_definition(name)['attrs']
        incoming = header['variables'][name]['attrs']
        result[name] = {key: incoming[key] for key in known if key in incoming}
    return result


class NativeFileIds:
    """Isolated GNU/Linux rand() words used by MPAS random_id.c:20-47.

    MPAS seeds once at stream-manager initialization from wall-clock seconds
    (mpas_stream_manager.F:143-144). Retain the seed as evidence, without
    modifying process-global libc RNG state. Independently timed native runs
    have different IDs; this does not make archived full hashes reproducible.
    """
    def __init__(self, seed=None):
        self.seed = int(time()) if seed is None else int(seed)
        first = (self.seed & 0xffffffff) or 1
        words = [first]
        for _ in range(30):
            previous = words[-1]
            if previous >= 0x80000000:
                previous -= 0x100000000
            words.append((16807 * previous) % 2147483647)
        words.extend(words[:3])
        for i in range(34, 344):
            words.append((words[i-31] + words[i-3]) & 0xffffffff)
        self._words = deque(words[-31:], maxlen=31)

    def next(self):
        result = ''
        for _ in range(10):
            word = (self._words[0] + self._words[-3]) & 0xffffffff
            self._words.append(word)
            result += '0123456789abcdefghijklmnopqrstuvwxyz'[(word >> 1) % 36]
        return result
