#!/usr/bin/env python3
"""Observe live sea-ice pointer inputs for comparison, never as replay data."""
# SPDX-License-Identifier: Apache-2.0
import gzip
import hashlib
import json
import linecache
import struct
import sys
import threading
import time

import observe_cupti_outer as base
import observe_species_pointers as species


def seaice_record(frame, observer, step):
    arrays, dummy = frame.f_locals['arrays'], frame.f_locals['dummy']
    names = frame.f_globals['FIELDS']
    flux = frame.f_globals['_SEA_FLUX_FIELDS']
    if (type(arrays) is not dict or not names or len(names) != len(set(names)) or
            not set(flux).issubset(names)):
        raise ValueError('invalid seaice pointer inputs')
    entries, words = [], bytearray()
    for name in names:
        if name in arrays:
            kind, array = 'live', arrays[name]
        elif name in flux:
            kind, array = 'null', None
        else:
            kind, array = 'dummy', dummy
        address = 0 if array is None else int(array.data.ptr)
        if not 0 <= address < (1 << 64) or (array is not None and address == 0):
            raise ValueError('invalid seaice device pointer')
        words.extend(struct.pack('<Q', address))
        entries.append({'name': name, 'kind': kind, 'address': address,
                        'view': None if array is None else observer.array_metadata(array)})
    return {'schema': 'ad.binder.phys-seaice-pointers/v1', 'complete': False,
            'rule': 'seaice_v842_pointers', 'step': int(step),
            'clock_ns': time.monotonic_ns(), 'thread_id': threading.get_native_id(),
            'source': frame.f_code.co_filename, 'line': frame.f_lineno,
            'present': [name for name in names if name in arrays],
            'bytes': len(words), 'sha256': hashlib.sha256(words).hexdigest(), 'entries': entries}


class SeaiceObserver(species.PointerObserver):
    def __init__(self, root):
        super().__init__(root)
        self.seaice_output = gzip.open(root / 'seaice-pointers.jsonl.gz', 'wt',
                                       encoding='utf-8', compresslevel=1)

    def record_seaice(self, frame):
        parent = frame
        while parent and not (parent.f_code.co_name == '_run_native_dynamics' and
                              parent.f_code.co_filename.endswith('/hexcore/mpasphys/runtime_v842.py')):
            parent = parent.f_back
        if parent is None or not isinstance(parent.f_locals.get('step'), int):
            raise RuntimeError('seaice pointer expression has no forecast step')
        row = seaice_record(frame, self, parent.f_locals['step'])
        with self.lock:
            self.seaice_output.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
            self.seaice_output.flush()

    def close(self):
        try:
            self.seaice_output.close()
        finally:
            super().close()


def boundary_species_and_seaice_trace(observer):
    prior = species.boundary_and_species_trace(observer)
    def global_trace(frame, event, arg):
        if (event == 'call' and frame.f_code.co_name == '_pointers' and
                frame.f_code.co_filename.endswith('/hexcore/mpasphys/seaice_v842.py')):
            lines = linecache.getlines(frame.f_code.co_filename)
            target = [i + 1 for i, line in enumerate(lines)
                      if i + 1 >= frame.f_code.co_firstlineno and
                      'pointers = cp.asarray(np.asarray([arrays[name].data.ptr if name in arrays else' in line]
            if len(target) != 1:
                raise RuntimeError('seaice pointer expression is missing or ambiguous')
            seen = False
            def local(current, kind, value):
                nonlocal seen
                if not seen and kind == 'line' and current.f_lineno == target[0]:
                    observer.record_seaice(current)
                    seen = True
                return local
            return local
        return prior(frame, event, arg)
    return global_trace


if __name__ == '__main__':
    base.Observer = SeaiceObserver
    base.boundary_trace = boundary_species_and_seaice_trace
    sys.exit(base.main())
