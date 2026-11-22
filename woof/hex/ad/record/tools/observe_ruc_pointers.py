#!/usr/bin/env python3
"""Observe RUC pointer inputs and radiation draw counts for checking only."""
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
import observe_host_asarray as host
import observe_seaice_pointers as seaice


def source_stack_with_radiation_counts():
    result = []
    frame = sys._getframe(1)
    while frame and len(result) < 12:
        path = frame.f_code.co_filename
        if (('/hexcore/' in path and '/hexcore/ad/record/' not in path) or '/woof/' in path) and 'stochastic' not in path.lower():
            context = {}
            for key in ('name', 'pool', 'level', 'nlevels', 'index', 'stem',
                        'draws', 'gpoints', 'nl', 'cldovrlp'):
                value = frame.f_locals.get(key)
                if isinstance(value, (str, int, bool)) and len(str(value)) <= 128:
                    context[key] = value
            result.append({'path': path, 'line': frame.f_lineno,
                           'function': frame.f_code.co_name, 'context': context})
        frame = frame.f_back
    return result


class RucObserver(host.HostAsarrayObserver):
    def __init__(self, root):
        super().__init__(root)
        self.ruc_output = gzip.open(root / 'ruc-pointers.jsonl.gz', 'wt',
                                    encoding='utf-8', compresslevel=1)

    def record_ruc(self, frame):
        local = frame.f_locals
        owner = local['self']
        layout = local['layout']
        arguments = local['arguments']
        optional = [field for field in layout if field['optional']]
        if len(optional) != len(owner.presence):
            raise RuntimeError('RUC optional flag count differs from layout')
        fields = [{'name': field['name'], 'type': field['type'],
                   'optional': bool(field['optional']),
                   'view': self.array_metadata(arguments[field['name']])}
                  for field in layout]
        flags = [{'name': field['name'], 'view': self.array_metadata(array)}
                 for field, array in zip(optional, owner.presence)]
        values = [int(value) for value in local['pointers']]
        payload = b''.join(struct.pack('<Q', value) for value in values)
        row = {'schema': 'ad.binder.phys-ruc-pointers/v1', 'complete': False,
               'entry': local['entry'], 'step': base.forecast_step(),
               'source': frame.f_code.co_filename, 'line': frame.f_lineno,
               'clock_ns': time.monotonic_ns(), 'thread_id': threading.get_native_id(),
               'n': int(local['n']), 'fields': fields, 'flags': flags,
               'tile_stop': (None if owner.tile_stop is None else self.array_metadata(owner.tile_stop)),
               'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        with self.lock:
            self.ruc_output.write(json.dumps(row, separators=(',', ':')) + '\n')
            self.ruc_output.flush()

    def close(self):
        try:
            self.ruc_output.close()
        finally:
            super().close()


def boundary_ruc_and_existing_trace(observer):
    prior = seaice.boundary_species_and_seaice_trace(observer)
    def global_trace(frame, event, arg):
        if (event == 'call' and frame.f_code.co_name == '__init__' and
                frame.f_code.co_filename.endswith('/hexcore/mpasphys/rrfs/resident_ruc_rrfs.py')):
            lines = linecache.getlines(frame.f_code.co_filename)
            target = [i + 1 for i, line in enumerate(lines)
                      if i + 1 >= frame.f_code.co_firstlineno and
                      'self.ptrs = cp.asarray(pointers, dtype=cp.uint64)' in line]
            if len(target) == 1:
                def local(current, kind, value):
                    if kind == 'line' and current.f_lineno == target[0]:
                        observer.record_ruc(current)
                    return local
                return local
        return prior(frame, event, arg)
    return global_trace


if __name__ == '__main__':
    host.source_stack = source_stack_with_radiation_counts
    base.Observer = RucObserver
    base.boundary_trace = boundary_ruc_and_existing_trace
    sys.exit(base.main())
