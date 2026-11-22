#!/usr/bin/env python3
"""Observe the reference's live species-pointer expression without storing its output.

The six-step observation checks the named source rule and allocation identities;
it does not classify other host work or prove the complete physics replay.
"""
# SPDX-License-Identifier: Apache-2.0
import gzip
import hashlib
import json
import linecache
import struct
import sys
import threading
import time

import observe_callbacks_v2 as callbacks
import observe_cupti_outer as base


def species_record(frame, observer, step):
    owner = frame.f_locals['self']
    species = frame.f_locals['species']
    names = list(owner.names)
    if not names or len(names) != len(set(names)) or not all(isinstance(name, str) for name in names):
        raise ValueError('invalid species names at pointer-table source')
    entries = []
    words = bytearray()
    for name in names:
        array = species[name]
        address = int(array.data.ptr)
        if not 0 < address < (1 << 64):
            raise ValueError('invalid live species device pointer')
        words.extend(struct.pack('<Q', address))
        entries.append({'name': name, 'address': address, 'view': observer.array_metadata(array)})
    return {'schema': 'ad.binder.phys-species-pointers/v1', 'complete': False,
            'rule': 'resident_v842_species', 'step': int(step),
            'clock_ns': time.monotonic_ns(), 'thread_id': threading.get_native_id(),
            'source': frame.f_code.co_filename, 'line': frame.f_lineno,
            'bytes': len(words), 'sha256': hashlib.sha256(words).hexdigest(), 'entries': entries}


class PointerObserver(callbacks.CallbackObserverV2):
    def __init__(self, root):
        super().__init__(root)
        self.pointer_output = gzip.open(root / 'species-pointers.jsonl.gz', 'wt',
                                        encoding='utf-8', compresslevel=1)

    def record_species(self, frame):
        parent = frame
        while parent and not (parent.f_code.co_name == '_run_native_dynamics' and
                              parent.f_code.co_filename.endswith('/hexcore/mpasphys/runtime_v842.py')):
            parent = parent.f_back
        if parent is None or not isinstance(parent.f_locals.get('step'), int):
            raise RuntimeError('species pointer expression has no forecast step')
        row = species_record(frame, self, parent.f_locals['step'])
        with self.lock:
            self.pointer_output.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
            self.pointer_output.flush()

    def close(self):
        try:
            self.pointer_output.close()
        finally:
            super().close()


original_boundary_trace = base.boundary_trace


def boundary_and_species_trace(observer):
    boundary = original_boundary_trace(observer)
    def global_trace(frame, event, arg):
        if (event == 'call' and frame.f_code.co_name == 'prepare' and
                frame.f_code.co_filename.endswith('/hexcore/mpasphys/resident_v842.py')):
            lines = linecache.getlines(frame.f_code.co_filename)
            target = [i + 1 for i, line in enumerate(lines)
                      if i + 1 >= frame.f_code.co_firstlineno and
                      'pointers = cp.asarray(np.array([species[name].data.ptr for name in self.names]' in line]
            if len(target) != 1:
                raise RuntimeError('species pointer source expression is missing or ambiguous')
            def local(current, kind, value):
                if kind == 'line' and current.f_lineno == target[0]:
                    observer.record_species(current)
                return local
            return local
        return boundary(frame, event, arg)
    return global_trace


if __name__ == '__main__':
    base.Observer = PointerObserver
    base.boundary_trace = boundary_and_species_trace
    sys.exit(base.main())
