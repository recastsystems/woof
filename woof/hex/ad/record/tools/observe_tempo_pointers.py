#!/usr/bin/env python3
"""Observe TEMPO pointer table inputs during an ordinary forecast.

These descriptors and hashes are checks only. The native replay must obtain
all array views and optional-presence decisions from its own live state.
"""
# SPDX-License-Identifier: Apache-2.0
import functools
import gzip
import hashlib
import json
import struct
import sys
import threading
import time

import observe_cupti_outer as base
import observe_host_asarray as host
import observe_ruc_pointers as ruc
import observe_stack_pointers as stack


class TempoObserver(stack.StackObserver):
    def __init__(self, root):
        super().__init__(root)
        self.tempo_output = gzip.open(root / 'tempo-pointers.jsonl.gz', 'wt',
                                      encoding='utf-8', compresslevel=1)

    def install(self):
        super().install()
        from woof.hex.mpasphys.rrfs import tempo_v303 as tempo
        original, observer = tempo._pointer_array, self

        @functools.wraps(original)
        def observed(arrays, *, shape, device, optional=False):
            step = base.forecast_step()
            if not step:
                return original(arrays, shape=shape, device=device, optional=optional)
            words = [0 if array is None else int(array.data.ptr) for array in arrays]
            payload = b''.join(struct.pack('<Q', word) for word in words)
            row = {'schema': 'ad.binder.phys-tempo-pointers/v1', 'complete': False,
                   'step': step, 'thread_id': threading.get_native_id(),
                   'stack': host.source_stack(), 'shape': None if shape is None else list(shape),
                   'device': int(device), 'optional': bool(optional),
                   'inputs': [None if array is None else observer.array_metadata(array) for array in arrays],
                   'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
                   'enter_ns': time.monotonic_ns()}
            try:
                result = original(arrays, shape=shape, device=device, optional=optional)
            except BaseException:
                row['exit_ns'] = time.monotonic_ns()
                row['success'] = False
                with observer.lock:
                    observer.tempo_output.write(json.dumps(row, separators=(',', ':')) + '\n')
                    observer.tempo_output.flush()
                raise
            row['exit_ns'] = time.monotonic_ns()
            row['success'] = True
            row['output'] = observer.array_metadata(result)
            with observer.lock:
                observer.tempo_output.write(json.dumps(row, separators=(',', ':')) + '\n')
                observer.tempo_output.flush()
            return result

        tempo._pointer_array = observed

    def close(self):
        try:
            self.tempo_output.close()
        finally:
            super().close()


if __name__ == '__main__':
    host.source_stack = ruc.source_stack_with_radiation_counts
    base.Observer = TempoObserver
    base.boundary_trace = ruc.boundary_ruc_and_existing_trace
    sys.exit(base.main())
