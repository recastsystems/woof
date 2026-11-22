#!/usr/bin/env python3
"""Observe CuPy stack inputs; pointer hashes are checks, never replay inputs."""
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


class StackObserver(ruc.RucObserver):
    def __init__(self, root):
        super().__init__(root)
        self.stack_output = gzip.open(root / 'stack-inputs.jsonl.gz', 'wt',
                                      encoding='utf-8', compresslevel=1)

    def install(self):
        super().install()
        import cupy as cp
        original = cp.stack
        observer = self

        @functools.wraps(original)
        def observed(arrays, *args, **kwargs):
            step = base.forecast_step()
            if not step or not isinstance(arrays, (list, tuple)) or not all(
                    isinstance(array, cp.ndarray) for array in arrays):
                return original(arrays, *args, **kwargs)
            addresses = [int(array.data.ptr) for array in arrays]
            digest = hashlib.sha256(b''.join(struct.pack('<Q', address) for address in addresses)).hexdigest()
            row = {'schema': 'ad.binder.phys-stack-inputs/v1', 'complete': False,
                   'step': step, 'thread_id': threading.get_native_id(),
                   'stack': host.source_stack(),
                   'axis': int(kwargs.get('axis', args[0] if args else 0)),
                   'inputs': [observer.array_metadata(array) for array in arrays],
                   'pointer_bytes': len(addresses) * 8, 'pointer_sha256': digest,
                   'enter_ns': time.monotonic_ns()}
            try:
                result = original(arrays, *args, **kwargs)
            except BaseException:
                row['exit_ns'] = time.monotonic_ns()
                row['success'] = False
                with observer.lock:
                    observer.stack_output.write(json.dumps(row, separators=(',', ':')) + '\n')
                    observer.stack_output.flush()
                raise
            row['exit_ns'] = time.monotonic_ns()
            row['success'] = True
            row['output'] = observer.array_metadata(result)
            with observer.lock:
                observer.stack_output.write(json.dumps(row, separators=(',', ':')) + '\n')
                observer.stack_output.flush()
            return result

        cp.stack = observed

    def close(self):
        try:
            self.stack_output.close()
        finally:
            super().close()


if __name__ == '__main__':
    host.source_stack = ruc.source_stack_with_radiation_counts
    base.Observer = StackObserver
    base.boundary_trace = ruc.boundary_ruc_and_existing_trace
    sys.exit(base.main())
