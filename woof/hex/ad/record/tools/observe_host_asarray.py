#!/usr/bin/env python3
"""Identify live host-to-device upload producers in an ordinary forecast.

Only hashes, array descriptors, and source locations are retained; these
records check replay rules and cannot be used as substitute host outputs.
"""
# SPDX-License-Identifier: Apache-2.0
import functools
import gzip
import hashlib
import json
import sys
import threading
import time

import observe_cupti_outer as base
import observe_layer_groups as layer
import observe_seaice_pointers as seaice


def source_stack():
    result = []
    frame = sys._getframe(1)
    while frame and len(result) < 12:
        path = frame.f_code.co_filename
        if (('/hexcore/' in path and '/hexcore/ad/record/' not in path) or '/woof/' in path) and 'stochastic' not in path.lower():
            context = {}
            for key in ('name', 'pool', 'level', 'nlevels', 'index', 'stem'):
                value = frame.f_locals.get(key)
                if isinstance(value, (str, int, bool)) and len(str(value)) <= 128:
                    context[key] = value
            result.append({'path': path, 'line': frame.f_lineno,
                           'function': frame.f_code.co_name, 'context': context})
        frame = frame.f_back
    return result


class HostAsarrayObserver(layer.GroupObserver):
    def __init__(self, root):
        super().__init__(root)
        self.host_asarray_output = gzip.open(root / 'host-asarray.jsonl.gz', 'wt',
                                              encoding='utf-8', compresslevel=1)

    def install(self):
        super().install()
        import cupy as cp
        import numpy as np
        install_upload_hooks(cp, np, self)

    def close(self):
        try:
            self.host_asarray_output.close()
        finally:
            super().close()


#: The host-to-device constructors whose call site names an upload: (name, position of `order` in the positional
#: arguments after the value, host types taken). cp.array takes a host ndarray only: its list or tuple form can hold
#: device arrays and stack them, and a stacked pointer table must keep no site (the sealer's cupy_stack_same_size
#: rule). THE BREAKAGE THIS PREVENTS: an upload made by cp.array (resident_output_diagnostics_v842 current_qv, the
#: first output step's moisture slot) had no site, so the sealer could not name its rule and the 12 km release example
#: refused at seal (one 'upload_needs_live_rule' word, step 15; port rowrec r2, 2026-11-16).
UPLOAD_CONSTRUCTORS = (('asarray', 1, 'any'), ('array', 2, 'ndarray'))


def install_upload_hooks(cp, np, observer):
    for name, order_at, takes in UPLOAD_CONSTRUCTORS:
        setattr(cp, name, observe_upload(getattr(cp, name), np, observer, order_at, takes))


def observe_upload(original, np, observer, order_at, takes):
    host_types = (np.ndarray, np.generic) if takes == 'ndarray' else (np.ndarray, list, tuple, int, float, bool, np.generic)

    @functools.wraps(original)
    def observed(value, *args, **kwargs):
            step = base.forecast_step()
            if type(value).__module__.startswith(('woof.hex.stochastic', 'gpuwm.stochastic')):
                return original(value, *args, **kwargs)
            if not step or not isinstance(value, host_types):
                return original(value, *args, **kwargs)
            dtype = kwargs.get('dtype', args[0] if args else None)
            order = kwargs.get('order', args[order_at] if len(args) > order_at else None)
            record = {'schema': 'ad.binder.phys-host-asarray-observation/v1',
                      'complete': False, 'step': step, 'thread_id': threading.get_native_id(),
                      'stack': source_stack()}
            try:
                check = np.asarray(value, dtype=dtype, order=order)
                if not check.dtype.hasobject:
                    record['input'] = {'shape': list(check.shape), 'dtype': check.dtype.str,
                        'strides': list(check.strides), 'bytes': int(check.nbytes),
                        'address': int(check.__array_interface__['data'][0]),
                        'sha256_c': hashlib.sha256(check.tobytes(order='C')).hexdigest(),
                        'sha256_f': hashlib.sha256(check.tobytes(order='F')).hexdigest()}
            except (TypeError, ValueError, OverflowError):
                record['input_error'] = 'host check unavailable'
            record['enter_ns'] = time.monotonic_ns()
            try:
                result = original(value, *args, **kwargs)
            except BaseException:
                record['exit_ns'] = time.monotonic_ns()
                record['success'] = False
                with observer.lock:
                    observer.host_asarray_output.write(json.dumps(record, separators=(',', ':')) + '\n')
                    observer.host_asarray_output.flush()
                raise
            record['exit_ns'] = time.monotonic_ns()
            record['success'] = True
            record['output'] = {'address': int(result.data.ptr), 'bytes': int(result.nbytes),
                                'shape': list(result.shape), 'dtype': result.dtype.str}
            with observer.lock:
                observer.host_asarray_output.write(json.dumps(record, separators=(',', ':')) + '\n')
                observer.host_asarray_output.flush()
            return result

    return observed


if __name__ == '__main__':
    base.Observer = HostAsarrayObserver
    base.boundary_trace = seaice.boundary_species_and_seaice_trace
    sys.exit(base.main())
