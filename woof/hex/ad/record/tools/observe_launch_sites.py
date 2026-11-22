#!/usr/bin/env python3
"""Observe ordinary RawKernel caller sites and scalar-name candidates.

The locations and value-equality candidates are checks for later source
analysis, never launch inputs or proof of a live replay rule.  Do not inspect
the held stochastic implementation or retain host scalar values here.
"""
# SPDX-License-Identifier: Apache-2.0
import gzip
import json
import sys
import threading
import time
import warnings

import numpy as np

import observe_cupti_outer as base
import observe_host_asarray as host
import observe_ruc_pointers as ruc
import observe_tempo_pointers as tempo


def sites(args, frame):
    """Find source frames and names whose scalar values match call arguments.

    Equality is only a candidate association: different live variables may
    have the same value. The host code and the owner's lifetime still need an
    independent proof before the resulting rule can be used in replay.
    """
    targets = {}
    for index, arg in enumerate(args):
        if isinstance(arg, (np.generic, int, float, bool)):
            value = np.asarray(arg)
            if value.dtype.kind in 'biuf' and value.ndim == 0:
                targets.setdefault(value.dtype.str, {}).setdefault(value.tobytes(), []).append(index)
    result = []
    excluded = 0
    while frame and len(result) < 12:
        path = frame.f_code.co_filename
        if 'stochastic' in path.lower():
            excluded += 1
        elif ('/hexcore/' in path and '/hexcore/ad/record/' not in path) or '/woof/' in path:
            candidates = {}
            for name, value in frame.f_locals.items():
                if (any(part in name.lower() for part in ('password', 'secret', 'credential', 'token'))
                        or not isinstance(value, (np.integer, np.floating, np.bool_, int, float, bool))):
                    continue
                for dtype, by_value in targets.items():
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            warnings.simplefilter('ignore', DeprecationWarning)
                            encoded = np.asarray(value, dtype=np.dtype(dtype)).tobytes()
                    except (TypeError, ValueError, OverflowError):
                        continue
                    for index in by_value.get(encoded, ()):
                        candidates.setdefault(str(index), []).append(name)
            result.append({'path': path, 'line': frame.f_lineno,
                           'function': frame.f_code.co_name, 'scalar_candidates': candidates})
        frame = frame.f_back
    return result, excluded


class LaunchSiteObserver(tempo.TempoObserver):
    def __init__(self, root):
        super().__init__(root)
        self.site_output = gzip.open(root / 'launch-sites.jsonl.gz', 'wt',
                                     encoding='utf-8', compresslevel=1)

    def record(self, cp, name, image, grid, block, args, kwargs):
        step = base.forecast_step()
        if not step:
            return super().record(cp, name, image, grid, block, args, kwargs)
        with self.lock:
            caller_sites, excluded = sites(args, sys._getframe(1))
            super().record(cp, name, image, grid, block, args, kwargs)
            row = {'schema': 'ad.binder.phys-launch-site-observation/v1', 'complete': False,
                   'step': step, 'ordinal': self.count, 'thread_id': threading.get_native_id(),
                   'clock_ns': time.monotonic_ns(), 'kernel': name, 'sites': caller_sites,
                   'held_frames_excluded': excluded}
            self.site_output.write(json.dumps(row, separators=(',', ':')) + '\n')
            if self.count % 64 == 0:
                self.site_output.flush()

    def close(self):
        try:
            self.site_output.close()
        finally:
            super().close()


if __name__ == '__main__':
    host.source_stack = ruc.source_stack_with_radiation_counts
    base.Observer = LaunchSiteObserver
    base.boundary_trace = ruc.boundary_ruc_and_existing_trace
    sys.exit(base.main())
