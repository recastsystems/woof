#!/usr/bin/env python3
"""Check radiation grouping with live inputs from an ordinary forward run.

The extra device reads deliberately make this an observation run, not a
reference command stream. Recorded values are for equality checking only and
must never be substituted for inputs during native replay.
"""
# SPDX-License-Identifier: Apache-2.0
import functools
import gzip
import json
import sys
import threading
import time

import observe_copy_plans as copies
import observe_cupti_outer as base
import observe_seaice_pointers as seaice


class GroupObserver(copies.CopyPlanObserver):
    def __init__(self, root):
        super().__init__(root)
        self.group_output = gzip.open(root / 'layer-groups.jsonl.gz', 'wt',
                                      encoding='utf-8', compresslevel=1)
        self.chunk_output = gzip.open(root / 'layer-chunks.jsonl.gz', 'wt',
                                      encoding='utf-8', compresslevel=1)

    def install(self):
        super().install()
        import woof.hex.mpasphys.rrtmg_lw_v842 as lw
        original = lw._layer_groups
        original_chunk = lw._lw_wrapper_chunk
        observer = self

        @functools.wraps(original)
        def observed(counts, nm, *, need_members=False):
            step = base.forecast_step()
            # Only check data: retain the original counts allocation and call.
            values = counts.get().tolist()
            groups = original(counts, nm, need_members=need_members)
            checked = []
            for layers, indices, members in groups:
                checked.append({'layers': int(layers),
                                'indices': indices.get().tolist(),
                                'members': members})
            row = {'schema': 'ad.binder.phys-layer-groups-observation/v1',
                   'complete': False, 'step': step, 'nm': int(nm),
                   'need_members': bool(need_members), 'counts': values,
                   'groups': checked, 'clock_ns': time.monotonic_ns(),
                   'thread_id': threading.get_native_id()}
            with observer.lock:
                observer.group_output.write(json.dumps(row, separators=(',', ':')) + '\n')
                observer.group_output.flush()
            return groups

        lw._layer_groups = observed

        @functools.wraps(original_chunk)
        def observed_chunk(ncol, nlayers):
            import cupy as cp
            original_info = cp.cuda.runtime.memGetInfo
            thread_id = threading.get_native_id()
            calls = []

            def checked_info():
                value = original_info()
                if threading.get_native_id() == thread_id:
                    calls.append(tuple(map(int, value)))
                return value

            cp.cuda.runtime.memGetInfo = checked_info
            try:
                result = original_chunk(ncol, nlayers)
            finally:
                cp.cuda.runtime.memGetInfo = original_info
            row = {'schema': 'ad.binder.phys-layer-chunk-observation/v1',
                   'complete': False, 'step': base.forecast_step(),
                   'ncol': int(ncol), 'nlayers': int(nlayers),
                   'memory_info_calls': calls, 'result': int(result)}
            with observer.lock:
                observer.chunk_output.write(json.dumps(row, separators=(',', ':')) + '\n')
                observer.chunk_output.flush()
            return result

        lw._lw_wrapper_chunk = observed_chunk

    def close(self):
        try:
            try:
                self.group_output.close()
            finally:
                self.chunk_output.close()
        finally:
            super().close()


if __name__ == '__main__':
    base.Observer = GroupObserver
    base.boundary_trace = seaice.boundary_species_and_seaice_trace
    sys.exit(base.main())
