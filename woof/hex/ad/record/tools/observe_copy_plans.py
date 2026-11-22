#!/usr/bin/env python3
"""Observe RRFS copy-plan inputs while the ordinary forward computes them.

Only live input descriptors and a check digest are recorded; the assembled
host table is never retained as replay data. This is not a completed census.
"""
# SPDX-License-Identifier: Apache-2.0
import functools
import gzip
import hashlib
import json
import math
import struct
import sys
import threading
import time

import observe_cupti_outer as base
import observe_module_globals as globals_observer
import observe_seaice_pointers as seaice


def copy_plan_row(dst, src, dst_stride, src_stride, shape, fill, mode):
    dst, src, fill, mode = int(dst), int(src), int(fill), int(mode)
    shape, dst_stride, src_stride = tuple(map(int, shape)), tuple(map(int, dst_stride)), tuple(map(int, src_stride))
    if (len(shape) != 4 or len(dst_stride) != 4 or len(src_stride) != 4
            or any(x < 0 or x >= 1 << 63 for x in shape)
            or not 0 < dst < 1 << 64 or not 0 <= src < 1 << 64
            or not 0 <= fill < 1 << 32 or mode not in (0, 1)
            or (mode == 0 and src == 0) or (mode == 1 and src != 0)):
        raise ValueError('invalid copy-plan source descriptor')
    count = math.prod(shape)
    if count >= 1 << 63:
        raise ValueError('copy-plan element count exceeds int64')
    order = sorted(range(4), key=lambda k: (shape[k] > 1, -abs(dst_stride[k])))
    row = {'dst': dst, 'src': src, 'dst_stride': list(dst_stride),
           'src_stride': list(src_stride), 'shape': list(shape), 'fill': fill, 'mode': mode}
    encoded = struct.pack('<QQ4q4q4qqIi', dst, src,
        *(dst_stride[k] for k in order), *(src_stride[k] for k in order),
        *(shape[k] for k in order), count, fill, mode)
    return row, encoded


class CopyPlanObserver(globals_observer.GlobalObserver):
    def __init__(self, root):
        super().__init__(root)
        self.copy_output = gzip.open(root / 'copy-plan-inputs.jsonl.gz', 'wt',
                                     encoding='utf-8', compresslevel=1)

    def install(self):
        super().install()
        from woof.hex.mpasphys.rrfs.glue_plan import CopyPlan
        observer = self
        original_init, original_add, original_build = CopyPlan.__init__, CopyPlan._add, CopyPlan.build

        @functools.wraps(original_init)
        def init(plan, *args, **kwargs):
            original_init(plan, *args, **kwargs)
            plan._binder_input_rows = []

        @functools.wraps(original_add)
        def add(plan, dst, src, dst_stride, src_stride, shape, fill, mode):
            # The ordinary method remains responsible for producing its table.
            result = original_add(plan, dst, src, dst_stride, src_stride, shape, fill, mode)
            row, encoded = copy_plan_row(dst, src, dst_stride, src_stride, shape, fill, mode)
            plan._binder_input_rows.append((row, encoded))
            return result

        @functools.wraps(original_build)
        def build(plan):
            step = base.forecast_step()
            if step and plan._binder_input_rows:
                inputs = [row for row, _ in plan._binder_input_rows]
                digest = hashlib.sha256(b''.join(encoded for _, encoded in plan._binder_input_rows)).hexdigest()
                record = {'schema': 'ad.binder.phys-copy-plan-input/v1', 'complete': False,
                          'step': step, 'clock_ns': time.monotonic_ns(),
                          'thread_id': threading.get_native_id(), 'name': plan.name,
                          'bytes': len(inputs) * 128, 'sha256': digest, 'inputs': inputs}
                with observer.lock:
                    observer.copy_output.write(json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n')
                    observer.copy_output.flush()
            return original_build(plan)

        CopyPlan.__init__, CopyPlan._add, CopyPlan.build = init, add, build

    def close(self):
        try:
            self.copy_output.close()
        finally:
            super().close()


if __name__ == '__main__':
    base.Observer = CopyPlanObserver
    base.boundary_trace = seaice.boundary_species_and_seaice_trace
    sys.exit(base.main())
