"""Replay full real CopyPlans at 81 times their column count, with bounded memory."""
import json
import os
import statistics
import numpy as np


def run(plan, old, gp, cp, step, arm=''):
    replica = gp.CopyPlan(plan.name + ' 81x columns')
    checks = []
    logical_bytes = 0
    for row in plan._rows:
        shape = tuple(int(x) for x in row['shape'])
        reps = [1] * 4
        # These real plans have one horizontal axis. Soil/level/species axes
        # are small; scalar flags stay scalar.
        axis = int(np.argmax(shape))
        if shape[axis] >= 1000:
            reps[axis] = 81
        expanded_shape = tuple(n * r for n, r in zip(shape, reps))
        dst = cp.empty(expanded_shape, cp.uint32)
        if row['mode']:
            source = np.uint32(row['fill'])
            replica.fill(dst, source)
            logical_bytes += dst.nbytes
        else:
            owner = next(a for a in plan._keep if a.data.ptr == int(row['src']))
            view = cp.ndarray(shape, dtype=cp.uint32, memptr=owner.data,
                              strides=tuple(int(s)*4 for s in row['src_stride']))
            source = cp.tile(view, reps)
            if row['src_stride'][2] == 1 and row['dst_stride'][3] == 1 and shape[:2] == (1, 1):
                source = cp.asfortranarray(source)
            replica.copy(dst, source)
            logical_bytes += dst.nbytes * 2
        checks.append((dst, source))
    replica.build()
    def verify():
        return sum(int(cp.count_nonzero(dst != src).get()) for dst, src in checks)
    def measure(kernel):
        values = []
        for _ in range(9):
            a, b = cp.cuda.Event(), cp.cuda.Event()
            a.record(); kernel(replica._grid, (gp._THREADS,), (replica._device,)); b.record(); b.synchronize()
            values.append(cp.cuda.get_elapsed_time(a, b))
        return statistics.median(values)
    old(replica._grid, (gp._THREADS,), (replica._device,))
    base_different = verify()
    tip = gp._kernel(cp, replica._tiled)
    tip(replica._grid, (gp._THREADS,), (replica._device,))
    tip_different = verify()
    assert base_different == tip_different == 0
    result = dict(plan=plan.name, arm=arm, step=step, scale=81, words=sum(d.size for d, _ in checks),
                  base_different=base_different, tip_different=tip_different,
                  bytes=logical_bytes, base_ms=measure(old), tip_ms=measure(tip))
    with open(os.environ['GLUE_AB_OUT'] + '.full-tiled.jsonl', 'a') as stream:
        stream.write(json.dumps(result) + '\n')
