"""Real-call CopyPlan A/B: base source versus tip, all destination words.

Reuses the existing glue A/B word comparator. All synchronization and CUDA
events are confined to this proof launcher, never the production path.
"""
import atexit
import contextlib
import json
import os
import runpy
import statistics
import sys

from ab_launch import _compare


def install():
    import cupy as cp
    import subprocess
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.used,utilization.gpu',
                          '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    neighbors = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    with open(os.environ['GLUE_AB_OUT'] + '.gpu.json', 'w') as stream:
        json.dump(dict(gpu=gpu.stdout, gpu_error=gpu.stderr,
                       processes=neighbors.stdout, process_error=neighbors.stderr), stream, indent=2)
    from woof.hex.mpasphys.rrfs import glue_plan as gp
    from woof.hex.mpasphys import runtime_v842 as rt
    base = runpy.run_path(os.environ['GLUE_BASE_SOURCE'])
    old = cp.RawModule(code=base['_SOURCE'], options=('--std=c++17',)).get_function('rrfs_glue_copy_plan')
    original = gp.CopyPlan.__call__
    stage_original = rt.timed_stage
    step = 0
    steps = {1, 12, 24}
    timings = []
    tiled = False

    @contextlib.contextmanager
    def stage(rows, name, *args, **kwargs):
        nonlocal step
        if name == 'outer-physics':
            step += 1
        with stage_original(rows, name, *args, **kwargs) as value:
            yield value
    rt.timed_stage = stage

    def destinations(plan):
        out = {}
        for i, row in enumerate(plan._rows):
            owner = next(a for a in plan._keep if a.data.ptr == int(row['dst']))
            out[(str(i),)] = cp.ndarray(tuple(int(n) for n in row['shape']), dtype=cp.uint32,
                memptr=owner.data, strides=tuple(int(s)*4 for s in row['dst_stride']))
        return out

    def measure(call):
        values = []
        for _ in range(9):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record(); call(); end.record(); end.synchronize()
            values.append(cp.cuda.get_elapsed_time(start, end))
        return statistics.median(values)

    def run(plan):
        nonlocal tiled
        if step not in steps or not plan._rows:
            return original(plan)
        if plan._device is None:
            plan.build()
        dst = destinations(plan)
        before = {k: a.copy() for k, a in dst.items()}
        old(plan._grid, (gp._THREADS,), (plan._device,))
        expected = {k: a.copy() for k, a in dst.items()}
        for k, a in dst.items():
            a[...] = before[k]
        original(plan)
        _compare(plan.name, step, expected, dst)
        if step == 24:
            if os.environ.get('GLUE_TILE_ALL') == '1':
                from tiled_plans import run as run_tiled
                run_tiled(plan, old, gp, cp, step)
            base_ms = measure(lambda: old(plan._grid, (gp._THREADS,), (plan._device,)))
            tip_kernel = gp._kernel(cp, plan._tiled)
            tip_ms = measure(lambda: tip_kernel(plan._grid, (gp._THREADS,), (plan._device,)))
            timings.append(dict(plan=plan.name, shape='x1.2562', base_ms=base_ms, tip_ms=tip_ms,
                bytes=sum(int(r['count'])*(4 if r['mode'] else 8) for r in plan._rows)))
            if not tiled:
                for row in plan._rows:
                    shape = tuple(int(n) for n in row['shape'])
                    if (shape[:2] == (1, 1) and not row['mode']
                            and row['dst_stride'][3] == row['src_stride'][2] == 1
                            and 2562 in shape):
                        owner = next(a for a in plan._keep if a.data.ptr == int(row['src']))
                        src = cp.ndarray(shape, dtype=cp.uint32, memptr=owner.data,
                            strides=tuple(int(s)*4 for s in row['src_stride']))[0, 0]
                        axis = src.shape.index(2562)
                        reps = [1, 1]; reps[axis] = 81
                        expanded = cp.asfortranarray(cp.tile(src, reps))
                        target = cp.empty(expanded.shape, cp.uint32, order='C')
                        test = gp.CopyPlan('real transpose tiled 207522 columns')
                        test.copy(target, expanded); test.build()
                        old(test._grid, (gp._THREADS,), (test._device,))
                        expected_tile = target.copy()
                        original(test)
                        _compare(test.name, step, {('target',): expected_tile}, {('target',): target})
                        timings.append(dict(plan=test.name, shape=list(target.shape),
                            base_ms=measure(lambda: old(test._grid, (gp._THREADS,), (test._device,))),
                            tip_ms=measure(lambda: gp._kernel(cp, test._tiled)(test._grid, (gp._THREADS,), (test._device,))), bytes=target.nbytes*2))
                        tiled = True
                        break
        return None
    gp.CopyPlan.__call__ = run
    from add_pair import install as install_add
    install_add(lambda: step, _compare)
    if os.environ.get('GLUE_BASE_RUC'):
        from ruc_ab import install as install_ruc
        install_ruc()

    def dump():
        with open(os.environ['GLUE_AB_OUT'] + '.timings.json', 'w') as stream:
            json.dump(timings, stream, indent=2)
    atexit.register(dump)


if __name__ == '__main__':
    install()
    runpy.run_path(os.environ['GLUE_FORECAST_LAUNCHER'], run_name='__main__')
