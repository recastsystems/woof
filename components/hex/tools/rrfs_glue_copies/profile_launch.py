"""One-off NVTX call-site and logical-byte attribution; never a timing run.

No device values, synchronizations or CUDA events are read/added. Python call
ranges let Nsight attribute even assignment-generated CuPy kernels. Explicit
copy/add wrappers record logical bytes (reads plus writes), not DRAM counters.
"""
import atexit
from collections import Counter
import functools
import json
import os
import runpy
import sys


def install():
    import cupy as cp
    from cupy.cuda import nvtx
    from woof.hex.mpasphys.rrfs.glue_plan import CopyPlan
    from woof.hex.cuda_driver import CudaDryDycoreDriver

    counts = Counter()
    active = False
    step = 0

    def site(frame):
        names = []
        while frame and len(names) < 3:
            path = frame.f_code.co_filename
            if '/hexcore/' in path:
                names.append(path.split('/hexcore/', 1)[1] + ':' + str(frame.f_lineno))
            frame = frame.f_back
        return ' <- '.join(names)

    def record(kind, where, size, shape=()):
        if active:
            counts[(step, kind, where, int(size), tuple(shape))] += 1

    def wrap_copy(name):
        original = getattr(cp, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            where = site(sys._getframe(1))
            nvtx.RangePush('copy:' + name + ':' + where)
            try:
                result = original(*args, **kwargs)
                if name == 'copyto':
                    dst = args[0]
                    record(name, where, 2 * dst.nbytes, dst.shape)
                elif args and isinstance(args[0], cp.ndarray):
                    src = args[0]
                    if result.data.ptr != src.data.ptr:
                        record(name, where, src.nbytes + result.nbytes, result.shape)
                return result
            finally:
                nvtx.RangePop()
        setattr(cp, name, wrapped)
    for name in ('array', 'copy', 'copyto', 'ascontiguousarray', 'asfortranarray'):
        wrap_copy(name)

    original_plan = CopyPlan.__call__
    def plan(self):
        where = self.name + ' @ ' + site(sys._getframe(1))
        nvtx.RangePush('plan:' + where)
        try:
            for row in self._rows:
                shape = tuple(int(x) for x in row['shape'])
                size = int(row['count']) * (4 if int(row['mode']) else 8)
                record('plan-fill' if int(row['mode']) else 'plan-copy', where, size, shape)
            return original_plan(self)
        finally:
            nvtx.RangePop()
    CopyPlan.__call__ = plan

    original_add = CudaDryDycoreDriver._add
    def add(self, target, increment):
        where = site(sys._getframe(1))
        nvtx.RangePush('add:' + where)
        try:
            record('add', where, 3 * target.nbytes, target.shape)
            return original_add(self, target, increment)
        finally:
            nvtx.RangePop()
    CudaDryDycoreDriver._add = add

    from woof.hex.mpasphys import runtime_v842 as runtime
    import contextlib
    original_stage = runtime.timed_stage
    @contextlib.contextmanager
    def stage(rows, name, *args, **kwargs):
        nonlocal active, step
        if name == 'outer-physics':
            step += 1
        old = active
        active = name in ('outer-physics', 'dycore-and-post-transport')
        nvtx.RangePush(f'step:{step}:stage:{name}')
        try:
            with original_stage(rows, name, *args, **kwargs) as value:
                yield value
        finally:
            nvtx.RangePop()
            active = old
    runtime.timed_stage = stage

    # C calls include ndarray.copy/astype. Python ranges enclose implicit
    # assignment kernels which do not produce c_call profile events.
    def profile(frame, event, arg):
        path = frame.f_code.co_filename
        if not active or '/hexcore/' not in path:
            return
        if event == 'call':
            nvtx.RangePush('py:' + site(frame))
        elif event == 'return':
            if (path.endswith('/resident_output_diagnostics_v842.py')
                    and frame.f_code.co_name == 'current'):
                value = frame.f_locals.get('value')
                if isinstance(value, cp.ndarray):
                    where = (path.split('/hexcore/', 1)[1] + ':'
                             + str(frame.f_code.co_firstlineno) + ' <- ' + site(frame.f_back))
                    # Keep the same three-frame key as the enclosing Python range.
                    where = ' <- '.join(where.split(' <- ')[:3])
                    record('implicit-current', where, 2 * value.nbytes, value.shape)
            nvtx.RangePop()
        elif event == 'c_call':
            name = getattr(arg, '__name__', '')
            nvtx.RangePush('c:' + name + ':' + site(frame))
            owner = getattr(arg, '__self__', None)
            if name in ('copy', 'reshape', 'astype') and isinstance(owner, cp.ndarray):
                record('ndarray.' + name, site(frame), 2 * owner.nbytes, owner.shape)
        elif event in ('c_return', 'c_exception'):
            nvtx.RangePop()

    def dump():
        sys.setprofile(None)
        records = [dict(step=k[0], kind=k[1], site=k[2], bytes=k[3], shape=k[4], calls=n)
                   for k, n in sorted(counts.items())]
        with open(os.environ['GLUE_COPY_OUT'], 'w') as stream:
            json.dump(dict(records=records, steps=step,
                           byte_definition='logical source reads plus destination writes'), stream, indent=2)
    atexit.register(dump)
    sys.setprofile(profile)


if __name__ == '__main__':
    install()
    runpy.run_path(os.environ['GLUE_FORECAST_LAUNCHER'], run_name='__main__')
