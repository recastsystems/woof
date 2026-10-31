"""Shared-driver proposal, measured only. No production dispatch is changed."""
import json
import os
import statistics
import sys
import numpy as np
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS

SOURCE = CUDA_FTZ_HELPERS + r'''
extern "C" __global__ void rrfs_add_pair_f32(
    int count, const float* first, const float* second, float* target) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<count; i+=gridDim.x*blockDim.x) {
        float intermediate=mpas_add(target[i],first[i]);
        target[i]=mpas_add(intermediate,second[i]);
    }
}
'''


def install(get_step, compare):
    import cupy as cp
    from woof.hex.cuda_driver import CudaDryDycoreDriver as Driver
    from woof.hex.cuda_compile_rrfs import module
    kernel = None
    original = Driver._add
    pending = {}
    timed = set()

    def fused(target, first, second):
        nonlocal kernel
        if kernel is None:
            kernel = module(SOURCE, 'hexcore.tools.rrfs_glue_copies.add_pair').get_function('rrfs_add_pair_f32')
        kernel(((target.size+127)//128,), (128,), (np.int32(target.size), first, second, target))

    def measure(call):
        samples = []
        for _ in range(9):
            a, b = cp.cuda.Event(), cp.cuda.Event()
            a.record(); call(); b.record(); b.synchronize()
            samples.append(cp.cuda.get_elapsed_time(a, b))
        return statistics.median(samples)

    def add(driver, target, increment):
        step = get_step()
        line = sys._getframe(1).f_lineno
        if step not in (1, 12, 24) or line not in (4329, 4331, 4369, 4371):
            return original(driver, target, increment)
        if line in (4329, 4369):
            pending[target.data.ptr] = target.copy(), increment.copy()
            return original(driver, target, increment)
        before, first = pending.pop(target.data.ptr)
        original(driver, target, increment)
        expected = target.copy()
        target[...] = before
        fused(target, first, increment)
        compare('shared add-pair proposal ' + str(line), step, {('target',): expected}, {('target',): target})
        if step == 24 and target.shape not in timed:
            timed.add(target.shape)
            records = []
            for repeat in (1, 81):
                dst, x, y = (cp.tile(a, (1, repeat)) for a in (before, first, increment))
                original(driver, dst, x); original(driver, dst, y)
                want = dst.copy(); dst[...] = cp.tile(before, (1, repeat))
                fused(dst, x, y)
                compare('tiled add-pair proposal ' + str(repeat), step, {('target',): want}, {('target',): dst})
                records.append(dict(shape=list(dst.shape),
                    base_ms=measure(lambda: (original(driver, dst, x), original(driver, dst, y))),
                    tip_ms=measure(lambda: fused(dst, x, y)),
                    base_bytes=6*dst.nbytes, tip_bytes=4*dst.nbytes))
            with open(os.environ['GLUE_AB_OUT'] + '.add-timings.jsonl', 'a') as stream:
                stream.write(json.dumps(records) + '\n')
        return None
    Driver._add = add
