"""Assemble only MYNN's existing, qualified math helpers for the new unit."""
from woof.core.kernels import module_source


def cuda_source(work):
    old = module_source('mynn_pbl')
    prefix = old[:old.index('extern "C" __global__')]
    start = old.index('__device__ __forceinline__ real mynn_scale_exponent(')
    end = old.index('// Fortran real**real,', start)
    return prefix + old[start:end] + '\n' + (work / 'mynn471.cu').read_text()
