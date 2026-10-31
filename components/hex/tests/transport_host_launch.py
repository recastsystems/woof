"""Execute unchanged transport CUDA bodies as C++ for host orchestration checks.

This is not a CUDA/native oracle. It checks launch slicing and copy decisions
against the one-species path using the same arithmetic bodies. GPU bitwise
checks remain separate. The shim substitutes CUDA thread indices and intrinsics.
"""
import ctypes as ct
import re
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np


class HostArrays:
    def __init__(self):
        self.pool = SimpleNamespace(malloc=lambda: None, free_bytes=lambda: 0)
        self.cuda = SimpleNamespace(
            runtime=SimpleNamespace(memGetInfo=lambda: (8 * 1024**3, 8 * 1024**3)),
            get_allocator=lambda: self.pool.malloc,
            get_current_stream=lambda: SimpleNamespace(is_capturing=lambda: False))

    def __getattr__(self, name):
        return getattr(np, name)

    def get_default_memory_pool(self):
        return self.pool

    @staticmethod
    def asnumpy(a):
        return np.asarray(a)

    @staticmethod
    def ReductionKernel(*args):
        def reduce(*inputs, axis):
            words = inputs[0]
            for value in inputs[1:]:
                words = words | value
            for dim in sorted(axis, reverse=True):
                words = np.bitwise_or.reduce(words, axis=dim)
            return words
        return reduce


class HostLaunch:
    def __init__(self, directory, source):
        compiler = shutil.which('g++')
        if compiler is None:
            raise RuntimeError('g++ required for transport host launch checks')
        declarations = re.findall(r'extern "C" __global__ void (\w+)\((.*?)\)\s*\{', source, re.S)
        shim = r'''
#include <cmath>
#include <cstring>
#include <algorithm>
using std::isfinite;
using std::min;
using std::max;
#define __global__
#define __device__
#define __forceinline__ inline
#define __constant__
struct Dim { int x; };
thread_local Dim blockDim, blockIdx, threadIdx, gridDim;
inline unsigned int __float_as_uint(float v) { unsigned int u; memcpy(&u,&v,4); return u; }
inline float __uint_as_float(unsigned int u) { float v; memcpy(&v,&u,4); return v; }
inline long long __double_as_longlong(double v) { long long u; memcpy(&u,&v,8); return u; }
inline float __double2float_rn(double v) { return (float)v; }
inline void atomicExch(int* p, int v) { __atomic_store_n(p,v,__ATOMIC_RELAXED); }
'''
        self.types = {}
        for name, text in declarations:
            args = [part.strip() for part in text.split(',')]
            names = [re.search(r'(\w+)\s*$', part).group(1) for part in args]
            self.types[name] = [ct.c_void_p if '*' in part else ct.c_float if 'float' in part else ct.c_int for part in args]
            shim += '\n'
            # Definitions follow the original source below.
        wrappers = []
        for name, text in declarations:
            args = [part.strip() for part in text.split(',')]
            names = [re.search(r'(\w+)\s*$', part).group(1) for part in args]
            wrappers.append(f'''extern "C" void host_{name}(int count, {text}) {{
                const int blocks = (count + 127) / 128;
                #pragma omp parallel for if(count > 4096)
                for (int i=0; i<blocks*128; ++i) {{
                    blockDim.x=128; gridDim.x=blocks; blockIdx.x=i/128; threadIdx.x=i%128;
                    {name}({', '.join(names)});
                }}
            }}''')
        cpp = directory / 'transport.cpp'
        so = directory / 'transport.so'
        cpp.write_text(shim + source + '\n' + '\n'.join(wrappers))
        result = subprocess.run([compiler, '-shared', '-fPIC', '-O2', '-fopenmp', '-ffp-contract=off', str(cpp), '-o', str(so)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        self.library = ct.CDLL(str(so))
        self.calls = []

    def __call__(self, name, count, args, cache=None):
        self.calls.append(name)
        function = getattr(self.library, 'host_' + name)
        function.argtypes = [ct.c_int, *self.types[name]]
        function.restype = None
        function(int(count), *(ct.c_void_p(v.ctypes.data) if isinstance(v, np.ndarray) else v for v in args))
