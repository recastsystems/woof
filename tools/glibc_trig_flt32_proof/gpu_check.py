from pathlib import Path
import ctypes
import datetime
import time
import hashlib
import numpy as np
import cupy as cp

root=Path(__file__).resolve().parent
names=['sinf','cosf','tanf','asinf','acosf','atanf']
lib=ctypes.CDLL(str(root/'gpu_oracle.so'))
ptr=ctypes.c_void_p
lib.fill.argtypes=[ptr,ctypes.c_uint64]
lib.reference.argtypes=[ctypes.c_int,ptr,ptr,ctypes.c_uint64]
header=(root/'glibc_trig_flt32.cuh').read_text()
pre=(root/'glibc_flt32.cuh').read_text()
src=pre+'\n'+header+'\n'
for name in names:
    src+=f'extern "C" __global__ void check_{name}(const float* x, unsigned int* y, unsigned long long n) {{ unsigned long long k=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x; if(k<n) y[k]=__float_as_uint(glibc_{name}(x[k])); }}\n'
options=('-std=c++17','--fmad=false','--ftz=true')
module=cp.RawModule(code=src,options=options,backend='nvrtc')
kernels=[module.get_function('check_'+name) for name in names]
print('CuPy',cp.__version__,'NVRTC',cp.cuda.nvrtc.getVersion(),'options',options,flush=True)
print('header_sha256',hashlib.sha256(header.encode()).hexdigest(),flush=True)
print('GPU',cp.cuda.runtime.getDeviceProperties(0)['name'],flush=True)
base=np.empty(1<<26,dtype=np.float32)
lib.fill(base.ctypes.data,base.size)
# Explicit NaN payloads, signed zeros, infinities, endpoints, range boundaries,
# every CORE-MATH exceptional input, and their immediate neighbours.
bits=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
      0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7f800001,0xff800001,
      0x7fc00000,0xffc00000,0x7fffffff,0xffffffff,0x3f800000,0xbf800000,
      0x3f3fffff,0x3f400000,0x42efffff,0x42f00000,0x4d800000,
      0x3f2ab445,0x3f083a1a,0x328885a3,0x39826222,0x4c700518,
      0x3f8a1f62,0x4d56d355,0x57d7b0ed,0x5980445e,
      0x63fc86fe,0x6a662711,0x6ad36709,0x72b505bb]
extras=np.array(sorted({((b+d)&0xffffffff)^s for b in bits for d in [-2,-1,0,1,2] for s in [0,0x80000000]}),dtype=np.uint32).view(np.float32)
inputs=np.concatenate([base,extras])
del base
expected=np.empty(inputs.size,dtype=np.uint32)
owner=Path.home()/'card-lease/OWNER'
def stamp(action):
    with owner.open('a') as f:
        f.write('urban-trig-proof '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+action+'\n')
# No launch occurs before the start record. Hold through every copy and launch.
start=time.monotonic()
stamp('start')
try:
    dx=cp.asarray(inputs)
    dy=cp.empty(inputs.size,dtype=cp.uint32)
    for f,name in enumerate(names):
        lib.reference(f,inputs.ctypes.data,expected.ctypes.data,inputs.size)
        begin,end=cp.cuda.Event(),cp.cuda.Event()
        begin.record()
        kernels[f](((inputs.size+255)//256,),(256,),(dx,dy,np.uint64(inputs.size)))
        end.record();end.synchronize()
        actual=cp.asnumpy(dy)
        bad=np.flatnonzero(actual!=expected)
        print(name,'inputs',inputs.size,'mismatches',bad.size,'gpu_ms',cp.cuda.get_elapsed_time(begin,end),
              'examples',[(hex(int(inputs.view(np.uint32)[k])),hex(int(expected[k])),hex(int(actual[k]))) for k in bad[:8]],flush=True)
        if time.monotonic()-start>=270:
            raise RuntimeError('GPU reservation approached five minute cap')
    # Compile and run a second configuration used by FTZ callers.
    # The full subnormal coverage above is reused to check the RN shortcuts.
    ftz=cp.RawModule(code=src,options=('-std=c++17','--fmad=true','--ftz=true'),backend='nvrtc')
    for f,name in enumerate(names):
        lib.reference(f,inputs.ctypes.data,expected.ctypes.data,inputs.size)
        ftz.get_function('check_'+name)(((inputs.size+255)//256,),(256,),(dx,dy,np.uint64(inputs.size)))
        actual=cp.asnumpy(dy)
        bad=np.flatnonzero(actual!=expected)
        print('FTZ',name,'inputs',inputs.size,'mismatches',bad.size,
              'examples',[(hex(int(inputs.view(np.uint32)[k])),hex(int(expected[k])),hex(int(actual[k]))) for k in bad[:8]],flush=True)
        if time.monotonic()-start>=270:raise RuntimeError('GPU reservation approached five minute cap')
finally:
    cp.cuda.runtime.deviceSynchronize()
    stamp('release')
    print('GPU reservation seconds',time.monotonic()-start,flush=True)
