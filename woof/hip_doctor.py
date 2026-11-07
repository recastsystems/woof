"""Measured ROCm/HIP install and operation checks for both model engines."""
from __future__ import annotations
import argparse
import json


def run():
    result={'backend':'amd','experimental':True,'rocm_version':None,
            'gfx_target':None,'wave_size':None,'kernel_smoke':False,'checks':[]}
    def check(name,fn):
        try:value=fn();result['checks'].append(dict(name=name,ok=True,detail=value));return value
        except Exception as error:
            result['checks'].append(dict(name=name,ok=False,error=f'{type(error).__name__}: {error}'));return None
    from woof.backend import get_backend
    backend=check('backend',lambda:get_backend().name)
    if backend not in ('hip','hip-nvidia'):
        result['checks'].append(dict(name='ROCm selection',ok=False,error='Select --backend amd with ROCm-built CuPy; see docs/amd-backend.md'))
        result['ok']=False;return result
    result['backend']='amd' if backend == 'hip' else backend
    import cupy as cp
    import numpy as np
    from woof.hip_backend import device_info,compile_fingerprint,supports_graphs
    info=check('device',device_info)
    if info:
        result.update({key:info[key] for key in ('rocm_version','gfx_target','wave_size')})
    check('compiler',compile_fingerprint)
    def rust_driver():
        import ctypes
        from woof.hip_rust import driver
        path=driver();library=ctypes.CDLL(str(path))
        assert library.gpuwm_cuInit(0)==0,'HIP driver initialization'
        text=ctypes.create_string_buffer(256)
        assert library.gpuwmArchitecture(0,text,len(text))==0,'HIP driver architecture'
        return dict(library=str(path),architecture=text.value.decode())
    check('Rust driver ABI',rust_driver)
    def kernel():
        out=cp.empty(64,cp.uint32)
        source='''extern "C" __global__ void warp_probe(unsigned* p){
          int i=threadIdx.x; unsigned b=__ballot_sync(0xffffffffu,(i&31)==0);
          p[i]=b + __shfl_sync(0xffffffffu,(unsigned)i,0); }'''
        cp.RawKernel(source,'warp_probe')((1,),(64,),(out,))
        expected=np.repeat(np.array([1,33],np.uint32),32)
        np.testing.assert_array_equal(out.get(),expected)
        return '64 threads: independent 32-lane ballots and shuffles'
    result['kernel_smoke']=check('kernel/warp grouping',kernel) is not None
    def blas():
        a=cp.arange(64,dtype=cp.float32).reshape(8,8)
        np.testing.assert_array_equal((a@a).get(),np.arange(64,dtype='f4').reshape(8,8)@np.arange(64,dtype='f4').reshape(8,8))
        return 'CuPy GEMM through '+('hipBLAS/rocBLAS' if cp.cuda.runtime.is_hip else 'cuBLAS (HIP-on-NVIDIA)')
    check('BLAS',blas)
    def fft():
        x=cp.arange(64,dtype=cp.float32)
        np.testing.assert_allclose(cp.fft.ifft(cp.fft.fft(x)).real.get(),np.arange(64),atol=2e-5)
        return 'CuPy FFT through '+('hipFFT/rocFFT' if cp.cuda.runtime.is_hip else 'cuFFT (HIP-on-NVIDIA)')
    check('FFT',fft)
    def rng():
        x=cp.random.RandomState(7).random_sample(1024)
        assert bool(cp.isfinite(x).all()) and bool(((x>=0)&(x<1)).all())
        return ('CuPy RNG through '+('hipRAND/rocRAND' if cp.cuda.runtime.is_hip else 'cuRAND (HIP-on-NVIDIA)')
                +'; cross-vendor RNG identity is not asserted')
    check('RNG',rng)
    def memory():
        mem=cp.cuda.alloc_pinned_memory(256)
        host=np.frombuffer(mem,dtype='f4',count=64);host[:]=np.arange(64)
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            x=cp.asarray(host);x+=1
            event=cp.cuda.Event();event.record()
        event.synchronize();x.get(out=host,stream=stream,blocking=True)
        np.testing.assert_array_equal(host,np.arange(64)+1)
        return 'Pinned host memory, ordered stream transfer and event completion'
    check('streams/events/pinned memory',memory)
    check('graphs policy',lambda:dict(capture=supports_graphs(),fallback='ordered stream launches'))
    from woof import gpu_smi
    check('telemetry',lambda:gpu_smi.inventory() if cp.cuda.runtime.is_hip else {'command':gpu_smi.command()})
    result['ok']=all(c['ok'] for c in result['checks']);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--json',action='store_true');p.parse_args(argv)
    result=run();print(json.dumps(result,indent=2));return int(not result['ok'])
if __name__=='__main__':raise SystemExit(main())
