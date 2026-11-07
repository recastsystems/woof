"""Measure the actual NTU runtime ERF code, including IEEE edge values."""
import ctypes
import ctypes.util
import json
from pathlib import Path
import numpy as np

def runtime_source():
    from woof.core import kernels
    rt=Path(__file__).with_name('ntu_rt.cuh').read_text()
    start=rt.index('struct NtuDD')
    end=rt.index('\n#endif',start)
    return kernels._extra_header_text('ntu')+'\n'+rt[start:end].replace('NTU_DEV','__device__')

def main(output):
    import cupy as cp
    from woof.core import ntu
    library=ctypes.CDLL(ctypes.util.find_library('m'))
    source=runtime_source()+"""
extern "C" __global__ void words(const double* x,double* d,float* f,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){d[i]=ntu_erf(x[i]);f[i]=ntu_erff((float)x[i]);}
}
"""
    from woof.nvrtc_ptx_cache import compile_using_nvrtc
    image,_=compile_using_nvrtc(source,ntu.compile_options(),filename='ntu_runtime_erf.cu')
    module=cp.cuda.function.Module()
    module.load(image.encode() if isinstance(image,str) else image)
    kernel=module.get_function('words')
    result={}
    rng=np.random.default_rng(56)
    for name,dtype,ctype in (('erff','f4',ctypes.c_float),('erf','f8',ctypes.c_double)):
        tiny=np.nextafter(np.array(0,dtype=dtype),np.array(1,dtype=dtype)).item()
        edge=np.array([0.,-0.,tiny,-tiny,np.finfo(dtype).tiny,-np.finfo(dtype).tiny,
                       6.,-6.,np.nextafter(np.array(6,dtype=dtype),np.array(0,dtype=dtype)).item(),
                       np.inf,-np.inf,np.nan],dtype=dtype)
        x=np.concatenate((np.linspace(-6,6,16385).astype(dtype),
                          rng.uniform(-6,6,8192).astype(dtype),edge))
        inputs=cp.asarray(x.astype('f8'));d=cp.empty(len(x),'f8');f=cp.empty(len(x),'f4')
        kernel(((len(x)+127)//128,),(128,),(inputs,d,f,np.int32(len(x))))
        got=cp.asnumpy(f if dtype=='f4' else d)
        fn=getattr(library,name);fn.restype=ctype;fn.argtypes=[ctype]
        expected=np.array([fn(float(a)) for a in x],dtype=dtype)
        finite=np.isfinite(expected)
        unsigned='u4' if dtype=='f4' else 'u8'
        bits=got.view(unsigned);ref=expected.view(unsigned)
        diff=np.maximum(bits[finite],ref[finite])-np.minimum(bits[finite],ref[finite])
        bad=np.flatnonzero(finite & (bits!=ref))
        result[name]={'words':len(x),'finite_words':int(finite.sum()),
                      'bit_mismatches':len(bad),'max_ulp':int(diff.max()),
                      'max_abs':float(np.max(np.abs(got[finite].astype('f8')-expected[finite].astype('f8')))),
                      'special_words_match':bool(np.array_equal(bits[-12:-10],ref[-12:-10]) and
                                                 np.array_equal(bits[-6:-1],ref[-6:-1]) and
                                                 np.isnan(got[-1]) and np.all(got[-10:-6]!=0)),
                      'ieee_edges':[{'gpu_bits':int(a),'host_bits':int(b)}
                                    for a,b in zip(bits[-12:],ref[-12:])],
                      'examples':[{'x':float(x[i]),'gpu_bits':int(bits[i]),'host_bits':int(ref[i])} for i in bad[:8]]}
    result['options']=list(ntu.compile_options())
    Path(output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    import sys
    main(sys.argv[1])
