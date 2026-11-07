"""Measure device intrinsic words against the oracle host's math calls."""
import ctypes,ctypes.util,json,sys

REQUIRED_EXACT = ("expf:gfk_exp(x)", "logf:gfk_log(x)", "powf:gfk_pow(x, 0.3333333432674407958984375f)", "sqrtf:sqrtf(x)", "log10f:ntu_log10(x)", "cosf:glibc_cosf(x)")
from pathlib import Path
import numpy as np

def main(output):
    import cupy as cp
    from woof.core import kernels
    from woof.core.fp32_ulp import fp32_ulp_distance
    library=ctypes.CDLL(ctypes.util.find_library('m'))
    specs=[('expf','gfk_exp(x)',np.linspace(-80,80,4097)),('logf','gfk_log(x)',np.logspace(-35,35,4097)),('powf','gfk_pow(x, 0.3333333432674407958984375f)',np.logspace(-30,30,4097)),('log10f','((float)log10((double)x))',np.logspace(-8,6,4097)),('erff','((float)erf((double)x))',np.linspace(-10,10,4097)),('cosf','cosf(x)',np.linspace(0,4,4097)),('sqrtf','sqrtf(x)',np.logspace(-35,35,4097))]
    header=kernels._extra_header_text('ntu')
    rt=Path(__file__).with_name('ntu_rt.cuh').read_text()
    local=rt[rt.index('NTU_DEV float ntu_log10'):rt.index('\n#endif',rt.index('NTU_DEV float ntu_log10'))]
    header+='\n'+local.replace('NTU_DEV','__device__')
    specs += [('log10f','ntu_log10(x)',np.logspace(-8,6,4097)),('cosf','glibc_cosf(x)',np.linspace(0,4,4097))]
    source=header+'\nextern "C" __global__ void words(const float* in, float* out, int n){int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n)return; float x=in[i];\n'
    for row,(_,expr,_) in enumerate(specs): source+=f'out[{row}*n+i]={expr};\n'
    source+='}\n'
    kernel=cp.RawKernel(source,'words',options=('-std=c++17','--ftz=false','--fmad=false'))
    result={}
    for row,(name,expr,values) in enumerate(specs):
        x=values.astype(np.float32); device=cp.asarray(x); got=cp.empty((len(specs),len(x)),dtype=cp.float32)
        kernel(((len(x)+127)//128,),(128,),(device,got,np.int32(len(x))))
        actual=cp.asnumpy(got[row]); f=getattr(library,name); f.restype=ctypes.c_float; f.argtypes=[ctypes.c_float]*(2 if name=='powf' else 1)
        expected=np.array([f(float(v),float(np.float32(1/3))) if name=='powf' else f(float(v)) for v in x],dtype=np.float32)
        d=fp32_ulp_distance(actual,expected); bad=np.flatnonzero(actual.view(np.uint32)!=expected.view(np.uint32))
        result[name+':'+expr]={'expression':expr,'words':len(x),'bit_mismatches':int(len(bad)),'max_ulp':int(d.max()),'examples':[{'x':float(x[i]),'gpu_bits':int(actual.view(np.uint32)[i]),'host_bits':int(expected.view(np.uint32)[i]),'ulp':int(d[i])} for i in bad[:8]]}
    Path(output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    return result
if __name__=='__main__':
    result=main(sys.argv[1])
    if '--require-exact' in sys.argv[2:]:
        sys.exit(1 if any(result[key]['bit_mismatches'] for key in REQUIRED_EXACT) else 0)
