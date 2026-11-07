"""Test the binary64 intrinsic words used during NTU table construction."""
import ctypes,ctypes.util,json,sys
from pathlib import Path
import numpy as np
import cupy as cp
from woof.core import kernels
library=ctypes.CDLL(ctypes.util.find_library('m'))
source=kernels._extra_header_text('ntu')+'''
extern "C" __global__ void words(const double* in,double* out,int n){int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;double x=in[i];out[i]=glibc_exp(x);out[n+i]=glibc_log(x);out[2*n+i]=erf(x);}
'''
kernel=cp.RawKernel(source,'words',options=('-std=c++17','--ftz=false','--fmad=false'))
result={}
for row,(name,values) in enumerate((('exp',np.linspace(-500,500,4097)),('log',np.logspace(-100,100,4097)),('erf',np.linspace(-10,10,4097)))):
 x=values.astype(np.float64); device=cp.asarray(x); got=cp.empty((3,len(x)),dtype=cp.float64); kernel(((len(x)+127)//128,),(128,),(device,got,np.int32(len(x))))
 actual=cp.asnumpy(got[row]); f=getattr(library,name); f.restype=ctypes.c_double; f.argtypes=[ctypes.c_double]; expected=np.array([f(float(v)) for v in x],dtype=np.float64)
 a=actual.view(np.uint64); b=expected.view(np.uint64); bad=np.flatnonzero(a!=b)
 key=lambda bits: -int(bits&np.uint64(0x7fffffffffffffff)) if bits&np.uint64(0x8000000000000000) else int(bits)
 distance=[abs(key(aa)-key(bb)) for aa,bb in zip(a,b)]
 result[name]={'bit_mismatches':int(len(bad)),'max_ulp':max(distance),'words':len(x),'examples':[{'x':float(x[i]),'gpu_bits':int(a[i]),'host_bits':int(b[i]),'ulp':distance[i]} for i in bad[:8]]}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
