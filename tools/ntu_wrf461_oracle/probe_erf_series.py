"""Check an independent ERF recurrence built on WOOF binary64 exp."""
import ctypes,ctypes.util,json,sys
from pathlib import Path
import numpy as np
from woof.core import kernels
SOURCE=r'''
struct NtuDD {double hi,lo;};
__device__ NtuDD ntu_dd_norm(double a,double b){double h=__dadd_rn(a,b);return {h,__dsub_rn(b,__dsub_rn(h,a))};}
__device__ NtuDD ntu_dd_add(NtuDD a,NtuDD b){double h=__dadd_rn(a.hi,b.hi),v=__dsub_rn(h,a.hi);double e=__dadd_rn(__dsub_rn(a.hi,__dsub_rn(h,v)),__dsub_rn(b.hi,v));e=__dadd_rn(e,__dadd_rn(a.lo,b.lo));return ntu_dd_norm(h,e);}
__device__ NtuDD ntu_dd_mul(NtuDD a,NtuDD b){double h=__dmul_rn(a.hi,b.hi);double e=__fma_rn(a.hi,b.hi,-h);e=__dadd_rn(e,__dadd_rn(__dmul_rn(a.hi,b.lo),__dmul_rn(a.lo,b.hi)));return ntu_dd_norm(h,e);}
__device__ NtuDD ntu_dd_div(NtuDD a,double b){double q=__ddiv_rn(a.hi,b);double r=__dadd_rn(__fma_rn(-q,b,a.hi),a.lo);return ntu_dd_norm(q,__ddiv_rn(r,b));}
__device__ double ntu_series_erf(double x){
 if(x!=x)return x+x;
 double a=x<0?-x:x;
 if(a>=6)return x<0?-1:1;
 if(a==0)return x;
 NtuDD y=ntu_dd_mul({a,0},{a,0}),z=ntu_dd_mul(y,{2,0}),term={a,0},sum=term;
 for(int n=1;n<240;++n){term=ntu_dd_div(ntu_dd_mul(term,z),(double)(2*n+1));sum=ntu_dd_add(sum,term);if(term.hi<sum.hi*0x1p-105)break;}
 NtuDD expx=ntu_dd_mul({glibc_exp(-y.hi),0},{1,-y.lo});
 NtuDD answer=ntu_dd_mul(ntu_dd_mul({1.1283791670955126,1.533545961316588e-17},expx),sum);
 double result=__dadd_rn(answer.hi,answer.lo);if(result>1)result=1;return x<0?-result:result;
}
extern "C" __global__ void words(const double* in,double* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=ntu_series_erf(in[i]);}
'''
def main():
 import cupy as cp
 from woof.core.fp32_ulp import fp32_ulp_distance
 library=ctypes.CDLL(ctypes.util.find_library('m'))
 kernel=cp.RawKernel(kernels._extra_header_text('ntu')+SOURCE,'words',options=('-std=c++17','--ftz=false','--fmad=false'))
 result={}
 for name,dtype,ctype in (('erff','f4',ctypes.c_float),('erf','f8',ctypes.c_double)):
  x=np.linspace(-6,6,16385).astype(dtype);inputs=cp.asarray(x.astype('f8'));out=cp.empty(len(x),dtype='f8');kernel(((len(x)+127)//128,),(128,),(inputs,out,np.int32(len(x))));got=cp.asnumpy(out).astype(dtype)
  fn=getattr(library,name);fn.restype=ctype;fn.argtypes=[ctype];expected=np.array([fn(float(a)) for a in x],dtype=dtype)
  unsigned='u4' if dtype=='f4' else 'u8';bits=got.view(unsigned);ref=expected.view(unsigned);bad=np.flatnonzero(bits!=ref)
  ulp=fp32_ulp_distance(got,expected) if dtype=='f4' else np.abs(bits.astype('i8')-ref.astype('i8'))
  result[name]={'words':len(x),'bit_mismatches':len(bad),'max_ulp':int(ulp.max()),'max_abs':float(np.max(np.abs(got.astype('f8')-expected.astype('f8')))),'examples':[{'x':float(x[i]),'got':int(bits[i]),'reference':int(ref[i])} for i in bad[:8]]}
 Path(sys.argv[1]).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
