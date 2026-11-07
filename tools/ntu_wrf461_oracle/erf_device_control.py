"""Counterfactual host ERF words identify intrinsic rounding causes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from woof.core import ntu
sys.path.insert(0,str(Path(__file__).parent))
from run_gpu import main as run

def main():
 ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('trace',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--option',action='append',default=[]);a=ap.parse_args()
 records=np.fromfile(a.trace,'<u8').reshape(-1,3);source='__device__ unsigned int ntu_erf_miss[2]={0,0};\n';sizes={}
 for op,kind in ((1,'f'),(2,'d')):
  pairs={int(row[1]):int(row[2]) for row in records if row[0]==op};items=sorted(pairs.items());n=len(items);sizes[kind]=n;typ='unsigned int' if op==1 else 'unsigned long long';suffix='u' if op==1 else 'ull'
  source+=f'__device__ const {typ} ntu_erf_{kind}_x[{n}]={{'+','.join(hex(x)+suffix for x,y in items)+'};\n'
  source+=f'__device__ const {typ} ntu_erf_{kind}_y[{n}]={{'+','.join(hex(y)+suffix for x,y in items)+'};\n'
  ftype='float' if op==1 else 'double';bits='__float_as_uint(x)' if op==1 else '(unsigned long long)__double_as_longlong(x)';value='__uint_as_float' if op==1 else '__longlong_as_double';fallback='(float)erf((double)x)' if op==1 else 'erf(x)'
  source+=f'__device__ {ftype} ntu_erf_control_{kind}({ftype} x){{{typ} key={bits};int low=0,high={n}-1;while(low<=high){{int middle=(low+high)/2;{typ} k=ntu_erf_{kind}_x[middle];if(k==key)return {value}(ntu_erf_{kind}_y[middle]);if(k<key)low=middle+1;else high=middle-1;}}atomicAdd(&ntu_erf_miss[{op-1}],1u);return {fallback};}}\n'
 source+='extern "C" __global__ void ntu_erf_counts(unsigned int* out){out[0]=ntu_erf_miss[0];out[1]=ntu_erf_miss[1];}\n'
 original=ntu.module_source
 def altered(*args,**kw):
  s=original(*args,**kw);pos=s.index('// ===========================================================================\n// NTU multi-moment');s=s[:pos]+source+s[pos:];return s.replace('#define NTU_ERFF(x)   ((float)erf((double)(x)))','#define NTU_ERFF(x)   ntu_erf_control_f(x)').replace('#define NTU_ERF(x)    erf(x)','#define NTU_ERF(x)    ntu_erf_control_d(x)').replace('#define NTU_ERFF(x)   ntu_erff(x)','#define NTU_ERFF(x)   ntu_erf_control_f(x)').replace('#define NTU_ERF(x)    ntu_erf(x)','#define NTU_ERF(x)    ntu_erf_control_d(x)')
 ntu.module_source=altered;opts=ntu.compile_options;ntu.compile_options=lambda:opts()+tuple(a.option);run(a.input,a.output)
 import cupy as cp
 counts=cp.zeros(2,dtype=cp.uint32);ntu._kernel('ntu_erf_counts',64)((1,),(1,),(counts,));a.output.with_suffix('.lookup.json').write_text(json.dumps({'table_sizes':sizes,'lookup_misses':cp.asnumpy(counts).tolist(),'options':ntu.compile_options()},indent=2))
if __name__=='__main__':main()
