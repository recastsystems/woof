#!/usr/bin/env python3
"""Finish offline F2C lowering; generated numeric code has no libf2c ABI."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('work',type=Path)
p.add_argument('--global-scratch',action='store_true');a=p.parse_args()
s=(a.work/'model.c').read_text();meta=json.loads((a.work/'schema.json').read_text())
s=re.sub(r'^/\* model.f.*?\*/','// WRF 4.7.1 public-domain MYNN numerical source; generated offline.',s,count=1,flags=re.S)
s=s.replace('#include "f2c.h"','')
s=re.sub(r'\bextern\b[^;]*;','',s)
s=re.sub(r'\bdouble\s+\w+\s*\([^;]*;','',s)
s=s.replace('(doublereal)', '')
prototypes=[]
scratch_bytes=0
routine_names=[]
pat=r'((?:/\* Subroutine \*/ )?(?:int|integer|real|doublereal))\s+(\w+_)\((.*?)\)\s*\{'
def signature(m):
    routine_names.append(m[2])
    args=('unsigned char *m471_work, ' if a.global_scratch else '')+m[3]
    d=f'__device__ __noinline__ {m[1]} {m[2]}({args})'
    prototypes.append(d+';');return d+' {'
s=re.sub(pat,signature,s,flags=re.S)
# F2C makes power scratch double even for REAL**REAL. WRF uses powf.
# Explicit DOUBLE PRECISION closure temporaries retain their native type.
# F2C power arguments remain ABI doubles; pow_dd narrows at the call.
# Native's cold initializer reads local rmol before its later assignment.
# Match the retained fresh-process native frame (zero), deterministically;
# leaving C++ undefined here reuses a prior launch's stack contents.
s=s.replace('real sqcg, flqv, rmol, fltv, red001, delt2,',
            'real rmol = 0.f;\n    real sqcg, flqv, fltv, red001, delt2,')
boundary=s.index('__device__ __noinline__')
s=s[:boundary].replace('static ','static __device__ ')+s[boundary:].replace('static ','')
def arena_array(typ,name,count):
    global scratch_bytes
    alignment=8 if typ=="doublereal" else 4
    scratch_bytes=(scratch_bytes+alignment-1)//alignment*alignment
    declaration=f'{typ} *{name}=({typ}*)(m471_work+{scratch_bytes});'
    scratch_bytes+=int(count)*(8 if typ=='doublereal' else 4)
    return declaration
if a.global_scratch:
    # Explicit bounded scratch avoids CUDA's context-wide local-stack reserve.
    # Each column and routine owns a disjoint slice; calls never share state.
    def declaration(m):
        typ,entries=m[1],m[2]
        result=[];remaining=[]
        for entry in entries.split(','):
            clean=re.sub(r'/\*.*?\*/','',entry,flags=re.S)
            array=re.fullmatch(r'\s*(\w+)\[\s*(\d+)\s*\]\s*',clean)
            if array:result.append(arena_array(typ,array[1],array[2]))
            else:remaining.append(entry)
        if remaining:result.insert(0,typ+' '+','.join(remaining)+';')
        return '\n'.join(result)
    s=re.sub(r'\b(real|integer|logical|doublereal)\s+([^;{}]+);',declaration,s)
    for name in routine_names:
        # Definitions already carry the typed scratch argument.
        s=re.sub(r'\b'+re.escape(name)+r'\((?!unsigned char)',name+'(m471_work, ',s)
insert=s.index('/* Table of constant values */')
s=s[:insert]+'\n'.join(prototypes)+'\n'+s[insert:]
# The native functions are single precision except explicitly declared
# DOUBLE PRECISION closure algebra. Keep overloads rather than blanket casts.
helpers='''
#if GPUWM_WRF_EXACT
#define M471_CALL __noinline__
#define M471_LAUNCH
#else
#define M471_CALL __forceinline__
#define M471_LAUNCH __launch_bounds__(128,4)
#endif
typedef int integer;
typedef int logical;
typedef double doublereal;
#define TRUE_ 1
#define FALSE_ 0
#define abs(x) ((x)>=0 ? (x) : -(x))
#define dabs(x) abs(x)
#define dmin(a,b) ((a)<=(b)?(a):(b))
#define dmax(a,b) ((a)>=(b)?(a):(b))
#define min(a,b) dmin(a,b)
#define max(a,b) dmax(a,b)
__device__ float pow_dd(double*a,double*b){return mynn_glibc_powf((float)*a,(float)*b);}
__device__ double d_sign(double*a,double*b){return *b>=0.?fabs(*a):-fabs(*a);}
__device__ float r_lg10(float*a){return mynn_glibc_log10f(*a);}
__device__ float erf_(float*a){
#if GPUWM_WRF_EXACT
 return mynn471_erff(*a);
#else
 return erff(*a);
#endif
}
__device__ float m471_sqrt(float x){return sqrtf(x);}
__device__ double m471_sqrt(double x){return sqrt(x);}
__device__ float m471_exp(float x){return mynn_glibc_expf(x);}
__device__ double m471_exp(double x){return exp(x);}
__device__ float m471_log(float x){return mynn_glibc_logf(x);}
__device__ float m471_atan(float x){return mynn_glibc_atanf(x);}
__device__ float m471_tanh(float x){return mynn_tanhf(x);}
#define sqrt(x) m471_sqrt(x)
#define exp(x) m471_exp(x)
#define log(x) m471_log(x)
#define atan(x) m471_atan(x)
#define tanh(x) m471_tanh(x)
'''
helpers=(Path(__file__).parents[2]/'woof/data/mynn471/erff.cuh').read_text()+helpers
s=s.replace('__device__ __noinline__','__device__ M471_CALL')
load=[];store=[];decls=[];args=[];offsets={'f32':0,'i32':0}
for name,f in meta['arguments'].items():
    if f['type']=='character':continue
    typ='float' if f['type']=='real' else 'int'
    slab='f32' if typ=='float' else 'i32';count=1
    if f['dims']:
        for dim in f['dims'].split(','):
            parts=dim.split(':');lo,hi=parts if len(parts)==2 else ('1',parts[0])
            def size(expr):
                expr=re.sub(r'\bkts\b','1',expr);expr=re.sub(r'\bkte\b',str(meta['nz']),expr)
                expr=re.sub(r'\b(nchem|ndvel)\b','1',expr)
                if not re.fullmatch(r'[\d\s+*/()-]+',expr):raise ValueError(expr)
                return int(eval(expr,{'__builtins__':{}},{}))
            count*=size(hi)-size(lo)+1
    off=offsets[slab];offsets[slab]+=count;f.update(slab=slab,offset=off,count=count)
    var='a_'+name
    decls.append(arena_array(typ,var,count) if a.global_scratch and f['dims']
                 else f'{typ} {var}'+(f'[{count}]' if f['dims'] else '')+';')
    addr=var+'[k]' if f['dims'] else var
    if f['dims']:
        load.append(f'for(int k=0;k<{count};++k){addr}={slab}[({off}+k)*n+c];')
        if f['intent']!='in':store.append(f'for(int k=0;k<{count};++k){slab}[({off}+k)*n+c]={addr};')
    else:
        load.append(f'{addr}={slab}[{off}*n+c];')
        if f['intent']!='in':store.append(f'{slab}[{off}*n+c]={addr};')
    args.append(var if f['dims'] else '&'+var)
scratch_bytes=(scratch_bytes+3)//4*4
engine='''
extern "C" __global__ M471_LAUNCH void mynn471_step(int n,float*f32,int*i32'''+(',unsigned char*arena' if a.global_scratch else '')+'''){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
'''+(f'unsigned char*m471_work=arena+c*{scratch_bytes};\n' if a.global_scratch else '')+ '\n'.join(decls+load)+ '\n mynnedmf_('+('m471_work,' if a.global_scratch else '')+','.join(args)+');\n'+'\n'.join(store)+'\n}\n'
meta['slab_fields']=offsets
meta['scratch_bytes']=scratch_bytes if a.global_scratch else 0
# Fixed-size private arrays make the f2c whole-array loops compile-time
# bounded. Unrolling all of them explodes NVVM's optimizer, which crashed
# on the full native driver. Keep the native sequential vertical walks.
unit = re.sub(r'\bfor\s*\(', '\n#pragma unroll 1\nfor (', helpers+s+engine)
(a.work/'mynn471.cu').write_text(unit)
(a.work/'schema.json').write_text(json.dumps(meta,indent=2)+'\n')
