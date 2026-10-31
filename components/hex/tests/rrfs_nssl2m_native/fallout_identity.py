"""CPU witness for a rolling native fallout flux, before any production change."""
import ctypes,json,os,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from woof.hex.mpasphys.rrfs import mp_nssl2m as m
from woof.hex.mpasphys.rrfs.nssl_phases import _block
from prepare import HOST_MATH
start=m.MODULE_SOURCE.index('__device__ __noinline__ void f_fallout1d(')
o,end=_block(m.MODULE_SOURCE,start);original=m.MODULE_SOURCE[start:end+1]
from woof.hex.mpasphys.rrfs.nssl_sediment import stream_fallout as stream
new=stream(original).replace('void f_fallout1d(','void f_stream(')
root=Path(sys.argv[1]);root.mkdir(exist_ok=True)
wrapper=r'''
extern "C" void run(int variant,int nz,int id,int stride,float* a,float* vt,float* db,float* dtz,float* xf) {
 std::vector<unsigned char> memory(65536);NsslArena arena{memory.data(),0};Probe p{arena};
 auto aa=nssl_sequence<float,4>(a,stride,{1,1,1,1},{1,1,nz,1});
 auto vv=nssl_sequence<float,2>(vt,stride,{1,1},{nz+1,1});
 auto dd=nssl_sequence<float,2>(db,stride,{1,1},{1,nz+1});
 auto dz=nssl_sequence<float,3>(dtz,stride,{1,1,0},{nz+1,1,1});
 auto xx=nssl_sequence<float,3>(xf,stride,{1,1,1},{1,1,1});
 if(variant)p.f_stream(1,1,nz,0,1,0.1f,0.25f,1,vv,aa,dd,1,id,xx,dz,1);
 else p.f_fallout1d(1,1,nz,0,1,0.1f,0.25f,1,vv,aa,dd,1,id,xx,dz,1);
}
'''
cpp=root/'probe.cpp';lib=root/'probe.so'
cpp.write_text(HOST_MATH+'\n#include <vector>\n'+m.ARRAY_SUPPORT+m.OPERATOR_SUPPORT+'\nstruct Probe { NsslArena& arena;\n'+original+'\n'+new+'\n};\n'+wrapper)
subprocess.run(['timeout','60','nice','-n','10','g++','-std=c++17','-O1','-fno-fast-math','-ffp-contract=off','-shared','-fPIC',str(cpp),'-o',str(lib)],check=True)
f=ctypes.CDLL(str(lib)).run;f.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*5
rng=np.random.default_rng(912);words=0;cases=0;diff=0
for nz in (1,2,3,12,59):
 for stride in (1,64):
  for id in (0,1):
   for pattern in range(12):
    sizes=(nz,nz+1,nz+1,2*(nz+1),1)
    arrays=[rng.uniform(0,1,size*stride).astype('f4') for size in sizes]
    if pattern<6:
     arrays[0][::stride]=0
     if pattern<nz:arrays[0][pattern*stride]=0.1
    elif pattern>=7:
     special=np.array([0,0x80000000,0x7f800000,0xff800000,0x7fc00001,0xffc00002,0x7f800001,1,0x80000001],dtype='u4').view('f4')
     arrays[0][::stride]=rng.choice(special,nz)
    left=[a.copy() for a in arrays];right=[a.copy() for a in arrays]
    f(0,nz,id,stride,*(a.ctypes.data for a in left));f(1,nz,id,stride,*(a.ctypes.data for a in right))
    for a,b in zip(left,right):words+=a.size;diff+=int(np.count_nonzero(a.view('u4')!=b.view('u4')))
    cases+=1
result=dict(cases=cases,words=words,different_words=diff,passed=diff==0);print(json.dumps(result),flush=True);assert diff==0
