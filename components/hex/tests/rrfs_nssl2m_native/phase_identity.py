"""Compare phased and original drivers with identical host arithmetic."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import HOST_MATH
from replay import Host, fixture, compare
from woof.hex.mpasphys.rrfs import mp_nssl2m as m
from woof.hex.mpasphys.rrfs.nssl_phases import _block

root = Path(sys.argv[1]).resolve()
root.mkdir(exist_ok=True)
source = m.cuda_source(shared_math_source='')
start = m.ENTRY_SOURCE.index('extern "C" __device__ void nssl_step_body(')
opening, closing = _block(m.ENTRY_SOURCE, start)
source += '\n' + m.ENTRY_SOURCE[start:closing+1]
signature = m.ENTRY_SOURCE[start:opening].replace('__device__ ', '').replace('nssl_step_body(', 'nssl_host_phase(')
names = 'active,workspace_column,nc,nz,nt,nmicro,diagnose,dbz_due,reqc,reqi,reqs,dt,state,environment,surface,diagnostics,refl1km,rainprod,evapprod,parameters,workspace,workspace_stride,errors'
wrapper = '\n#include <vector>\n' + signature + '{ NsslPhaseState a,b; NsslPhaseState* input=&a; NsslPhaseState* output=&b; std::vector<unsigned char> collection(32*65536);\n'
wrapper += f'nssl_phase_body<0>(0,input,{names});\n'
wrapper += 'for(int step=1;step<=nmicro;++step) {\n'
wrapper += f'nssl_phase_body<1>(step,input,{names});\n'
wrapper += '*output=*input;\n'
wrapper += f'for(int level=1;level<=nz;++level) nssl_collect_level_body(level,collection.data(),65536,0,output,step,input,{names});\n'
wrapper += 'auto temp=input;input=output;output=temp;\n'
wrapper += f'nssl_phase_body<3>(step,input,{names});\n'
wrapper += f'}} nssl_phase_body<4>(nmicro+1,input,{names}); }}\n'
wrapper += 'extern "C" unsigned long long nssl_host_parameter_bytes(){return sizeof(NsslState);}\n'
cpp=root/'phase.cpp'; library=root/'phase.so'
cpp.write_text(HOST_MATH+'\n#define __forceinline__ inline\n'+source+wrapper)
subprocess.run(['timeout','180','g++','-std=c++17','-O1','-fno-fast-math',
                '-ffp-contract=off','-fno-strict-aliasing','-shared','-fPIC',str(cpp),'-o',str(library)],check=True)
rows=[]
for moments in (2,3):
    original=Host(library,moments=moments)
    phased=Host(library,moments=moments)
    phased.lib.nssl_host_phase.argtypes=phased.lib.nssl_step_body.argtypes
    phased.lib.nssl_host_phase.restype=None
    phased.lib.nssl_step_body=phased.lib.nssl_host_phase
    a=fixture(nc=4,nz=12,moments=moments)
    b={n:x.copy() for n,x in a.items()}
    for nt,nmicro in ((1,1),(2,3),(3,2)):
        args=dict(nt=nt,nmicro=nmicro,diagnose=nt%2,dbz_due=1,reqc=1,reqi=1,reqs=1,dt=5.)
        original.run(a,**args);phased.run(b,**args)
        different=sum(int(np.count_nonzero(a[n].view(np.uint32)!=b[n].view(np.uint32))) for n in a)
        rows.append(dict(moments=moments,**args,different_words=different,words=sum(x.size for x in a.values())))
        assert different==0,compare(a,b)
(root/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows),flush=True)
