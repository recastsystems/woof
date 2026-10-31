"""Compare CUDA-initialized NSSL module storage with strict native ELF symbols."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import cupy as cp
import translate
from replay import Native
from woof.hex.mpasphys.rrfs import mp_nssl2m as m
from woof.hex.mpasphys.rrfs.nssl_math import source as math_source


def execute(a):
    original = Path.cwd()
    try:
        os.chdir(a.namelist.resolve().parent)
        native = Native(a.native)
    finally:
        os.chdir(original)
    runtime = (m.load_resources(json.loads(a.resources.read_text()), cp.cuda.Device())
               if a.resources else m.Nssl2Moment(shared_math_source=math_source()))
    runtime.initialize()
    variables, _ = translate.parse_modules([a.source])
    eligible = [v for v in variables.values() if not v.parameter and v.kind in ('float','double','int')]
    declarations = m.MODULE_SOURCE.split('struct NsslColumn',1)[0]
    code = [m.ARRAY_SUPPORT, declarations,
        'extern "C" __global__ void inspect(const unsigned char* parameters,unsigned long long* out) {',
        'const NsslState& s=*reinterpret_cast<const NsslState*>(parameters);']
    for i,v in enumerate(eligible):
        pointer,count = (f's.{v.name}.data',f's.{v.name}.size()') if v.dimensions else (f'&s.{v.name}','1')
        code.append(f'out[{3*i}]=(unsigned long long){pointer};out[{3*i+1}]={count};out[{3*i+2}]=sizeof({v.kind});')
    code.append('}')
    kernel=cp.RawKernel('\n'.join(code),'inspect',options=('--std=c++17',))
    descriptors=cp.empty((len(eligible),3),cp.uint64)
    kernel((1,),(1,),(runtime._parameters,descriptors))
    prefix='__module_mp_nssl_2mom_MOD_'
    symbols={}
    for line in subprocess.check_output(['nm','-S','--defined-only',str(a.native)],text=True).splitlines():
        row=line.split()
        if len(row)==4 and row[3].startswith(prefix):symbols[row[3][len(prefix):]]=(int(row[0],16),int(row[1],16))
    anchor=ct.cast(getattr(native.lib,prefix+'nssl_2mom_init'),ct.c_void_p).value
    base=anchor-symbols['nssl_2mom_init'][0]
    rows={}
    for v,desc in zip(eligible,descriptors.get()):
        pointer,count,width=map(int,desc);offset,size=symbols[v.name]
        assert size==count*width,(v.name,size,count,width)
        dtype=np.uint64 if width==8 else np.uint32
        expected=np.frombuffer((ct.c_ubyte*size).from_address(base+offset),dtype=dtype)
        memory=cp.cuda.UnownedMemory(pointer,size,runtime)
        actual=cp.ndarray((count,),dtype=dtype,memptr=cp.cuda.MemoryPointer(memory,0)).get()
        different=np.flatnonzero(expected!=actual)
        rows[v.name]=dict(words32=size//4,different_words32=int(np.count_nonzero(expected.view(np.uint32)!=actual.view(np.uint32))),
            witnesses=[dict(index=int(i),native=hex(int(expected[i])),cuda=hex(int(actual[i]))) for i in different[:4]])
    result=dict(native_sha256=hashlib.sha256(a.native.read_bytes()).hexdigest(),source_sha256=m.SOURCE_SHA256,
        initialization_storage_bytes=runtime.initialization_storage_bytes,variables=rows,
        words32=sum(r['words32'] for r in rows.values()),different_words32=sum(r['different_words32'] for r in rows.values()))
    result['passed']=result['different_words32']==0
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='variables'}))
    return 0 if result['passed'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('native','source','namelist','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument("--resources", type=Path)
    raise SystemExit(execute(p.parse_args()))
