"""Compile-only local frame measurement, without a CUDA device or context."""
from pathlib import Path
import argparse, hashlib, json, re, subprocess
from cupy_backends.cuda.libs import nvrtc
from woof.core import kernels
from woof.certify.compile_platform import nvrtc_build, nvrtc_build_id
from woof.nvrtc_cache_key import compile_program

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ptxas',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
assembler=args.ptxas.resolve()
out=args.output.resolve()
out.mkdir(parents=True,exist_ok=True)
rows={}
for name in ('qnsesfc','qnsepbl'):
    source=kernels.module_source(name)
    options=(*kernels.module_options(name),'--gpu-architecture=compute_89')
    # Keyed compile (A160): the fixed program name cannot be answered from
    # the ComputeCache with PTX built under the other flush mode.
    image=compile_program(source,options,name=name+'.cu',nvrtc=nvrtc)
    ptx=out/(name+'.ptx')
    ptx.write_bytes(image)
    command=[str(assembler),'-v','--gpu-name=sm_89',str(ptx),'-o',str(out/(name+'.cubin'))]
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    log=result.stdout+result.stderr
    (out/(name+'.log')).write_text(log)
    entries={}
    entry=None
    for line in log.splitlines():
        if m:=re.search(r"Compiling entry function '([^']+)'",line): entry=m.group(1)
        if m:=re.search(r'(\d+) bytes stack frame',line):
            assert entry
            entries[entry]=int(m.group(1))
    assert entries,(name,log)
    rows[name]={'local_size_bytes':max(entries.values()),'entries':entries,'command':command,
                'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                'ptx_sha256':hashlib.sha256(image).hexdigest(),'options':options,'log':log}
data={'measurement':'compile-only NVRTC PTX plus ptxas; no GPU context or execution',
      'compute_capability':'89','nvrtc_build':nvrtc_build(),'nvrtc_build_id':nvrtc_build_id(),
      'assembler':subprocess.check_output([str(assembler),'--version'],text=True),
      'hostname':subprocess.check_output(['hostname'],text=True).strip(),'rows':rows}
(out/'compile-only-qnse-frames.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
