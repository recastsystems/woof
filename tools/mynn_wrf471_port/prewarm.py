#!/usr/bin/env python3
"""Build source-bound MYNN images without a serial NVRTC cold forecast."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from woof.core.mynn471_gpu import source, DATA

p=argparse.ArgumentParser();p.add_argument('--nz',type=int,default=49)
p.add_argument('--arch',default='sm_120');p.add_argument('--out',type=Path,default=DATA/'images')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
text=source(a.nz);digest=sha256(text.encode()).hexdigest()
def build(exact):
 mode='strict' if exact else 'fast'
 binary=a.out/f'{a.arch}-nz{a.nz}-{mode}-{digest[:16]}.cubin'
 unit=binary.with_suffix('.cu');unit.write_text(text)
 command=['/usr/local/cuda/bin/nvcc','--cubin','-std=c++17','-arch='+a.arch,
          '--ftz=false','--fmad='+('false' if exact else 'true'),'--prec-div=true',
          '--prec-sqrt=true','--split-compile=4']
 if exact:command+=['-DGPUWM_WRF_EXACT=1']
 command += [str(unit),'-o',str(binary)]
 with binary.with_suffix('.log').open('w') as log:
  subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 manifest=dict(source_sha256=digest,image_sha256=sha256(binary.read_bytes()).hexdigest(),
               mode=mode,nz=a.nz,arch=a.arch,command=command)
 binary.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
 unit.unlink();print(json.dumps(manifest),flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(build,(True,False)))
