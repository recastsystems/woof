"""Replay a native CU_KSAS capture. Run under GPUWM_NUMERICS=strict."""
from pathlib import Path
import argparse
import sys, json,hashlib,platform,os
import numpy as np
import cupy as cp
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('inputs',type=Path)
parser.add_argument('native_outputs',type=Path)
parser.add_argument('--receipt',type=Path,required=True)
args=parser.parse_args()
import gen_cases
from woof.core.sas import SAS_VARIANTS,launch_column
from woof.core.kernels import module_source,module_options
groups=gen_cases.read(args.inputs)
answers=gen_cases.read_outputs(args.native_outputs,groups)
ledger={}
for g,want in zip(groups,answers):
    a=g['arrays']; names=('t3d','qv3d','qc3d','qi3d','p3d','pi3d','rho3d','dz8w','u3d','v3d','w','xland','hpbl','hpbl_hold')
    inputs={name:cp.asarray(a[name]) for name in names}
    out=launch_column(SAS_VARIANTS[14],inputs=inputs,znu=cp.asarray(g['znu']),dt=g['dt'],dx=g['dx'],stepcu=g['stepcu'],mp_physics=g['mp_physics'],f_qc=g['f_qc'],f_qi=g['f_qi'])
    for name in want:
        if name=='rqicuten' and not g['f_qi'] or name=='rqccuten' and not g['f_qc']: continue
        got=cp.asnumpy(out[name]); different=got.view(np.uint32)!=want[name].view(np.uint32)
        row={'words':int(different.size),'differing_words':int(different.sum()),'nonfinite_native':int((~np.isfinite(want[name])).sum()),'nonfinite_gpu':int((~np.isfinite(got)).sum()),'native_nonzero':int((want[name]!=0).sum()),'max_absolute_difference':float(np.nanmax(np.abs(got-want[name])))}
        flat=np.flatnonzero(different)[:12]
        row['first_differences']=[{'offset':int(i),'native':hex(int(want[name].view(np.uint32).ravel()[i])),'gpu':hex(int(got.view(np.uint32).ravel()[i]))} for i in flat]
        ledger[name]=row
receipt={'scope':'native forecast CU_KSAS module call, MPI rank 0, after 20 real model steps', 'groups':[ {k:float(v) if isinstance(v,np.floating) else v for k,v in g.items() if k not in ('arrays','znu')} for g in groups], 'words':sum(v['words'] for v in ledger.values()),'differing_words':sum(v['differing_words'] for v in ledger.values()),'ledger':ledger,'environment':{k:os.environ.get(k) for k in ('GPUWM_NUMERICS','GPUWM_WRF_EXACT','GPUWM_WRF_GLIBC','CUDA_PATH')},'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'libc':platform.libc_ver(),'module_options':module_options('sas'),'module_source_sha256':hashlib.sha256(module_source('sas').encode()).hexdigest(),'inputs_sha256':hashlib.sha256(args.inputs.read_bytes()).hexdigest(),'outputs_sha256':hashlib.sha256(args.native_outputs.read_bytes()).hexdigest()}
args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
assert receipt['differing_words']==0
