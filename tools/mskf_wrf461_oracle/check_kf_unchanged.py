"""Compare cu=1 output bytes with a saved candidate module on real WRF input.

The baseline must be extracted from the exact pre-port candidate with git show.
This checks the closest column scheme; it does not claim whole-model identity.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from replay_native_capture import read_capture


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('--baseline-module',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import cupy as cp
    from woof.core import kf
    h,sc,coords,values,initial,expected=read_capture(args.capture)
    nz,ncol=map(int,h[:2])
    def dev3(x):return cp.asarray(np.ascontiguousarray(x.T.reshape(nz,1,ncol)))
    arrays=[dev3(values[:,j*nz:(j+1)*nz]) for j in range(9)]
    spec=importlib.util.spec_from_file_location('woof.core._mskf_baseline_kf',args.baseline_module)
    baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
    # Resolve the same pinned coefficient asset; only Python source differs.
    baseline.__file__=kf.__file__
    def run(module):
        return module.launch_kf(*arrays,dt=float(sc[0]),dx=float(sc[1]),cudt_minutes=float(sc[2]),
            phase_mode=module.KFPhaseMode.SEPARATE_ICE_SNOW,f_qr=bool(h[6]),cloud_diagnostics=True)
    old,new=run(baseline),run(kf)
    fields={}
    for key in sorted(old.keys()|new.keys()):
        a,b=cp.asnumpy(old[key]),cp.asnumpy(new[key])
        fields[key]={'shape':list(a.shape),'dtype':str(a.dtype),
            'candidate_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
            'port_sha256':hashlib.sha256(b.tobytes()).hexdigest(),
            'differing_bytes':int(np.count_nonzero(a.view('u1')!=b.view('u1')))}
    receipt={'baseline_sha256':hashlib.sha256(args.baseline_module.read_bytes()).hexdigest(),
             'capture':args.capture.name,'columns':ncol,'nz':nz,'fields':fields,
             'bytes':sum(cp.asnumpy(value).nbytes for value in old.values()),
             'differing_bytes':sum(value['differing_bytes'] for value in fields.values())}
    args.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='fields'}))
    return bool(receipt['differing_bytes'])

if __name__=='__main__':sys.exit(main())
