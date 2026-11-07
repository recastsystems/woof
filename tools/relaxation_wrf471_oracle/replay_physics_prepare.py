"""Replay real native dry physics preparation feeding Held--Suarez."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.relaxation_wrf471_oracle.replay_geometry import read_capture
from tools.relaxation_wrf471_oracle.replay import grade


def run(directory):
    import cupy as cp
    from woof.core.kernels import load_module_no_ftz
    schema=json.loads((directory/'schema.json').read_text())['physics_prepare']
    dims,data,expected,capture=read_capture(directory/'physics_prepare.bin',schema)
    nx,ny,nz=dims
    a={name:cp.asarray(np.ascontiguousarray(value))for name,value in data.items()if value.ndim}
    a.update({name:cp.empty(value.shape,cp.float32)for name,value in expected.items()})
    load_module_no_ftz('global_geometry').get_function('global_physics_prepare')(
        ((nx*ny*nz+127)//128,),(128,),(*map(np.int32,dims),np.float32(data['rcp']),
        *(a[n]for n in ('p','pb','ph','phb','t','pressure','exner','temperature','ps'))))
    outputs={name:grade(cp.asnumpy(a[name]),reference)for name,reference in expected.items()}
    return dict(capture=capture,outputs=outputs,words=sum(g['words']for g in outputs.values()),
                different_words=sum(g['different_words']for g in outputs.values()),
                passed=all(g['different_words']==0 for g in outputs.values()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('receipt',type=Path)
    args=parser.parse_args();result=run(args.directory)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed']else 1)
