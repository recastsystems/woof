"""Grade the staged radtpl CUDA kernel against the real 64-column artifact."""
from pathlib import Path
import argparse
import json
import numpy as np
from woof.core.cam3_paths import lw_paths, PATH_FIELDS
from validate import load, compare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('oracle',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import cupy as cp
    data=load(args.oracle)
    def columns(field):
        return np.stack([data[('lw',i,'input:'+field)] for i in range(1,65)])
    result=lw_paths(columns('tnm'),columns('qnm'),columns('pint'),columns('pmln'),
                    columns('piln'),columns('lwupcgs')[:,0],data[('constants',1,'input:radtpl')])
    cp.cuda.Device().synchronize()
    expected={key:value for key,value in data.items() if key[0]=='paths'}
    actual={}
    for field in PATH_FIELDS:
        values=cp.asnumpy(result[field])
        if values.ndim==3:
            values=values.transpose(0,2,1).reshape(64,-1)
        for i in range(1,65):
            actual[('paths',i,'output:'+field)]=values[i-1].copy()
    strict=compare(expected,actual)
    default=compare(expected,actual,strict=False)
    key=('paths',1,'output:s2c')
    mutated={key:value.copy() for key,value in actual.items()}
    mutated[key].view(np.uint64)[0]^=np.uint64(1)
    mutation=compare(expected,mutated)
    import os
    from woof.core.kernels import module_options
    receipt={'strict':strict,'default':default,'mutation_detected':not mutation['pass'],
             'exact_env':os.environ.get('GPUWM_WRF_EXACT','0'),
             'device':cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode(),
             'options':module_options('cam3_paths')}
    args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'pass':strict['pass'],'words':strict['words'],
                      'max_ulp':max((x['max_ulp'] for x in strict['mismatches']),default=0),
                      'mutation_detected':receipt['mutation_detected']}))
    raise SystemExit(0 if strict['pass'] and receipt['mutation_detected'] else 1)


if __name__=='__main__':
    main()
