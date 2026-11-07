"""Compare the actual shared-loader GPU kernel with pristine WRF columns."""
import argparse
import hashlib
import json
from pathlib import Path
import cupy as cp
import numpy as np
from woof.core.kernels import get_kernel, module_source

def run(oracle, mutation=False):
    rows=np.loadtxt(oracle,skiprows=1)
    if mutation:
        original=np.float32(0.000277778).view(np.uint32).item()
        changed=np.array([original ^ 1],np.uint32).view(np.float32)[0]
        src=module_source('irrigation').replace('0.000277778f',float(changed).hex()+'f')
        from woof.core.kernels import MODULE_KEY_ROOT, _compile_observed
        from woof.certify.kernel_manifest import record_module
        mod=cp.RawModule(code=src,options=('-std=c++17',),name_expressions=None)
        _compile_observed(mod, MODULE_KEY_ROOT+':irrigation_mutation')
        record_module(MODULE_KEY_ROOT+':irrigation_mutation',source=src,options=('-std=c++17',),module=mod)
        kernel=mod.get_function('irrigation_column')
    else:
        kernel=get_kernel('irrigation','irrigation_column')
    results={str(o):{name:{'max_ulp':0,'mismatches':0,'changed':0} for name in ('rainbl','channel','qr')} for o in (1,2,3)}
    digest=hashlib.sha256()
    for row in rows:
        option,c,step,i,j,ph,freq,sh,nh,sj,ej,rnd=map(int,row[:12])
        f,amount,jul,xt,gmt,dt,rho,dz=row[12:20].astype(np.float32)
        expected=row[20:].astype(np.float32)
        if step==1:
            initial=np.float32(0.01 if c%13==0 else (0.010000001 if c%11==0 else 0.))
            rain=cp.asarray([initial],dtype=cp.float32)
            chan=cp.asarray([123.],dtype=cp.float32)
            qr=cp.asarray([0.00002],dtype=cp.float32)
            fraction=cp.asarray([f]); random=cp.asarray([rnd],dtype=cp.int32)
            density=cp.asarray([rho]); depth=cp.asarray([dz])
        before=np.array([rain.item(),chan.item(),qr.item()],np.float32)
        args=(fraction,random,rain,chan,qr,density,depth,np.int32(option),amount,
              *map(np.int32,(sh,nh,sj,ej,freq,ph)),jul,xt,gmt,dt,
              *map(np.int32,(1,1,i,j,0)))
        kernel((1,),(32,),args)
        actual=np.array([rain.item(),chan.item(),qr.item()],np.float32)
        digest.update(actual.tobytes())
        ulp=np.abs(actual.view(np.uint32).astype(np.int64)-expected.view(np.uint32).astype(np.int64))
        for k,name in enumerate(('rainbl','channel','qr')):
            entry=results[str(option)][name]
            entry['max_ulp']=max(entry['max_ulp'],int(ulp[k]))
            entry['mismatches']+=int(ulp[k]!=0)
            entry['changed']+=int(actual[k]!=before[k])
    return {'columns_per_option':192,'steps':4,'rows':len(rows),'mutation':mutation,'constant_bit_xor':1 if mutation else 0,'fields':results,'full_output_sha256':digest.hexdigest(),'pass':all(v['max_ulp']==0 for o in results.values() for v in o.values())}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('oracle'); p.add_argument('receipt'); p.add_argument('--mutation',action='store_true'); a=p.parse_args()
    result=run(a.oracle,a.mutation); Path(a.receipt).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['pass'] != a.mutation else 1)
