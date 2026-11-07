"""Compare every isolated WRF emissivity output on the queued GPU."""
from pathlib import Path
import argparse,json,os
import numpy as np
from validate import load,compare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('oracle',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import cupy as cp
    from woof.core.kernels import get_kernel,module_options
    root=Path(__file__).resolve().parents[2]
    metadata=json.loads((root/'woof/core/kernels/cam3_emissivity.json').read_text())['radems']
    data=load(args.oracle)
    values=[];expected={};actual={};roles=[]
    state=cp.asarray(np.frombuffer((args.oracle/'state.bin').read_bytes(),dtype=np.uint8))
    for field in metadata['fields']:
        role='output' if ('radems',1,'output:'+field) in data else 'input'
        array=np.stack([data[('radems',i,role+':'+field)] for i in range(1,65)])
        values.append(cp.zeros(array.shape,dtype=cp.float64) if role=='output' else cp.asarray(array))
        roles.append(role)
    workspace=cp.empty((64,eval(metadata['workspace_expression'],{'__builtins__':{}},{'CAM_LEVELS':32})),cp.uint8)
    get_kernel('cam3_emissivity','cam_emissivity')((8,),(8,),(np.int32(64),state,workspace,*values))
    cp.cuda.Device().synchronize()
    for field,role,array in zip(metadata['fields'],roles,values):
        if role!='output':continue
        result=cp.asnumpy(array)
        for i in range(1,65):
            key=('radems',i,'output:'+field)
            expected[key]=data[key];actual[key]=result[i-1]
    strict=compare(expected,actual);default=compare(expected,actual,strict=False)
    receipt={'strict':strict,'default':default,'exact_env':os.environ.get('GPUWM_WRF_EXACT','0'),
             'options':module_options('cam3_emissivity'),
             'device':cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode()}
    args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'pass':strict['pass'],'words':strict['words'],'failed_fields':len(strict['mismatches']),
                     'max_ulp':max((x['max_ulp'] for x in strict['mismatches']),default=0)}))
    raise SystemExit(not strict['pass'])


if __name__=='__main__':main()
