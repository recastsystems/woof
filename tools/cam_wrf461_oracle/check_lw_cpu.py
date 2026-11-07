"""Compare the entire translated LW column with unmodified WRF outputs."""
from pathlib import Path
import argparse,ctypes,json
import numpy as np
from make_driver import arguments
from validate import load,compare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('translation',type=Path)
    parser.add_argument('oracle',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--normalized',action='store_true')
    args=parser.parse_args()
    library=ctypes.CDLL(str(args.translation/('lw_normalized.so' if args.normalized else 'lw.so')))
    payload=(args.oracle/'state.bin').read_bytes()
    common=(ctypes.c_ubyte*len(payload)).in_dll(library,'cam_state_' if args.normalized else 'cam_state__')
    ctypes.memmove(ctypes.addressof(common),payload,len(payload))
    data=load(args.oracle)
    formal,decl=arguments(args.source.read_text(),'radclwmx')
    expected={key:value for key,value in data.items() if key[0]=='lw' and key[2].startswith('output:')}
    actual={}
    for column in range(1,65):
        buffers=[]
        for field in formal:
            if field in ('lchnk','ncol','pcols','pver','pverp'):
                buffers.append(np.array([{'lchnk':1,'ncol':1,'pcols':1,'pver':32,'pverp':33}[field]],np.int32))
            elif field=='doabsems':buffers.append(np.array([1],np.int32))
            elif field=='nmxrgn':
                buffers.append(data[('lw',column,'input:nmxrgn')].astype(np.int32))
            else:
                role='output' if 'intent(out)' in decl[field] else 'input'
                array=data[('lw',column,role+':'+field)]
                buffers.append(np.zeros_like(array) if role=='output' else array.copy())
        library.radclwmx_(*[ctypes.c_void_p(x.ctypes.data) for x in buffers])
        for field,value in zip(formal,buffers):
            if 'intent(out)' in decl[field]:actual[('lw',column,'output:'+field)]=value
    result=compare(expected,actual)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pass':result['pass'],'words':result['words'],'failed_fields':len(result['mismatches']),
                      'max_ulp':max((x['max_ulp'] for x in result['mismatches']),default=0)}))
    for row in result['mismatches'][:4]:print(row)
    raise SystemExit(not result['pass'])


if __name__=='__main__':main()
