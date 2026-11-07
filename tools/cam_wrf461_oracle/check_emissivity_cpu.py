"""Grade translated C statements against isolated WRF emissivity outputs."""
from pathlib import Path
import argparse,ctypes,json,sys
import numpy as np
from make_driver import arguments
from validate import load,compare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('translation',type=Path)
    parser.add_argument('oracle',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    library=ctypes.CDLL(str(args.translation/'radems.so'))
    payload=(args.oracle/'state.bin').read_bytes()
    layout=json.loads((args.translation/'state-layout.json').read_text())
    size=sum(item['bytes'] for item in layout)
    if size!=len(payload):
        raise ValueError(f'state layout {size} differs from WRF capture {len(payload)}')
    common=(ctypes.c_ubyte*size).in_dll(library,'cam_state__')
    ctypes.memmove(ctypes.addressof(common),payload,size)
    fields=load(args.oracle)
    formal,declarations=arguments(args.source.read_text(),'radems')
    expected={key:value for key,value in fields.items() if key[0]=='radems' and key[2].startswith('output:')}
    actual={}
    for column in range(1,65):
        buffers=[]
        for arg in formal:
            if arg in ('lchnk','ncol','pcols','pver','pverp'):
                value={'lchnk':1,'ncol':1,'pcols':1,'pver':32,'pverp':33}[arg]
                buffers.append(np.array([value],dtype=np.int32))
            else:
                role='output' if 'intent(out)' in declarations[arg] else 'input'
                values=fields[('radems',column,role+':'+arg)]
                buffers.append(np.zeros_like(values) if role=='output' else values.copy())
        library.radems_(*[ctypes.c_void_p(value.ctypes.data) for value in buffers])
        for arg,value in zip(formal,buffers):
            if 'intent(out)' in declarations[arg]:
                actual[('radems',column,'output:'+arg)]=value
    result=compare(expected,actual)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pass':result['pass'],'words':result['words'],'failed_fields':len(result['mismatches']),
                      'max_ulp':max((r['max_ulp'] for r in result['mismatches']),default=0)}))
    for item in result['mismatches'][:5]:print(item)
    raise SystemExit(not result['pass'])


if __name__=='__main__':main()
