"""Isolate camrad packing and column physics from independently read gas values."""
from pathlib import Path
import argparse,ctypes,json
import numpy as np
from make_driver import arguments
from validate import load,compare
PARAMS={'ids':1,'ide':2,'jds':1,'jde':2,'kds':1,'kde':33,'ims':1,'ime':1,'jms':1,'jme':1,'kms':1,'kme':33,'its':1,'ite':1,'jts':1,'jte':1,'kts':1,'kte':32,'levsiz':59,'num_months':13,'paerlev':29,'naer_c':13,'cam_abs_dim1':4,'cam_abs_dim2':33,'n_cldadv':3}

def main():
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('translation',type=Path);p.add_argument('oracle',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 lib=ctypes.CDLL(str(a.translation/'camrad.so'));payload=(a.oracle/'state.bin').read_bytes();common=(ctypes.c_ubyte*len(payload)).in_dll(lib,'cam_state__');ctypes.memmove(ctypes.addressof(common),payload,len(payload));gas=(ctypes.c_double*5).in_dll(lib,'cam_gas_current')
 data=load(a.oracle);formal,decl=arguments(a.source.read_text(),'camrad');expected={};actual={}
 for column in range(1,65):
  for i,value in enumerate(data[('camrad',column,'input:gases')]):gas[i]=value
  buffers=[]
  for field in formal:
   if field in PARAMS:array=np.array([PARAMS[field]],np.int32)
   else:
    role='input' if ('camrad',column,'input:'+field) in data else 'output';array=data[('camrad',column,role+':'+field)]
    if 'ims:ime' in decl[field]:array=array[::2]
    dtype=np.int32 if decl[field].startswith(('integer','logical')) else np.float64 if decl[field].startswith('real(r8)') else np.float32
    array=array.astype(dtype).copy() if role=='input' else np.zeros(array.shape,dtype)
   buffers.append(array)
  lib.camrad_(*[ctypes.c_void_p(x.ctypes.data) for x in buffers])
  for field,array in zip(formal,buffers):
   key=('camrad',column,'output:'+field)
   if key not in data:continue
   reference=data[key];reference=reference[::2] if 'ims:ime' in decl[field] else reference
   expected[key]=reference;actual[key]=array.astype(np.float64)
 result=compare(expected,actual);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':result['pass'],'words':result['words'],'failed_fields':len(result['mismatches']),'max_ulp':max((x['max_ulp'] for x in result['mismatches']),default=0)}));print(result['mismatches'][:3]);raise SystemExit(not result['pass'])
if __name__=='__main__':main()
