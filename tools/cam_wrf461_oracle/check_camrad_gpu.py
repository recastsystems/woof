"""Grade the full CUDA CAM driver against unmodified WRF camrad."""
from pathlib import Path
import argparse,json,os
import numpy as np
from check_camrad_cpu import PARAMS
from make_driver import arguments
from validate import load,compare

def main():
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('oracle',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--levels',type=int,default=32);p.add_argument('--native-tables',type=Path);p.add_argument('--workers',type=int,default=64);a=p.parse_args()
 import cupy as cp
 from woof.core.kernels import get_kernel_int_defines,module_options
 params={**PARAMS,'kte':a.levels,'kde':a.levels+1,'kme':a.levels+1,'cam_abs_dim2':a.levels+1}
 root=Path(__file__).resolve().parents[2];meta=json.loads((root/'woof/core/kernels/cam3_emissivity.json').read_text())['camrad'];data=load(a.oracle);_,decl=arguments(a.source.read_text(),'camrad');values=[];expected={};actual={}
 if a.workers<1 or a.workers%8:raise ValueError('--workers must be a positive multiple of 8')
 work_bytes=eval(meta['workspace_expression'],{'__builtins__':{}},{'CAM_LEVELS':a.levels});workspace=cp.empty((a.workers,work_bytes),cp.uint8)
 tables=None
 if a.native_tables:
  from woof.core.cam3_tables import Cam3Tables
  tables=Cam3Tables(a.native_tables)
 physical=np.array([9.81,1004.5,0.622,5.67e-8,287.,461.6,0.622,2.501e6,3.336e5,461.6,1004.5,273.16],np.float64)
 state=cp.asarray(tables.initialized_state(physical,3) if tables else np.frombuffer((a.oracle/'state.bin').read_bytes(),np.uint8))
 for field in meta['parameters']:
  name=field['name'];dtype={'int':np.int32,'float':np.float32,'double':np.float64}[field['dtype']]
  if name in params:array=np.full((64,1),params[name],dtype)
  else:
   role='input' if ('camrad',1,'input:'+name) in data else 'output'
   if name=='gases' and tables:
    array=np.stack([tables.gases(data[('camrad',i,'input:yr')][0],data[('camrad',i,'input:julian')][0]) for i in range(1,65)])
   else:array=np.stack([data[('camrad',i,role+':'+name)] for i in range(1,65)])
   if name!='gases' and 'ims:ime' in decl[name]:array=array[:,::2]
   array=array.astype(dtype) if role=='input' else np.zeros(array.shape,dtype)
  count=eval(field.get('count_expression',str(field['count'])),{'__builtins__':{}},{'CAM_LEVELS':a.levels})
  if array.shape!=(64,count):raise ValueError(str((name,array.shape,field['count'])))
  values.append(cp.asarray(array))
 get_kernel_int_defines('cam3_emissivity','cam_driver',(('CAM_LEVELS',a.levels),))((a.workers//8,),(8,),(np.int32(64),state,workspace,*values));cp.cuda.Device().synchronize()
 for field,array in zip(meta['parameters'],values):
  name=field['name']
  if ('camrad',1,'output:'+name) not in data:continue
  result=cp.asnumpy(array).astype(np.float64)
  for i in range(1,65):
   key=('camrad',i,'output:'+name);reference=data[key];reference=reference[::2] if 'ims:ime' in decl[name] else reference
   expected[key]=reference;actual[key]=result[i-1]
 bad=[{'field':str(k),'gpu_nonfinite':int((~np.isfinite(v)).sum()),'oracle_nonfinite':int((~np.isfinite(expected[k])).sum())} for k,v in actual.items() if not np.isfinite(v).all() or not np.isfinite(expected[k]).all()]
 if bad:
  a.out.write_text(json.dumps({'pass':False,'nonfinite_fields':bad},indent=2)+'\n');print(bad[:10]);raise SystemExit(2)
 key=('camrad',1,'output:rthratenlw');changed=actual[key].astype(np.float32);changed.view(np.uint32)[0]^=np.uint32(1)
 planted=dict(actual);planted[key]=changed.astype(np.float64);mutation=compare(actual,planted)
 if mutation['pass'] or len(mutation['mismatches'])!=1:raise RuntimeError('one-bit CAM driver corruption was not isolated')
 strict=compare(expected,actual);default=compare(expected,actual,strict=False)
 receipt={'workers':a.workers,'mutation_caught':not mutation['pass'],'mutation_native_dtype':'float32','mutation_native_changed_bits':1,'strict':strict,'default':default,'exact_env':os.environ.get('GPUWM_WRF_EXACT','0'),'options':module_options('cam3_emissivity'),'note':'native float32 driver outputs stored as exact binary64 values for the comparator; gas inputs provided by native WOOF tables when --native-tables is set'}
 a.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'pass':strict['pass'],'default_pass':default['pass'],'words':strict['words'],'failed_fields':len(strict['mismatches']),'max_ulp':max((x['max_ulp'] for x in strict['mismatches']),default=0)}));print(strict['mismatches'][:4]);raise SystemExit(not (strict if os.environ.get('GPUWM_WRF_EXACT')=='1' else default)['pass'])
if __name__=='__main__':main()
