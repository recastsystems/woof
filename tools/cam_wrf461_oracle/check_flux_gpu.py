"""Grade all returned CAM column flux and heating fields on a queued card."""
from pathlib import Path
import argparse,json,os
import numpy as np
from validate import load,compare

def main():
 p=argparse.ArgumentParser();p.add_argument('oracle',type=Path);p.add_argument('--spectrum',choices=('lw','sw','emissivity'),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 import cupy as cp
 from woof.core.kernels import get_kernel,module_options
 routine={'lw':'radclwmx','sw':'radcswmx','emissivity':'radems'}[a.spectrum]
 spectrum='radems' if a.spectrum=='emissivity' else a.spectrum
 root=Path(__file__).resolve().parents[2];meta=json.loads((root/'woof/core/kernels/cam3_emissivity.json').read_text())[routine]
 data=load(a.oracle);values=[];expected={};actual={}
 workspace=cp.empty((64,eval(meta['workspace_expression'],{'__builtins__':{}},{'CAM_LEVELS':32})),cp.uint8)
 state=cp.asarray(np.frombuffer((a.oracle/'state.bin').read_bytes(),np.uint8))
 for field in meta['parameters']:
  name=field['name'];role='output' if field['output'] else 'input';dtype={'int':np.int32,'float':np.float32,'double':np.float64}[field['dtype']]
  if name in ('doabsems','jj'):array=np.ones((64,field['count']),dtype)
  else:array=np.stack([data[(spectrum,i,role+':'+name)] for i in range(1,65)]).astype(dtype)
  if array.shape!=(64,field['count']):raise ValueError(name+' shape mismatch')
  values.append(cp.zeros_like(cp.asarray(array)) if role=='output' else cp.asarray(array))
 kernel={'lw':'cam_lw','sw':'cam_sw','emissivity':'cam_emissivity'}[a.spectrum]
 get_kernel('cam3_emissivity',kernel)((8,),(8,),(np.int32(64),state,workspace,*values));cp.cuda.Device().synchronize()
 for field,array in zip(meta['parameters'],values):
  if not field['output']:continue
  result=cp.asnumpy(array)
  for i in range(1,65):
   key=(spectrum,i,'output:'+field['name']);expected[key]=data[key];actual[key]=result[i-1].astype(np.float64)
 strict=compare(expected,actual);default=compare(expected,actual,strict=False)
 receipt={'strict':strict,'default':default,'exact_env':os.environ.get('GPUWM_WRF_EXACT','0'),'options':module_options('cam3_emissivity'),'device':cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode()}
 a.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'spectrum':spectrum,'pass':strict['pass'],'default_pass':default['pass'],'words':strict['words'],'failed_fields':len(strict['mismatches']),'max_ulp':max((x['max_ulp'] for x in strict['mismatches']),default=0)}));raise SystemExit(not strict['pass'])
if __name__=='__main__':main()
