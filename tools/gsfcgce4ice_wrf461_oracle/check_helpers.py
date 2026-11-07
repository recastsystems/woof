"""Grade actual standalone helper calls against WRF Fortran output words."""
from pathlib import Path
import argparse,json,hashlib,os
import numpy as np
from woof.core.gsfcgce4ice import launch_negcor, launch_auto_conversion, launch_refl10cm_gsfc, radar_table
from check import compare_words

def main():
 import cupy as cp
 ap=argparse.ArgumentParser();ap.add_argument('oracle',type=Path);ap.add_argument('--json',type=Path,required=True);ap.add_argument('--plant',action='store_true');args=ap.parse_args();root=args.oracle
 strict=os.environ.get('GPUWM_WRF_EXACT')=='1';records={};nc,nz,na=80,40,160;module=None
 if args.plant:
  from woof.core.kernels import module_source_int_defines,module_options
  from woof.wrf_exact import effective_options
  source=module_source_int_defines('gsfcgce4ice_aux',(('G4_AUX_KMAX',nz),));needle='1.9e11f * N';assert source.count(needle)==1
  assert np.float32(1.90000021504e11).view(np.uint32)-np.float32(1.9e11).view(np.uint32)==1
  module=cp.RawModule(code=source.replace(needle,'1.90000021504e11f * N'),options=effective_options(module_options('gsfcgce4ice_aux')))
  module.compile()
 def grade(name,a,b,**extra):
  field = ('refl_10cm' if name=='refl10cm_gsfc' else 're_cloud_gsfc' if name=='auto_radius' else 'qs' if name.startswith('negcor') else name)
  r=compare_words(field,np.ascontiguousarray(a,np.float32),np.ascontiguousarray(b,np.float32),strict=strict);r.update(extra);r['actual_sha256']=hashlib.sha256(a.tobytes()).hexdigest();r['expected_sha256']=hashlib.sha256(b.tobytes()).hexdigest();records[name]=r
 for n in range(1,6):
  if n<=3:
   raw=np.fromfile(root/f'neg{n}.input',np.float32).reshape(3,nz,1,nc);kwargs={};b=np.fromfile(root/f'neg{n}.expected',np.float32).reshape(nz,1,nc)
  else:
   raw=np.fromfile(root/f'neg{n}.input',np.float32).reshape(3,2,nz,40).transpose(0,2,1,3).copy();kwargs={'i_start':4,'i_end':36 if n==4 else 4,'j_start':0,'j_end':2,'k_start':1,'k_end':39};b=np.fromfile(root/f'neg{n}.expected',np.float32).reshape(2,nz,40).transpose(1,0,2).copy()
  x,rho,dz=[cp.asarray(a) for a in raw];launch_negcor(x,rho,dz,module=module,**kwargs);a=cp.asnumpy(x);grade(f'negcor-{n}',a,b)
 raw=np.fromfile(root/'auto.input',np.float32).reshape(2,na);liquid,number=[cp.asarray(a) for a in raw];rate=cp.zeros(na,cp.float32);radius=cp.full(na,-999.,cp.float32);launch_auto_conversion(liquid,number,rate,radius,module=module);a=cp.asnumpy(rate);b=np.fromfile(root/'auto.expected',np.float32).reshape(2,na);grade('auto_rate',a,b[0]);a=cp.asnumpy(radius);defined=(raw[0]>np.float32(1.e-32))&(raw[1]>0);grade('auto_radius',a[defined],b[1,defined],undefined_words=int((~defined).sum()))
 raw=np.fromfile(root/'radar.input',np.float32).reshape(6,nz,1,nc);fields=[cp.asarray(a) for a in raw];result=cp.empty((nz,1,nc),cp.float32);launch_refl10cm_gsfc(*fields,result,module=module);a=cp.asnumpy(result);b=np.fromfile(root/'radar.expected',np.float32).reshape(nz,1,nc);grade('refl10cm_gsfc',a,b)
 a=cp.asnumpy(radar_table());b=np.fromfile(root/'radar-init.expected',np.float64);records['radar_init']={'words':a.size,'different_words':int((a.view(np.uint64)!=b.view(np.uint64)).sum()),'actual_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'expected_sha256':hashlib.sha256(b.tobytes()).hexdigest(),'max_abs':float(np.max(np.abs(a-b)))}
 passed=all(r.get('rejected',0)==0 for r in records.values()) and (not strict or records['radar_init']['different_words']==0)
 report={'arm':'strict' if strict else 'default','passed':passed,'planted':args.plant,'records':records};args.json.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0 if (records['auto_rate']['rejected']>0 if args.plant else passed) else 1
if __name__=='__main__':raise SystemExit(main())
