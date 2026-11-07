import sys
from pathlib import Path
import json,hashlib,numpy as np,cupy as cp
from woof.core.kernels import get_kernel
from woof.core.wrf_stock_aer3 import optics_device
from woof.core.rrtmg_aerosol_optics import aer3_sw_optics_device
root=Path(sys.argv[1]).resolve();files=sorted((root/'capture-aer3').glob('*.bin'));p=[];a=[[] for _ in range(5)]
for f in files:
 b=f.read_bytes();nz=int(np.frombuffer(b[:4],'<i4')[0]);v=np.frombuffer(b[4:],'<f4');p.append(v[:nz]);j=nz+1
 for x in a:x.append(v[j:j+nz]);j+=nz
# Repeat real distinct captured columns to the 4,380-column forecast width.
pack=lambda x:cp.asarray(np.resize(np.asarray(x,dtype='f4'),(4380,nz)))
p=pack(p);t,qv,dz,nwfa,nifa=map(pack,a);n,nz=p.shape

def timing(fn):
 for _ in range(3):fn()
 s=cp.cuda.Stream(non_blocking=True)
 with s:
  s.begin_capture()
  for _ in range(20):fn()
  g=s.end_capture();a,b=cp.cuda.Event(),cp.cuda.Event();a.record();g.launch(s);b.record();b.synchronize()
 return cp.cuda.get_elapsed_time(a,b)/20
stock=lambda:optics_device(p,t,qv,dz,nwfa,nifa)
fork=lambda:aer3_sw_optics_device(p,t,qv,dz,nwfa,nifa,nz+1)
r={'columns':n,'levels':nz,'scope':'20-call CUDA graph, production adapters including device packing/allocation; repeated real WRF captured columns', 'aer3_stock_ms':timing(stock),'aer3_hrrr_ms':timing(fork)}
cu=get_kernel('cu_radiation','cu_radiation_merge');out=[cp.zeros((n,nz),cp.float32) for _ in range(4)];args=[cp.ones_like(p)*cp.float32(v) for v in (.00001,.000001,.2,.1)]
for mode in (1,2):
 r['cu_merge_mode'+str(mode)+'_ms']=timing(lambda:cu(((n*nz+127)//128,),(128,),(np.int64(n*nz),np.int32(mode),np.int32(1),np.int32(1),*args,*out)))
r['cu_nearest_ms']=timing(lambda:(cp.add(args[0],args[1],out=out[0]),cp.add(args[2],args[3],out=out[1])))
r['cu_nearest_scope']='two condensate additions, equal output volume; mode 2 additionally merges and publishes fractions' 
(root/'evidence/resume/feedback-aerosol-profile.json').write_text(json.dumps(r,indent=2));print(r)
