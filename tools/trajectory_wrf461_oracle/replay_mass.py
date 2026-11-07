"""Replay native face mass, including the physical boundary faces."""
from pathlib import Path
import sys,json,os
import numpy as np,cupy as cp
from woof.core.kernels import get_kernel
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tracer_wrf461_oracle'))
from replay import Stream,compare
rows=[]
if os.environ.get('GPUWM_WRF_EXACT')!='1':raise ValueError('strict proof requires GPUWM_WRF_EXACT=1')
for path in Path(sys.argv[1]).glob('trajectory-[0-9][0-9].in.bin'):
 s=Stream(path);h=s.read((21,),'i4');_,domain,n,*b=map(int,h)
 ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,*_=b
 s.read((3,));s.read((5,1000));shape=(jme-jms+1,kme-kms+1,ime-ims+1)
 for _ in range(3):s.read(shape)
 a=[s.read((shape[0],shape[2])) for _ in range(6)]
 nx=ide-ids;ny=jde-jds
 mut=cp.asarray(np.ascontiguousarray(a[0][jds-jms:jds-jms+ny,ids-ims:ids-ims+nx]))
 muu=cp.empty((ny,nx+1),cp.float32);muv=cp.empty((ny+1,nx),cp.float32)
 get_kernel('trajectory','trajectory_face_mass')(((max(muu.size,muv.size)+127)//128,),(128,),(mut,muu,muv,np.int32(nx),np.int32(ny)))
 for key,out,ref in [('muus',cp.asnumpy(muu),a[1][jds-jms:jds-jms+ny,ids-ims:ids-ims+nx+1]),('muvs',cp.asnumpy(muv),a[2][jds-jms:jds-jms+ny+1,ids-ims:ids-ims+nx])]:
  row=compare(out,ref);row.update(domain=domain,field=key);rows.append(row)
r=dict(rows=rows,words=sum(r['words'] for r in rows),differing_words=sum(r['differing_words'] for r in rows))
Path(sys.argv[2]).write_text(json.dumps(r,indent=2)+'\n');print(r)
assert rows and r['differing_words']==0
