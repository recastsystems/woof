"""Replay real native trajectory-driver input buffers, not later history."""
from pathlib import Path
import json,os,sys
import numpy as np
import cupy as cp
import netCDF4
from woof.core.kernels import get_kernel
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tracer_wrf461_oracle'))
from replay import Stream,compare,digest

def load_capture(path):
 stream=Stream(path);version,domain,n,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme=map(int,stream.read((15,),'i4'))
 capacity=1000;coords=stream.read((5,capacity));shape=(jme-jms+1,kme-kms+1,ime-ims+1)
 names=('QVAPOR','tr17_1','tr17_8','P','PB','THM','PH','PHB','U','V','W')
 nx,ny,nz=ide-ids,jde-jds,kde-kds
 arrays={}
 for name in names:
  a=stream.read(shape);sx=int(name=='U');sy=int(name=='V');sz=int(name in ('PH','PHB','W'))
  a=a[jds-jms:jds-jms+ny+sy,kds-kms:kds-kms+nz+sz,ids-ims:ids-ims+nx+sx].transpose(1,0,2)
  arrays[name]=cp.asarray(np.ascontiguousarray(a))
 stream.finish()
 return domain,n,nx,ny,nz,coords,arrays

def replay(path):
 domain,n,nx,ny,nz,coords,arrays=load_capture(path)
 capacity=1000
 variables=[('QVAPOR',None,None,0),('tr17_1',None,None,0),('tr17_8',None,None,0),('P','PB',None,4),('THM','P','PB',5),('PH','PHB',None,6),('U',None,None,1),('V',None,None,2),('W',None,None,3)]
 ptrs=cp.asarray(np.asarray([[arrays[a].data.ptr,arrays[b].data.ptr if b else 0,arrays[p].data.ptr if p else 0] for a,b,p,k in variables],np.uint64))
 desc=cp.asarray(np.asarray([[k,arrays[a].shape[2],arrays[a].shape[1],0] for a,b,p,k in variables],np.int32));out=cp.empty((14,n),np.float32)
 get_kernel('trajectory','trajectory_sample')(((14*n+127)//128,),(128,),
  (cp.asarray(coords),np.int32(capacity),np.int32(n),ptrs,desc,cp.ones((9,n),np.int32),cp.ones(n,np.int32),out,np.int32(0),np.int32(9),np.int32(nx),np.int32(ny),np.int32(nz)))
 got=out.get();rows=[]
 with netCDF4.Dataset(path.parent/f'wrfout_traj_d{domain:02d}') as points:
  keys=['traj_'+k for k in ('i','j','k','long','lat')]+[k+'_traj' for k in ('QVAPOR','tr17_1','tr17_8','p','T','z','u','v','w')]
  for i,key in enumerate(keys):
   ref=np.asarray(points[key][0],np.float32);row=compare(got[i],ref)
   row.update(domain=domain,field=key,reference=ref.tolist(),computed=got[i].tolist());rows.append(row)
 return rows

if __name__=='__main__':
 if os.environ.get('GPUWM_WRF_EXACT')!='1':raise ValueError('strict proof requires GPUWM_WRF_EXACT=1')
 root=Path(sys.argv[1]);files=sorted(root.glob('trajectory-sampler-*.in.bin'));rows=[row for path in files for row in replay(path)]
 receipt=dict(rows=rows,words=sum(r['words'] for r in rows),differing_words=sum(r['differing_words'] for r in rows),captures={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files})
 Path(sys.argv[2]).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
 if not rows or receipt['differing_words']:raise SystemExit(1)
