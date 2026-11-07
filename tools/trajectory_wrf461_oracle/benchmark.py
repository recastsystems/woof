"""CUDA event timings of real-input trajectory stages, default fast mode."""
from pathlib import Path
import json,sys
import numpy as np
import cupy as cp
from woof.core.kernels import get_kernel
from woof.wrf_exact import ENABLED
from replay_sampler import load_capture
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tracer_wrf461_oracle'))
from replay import Stream
from benchmark import timed,transport

def main(root,output,sampler_root):
 if ENABLED:raise ValueError('timings require fast numerics')
 path=root/'trajectory-01.in.bin';s=Stream(path)
 _,_,count,*b=map(int,s.read((21,),'i4'))
 ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,*_=b
 dt,rdx,rdy=s.read((3,));initial=s.read((5,1000));shape=(jme-jms+1,kme-kms+1,ime-ims+1)
 three=[s.read(shape) for _ in range(3)];two=[s.read((shape[0],shape[2])) for _ in range(6)];vertical=[s.read((shape[1],)) for _ in range(5)];code=int(s.read((1,),'i4')[0]);proj=cp.asarray(s.read((14,)));s.finish()
 nx,ny,nz=ide-ids,jde-jds,kde-kds
 def dev(a,stagger=''):
  sx=int(stagger=='u');sy=int(stagger=='v');sz=int(stagger=='w')
  a=a[jds-jms:jds-jms+ny+sy,kds-kms:kds-kms+nz+sz,ids-ims:ids-ims+nx+sx].transpose(1,0,2) if a.ndim==3 else a[jds-jms:jds-jms+ny+sy,ids-ims:ids-ims+nx+sx]
  return cp.asarray(np.ascontiguousarray(a))
 fields=[dev(a,t) for a,t in zip(three,('u','v','w'))]+[dev(a,t) for a,t in zip(two,('','u','v','','u','v'))]
 vert=[cp.asarray(a[kds-kms:kds-kms+nz+1]) for a in vertical]
 dims=tuple(np.int32(x) for x in (nx,ny,nz));rows=[]
 def launch(name,n,args):get_kernel('trajectory',name)(((n+127)//128,),(128,),args)
 for n in (count,1000):
  coords=cp.asarray(np.tile(initial[:,:count],(1,(1000+count-1)//count))[:,:1000].copy())
  moving=cp.ones(n,np.int32);enabled=cp.ones(n,np.int32);intervals=cp.asarray(np.tile([0.,21600.,1.],(n,1)))
  _,_,_,_,_,_,arrays=load_capture(sampler_root/'trajectory-sampler-01.in.bin')
  variables=[('QVAPOR',None,None,0),('tr17_1',None,None,0),('tr17_8',None,None,0),('P','PB',None,4),('THM','P','PB',5),('PH','PHB',None,6),('U',None,None,1),('V',None,None,2),('W',None,None,3)]
  ptrs=cp.asarray(np.asarray([[arrays[a].data.ptr,arrays[b].data.ptr if b else 0,arrays[p].data.ptr if p else 0] for a,b,p,k in variables],np.uint64))
  desc=cp.asarray(np.asarray([[k,arrays[a].shape[2],arrays[a].shape[1],0] for a,b,p,k in variables],np.int32))
  requested=cp.ones((9,n),np.int32);out=cp.empty((14,n),cp.float32)
  clock=lambda:launch('trajectory_clock',n,(intervals,np.float64(3600.),moving,enabled,np.int32(n)))
  face=lambda:launch('trajectory_face_mass',max(fields[4].size,fields[5].size),(fields[3],fields[4],fields[5],*dims[:2]))
  # dt=0 holds the captured particles fixed across timing repetitions;
  # interpolation, projection and eta searches still run normally.
  advance=lambda:launch('trajectory_advance',n,(coords,np.int32(1000),np.int32(n),moving,np.int32(code),proj,np.float32(0.),rdx,rdy,*fields,*vert,*dims))
  sample9=lambda:launch('trajectory_sample',14*n,(coords,np.int32(1000),np.int32(n),ptrs,desc,requested,enabled,out,np.int32(0),np.int32(9),*dims))
  values={key:timed(call,100) for key,call in [('clock',clock),('face_mass',face),('advance',advance),('sample_nine_variables',sample9)]}
  values['total']=sum(values[k] for k in ('clock','face_mass','advance','sample_nine_variables'))
  rows.append(dict(particles=n,milliseconds=values))
 nearest=transport(tracer=True,mp=0)
 receipt=dict(mode='fast',gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),units='ms per stage',dimensions=[nz,ny,nx],rows=rows,nearest='existing passive scalar transport (eight tracers plus three water fields)',nearest_milliseconds=nearest,ratios_to_nearest_pd={str(row['particles']):row['milliseconds']['total']/nearest['pd'] for row in rows},notes='Projection, face mass, clock, movement and nine-variable sampling execute on CUDA; asynchronous Rust point-file output is outside the step stream. The nine actual native sampler bindings include pressure addition, temperature power, geopotential height and staggered U/V/W averaging; inputs come from the real first-step driver capture.')
 output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
