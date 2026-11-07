"""Cold trajectory initialization from real native history and captured projection."""
from pathlib import Path
import sys,json
import numpy as np
import cupy as cp
import netCDF4
from woof.core.kernels import get_kernel
from woof.core.trajectory_contract import parse_points,COORDINATES
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tracer_wrf461_oracle'))
from replay import Stream,compare,digest

def exact_differences(got,ref):
 got=np.ascontiguousarray(got,np.float32);ref=np.ascontiguousarray(ref,np.float32)
 return [dict(index=i.tolist(),native=hex(int(ref.view('u4')[tuple(i)])),cuda=hex(int(got.view('u4')[tuple(i)]))) for i in np.argwhere(got.view('u4')!=ref.view('u4'))]

def run(root,domain):
 path=root/f'trajectory-{domain:02d}.in.bin';stream=Stream(path)
 h=stream.read((21,),'i4');_,_,count,*bounds=map(int,h)
 ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,*_=bounds
 stream.read((3,));stream.read((5,1000))
 shape=(jme-jms+1,kme-kms+1,ime-ims+1)
 for _ in range(3):stream.read(shape)
 for _ in range(6):stream.read((shape[0],shape[2]))
 for _ in range(5):stream.read((shape[1],))
 code=int(stream.read((1,),'i4')[0]);proj=stream.read((14,));stream.finish()
 history=root/f'wrfout_d{domain:02d}_2019-09-04_12:00:00'
 with netCDF4.Dataset(history) as ds:
  def read(name):return np.asarray(ds[name][0],np.float32)
  native=np.stack([read(name.upper()) for name in COORDINATES])
  ph,phb=map(cp.asarray,(read('PH'),read('PHB')))
  pr=cp.zeros(14,cp.float32)
  get_kernel('trajectory','trajectory_projection')((1,),(1,),
   (pr,np.int32(code),np.float32(ds.DX),np.float32(ds.DY),read('XLAT')[0,0],read('XLONG')[0,0],np.float32(ds.TRUELAT1),np.float32(ds.TRUELAT2),np.float32(ds.STAND_LON)))
  live=cp.asnumpy(pr)
  indices=[0,1,2,3,4,5,6,7,8,10,11] if code==1 else list(range(14))
  projection=compare(live[indices],proj[indices])
  projection.update(exact_words=exact_differences(live[indices],proj[indices]),got=live.tolist(),reference=proj.tolist(),compared_indices=indices)
  points=parse_points((root/f'wrfinput_traj_d{domain:02d}').read_text())
  source=cp.asarray(np.array([[p['lon'],p['lat'],p['lev']] for p in points],np.float32))
  agl=cp.asarray(np.array([p['z_is_agl'] for p in points],np.int32))
  coords=cp.full((5,1000),-9999.,cp.float32)
  get_kernel('trajectory','trajectory_initialize')((8,),(128,),
   (coords,np.int32(1000),np.int32(len(points)),source,agl,np.int32(code),pr,ph,phb,np.int32(0),np.int32(ide-ids),np.int32(jde-jds),np.int32(kde-kds)))
  got=cp.asnumpy(coords)
  result=compare(got,native)
  result.update(active_got=got[:,:len(points)].tolist(),active_reference=native[:,:len(points)].tolist())
  return dict(domain=domain,projection=projection,initialization=result,history=dict(name=history.name,sha256=digest(history)),points_sha256=digest(root/f'wrfinput_traj_d{domain:02d}')),coords,native
if __name__=='__main__':
 root=Path(sys.argv[1]);captures=[run(root,d) for d in (1,2)]
 # Native startup feedback copies child k/lon/lat into the parent before
 # these histories are written. Keep both cold kernels and that handoff
 # in the replay, including the native sequential missing-value rule.
 from types import SimpleNamespace
 from woof.core.trajectory import feedback
 feedback(SimpleNamespace(traj_coordinates=captures[0][1]),SimpleNamespace(traj_coordinates=captures[1][1]))
 rows=[]
 for row,coords,reference in captures:
  row['before_feedback_differences']=row['initialization']['differing_words']
  row['initialization']=compare(cp.asnumpy(coords),reference)
  row['initialization']['exact_words']=exact_differences(cp.asnumpy(coords),reference)
  rows.append(row)
 receipt={'rows':rows,'words':sum(x['projection']['words']+x['initialization']['words'] for x in rows),'differing_words':sum(x['projection']['differing_words']+x['initialization']['differing_words'] for x in rows)}
 Path(sys.argv[2]).write_text(json.dumps(receipt,indent=2)+'\n')
 if receipt['differing_words']:raise SystemExit(1)
 print(json.dumps([{'domain':r['domain'],'projection_differences':r['projection']['differing_words'],'initial_differences':r['initialization']['differing_words']} for r in rows]))
