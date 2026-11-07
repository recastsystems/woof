"""The same real native capture, now entering through the shipped Rust files."""
from pathlib import Path
import runpy,contextlib,io,sys,json
from types import SimpleNamespace
import numpy as np
import cupy as cp
from woof.wind_table import read_table
from woof.core.wind_mav import MavFarm
from woof.core.wind_induction import InductionFarm
import argparse
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('mav_capture',type=Path)
ap.add_argument('induction_capture',type=Path)
ap.add_argument('--rotation',type=float,required=True)
args=ap.parse_args();here=Path(__file__).resolve().parent;result={}
for scheme,case,script in [('mav',args.mav_capture,'replay_mav.py'),('fitch',args.induction_capture,'replay_induction.py')]:
 sys.argv=[script,str(case)]
 with contextlib.redirect_stdout(io.StringIO()):x=runpy.run_path(str(here/script))
 path=case/('windturbines-xy.txt' if scheme=='mav' else 'windturbines-ij.txt')
 cfg=SimpleNamespace(wind_farm_scheme=scheme,wind_farm_table=str(path),windfarm_ij=1,
  windfarm_induction=int(scheme=='fitch'),dx=float(x['meta'][0]),dy=float(x['meta'][0]),dt=float(x['meta'][1]),
  windfarm_deg=args.rotation,windfarm_tke_factor=.25,windfarm_wake_model=7,windfarm_overlap_method=4,bl_mynn_tkeadvect=True)
 shape=tuple(x['u'].shape);farm=MavFarm(cfg,shape,prepared=read_table(path,scheme,windfarm_ij=1)) if scheme=='mav' else InductionFarm(cfg,shape)
 gpu=lambda a:cp.asarray(np.ascontiguousarray(a));a={k:gpu(x[k]) for k in ['u','v','dz']};a['z_interface']=gpu(x['zw']);a['rho']=cp.full(shape,1.23,cp.float32)
 du,dv=gpu(x['du']),gpu(x['dv']);q=gpu(x['qke']) if scheme=='fitch' else cp.zeros(shape,cp.float32);fields=dict(qke=q,qke_adv=q.copy(),exch_m=cp.zeros(shape,cp.float32))
 if scheme=='mav':
  farm.apply(a,gpu(x['land']),gpu(x['co']),gpu(x['si']),du,dv,q,fields['qke_adv'],z_mass=gpu(x['zm']))
  actual=dict(du=du,dv=dv,tke=farm.tke_increment,POWER=farm.power)
 else:
  farm.apply_mynn(a,fields,dict(du=du,dv=dv));actual=dict(du=du,dv=dv,qke=q,POWER=farm.power,THRUST=farm.thrust)
 cp.cuda.get_current_stream().synchronize();r={}
 for name,v in actual.items():
  aa=cp.asnumpy(v);ee=x['expected'][name];r[name]=dict(words=aa.size,different=int(np.count_nonzero(aa.view('u4')!=ee.view('u4'))),max_absolute=float(np.max(abs(aa-ee))))
 result[scheme]=dict(table=str(path),table_sha256=farm.table_sha256,native_capture=x['hashlib'].sha256(x['raw']).hexdigest(),outputs=r,numerics='strict' if x['ENABLED'] else 'fast',compiler=x['compile_receipt']())
print(json.dumps(result,indent=2))

assert all(v['different']==0 for row in result.values() for v in row['outputs'].values()), 'native input path differs'
