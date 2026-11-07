"""Bounded CUDA-event stage timing on tiled real native input columns."""
from pathlib import Path
import argparse, json
import cupy as cp
import numpy as np
from tools.keps_wrf471_oracle.replay import read_capture
from woof.core.keps import keps_step,load_keps_module
from woof.core.ysu import launch_ysu
from woof.core.myjpbl import myj_pbl_step
from woof.core.kernel_local_size import kernel_local_size_bytes
from woof.wrf_exact import ENABLED
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[2]/'woof/data/keps/wrf471/original/keps-000180.bin.xz')
parser.add_argument('--receipt',type=Path,required=True)
parser.add_argument('--nx',type=int,default=250)
parser.add_argument('--ny',type=int,default=200)
args=parser.parse_args()
if args.nx<1 or args.ny<1:parser.error('nx and ny must be positive')
i,s,w,meta=read_capture(args.input)
ny,nx=args.ny,args.nx
input_ny,input_nx=i['u'].shape[-2:]

def tile(a):
 if a.ndim==3:return cp.asarray(np.ascontiguousarray(np.tile(a,(1,(ny+input_ny-1)//input_ny,(nx+input_nx-1)//input_nx))[:,:ny,:nx]))
 return cp.asarray(np.ascontiguousarray(np.tile(a,((ny+input_ny-1)//input_ny,(nx+input_nx-1)//input_nx))[:ny,:nx]))
i={k:tile(v) for k,v in i.items()};s={k:tile(v) for k,v in s.items()}
zero=cp.zeros_like(i['u']);two=cp.ones((ny,nx),cp.float32)
ysu_args=[i['u'],i['v'],i['th'],i['qv'],i['qc'],zero,
          cp.ascontiguousarray((i['p8w'][:-1]+i['p8w'][1:])*cp.float32(.5)),
          i['p8w'],i['pi'],i['dz8w'],i['rthraten']]
ysu_kw=dict(psfc=cp.ascontiguousarray(i['p8w'][0]),znt=i['znt'],ust=i['ust'],hfx=i['hfx'],qfx=i['qfx'],
            wspd=cp.sqrt(i['u'][0]**2+i['v'][0]**2),br=i['br'],psim=i['psim'],psih=i['psih'],
            xland=two,u10=cp.ascontiguousarray(i['u'][0]),v10=cp.ascontiguousarray(i['v'][0]),dt=meta['dt'])

def measure(fn):
 for _ in range(3):fn()
 cp.cuda.get_current_stream().synchronize()
 # Five calls are enough to average launch noise without a proof campaign.
 samples=[]
 for _ in range(5):
  begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record();fn();end.record();end.synchronize()
  samples.append(float(cp.cuda.get_elapsed_time(begin,end)))
 return dict(ms=samples,median_ms=float(np.median(samples)))
a=measure(lambda:keps_step(i,s,dt=meta['dt'],itimestep=max(3,meta['itimestep']),cp_air=meta['cp_air'],g=meta['g']))
b=measure(lambda:launch_ysu(*ysu_args,**ysu_kw))
columns=dict(dz=i['dz8w'],u=i['u'],v=i['v'],t=i['th']*i['pi'],
             th=i['th'],exner=i['pi'],qv=i['qv'],qc=i['qc'],qi=zero,
             p=ysu_args[6])
def plane(a): return cp.ascontiguousarray(a[0])
surface=dict(psfc=ysu_kw['psfc'],ust=i['ust'],tsk=plane(columns['t'])+cp.float32(2),
             chklowq=two,xland=two,sice=two*0,snow=two*0,
             akhs=two*cp.float32(.05),akms=two*cp.float32(.1),
             elflx=i['qfx']*cp.float32(2.5e6),uz0=plane(i['u']),vz0=plane(i['v']),ht=two*0)
myj_state=dict(thz0=plane(i['th']).copy(),qz0=plane(i['qv']).copy(),
               qsfc=plane(i['qv']).copy(),ct=two*cp.float32(.1))
tke=cp.full_like(i['u'],cp.float32(.1))
c=measure(lambda:myj_pbl_step(columns,surface,myj_state,tke,dtturbl=meta['dt'],flqi=True))
f=load_keps_module().get_function('keps_columns')
r=dict(schema='keps-stage-events-v1',strict=ENABLED,input=meta,shape=[i['u'].shape[0],ny,nx],
       keps=a,ysu=b,myj=c,ratio=a['median_ms']/b['median_ms'],
       ratio_myj=a['median_ms']/c['median_ms'],
       myj_surface='Eta transfer coefficients declared: AKHS=.05 AKMS=.1 m/s; THZ0/QZ0 from native lowest level, TSK=Tlow+2K',
       kernel_attributes=f.attributes,device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
args.receipt.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
