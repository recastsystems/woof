"""GPU regression: existing aerosol column versus bounded storage, all output words."""
import os,json
from pathlib import Path
import numpy as np
import cupy as cp
from woof.hex.mpasphys import thompson_aero_v842 as a, thompson_v842 as c
from woof.hex.mpasphys.resources_build import load_packet,resolve_resource
r=a.load_resources(load_packet(resolve_resource('physics-resources'))['mp_thompson_aerosols'],cp)
nz,nc=55,257
z=np.linspace(100,16000,nz,dtype=np.float32)[:,None]
x=np.arange(nc,dtype=np.float32)[None,:]
p=np.broadcast_to(100000*np.exp(-z/8000),(nz,nc)).copy();pi=(p/100000)**np.float32(287.04/1004.5)
t=np.broadcast_to(np.maximum(210,294-.006*z),(nz,nc)).copy()
fields={n:cp.zeros((nz,nc),dtype=cp.float32) for n in (*c.FIELD_ARGUMENTS,*a.AEROSOL_FIELDS)}
vals=dict(qv=.012*np.exp(-z/2200)*np.ones_like(x),qc=np.where((z<5000)&(x%4==0),.0004,0),qi=np.where((z>6000)&(z<9500)&(x%4==1),.0002,0),qr=np.where((z<4000)&(x%4==2),.0002,0),qs=np.where((z>4000)&(z<9000)&(x%4==2),.0003,0),qg=np.where((z>3000)&(z<8000)&(x%4==3),.0001,0),ni=10000,nr=100000,nc=1e8,nifa=1e5,nwfa=1e8,th=t/pi,pii=pi,p=p,w=np.broadcast_to((x%5)-2,(nz,nc)),dz=300)
for k,v in vals.items():fields[k][...]=cp.asarray(v,dtype=cp.float32)
surf={n:cp.zeros(nc,dtype=cp.float32) for n in (*c.SURFACE_ARGUMENTS,'nifa2d','nwfa2d')}
for k,v in dict(nt_c=1e8,mu_c=0,nifa2d=5,nwfa2d=500).items():
 if k in surf:surf[k].fill(v)
base_f={k:v.copy() for k,v in fields.items()};base_s={k:v.copy() for k,v in surf.items()}
a.launch_driver(fields,surf,**r,dt=18,workspace=False)
old={**{k:cp.asnumpy(v) for k,v in fields.items()},**{k:cp.asnumpy(v) for k,v in surf.items()}}
rows=[]
for tile in (32,128,256):
 os.environ['WOOF_HEX_THOMPSON_TILE_COLUMNS']=str(tile)
 f={k:v.copy() for k,v in base_f.items()};s={k:v.copy() for k,v in base_s.items()}
 a.launch_driver(f,s,**r,dt=18,workspace=True)
 cp.cuda.Stream.null.synchronize()
 diffs={k:int(np.count_nonzero(cp.asnumpy(v).view(np.uint32)!=old[k].view(np.uint32))) for k,v in {**f,**s}.items()}
 assert not any(diffs.values()),diffs
 assert all(np.isfinite(v).all() for v in old.values())
 rows.append(dict(tile=tile,differing_words=diffs))
k0=a._aero_module(0,r['device_math_source'],nz,False).get_function('mpas_thompson_step')
k1=a._aero_module(0,r['device_math_source'],nz,True).get_function('mpas_thompson_step')
print(json.dumps(dict(shape=[nz,nc],legacy_local_bytes=cp.cuda.driver.funcGetAttribute(3,k0.ptr),workspace_local_bytes=cp.cuda.driver.funcGetAttribute(3,k1.ptr),rows=rows),indent=2))
