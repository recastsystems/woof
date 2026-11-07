"""Resident WRF-v4 groundwater: cold initialization, eight-neighbor flow and WTABLE.

Only option five allocates these fields. Device neighbors use the complete
mass grid, including its physical edges; checkpoint fields own all evolving
state. The absolute model timestep controls native WTDDT cadence.
"""
import math
import numpy as np
from woof.core.noahmp_slab_libm import copy_slab_slots,split_slab_slots
STATIC=('fdepthxy','areaxy','ht','eqzwt','pexpxy','riverbedxy','rivercondxy','rechclim','xland','xice')
STATE=('zwtxy','smcwtdxy','deeprechxy','qslatxy','qrfsxy','qspringsxy','rechxy','qlatxy','qrfxy','qspringxy')
CARRIERS=tuple(dict.fromkeys((*STATE,*STATIC[:8])))
GEOGRAPHY={'fdepthxy':'FDEPTH','eqzwt':'EQZWT','riverbedxy':'RIVERBED','rechclim':'RECHCLIM'}

def shapes(shape):return {**{name:tuple(shape)for name in CARRIERS},'smoiseq':(4,*shape)}
def validate(value):
 if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or value<0:raise ValueError('wtddt must be finite and nonnegative minutes, as in WRF')
 return float(value)
def steps(value,dt):
 ratio=np.float32(np.float32(validate(value))*np.float32(60.)/np.float32(dt))
 if not np.isfinite(ratio)or ratio>2147483646:raise ValueError('wtddt/DT exceeds native INTEGER cadence')
 return max(1,int(math.floor(float(ratio)+.5)))
def module():
 from woof.core.noahmp_irrigation_slab import module
 import cupy as cp
 return module(cp.cuda.runtime.getDevice())
def _pack(f,names):
 import cupy as cp
 n=f['zwtxy'].size;slots=[];width=0
 for name in names:
  a=f[name];w=1 if a.ndim==2 else a.shape[0]
  slots.append((width,a.reshape(n)if w==1 else cp.ascontiguousarray(a.reshape(w,n).T)));width+=w
 out=cp.empty((n,width),cp.float32);copy_slab_slots(out,slots);return out

def _unpack(row,f,names):
 slots={};offset=0
 for name in names:
  w=1 if f[name].ndim==2 else f[name].shape[0];slots[name]=(offset,w);offset+=w
 for name,a in split_slab_slots(row,slots).items():f[name][...]=a.reshape(f[name].shape)if a.ndim==1 else a.T.reshape(f[name].shape)
def _parameters(f,params,*,cold=False):
 import cupy as cp
 from woof.core.noahmp import SOILPARM_SECTION
 key=cp.cuda.runtime.getDevice();cache=params.__dict__.setdefault('_groundwater_tables',{})
 if key not in cache:
  table=params.bundle.soil[SOILPARM_SECTION]
  values=np.array([table.column(k)for k in ('BB','SATDK','SATPSI','MAXSMC','WLTSMC','SATDW')],np.float32).T
  values[:,2]*=np.float32(-1.);cache[key]=cp.asarray(values)
 p=cp.ascontiguousarray(cache[key][f['isltyp'].reshape(-1).astype(cp.int32)-1])
 v=f['ivgtyp'].reshape(-1);u=v==params.land_use.isurban
 if cold:
  for lc in params.land_use.lcz:u|=v==lc
 p[:,3]=cp.where(u,np.float32(.45),p[:,3]);p[:,4]=cp.where(u,np.float32(.4),p[:,4]);return p

def ingest(f,values,*,dx,dy,mapfac_x,mapfac_y):
 import cupy as cp
 if values is None or any(w not in values for w in GEOGRAPHY.values()):raise ValueError('Noah-MP opt_run=5 requires FDEPTH, EQZWT, RIVERBED and RECHCLIM geography')
 for name,wrf in GEOGRAPHY.items():
  a=np.asarray(values[wrf],np.float32)
  if a.shape!=f[name].shape or not np.isfinite(a).all():raise ValueError(f'Noah-MP {wrf} geography has invalid shape or nonfinite values')
  f[name][...]=cp.asarray(a)
 f['areaxy'][...]=(np.float32(dx)*np.float32(dy))/(cp.asarray(mapfac_x,dtype=cp.float32)*cp.asarray(mapfac_y,dtype=cp.float32))
 f['ht'][...]=cp.asarray(values['HGT'],dtype=cp.float32)

def evaluate_rows(p,s,y,z,soil,land,dzs,shape,dt,*,cold=False,flow_land=None):
 """Shared native-replay and resident entry point, with no host field reads."""
 import cupy as cp
 nx=shape[1];ny=shape[0];n=nx*ny;blocks=((n+127)//128,);threads=(128,);m=module();dz=cp.asarray(dzs,cp.float32);kh=cp.empty((n,2),cp.float32)
 def lateral(interval):
  m.get_function('k_groundwater_heads')(blocks,threads,(p,s,y,soil,kh,np.int32(n)))
  m.get_function('k_groundwater_lateral')(blocks,threads,(s,kh,land if flow_land is None else flow_land,y,np.int32(nx),np.int32(ny),np.float32(interval)))
 if cold:
  for _ in range(500):
   lateral(np.float32(31536000.))
   m.get_function('k_groundwater_cold_adjust')(blocks,threads,(y,land,np.int32(n)))
  m.get_function('k_groundwater_cold_river')(blocks,threads,(s,y,np.int32(n)))
  lateral(dt)
  m.get_function('k_groundwater_cold_soil')(blocks,threads,(p,s,dz,land,y,z,np.int32(n),np.float32(dt)))
 else:
  lateral(dt)
  m.get_function('k_groundwater_update')(blocks,threads,(p,s,dz,land,y,z,np.int32(n),np.float32(dt)))
 return s,y,z

def initialize(f,params,dzs,wtddt):
 import cupy as cp
 f['pexpxy'].fill(1);f['zwtxy'][...]=f['eqzwt'];f['waxy'].fill(0)
 p=_parameters(f,params,cold=True);s=_pack(f,STATIC);y=_pack(f,STATE);z=_pack(f,('smois','sh2o','smoiseq'))
 v=f['ivgtyp'];land=cp.ascontiguousarray(((v!=params.land_use.iswater)&(v!=params.land_use.isice)).reshape(-1).astype(cp.int32));soil=cp.ascontiguousarray(f['isltyp'].reshape(-1).astype(cp.int32))
 evaluate_rows(p,s,y,z,soil,land,dzs,v.shape,np.float32(wtddt)*np.float32(60),cold=True)
 _unpack(s,f,STATIC[:8]+('xland','xice'));_unpack(y,f,STATE);_unpack(z,f,('smois','sh2o','smoiseq'))

def advance(f,params,dzs,wtddt,dt,itimestep):
 if int(itimestep)%steps(wtddt,dt):return
 import cupy as cp
 v=f['ivgtyp'];land=cp.ascontiguousarray(((f['xland']<np.float32(1.5))&(f['xice']<np.float32(params.xice_threshold))&(v!=params.land_use.isice)).reshape(-1).astype(cp.int32));soil=cp.ascontiguousarray(f['isltyp'].reshape(-1).astype(cp.int32))
 p=_parameters(f,params);s=_pack(f,STATIC);y=_pack(f,STATE);z=_pack(f,('smois','sh2o','smoiseq'))
 evaluate_rows(p,s,y,z,soil,land,dzs,v.shape,np.float32(wtddt)*np.float32(60))
 _unpack(y,f,STATE);_unpack(z,f,('smois','sh2o','smoiseq'))
