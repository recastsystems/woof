"""Resident WRF-v4 irrigation prefix, using the complete native procedures."""
from functools import lru_cache
import numpy as np
from woof.core.noahmp_slab_libm import copy_slab_slots,split_slab_slots
PARAMETERS=('irr_frac','irr_har','irr_lai','irr_mad','filoss','sprir_rate','micir_rate','firtfac','ir_rain')
LAYERS=('smcmax','smcwlt','smcref','bexp','dksat','dwsat')
PARAMETER_LAYOUT={**{name:(4*k,4) for k,name in enumerate(LAYERS)},**{name:(24+k,1) for k,name in enumerate((*PARAMETERS,'pltday','hsday'))}}
INPUT_LAYOUT={'zsoil':(0,4),'sh2o':(4,4),'smc':(8,4),**{name:(12+k,1) for k,name in enumerate(('dt','sfctmp','uu','vv','eair','rain','fveg','lai','julian','irrfra','sifra','mifra','fifra','iramtsi','iramtmi','iramtfi'))}}
INTS=('croplu','nroot','ircntsi','ircntmi','ircntfi')
OUTPUTS=('rain','firr','eirr','irsirate','irevplos','iramtsi','iramtmi','iramtfi','sifac','mifac','fifac')
@lru_cache(maxsize=None)
def module(card):
 from woof.core.noahmp_kernel_sources import compile_runtime_unit
 return compile_runtime_unit('noahmp_water',selected_options=True,module_key='noahmp_water_selected')
def _take(cp,fields,name,n,width=1,dtype=None):
 a=cp.asarray(fields[name],dtype=cp.float32 if dtype is None else dtype)
 if a.shape!=((n,) if width==1 else (n,width)):raise ValueError(f'irrigation {name} has shape {a.shape}, expected {n} columns, width {width}')
 return cp.ascontiguousarray(a)
def parameter_slab(fields,n):
 import cupy as cp
 a=cp.empty((n,35),dtype=cp.float32)
 copy_slab_slots(a,[(offset,_take(cp,fields,name,n,width))for name,(offset,width)in PARAMETER_LAYOUT.items()]);return a
def evaluate_prefix(fields,n,*,opt_irr,opt_irrm,calculate_soil):
 import cupy as cp
 if type(opt_irr)is not int or opt_irr not in (1,2,3) or type(opt_irrm)is not int or opt_irrm not in (0,1,2,3):raise ValueError('native irrigation requires opt_irr=1/2/3 and opt_irrm=0/1/2/3')
 p=parameter_slab(fields,n);x=cp.empty((n,28),dtype=cp.float32);i=cp.empty((n,5),dtype=cp.int32)
 copy_slab_slots(x,[(offset,_take(cp,fields,name,n,width))for name,(offset,width)in INPUT_LAYOUT.items()])
 copy_slab_slots(i,[(k,_take(cp,fields,name,n,dtype=cp.int32))for k,name in enumerate(INTS)])
 y=cp.empty((n,len(OUTPUTS)),dtype=cp.float32);iy=cp.empty((n,3),dtype=cp.int32)
 if n:module(cp.cuda.runtime.getDevice()).get_function('k_irrigation_prefix')(((n+127)//128,),(128,),(p,x,i,y,iy,np.int32(n),np.int32(opt_irr),np.int32(opt_irrm),np.int32(calculate_soil)))
 out=split_slab_slots(y,{name:(k,1)for k,name in enumerate(OUTPUTS)})
 out.update(split_slab_slots(iy,{name:(k,1)for k,name in enumerate(INTS[2:])}));return out

# Pinned Registry/registry.noahmp:234-248. IRWAT amounts are restart-only
# and deliberately absent from the native nest feedback inventory.
MAPS=('irfract','sifract','mifract','fifract')
COUNTERS=('irnumsi','irnummi','irnumfi')
AMOUNTS=('irwatsi','irwatmi','irwatfi')
ACCUMULATED=('irsivol','irmivol','irfivol','ireloss','irrsplh')
STATIC=(*MAPS,'planting','harvest')
STATE=(*AMOUNTS,*ACCUMULATED)
GATHER={'irrfra':'irfract','sifra':'sifract','mifra':'mifract','fifra':'fifract',
 'ircntsi':'irnumsi','ircntmi':'irnummi','ircntfi':'irnumfi',
 'iramtsi':'irwatsi','iramtmi':'irwatmi','iramtfi':'irwatfi'}
def cropland(vegtyp,dataset):
 if dataset=='USGS':return (vegtyp>=3)&(vegtyp<=6)
 if dataset=='MODIFIED_IGBP_MODIS_NOAH':return (vegtyp==12)|(vegtyp==14)
 return vegtyp!=vegtyp
