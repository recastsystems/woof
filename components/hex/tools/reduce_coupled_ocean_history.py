"""Reduce every forecast frame to evaluation fields; hash all arrays for off proof."""
from pathlib import Path
import argparse, hashlib,json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from netCDF4 import Dataset
FIELDS=('latCell','lonCell','tsk','sst','u10','v10','surface_pressure','t2','ocean_depth','ocean_residual','ocean_pwp_mixing_failures','ocean_initial_sst','ocean_net_flux','ter','ocean_pwp_active','ocean_temperature','ocean_salinity','ocean_layer_bottom','landmask','ocean_ohc26','ocean_d26','ocean_d26_censored','ocean_ild','ocean_mld','ocean_barrier_layer')
def reduce(task):
 p,out=map(Path,task[:2]); bounds=task[2] if len(task)>2 else None; hashes={}; data={}; columns={}
 with Dataset(p) as d:
  for name,v in d.variables.items():
   a=np.asarray(v[:]); hashes[name]=dict(shape=a.shape,dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
   if name in FIELDS:
    data[name]=a.squeeze()
    if 'nOceanLevels' in v.dimensions:
     dims=[k for k in v.dimensions if k!='Time'];data[name]=np.moveaxis(data[name],dims.index('nOceanLevels'),0)
   if name in ('pressure','theta','u_zonal','v_meridional'):columns[name]=a.squeeze()
  if 'xtime' in d.variables:data['valid']=np.asarray(d.variables['xtime'][:]).tobytes().decode().strip()
 from woof.hex.storm_tracker import sea_level_pressure,interpolate_to_pressure
 if all(k in columns for k in ('pressure','theta')):
  pressure=columns['pressure'];temperature=columns['theta'][:,0]*(pressure[:,0]/100000.)**(287./1004.)
  data['mslp_pa']=sea_level_pressure(data['surface_pressure'],data['ter'],temperature)
  for level in (85000,70000,50000,30000,20000):
   for key,var in [('u','u_zonal'),('v','v_meridional')]:
    if var in columns:data[f'{key}_{level}']=interpolate_to_pressure(pressure,columns[var],level)
 if bounds is not None:
  west,east,south,north=map(float,bounds)
  lat=np.rad2deg(data['latCell']);lon=(np.rad2deg(data['lonCell'])+180)%360-180
  keep=(lat>=south)&(lat<=north)&(lon>=west)&(lon<=east);n=lat.size
  for key,value in list(data.items()):
   if isinstance(value,np.ndarray) and value.ndim and value.shape[-1]==n:data[key]=value[...,keep]
  data['retained_bounds_deg']=np.asarray(bounds);data['source_cell_indices']=np.flatnonzero(keep)
 data['coordinate_units']='radians'
 np.savez_compressed(out/(p.stem+'.npz'),**data)
 return dict(file=p.name,arrays=hashes)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('input',type=Path);a.add_argument('output',type=Path);args=a.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 files=sorted(args.input.glob('cuda-history.*.nc'))
 with ProcessPoolExecutor(max_workers=6) as pool: results=list(pool.map(reduce,[(str(p),str(args.output)) for p in files]))
 (args.output/'array-hashes.json').write_text(json.dumps(results,indent=2)+'\n');print(f'Reduced {len(files)} frames')
