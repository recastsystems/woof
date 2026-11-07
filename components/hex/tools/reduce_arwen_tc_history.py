"""Reduce WOOF history for the same observation scoreboard used by hex."""
import argparse,hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from netCDF4 import Dataset
from woof.hex.storm_tracker import sea_level_pressure,interpolate_to_pressure
from woof.da.obsop import earth_relative_winds

def reduce(task):
    p,out=map(Path,task);hashes={};data={}
    with Dataset(p) as d:
        for name,v in d.variables.items():
            a=np.asarray(v[:]);hashes[name]=dict(shape=a.shape,dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
        def a(name):return np.asarray(d[name][0],dtype=float)
        data['valid']=np.asarray(d['Times'][0]).tobytes().decode().strip()
        data['coordinate_units']='degrees'
        for dest,src in [('latCell','XLAT'),('lonCell','XLONG'),('native_sst','SST'),('tsk','TSK'),('t2','T2'),('surface_pressure','PSFC'),('ter','HGT'),('landmask','LANDMASK'),('ocean_pwp_active','OM_ACTIVE')]:
            if src in d.variables:data[dest]=a(src).ravel()
        # The atmosphere receives the live water temperature in TSK. SST is
        # the prescribed input carrier and remains unchanged during PWP.
        # Use the same carrier for ON and OFF; verification masks land/lakes.
        data['sst']=data['tsk'].copy()
        data['sst_source']='TSK, live atmosphere-facing water temperature'
        sina,cosa=a('SINALPHA'),a('COSALPHA');u10,v10=earth_relative_winds(a('U10'),a('V10'),sina,cosa);data['u10']=u10.ravel();data['v10']=v10.ravel()
        pressure=a('P')+a('PB');temperature=(a('T')+300)*(pressure/100000.)**(287./1004.)
        data['mslp_pa']=sea_level_pressure(a('PSFC'),a('HGT'),temperature[0]).ravel()
        u,v=a('U'),a('V');u=(u[...,:-1]+u[...,1:])/2;v=(v[:,:-1,:]+v[:,1:,:])/2
        u,v=earth_relative_winds(u,v,sina,cosa);columns=lambda x:np.moveaxis(x,0,-1).reshape(-1,x.shape[0])
        for lev in (85000,70000,50000,30000,20000):
            data[f'u_{lev}']=interpolate_to_pressure(columns(pressure),columns(u),lev);data[f'v_{lev}']=interpolate_to_pressure(columns(pressure),columns(v),lev)
        for dest,src in [('ocean_temperature','OM_TMP'),('ocean_salinity','OM_S'),('ocean_layer_bottom','OM_DEPTH')]:
            if src in d.variables:
                arr=a(src);data[dest]=arr.reshape(arr.shape[0],-1)
        for name in ('OCEAN_SST','OCEAN_SST_CHANGE','OCEAN_OHC26','OCEAN_D26',
                     'OCEAN_D26_CENSORED','OCEAN_ILD','OCEAN_MLD','OCEAN_BARRIER_LAYER',
                     'RAINNC','RAINC','LAKEMASK'):
            if name in d.variables:data[name.lower()]=a(name).ravel()
    target=out/p.name.split('_')[1];target.mkdir(exist_ok=True)
    np.savez_compressed(target/(p.name.replace(':','.')+'.npz'),**data)
    return dict(file=p.name,arrays=hashes)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    files=sorted(x for x in a.input.rglob('wrfout_d0*') if x.is_file() and not x.name.endswith('.json'))
    with ProcessPoolExecutor(6) as pool:rows=list(pool.map(reduce,[(str(x),str(a.output)) for x in files]))
    (a.output/'array-hashes.json').write_text(json.dumps(rows,indent=2)+'\n');print(f'Reduced {len(files)} frames')
