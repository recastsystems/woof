"""Hourly pressure tracking, using no future observation positions."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json,sys,time
import numpy as np
from netCDF4 import Dataset,chartostring
from woof.core.mslp import _dcomputeseaprs,_nine_point_smooth

def distance(lat,lon,center):
    a,b=np.radians(center);x,y=np.radians(lat),np.radians(lon)
    return 12742*np.arcsin(np.minimum(1,np.sqrt(np.sin((x-a)/2)**2+np.cos(a)*np.cos(x)*np.sin((y-b)/2)**2)))

def read(path):
    with Dataset(path) as f:
        def v(name):return np.asarray(f[name][0],dtype=np.float64)
        lat,lon=v('XLAT'),v('XLONG');p=v('P')+v('PB');theta=v('T')+300
        phi=v('PH')+v('PHB');z=.5*(phi[1:]+phi[:-1])/9.80665
        slp=_dcomputeseaprs(p,theta*(p/100000)**(287/1004.5),v('QVAPOR'),z)
        smoothed=_nine_point_smooth(slp,3,4)
        wind=np.hypot(v('U10'),v('V10'));ps=v('PSFC')/100;hgt=v('HGT')
        stamp=str(chartostring(f['Times'][:])[0]).replace('_','T')+'Z'
        return dict(path=str(path),valid=stamp,lat=lat,lon=lon,slp=slp,smooth=smoothed,wind=wind,ps=ps,hgt=hgt)

def main():
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.parent.mkdir(parents=True,exist_ok=True)
    paths=sorted(root.glob('wrfout_d0[12]_*'));start=time.perf_counter()
    center=(22.5,-93.4);fixes=[];bytime={}
    # A bounded batch keeps array memory below 1 GB and uses processes because
    # independent netCDF/HDF5 handles are not thread safe.
    with ProcessPoolExecutor(max_workers=6) as pool:
        for begin in range(0,len(paths),6):
            for row in pool.map(read,paths[begin:begin+6]):bytime.setdefault(row['valid'],[]).append(row)
            # The whole per-hour set is small for this 218x180 / 160x160 case.
    for valid,rows in sorted(bytime.items()):
        candidates=[]
        for row in rows:
            near=distance(row['lat'],row['lon'],center)<=400
            if not near.any():continue
            j,i=np.unravel_index(np.argmin(np.where(near,row['smooth'],np.inf)),near.shape)
            candidates.append((row['smooth'][j,i],row,j,i))
        if not candidates:raise RuntimeError(f'{valid}: no grid within continuity search')
        _,row,j,i=min(candidates,key=lambda x:x[0]);center=float(row['lat'][j,i]),float(row['lon'][j,i])
        wind=[];pressure=[];ring=[]
        for r in rows:
            near=distance(r['lat'],r['lon'],center)<=300
            wind.append(float(r['wind'][near].max()));pressure.append(float(r['slp'][near].min()))
            if 'wrfout_d01_' in r['path']:
                radius=distance(r['lat'],r['lon'],(22.5,-93.4));sel=(radius>=150)&(radius<=300)&(r['hgt']<5)
                ring=r['ps'][sel].tolist()
        fixes.append(dict(valid=valid,lat=center[0],lon=center[1],vmax_kt=max(wind)*1.94384449,mslp_hpa=min(pressure),center_grid=row['path'],center_edge_cells=int(min(j,i,row['lat'].shape[0]-1-j,row['lat'].shape[1]-1-i)),initial_ocean_ring_psfc_mean_hpa=float(np.mean(ring)),initial_ocean_ring_psfc_min_hpa=float(np.min(ring)),initial_ocean_ring_psfc_max_hpa=float(np.max(ring))))
    out.write_text(json.dumps(dict(schema='tc-hafs-init.history-diagnostics/v1',method='Minimum three-pass smoothed MSLP within 400 km of previous center, initialized only from cycle vitals; unsmoothed intensity extrema within 300 km over both grids; hourly samples.',fixes=fixes,seconds=time.perf_counter()-start),indent=2)+'\n')
if __name__=='__main__':main()
