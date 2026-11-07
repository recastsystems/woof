"""Fixed physical scales for every arm, drawn only through woof.rustwx."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json,sys,time
from datetime import datetime,timezone
import numpy as np
from netCDF4 import Dataset
from score_history import read
from woof import rustwx

def convert(task):
    path,out=map(Path,task);row=read(path)
    with Dataset(path) as f:
        def v(name):return np.asarray(f[name][0],dtype=np.float64)
        p=v('P')+v('PB');hydro=sum(v(n) for n in ('QCLOUD','QRAIN','QICE','QSNOW','QGRAUP'))
        hydro=hydro/(1+v('QVAPOR')+hydro)
        column=np.sum(.5*(hydro[:-1]+hydro[1:])*np.maximum(p[:-1]-p[1:],0),axis=0)/9.80665
    out.mkdir(exist_ok=True,parents=True);nc=out/(path.name+'.nc')
    with Dataset(nc,'w',format='NETCDF3_64BIT_OFFSET') as f:
        ny,nx=row['lat'].shape;f.createDimension('time',1);f.createDimension('y',ny);f.createDimension('x',nx)
        t=f.createVariable('time','f8',('time',));t[:]=[datetime.fromisoformat(row['valid']).timestamp()];t.units='seconds since 1970-01-01 00:00:00';t.standard_name='time'
        for name,key,unit in [('latitude','lat','degrees_north'),('longitude','lon','degrees_east')]:
            a=f.createVariable(name,'f8',('y','x'));a[:]=row[key];a.units=unit;a.standard_name=name
        for name,data in [('wind',row['wind']*1.94384449),('mslp',row['slp']),('hydrometeors',column)]:
            a=f.createVariable(name,'f4',('time','y','x'));a[0]=data;a.units='1';a.coordinates='latitude longitude'
        f.Conventions='CF-1.8'
    return dict(source=str(path),cf=str(nc),valid=row['valid'],max_wind_kt=float(row['wind'].max()*1.94384449),max_hydrometeor_path_kg_m2=float(column.max()))

def main():
    root,out=map(Path,sys.argv[1:3]);label=sys.argv[3] if len(sys.argv)>3 else "WOOF tc/hafs-init; GFS baseline";out.mkdir(exist_ok=True,parents=True);start=time.perf_counter()
    initial=datetime(2024,10,7,tzinfo=timezone.utc)
    files=list(root.glob('wrfout_d01_2024-10-07_00_00_00'))
    for p in sorted(root.glob('wrfout_d02_*')):
        stamp=datetime.strptime(p.name.removeprefix('wrfout_d02_'),'%Y-%m-%d_%H_%M_%S').replace(tzinfo=timezone.utc)
        h=(stamp-initial).total_seconds()/3600
        if 0<=h<=24 and h%3==0:files.append(p)
    with ProcessPoolExecutor(max_workers=6) as pool:rows=list(pool.map(convert,[(str(p),str(out/'planes')) for p in files]))
    table=[]
    for name,title,unit,levels in [('wind','10 m wind speed','kt',list(range(0,201,10))),('mslp','Mean sea level pressure','hPa',list(range(880,1021,5))),('hydrometeors','Resolved pressure-layer condensate','kg m-2',[0,.05,.2,.5,1,2,5,10,20,40])]:
        table.append(dict(names=[name],standard_names=[],product=name,title=title,default_units='1',display_units=unit,unit_factors={'1':1.},levels=levels))
    tablepath=out/'products.json';tablepath.write_text(json.dumps(table))
    theme=out/'theme.json';theme.write_text(json.dumps(dict(extends='woof-light',text=dict(source_label=label,model_label='WOOF'))))
    rings=out/'rings.json';rings.write_text(json.dumps({'rings':[{'lat':22.5,'lon':-93.4,'radii_km':[150,300],'color':'#101418','width':2,'segments':180}]}))
    renderer=rustwx.find_renderer()
    for row in rows:
        for name in ('wind','mslp','hydrometeors'):
            code=rustwx.run_cf_renderer(renderer,Path(row['cf']),variable=name,out_dir=out/'pictures'/Path(row['source']).name,valid=row['valid'],table=tablepath,bounds='-97,-88,19,27',size=(1000,850),overlays=rings if row['valid']=='2024-10-07T00:00:00Z' else None,theme=str(theme.resolve()))
            if code:raise RuntimeError((row,name,code))
    (out/'receipt.json').write_text(json.dumps(dict(frames=rows,seconds=time.perf_counter()-start,pressure_method='Existing core DCOMPUTESEAPRS',condensate_method='Sum five mass mixing ratios, convert to moist mass fraction and trapezoidal dp/g between supplied mass levels'),indent=2)+'\n')
if __name__=='__main__':main()
