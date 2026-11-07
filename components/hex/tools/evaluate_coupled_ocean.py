"""Observation evaluation for reduced native-mesh histories; all maps use Rust."""
import argparse,csv,gzip,json,os
from pathlib import Path
from datetime import datetime,timedelta
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from netCDF4 import Dataset,num2date

def read_oi(task):
 path,bounds=task;west,east,south,north=bounds
 with Dataset(path) as d:
  lat=np.asarray(d['lat'][:]); lon=(np.asarray(d['lon'][:])+180)%360-180
  iy=np.flatnonzero((lat>=south)&(lat<=north));ix=np.flatnonzero((lon>=west)&(lon<=east))
  v=np.ma.asarray(d['sst'][0,0,iy,ix]).filled(np.nan)
  valid=num2date(d['time'][0],d['time'].units).isoformat()
 return dict(valid=valid,lat=lat[iy],lon=lon[ix],sst=v)

def xyz(lat,lon):
 a,b=np.radians(lat),np.radians(lon);return np.column_stack([np.cos(a)*np.cos(b),np.cos(a)*np.sin(b),np.sin(a)])

def main():
 # Evidence rendering is CPU work and must not acquire an unleased GPU.
 os.environ['RUSTWX_RASTER_DEVICE']='cpu'
 os.environ['CUDA_VISIBLE_DEVICES']='-1'
 a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True);args=a.parse_args();cfg=json.loads(args.config.read_text())
 out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True);work=Path(cfg['work']);work.mkdir(parents=True,exist_ok=True)
 west,east,south,north=cfg['bounds']
 frames={mode:[dict(np.load(p)) for p in sorted(Path(root).glob('*.npz'))] for mode,root in cfg['arms'].items()}
 with Dataset(cfg['static']) as d:mask=np.asarray(d['landmask'][:]).ravel()==0
 for mode,fs in frames.items():
  for f in fs:
   if 'sst' not in f:f['sst']=f['tsk'] # prescribed open-water skin is the fixed SST; omit frozen oceans from regional scores
 sample=frames['pwp'][0];lat=np.degrees(sample['latCell']);lon=(np.degrees(sample['lonCell'])+180)%360-180
 # History coordinates are in radians, exactly as in the native grid.
 region=mask&(lat>=south)&(lat<=north)&(lon>=west)&(lon<=east)
 tree=cKDTree(xyz(lat[mask],lon[mask]));ocean_idx=np.flatnonzero(mask)
 with ProcessPoolExecutor(max_workers=6) as pool:oi=list(pool.map(read_oi,[(p,cfg['bounds']) for p in cfg['oisst']]))
 yl,xl=np.meshgrid(oi[0]['lat'],oi[0]['lon'],indexing='ij');_,idx=tree.query(xyz(yl.ravel(),xl.ravel()),workers=6);idx=ocean_idx[idx].reshape(yl.shape)
 start=datetime.fromisoformat(cfg['start']);clock=lambda f:datetime.strptime(str(f['valid']),'%Y-%m-%d_%H:%M:%S')
 records=[];maps=[]
 def metrics(x,y):
  ok=np.isfinite(x)&np.isfinite(y);e=x[ok]-y[ok];w=np.cos(np.radians(yl[ok]));result=dict(count=int(ok.sum()),bias_K=float(np.average(e,weights=w)),rmse_K=float(np.sqrt(np.average(e*e,weights=w))))
  west,east,south,north=cfg['wake_bounds'];wake=ok&(xl>=west)&(xl<=east)&(yl>=south)&(yl<=north)
  e=x[wake]-y[wake];w=np.cos(np.radians(yl[wake]));result.update(wake_count=int(wake.sum()),wake_bias_K=float(np.average(e,weights=w)),wake_rmse_K=float(np.sqrt(np.average(e*e,weights=w))),wake_model_min_K=float(x[wake].min()),wake_observed_min_K=float(y[wake].min()))
  return result
 obs_by_time={datetime.fromisoformat(x['valid']):x for x in oi}
 for mode,fs in frames.items():
  for f in fs:
   t=clock(f);sst=f['sst'][idx]-273.15
   row=dict(mode=mode,valid=t.isoformat(),hours=(t-start).total_seconds()/3600,regional_sst_min_K=float(np.min(f['sst'][region])),regional_sst_max_K=float(np.max(f['sst'][region])))
   if 'ocean_pwp_mixing_failures' in f:row['max_mixing_failures']=float(f['ocean_pwp_mixing_failures'].max());row['max_abs_residual_J_m2']=float(np.abs(f['ocean_residual'][mask]).max())
   if t in obs_by_time:row.update(metrics(sst,obs_by_time[t]['sst']))
   records.append(row)
 # Daily matched changes, starting with the first daily-mean OISST nominal time.
 from woof.hex.ocean_products import physical_renderer,write_plane
 from woof.rustwx import run_cf_renderer
 from types import SimpleNamespace
 renderer=physical_renderer();fakecp=SimpleNamespace(asnumpy=np.asarray,asarray=np.asarray)
 levels=[-5,-3,-2,-1,-.5,-.1,.1,.5,1,2,3,5];colors=[[5,48,97],[33,102,172],[67,147,195],[146,197,222],[209,229,240],[247,247,247],[253,219,199],[244,165,130],[214,96,77],[178,24,43],[103,0,31]]
 table=work/'cf-table.json';table.write_text(json.dumps([dict(names=['sst_change'],standard_names=[],product='sst_change',title='SST change since first matched daily analysis',default_units='K',display_units='K',unit_factors={'K':1},scale=dict(levels=levels,colors=colors))]))
 matched=[f for f in frames['pwp'] if clock(f) in obs_by_time]
 if matched:
  first=matched[0];firstobs=obs_by_time[clock(first)]['sst']
  for f in matched[1:]:
   t=clock(f);stamp=t.strftime('%Y%m%d%H');mod=(f['sst'][idx]-first['sst'][idx]);obs=obs_by_time[t]['sst']-firstobs
   mod=np.where(np.isfinite(obs),mod,np.nan)
   for label,plane in [('pwp',mod),('oisst',obs)]:
    nc=work/f'{label}-{stamp}.nc';write_plane(nc,fakecp,yl,xl,t,{'sst_change':('K',plane)},source='NOAA OISST v2.1 daily analysis' if label=='oisst' else 'PWP coupled forecast')
    target=work/f'render-{label}-{stamp}';receipt=out/f'{label}-{stamp}.render.json'
    rc=run_cf_renderer(renderer,nc,variable='sst_change',out_dir=target,valid=t.isoformat()+'Z',bounds=','.join(map(str,cfg['bounds'])),table=table,size=(1000,740),receipt=receipt)
    if rc:raise RuntimeError(f'render failed {rc}')
    import shutil
    pics=list(target.rglob('*.png'))
    if len(pics)!=1:raise RuntimeError(f'unexpected pictures {pics}')
    dest=out/f'{label}-{stamp}.png';shutil.copyfile(pics[0],dest);maps.append(dict(file=dest.name,valid=t.isoformat(),field=label))
    if f is matched[-1]:
     # An additional, explicitly clipped scale exposes weak modeled changes;
     # the full-range pair above remains the magnitude comparison of record.
     fine=work/'cf-fine-table.json'
     fine.write_text(json.dumps([dict(names=['sst_change'],standard_names=[],product='sst_change',title='SST change; detail scale clipped at +/-1.5 K',default_units='K',display_units='K',unit_factors={'K':1},scale=dict(levels=[-1.5,-1,-.75,-.5,-.25,-.05,.05,.25,.5,.75,1,1.5],colors=colors))]))
     target=work/f'detail-{label}-{stamp}'
     rc=run_cf_renderer(renderer,nc,variable='sst_change',out_dir=target,valid=t.isoformat()+'Z',bounds=','.join(map(str,cfg['bounds'])),table=fine,size=(1000,740),receipt=out/f'{label}-{stamp}-detail.render.json')
     if rc:raise RuntimeError(f'detail render failed {rc}')
     pics=list(target.rglob('*.png'))
     if len(pics)!=1:raise RuntimeError(f'unexpected detail pictures {pics}')
     dest=out/f'{label}-{stamp}-detail.png';shutil.copyfile(pics[0],dest);maps.append(dict(file=dest.name,valid=t.isoformat(),field=label,scale_clipped_K=1.5))
   records.append(dict(mode='pwp-change',valid=t.isoformat(),hours=(t-start).total_seconds()/3600,**metrics(mod,obs)))
   records.append(dict(mode='fixed-change',valid=t.isoformat(),hours=(t-start).total_seconds()/3600,**metrics(np.zeros_like(obs),obs)))
 # Time series are analysis charts, not weather fields.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained');buoyrows=[]
 for ax,b in zip(axs,cfg['buoys']):
  _,ii=tree.query(xyz([b['lat']],[b['lon']]));cell=int(ocean_idx[ii[0]])
  obs=[]
  with gzip.open(b['path'],'rt') as stream:
   for line in stream:
    if line.startswith('#'):continue
    v=line.split();t=datetime(*map(int,v[:5]));temp=float(v[14])
    if start<=t<=max(clock(f) for f in frames['pwp']) and -2<temp<40:obs.append((t,temp))
  ax.plot([x[0] for x in obs],[x[1] for x in obs],color='black',label='Buoy')
  for mode,fs in frames.items():
   ts=[clock(f) for f in fs];vals=[float(f['sst'][cell])-273.15 for f in fs];ax.plot(ts,vals,'o-',label=mode,ms=3)
   for t,v in zip(ts,vals):
    near=[ov for ot,ov in obs if abs((ot-t).total_seconds())<=1800]
    if near:buoyrows.append(dict(station=b['id'],mode=mode,valid=t.isoformat(),model_C=v,obs_C=float(np.mean(near)),error_K=v-float(np.mean(near))))
  ax.set(title=f"Buoy {b['id']}: nearest {cfg['mesh_label']} ocean cell",ylabel='Water temperature (C)');ax.grid(alpha=.25);ax.legend(ncol=3)
 fig.savefig(out/'buoy-temperatures.png',dpi=130);plt.close(fig)
 # Best-track comparison uses the greatest 10 m speed within 500 km of the observed center.
 track={}
 with gzip.open(cfg['best_track'],'rt') as stream:
  for line in stream:
   v=[x.strip() for x in line.split(',')];t=datetime.strptime(v[2],'%Y%m%d%H')
   if v[4]!='BEST' or not start<=t<=max(clock(f) for f in frames['pwp']):continue
   signed=lambda s:float(s[:-1])*.1*(-1 if s[-1] in 'SW' else 1)
   track[t]=(signed(v[6]),signed(v[7]),float(v[8])*.514444,float(v[9]))
 stormrows=[];fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
 ax.plot(list(track),[v[2] for v in track.values()],color='black',label='Best-track maximum sustained wind')
 for mode,fs in frames.items():
  tt=[];vv=[]
  for f in fs:
   t=clock(f)
   if t not in track:continue
   center=track[t];dist=np.linalg.norm(xyz(lat,lon)-xyz([center[0]],[center[1]])[0],axis=1)*6371
   ix=(dist<=cfg['storm_radius_km'])&mask;wind=np.hypot(f['u10'],f['v10']);value=float(wind[ix].max());tt.append(t);vv.append(value)
   stormrows.append(dict(mode=mode,valid=t.isoformat(),model_wind_ms=value,best_track_wind_ms=center[2],error_ms=value-center[2]))
   if (t-start).total_seconds()/3600 in cfg.get('wind_map_hours',[]):
    stamp=t.strftime('%Y%m%d%H');name=f'{mode}-wind-{stamp}'
    nc=work/f'{name}.nc'
    write_plane(nc,fakecp,yl,xl,t,{'wind10':('m/s',np.where(np.isfinite(oi[0]['sst']),wind[idx],np.nan))},source=f'{mode} forecast')
    windtable=work/'cf-wind-table.json'
    windtable.write_text(json.dumps([dict(names=['wind10'],standard_names=[],product='wind10',title=f'Ocean wind; observed center ring, maximum {center[2]:.1f} m/s',default_units='m/s',display_units='m/s',unit_factors={'m/s':1},scale=dict(levels=[0,2,4,6,8,10,12,15,18,22,26,30],colors=[[247,252,245],[229,245,224],[199,233,192],[161,217,155],[116,196,118],[65,171,93],[35,139,69],[0,109,44],[0,68,27],[128,0,38],[255,0,50]]))]))
    overlay=work/f'{name}-overlay.json'
    overlay.write_text(json.dumps(dict(points=[dict(lat=center[0],lon=center[1],shape='circle',radius_px=9,width_px=2,color='#111111')])))
    target=work/f'render-{name}'
    rc=run_cf_renderer(renderer,nc,variable='wind10',out_dir=target,valid=t.isoformat()+'Z',bounds=','.join(map(str,cfg['bounds'])),table=windtable,overlays=overlay,size=(1000,740),receipt=out/f'{name}.render.json')
    if rc:raise RuntimeError(f'wind render failed {rc}')
    pics=list(target.rglob('*.png'))
    if len(pics)!=1:raise RuntimeError(f'unexpected wind pictures {pics}')
    dest=out/f'{name}.png';shutil.copyfile(pics[0],dest);maps.append(dict(file=dest.name,valid=t.isoformat(),field='wind10',mode=mode,observed_center=center[:2]))
  ax.plot(tt,vv,'o-',label=mode)
 ax.set(title=f"Coarse-grid intensity proxy within {cfg['storm_radius_km']} km of observed center",ylabel='10 m wind speed (m/s)');ax.grid(alpha=.25);ax.legend();fig.savefig(out/'storm-wind.png',dpi=130);plt.close(fig)
 def savecsv(name,rows):
  if not rows:return
  with (out/name).open('w') as stream:
   writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
 savecsv('sst-metrics.csv',records);savecsv('buoy-matches.csv',buoyrows);savecsv('storm-wind.csv',stormrows)
 result=dict(frames={k:len(v) for k,v in frames.items()},maps=maps,sst=records,buoys=buoyrows,storm=stormrows,scope='Nearest ocean cell; OISST daily analysis at nominal noon, not instantaneous SST. No eyewall skill claim. Spatial RMSE uses latitude weights and valid OISST mask; coastal nearest-cell representativeness remains.')
 (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
