"""Rust physical weather maps and chart-only matplotlib verification figures."""
import argparse,json,os,shutil
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import numpy as np

COLORS=[[5,48,97],[33,102,172],[67,147,195],[146,197,222],[209,229,240],[247,247,247],[253,219,199],[244,165,130],[214,96,77],[178,24,43],[103,0,31]]

def weather(task):
    path,out,arm,field,detail=task;path,out=Path(path),Path(out)
    from woof.hex.ocean_products import write_plane,physical_renderer
    from woof.rustwx import run_cf_renderer
    with np.load(path) as z:d={k:z[k] for k in z.files}
    lat=np.unique(d['lat']);lon=np.unique(d['lon']);y,x=np.meshgrid(lat,lon,indexing='ij')
    iy=np.searchsorted(lat,d['lat']);ix=np.searchsorted(lon,d['lon']);plane=np.full(y.shape,np.nan)
    values=d['on']-d['off'] if arm=='difference' else d[arm+'_change']
    mask=d['mask'] if arm=='difference' else d['change_mask'];plane[iy,ix]=np.where(mask,values,np.nan)
    levels=[-5,-3,-2,-1,-.5,-.1,.1,.5,1,2,3,5]
    title='SST change since first complete forecast day'
    if detail:
        extent=max(.1,float(np.nanpercentile(np.abs(plane),99.5)));extent=float(np.ceil(extent*10)/10)
        levels=np.linspace(-extent,extent,12).tolist();title=f'ON minus OFF SST; detail clipped at +/-{extent:g} K'
    name=f'{path.stem}-{arm}';work=out/'work'/name;work.mkdir(parents=True,exist_ok=True)
    valid=datetime.strptime(path.stem.split('-')[-1],'%Y%m%d').replace(hour=12)
    source={'on':'PWP ON daily mean','off':'Fixed-SST OFF daily mean','obs':'NOAA OISST v2.1 daily analysis','difference':'PWP ON minus fixed-SST OFF'}[arm]
    nc=work/'field.nc';write_plane(nc,SimpleNamespace(asnumpy=np.asarray,asarray=np.asarray),y,x,valid,{'change':('K',plane)},source=source)
    table=work/'table.json';table.write_text(json.dumps([dict(names=['change'],standard_names=[],product='change',title=title,default_units='K',display_units='K',unit_factors={'K':1},scale=dict(levels=levels,colors=COLORS))]))
    bounds=','.join(str(float(v)) for v in [lon.min(),lon.max(),lat.min(),lat.max()]);receipt=out/(name+'.render.json')
    rc=run_cf_renderer(physical_renderer(),nc,variable='change',out_dir=work/'render',valid=valid.isoformat()+'Z',bounds=bounds,table=table,size=(1000,740),receipt=receipt)
    if rc:raise RuntimeError(f'{name}: renderer {rc}')
    pics=list((work/'render').rglob('*.png'))
    if len(pics)!=1:raise RuntimeError(pics)
    target=out/(name+'.png');shutil.copyfile(pics[0],target)
    return dict(file=target.name,source=source,valid=valid.isoformat(),scale=levels,minimum=float(np.nanmin(plane)),maximum=float(np.nanmax(plane)),clipped=int(np.sum((plane<levels[0])|(plane>levels[-1]))))

def charts(result,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    label=result['case']['case'];fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,layout='constrained')
    for arm,color in [('on','#2166ac'),('off','#b2182b')]:
        rows=result['track_rows'][arm];t=[datetime.fromisoformat(r['valid']) for r in rows]
        for ax,key in zip(axes,['vmax10_ms','mslp_min_hpa','track_error_km']):ax.plot(t,[r.get(key) if r['ok'] else None for r in rows],'o-',ms=3,label=arm.upper(),color=color)
        if arm=='on':
            for ax,key in zip(axes[:2],['observed_wind_ms','observed_pressure_hpa']):ax.plot(t,[r.get(key) for r in rows],'k-',label='IBTrACS USA')
    for ax,title,ylabel in zip(axes,[label+' maximum wind',label+' minimum pressure',label+' model-only track error'],['m/s','hPa','km']):ax.set(title=title,ylabel=ylabel);ax.grid(alpha=.25);ax.legend()
    fig.savefig(out/'storm-evolution.png',dpi=140);plt.close(fig)
    days=result['sst']['days'];fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for arm,color in [('on','#2166ac'),('off','#b2182b')]:
        t=[datetime.fromisoformat(r['date']) for r in days]
        for ax,key in zip(axes,['wake_'+arm,'wake_change_'+arm]):ax.plot(t,[r[key]['rmse'] for r in days],'o-',label=arm.upper(),color=color)
    for ax,title in zip(axes,['SST RMSE inside observed-track corridor','SST-change RMSE inside observed-track corridor']):ax.set(title=label+' '+title,ylabel='K');ax.grid(alpha=.25);ax.legend()
    fig.savefig(out/'sst-scores.png',dpi=140);plt.close(fig)
    buoys=[b for b in result['buoys'] if b.get('rows')]
    if buoys:
        fig,axes=plt.subplots(len(buoys),1,figsize=(10,3.5*len(buoys)),squeeze=False,layout='constrained')
        for ax,b in zip(axes.ravel(),buoys):
            t=[datetime.fromisoformat(r['valid']) for r in b['rows']]
            for arm,color in [('obs','black'),('on','#2166ac'),('off','#b2182b')]:ax.plot(t,[r[arm] for r in b['rows']],'o-',ms=3,color=color,label=arm.upper())
            ax.set(title=label+' buoy '+b['station'],ylabel='C');ax.grid(alpha=.25);ax.legend()
        fig.savefig(out/'buoy-temperatures.png',dpi=140);plt.close(fig)
    argo=result['argo']
    if 'depth_bottom_m' in argo:
        fig,axes=plt.subplots(1,2,figsize=(9,6),sharey=True,layout='constrained');z=np.asarray(argo['depth_bottom_m']);mid=(z+np.r_[0,z[:-1]])/2
        for ax,key,unit in zip(axes,['TEMP','PSAL'],['C','PSU']):
            if 'model' not in argo[key]:continue
            ax.plot(argo[key]['model'],mid,label='PWP ON');ax.plot(argo[key]['observed_layer_mean'],mid,'k.-',label='Argo QC 1/2');ax.set(xlabel=unit,title=key);ax.grid(alpha=.25);ax.legend()
        axes[0].invert_yaxis();axes[0].set_ylabel('Depth (m)');fig.suptitle(label+' '+argo['source']['time']);fig.savefig(out/'argo-profile.png',dpi=140);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['RUSTWX_RASTER_DEVICE']='cpu';result=json.loads((a.evaluation/'metrics.json').read_text());files=sorted(a.evaluation.glob('sst-*.npz'));tasks=[]
    for f in files[1:]:
        for arm in ('on','obs'):tasks.append((str(f),str(a.output),arm,'change',False))
    if files:
        tasks.extend([(str(files[-1]),str(a.output),'off','change',False),(str(files[-1]),str(a.output),'difference','difference',True)])
    with ProcessPoolExecutor(4) as pool:maps=list(pool.map(weather,tasks))
    charts(result,a.output);(a.output/'pictures.json').write_text(json.dumps(maps,indent=2)+'\n')
if __name__=='__main__':main()
