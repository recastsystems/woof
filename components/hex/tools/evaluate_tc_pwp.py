"""Read-only TC/PWP verification against independent observations.

Model tracker receives only the initial observed centre. Subsequent observed
positions define verification distances and a 200 km SST corridor, never a fix.
Daily SST means use all five six-hour endpoints with trapezoidal weights.
"""
from __future__ import annotations
import argparse,csv,gzip,json
from datetime import datetime,timedelta
from pathlib import Path
from dataclasses import asdict
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from woof.hex.storm_tracker import TrackSession,TrackerConfig,StateFields,distance_km,xyz

LEVELS=(85000,70000,50000,30000,20000)

def stamp(value):
    return datetime.fromisoformat(str(value).strip().replace('_','T').rstrip('Z'))

def metric(model,obs):
    model,obs=np.broadcast_arrays(np.asarray(model,float),np.asarray(obs,float))
    good=np.isfinite(model)&np.isfinite(obs);e=(model-obs)[good]
    return dict(n=int(e.size),bias=float(e.mean()) if e.size else None,
                mae=float(np.abs(e).mean()) if e.size else None,
                rmse=float(np.sqrt(np.mean(e*e))) if e.size else None)

def read(path,profiles=False):
    omitted={'ocean_temperature','ocean_salinity','ocean_layer_bottom'}
    with np.load(path) as z:return {k:z[k] for k in z.files if profiles or k not in omitted}

def index(folder):
    result={}
    for p in sorted(Path(folder).glob('*.npz')):
        with np.load(p) as z:result[stamp(z['valid'].item())]=p
    if not result:raise ValueError(f'No reduced history in {folder}')
    return result

def coords(d):
    # Reducers declare angular units explicitly; legacy hex history is radians.
    lat,lon=np.asarray(d['latCell']).ravel(),np.asarray(d['lonCell']).ravel()
    if str(d.get('coordinate_units','radians'))!='degrees':lat,lon=np.degrees(lat),np.degrees(lon)
    return lat,(lon+180)%360-180

def sst(d):return np.asarray(d.get('sst',d['tsk']),float).ravel()-273.15

def nearest(d,lat,lon,max_km,wet_only=False):
    a,b=coords(d);indices=np.arange(a.size)
    if wet_only and 'landmask' in d:indices=indices[np.asarray(d['landmask']).ravel()<.5]
    dist,idx=cKDTree(xyz(a[indices],b[indices])).query(xyz(np.asarray(lat).ravel(),np.asarray(lon).ravel()),workers=6)
    return indices[idx],dist*6371.229<=max_km

def wind10(d):
    # Published initializer MSG (-1.0e30f), not a physical wind or a clip limit.
    u,v=np.asarray(d['u10'],float),np.asarray(d['v10'],float)
    sentinel=float(np.float32(-1.e30))
    valid=np.isfinite(u)&np.isfinite(v)&(u!=sentinel)&(v!=sentinel)
    return np.where(valid,np.hypot(u,v),np.nan)

def track(files,observed,start):
    initial=observed[start];session=TrackSession(TrackerConfig(initial_center_deg=[float(initial['LAT']),float(initial['LON'])]));rows=[]
    for valid,p in sorted(files.items()):
        d=read(p);lat,lon=coords(d);w10=wind10(d)
        fields=StateFields(lat=lat,lon=lon,mslp_pa=d['mslp_pa'].ravel(),
            u_p={k:d[f'u_{k}'].ravel() for k in LEVELS if f'u_{k}' in d},
            v_p={k:d[f'v_{k}'].ravel() for k in LEVELS if f'v_{k}' in d},
            wind10=w10.ravel() if np.isfinite(w10).any() else None)
        fix=session.observe((valid-start).total_seconds(),fields);r=fix.as_dict();r['valid']=valid.isoformat();r['valid_wind10_cells']=int(np.isfinite(w10).sum())
        if valid in observed:
            o=observed[valid];r['observed_lat']=float(o['LAT']);r['observed_lon']=float(o['LON'])
            for dest,source,scale in [('observed_wind_ms','USA_WIND',.514444),('observed_pressure_hpa','USA_PRES',1.)]:
                r[dest]=float(o[source])*scale if o.get(source) else None
            if fix.ok:r['track_error_km']=float(distance_km(fix.lat,fix.lon,float(o['LAT']),float(o['LON'])))
        rows.append(r)
    good=[r for r in rows if 'track_error_km' in r]
    summary=dict(fixes=len(rows),misses=sum(not r['ok'] for r in rows),matched_fixes=len(good),
        track_km=metric([r['track_error_km'] for r in good],0.),
        wind_ms=metric([r.get('vmax10_ms',np.nan) or np.nan for r in good],[r.get('observed_wind_ms') or np.nan for r in good]),
        pressure_hpa=metric([r.get('mslp_min_hpa',np.nan) or np.nan for r in good],[r.get('observed_pressure_hpa') or np.nan for r in good]))
    return rows,summary

def oi(path):
    with Dataset(path) as d:
        lat=np.asarray(d['lat'][:]);lon=np.asarray(d['lon'][:]);v=np.ma.filled(d['sst'][:],np.nan).squeeze()
    y,x=np.meshgrid(lat,lon,indexing='ij');return y.ravel(),((x+180)%360-180).ravel(),v.ravel()

def daily_sst(on,off,observed,shared,out,max_km):
    first,last=min(on),max(on);day=first.replace(hour=0,minute=0,second=0)
    if day<first:day+=timedelta(days=1)
    # Crop the global observation grid to the actual first model footprint.
    key=day.strftime('%Y%m%d');lat,lon,_=oi(shared/'oisst'/key/f'oisst-avhrr-v02r01.{key}.nc')
    base=read(on[first]);mlat,mlon=coords(base);sel=(lat>=mlat.min()-.3)&(lat<=mlat.max()+.3)&(lon>=mlon.min()-.3)&(lon<=mlon.max()+.3)
    lat,lon=lat[sel],lon[sel]
    positions=[(float(r['LAT']),float(r['LON'])) for t,r in observed.items() if first<=t<=last]
    corridor=np.min(np.stack([distance_km(lat,lon,a,b) for a,b in positions]),axis=0)<=200
    rows=[];initial=None;allvalues={a:[] for a in ('on','off','obs')};wakevalues={a:[] for a in ('on','off','obs')}
    while day+timedelta(days=1)<=last:
        times=[day+timedelta(hours=h) for h in (0,6,12,18,24)]
        if not all(t in on and t in off for t in times):raise ValueError(f'Missing daily mean endpoint {day}')
        key=day.strftime('%Y%m%d');_,_,obs=oi(shared/'oisst'/key/f'oisst-avhrr-v02r01.{key}.nc');obs=obs[sel]
        values={};valid=np.isfinite(obs)
        for arm,files in [('on',on),('off',off)]:
            pieces=[]
            for t in times:
                d=read(files[t]);idx,near=nearest(d,lat,lon,max_km);temp=sst(d)[idx]
                # The ON active mask defines a common coverage mask for both arms.
                reference=d if arm=='on' else read(on[t]);ri,rnear=nearest(reference,lat,lon,max_km)
                active=np.asarray(reference['ocean_pwp_active']).ravel()[ri]>.5
                near &= rnear&active&np.isfinite(temp);valid &= near;pieces.append(temp)
            values[arm]=np.average(pieces,axis=0,weights=[.5,1,1,1,.5])
        values['obs']=obs
        if initial is None:initial={a:v.copy() for a,v in values.items()};initial['mask']=valid.copy()
        changevalid=valid&initial['mask'];wake=valid&corridor
        row=dict(date=day.isoformat(),pixels=int(valid.sum()),wake_pixels=int(wake.sum()),on=metric(values['on'][valid],obs[valid]),off=metric(values['off'][valid],obs[valid]),
            wake_on=metric(values['on'][wake],obs[wake]),wake_off=metric(values['off'][wake],obs[wake]))
        for arm in ('on','off'):
            row['change_'+arm]=metric((values[arm]-initial[arm])[changevalid],(obs-initial['obs'])[changevalid])
            row['wake_change_'+arm]=metric((values[arm]-initial[arm])[changevalid&corridor],(obs-initial['obs'])[changevalid&corridor])
        np.savez_compressed(out/f'sst-{key}.npz',lat=lat,lon=lon,mask=valid,wake=corridor,change_mask=changevalid,
            **values,**{a+'_change':v-initial[a] for a,v in values.items()})
        for a in allvalues:allvalues[a].extend(values[a][valid]);wakevalues[a].extend(values[a][wake])
        rows.append(row);day+=timedelta(days=1)
    return dict(days=rows,all_on=metric(allvalues['on'],allvalues['obs']),all_off=metric(allvalues['off'],allvalues['obs']),
                wake_on=metric(wakevalues['on'],wakevalues['obs']),wake_off=metric(wakevalues['off'],wakevalues['obs']))

def buoy_text(path):
    raw=path.read_bytes();raw=gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw
    lines=raw.decode().splitlines();header=lines[0].lstrip('#').split();rows=[]
    for line in lines[1:]:
        if line.startswith('#'):continue
        values=line.split()
        if len(values)!=len(header):continue
        d=dict(zip(header,values))
        try:
            temp=float(d['WTMP']);year=int(d.get('YY',d.get('YYYY')))
            if year<100:year+=2000
            if temp>45 or temp< -3:continue
            rows.append((datetime(year,int(d['MM']),int(d['DD']),int(d['hh']),int(d.get('mm',0))),temp))
        except (ValueError,KeyError):continue
    return rows

def buoys(on,off,case,shared,coordinates,max_km):
    result=[]
    for station in case['buoys']:
        meta=next((r for r in coordinates if r['station']==station and r['year']==case['year']),{})
        if 'latitude' not in meta:result.append(dict(station=station,error=meta.get('error','missing historical coordinate')));continue
        obs=buoy_text(shared/'ndbc'/str(case['year'])/f'{station}h{case["year"]}.txt.gz');rows=[]
        for valid in sorted(on.keys()&off.keys()):
            sample=[v for t,v in obs if abs((t-valid).total_seconds())<=1800]
            if not sample:continue
            r=dict(valid=valid.isoformat(),obs=float(np.mean(sample)))
            for arm,files in [('on',on),('off',off)]:
                d=read(files[valid]);idx,near=nearest(d,[meta['latitude']],[meta['longitude']],max_km,wet_only=True)
                r[arm]=float(sst(d)[idx[0]]) if near[0] else None
            rows.append(r)
        result.append(dict(station=station,coordinates=meta,rows=rows,
            on=metric([r['on'] if r['on'] is not None else np.nan for r in rows],[r['obs'] for r in rows]),
            off=metric([r['off'] if r['off'] is not None else np.nan for r in rows],[r['obs'] for r in rows])))
    return result

def argo(on,case,receipt,max_km):
    info=next((r for r in receipt['selected'] if r['case']==case['case']),None)
    if not info or 'path' not in info:return dict(error='No profile acquired')
    t=stamp(info['time']);times=sorted(on);before=[x for x in times if x<=t];after=[x for x in times if x>=t]
    if not before or not after:return dict(error='Profile outside forecast window',source=info)
    ta,tb=before[-1],after[0];w=(t-ta).total_seconds()/max((tb-ta).total_seconds(),1)
    profiles=[]
    for tt in (ta,tb):
        d=read(on[tt],profiles=True);idx,near=nearest(d,[float(info['index']['latitude'])],[float(info['index']['longitude'])],max_km)
        if not near[0] or np.asarray(d['ocean_pwp_active']).ravel()[idx[0]]<.5:return dict(error='Profile outside coupled footprint',source=info)
        profiles.append({k:np.asarray(d[k])[:,idx[0]] for k in ('ocean_temperature','ocean_salinity','ocean_layer_bottom')})
    model={k:(1-w)*profiles[0][k]+w*profiles[1][k] for k in profiles[0]};z=model['ocean_layer_bottom'];top=np.r_[0,z[:-1]]
    result=dict(source=info,bracketing_times=[ta.isoformat(),tb.isoformat()],pressure_depth_approximation='1 dbar treated as 1 m in upper 200 m',depth_bottom_m=z.tolist())
    with Dataset(info['path']) as nc:
        for name,key,offset in [('TEMP','ocean_temperature',273.15),('PSAL','ocean_salinity',0)]:
            chosen=name+'_ADJUSTED' if name+'_ADJUSTED' in nc.variables else name
            pchosen='PRES_ADJUSTED' if 'PRES_ADJUSTED' in nc.variables else 'PRES'
            p=np.ma.filled(nc[pchosen][0],np.nan);v=np.ma.filled(nc[chosen][0],np.nan)
            q=np.asarray(nc[chosen+'_QC'][0]).astype('U1');pq=np.asarray(nc[pchosen+'_QC'][0]).astype('U1')
            good=np.isfinite(p)&np.isfinite(v)&np.isin(q,['1','2'])&np.isin(pq,['1','2']);p,v=p[good],v[good];order=np.argsort(p);p,v=p[order],v[order]
            if len(p)<2:result[name]=dict(error='Insufficient QC 1 or 2 profile samples');continue
            means=[]
            for lo,hi in zip(top,z):
                if lo<p.min() or hi>p.max():means.append(np.nan);continue
                knots=np.r_[lo,p[(p>lo)&(p<hi)],hi];means.append(float(np.trapezoid(np.interp(knots,p,v),knots)/(hi-lo)))
            mod=model[key]-offset;result[name]=dict(variable=chosen,metric=metric(mod,means),model=mod.tolist(),observed_layer_mean=means)
    return result

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--on',type=Path,required=True);p.add_argument('--off',type=Path,required=True);p.add_argument('--surface-on',type=Path);p.add_argument('--surface-off',type=Path);p.add_argument('--observations',type=Path,required=True);p.add_argument('--shared',type=Path,default=Path('/workspace/atlas/_shared'));p.add_argument('--out',type=Path,required=True);p.add_argument('--max-distance-km',type=float,default=25.);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);case=next(c for c in json.loads(a.cases.read_text()) if c['case']==a.case);start=stamp(case['start']);observed={stamp(r['ISO_TIME']):r for r in json.loads((a.observations/(a.case+'-besttrack.json')).read_text())};on,off=index(a.on),index(a.off)
    rows_on,m_on=track(on,observed,start);rows_off,m_off=track(off,observed,start);son=index(a.surface_on) if a.surface_on else on;soff=index(a.surface_off) if a.surface_off else off
    result=dict(case=case,tracker=dict(on=m_on,off=m_off),track_rows=dict(on=rows_on,off=rows_off),sst=daily_sst(son,soff,observed,a.shared,a.out,a.max_distance_km),
        buoys=buoys(son,soff,case,a.shared,json.loads((a.observations/'buoy-coordinates.json').read_text()),a.max_distance_km),argo=argo(son,case,json.loads((a.observations/'argo-fetch.json').read_text()),a.max_distance_km),
        conventions=dict(track='Published model-only tracker, seed at initialization only; misses retained',wind='IBTrACS USA_WIND one-minute knots converted to m/s; gridded model maxima are resolution limited; initializer MSG=-1e30f is missing, with valid cell counts retained',sst='OISST daily mean against trapezoidal six-hour forecast means on common PWP-supported cells; wake corridor within 200 km of observed track',argo='QC 1/2 adjusted profiles where available; forecast-time profile is independent of initial analysis; no fabricated OFF subsurface ocean'))
    (a.out/'metrics.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(clean({k:v for k,v in result.items() if k in ('case','tracker')}),indent=2))
if __name__=='__main__':main()
