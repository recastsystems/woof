"""Isolated Earth2Studio GPU driver. Does not decode atmospheric file formats."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def main(request):
    start=time.monotonic()
    import numpy as np
    import torch
    import torch.nn.functional as F
    import cupy as cp
    from earth2studio.models.dx import TCTrackerWuDuan, TCTrackerVitart
    from datetime import datetime
    r=json.loads(Path(request).read_text());config=r['config'];member=r['member']
    torch.set_num_threads(os.cpu_count() or 1)
    stages=[dict(stage='gpu_backend_import',seconds=time.monotonic()-start,parallelism='1 Python orchestration thread; framework loads')]
    peak=[0];stop=threading.Event()
    def sample():
        while not stop.is_set():
            p=subprocess.run([*__import__('woof.gpu_smi', fromlist=['command']).command(),'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
            for line in p.stdout.splitlines():
                parts=line.split(',')
                if len(parts)==2 and parts[0].strip()==str(os.getpid()):
                    try:peak[0]=max(peak[0],int(parts[1]))
                    except ValueError:pass
            stop.wait(0.1)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    init=datetime.fromisoformat(member['init'].replace('Z','+00:00'))
    cls=TCTrackerWuDuan if r['backend']=='wuduan' else TCTrackerVitart
    kw=config['constructor']
    tracker=cls(**kw).to('cuda');tracker.reset_path_buffer()
    coords=tracker.input_coords();coords['batch']=np.array([0])
    fields=list(coords['variable'])
    if set(fields)!=set(config['fields']):raise ValueError('tracker field list differs from authority')
    state={};counters=[];edge_bytime=[];rejected=[0]
    original=tracker._find_centers
    def filter_centers(centers):
        finite=torch.isfinite(centers).all(-1)&(centers[:,0]!=tracker.PATH_FILL_VALUE)
        iy=((90-centers[:,0])*4).round().long().clamp(0,720)
        ix=(centers[:,1]*4).round().long()%1440
        keep=finite&state['mask'][iy,ix]
        rejected[0]+=int((finite&~keep).sum().item())
        return centers[keep] if bool(keep.any()) else torch.full((1,4),tracker.PATH_FILL_VALUE,device='cuda')
    if r['backend']=='wuduan':
        def masked(lat,lon,vort,w10,msl,**kw):
            # Remove derivative discontinuities in a one-cell halo. Filled cells
            # cannot form a connected vorticity object or a candidate centre.
            vort=vort.masked_fill(~state['interior'],0)
            kw.setdefault('vort850_threshold',config['vorticity_threshold'])
            return filter_centers(original(lat,lon,vort,w10.masked_fill(~state['mask'],0),msl.masked_fill(~state['mask'],float('inf')),**kw))
    else:
        def masked(lat,lon,msl,w10,vort,temp,dz,**kw):
            mask=state['mask'];inside=state['interior']
            vort=vort.masked_fill(~inside,0)
            # Upstream assumes each extrema list is nonempty before dist.min().
            # A masked regional field can legitimately contain none.
            options=dict(min_distance=10,exclude_border=kw.get('exclude_border',True))
            extrema=(tracker.get_local_max(vort,threshold_abs=kw.get('vorticity_threshold',3.5e-5),**options),
                     tracker.get_local_max(temp.masked_fill(~mask,float('-inf')),**options),
                     tracker.get_local_max(dz.masked_fill(~mask,float('-inf')),exclude_border=kw.get('exclude_border',True)))
            if any(len(indices)==0 for indices in extrema):
                return torch.full((1,4),tracker.PATH_FILL_VALUE,device='cuda')
            return filter_centers(original(lat,lon,msl.masked_fill(~mask,float('inf')),w10.masked_fill(~mask,0),vort,temp.masked_fill(~mask,float('-inf')),dz.masked_fill(~mask,float('-inf')),**kw))
    tracker._find_centers=masked
    routes=set();output=None
    latglobal=torch.linspace(90,-90,721,device='cuda',dtype=torch.float64)
    longlobal=torch.arange(1440,device='cuda',dtype=torch.float64)/4
    for fi,frame in enumerate(r['frames']):
        ts=time.monotonic();d=Path(frame['directory']);grid=json.loads((d/'grid.json').read_text());routes.add(grid['route'])
        ys=torch.tensor(grid['lat'],device='cuda');xs=torch.tensor(grid['lon'],device='cuda')%360
        iy=((90-ys)*4).round().long();ix=(xs*4).round().long()
        cropmask=torch.from_numpy(np.fromfile(d/'mask.u8',dtype=np.uint8).reshape(grid['shape'])).to('cuda').bool()
        mask=torch.zeros((721,1440),dtype=torch.bool,device='cuda');mask[iy[:,None],ix[None,:]]=cropmask
        # Periodic longitude, replicated polar rows. No fictional halo data.
        padded=F.pad(mask[None,None].float(),(1,1,0,0),mode='circular');padded=F.pad(padded,(0,0,1,1),mode='replicate')
        interior=F.avg_pool2d(padded,3,stride=1)[0,0]>=1
        state.update(mask=mask,interior=interior)
        native=grid['route']=='global_exact_lattice' and bool(mask.all())
        # Complete global input uses the upstream detector on its own lattice.
        # Only crops or missing signal need the coverage-aware wrapper.
        tracker._find_centers=original if native else masked
        x=torch.zeros((1,len(fields),721,1440),device='cuda')
        for vi,name in enumerate(fields):
            raw=np.fromfile(d/f'{name}.f32',dtype='<f4').reshape(grid['shape'])
            plane=torch.from_numpy(raw).to('cuda')
            x[0,vi,iy[:,None],ix[None,:]]=torch.nan_to_num(plane)
        stages.append(dict(stage='gpu_stage_frame',valid=frame['valid'],seconds=time.monotonic()-ts,parallelism='GPU + native float buffer reads'))
        ts=time.monotonic();output,_=tracker(x,coords)
        # One bounded result copy per completed valid time, also completes the
        # tracker call. No profiling synchronization is added to the algorithm.
        values=output[0].detach().cpu().numpy()
        edge=mask&~interior;ey,ex=torch.where(edge)
        edgelat=latglobal[ey];edgelon=longlobal[ex]
        flags={}
        for pi in range(values.shape[0]):
            la,lo=values[pi,-1,:2]
            if not np.isfinite(la+lo):continue
            if len(ey):
                a=torch.deg2rad(edgelat);b=float(la)*np.pi/180
                h=torch.sin((a-b)/2)**2+torch.cos(a)*np.cos(b)*torch.sin(torch.deg2rad(edgelon-float(lo))/2)**2
                km=12742*torch.asin(torch.sqrt(h.clamp(0,1)))
                flags[pi]=['edge'] if float(km.min())<=config['edge_radius_km'] else []
            else:flags[pi]=[]
        edge_bytime.append(flags)
        counters.append(dict(valid=frame['valid'],real_cells=int(mask.sum().item()),filled_cells=721*1440-int(mask.sum().item()),coverage_mask_applied=not native,rejected_filled_cell_detections=rejected[0]))
        stages.append(dict(stage='gpu_detect_associate',valid=frame['valid'],seconds=time.monotonic()-ts,parallelism='GPU detection; upstream CPU shape/short path association'))
        print(f"{frame['valid']} paths={values.shape[0]} fixes={int(np.isfinite(values[:,-1,0]).sum())}",flush=True)
    tracks=[]
    for pi,path in enumerate(values):
        fixes=[]
        for fi,(frame,v) in enumerate(zip(r['frames'],path)):
            finite=bool(np.isfinite(v).all())
            fixes.append(dict(valid=frame['valid'],lead_hours=(datetime.fromisoformat(frame['valid'].replace('Z','+00:00'))-init).total_seconds()/3600,
                              lat=float(v[0]) if finite else None,lon=(float(v[1])+180)%360-180 if finite else None,
                              mslp_pa=float(v[2]) if finite else None,max_wind10_ms=float(v[3]) if finite else None,
                              flags=edge_bytime[fi].get(pi,[]) if finite else ['missing']))
        tracks.append(dict(track_id=f'{member["member_id"]}-{pi:03d}',member_id=member['member_id'],weight=member['weight'],fixes=fixes))
    if not tracks:
        tracks=[dict(track_id=member['member_id']+'-none',member_id=member['member_id'],weight=member['weight'],fixes=[dict(valid=f['valid'],lead_hours=(datetime.fromisoformat(f['valid'].replace('Z','+00:00'))-init).total_seconds()/3600,lat=None,lon=None,mslp_pa=None,max_wind10_ms=None,flags=['missing']) for f in r['frames']])]
    peak[0]=max(peak[0],round((torch.cuda.max_memory_reserved()+cp.get_default_memory_pool().total_bytes())/1048576))
    stop.set();monitor.join()
    result=dict(tracks=tracks,stages=stages,route=sorted(routes),peak_vram_mib=peak[0],coverage_route='native_global_lattice' if all(not c['coverage_mask_applied'] for c in counters) else 'explicit_no_signal_mask',mask_receipts=counters,earth2studio_version=__import__('importlib.metadata').metadata.version('earth2studio'))
    Path(r['out']).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main(sys.argv[1])
