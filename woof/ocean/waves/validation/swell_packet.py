"""Source-free 3000 km transport check in the single authorized smoke.

Finite Gaussian packet, Cartesian 25 km cells, periodic transverse boundary.
Compare the exact translation and the analytic constant-tensor GSE diffusion.
No atmospheric forcing, physical source, partition or frequency operator is run.
All field work and reductions are on the GPU; output is an analysis chart.
"""
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import cupy as cp
import numpy as np
from woof.ocean.waves.backends.structured import StructuredBackend


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    nx,ny,dx,dy=256,80,25000.,25000.
    shape=(ny,nx);nc=nx*ny;nf,nd=36,36
    grid=SimpleNamespace(nx=nx,ny=ny,dx=dx,dy=dy,
        latitude=cp.zeros(shape,cp.float32),longitude=cp.zeros(shape,cp.float32),
        cosalpha=cp.ones(shape,cp.float32),sinalpha=cp.zeros(shape,cp.float32))
    bathy=SimpleNamespace(depth=cp.full(shape,4000,cp.float32),wet=cp.ones(shape,cp.float32))
    b=StructuredBackend(grid,bathy,kind='curvilinear',map_factor=cp.ones(shape,cp.float32))
    cells=cp.arange(nc,dtype=cp.int32);row,col=cp.divmod(cells,nx)
    b.neighbor[:,2]=((row-1)%ny)*nx+col;b.neighbor[:,3]=((row+1)%ny)*nx+col
    b.boundary_slot[:,2:]=-1
    frequency=.035*cp.power(cp.float32(1.1),cp.arange(nf,dtype=cp.float32))
    fbin=4;cg=cp.broadcast_to((9.80665/(4*math.pi*frequency))[None,:],(nc,nf)).copy()
    speed=float(cg[0,fbin]);period=1/float(frequency[fbin])
    x=(col+.5)*dx;y=(row+.5)*dy
    sx,sy=200000.,200000.;x0,y0=1000000.,ny*dy/2
    variance=cp.exp(-.5*((x-x0)/sx)**2-.5*((y-y0)/sy)**2).astype(cp.float32)
    df=float(frequency[fbin])*(math.sqrt(1.1)-1/math.sqrt(1.1));weight=2*math.pi*float(frequency[fbin])*df*2*math.pi/nd
    state=cp.zeros((nc,nf,nd),cp.float32);state[:,fbin,0]=variance/weight
    bound=b.max_dt(.6*9.80665/(2*math.pi*.035))
    target=3000000./speed;steps=math.ceil(target/600);dt=target/steps
    pieces=math.ceil(dt/bound);h=dt/pieces
    # Allocate backend work before capture; the probe is not copied into state.
    work=b.advect_group(state,cg,h,cp.cuda.get_current_stream(),None,0,6)
    cp.copyto(work[:,:6],state[:,:6])
    cp.cuda.get_current_stream().synchronize()
    def advance():
        for _ in range(pieces):
            advanced=b.advect_group(state,cg,h,cp.cuda.get_current_stream(),None,0,6)
            cp.copyto(state[:,:6],advanced[:,:6])
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture();advance();graph=stream.end_capture()
    cp.cuda.get_current_stream().synchronize()
    mass0=float(cp.sum(variance,dtype=cp.float64));records=[];profiles=[]
    dss=(speed*(math.sqrt(1.1)-1/math.sqrt(1.1)))**2*3600/12
    dnn=(speed*2*math.pi/nd)**2*3600/12
    def sample(step):
        t=step*dt;v=state[:,fbin,0]*weight
        exact=cp.exp(-.5*((x-x0-speed*t)/sx)**2-.5*((y-y0)/sy)**2)
        sxn=math.sqrt(sx*sx+2*dss*t);syn=math.sqrt(sy*sy+2*dnn*t)
        analytic=(sx*sy/(sxn*syn))*cp.exp(-.5*((x-x0-speed*t)/sxn)**2-.5*((y-y0)/syn)**2)
        mass=cp.sum(v,dtype=cp.float64)
        records.append({'distance_km':speed*t/1000,'hour':t/3600,
            'relative_mass_error':float(mass/mass0-1),
            'centroid_error_m':float(cp.sum(v*x,dtype=cp.float64)/mass-(x0+speed*t)),
            'peak_hs_ratio_to_translation':float(cp.sqrt(cp.max(v)/cp.max(exact))),
            'peak_hs_ratio_to_gse_analytic':float(cp.sqrt(cp.max(v)/cp.max(analytic))),
            'relative_l2_to_gse_analytic':float(cp.sqrt(cp.sum((v-analytic)**2)/cp.sum(analytic**2))),
            'minimum_variance':float(cp.min(v))})
        profiles.append({'distance_km':speed*t/1000,
            'x_relative_km':cp.asnumpy((x.reshape(shape)[ny//2]-x0-speed*t)/1000).tolist(),
            'hs_model':cp.asnumpy(4*cp.sqrt(cp.maximum(v.reshape(shape)[ny//2],0))).tolist(),
            'hs_translation':cp.asnumpy(4*cp.sqrt(exact.reshape(shape)[ny//2])).tolist(),
            'hs_gse_analytic':cp.asnumpy(4*cp.sqrt(analytic.reshape(shape)[ny//2])).tolist()})
    sample(0)
    for step in range(1,steps+1):
        graph.launch(stream=cp.cuda.get_current_stream())
        if step in (steps//3,2*steps//3,steps):sample(step)
    last=records[-1]
    passed=(abs(last['relative_mass_error'])<2e-4 and abs(last['centroid_error_m'])<dx
            and last['peak_hs_ratio_to_translation']>.95 and last['relative_l2_to_gse_analytic']<.05
            and all(r['minimum_variance']>=-1e-7 for r in records))
    receipt={'status':'PASS' if passed else 'FAIL','source_terms':'none; backend transport only',
        'geometry':'Cartesian; periodic transverse direction; open along track',
        'grid':[ny,nx],'dx_m':dx,'dy_m':dy,'nf':nf,'nd':nd,'active_frequency_bin':fbin,
        'period_s':period,'group_speed_ms':speed,'distance_m':3000000,'steps':steps,
        'dt_seconds':dt,'pieces_per_call':pieces,'gse_horizon_seconds':3600,
        'gse_diffusion_m2s':[dss,dnn],'initial_gaussian_sigma_m':[sx,sy],
        'acceptance':{'mass_error':2e-4,'centroid_error_m':dx,'peak_hs_retention':.95,'gse_relative_l2':.05},
        'records':records,'profiles':profiles,'wall_seconds':time.perf_counter()-started}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def charts(output,receipt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for p in receipt['profiles']:
        axes[0].plot(p['x_relative_km'],p['hs_model'],label=f"{p['distance_km']:.0f} km")
    p=receipt['profiles'][-1]
    axes[0].plot(p['x_relative_km'],p['hs_translation'],'k--',label='Exact translation')
    axes[0].plot(p['x_relative_km'],p['hs_gse_analytic'],'k:',label='Analytic GSE diffusion')
    axes[0].set(xlim=(-800,800),xlabel='Distance from moving center (km)',ylabel='Hs (m)')
    r=receipt['records'];d=[v['distance_km'] for v in r]
    axes[1].plot(d,[v['peak_hs_ratio_to_translation'] for v in r],'o-',label='Peak Hs / translated peak')
    axes[1].plot(d,[1+v['relative_mass_error'] for v in r],'o-',label='Integrated variance / initial')
    axes[1].set(xlabel='Path length (km)',ylabel='Retained fraction',ylim=(.94,1.01))
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle(f"Source-free {receipt['period_s']:.2f} s packet, 25 km grid, {receipt['status']}")
    fig.savefig(Path(output)/'packet.png',dpi=140);plt.close(fig)
