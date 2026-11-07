"""Source-free geodesic packet, exact Hamiltonian backward-ray reference.

Holthuijsen (2007), doi:10.1017/CBO9780511618536, ch.7.
Only geographic and angular operators run, with their production ordering.
All field calculations/reductions run on CUDA. Figures are analysis charts.
"""
import json,math,time
from pathlib import Path
from types import SimpleNamespace
import cupy as cp
import numpy as np
from ..action import ActionSolver
from ..backends.structured import StructuredBackend
R=6371000.


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    lo,la=cp.meshgrid(cp.arange(155.,215.01,.5,dtype=cp.float32),cp.arange(8.,58.01,.5,dtype=cp.float32))
    ny,nx=lo.shape
    b=StructuredBackend(SimpleNamespace(nx=nx,ny=ny,latitude=la,longitude=lo),
        SimpleNamespace(depth=cp.full(lo.shape,4000,cp.float32),wet=cp.ones(lo.shape,cp.float32)))
    m=ActionSolver(None,None,{'grid_backend':b,'source_pack':'st4'})
    nf,nd=m.nf,m.nd;fb=4;f=float(m.frequency[fb]);speed=float(m.cg[0,fb]);delta=2*math.pi/nd
    weight=2*math.pi*f*float(m.df[fb])*delta
    lon0,lat0,lon1,lat1=map(math.radians,(170.,45.,198.,23.))
    start=cp.asarray([math.cos(lat0)*math.cos(lon0),math.cos(lat0)*math.sin(lon0),math.sin(lat0)],cp.float64)
    end=cp.asarray([math.cos(lat1)*math.cos(lon1),math.cos(lat1)*math.sin(lon1),math.sin(lat1)],cp.float64)
    arc=math.acos(float(cp.sum(start*end)));distance=R*arc;tangent=(end-start*math.cos(arc))/math.sin(arc)
    east0=cp.asarray([-math.sin(lon0),math.cos(lon0),0.]);north0=cp.asarray([-math.sin(lat0)*math.cos(lon0),-math.sin(lat0)*math.sin(lon0),math.cos(lat0)])
    theta0=math.atan2(float(cp.sum(tangent*north0)),float(cp.sum(tangent*east0)))
    lon=b.lon.astype(cp.float64)*math.pi/180;lat=b.lat.astype(cp.float64)*math.pi/180
    pos=cp.stack((cp.cos(lat)*cp.cos(lon),cp.cos(lat)*cp.sin(lon),cp.sin(lat)),axis=-1)
    east=cp.stack((-cp.sin(lon),cp.cos(lon),cp.zeros_like(lon)),axis=-1)
    north=cp.stack((-cp.sin(lat)*cp.cos(lon),-cp.sin(lat)*cp.sin(lon),cp.cos(lat)),axis=-1)
    theta=cp.arange(nd,dtype=cp.float64)*delta
    directions=east[:,None,:]*cp.cos(theta)[None,:,None]+north[:,None,:]*cp.sin(theta)[None,:,None]
    sigma_space=200000.;sigma_angle=math.radians(10.)
    # Bin-averaged exact solution. A narrow arriving spectrum becomes much
    # narrower than a direction bin; point samples alias its peak severely.
    nodes,weights=np.polynomial.legendre.leggauss(16)
    def exact(t):
        a=speed*t/R;answer=cp.zeros((m.nc,nd),cp.float64)
        for node,quadrature_weight in zip(nodes,weights):
            angles=theta+float(node)*delta/2
            directions=east[:,None,:]*cp.cos(angles)[None,:,None]+north[:,None,:]*cp.sin(angles)[None,:,None]
            r=pos[:,None,:]*math.cos(a)-directions*math.sin(a)
            v=pos[:,None,:]*math.sin(a)+directions*math.cos(a)
            l=cp.arctan2(r[:,:,1],r[:,:,0]);p=cp.arcsin(cp.clip(r[:,:,2],-1,1))
            ve=-v[:,:,0]*cp.sin(l)+v[:,:,1]*cp.cos(l)
            vn=-v[:,:,0]*cp.sin(p)*cp.cos(l)-v[:,:,1]*cp.sin(p)*cp.sin(l)+v[:,:,2]*cp.cos(p)
            angle=cp.arctan2(vn,ve)-theta0
            radius=R*cp.arccos(cp.clip(cp.sum(r*start,axis=2),-1,1))
            answer+=float(quadrature_weight)/2*cp.exp(-.5*(radius/sigma_space)**2+(cp.cos(angle)-1)/sigma_angle**2)
        return answer
    initial=exact(0);norm=float(cp.max(cp.sum(initial,axis=1)));initial/=norm
    m.state.fill(0);m.state[:,fb,:]=(initial/weight).astype(cp.float32)
    total=distance/speed;steps=math.ceil(total/600);dt=total/steps
    group=next(g for g in m.transport_groups if g[0]<=fb<g[0]+g[1]);first,count,bound=group
    pieces=math.ceil(dt/bound);h=dt/pieces;threads=1<<(nd-1).bit_length()
    def advance():
        for _ in range(pieces):
            advanced=b.advect_group(m.state,m.cg,h,cp.cuda.get_current_stream(),None,first,count)
            cp.copyto(m.state[:,first:first+count],advanced[:,first:first+count])
            m.kernels['wave_refract_geometry']((m.nc*count,),(threads,),
                (m.state,b.depth,b.lat,b.depth_gradient_east,b.depth_gradient_north,m.frequency,m.k,m.cg,np.int32(m.nc),np.int32(nf),np.int32(nd),np.float32(h),np.int32(first),np.int32(count)),shared_mem=5*threads*4)
    # Allocate backend scratch before capture without copying the probe to state.
    b.advect_group(m.state,m.cg,h,cp.cuda.get_current_stream(),None,first,count)
    cp.cuda.get_current_stream().synchronize();stream=cp.cuda.Stream(non_blocking=True)
    with stream:stream.begin_capture();advance();graph=stream.end_capture()
    area=b.area.astype(cp.float64);mass0=float(cp.sum(initial*area[:,None]));records=[];profiles=[]
    def sample(step):
        t=step*dt;ref=exact(t)/norm;v=m.state[:,fb,:]*weight
        bulk=cp.sum(v,axis=1);rb=cp.sum(ref,axis=1)
        peak=int(cp.argmax(bulk));refpeak=int(cp.argmax(rb))
        # Local circular angular width at the respective peak cells.
        def angular(s):
            z=cp.sum(s*cp.exp(1j*theta))/cp.sum(s)
            return float(cp.angle(z)*180/math.pi),float(cp.sqrt(cp.maximum(0.,-2*cp.log(cp.abs(z))))*180/math.pi)
        md,ms=angular(v[peak]);rd,rs=angular(ref[refpeak])
        centroid=cp.sum(pos*bulk[:,None]*area[:,None],axis=0);centroid/=cp.linalg.norm(centroid)
        rc=cp.sum(pos*rb[:,None]*area[:,None],axis=0);rc/=cp.linalg.norm(rc)
        record={'distance_km':speed*t/1000,'hour':t/3600,'mass_relative_error':float(cp.sum(bulk*area)/mass0-1),
            'reference_mass_relative_error':float(cp.sum(rb*area)/mass0-1),
            'peak_hs_retention':float(cp.sqrt(cp.max(bulk))), 'analytic_peak_hs_retention':float(cp.sqrt(cp.max(rb))),
            'peak_hs_ratio_to_analytic':float(cp.sqrt(cp.max(bulk)/cp.max(rb))),
            'angular_spread_degrees':ms,'analytic_spread_degrees':rs,'angular_mean_error_degrees':(md-rd+180)%360-180,
            'centroid_error_m':float(R*cp.arccos(cp.clip(cp.sum(centroid*rc),-1,1))),
            'minimum_variance':float(cp.min(v)),'relative_l2_bulk':float(cp.sqrt(cp.sum((bulk-rb)**2*area)/cp.sum(rb*rb*area)))}
        records.append(record)
        profiles.append({'distance_km':record['distance_km'],'direction_degrees':cp.asnumpy(theta*180/math.pi).tolist(),
            'peak_directional_variance':cp.asnumpy(v[peak]).tolist(),'analytic_peak_directional_variance':cp.asnumpy(ref[refpeak]).tolist()})
    sample(0)
    for step in range(1,steps+1):
        graph.launch(stream=cp.cuda.get_current_stream())
        if step in (steps//3,2*steps//3,steps):sample(step)
    last=records[-1]
    acceptance={'mass_error':.002,'peak_hs_ratio_to_analytic_min':.95,'peak_hs_ratio_to_analytic_max':1.05,
        'added_angular_spread_degrees_max':3.,'angular_mean_error_degrees_max':10.,'centroid_error_m_max':50000.}
    passed=(abs(last['mass_relative_error'])<.002 and .95<last['peak_hs_ratio_to_analytic']<1.05
        and last['angular_spread_degrees']-last['analytic_spread_degrees']<3 and abs(last['angular_mean_error_degrees'])<10
        and last['centroid_error_m']<50000 and min(r['minimum_variance'] for r in records)>=-1e-7)
    receipt={'status':'PASS' if passed else 'FAIL','source_terms':'none; production geographic and angular operators only',
        'start_lon_lat':[170.,45.],'end_lon_lat':[-162.,23.],'distance_m':distance,'grid':[ny,nx],'spacing_degrees':.5,
        'period_seconds':1/f,'frequency_bin':fb,'direction_sigma_degrees':10,'spatial_sigma_m':sigma_space,
        'dt_seconds':dt,'steps':steps,'pieces_per_call':pieces,'group_speed_ms':speed,
        'reference_direction_quadrature':16,'reference':'Direction-bin-averaged exact backward great-circle characteristics in spherical phase space; no physical sources or GSE diffusion in reference. Production GSE stays enabled.',
        'acceptance':acceptance,'records':records,'profiles':profiles,'wall_seconds':time.perf_counter()-began}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def charts(output,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,4),constrained_layout=True);rows=r['records'];x=[v['distance_km'] for v in rows]
    for key,label in [('peak_hs_retention','Model'),('analytic_peak_hs_retention','Exact rays')]:ax[0].plot(x,[v[key] for v in rows],'o-',label=label)
    for key,label in [('angular_spread_degrees','Model'),('analytic_spread_degrees','Exact rays')]:ax[1].plot(x,[v[key] for v in rows],'o-',label=label)
    p=r['profiles'][-1];ax[2].plot(p['direction_degrees'],p['peak_directional_variance'],label='Model peak');ax[2].plot(p['direction_degrees'],p['analytic_peak_directional_variance'],'--',label='Exact peak')
    ax[0].set(xlabel='Distance (km)',ylabel='Peak Hs / initial peak Hs');ax[1].set(xlabel='Distance (km)',ylabel='Local directional spread (degrees)');ax[2].set(xlabel='Travel direction CCW from east (degrees)',ylabel='Variance by direction (m2)')
    for a in ax:a.grid(alpha=.25);a.legend(fontsize=8)
    fig.suptitle(f"Spherical source-free {r['period_seconds']:.2f} s packet: 45N 170E to 23N 162W; {r['status']}")
    fig.savefig(Path(output)/'spherical-packet.png',dpi=140);plt.close(fig)
