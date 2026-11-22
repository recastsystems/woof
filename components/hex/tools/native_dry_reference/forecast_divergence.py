#!/usr/bin/env python3
"""Measure five-day dry hex/native and native/native divergence on one mesh.

The analysis uses the same wind reconstruction and log-pressure interpolation
for every arm. RMS height is in metres; RMS vector wind is sqrt(mean(du^2+dv^2))
in m/s, weighted by cell area. A complete five-day deck is required.
"""
from __future__ import annotations
import os as _hex_os
_hex_os.environ["WOOF_HEX_NUMERICS"] = "strict"  # a bitwise proof against native MPAS runs strict numerics (fast is the default, strict the opt-in)
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import platform
import subprocess
import numpy as np
from netCDF4 import Dataset


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def read_level_cell(d,name):
    v=d[name]; data=np.asarray(v[:])
    dims=list(v.dimensions)
    if 'Time' in dims:
        axis=dims.index('Time');data=np.take(data,0,axis=axis);dims.pop(axis)
    entity='nEdges' if 'nEdges' in dims else 'nCells'
    return np.asarray(data.transpose(dims.index('nVertLevels'),dims.index(entity)),np.float64)


def reconstruct(normal, mesh):
    normal=np.asarray(normal,np.float64)
    edges=np.asarray(mesh['edgesOnCell'],np.int64)
    # Mesh loaders convert connectivity to zero-based indices.
    coeff=np.asarray(mesh['coeffs_reconstruct'],np.float64)
    counts=np.asarray(mesh['nEdgesOnCell'],np.int64)
    if edges.shape[0]!=counts.size: edges=edges.T
    if coeff.shape[0]!=counts.size: coeff=coeff.transpose(2,1,0)
    elif coeff.shape[1]==3 and coeff.shape[2]==edges.shape[1]: coeff=coeff.transpose(0,2,1)
    if not np.isfinite(coeff).all() or np.any(np.all(coeff==0,axis=(1,2))):
        raise ValueError('missing/zero wind reconstruction coefficients')
    nlev=normal.shape[0];vector=np.zeros((nlev,len(counts),3),np.float64)
    for slot in range(edges.shape[1]):
        active=slot<counts
        vector[:,active,:]+=normal[:,edges[active,slot],None]*coeff[active,slot,:]
    lat=np.asarray(mesh['latCell'],np.float64);lon=np.asarray(mesh['lonCell'],np.float64)
    east=-vector[:,:,0]*np.sin(lon)+vector[:,:,1]*np.cos(lon)
    north=-(vector[:,:,0]*np.cos(lon)+vector[:,:,1]*np.sin(lon))*np.sin(lat)+vector[:,:,2]*np.cos(lat)
    return east,north


def at_pressure(pressure, values, target, *, allow_outside=False):
    """Interpolate only an unambiguous downward crossing, never extrapolate.

The native real initial state can have inversions away from the requested
surface. Such an inversion does not invalidate a unique 500/250 hPa crossing.
The strict unit-test path requires a monotone column; the analysis masks
outside/ambiguous crossings and uses a common mask for all three arms.
"""
    p=np.asarray(pressure,np.float64);v=np.asarray(values,np.float64)
    if p.shape!=v.shape or not np.isfinite(p).all() or not np.isfinite(v).all() or np.any(p<=0):
        raise ValueError('invalid pressure/field column')
    if p.shape[0]<2: raise ValueError('at least two vertical levels are required')
    if not allow_outside and not np.all(np.diff(p,axis=0)<0):
        raise ValueError('non-monotonic pressure column')
    crossing=(p[:-1]>=target)&(p[1:]<target)
    upward=(p[:-1]<target)&(p[1:]>=target)
    valid=(np.sum(crossing,axis=0)==1)&(np.sum(upward,axis=0)==0)
    exact_top=(p[-1]==target)&(np.sum(upward,axis=0)==0)
    valid|=exact_top
    outside=~valid
    if outside.any() and not allow_outside: raise ValueError('requested pressure is outside or ambiguous in a column')
    lower=np.argmax(crossing,axis=0)
    lower=np.clip(lower,0,p.shape[0]-2);upper=lower+1
    column=np.arange(p.shape[1]);p1=p[lower,column];p2=p[upper,column]
    weight=(np.log(target)-np.log(p1))/(np.log(p2)-np.log(p1))
    result=v[lower,column]*(1-weight)+v[upper,column]*weight
    result[exact_top]=v[-1,exact_top]
    result[outside]=np.nan
    return result


def metrics(pressure, normal, height, mesh):
    east,north=reconstruct(normal,mesh)
    return {'z500':at_pressure(pressure,height,50000,allow_outside=True),
            'u250':at_pressure(pressure,east,25000,allow_outside=True),
            'v250':at_pressure(pressure,north,25000,allow_outside=True)}


def rms_pair(a,b,area):
    z=np.isfinite(a['z500'])&np.isfinite(b['z500'])
    wind=np.isfinite(a['u250'])&np.isfinite(b['u250'])&np.isfinite(a['v250'])&np.isfinite(b['v250'])
    if not z.any() or not wind.any(): raise ValueError('no common above-ground cells')
    return {'z500_m':float(np.sqrt(np.average((a['z500'][z]-b['z500'][z])**2,weights=area[z]))),
            'wind250_ms':float(np.sqrt(np.average((a['u250'][wind]-b['u250'][wind])**2+(a['v250'][wind]-b['v250'][wind])**2,weights=area[wind])))}


def native_metrics(path,height,mesh):
    with Dataset(path) as d:
        d.set_auto_mask(False)
        p=read_level_cell(d,'pressure_base')+read_level_cell(d,'pressure_p')
        normal=read_level_cell(d,'u')
    return metrics(p,normal,height,mesh)


def time_of(path):
    return datetime.strptime(path.name.removeprefix('tracer.').removesuffix('.nc'),'%Y-%m-%d_%H.%M.%S')


def summarize(rows):
    # A conservative envelope criterion supports a narrower statement than
    # an unqualified claim about all meshes, physics or initial states.
    result={}
    for metric in ('z500_m','wind250_ms'):
        result[metric]={}
        baseline=np.array([r['native_strict_vs_optimized'][metric] for r in rows])
        days=np.array([r['hours']/24 for r in rows])
        for pair in ('hex_vs_strict','hex_vs_optimized'):
            values=np.array([r[pair][metric] for r in rows])
            positive=(days>=1)&(baseline>0)&(values>0)
            growth_hex=growth_native=None
            native_positive=(days>=1)&(baseline>0)
            if native_positive.sum()>=3:
                growth_native=float(np.polyfit(days[native_positive],np.log(baseline[native_positive]),1)[0])
            if positive.sum()>=3:
                growth_hex=float(np.polyfit(days[positive],np.log(values[positive]),1)[0])
            identically_zero=bool(np.all(values==0))
            envelope=bool(np.all(values<=baseline))
            result[metric][pair]={'within_native_envelope_at_every_sample':envelope,
              'log_rms_growth_per_day':growth_hex,'native_log_rms_growth_per_day':growth_native,
              'identically_zero_at_every_sample':identically_zero,
              # log(0) has no fit; identically zero error cannot grow.
              'no_faster_log_growth':True if identically_zero else None if growth_hex is None else growth_hex<=growth_native,
              'day5_rms':float(values[-1]),'day5_native_rms':float(baseline[-1])}
    return result


def chart(rows,path,*,cells=None,case_label='Common real initial state',mixing='off'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True,layout='constrained')
    days=[r['hours']/24 for r in rows]
    for ax,metric,label in zip(axes,('z500_m','wind250_ms'),('500 hPa height RMS difference (m)','250 hPa vector wind RMS difference (m/s)')):
        for pair,title,color in [('native_strict_vs_optimized','Native strict vs optimized','#222222'),('hex_vs_strict','Hex vs native strict','#0072b2'),('hex_vs_optimized','Hex vs native optimized','#d55e00')]:
            ax.plot(days,[r[pair][metric] for r in rows],label=title,color=color,marker='o',markersize=3)
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=9)
    axes[-1].set_xlabel('Forecast days')
    fig.suptitle(f'MPAS-A v8.4.2 dry comparison, {cells} cells\n{case_label}; physics off, mixing {mixing}; area-weighted RMS')
    fig.savefig(path,dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('init','strict','optimized','out','cache-root'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--dt',type=float,required=True)
    p.add_argument('--order',type=int,choices=(2,3),default=2)
    p.add_argument('--substeps',type=int,default=4)
    p.add_argument('--case-label',default='Common real initial state')
    p.add_argument('--mixing',choices=('off','smag'),default='off')
    p.add_argument('--smdiv',type=float,default=0.0)
    p.add_argument('--xnutr',type=float,default=0.0)
    p.add_argument('--reference-profiles',type=Path,
                   help='native u_init/v_init dump if the initial file omits those fields')
    a=p.parse_args()
    tree=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(tree/'src'))
    from woof.hex.driver import load_mpas_vertical_grid,load_mpas_initial_state
    from woof.hex.dynamics_v841 import load_v841_reference_wind_profiles
    from woof.hex.config_v841 import V841DryDycoreConfig
    from woof.hex.mesh import Mesh
    from woof.hex.vector import initialize_reconstruction_coefficients
    from woof.hex.rust_geometry import geometry_engine
    from woof.hex.cuda_backend import require_cuda
    from woof.hex.cuda_backend.runtime import KernelCache
    from woof.hex.cuda_driver import CudaDryDycoreDriver
    import cupy as cp
    strict=sorted(a.strict.glob('tracer.*.nc'));optimized=sorted(a.optimized.glob('tracer.*.nc'))
    if len(strict)!=21 or [x.name for x in strict]!=[x.name for x in optimized]:
        raise ValueError('requires matching 21 outputs at six-hour intervals through five days')
    times=np.array([(time_of(x)-time_of(strict[0])).total_seconds() for x in strict])
    if not np.array_equal(times,np.arange(21)*21600): raise ValueError('missing or wrong forecast times')
    if not (432000/a.dt).is_integer(): raise ValueError('dt must close five days')
    # Refuse deck/config mismatches before executing an expensive comparison.
    import re
    expected={'config_dt':str(a.dt),'config_time_integration_order':str(a.order),
              'config_number_of_sub_steps':str(a.substeps),'config_dynamics_split_steps':'3',
              'config_smdiv':str(a.smdiv),'config_xnutr':str(a.xnutr),'config_physics_suite':'none',
              'config_visc4_2dsmag':'0.05','config_smagorinsky_coef':'0.125','config_del4u_div_factor':'10.0'}
    init_hash=digest(a.init)
    for root in (a.strict,a.optimized):
        text=(root/'namelist.atmosphere').read_text()
        for key,value in expected.items():
            match=re.search(r'\b'+key+r'\s*=\s*([^\s,\n/]+)',text)
            if not match: raise ValueError('missing native option '+key)
            raw=match.group(1).strip('"\'')
            if value=='none': same=raw==value
            else: same=float(raw)==float(value)
            if not same: raise ValueError('native option differs: '+key)
        native_mixing='2d_smagorinsky' if a.mixing=='smag' else '2d_fixed'
        if f'config_horiz_mixing = "{native_mixing}"' not in text:
            raise ValueError('native mixing differs from hex')
        if digest(root/(a.init.name if (root/a.init.name).exists() else 'x1.40962.init.nc'))!=init_hash:
            raise ValueError('native initial file differs')
        if 'Finished running the atmosphere core' not in (root/'log.atmosphere.0000.out').read_text():
            raise ValueError('native run did not complete')
    a.out.mkdir(parents=True,exist_ok=True)
    mesh=Mesh.from_netcdf(a.init)
    # MPAS initializes these at runtime. A static init may carry only zeros.
    # Rebuild the common RBF diagnostic from geometry for every arm.
    coefficients=initialize_reconstruction_coefficients(mesh,include_halos=True)
    mesh.arrays['coeffs_reconstruct']=coefficients
    reconstruction_sha256=hashlib.sha256(np.ascontiguousarray(coefficients).tobytes()).hexdigest()
    grid=load_mpas_vertical_grid(a.init,mesh,config_coef_3rd_order=.25)
    profile_path=a.reference_profiles or a.init
    profiles=load_v841_reference_wind_profiles(profile_path,n_vert_levels=grid.vertical_grid.n_vert_levels)
    state,reference,saved=load_mpas_initial_state(strict[0],mesh,grid.vertical_grid,scalar_names=('qv',),
       reference_path=strict[0],terrain_metrics=grid.terrain_metrics,return_saved_diagnostics=True)
    for name in ('defc_a','defc_b'):
        mesh.arrays[name]=np.zeros((len(mesh.arrays['nEdgesOnCell']),mesh.arrays['edgesOnCell'].shape[1]),np.float32)
    config=V841DryDycoreConfig(config_dt=a.dt,config_time_integration_order=a.order,
        config_number_of_sub_steps=a.substeps,config_dynamics_split_steps=3,config_split_dynamics_transport=True,
        config_monotonic=True,config_positive_definite=False,config_scalar_adv_order=3,config_scalar_vadv_order=3,
        config_coef_3rd_order=.25,config_apvm_upwinding=.5,config_smdiv=a.smdiv,config_divergence_damping=a.smdiv>0,
        config_xnutr=a.xnutr,config_zd=22000.,config_visc4_2dsmag=.05 if a.mixing=='smag' else 0.,
        config_smagorinsky_coef=.125,config_del4u_div_factor=10.,config_h_ScaleWithMesh=True,
        config_horiz_mixing='2d_smagorinsky' if a.mixing=='smag' else 'off')
    config.validate()
    driver=CudaDryDycoreDriver.from_host(mesh,state,grid.vertical_grid,reference,config,
      saved_diagnostics=saved,terrain_metrics=grid.terrain_metrics,reference_wind_profiles=profiles,
      kernel_cache=KernelCache(capability=require_cuda(min_compute=(7,5)),cache_dir=str(a.cache_root)))
    with Dataset(a.init) as d:
        z=np.asarray(d['zgrid'][:],np.float64)
    height=((z[:,1:]+z[:,:-1])/2).T
    area=np.asarray(mesh.arrays['areaCell'],np.float64)
    rows=[];started=time.perf_counter()
    for index,seconds in enumerate(times):
        if index:
            for _ in range(round(21600/a.dt)):
                driver.atmosphere=driver.step_device().atmosphere
        pressure=np.asarray(reference.pressure_base,np.float64)+cp.asnumpy(driver.atmosphere.saved.pressure_perturbation).astype(np.float64)
        normal=cp.asnumpy(driver.atmosphere.saved.normal_velocity)
        actual=metrics(pressure,normal,height,mesh)
        s=native_metrics(strict[index],height,mesh);o=native_metrics(optimized[index],height,mesh)
        # Use exactly the same above-ground sample for all three pairs.
        common_z=np.isfinite(actual['z500'])&np.isfinite(s['z500'])&np.isfinite(o['z500'])
        common_wind=np.isfinite(actual['u250'])&np.isfinite(s['u250'])&np.isfinite(o['u250'])
        for arm in (actual,s,o):
            arm['z500'][~common_z]=np.nan
            for name in ('u250','v250'): arm[name][~common_wind]=np.nan
        row={'hours':float(seconds/3600),'native_strict_vs_optimized':rms_pair(s,o,area),
             'hex_vs_strict':rms_pair(actual,s,area),'hex_vs_optimized':rms_pair(actual,o,area),
             'z500_cells':int(common_z.sum()),'wind250_cells':int(common_wind.sum()),
             'strict_output_sha256':digest(strict[index]),'optimized_output_sha256':digest(optimized[index])}
        rows.append(row)
        (a.out/'progress.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    receipt={'schema':'gpuwm-hex.forecast-divergence/v1','init_sha256':init_hash,
      'reference_profiles_sha256':digest(profile_path),
      'wind_reconstruction_sha256':reconstruction_sha256,
      'geometry_engine_sha256':geometry_engine()[1],
      'hex_commit':subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip(),
      'analysis_sha256':digest(Path(__file__)),
      'host':platform.node(),'python':platform.python_version(),'numpy':np.__version__,'cupy':cp.__version__,
      'card':cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode(),
      'cuda_runtime':cp.cuda.runtime.runtimeGetVersion(),
      'native_decks':{str(root):{'namelist_sha256':digest(root/'namelist.atmosphere'),
                                'streams_sha256':digest(root/'streams.atmosphere')} for root in (a.strict,a.optimized)},
      'dt_seconds':a.dt,'order':a.order,'acoustic_substeps':a.substeps,'cells':len(area),
      'levels':height.shape[0],'physics':'none','mixing':a.mixing,'smdiv':a.smdiv,'xnutr':a.xnutr,'area_weighted':True,
      'analysis':'same double-precision reconstruction and log-pressure interpolation at a unique downward crossing for all arms; no extrapolation; same common above-ground/unambiguous mask for all pairs at each time',
      'hex_start':'native strict time-zero internal state from the same initial file',
      'hex_wall_seconds':time.perf_counter()-started,'rows':rows,'summary':summarize(rows)}
    (a.out/'forecast-divergence.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    with (a.out/'forecast-divergence.csv').open('w',newline='') as f:
        writer=csv.writer(f);pairs=('native_strict_vs_optimized','hex_vs_strict','hex_vs_optimized')
        writer.writerow(['hours',*[p+'_'+m for m in ('z500_m','wind250_ms') for p in pairs]])
        for r in rows: writer.writerow([r['hours'],*[r[p][m] for m in ('z500_m','wind250_ms') for p in pairs]])
    chart(rows,a.out/'forecast-divergence.png',cells=len(area),case_label=a.case_label,mixing=a.mixing)
    print(json.dumps(receipt['summary'],indent=2))


if __name__=='__main__': main()
