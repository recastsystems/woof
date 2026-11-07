"""Public stock-namelist runner for native dry spherical hydrostatic grids.

CUDA owns initialization and integration, the shared Rust bridges own NetCDF
IO, and the ordinary restart implementation owns durable continuation.
"""
from dataclasses import replace
from datetime import datetime,timedelta
import hashlib,json,time
from pathlib import Path
import numpy as np


def _native_fields(path):
    from woof.netcdf_bridge import Dataset
    from woof.core.global_ideal_init import PROFILE_NAMES
    aliases=dict(t_2='T',mu_2='MU',ph_2='PH',xlat='XLAT',xlong='XLONG',
        msftx='MAPFAC_MX',msfty='MAPFAC_MY',msfux='MAPFAC_UX',msfuy='MAPFAC_UY',
        msfvx='MAPFAC_VX',msfvy='MAPFAC_VY',msfvx_inv='MF_VX_INV')
    names=(*PROFILE_NAMES,'u','v','w','pb','alb','phb','al','p','mub','f','e',*aliases)
    with Dataset(path)as ds:
        if int(ds.MAP_PROJ)!=0:raise ValueError('polar hydrostatic input requires native MAP_PROJ=0 directional geometry')
        fields={name:np.ascontiguousarray(ds.variables[aliases.get(name,name.upper())][0])for name in names}
        scalar=lambda name:float(np.asarray(ds.variables[name][0]).reshape(-1)[0])
        fields['scalars']=np.asarray((scalar('P_TOP'),float(ds.DX),float(ds.DY),
            *(scalar(name)for name in ('CF1','CF2','CF3','CFN','CFN1'))),np.float32)
    return fields


def prepare(directory):
    """Import the stock namelist and construct one ordinary domain state."""
    import cupy as cp
    from woof.namelist_import import import_global_ideal_namelist
    from woof.core.state import DomainState
    from woof.core.global_state import bind
    from woof.forecast_gpu import require_forecast_gpu
    cfg,receipt=import_global_ideal_namelist(Path(directory)/'namelist.input')
    require_forecast_gpu()
    free,_=cp.cuda.runtime.memGetInfo()
    estimate=cfg.nx*cfg.ny*cfg.nz*1024+64*1024**2
    if estimate>free:raise MemoryError(f'global resident state needs approximately {estimate} bytes; device has {free} free')
    source=Path(directory)/'wrfinput_d01'
    if source.is_file():
        raw=_native_fields(source)
        receipt['input']=dict(path=str(source.resolve()),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    else:
        from woof.core.global_ideal_init import initialize
        random=cp.random.random((cfg.nz,cfg.ny,cfg.nx),dtype=cp.float32)
        raw=initialize(cfg.nx,cfg.ny,cfg.nz,ztop=cfg.ztop,hybrid_opt=cfg.hybrid_opt,
                       etac=cfg.etac,random_samples=random)
        receipt['rng']='GPU float32 uniform samples; native Fortran random_seed sequence is unspecified'
    scalars=cp.asnumpy(cp.asarray(raw['scalars']))
    cfg=replace(cfg,dx=float(scalars[1]),dy=float(scalars[2]))
    state=bind(DomainState(cfg),cfg,raw)
    receipt['spacing_m']=[cfg.dx,cfg.dy]
    return state,cfg,receipt


def time_string(value):
    return f'{value.year:04d}-{value.month:02d}-{value.day:02d}_{value.hour:02d}:{value.minute:02d}:{value.second:02d}'


def history_fields(state,cfg):
    """Publish native carriers; transforms stay on CUDA before the IO copy."""
    import cupy as cp
    fields=dict(U=state.u,V=state.v,W=state.w,T=state.thp,PH=state.php,MU=state.mup,
        P=state.p_perturbation,PB=state.pb,PHB=state.phb,MUB=state.mub2d,HGT=state.ht,
        XLAT=state.global_lat,XLONG=state.global_lon,F=state.f,E=state.e,
        COSALPHA=state.cosa,SINALPHA=state.sina,
        MAPFAC_MX=state.global_tx,MAPFAC_MY=state.msft,MAPFAC_UX=state.global_ux,
        MAPFAC_UY=state.msfu,MAPFAC_VX=state.msfv,MAPFAC_VY=state.global_vy,
        MF_VX_INV=state.msfv_inv,MAPFAC_M=state.msft,MAPFAC_U=state.msfu,MAPFAC_V=state.msfv)
    for name in ('znu','znw','dnw','rdnw','dn','rdn','fnm','fnp','c1h','c2h','c3h','c4h','c1f','c2f','c3f','c4f'):
        fields[name.upper()]=getattr(state,name)
    # Dry atmosphere, rather than a fabricated moisture diagnostic.
    fields['QVAPOR']=state.scratch(state.thp.shape,'global_rk_history_qv')
    fields['QVAPOR'].fill(0)
    fields['GLW']=state.scratch((cfg.ny,cfg.nx),'global_rk_history_glw')
    fields['GLW'].fill(0)
    fields['RTHRATEN']=state.global_rthraten
    for name,value in fields.items():
        if not bool(cp.isfinite(value).all().get()):
            raise FloatingPointError(f'global forecast produced nonfinite {name}')
    return {name:cp.asnumpy(value)for name,value in fields.items()}


def run(directory,outdir,*,restart=None,render_products=None):
    import cupy as cp
    from woof.core import dycore
    from woof.core.global_state import build_stepper
    from woof.io.restart import write_restart,restore_restart
    from woof.io.wrfout import WrfoutWriter
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    if list(outdir.glob('wrfout*')) and restart is None:
        raise ValueError(f'{outdir} already contains forecast histories; choose a new output directory')
    state,cfg,receipt=prepare(directory)
    start=datetime.fromisoformat(receipt['start'])
    if restart is None and receipt['document'].get('time_control',{}).get('restart',[False])[0]:
        raise ValueError('restart=true requires --restart pointing at the shared durable checkpoint; cold initialization would discard the requested continuation')
    if restart is not None:restore_restart(restart,state,cfg)
    build_stepper(state,cfg)
    attrs=dict(MAP_PROJ=0,MAP_PROJ_CHAR='Latitude-Longitude',START_DATE=time_string(start),
        SIMULATION_START_DATE=time_string(start),DT=cfg.dt,RA_LW_PHYSICS=cfg.ra_lw_physics,
        RA_SW_PHYSICS=0,MP_PHYSICS=0,BL_PBL_PHYSICS=0,CU_PHYSICS=0,SF_SURFACE_PHYSICS=0,
        SF_SFCLAY_PHYSICS=0,HYBRID_OPT=cfg.hybrid_opt,TRUELAT1=0.,TRUELAT2=0.,STAND_LON=0.,
        CEN_LAT=0.,CEN_LON=0.,POLE_LAT=90.,POLE_LON=0.,TITLE='WOOF native hydrostatic global ideal forecast')
    frames=[];checkpoints=[];writer=None;in_file=0
    frames_per_file=receipt['frames_per_outfile']
    def output():
        nonlocal writer,in_file
        stamp=time_string(start+timedelta(seconds=state.elapsed_seconds))
        fields=history_fields(state,cfg)
        if writer is None or in_file==frames_per_file:
            if writer is not None:writer.close()
            path=outdir/('wrfout_d01_'+stamp)
            if path.exists():raise FileExistsError(path)
            writer=WrfoutWriter(path,nx=cfg.nx,ny=cfg.ny,nz=cfg.nz,dx=cfg.dx,dy=cfg.dy,
                title=attrs['TITLE'],global_attrs=attrs,field_schema=fields)
            frames.append(str(path));in_file=0
        writer.write_frame(stamp,fields);in_file+=1
    begin=cp.cuda.Event();end=cp.cuda.Event();wall=time.monotonic()
    steps=int(round((cfg.run_seconds-state.elapsed_seconds)/cfg.dt))
    if steps<0:raise ValueError('restart is later than the requested forecast end')
    try:
        output();begin.record()
        for _ in range(steps):
            dycore.step(state,cfg)
            if abs(state.elapsed_seconds/cfg.output_interval_s-round(state.elapsed_seconds/cfg.output_interval_s))<1e-8:output()
            if cfg.restart_interval_s and abs(state.elapsed_seconds/cfg.restart_interval_s-round(state.elapsed_seconds/cfg.restart_interval_s))<1e-8:
                stamp=time_string(start+timedelta(seconds=state.elapsed_seconds))
                path=outdir/('restart_d01_'+stamp+'.npz')
                write_restart(path,state,cfg);checkpoints.append(str(path))
    finally:
        if writer is not None:writer.close()
    end.record();end.synchronize()
    result=dict(schema='gpuwm-global-forecast-v1',import_receipt=receipt,
        frames=frames,checkpoints=checkpoints,steps=steps,elapsed_seconds=state.elapsed_seconds,
        wall_seconds=time.monotonic()-wall,device_envelope_ms=cp.cuda.get_elapsed_time(begin,end),
        numerics=__import__('os').environ.get('GPUWM_NUMERICS','fast'),
        integration='resident CUDA graph; no per-step host readbacks',
        history='shared Rust WrfoutWriter',restart='shared woof.io.restart')
    (outdir/'run_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if render_products and render_products!='none':
        from woof.rustwx import find_renderer,run_renderer
        renderer=find_renderer()
        for frame in frames:
            okay,errors,*_=run_renderer(renderer,Path(frame),store_root=outdir/'render-store',
                out_dir=outdir/'maps',products=render_products,frames='all',source_label=attrs['TITLE'])
            if not okay or errors:raise RuntimeError(f'global Rust rendering failed: {errors}')
    print(f'go: global forecast completed {steps} steps, {len(frames)} Rust history files')
    return 0
