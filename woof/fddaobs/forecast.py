"""Setup-only Rust observation packing and captured CUDA FDDA stage.

The dycore launches the graph into its held, undivided forward-tendency
slots. No observation processing, time decision, or device readback runs
in Python between model steps. Checkpoints carry the native mutable
report/innovation/weight inventory and the device clock.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import numpy as np


def _record(values, types):
    return np.array(tuple(values), dtype=[('v'+str(i), t) for i,t in enumerate(types)])[()]


def attach(state, cfg, driver, *, start_time, latitude, longitude):
    if not cfg.obs_nudge_opt:
        return None
    from .config import validate
    validate(cfg)
    import cupy as cp
    from woof import bridges
    from woof.certify.kernel_manifest import record_module
    if start_time is None or latitude is None or longitude is None:
        raise ValueError('FDDAOBS_GEOGRAPHY: bind the original UTC start time and mass-grid latitude/longitude')
    if getattr(state, '_fddaobs_graph', None) is not None:
        raise ValueError('observation FDDA is already attached to this domain')
    binary = bridges.find_artifact('GPUWM_RW_FDDAOBS', bridges.executable_name('rw_fddaobs'),
                                  crate_relative=bridges.RUSTWX_CRATE_RELATIVE)
    if binary is None:
        raise FileNotFoundError('FDDAOBS_RUST_READER: rw_fddaobs is missing; install its bridge bundle or build tools/rustwx with cargo build --release --locked -p rw-obs --bin rw_fddaobs')
    nz, ny, nx = state.p.shape
    def origin(a):
        # Setup metadata only; the complete observation data path is Rust.
        value = a.reshape(-1)[:1]
        return float(cp.asnumpy(value)[0] if isinstance(value, cp.ndarray) else value[0])
    projection = dict(map_proj=cfg.map_proj, latitude_origin=origin(latitude), longitude_origin=origin(longitude),
                      truelat1=cfg.obs_truelat1, truelat2=cfg.obs_truelat2, stand_lon=cfg.obs_stand_lon,
                      dx=cfg.dx, dy=cfg.dy, e_we=nx+1, e_sn=ny+1)
    inputfile = Path(cfg.obs_inputfile or f'OBS_DOMAIN{cfg.obs_domain_id}01').resolve()
    if not inputfile.is_file():
        raise FileNotFoundError(f'WRF observation FDDA requires {inputfile}')
    from datetime import timezone
    original = start_time.replace(tzinfo=timezone.utc) if start_time.tzinfo is None else start_time
    with tempfile.TemporaryDirectory(prefix='gpuwm-fddaobs-') as directory:
        directory = Path(directory)
        request = dict(input=str(inputfile), output=str(directory/'observations.pack'), projection=projection,
                       original_start_unix_seconds=int(original.timestamp()), rovg=float(np.float32(287.)/np.float32(9.81)),
                       window=dict(capacity=cfg.max_obs,begin=cfg.fdda_start,end=cfg.fdda_end,
                                   hours=cfg.obs_twindo,dt_minutes=cfg.dt/60.,
                                   interval=cfg.obs_ionf*cfg.obs_parent_grid_ratio**cfg.obs_levelidn,adaptive=cfg.use_adaptive_time_step))
        (directory/'request.json').write_text(json.dumps(request))
        subprocess.run([str(binary),str(directory/'request.json')],check=True,capture_output=True,text=True)
        packed=(directory/'observations.pack').read_bytes()
    if packed[:8] != b'GPWMOBS1':
        raise ValueError('observation pack ABI differs from gpuwm-fddaobs-pack-v1')
    length=int.from_bytes(packed[8:12],'little')
    metadata=json.loads(packed[12:12+length]);count=metadata['rows']
    host=np.frombuffer(packed, dtype='<f4', offset=12+length)
    if host.size!=count*14:
        raise ValueError('observation pack carrier length is inconsistent')
    # This digest binds immutable observation content, geometry and original
    # clock into the common setup fingerprint before any checkpoint restore.
    identity={k:v for k,v in metadata.items() if k != "workers"}
    identity['cuda_source_sha256']=hashlib.sha256((Path(__file__).with_name('kernels.cu').read_text()+'\n'+Path(__file__).with_name('forecast.cu').read_text()).encode()).hexdigest()
    state._fddaobs_identity=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    rows=[cp.asarray(host[n*count:(n+1)*count]) for n in range(8)]
    rio,rjo,initial_k,timeob,nlev,lev,platform,elevation=rows
    state.fddaobs_rko=initial_k
    state.fddaobs_varobs=cp.asarray(host[8*count:])
    state.fddaobs_errf=cp.zeros(count*10,dtype='f4')
    nxy=(nx+1)*(ny+1);volume=nxy*nz
    state.fddaobs_savwt=cp.zeros(volume*6,dtype='f4')
    state.fddaobs_clock=cp.array([round(state.elapsed_seconds/cfg.dt)],dtype='i8')
    state.fddaobs_frame=cp.zeros(32,dtype='u1')
    state.fddaobs_frame.view('i4')[3]=-1  # first native call refreshes the report window
    state.fddaobs_uratx=cp.zeros(nxy,dtype='f4')
    state.fddaobs_vratx=cp.zeros(nxy,dtype='f4')
    state.fddaobs_tratx=cp.zeros(nxy,dtype='f4')
    # Every bulk packing and weighting pass is a CUDA kernel. Fortran strides
    # remain the replay ABI; conversion to model C order is fused with folding.
    volume_names=('ub','vb','tb','qv','pb','pp','z','au','av','at','aq','wt','w2')
    plane_names=('up','vp','mu','muu','muv','mx','my','mux','muy','vx','vy','ht','regime','pbl')
    work={name:cp.zeros(volume if name not in ('wt','w2') or cfg.obs_scl_neg_qv_innov else 1,dtype='f4') for name in volume_names}
    work.update({name:cp.zeros(nxy,dtype='f4') for name in plane_names})
    work['kp']=cp.zeros(nxy,dtype='i4')
    vif={suffix:cp.asarray([getattr(cfg,f'obs_nudgez{part}r{r}_{suffix}') for r in (1,2,4) for part in ('full','ramp')],dtype='f4') for suffix in ('uv','t','q')}
    heights={suffix:(cp.zeros(nxy,dtype='f4'),cp.zeros(nxy,dtype='f4')) for suffix in vif}
    surface_top={suffix:cp.zeros(nxy,dtype='i4') for suffix in vif}
    g=_record([1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz,1,nx,1,ny,1,nz],['i4']*18)
    source=Path(__file__).with_name('kernels.cu').read_text()+'\n'+Path(__file__).with_name('forecast.cu').read_text()
    module=cp.RawModule(code=source,options=('--std=c++17',))
    names=('fddaobs_clock','fddaobs_pack','fddaobs_ratios','fddaobs_errob','fddaobs_nudob_rows','fddaobs_vif','fddaobs_fold')
    kernels={name:module.get_function(name) for name in names}
    stamp=module.get_function('fddaobs_stamp')
    timer_start=cp.zeros(16,dtype='u8')
    timer_sum=cp.zeros(16,dtype='u8')
    timer_count=cp.zeros(16,dtype='u8')
    error=cp.zeros(1,dtype='u4')
    published=cp.asarray([state.elapsed_seconds,cfg.dt],dtype='f4')
    publish=module.get_function('fddaobs_publish')
    labels=[]
    launches=[]
    def measured(name, kernel):
        def launch(grid, block, arguments):
            stage=len(labels)
            labels.append(name)
            launches.append((name,kernel,grid,block,arguments))
            stamp((1,),(1,),(timer_start,timer_sum,timer_count,np.int32(stage),np.int32(0)))
            kernel(grid,block,arguments)
            stamp((1,),(1,),(timer_start,timer_sum,timer_count,np.int32(stage),np.int32(1)))
        return launch
    kernels={name:measured(name,kernel) for name,kernel in kernels.items()}
    record_module('woof.fddaobs.forecast',source=source,options=('--std=c++17',),module=module)
    block=(128,);grid=((volume+127)//128,)
    from woof.core.dycore import _smag2d_specs
    # Allocate before graph capture and checkpoint restore. The driver reads
    # these held slots on every RK stage even when all mixing is switched off.
    for field,_tend,_xk,_c1,_c2,slot,_stag in _smag2d_specs(state,None,None,time_t=True):
        state.scratch(field.shape,slot)
    targets=tuple(state.scratch(shape,slot) for shape,slot in ((state.u.shape,'smag_ru'),(state.v.shape,'smag_rv'),(state.thp.shape,'smag_rth'),(state.qv.shape,'smag_rqv')))
    f=driver.fields
    interval=cfg.obs_ionf*cfg.obs_parent_grid_ratio**cfg.obs_levelidn
    controls={}
    for ivar,suffix in ((1,'uv'),(2,'uv'),(3,'t'),(4,'q')):
        no_pbl=getattr(cfg,'obs_no_pbl_nudge_'+suffix)
        coef=cfg.obs_coef_wind if ivar<3 else cfg.obs_coef_temp if ivar==3 else cfg.obs_coef_mois
        controls[ivar]=_record([cfg.obs_domain_id,0,0,0,interval,1-no_pbl,cfg.obs_sfc_scheme_horiz,cfg.obs_sfc_scheme_vert,cfg.obs_scl_neg_qv_innov,
            0.,cfg.dt/60.,cfg.obs_rinxy,cfg.dx,cfg.obs_twindo,cfg.obs_sfcfact,cfg.obs_sfcfacr,1.,2.,50.,1./cfg.obs_dpsmx,1.,coef,cfg.obs_max_sndng_gap],['i4']*9+['f4']*14)
    # WRF's analytic base-state first half level, not the terrain-following
    # model column, raises the minimum VIF depth when fullmin is positive.
    znu0=float(cp.asnumpy(state.znu[:1])[0])
    p0=np.float32(znu0)*(np.float32(100000.)-np.float32(state.p_top))+np.float32(state.p_top)
    logterm=np.log(p0/np.float32(100000.))
    invg=np.float32(1.)/np.float32(9.81)
    base=np.float32(.5)*np.float32(287.)*np.float32(50.)*invg*logterm*logterm-np.float32(287.)*np.float32(cfg.base_temp)*invg*logterm
    fullmin=max(np.float32(cfg.obs_nudgezfullmin),np.float32(1.1)*base) if cfg.obs_nudgezfullmin>0 else np.float32(0.)
    ints=tuple(np.int32(v) for v in (nx,ny,nz,state.thb.ndim==3,state.pb.ndim==3,state.phb.ndim==3,getattr(state,'p_perturbation',None) is not None,
          cfg.sf_sfclay_physics in (1,5,7,91),cfg.sf_sfclay_physics!=0,driver.stepbl))
    frame=state.fddaobs_frame
    def record():
        kernels['fddaobs_clock']((1,),(1,),(state.fddaobs_clock,frame,timeob,nlev,lev,np.int32(count),np.int32(cfg.max_obs),
            np.float32(cfg.dt),np.float32(cfg.fdda_start),np.float32(cfg.fdda_end),np.float32(cfg.obs_twindo),np.int32(cfg.obs_idynin),np.float32(cfg.obs_dtramp),np.int32(interval),published))
        kernels['fddaobs_pack'](grid,block,(state.u,state.v,state.thp,state.thb,state.qv,state.p,state.pb,
            getattr(state,'p_perturbation',None) if getattr(state,'p_perturbation',None) is not None else state.p,state.php,state.phb,
            state.ht,state.mub2d,state.mup,state.msft,state.msfu,state.msfv,f['u10'],f['v10'],f['th2'],f['regime'],f['pblh'],f['kpbl'],
            *(work[n] for n in ('ub','vb','tb','qv','pb','pp','z')),state.fddaobs_uratx,state.fddaobs_vratx,state.fddaobs_tratx,
            *(work[n] for n in ('mu','muu','muv','mx','my','mux','muy','vx','vy','ht','regime','pbl','kp','au','av','at','aq')),*ints,frame))
        kernels['fddaobs_ratios'](((nx+16)//16,(ny+16)//16),(16,16),(state.fddaobs_uratx,state.fddaobs_vratx,work['up'],work['vp'],g))
        if count:
            kernels['fddaobs_errob'](((count+127)//128,),block,(*(work[n] for n in ('ub','vb','tb','qv','pb','pp','z','up','vp')),state.fddaobs_tratx,work['kp'],
                rio,rjo,state.fddaobs_rko,state.fddaobs_varobs,state.fddaobs_errf,np.int32(count),np.int32(6),np.int32(10),np.int32(cfg.obs_nudge_wind),np.int32(cfg.obs_nudge_temp),np.int32(cfg.obs_nudge_mois),np.float32(300.),np.float32(287./1004.5),g,frame))
        for suffix in vif:
            kernels['fddaobs_vif'](((nx+15)//16,(ny+15)//16),(16,16),(vif[suffix],work['regime'],work['pbl'],*heights[suffix],np.float32(cfg.obs_nudgezmax),np.float32(fullmin),np.float32(cfg.obs_nudgezrampmin),g,error,work['z'],work['ht'],surface_top[suffix]))
        for ivar,suffix in ((1,'uv'),(2,'uv'),(3,'t'),(4,'q')):
            if not (cfg.obs_nudge_wind if ivar<3 else cfg.obs_nudge_temp if ivar==3 else cfg.obs_nudge_mois):
                continue
            mu=work['muu' if ivar==1 else 'muv' if ivar==2 else 'mu']
            mx,my=(work['mux'],work['muy']) if ivar==1 else (work['vx'],work['vy']) if ivar==2 else (work['mx'],work['my'])
            row_order = ivar == 4 and cfg.obs_scl_neg_qv_innov
            launch_grid, launch_block = ((((ny+31)//32,), (32,)) if row_order else (((nx+31)//32,ny,(nz+3)//4),(32,1,4)))
            kernels['fddaobs_nudob_rows'](launch_grid,launch_block,(work[{1:'au',2:'av',3:'at',4:'aq'}[ivar]],state.fddaobs_savwt,work['wt'],work['w2'],
                mu,state.c1h,state.c2h,mx,my,rio,rjo,state.fddaobs_rko,timeob,state.fddaobs_varobs,state.fddaobs_errf,nlev,lev,work['pb'],work['pp'],work['kp'],
                *heights[suffix],work['ht'],work['z'],work['qv'],np.int32(count),np.int32(6),np.int32(10),np.int32(ivar),controls[ivar],g,frame,surface_top[suffix]))
        kernels['fddaobs_fold'](((max(a.size for a in targets)+127)//128,),block,(*(work[n] for n in ('au','av','at','aq')),*targets,np.int32(nx),np.int32(ny),np.int32(nz)))
    # Record only. Launching a warmup would advance the native clock and
    # mutate report QC before the first actual model step.
    stream=cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream:
        stream.begin_capture();record();graph=stream.end_capture()
    state._fddaobs_graph=graph
    state._fddaobs_owner=dict(module=module,work=work,rows=rows,vif=vif,heights=heights,surface_top=surface_top,controls=controls,metadata=metadata,graph=graph,
        timer_start=timer_start,timer_sum=timer_sum,timer_count=timer_count,labels=labels,launches=launches,error=error,published=published,publish=publish)
    return state._fddaobs_owner


def receipt(state):
    """Read only at a forecast/output seam, after the existing device sync."""
    owner=getattr(state,'_fddaobs_owner',None)
    if owner is None:
        return None
    if int(owner['error'].get()[0]):
        raise RuntimeError('WRF fddaobs_driver STOP 6: surface-layer regime must be 1, 2, 3 or 4')
    sums=owner['timer_sum'].get()
    counts=owner['timer_count'].get()
    return dict(identity=state._fddaobs_identity,observations=owner['metadata'],
        completed_steps=int(state.fddaobs_clock.get()[0]),
        stages=[dict(name=name,calls=int(counts[i]),total_ns=int(sums[i]),
                     mean_ms=float(sums[i])/float(counts[i])/1.e6 if counts[i] else None)
                for i,name in enumerate(owner['labels'])])


def frame_module():
    """The exact production compilation unit for local-memory measurement."""
    import cupy as cp
    source=Path(__file__).with_name('kernels.cu').read_text()+'\n'+Path(__file__).with_name('forecast.cu').read_text()
    module=cp.RawModule(code=source,options=('--std=c++17',))
    from woof.certify.kernel_manifest import record_module
    record_module('woof.fddaobs.forecast:frame',source=source,
                  options=('--std=c++17',),module=module)
    return module,source


def launch(state,cfg):
    """Publish existing solve-entry metadata; all temporal selection is CUDA."""
    import cupy as cp
    owner=state._fddaobs_owner
    owner['publish']((1,),(1,),(owner['published'],np.float32(state.elapsed_seconds),np.float32(cfg.dt)))
    state._fddaobs_graph.launch(stream=cp.cuda.get_current_stream())


def restore_native(state, cfg, restored):
    """Restore WRF's three restart fields; rebuild its non-restart work.

    Registry.EM_COMMON saves only URATX/VRATX/TRATX. The original
    IN4DOB report book is read again and IFREST forces first-call weights
    even when the restart step is between ordinary observation calls.
    """
    if not cfg.obs_nudge_opt:
        return {}
    import cupy as cp
    from woof.ingest.wrfrst import _host_field
    nz, ny, nx = state.p.shape
    reasons = {}
    for name in ('uratx', 'vratx', 'tratx'):
        native = name.upper()
        key = 'fddaobs_' + name
        value = _host_field(restored.raw, native, (ny, nx), label=key)
        getattr(state, key)[...] = cp.pad(cp.asarray(value, dtype='f4'),
                                         ((0, 1), (0, 1)), mode='edge').ravel()
        reasons[key] = native
    state.fddaobs_clock[...] = restored.warm_start.clock.itimestep
    state.fddaobs_frame[...] = 0
    state.fddaobs_frame.view('i4')[3] = -1
    reasons.update({
        'fddaobs_clock': 'native ITIMESTEP',
        'fddaobs_frame': 'native IFREST first-call refresh',
        'fddaobs_rko': 'IN4DOB rebuilt from original OBS_DOMAIN input',
        'fddaobs_varobs': 'IN4DOB rebuilt from original OBS_DOMAIN input',
        'fddaobs_errf': 'ERROB rebuilt on first IFREST call',
        'fddaobs_savwt': 'NUDOB rebuilt on first IFREST call',
    })
    return reasons
