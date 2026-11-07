"""Native WRF diagnostic time series, GPU reductions and deferred output."""
from contextvars import ContextVar
import numpy as np
PERSISTENT=('misc_pk1m','misc_mu_2m','misc_dpsdt','misc_dmudt')
_sink=ContextVar('native_wrf_diagnostic_print',default=None)
def drop_inert(values):
    if values.get('diag_print',0)==0:values.pop('diag_print',None)

def scratch_shapes(cfg):
    if not cfg.diag_print:return {}
    p=(cfg.ny,cfg.nx);shapes={k:p for k in (*PERSISTENT,'misc_zero','misc_surface_pressure')}
    from woof.core.nwp_diagnostics import active
    if not active(cfg):
        shapes.update(misc_pressure_w=(cfg.nz+1,*p),misc_phb=(cfg.nz+1,*p),
                      misc_rho=(cfg.nz,*p),misc_zero3d=(cfg.nz,*p))
    return shapes
def prepare_microphysics(state,cfg):
    if not cfg.diag_print or not cfg.mp_physics:return
    from woof.core.kernels import get_kernel
    n=cfg.nx*cfg.ny
    get_kernel('misc_diag_print','misc_print_surface_pressure')(((n+255)//256,),(256,),
        (state.p,state.php,state.phb,state.existing_scratch('misc_surface_pressure'),
         np.int32(n),np.int32(state.phb.ndim==3)))

def bind_window(state,spec,domain_nx):
    y,x=spec.interior_in_tile('mass')
    state._misc_print_window=(x.start,x.stop,y.start,y.stop)
    state._misc_print_origin=(spec.ci0,spec.cj0,domain_nx)
def update(state,cfg):
    if not cfg.diag_print:return
    import cupy as cp
    from woof.core.kernels import get_kernel
    s=state.existing_scratch;n=cfg.nx*cfg.ny;i32=np.int32
    from woof.core.nwp_diagnostics import active
    ps=s('misc_surface_pressure') if cfg.mp_physics else s('nwp_surface_pw') if active(cfg) else s('misc_pressure_w')[0]
    mu=state.mup;p=state.physics;f=getattr(p,'fields',{})
    get_kernel('misc_diag_print','misc_print_fields')(((n+255)//256,),(256,),
        (ps,mu,*(s(k) for k in PERSISTENT),np.float32(cfg.dt),i32(state.elapsed_seconds==0.),i32(n)))
    if state.elapsed_seconds==0.:return
    zero=s('misc_zero');micro=getattr(p,'microphysics',None)
    def amount(owner,key):
        a=getattr(owner,key,None);return zero if a is None else a
    args=(s('misc_dpsdt'),s('misc_dmudt'),amount(p,'cu_raincv'),amount(micro,'rainncv'),
          amount(p,'rainc'),amount(micro,'rainnc'),f.get('sfcevp',zero),f.get('hfx',zero),f.get('lh',zero))
    window=getattr(state,'_misc_print_window',None)
    if window is None:x0,x1,y0,y1=0,cfg.nx,0,cfg.ny
    else:x0,x1,y0,y1=window
    blocks=((x1-x0)*(y1-y0)+255)//256
    partial=cp.empty((14,blocks),dtype='f4');out=cp.empty(14,dtype='f4')
    indices=cp.empty((3,blocks),dtype='i4');index=cp.empty(3,dtype='i4')
    ox,oy,dnx=getattr(state,'_misc_print_origin',(0,0,cfg.nx))
    get_kernel('misc_diag_print','misc_print_partial')((blocks,14),(256,),
        (*args,partial,indices,np.float32(cfg.dt),i32(cfg.nx),i32(x0),i32(x1),i32(y0),i32(y1),i32(cfg.diag_print),i32(ox),i32(oy),i32(dnx)))
    get_kernel('misc_diag_print','misc_print_finish')((14,),(128,),(partial,indices,out,index,i32(blocks)))
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        # Native patch sums are FP32 left folds. Preserve that association
        # only in the explicit proof mode; default runs retain parallel trees.
        owned=tuple(cp.ascontiguousarray(a[y0:y1,x0:x1]) for a in args)
        get_kernel('misc_diag_print','misc_print_native_order')((1,),(32,),
            (*owned,out,np.float32(cfg.dt),i32((x1-x0)*(y1-y0))))
        if cfg.diag_print==1:out[2:13]=0.
    event=cp.cuda.Event();event.record()
    record=(int(cfg.grid_id),float(state.elapsed_seconds)/60.,float(cfg.dt),int(cfg.diag_print),
            (x1-x0)*(y1-y0),out,index,event,partial,indices,dnx)
    sink=_sink.get()
    if sink is None:
        queue=getattr(state,'_misc_print_records',None)
        if queue is None:state._misc_print_records=queue=[]
        queue.append(record)
    else:sink.append(record)
def begin():
    rows=[];return rows,_sink.set(rows)
def end(token):_sink.reset(token)
def drain(rows,path):
    """Read completed GPU summaries at an existing forecast output boundary."""
    if not rows:return
    import cupy as cp,json
    from pathlib import Path
    groups={}
    for grid,time,dt,mode,n,out,index,event,partial,indices,dnx in rows:
        event.synchronize();groups.setdefault((grid,time,dt,mode),[]).append((n,out,index,dnx))
    result=[]
    for (grid,time,dt,mode),parts in sorted(groups.items()):
        total=sum(p[0] for p in parts);values=cp.stack([p[1] for p in parts]);v=cp.empty(14,dtype='f4')
        v[:11]=values[:,:11].sum(axis=0);v[11:]=values[:,11:].max(axis=0)
        ix=cp.stack([p[2] for p in parts]);winner=values[:,11:].argmax(axis=0)
        selected=ix[winner,cp.arange(3)]
        # Scaling is device arithmetic. The host only formats completed words.
        v[:2]/=np.float32(total);v[:2]*=np.float32(108.)
        v[2:5]/=np.float32(dt);v[2:5]/=np.float32(total);v[5:11]/=np.float32(total)
        a=cp.asnumpy(cp.concatenate((v,selected.astype('f4')))).tolist()
        coords=[None if a[11+k]==0. else [int(a[14+k])%parts[0][3]+1,int(a[14+k])//parts[0][3]+1] for k in range(3)]
        a=a[:14];result.append(dict(grid_id=grid,xtime_minutes=time,diag_print=mode,values=a,maxima_ij=coords,
            fields=['dpsdt_mb_3h','dmudt_mb_3h','drcdt_mm_s','drndt_mm_s','dardt_mm_s','rainc_mm','rainnc_mm','raint_mm','sfcevp_mm','hfx_W_m2','lh_W_m2','raincmax_mm','rainncmax_mm','dmumax_bar']))
        print(f'd{grid:02d} Domain average of dpsdt, dmudt (mb/3h): {time:g} {a[0]:.9g} {a[1]:.9g}',flush=True)
        print(f'd{grid:02d} Max mu change time step: {coords[2]} {a[13]:.9g}',flush=True)
        if mode==2:
            print(f'd{grid:02d} Domain average of dardt, drcdt, drndt (mm/sec): {time:g} {a[4]:.9g} {a[2]:.9g} {a[3]:.9g}',flush=True)
            print(f'd{grid:02d} Domain average of rt_sum, rc_sum, rnc_sum (mm): {time:g} {a[7]:.9g} {a[5]:.9g} {a[6]:.9g}',flush=True)
            print(f'd{grid:02d} Max Accum Resolved Precip, I,J (mm): {a[12]:.9g} {coords[1]}',flush=True)
            print(f'd{grid:02d} Max Accum Convective Precip, I,J (mm): {a[11]:.9g} {coords[0]}',flush=True)
            print(f'd{grid:02d} Domain average of sfcevp, hfx, lh: {time:g} {a[8]:.9g} {a[9]:.9g} {a[10]:.9g}',flush=True)
    path=Path(path);existing=json.loads(path.read_text()) if path.exists() else [];path.write_text(json.dumps(existing+result,indent=2)+'\n');rows.clear()
def registry_rows(registry):
    registry['parameters']['diag_print']={'type':'integer','default':0,'enum':[0,1,2],'per_domain':False,'consuming_read':'woof/core/misc_diag_print.py','source_pin':'f52c197ed39d12e087d02c50f412d90d418f6186'}
    registry.setdefault('diagnostic_state',{})['diag_print']={k:dict(history=False,restart=True,units='Pa s-1' if 'dt' in k else 'Pa',description='Native diagnostic time-series carrier') for k in PERSISTENT}
