"""One diagnostic driver for WRF's lightning packages. No state feedback."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from woof.lightning_contract import LIGHTNING_OPTIONS, FLASH_FIELDS, validate_lightning

def next_activation(activation, current, dt, interval):
    """WRF inclusive double clock test and repeated REAL clock addition."""
    activation=np.float32(activation)
    end=float(current)+float(np.float32(dt))
    if not float(current)<=float(activation)<=end:
        return None
    interval=np.float32(interval)
    while float(activation)<=end:
        updated=np.float32(activation+interval)
        if updated<=activation:
            raise ValueError("lightning_dt does not advance the float32 activation clock; WRF would loop forever")
        activation=updated
    return float(activation)

def initialize_lightning(state,cfg,*,latitude=None,longitude=None,start_time=None,inputs=None):
    if not cfg.lightning_option:
        return
    validate_lightning(cfg)
    from woof.core.state import _state_array_module
    xp = _state_array_module(state)
    shape=(cfg.ny,cfg.nx)
    for name in LIGHTNING_OPTIONS[cfg.lightning_option]["fields"]:
        state.scratch(shape,"ltng_"+name)
    for name in ("iccg_in_num","iccg_in_den"):
        target=state.scratch((12,*shape),"ltng_"+name)
        if inputs is not None and name in inputs:
            target[...] = xp.asarray(inputs[name], dtype=np.float32)
    if cfg.lightning_option!=3 and cfg.iccg_method in (0,2) and (latitude is None or longitude is None):
        raise ValueError("lightning's geographic IC:CG split needs latitude and longitude; absent coordinates would choose the wrong fraction")
    for name,value in (("lat",latitude),("lon",longitude)):
        target=state.scratch(shape,"ltng_"+name)
        if value is not None:
            target[...] = xp.asarray(value, dtype=np.float32)
    if cfg.iccg_method==4 and cfg.lightning_option!=3 and start_time is None:
        raise ValueError("monthly IC:CG inputs need a start calendar; no month can be selected without it")
    calendar=state.scratch((6,),"ltng_calendar",dtype=np.int32)
    if start_time is not None:
        from datetime import datetime
        stamp=start_time if isinstance(start_time,datetime) else datetime.fromisoformat(str(start_time).replace("Z","+00:00"))
        calendar[...] = xp.asarray((stamp.year,stamp.month,stamp.day,stamp.hour,stamp.minute,stamp.second),dtype=np.int32)
    else:
        calendar[...] = 0
    if cfg.lightning_option == 11:
        state.scratch(shape,"ltng_ktop_deep",dtype=np.int32)[...] = -1
    clock=state.scratch((1,),"ltng_ltngacttime")
    clock[0]=np.float32(cfg.lightning_start_seconds)
    state._lightning_host_clock = {
        "activation": float(np.float32(cfg.lightning_start_seconds)),
        "calendar": ([stamp.year,stamp.month,stamp.day,stamp.hour,stamp.minute,stamp.second]
                     if start_time is not None else [0]*6)}

def host_clock(state):
    """Host mirror of small driver clocks, rebuilt once after a restore.

    Array carriers remain the checkpoint authority. Tile sweeps transport
    this mirror alongside their ordinary scalar clocks. No domain field is
    transferred here, and an initialized step never reads a device scalar.
    """
    canonical = getattr(state, "_canonical_scalars", None)
    cached = (canonical.get("lightning_clock") if canonical is not None
              else getattr(state, "_lightning_host_clock", None))
    if cached is None:
        clock = state.existing_scratch("ltng_ltngacttime")
        if clock is None:
            raise ValueError("lightning was not initialized; activation time and IC:CG inputs are missing")
        from woof.core.state import _state_array_module
        xp = _state_array_module(state)
        calendar = state.existing_scratch("ltng_calendar")
        values = np.asarray(calendar if xp is np else xp.asnumpy(calendar))
        cached = {"activation": float(clock[0]), "calendar": [int(v) for v in values]}
        state._lightning_host_clock = cached
    return cached


def invalidate_host_clock(state):
    """A restored array carrier must replace a pre-restore driver mirror."""
    state._lightning_host_clock = None


def lightning_due(state,cfg):
    if not cfg.lightning_option:
        return False
    interval=cfg.lightning_dt if cfg.lightning_dt>0 else cfg.dt
    return next_activation(host_clock(state)["activation"],state.elapsed_seconds,cfg.dt,interval) is not None

@lru_cache(maxsize=128)
def _validated_tiles(nx,ny,tiles):
    """Validate static launch geometry once, without a domain-sized host mask."""
    area=0
    for index,(i0,i1,j0,j1) in enumerate(tiles):
        if not (0<=i0<i1<=nx and 0<=j0<j1<=ny):
            raise ValueError("lightning tile lies outside the domain")
        area+=(i1-i0)*(j1-j0)
        for a0,a1,b0,b1 in tiles[:index]:
            if max(i0,a0)<min(i1,a1) and max(j0,b0)<min(j1,b1):
                raise ValueError("lightning tiles must cover the domain exactly once; overlaps double count flashes")
    if area!=nx*ny:
        raise ValueError("lightning tiles must cover the domain exactly once; gaps omit flashes")
    return tiles


def _refuse_lightning(flags):
    causes={1:"cumulus cloud top is outside the column",
            2:"PR93 freezing-level search would read below the column",
            4:"input IC:CG ratio is -1 and divides by zero"}
    if flags:
        raise ValueError("lightning refused: "+"; ".join(
            text for bit,text in causes.items() if flags & bit))


def launch_lightning(fields,output,*,option,method,cellcount_method,dx,dy,
                     interval,factor=1.,num=0.,den=1.,adjust=0.,month=1,tiles=None):
    """Fields are contiguous [k,j,i], including WRF's top memory level.

    top is one-based. tiles are disjoint exhaustive (i0,i1,j0,j1) rectangles.
    Method 2 reduces the domain once; method 1 reduces each tile.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    nz,ny,nx=fields["t"].shape
    if not 1<=month<=12:
        raise ValueError("IC:CG input month must be 1..12")
    if cellcount_method==0:
        cellcount_method=(1 if nx*dx>1.e4 else 2) if option==1 else 1
    tiles=_validated_tiles(nx,ny,tuple(tuple(tile) for tile in (tiles or [(0,nx,0,ny)])))
    summary=cp.zeros(2,dtype=cp.float32)
    invalid=cp.zeros(1,dtype=cp.int32)
    i32=np.int32;f32=np.float32
    first=("refl","w","t","z")
    if option in (1,2):
        levels=cp.empty(2*nz,dtype=cp.float32)
        level_kernel=get_kernel("lightning","lightning_levels")
        patch=get_kernel("lightning","lightning_patch")
        def reduce(tile):
            geometry=tuple(i32(v) for v in tile)
            level_kernel((nz,),(256,),
                         (fields["refl"],fields["w"],i32(nz),i32(ny),i32(nx),*geometry,levels))
            patch((1,),(256,),
                  (fields["refl"],fields["t"],fields["z"],fields["ht"],fields["land"],levels,
                   i32(nz),i32(ny),i32(nx),i32(option),*geometry,summary))
        if cellcount_method==2:
            reduce((0,nx,0,ny))
    columns=get_kernel("lightning","lightning_columns")
    order=(*first,"dz","theta","pi","qc","qr","qi","qs","qg",
           "ht","land","lat","lon","top","inputnum","inputden")
    for tile in tiles:
        if option in (1,2) and cellcount_method==1:
            reduce(tile)
        args=tuple(fields[k] for k in order)+(summary,)+tuple(i32(v) for v in (nz,ny,nx,option,method,month))
        args+=tuple(f32(v) for v in (dx,dy,adjust,num,den,factor,interval))
        args+=tuple(i32(v) for v in tile)+tuple(output[k] for k in (*FLASH_FIELDS,"lpi"))+(invalid,)
        columns(((nx*ny+127)//128,),(128,),args)
    from woof.core.health_ledger import read_status
    code=read_status(invalid,site="lightning",describe=_refuse_lightning)
    _refuse_lightning(code)

def update_lightning(state,cfg,*,history_reflectivity_due=False):
    if not lightning_due(state,cfg):
        return
    import cupy as cp
    from datetime import datetime,timedelta
    from woof.core.physics import _prepare_atmosphere
    from woof.core.refl import consume_refl_10cm
    a=_prepare_atmosphere(state)
    nz,ny,nx=state.p.shape
    def extend(value,top=None):
        value=cp.broadcast_to(value,(nz,ny,nx))
        last=value[-1:] if top is None else cp.full((1,ny,nx),top,dtype=cp.float32)
        return cp.ascontiguousarray(cp.concatenate((value,last)))
    zero=cp.zeros((nz+1,ny,nx),dtype=cp.float32)
    z=cp.ascontiguousarray(0.5*(a["z_interface"][:-1]+a["z_interface"][1:]))
    fields={"t":extend(a["temperature"]),"theta":extend(a["theta"]),"pi":extend(a["exner"]),
            "z":extend(z),"dz":extend(a["dz"],0.),"w":cp.ascontiguousarray(state.w),
            "ht":state.ht,"land":state.physics.fields["xland"]}
    for name in ("lat","lon"):
        fields[name]=state.existing_scratch("ltng_"+name)
    for name in ("qc","qr","qi","qs","qg"):
        species=getattr(state,name,None)
        fields[name]=zero if species is None else extend(species,0.)
    if cfg.lightning_option in (1,2):
        refl=state.physics.refl_10cm
        if refl is None:
            raise ValueError("lightning has no reflectivity from this microphysics call; convective cells cannot be identified")
        fields["refl"]=extend(refl,-35.)
        if not history_reflectivity_due:
            consume_refl_10cm(state)
    else:
        fields["refl"]=zero
    top=state.existing_scratch("ltng_ktop_deep")
    if cfg.lightning_option==11 and top is None:
        raise ValueError("lightning_option=11 has no produced cumulus cloud top; PR92z cannot infer a flash rate")
    fields["top"]=cp.ones((ny,nx),dtype=cp.int32) if top is None else top
    for name,symbol in (("inputnum","iccg_in_num"),("inputden","iccg_in_den")):
        fields[name]=state.existing_scratch("ltng_"+symbol)
    output={name:state.scratch((ny,nx),"ltng_"+name) for name in LIGHTNING_OPTIONS[cfg.lightning_option]["fields"]}
    output.setdefault("lpi",cp.zeros((ny,nx),dtype=cp.float32))
    interval=cfg.lightning_dt if cfg.lightning_dt>0 else cfg.dt
    month = 1
    if cfg.iccg_method == 4 and cfg.lightning_option != 3:
        calendar=host_clock(state)["calendar"]
        month=(datetime(*(int(v) for v in calendar))+timedelta(seconds=state.elapsed_seconds)).month
    launch_lightning(fields,output,option=cfg.lightning_option,method=cfg.iccg_method,
                     cellcount_method=cfg.cellcount_method,dx=cfg.dx,dy=cfg.dy,
                     interval=interval,factor=cfg.flashrate_factor,num=cfg.iccg_prescribed_num,
                     den=cfg.iccg_prescribed_den,adjust=cfg.cldtop_adjustment,month=month)
    clock=state.existing_scratch("ltng_ltngacttime")
    value=next_activation(host_clock(state)["activation"],state.elapsed_seconds,cfg.dt,interval)
    clock[0]=np.float32(value)
    state._lightning_host_clock["activation"]=value
