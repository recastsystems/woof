"""Operational GSD diagnostics, NOAA-EMC/HRRR 40ee6058 (WRFV3.9).

Every reduction remains on the device. History reads do not reset these
windows. See docs/dev/operational-output-controls.md for source and limits.
"""
from __future__ import annotations
import numpy as np

HISTORY = {}
for names, units, description in (
    (("wspd10max","wspd10umax","wspd10vmax","wspd80max","wspd80umax","wspd80vmax"), "m s-1", "Wind at interval maximum speed"),
    (("w_up_max","w_dn_max","w_mean","wspd10","wspd80"), "m s-1", "GSD vertical or horizontal wind diagnostic"),
    (("grpl_max",), "kg m-2", "Maximum column integrated graupel"),
    (("refd_max","refdm10c_max"), "dbZ", "Maximum derived radar reflectivity"),
    (("composite_refl_10cm","refl_10cm_1km","refl_10cm_4km"), "dbZ", "GSD reflectivity at history time"),
    (("ltg1_max","ltg2_max","ltg3_max"), "flashes km-2 5min-1", "Maximum lightning threat"),
    (("lwp","iwp"), "g m-2", "Column cloud water path"),
    (("maxcldfra",), "0 to 1", "Maximum boundary layer cloud fraction"),
    (("rel_vort","rel_vort_max","rel_vort_max01"), "s-1", "Relative vertical vorticity"),
):
    for slot in names:
        symbol = slot if slot.startswith("rel_vort") else slot.upper()
        HISTORY[slot] = (symbol, units, description)
BANDS = (("",2000.,5000.),("16",1000.,6000.),("02",0.,2000.),("03",0.,3000.))
for suffix, lower, upper in BANDS:
    for prefix in ("up_heli_max", "up_heli_min", "uh"):
        slot=prefix+suffix
        HISTORY[slot]=(slot.upper(),"m2 s-2",f"Updraft helicity {lower:g}-{upper:g} m AGL")
HISTORY["up_heli_max"]=("UP_HELI_MAX","m2 s-2","MAX UPDRAFT HELICITY")
PERSISTENT = (*(slot for slot in HISTORY if slot != "up_heli_max"), "gsd_w_mean_sum", "gsd_step_count", "gsd_ltg1", "gsd_ltg2", "gsd_total_steps")
WORK = ("gsd_zero", "gsd_uh_use", "gsd_pressure_w", "gsd_temperature",
        "gsd_density", "gsd_wind10", "gsd_wind80")

def scratch_shapes(cfg):
    if not cfg.gsd_diagnostics:
        return {}
    plane=(cfg.ny,cfg.nx); mass=(cfg.nz,*plane)
    shapes={slot: mass if slot in ("rel_vort","gsd_zero","gsd_temperature","gsd_density") else plane
            for slot in (*PERSISTENT,*WORK)}
    shapes["gsd_pressure_w"]=(cfg.nz+1,*plane)
    return shapes

def initialize(state,cfg):
    for slot,shape in scratch_shapes(cfg).items():
        state.scratch(shape,slot)
    if cfg.gsd_diagnostics:
        for name in ("wspd10","wspd80"):
            state.existing_scratch(name)[...]=-99.

def update_helicity(state,cfg,*,history_due=False):
    """Sample start-of-step state, as module_first_rk_step_part2 does."""
    if not cfg.gsd_diagnostics:
        return
    from woof.core.kernels import get_kernel
    from woof.core.uh_diag import _supported_boundary_geometry
    if not _supported_boundary_geometry(cfg):
        raise NotImplementedError("gsd_diagnostics requires specified/nested or open x/y boundaries")
    nz,ny,nx=cfg.nz,cfg.ny,cfg.nx
    if nx<4 or ny<4 or nz<3:
        raise ValueError("gsd_diagnostics requires nx, ny >= 4 and nz >= 3 for helicity stencils")
    grid=((nx*ny+127)//128,); block=(128,)
    f=np.float32;i=np.int32
    zero=state.existing_scratch("gsd_zero")
    u, v = state.u, state.v
    if getattr(state, "_moving_frame_velocity", None):
        from woof.moving_frame import ground_relative_winds
        u, v = ground_relative_winds(state)
    get_kernel("gsd_diagnostics","gsd_prepare")(grid,block,
        (u,v,state.php,state.phb,i(state.phb.ndim==3),state.thp,
         state.thb,i(state.thb.ndim==3),state.p,state.alt,
         state.qv if state.qv is not None else zero,state.fnm,state.fnp,
         *(state.existing_scratch(name) for name in
           ("gsd_pressure_w","gsd_temperature","gsd_density","gsd_wind10","gsd_wind80")),
         i(nx),i(ny),i(nz),i(bool(history_due))))
    rv=state.existing_scratch("rel_vort")
    get_kernel("gsd_helicity","gsd_vorticity")(((nz*ny*nx+127)//128,),block,
        (state.u,state.v,state.php,state.phb,i(state.phb.ndim==3),
         state.msfu,state.msfv,state.dn,state.dnw,state.fnm,state.fnp,
         f(state.cf1),f(state.cf2),f(state.cf3),f(1)/f(cfg.dx),f(1)/f(cfg.dy),
         rv,i(nz),i(ny),i(nx)))
    for suffix,lower,upper in BANDS:
        column=state.existing_scratch("uh"+suffix)
        use=state.existing_scratch("gsd_uh_use")
        get_kernel("gsd_helicity","gsd_uh_columns")(grid,block,
            (state.w,state.php,state.phb,i(state.phb.ndim==3),state.ht,rv,
             column,use,f(lower),f(upper),i(nz),i(ny),i(nx)))
        get_kernel("gsd_helicity","gsd_uh_fold")(grid,block,
            (column,use,state.existing_scratch("up_heli_max"+suffix),
             state.existing_scratch("up_heli_min"+suffix),i(ny),i(nx)))
        for slot in ("up_heli_max"+suffix,"up_heli_min"+suffix):
            target=state.existing_scratch(slot)
            target[:,0]=target[:,1];target[0,:]=target[1,:]
    get_kernel("gsd_helicity","gsd_vorticity_fold")(grid,block,
        (rv,state.php,state.phb,i(state.phb.ndim==3),state.ht,
         state.existing_scratch("rel_vort_max"),state.existing_scratch("rel_vort_max01"),
         state.existing_scratch("gsd_total_steps"),
         i(max(1,int(f(cfg.reset_interval1*60)/f(cfg.dt)))),i(nz),i(ny),i(nx)))
    for slot in ("rel_vort_max", "rel_vort_max01"):
        target=state.existing_scratch(slot)
        target[:,0]=target[:,1];target[0,:]=target[1,:]

def reset_interval(state,cfg):
    if not cfg.gsd_diagnostics:
        return
    from woof.core.output_diagnostics import boundary_due
    if not boundary_due(state.elapsed_seconds,cfg.reset_interval1*60.):
        return
    # In the fork this reset follows the first-RK helicity calculation.
    for slot in (*PERSISTENT, "up_heli_max"):
        if slot.startswith("uh") or slot in ("rel_vort","gsd_ltg1","gsd_ltg2",
                "gsd_total_steps","composite_refl_10cm","refl_10cm_1km","refl_10cm_4km"):
            continue
        state.existing_scratch(slot)[...]=-99. if slot in ("wspd10","wspd80") else 0.
    for slot in ("hail_maxk1","hail_max2d"):
        value=state.existing_scratch(slot)
        if value is not None:
            value[...]=0.

def update(state,cfg,*,history_due):
    if not cfg.gsd_diagnostics:
        return
    from woof.core.kernels import get_kernel
    import cupy as cp
    nx,ny,nz=cfg.nx,cfg.ny,cfg.nz
    shape=(nz,ny,nx);plane=(ny,nx)
    zero=state.existing_scratch("gsd_zero")
    u, v = state.u, state.v
    if getattr(state, "_moving_frame_velocity", None):
        from woof.moving_frame import ground_relative_winds
        u, v = ground_relative_winds(state)
    def species(name):
        value=getattr(state,name,None)
        return zero if value is None else value
    fields=getattr(state.physics,"fields",{})
    def surface(name):
        return fields.get(name,zero[0])
    def cloud(name):
        return fields.get(name,zero)
    scratch=state.existing_scratch
    f=np.float32;i=np.int32
    get_kernel("gsd_diagnostics","gsd_columns")(((nx*ny+127)//128,),(128,),
        (u,v,state.w,state.php,state.phb,i(state.phb.ndim==3),
         state.thp,state.thb,i(state.thb.ndim==3),state.p,state.alt,
         *(species(name) for name in ("qv","qc","qr","qi","qs","qg","qh")),
         state.znw,scratch("gsd_density"),scratch("gsd_temperature"),
         scratch("gsd_wind10"),scratch("gsd_wind80"),
         state.msft,cloud("qc_bl"),cloud("cldfra_bl"),surface("u10"),surface("v10"),
         scratch("gsd_pressure_w"),
         *(scratch(name) for name in COLUMN_OUTPUTS),
         scratch("hail_maxk1") if scratch("hail_maxk1") is not None else zero[0],
         scratch("hail_max2d") if scratch("hail_max2d") is not None else zero[0],
         i(nx),i(ny),i(nz),f(state.p_top),i(bool(history_due)),i(cfg.icloud_bl),
         i(cfg.mp_physics in (8,28))))
    # Multi-pass device reductions, no host scalar extraction or synchronization.
    maxima=tuple(cp.max(scratch(slot)) for slot in ("ltg1_max","ltg2_max"))
    get_kernel("gsd_diagnostics","gsd_lightning3")(((nx*ny+255)//256,),(256,),
        (scratch("ltg1_max"),scratch("ltg2_max"),scratch("ltg3_max"),*maxima,i(nx*ny)))
    for slot in ("hail_maxk1","hail_max2d"):
        if scratch(slot) is not None:
            cp.maximum(scratch(slot+"_run"),scratch(slot),out=scratch(slot+"_run"))
    refl=scratch("refl_10cm")
    if history_due and refl is not None:
        rainc=getattr(state.physics,"cu_raincv",None)
        get_kernel("gsd_diagnostics","gsd_reflectivity")(((nx*ny+127)//128,),(128,),
            (refl,state.phb,i(state.phb.ndim==3),state.ht,
             zero[0] if rainc is None else rainc,
             *(scratch(name) for name in ("composite_refl_10cm","refl_10cm_1km","refl_10cm_4km")),
             i(nx),i(ny),i(nz),f(cfg.dt)))


def finalize_history_fields(fields):
    """Recompute the domain-wide LTG3 ratio after streamed tiles assemble.

    Tile-local ratios cannot represent the native whole-domain reduction.
    Only the two horizontal maxima planes cross the device on a host frame.
    """
    if not all(name in fields for name in ("LTG1_MAX","LTG2_MAX","LTG3_MAX")):
        return fields
    import cupy as cp
    from woof.core.kernels import get_kernel
    host=isinstance(fields["LTG1_MAX"],np.ndarray)
    a,b=(cp.asarray(fields[name]) for name in ("LTG1_MAX","LTG2_MAX"))
    out=cp.empty_like(a)
    get_kernel("gsd_diagnostics","gsd_lightning3")(((a.size+255)//256,),(256,),
        (a,b,out,cp.max(a),cp.max(b),np.int32(a.size)))
    fields["LTG3_MAX"]=cp.asnumpy(out) if host else out
    return fields

COLUMN_OUTPUTS=("wspd10max","wspd10umax","wspd10vmax","w_up_max","w_dn_max",
    "w_mean","gsd_w_mean_sum","gsd_step_count","grpl_max","refd_max","refdm10c_max",
    "ltg1_max","ltg2_max","gsd_ltg1","gsd_ltg2","wspd10","wspd80","lwp","iwp","maxcldfra")
