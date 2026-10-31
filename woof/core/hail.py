"""GPU hail diameter diagnostics. WRF public-domain source is pinned beside
tools/hail_wrf461/translate.py. PSD products use the 0.0005 m^-3 integrated
tail criterion. PSD fields are metres, HAILCAST fields millimetres.
Window maxima reset only after durable history; RUNMAX fields never reset.
"""
from __future__ import annotations
import numpy as np

HISTORY = {
    "hail_maxk1": ("HAIL_MAXK1", "m", "PSD maximum diameter at lowest model level"),
    "hail_max2d": ("HAIL_MAX2D", "m", "PSD maximum diameter over model column"),
    "hail_maxk1_run": ("HAIL_MAXK1_RUN", "m", "Run maximum PSD surface diameter"),
    "hail_max2d_run": ("HAIL_MAX2D_RUN", "m", "Run maximum PSD column diameter"),
    "hailcast_diam_max": ("HAILCAST_DIAM_MAX", "mm", "HAILCAST maximum surviving ice diameter"),
    "hailcast_diam_mean": ("HAILCAST_DIAM_MEAN", "mm", "HAILCAST mean surviving ice diameter"),
    "hailcast_diam_std": ("HAILCAST_DIAM_STD", "mm", "HAILCAST sample diameter standard deviation"),
    "hailcast_runmax": ("HAILCAST_RUNMAX", "mm", "Run maximum HAILCAST surviving ice diameter"),
}
for _index in range(1, 6):
    HISTORY[f"hailcast_dhail{_index}"] = (
        f"HAILCAST_DHAIL{_index}", "mm", f"HAILCAST embryo {_index} diameter")
EMBRYOS = tuple(f"hailcast_dhail{i}" for i in range(1, 6))
PERSISTENT = (*HISTORY, "hailcast_wup_mask", "hailcast_wdur")
# hailcast_version -> column kernel (woof/core/kernels/hail.cu).  Both are
# closed translations of the stone integration (tools/hail_wrf461/translate.py);
# the eligibility mask, cadence and window accumulation are shared.
HAILCAST_KERNELS = {"wrf_39_noaa": "hailcast_columns_wrf39",
                    "wrf_461": "hailcast_columns"}
WINDOW = ("hail_maxk1", "hail_max2d", "hailcast_diam_max",
          "hailcast_diam_mean", "hailcast_diam_std", *EMBRYOS)
# WRF diag_nwp's spherical PSD rows; column state supplies actual number and
# volume where transported. A missing moment never substitutes storm intensity.
PSD_ROWS = {
    2: {"density_g": 917., "intercept": 4.e4},
    6: {"density_g": ("wsm6_hail_opt", {0: 500., 1: 700.}),
        "intercept": ("wsm6_hail_opt", {0: 4.e6, 1: 4.e4})},
    8: {"density_g": 500., "thompson": True},
    9: {"density_g": 400., "density_h": 900.},
    10: {"density_g": ("morr_rimed_ice", {0: 400., 1: 900.})},
    16: {"density_g": ("wdm6_hail_opt", {0: 500., 1: 700.}),
         "intercept": ("wdm6_hail_opt", {0: 4.e6, 1: 4.e4})},
    18: {"density_g": 500., "density_h": 900., "alpha_h": 1.,
         "number_g": "qng", "number_h": "qnh"},
    24: {"density_g": 500., "intercept": 4.e6,
         "density_h": 912., "intercept_h": 4.e4},
    28: {"density_g": 500., "thompson": True},
    106: {"density_g": 500., "intercept": 4.e6},
}
PSD_SLOTS = ("hail_maxk1", "hail_max2d", "hail_maxk1_run", "hail_max2d_run")

def psd_parameters(cfg):
    """Resolve table-selected constants; a new spherical PSD is one row."""
    row = PSD_ROWS.get(cfg.mp_physics)
    if row is None:
        return None
    return {key: value[1][getattr(cfg, value[0])] if isinstance(value, tuple)
            else value for key, value in row.items()}

def enabled(cfg):
    return bool(cfg.gsd_diagnostics) or cfg.hailcast_opt == 1 or (cfg.hailcast_opt == -1 and
            cfg.dx <= 4000 and cfg.dy <= 4000 and cfg.mp_physics != 0)

def scratch_shapes(cfg):
    """Exact persistent and rebuilt diagnostic allocations for memory planning."""
    if not enabled(cfg):
        return {}
    plane = (cfg.ny, cfg.nx)
    mass = (cfg.nz, cfg.ny, cfg.nx)
    shapes = {slot: plane for slot in PERSISTENT
              if slot not in PSD_SLOTS or cfg.mp_physics in PSD_ROWS}
    shapes.update({slot: mass for slot in (
        "hail_zero", "hail_temperature", "hail_height", "hail_density")})
    shapes.update(hail_new_mask=plane, hail_new_duration=plane)
    return shapes


def initialize(state, cfg):
    if not enabled(cfg):
        return
    if cfg.nz > 256:
        raise ValueError("HAILCAST column storage supports at most 256 mass levels")
    for slot, shape in scratch_shapes(cfg).items():
        state.scratch(shape, slot)

def update(state, cfg):
    if not enabled(cfg):
        return
    from woof.core.kernels import get_kernel
    nz,ny,nx=state.p.shape
    shape=(nz,ny,nx); plane=(ny,nx)
    i32=np.int32; f32=np.float32
    grid=((nx*ny+127)//128,);block=(128,)
    def scratch(slot, shape=plane):
        return state.scratch(shape,slot)
    zero=scratch("hail_zero",shape)
    def species(name):
        value=getattr(state,name,None)
        return zero if value is None else value
    t=scratch("hail_temperature",shape)
    z=scratch("hail_height",shape)
    rho=scratch("hail_density",shape)
    mask=scratch("hail_new_mask")
    get_kernel("hail","hail_prepare")(grid,block,
        (state.total_theta(),state.p,state.alt,species("qv"),
         state.phb,state.php,state.w,t,z,rho,mask,
         i32(nx),i32(ny),i32(nz),i32(state.phb.ndim==1)))
    dur=scratch("hail_new_duration")
    get_kernel("hail","hail_duration")(grid,block,
        (mask,scratch("hailcast_wup_mask"),scratch("hailcast_wdur"),dur,
         i32(nx),i32(ny),f32(cfg.dt),i32(not (cfg.open_x or cfg.specified or cfg.nested)),i32(not (cfg.open_y or cfg.specified or cfg.nested))))
    scratch("hailcast_wup_mask")[...] = mask
    scratch("hailcast_wdur")[...] = dur
    row=psd_parameters(cfg)
    if row is not None and not cfg.gsd_diagnostics:
        ng=row.get("number_g", "ng");nh=row.get("number_h", "nh")
        vg="qvolg";vh="qvolh"
        get_kernel("hail","hail_psd")(grid,block,
            (rho,species("qg"),species(ng),species(vg),species("qh"),
             species(nh),species(vh),scratch("hail_maxk1"),scratch("hail_max2d"),
             scratch("hail_maxk1_run"),scratch("hail_max2d_run"),
             i32(nx),i32(ny),i32(nz),i32(getattr(state,ng,None) is not None),
             i32(getattr(state,vg,None) is not None),i32(getattr(state,nh,None) is not None),
             i32(getattr(state,vh,None) is not None),i32(row.get("thompson",False)),
             *(f32(row.get(key,default)) for key,default in (
                 ("density_g",500.),("density_h",900.),("alpha_g",0.),
                 ("alpha_h",0.),("intercept",0.),("intercept_h",0.)))))
    # WRF haildt is seconds. Cadence uses the clock, no GPU-host scalar copy.
    interval=cfg.haildt
    start=state.elapsed_seconds;end=start+cfg.dt
    if ((cfg.hailcast_opt == 1 or (cfg.hailcast_opt == -1 and
            cfg.dx <= 4000 and cfg.dy <= 4000 and cfg.mp_physics != 0)) and
            (interval<=0 or start==0 or int(end/interval)>int(start/interval))):
        get_kernel("hail",HAILCAST_KERNELS[cfg.hailcast_version])(((nx*ny+31)//32,),(32,),
            (t,z,state.p,rho,species("qv"),species("qi"),species("qc"),
             species("qr"),species("qs"),species("qg"),state.w,state.ht,dur,
             *(scratch(slot) for slot in EMBRYOS),scratch("hailcast_diam_max"),
             scratch("hailcast_diam_mean"),scratch("hailcast_diam_std"),
             scratch("hailcast_runmax"),i32(nx),i32(ny),i32(nz)))
def reset_window(state):
    from woof.core.uh_diag import _zero_domain_slot
    existing = getattr(state, "existing_scratch", None)
    gsd_active = existing is not None and existing("gsd_step_count") is not None
    for slot in WINDOW:
        if slot in ("hail_maxk1", "hail_max2d") and gsd_active:
            continue
        _zero_domain_slot(state,slot)


def registry_rows(registry):
    """Append the diagnostic's metadata without changing physics selectors."""
    registry.setdefault("diagnostic_packages", {})["hailcast_opt"] = {
        "-1": {"label": "Automatic at grid spacing up to 4 km", "package": "hail_auto", "fields": list(HISTORY)},
        "0": {"label": "Off", "package": "hail_none", "fields": []},
        "1": {"label": "WRF HAILCAST and PSD hail diameter", "package": "hail", "fields": list(HISTORY)},
    }
    registry.setdefault("diagnostic_state", {})["hail"] = {
        slot: {"description": desc, "units": units, "history": True, "restart": True}
        for slot, (_, units, desc) in HISTORY.items()}
    registry["diagnostic_state"]["hail"].update({
        "hailcast_wup_mask": {"units": "1", "history": False, "restart": True},
        "hailcast_wdur": {"units": "s", "history": False, "restart": True}})
    registry["parameters"].update({
        "hailcast_opt": {"type": "integer", "default": -1, "enum": [-1, 0, 1], "per_domain": True, "consuming_read": "woof/core/hail.py"},
        "haildt": {"type": "number", "default": 60.0, "minimum": 0.0, "maximum": 60.0, "per_domain": True, "consuming_read": "woof/core/hail.py"},
        "hailcast_version": {"type": "string", "default": "wrf_39_noaa", "enum": list(HAILCAST_KERNELS), "per_domain": True, "consuming_read": "woof/core/hail.py"}})
