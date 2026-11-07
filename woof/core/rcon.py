"""WRF 4.7.1 RCON driver, with pinned native initialized lookup tables.

The initialized table identity is checked before its device upload.
"""
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

TABLE_FILENAME = "rcon-wrf471-tables.bin"
TABLE_BYTES = 380970392
TABLE_SHA256 = "f8a1a5f9fca4b28baf6bd26be7ceb1d670d9e5d515f262ff84eb8746b676dc8d"
VOLUME = "th pii p w dz qv qc qi qr qs qg ni nr nc nwfa nifa nbca".split()
SURFACE = "nwfa2d nifa2d nbca2d rainnc rainncv cloudnc snownc snowncv graupelnc graupelncv sr".split()
DIAGNOSTIC = "refl_10cm re_cloud re_ice re_snow".split()


def table_root():
    return Path(os.environ.get("WOOF_RCON_TABLE_ROOT", Path.home()/".woof/tables/rcon"))


def table_schema():
    return json.loads((Path(__file__).resolve().parents[1]/"data/rcon/table-schema.json").read_text())


def table_identity():
    from woof.file_digest import sha256_file
    p=table_root()/TABLE_FILENAME
    if not p.is_file() or p.stat().st_size != TABLE_BYTES or sha256_file(p) != TABLE_SHA256:
        raise ValueError(f"RCON requires its pinned tables at {p}; run woof fetch-tables --schemes rcon --from DIR --replace-damaged")
    source=Path(__file__).parent/"kernels/rcon_generated.cuh"
    return {"generation":"wrf_471", "table_sha256":TABLE_SHA256,
            "table_bytes":TABLE_BYTES,"generated_source_sha256":sha256_file(source)}


@lru_cache(None)
def tables(root, device):
    import cupy as cp
    from woof.file_digest import sha256_file
    p=Path(root)/TABLE_FILENAME
    if p.stat().st_size != TABLE_BYTES or sha256_file(p) != TABLE_SHA256:
        raise ValueError(f"RCON requires pinned WRF 4.7.1 initialized tables: {p}")
    raw=p.read_bytes();offset=0;owner={}
    for name, spec in sorted(table_schema().items()):
        dtype={"double":"<f8","float":"<f4"}[spec["typ"]]
        count=int(np.prod(spec["shape"]))
        a=np.frombuffer(raw,dtype,count,offset)
        owner[name]=cp.asarray(a);offset+=a.nbytes
    assert offset == len(raw)
    pointers=cp.asarray(np.asarray([owner[n].data.ptr for n in sorted(owner)],dtype=np.uint64))
    cp.cuda.get_current_stream().synchronize()
    return owner,pointers


@lru_cache(None)
def module(device, capacity):
    from woof.core.kernels import load_module, load_module_int_defines
    return (load_module("rcon") if capacity == 64 else
            load_module_int_defines("rcon", (("RCON_COLUMN_KMAX",capacity),)))


def launch_driver(fields,dt,*,aer=1,wif=1,do_refl=False,do_radii=True,ke_diag=None,root=None):
    """Complete native driver boundary; arrays are C-order (k,j,i), radii m."""
    import cupy as cp
    nz=int(fields["qv"].shape[0]);columns=int(fields["qv"].size//nz)
    if not 1 <= nz <= 256:
        raise ValueError("RCON column workspace supports 1..256 levels")
    capacity=next(n for n in (64,128,256) if nz<=n)
    device=cp.cuda.Device().id
    owner,pointers=tables(str(root or table_root()),device)
    args=[fields.get(n) if fields.get(n) is not None else np.uint64(0) for n in VOLUME+SURFACE+DIAGNOSTIC]
    args.extend((np.int32(nz),np.int32(columns),np.float32(dt),np.int32(aer),np.int32(wif),
                 np.int32(do_refl),np.int32(do_radii),np.int32(ke_diag or nz),pointers))
    module(device,capacity).get_function("rcon_step")(((columns+31)//32,),(32,),tuple(args))


def apply(state, cfg, dt, *, refl_10cm_due=False):
    """Prepare the native RCON driver and publish its physical outputs."""
    import cupy as cp
    from woof.core import constants as C
    from woof.core.microphysics import MicrophysicsDiagnostics
    from woof.core.thompson_aerosol_state import launch_aerosol_exner
    from woof.core.refl import stash_refl_10cm
    nz,ny,nx=state.p.shape
    shape=state.p.shape
    th=state.scratch(shape,"mp_th")
    pii=state.scratch(shape,"mp_pii")
    dz=state.scratch(shape,"mp_dz8w")
    z=state.scratch((nz+1,ny,nx),"mp_z8w")
    thb=state.thb if state.thb.ndim==3 else state.thb[:,None,None]
    phb=state.phb if state.phb.ndim==3 else state.phb[:,None,None]
    th[...] = state.total_theta() if getattr(state,"use_theta_m",0) else thb+state.thp
    if getattr(state,"use_theta_m",0):
        from woof.core.microphysics import save_pre_mp_theta
        save_pre_mp_theta(state)
    else:
        state.h_diabatic[...] = th
    launch_aerosol_exner(state.p,pii)
    z[...] = (phb+state.php)/np.float32(C.G)
    dz[...] = z[1:]-z[:-1]
    surfaces={
        "rainnc":state.scratch((ny,nx),"mp_rainnc"),
        "rainncv":state.scratch((ny,nx),"mp_rainncv"),
        "snownc":state.scratch((ny,nx),"mp_snownc"),
        "snowncv":state.scratch((ny,nx),"mp_snowncv"),
        "graupelnc":state.scratch((ny,nx),"mp_graupelnc"),
        "graupelncv":state.scratch((ny,nx),"mp_graupelncv"),
        "sr":state.scratch((ny,nx),"mp_sr"),
    }
    fields={n:getattr(state,n) for n in ("p","w","qv","qc","qi","qr","qs","qg","ni","nr","nc","nwfa","nifa","nwfa2d","nifa2d")}
    fields.update(surfaces)
    fields.update(th=th,pii=pii,dz=dz,cloudnc=state.scratch((ny,nx),"mp_cloudnc"),
        refl_10cm=state.scratch(shape,"refl_10cm"),
        re_cloud=state.scratch(shape,"rcon_re_cloud"),
        re_ice=state.scratch(shape,"rcon_re_ice"),
        re_snow=state.scratch(shape,"rcon_re_snow"))
    launch_driver(fields,dt,aer=cfg.aer_init_opt,wif=cfg.wif_input_opt,
                  do_refl=refl_10cm_due,do_radii=True)
    for target,source in ((state.effc,"re_cloud"),(state.effi,"re_ice"),(state.effs,"re_snow")):
        target[...] = fields[source]*np.float32(1.e6)
    from woof.core.afwa_diagnostics import capture_microphysics_theta
    if not getattr(state,"use_theta_m",0):
        capture_microphysics_theta(state,cfg,th)
    if getattr(state,"use_theta_m",0):
        from woof.core.microphysics import moist_physics_finish
        moist_physics_finish(state,cfg,th,dt)
    elif not cfg.no_mp_heating:
        limit=np.float32(cfg.mp_tend_lim)*np.float32(dt)
        increment=cp.clip(th-state.h_diabatic,-limit,limit)
        state.thp[...] += increment
        state.h_diabatic[...] = increment/np.float32(dt)
    else:
        state.h_diabatic.fill(0)
    if refl_10cm_due:
        stash_refl_10cm(state,fields["refl_10cm"])
    return MicrophysicsDiagnostics(**surfaces)
