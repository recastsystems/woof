"""Model adapter for the corrected FAST_SBM column scheme.

The registry admission is separate from this adapter. Spectral distributions
are carried by DomainState and advanced by the existing scalar transport.
"""
import numpy as np
from woof.core.fast_sbm_contract import STATE_FIELDS

SURFACE_FIELDS=("xland","rainnc","rainncv","snownc","snowncv","graupelnc","graupelncv")
NUMBER_MAP={"qnc":"nc","qnr":"nr","qni":"ni","qns":"ns","qng":"ng","qna":"nn"}
#: Literal scratch slot names, so the restart classifier's static harvest
#: (tests/test_restart.py) sees every slot this adapter allocates.
_SURFACE_SLOTS={"rainnc":"mp_rainnc","rainncv":"mp_rainncv","snownc":"mp_snownc",
                "snowncv":"mp_snowncv","graupelnc":"mp_graupelnc","graupelncv":"mp_graupelncv"}


def frozen_fraction(total,snow,graupel):
    """SR, the frozen share of this step's surface precipitation, in [0, 1].

    The column kernel accumulates RAINNCV (all phases) and SNOWNCV and
    GRAUPELNCV in different summation orders, so in an all-snow column the
    FP32 sum snow+graupel can exceed RAINNCV by an ulp. Dividing by the
    larger of the two keeps SR <= 1 exactly (x/x == 1; x/y <= 1 for y >= x
    under round-to-nearest) and changes nothing where the sum is below the
    total. Breakage prevented: every winter real-data mp_physics=30 run
    stopped at its first snowing step with "microphysics SR is outside its
    validated range [0, 1]".
    """
    import cupy as cp
    frozen=snow+graupel
    return cp.where(total>0,frozen/cp.maximum(cp.maximum(total,frozen),np.float32(1.e-30)),np.float32(0))


def apply(state,cfg,dt,*,refl_10cm_due=False):
    import cupy as cp
    from woof.core import constants as c
    from woof.core.fast_sbm_cuda import launch
    from woof.core.microphysics import (save_pre_mp_theta,moist_physics_finish,
                                        MicrophysicsDiagnostics)
    if getattr(state,"fast_sbm_bins",None) is None:
        raise ValueError("FAST_SBM requires the 142 transported spectral fields; bulk moisture alone does not specify a spectral state")
    if refl_10cm_due:
        raise ValueError("FAST_SBM polar radar diagnostics are not ported; a requested reflectivity frame would lack its scheme-native scattering calculation")
    driver=getattr(state,"physics",None)
    if driver is None or "xland" not in driver.fields:
        raise ValueError("FAST_SBM requires the initialized surface xland field to select the natural continental or maritime aerosol spectrum")
    if getattr(state,"fast_sbm_itimestep",None) is None and float(getattr(state,"elapsed_seconds",0.)) == 0.:
        # A bulk analysis carries no spectra. Its condensate is carried into
        # the bins mass-exactly (woof/core/fast_sbm_bulk_start.py) instead
        # of being discarded; cells that already hold bins are untouched.
        from woof.core.fast_sbm_bulk_start import start_bins_from_bulk
        # The receipt goes to the run log only: a DomainState attribute
        # would be unclassified state to the restart manifest, which
        # refuses every checkpoint write that carries one.
        receipt=start_bins_from_bulk(state,driver.fields["xland"])
        if receipt.get("cells_filled"):
            import sys
            print(f"FAST_SBM bulk-to-bin start: {receipt}",file=sys.stderr,flush=True)
    nz,ny,nx=state.p.shape
    thb=state.thb if state.thb.ndim==3 else state.thb[:,None,None]
    phb=state.phb if state.phb.ndim==3 else state.phb[:,None,None]
    th=state.scratch((nz,ny,nx),"mp_th"); th[:]=thb+state.thp
    rho=state.scratch((nz,ny,nx),"mp_rho"); rho[:]=1./state.alt
    pii=state.scratch((nz,ny,nx),"mp_pii"); pii[:]=cp.power(state.p/np.float32(c.P0),np.float32(c.RCP))
    dz=state.scratch((nz,ny,nx),"mp_dz8w")
    z=state.scratch((nz+1,ny,nx),"mp_z8w"); z[:]=(phb+state.php)/np.float32(c.G)
    dz[:]=z[1:]-z[:-1]
    # A state already on the physics grid (woof/core/mpas_column_suite.py)
    # hands over its A-grid winds; an ARW state's are destaggered here.
    mass_winds=getattr(state,"mass_point_winds",None)
    if mass_winds is not None: u_mass,v_mass=mass_winds()
    else:
        u_mass=np.float32(.5)*(state.u[:,:,:-1]+state.u[:,:,1:])
        v_mass=np.float32(.5)*(state.v[:,:-1,:]+state.v[:,1:,:])
    if getattr(state, "_moving_frame_velocity", None):
        from woof.moving_frame import ground_relative_components
        u_mass,v_mass=ground_relative_components(state,u_mass,v_mass)
    source={"th":th,"rho":rho,"pii":pii,"dz8w":dz,"p":state.p,
            "w":state.w,"u":u_mass,"v":v_mass,
            "th_old":state.th_old,"qv_old":state.qv_old,"xland":driver.fields["xland"]}
    for name in ("qv","qc","qr","qi","qs","qg"):
        source[name]=getattr(state,name)
    for name,target in NUMBER_MAP.items(): source[name]=getattr(state,target)
    for name in SURFACE_FIELDS[1:]:
        source[name]=state.scratch((ny,nx),_SURFACE_SLOTS[name])
        previous=getattr(getattr(driver,"microphysics",None),name,None)
        if name.endswith("ncv") or previous is None: source[name].fill(0)
        else: source[name][:]=previous
    packed={}
    for name in STATE_FIELDS:
        if name=="chem":
            raw=state.scratch((143,ny+2,nz+1,nx+2),"fsbm_frame_chem")
            raw.fill(0)
            raw[1:,1:ny+1,:nz,1:nx+1]=state.fast_sbm_bins.transpose(0,2,1,3)
            packed[name]=raw.transpose(3,2,1,0)
        elif name in SURFACE_FIELDS:
            raw=state.scratch((ny+2,nx+2),"fsbm_frame_"+name)
            raw.fill(0); raw[1:ny+1,1:nx+1]=source[name]
            packed[name]=raw.T
        else:
            raw=state.scratch((ny+2,nz+1,nx+2),"fsbm_frame_"+name)
            raw.fill(0)
            a=source[name]; levels=a.shape[0]
            raw[1:ny+1,:levels,1:nx+1]=a.transpose(1,0,2)
            packed[name]=raw.transpose(2,1,0)
    # Global mass cells use WRF indices 1..nx and 1..ny. The specified
    # ring is excluded here, so its spectral carriers never enter CGS.
    width=max(int(cfg.spec_zone),0) if cfg.specified or cfg.nested else 0
    dims=(1,nx+1,1,ny+1,1,nz+1,0,nx+1,0,ny+1,1,nz+1,
          width+1,nx-width,width+1,ny-width,1,nz)
    prior=getattr(state,"fast_sbm_itimestep",None)
    if prior is None: prior=1 if float(getattr(state,"elapsed_seconds",0.))>0 else 0
    step=int(prior)+1
    save_pre_mp_theta(state)
    launch(step,dt,cfg.dx,cfg.dy,dims,packed,state.scratch)
    state.fast_sbm_itimestep=step
    state.fast_sbm_bins[:]=packed["chem"][1:nx+1,:nz,1:ny+1,1:].transpose(3,1,2,0)
    for name in ("qv","qc","qr","qi","qs","qg","th_old","qv_old"):
        source[name][:]=packed[name][1:nx+1,:nz,1:ny+1].transpose(1,2,0)
    th[:]=packed["th"][1:nx+1,:nz,1:ny+1].transpose(1,2,0)
    for name in NUMBER_MAP:
        source[name][:]=packed[name][1:nx+1,:nz,1:ny+1].transpose(1,2,0)
    for name in SURFACE_FIELDS[1:]:
        source[name][:]=packed[name][1:nx+1,1:ny+1].T
    moist_physics_finish(state,cfg,th,dt)
    sr=state.scratch((ny,nx),"mp_sr")
    sr[:]=frozen_fraction(source["rainncv"],source["snowncv"],source["graupelncv"])
    return MicrophysicsDiagnostics(rainnc=source["rainnc"],rainncv=source["rainncv"],sr=sr,
         snownc=source["snownc"],snowncv=source["snowncv"],graupelnc=source["graupelnc"],graupelncv=source["graupelncv"])
