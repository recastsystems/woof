"""Opt-in WRF 4.7.1 GECROS with the missing thermal emergence call.

Native WRF 4.7.1 fails here; this fixes it, not bitwise to WRF.
A cold start represents sowing. Native EMERG, using the native crop-specific
parameters and top-soil temperature, starts growth after its thermal threshold.
Warm emerged states enter the unchanged GECROS equations. All sixty state words
are resident carriers, including native reserved words 59/60.
"""
from functools import lru_cache
import numpy as np
STATE=tuple(f"gecros{k:02d}xy" for k in range(1,61))
LABEL="native WRF 4.7.1 fails here; this fixes it, not bitwise to WRF"

@lru_cache(None)
def _module_on(device,profile):
    from woof.core.noahmp_kernel_sources import compile_runtime_unit
    from woof.core.noahmp_vegeflux_gpu import _constant_tables,_copy_constant
    m=compile_runtime_unit("noahmp_gecros",module_key="woof.core.noahmp_gecros:"+profile,atan_profile=profile)
    for name,value in _constant_tables().items():_copy_constant(m,name,value)
    return m

def module(profile="glibc_2_39"):
    import cupy as cp
    return _module_on(cp.cuda.runtime.getDevice(),profile)

def initialize(fields,params,latitude):
    import cupy as cp
    from woof.core.noahmp_slab_libm import split_slab_slots
    shape=fields["lai"].shape;n=fields["lai"].size
    lat=cp.ascontiguousarray(cp.broadcast_to(cp.asarray(latitude,dtype=cp.float32),shape).ravel())
    gs=cp.zeros((n,60),cp.float32)
    if n:module(params.land_options.get("atan_profile","glibc_2_39")).get_function("k_gecros_init")(((n+63)//64,),(64,),(lat,gs,np.int32(n)))
    for name,a in split_slab_slots(gs,{name:(k,1)for k,name in enumerate(STATE)}).items():fields[name][...]=a.reshape(shape)
    land=(fields["xland"]<np.float32(1.5))&(fields["xice"]<np.float32(params.xice_threshold))
    v=fields["ivgtyp"];identity=params.land_use
    for excluded in (identity.iswater,identity.isbarren,identity.isice,identity.isurban,*identity.lcz):land &= v!=excluded
    a=fields["croptype"];crop=land&(a[4]>=np.float32(.5))
    fields["cropcat"][...]=cp.where(crop,cp.where(a[0]>a[2],2,cp.where(a[2]>0,1,0)),0).astype(cp.int32)
    if bool(cp.any((fields["cropcat"] > 0) & (cp.abs(lat.reshape(shape)) > 67))):
        raise ValueError(LABEL + ". Native GECROS ASTRO stops outside 67 degrees latitude; the selected crop geography crosses that limit.")

def evaluate(fields,n,active,gs,extra,*,atan_profile="glibc_2_39"):
    import cupy as cp
    from woof.core import noahmp_flux_slab as f
    from woof.core import noahmp_vegeflux_gpu as v
    from woof.core.noahmp_slab_libm import copy_slab_slots
    packed=f.pack_vege_flux_slab(fields,n)
    out=cp.empty((n,v.N_OUTPUT),cp.float32);gx=cp.empty((n,15),cp.float32)
    copy_slab_slots(gx,[(k,extra[name])for k,name in enumerate(("julian","swdown","prcp","fb","croptype"))]+[(5,extra["sh2o"]),(9,extra["smcwlt"]),(13,fields.get("thair",fields["sfctmp"])),(14,fields.get("CZIL",cp.zeros(n,cp.float32)))])
    fsno=cp.ascontiguousarray(cp.broadcast_to(cp.asarray(fields.get("fsno",0),dtype=cp.float32),(n,)))
    stcopt=int(fields.get("opt_stc",1));crsopt=int(fields.get("opt_crs",1));sfcopt=int(fields.get("opt_sfc",1))
    m=module(atan_profile)
    # Restore the missing call at the driver boundary. Warm active states pass
    # through without any arithmetic or lifecycle changes.
    if n:
        m.get_function("k_gecros_emerge")(((n+63)//64,),(64,),(gs,cp.ascontiguousarray(extra["croptype"],dtype=cp.int32),packed.stc,cp.ascontiguousarray(fields["dt"]),np.int32(n)))
        m.get_function("k_gecros_flux")(((n+63)//64,),(64,),(*packed,fsno,out,np.int32(n),cp.ascontiguousarray(active,dtype=cp.bool_),np.int32(stcopt),np.int32(crsopt),np.int32(sfcopt),gs,gx))
        m.get_function("k_gecros_harvest")(((n+63)//64,),(64,),(gs,cp.ascontiguousarray(extra["croptype"],dtype=cp.int32),cp.ascontiguousarray(fields["dt"]),np.int32(n)))
    return f._unpack(cp,out,v.OUTPUT_NAMES)
