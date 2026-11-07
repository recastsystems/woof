"""Native v4 depth-category and SR2006 soil parameters, resident on the GPU.

WRF Noah-MP 848f54ad, drivers/wrf/module_sf_noahmpdrv.F:745-754,
977-985,1712-1826. Original public-domain notice is retained in licenses/.
The driver supplies organic matter zero; missing sand/clay uses WRF's own
0.41/0.18 fallback. No MPAS/Noah-MP v5 implementation is used.
"""
import numpy as np

SOIL_PROPERTIES=("smcwlt","smcref","smcmax","bexp","psisat","dksat",
                 "dwsat","smcdry","quartz")
# WRF v4.7.1 e5c08598 driver:933-943 names these direct inputs but leaves
# them commented and SOILTYP unassigned. Initialize ancillary category values
# as in option 3, then replace these nine vectors after TRANSFER. Retain the
# category KDT/FRZX and urban override, exactly as the native pedotransfer arm.
DIRECT_SOIL_INPUTS = tuple(name.upper()+"_3D" for name in SOIL_PROPERTIES)
SOIL4_LABEL = "native WRF 4.7.1 fails here; this fixes it, not bitwise to WRF"
COEFFICIENT_GROUPS=(("theta_1500t","abcdefg"),("theta_1500","ab"),
    ("theta_33t","abcdefg"),("theta_33","abc"),
    ("theta_s33t","abcdefg"),("theta_s33","ab"),
    ("psi_et","abcdefg"),("psi_e","abc"),("smcmax","ab"))


def active_soil_inputs(opt_soil):
    return ({2:tuple(f"SOILCL{k}" for k in range(1,5)),
             3:("SOILCOMP",),4:DIRECT_SOIL_INPUTS}.get(int(opt_soil),()))


def soil_input_shapes(opt_soil,shape):
    return {name.lower():((8,*shape) if name=="SOILCOMP" else
                         (4,*shape) if name in DIRECT_SOIL_INPUTS else tuple(shape))
            for name in active_soil_inputs(opt_soil)}


def pedotransfer_coefficients(params):
    """Upload the actual pinned optional table once per device."""
    import cupy as cp
    key=cp.cuda.runtime.getDevice()
    cache=params.__dict__.setdefault("_pedotransfer_coefficients",{})
    if key not in cache:
        cache[key]=cp.asarray([np.float32(params.table(
            "noahmp_optional_parameters",f"sr2006_{group}_{letter}"))
            for group,letters in COEFFICIENT_GROUPS for letter in letters])
    return cache[key]


def pedotransfer(sand,clay,orgm,coefficients):
    """The full native routine; inputs are fractions, arrays (columns,4).

    Returns all nine native soil properties and the three INOUT vectors.
    KDT/FRZX remain the previously transferred native category values.
    """
    import cupy as cp
    from woof.core.noahmp_kernel_sources import selected_kernel
    arrays=tuple(cp.ascontiguousarray(x,dtype=cp.float32) for x in (sand,clay,orgm))
    if len({a.shape for a in arrays})!=1 or len(arrays[0].shape)!=2 or arrays[0].shape[1]!=4:
        raise ValueError("native pedotransfer requires equal four-layer vectors")
    n=arrays[0].shape[0]
    y=cp.empty((n,9,4),dtype=cp.float32)
    updated=cp.empty((n,3,4),dtype=cp.float32)
    if n:
        selected_kernel("noahmp_leaves","noahmp_pedotransfer_sr2006")(
            ((n*4+127)//128,),(128,),
            (*arrays,coefficients,y,updated,np.int32(n)))
    return {**{name:y[:,k,:] for k,name in enumerate(SOIL_PROPERTIES)},
            **{name:updated[:,k,:] for k,name in enumerate(("sand","clay","orgm"))}}


def select_soil_parameters(params,fields,layout,chunk,start,stop):
    """Select current per-depth categories or perform current pedotransfer.

    Every input identity is gathered from the current device fields. There is
    no host category enumeration or host synchronization in the step loop.
    """
    import cupy as cp
    soil=params.land_options.get("opt_soil",1)
    if soil==1:return
    if soil==4:
        direct_soil_parameters(chunk,layout["active"][start:stop])
        return
    if soil==3:
        composition=chunk.pop("soilcomp")
        values=pedotransfer(composition[:,:4]*np.float32(.01),
            composition[:,4:]*np.float32(.01),cp.zeros((stop-start,4),cp.float32),
            pedotransfer_coefficients(params))
        natural=~chunk["urban_flag"].astype(cp.bool_)
        for name in SOIL_PROPERTIES:
            if name in chunk:
                chunk[name]=cp.where(natural[:,None],values[name],chunk[name])
        return
    if soil!=2:raise ValueError("unknown native soil selector")
    from woof.core.noahmp_surface_device import parameter_pool
    from woof.core import health_ledger
    pool,lookup,nv,ns=parameter_pool(params,4)
    key=cp.cuda.runtime.getDevice()
    cache=params.__dict__.setdefault("_depth_parameter_tables",{})
    if key not in cache:
        from woof.core.noahmp import _SOIL_TOP_FIELDS,_TILEDRAIN_SCALAR_FIELDS,_TILEDRAIN_INTEGER_FIELDS
        top={name.lower() for name in (*_SOIL_TOP_FIELDS,*_TILEDRAIN_SCALAR_FIELDS,
                                       *_TILEDRAIN_INTEGER_FIELDS,"KDT","FRZX")}
        names=tuple(name for name in pool if name in SOIL_PROPERTIES or name in top)
        nr=next(iter(pool.values())).shape[0]
        widths=tuple(int(pool[name].size//nr) for name in names)
        offsets=tuple(sum(widths[:k]) for k in range(len(names)))
        assert all(w in (1,4) for w in widths)
        packed=cp.concatenate([pool[name].reshape(nr,-1).view(cp.float32)
                               for name in names],axis=1)
        descriptor=cp.asarray(list(zip(offsets,widths)),cp.int32)
        cache[key]=(names,widths,offsets,packed,descriptor)
    names,widths,offsets,packed,descriptor=cache[key]
    n=stop-start;record_width=sum(widths)
    out=cp.empty(n*record_width,cp.float32)
    flags=cp.zeros(1,cp.uint32)
    if n:
        from woof.core.noahmp_kernel_sources import selected_kernel
        selected_kernel("noahmp_leaves","noahmp_depth_soil_parameters")(
            ((n+127)//128,),(128,),(
                *(fields[f"soilcl{k}"] for k in range(1,5)),fields["xice"],
                layout["vegtyp"],layout["gather_j"][start:stop],layout["gather_i"][start:stop],
                layout["active"][start:stop],lookup,packed,descriptor,out,flags,
                np.int32(n),np.int32(fields["xice"].shape[1]),np.int32(nv),np.int32(ns),
                np.int32(len(names)),np.int32(record_width)))
    def refuse(bits):
        if bits:raise ValueError("Noah-MP SOILCL1-4 has an invalid category or native-unassigned FRZX row")
    ledger=health_ledger.active()
    if ledger is None:refuse(int(flags.item()))
    else:ledger.record("noahmp.soil.depth_categories",flags,refuse)
    for name,w,off in zip(names,widths,offsets):
        value=out[off*n:(off+w)*n]
        chunk[name]=(value if w==1 else value.reshape(n,w)).view(pool[name].dtype)


def direct_soil_parameters(chunk, active):
    """Complete the documented direct-property arm without changing the LSM.

    The resident gather has already produced (columns,4) arrays. No default
    substitution or clipping is allowed. Ancillary KDT/FRZX and the urban
    hardcoding follow the existing native option-3 contract.
    """
    import cupy as cp
    from woof.core import health_ledger
    natural = ~chunk["urban_flag"].astype(cp.bool_)
    values = {name:chunk.pop(name+"_3d") for name in SOIL_PROPERTIES}
    invalid = cp.zeros(active.shape,cp.bool_)
    for name,value in values.items():
        invalid |= cp.any(~cp.isfinite(value),axis=1)
        if name in ("bexp","psisat","dksat","dwsat"):
            invalid |= cp.any(value<=0,axis=1)
    dry,wilt,ref,sat = (values[n] for n in ("smcdry","smcwlt","smcref","smcmax"))
    invalid |= cp.any((dry<0)|(dry>wilt)|(wilt>=ref)|(ref>=sat)|(sat>1),axis=1)
    quartz=values["quartz"]
    invalid |= cp.any((quartz<0)|(quartz>1),axis=1)
    flags=cp.any(invalid & natural & active.astype(cp.bool_)).astype(cp.uint32).reshape(1)
    def refuse(bits):
        if bits:
            raise ValueError("opt_soil=4 requires finite direct soil properties: "
                             "0<=SMCDRY<=SMCWLT<SMCREF<SMCMAX<=1, positive "
                             "BEXP/PSISAT/DKSAT/DWSAT, 0<=QUARTZ<=1; "+SOIL4_LABEL)
    ledger=health_ledger.active()
    if ledger is None:refuse(int(flags.item()))
    else:ledger.record("noahmp.soil.direct_properties",flags,refuse)
    for name in SOIL_PROPERTIES:
        if name in chunk:
            chunk[name]=cp.where(natural[:,None],values[name],chunk[name])
