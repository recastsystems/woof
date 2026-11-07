"""Complete resident WRF v4 CARBON/CO2FLUX, without host state staging."""
import numpy as np

CARBON_PARAMETERS=("sla","folnmx","arm","rmf25","rmr25","rms25","wdpool","wrrat","ltovrc","tdlef","dilefw","dilefc","fragr","mrp","eblforest")
CARBON_POOLS=("lfmass","rtmass","stmass","wood","stblcp","fastcp")
CARBON_STATE_2D=tuple(name+"xy" for name in CARBON_POOLS)
CARBON_DIAGNOSTICS_2D=("neexy","gppxy","nppxy")
OUTPUTS=("lai","sai",*CARBON_POOLS,"gpp","npp","nee","autors","heters","totsc","totlb")

def dynamic(dveg):return int(dveg) in (2,5,6)

def evaluate_carbon_slab(s):
    import cupy as cp
    from woof.core.noahmp_kernel_sources import selected_kernel
    from woof.core.noahmp_slab_libm import copy_slab_slots,split_slab_slots
    n=int(s["dt"].size)
    scalars=("dt","tv","psn","foln","btran","igs","lai","sai",*CARBON_POOLS,
        "sla","folnmx","arm","rmf25","rmr25","rms25","wdpool","wrrat","ltovrc","tdlef","dilefw","dilefc","fragr","tmin","mrp")
    x=cp.empty((n,47),cp.float32);ix=cp.empty((n,6),cp.int32);y=cp.empty((n,15),cp.float32)
    blocks=[(k,s[name]) for k,name in enumerate(scalars)]
    blocks.extend(((29,s["smc"]),(33,s["smcmax"]),(37,s["dzsnso"][:,3:]),(41,s["zsoil"]),(45,s["stc"][:,3]),(46,s["eblforest"])))
    copy_slab_slots(x,blocks)
    copy_slab_slots(ix,[(k,s[name]) for k,name in enumerate(("vegtyp","nroot","iswater","isbarren","isice","urban_flag"))])
    if n:selected_kernel("noahmp_leaves","noahmp_carbon_flux")(((n+127)//128,),(128,),(x,ix,y,np.int32(n)))
    return split_slab_slots(y,{name:(k,1) for k,name in enumerate(OUTPUTS)})
