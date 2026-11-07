"""Resident WRF-v4 crop state, table transfer and complete crop carbon."""
import numpy as np
SCALARS=('q10mr', 'foln_mx', 'lfmr25', 'rtmr25', 'stmr25', 'grainmr25', 'fra_gr', 'lefreez', 'mrp', 'bio2lai', 'gddtbase', 'gddtcut', 'gdds1', 'gdds2', 'gdds3', 'gdds4', 'gdds5', 'pltday', 'hsday')
STAGES=('lfpt', 'stpt', 'rtpt', 'grainpt', 'lf_ovrc', 'rt_ovrc', 'st_ovrc', 'dile_fw', 'dile_fc', 'lfct', 'stct', 'rtct')
STATE=("grainxy","gddxy")
STATIC=("planting","harvest","season_gdd")
INTEGER=("cropcat",)

def apply_parameters(params,s):
    import cupy as cp
    from woof.core.noahmp_slab_libm import take_slab_rows,copy_slab_slots
    key=cp.cuda.runtime.getDevice()
    pools=params.__dict__.setdefault("_device_crop_pools",{})
    if key not in pools:
        group=params.bundle.mptable["noahmp_crop_parameters"]
        rows=[]
        for cat in range(1,6):
            rows.append([np.float32(0.0 if n=="mrp" else group.values[n.upper()][cat-1])for n in SCALARS]+[np.float32(group.values[n.upper()+"_S"+str(k)][cat-1])for n in STAGES for k in range(1,9)])
        photos=("C3PSN","KC25","AKC","KO25","AKO","AVCMX","VCMX25","BP","MP","FOLNMX","QE25")
        pools[key]=(cp.asarray(np.array(rows,np.float32)),{n:cp.asarray(np.array(group.values[n],np.float32))for n in photos})
    table,photos=pools[key];cat=s["croptype"];index=cp.minimum(cp.maximum(cat,1),5)-1;active=cat>0
    pars=cp.ascontiguousarray(table[index]);
    s["crop_pars"]=pars
    # Native TRANSFER_MP_PARAMETERS overrides crop photosynthesis before ENERGY.
    for name,values in photos.items():
        if name in s:s[name]=cp.where(active,values[index],s[name])
    # Driver 991-998 replaces crop dates and scales its five GDD thresholds.
    slots=[(SCALARS.index("mrp"),s["mrp"]),(SCALARS.index("pltday"),s["planting"].astype(cp.int32).astype(cp.float32)),(SCALARS.index("hsday"),s["harvest"].astype(cp.int32).astype(cp.float32))]
    scale=s["season_gdd"]/np.float32(1770.)
    for name in ("gdds1","gdds2","gdds3","gdds4","gdds5"):
        k=SCALARS.index(name);slots.append((k,scale*pars[:,k]))
    copy_slab_slots(pars,slots)

def evaluate_crop(s,base):
    import cupy as cp
    from woof.core.noahmp_kernel_sources import selected_kernel
    from woof.core.noahmp_slab_libm import copy_slab_slots,split_slab_slots
    from woof.core.noahmp_carbon_slab import CARBON_POOLS,OUTPUTS
    n=s["dt"].size
    names=("dt","tv","psn","foln","btran","julian","soldn","t2m","lai","sai",*CARBON_POOLS,"grain","gdd")
    x=cp.empty((n,44),cp.float32);ix=cp.empty((n,8),cp.int32);y=cp.empty((n,17),cp.float32);yi=cp.empty(n,cp.int32)
    copy_slab_slots(x,[(k,s[name])for k,name in enumerate(names)]+[(18,s["zsoil"]),(22,s["dzsnso"]),(29,s["stc"]),(36,s["smc"]),(40,s["smcmax"])])
    copy_slab_slots(ix,[(k,s[name])for k,name in enumerate(("vegtyp","nroot","iswater","isbarren","isice","urban_flag","croptype","pgs"))])
    if n:selected_kernel("noahmp_leaves","noahmp_crop_flux")(((n+127)//128,),(128,),(x,ix,s["crop_pars"],y,yi,np.int32(n)))
    result=split_slab_slots(y,{name:(k,1)for k,name in enumerate((*OUTPUTS,"grain","gdd"))})
    active=s["croptype"]>0
    for name,value in base.items():result[name]=cp.where(active,result[name],value)
    result["pgs"]=yi
    return result

def initialize(fields,params):
    import cupy as cp
    # Native NOAHMP_INIT 2150-2233, after the common vegetated initialization.
    land=(fields["xland"]<np.float32(1.5))&(fields["xice"]<np.float32(params.xice_threshold))
    v=fields["ivgtyp"];identity=params.land_use
    vegetation=land&(v!=identity.iswater)&(v!=identity.isbarren)&(v!=identity.isice)&(v!=identity.isurban)
    for lc in identity.lcz:vegetation &= v!=lc
    a=fields["croptype"];corn=(a[0]>a[1])&(a[0]>a[2])&(a[0]>a[3]);soy=(a[1]>a[0])&(a[1]>a[2])&(a[1]>a[3]);crop=vegetation&(a[4]>=np.float32(.5))
    default=int(params.bundle.mptable["noahmp_crop_parameters"].scalar("DEFAULT_CROP"))
    fields["cropcat"][...]=cp.where(vegetation,cp.where(crop&corn,1,cp.where(crop&soy,2,default)),0).astype(cp.int32)
    fields["grainxy"].fill(np.float32(1.e-10));fields["gddxy"].fill(0)
    fields["rtmassxy"][...]=cp.where(crop,np.float32(0),fields["rtmassxy"])
    fields["woodxy"][...]=cp.where(crop,np.float32(0),fields["woodxy"])
    factor=cp.where(corn,np.float32(.015),cp.where(soy,np.float32(.030),np.float32(.035)))
    fields["lfmassxy"][...]=cp.where(crop,fields["lai"]/factor,fields["lfmassxy"])
    fields["stmassxy"][...]=cp.where(crop,fields["xsaixy"]/np.float32(.003),fields["stmassxy"])

def ingest_statics(fields,values):
    """Validate official WRF crop geography once before NOAHMP_INIT."""
    import cupy as cp
    required=("croptype",*STATIC)
    if values is None or any(n not in values for n in required):
        raise ValueError("Noah-MP crop initialization requires CROPTYPE, PLANTING, HARVEST and SEASON_GDD geography")
    for n in required:
        a=np.asarray(values[n],np.float32)
        if a.shape!=fields[n].shape or not np.isfinite(a).all():raise ValueError(f"Noah-MP {n} crop geography has invalid shape or nonfinite data")
        if n=="croptype" and np.any((a<0)|(a>1)):raise ValueError("CROPTYPE fractions must be in 0..1")
        fields[n][...]=cp.asarray(a)
