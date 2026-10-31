"""CUDA launcher for the pinned CCPP TKE-EDMF column.
This module supplies the column core. Forecast admission requires the carried
TKE tracer, surface seam and registry metadata as well.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
from woof.core.device_cache import cuda_cache
from woof.core.kernels import load_module

_INPUTS=("u","v","t","q","swh","hlw","prsi","prsl","prslk","phii","phil","sfc")
WORKSPACE_BUDGET_BYTES=512*1024*1024

def _table_manifest():
    from woof import data_assets
    path=data_assets.data_path("satmedmf/manifest.json")
    from woof.tables import table_path
    return json.loads(table_path("satmedmf", path).read_text(encoding="utf-8"))

def build_table():
    """Generate with WOOF math and require the manifest's exact bytes."""
    import cupy as cp
    manifest=_table_manifest()
    table=cp.empty(manifest["elements"],dtype=cp.float64)
    load_module("satmedmf").get_function("satmedmf_fpvs_table")((30,),(256,),(table,))
    digest=hashlib.sha256(table.get().tobytes()).hexdigest()
    if digest!=manifest["sha256"]:
        raise ValueError(f"SATMEDMF fpvs table hash mismatch: {digest}, expected {manifest['sha256']}; wrong saturation pressures would change cloud buoyancy and tendencies")
    return table

@cuda_cache(maxsize=None,ready=True)
def _table():
    import cupy as cp
    import os,tempfile
    from woof.table_assets import generated_table_root
    from woof.core.thompson_contract import TableAsset,read_validated_table_asset
    manifest=_table_manifest()
    root=generated_table_root("satmedmf")
    asset=TableAsset("fpvs-f64.bin",manifest["bytes"],manifest["sha256"])
    path=root/asset.filename
    if not path.exists():
        import sys
        print(f"woof: SATMEDMF first use: generating {path} (size + SHA-256 verified)", file=sys.stderr, flush=True)
        raw=build_table().get().tobytes()
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root,prefix="fpvs-",suffix=".tmp",delete=False) as f:
            temporary=Path(f.name);f.write(raw)
        os.replace(temporary,path)
    from woof.tables import CorruptTableAssets
    try:
        raw=read_validated_table_asset(root,asset)
    except (OSError,ValueError) as error:
        raise CorruptTableAssets(f"SATMEDMF needs {path}: expected {asset.bytes} bytes, SHA-256 {asset.sha256}; run woof fetch-tables --schemes satmedmf --from DIR --replace-damaged") from error
    return cp.asarray(np.frombuffer(raw,dtype="<f8"))


def _validated(inputs,dt):
    if set(inputs)!=set(_INPUTS):
        raise ValueError("SATMEDMF column input fields differ from the pinned interface")
    if not math.isfinite(dt) or dt<=0:
        raise ValueError("SATMEDMF dt must be positive and finite")
    n,k=inputs["u"].shape
    nt=inputs["q"].shape[2]
    if not 5<=nt<=64:
        raise ValueError("SATMEDMF requires 5 to 64 tracers; the column layout includes four water slots and terminal TKE")
    if n<1 or k<8:
        raise ValueError("SATMEDMF needs at least one column and eight levels; shallower columns cannot supply the mass-flux searches")
    shapes={name:(n,k) for name in ("u","v","t","swh","hlw","prsl","prslk","phil")}
    shapes.update(q=(n,k,nt),prsi=(n,k+1),phii=(n,k+1),sfc=(n,16))
    for name in _INPUTS:
        if inputs[name].shape!=shapes[name] or inputs[name].dtype!=np.float64:
            raise ValueError(f"SATMEDMF {name}: expected float64 {shapes[name]}")
    return n,k,nt

def run_device(inputs,*,dt=60.0,strict=None):
    """Launch independent columns, keeping forecast arrays on their device."""
    import cupy as cp
    n,k,nt=_validated(inputs,dt)
    table=_table()
    shapes={"du":(n,k),"dv":(n,k),"tdt":(n,k),"rtg":(n,k,nt),"hpbl":(n,),
            "dkt":(n,k),"dku":(n,k),"sflux":(n,4)}
    out={key:cp.empty(shape,dtype=cp.float64,order="F") for key,shape in shapes.items()}
    out["kpbl"]=cp.empty(n,dtype=cp.int32)
    capacity=(220+16*nt)*k+4096
    chunk=max(1,WORKSPACE_BUDGET_BYTES//(capacity*8+((20+2*nt)*k+23)*8))
    kernel=load_module("satmedmf").get_function("satmedmf_column")
    # Default and strict arms use the same rounding-pinned column arithmetic.
    for first in range(0,n,chunk):
        last=min(n,first+chunk);nc=last-first
        packed=cp.asfortranarray(cp.concatenate([inputs[key][first:last].reshape(nc,-1,order="F") for key in _INPUTS],axis=1))
        packed_out=cp.empty((nc,(5+nt)*k+5),dtype=cp.float64,order="F")
        kpbl=cp.empty(nc,dtype=cp.int32)
        workspace=cp.empty((nc,capacity),dtype=cp.float64)
        highwater=cp.empty(nc,dtype=cp.int32)
        kernel(((nc+31)//32,),(32,),
               (np.int32(nc),np.int32(k),np.int32(nt),np.float64(dt),packed,packed_out,kpbl,workspace,table,highwater))
        if int(highwater.max().get())>capacity:
            raise ValueError("SATMEDMF workspace overflow would corrupt adjacent columns")
        offset=0
        for key,shape in shapes.items():
            size=int(np.prod(shape[1:])) if len(shape)>1 else 1
            out[key][first:last]=packed_out[:,offset:offset+size].reshape((nc,*shape[1:]),order="F")
            offset+=size
        out["kpbl"][first:last]=kpbl
    return out

def run_columns(inputs,*,dt=60.0,strict=True):
    """Qualification entry: transfer independent fixtures and return host results."""
    import cupy as cp
    _validated(inputs,dt)
    device={key:cp.asarray(inputs[key],order="F") for key in _INPUTS}
    return {key:value.get() for key,value in run_device(device,dt=dt,strict=strict).items()}

def bind_physics_columns(atmosphere,fields,state,cfg,*,shortwave,longwave):
    """Bind WOOF mass arrays to the pinned GFS bottom-to-top column interface."""
    import cupy as cp
    nz,ny,nx=state.p.shape; n=ny*nx
    col=lambda a: cp.asfortranarray(a.astype(cp.float64).reshape(a.shape[0],n).T)
    surf=lambda a: cp.asarray(a,dtype=cp.float64).reshape(n)
    rv=col(atmosphere["qv"])
    denom=1.0+rv
    from woof.core.moist import extra_moist_species
    base=("qv","qc","qr","qi")
    names=base+tuple(name for name in extra_moist_species(state) if name not in base and name!="tke_edmf")
    q=cp.empty((n,nz,len(names)+1),dtype=cp.float64,order="F")
    for j,name in enumerate(names):
        source=getattr(state,name,None)
        q[:,:,j]=0.0 if source is None else col(source)/denom
    q[:,:,-1]=col(state.tke_edmf)
    p=col(atmosphere["pressure"])
    pi=col(atmosphere["p_interface"])
    exner=cp.empty_like(p)
    psk=cp.empty(n,dtype=cp.float64)
    kernel=load_module("satmedmf").get_function("satmedmf_exner")
    kernel(((p.size+255)//256,),(256,),(p,pi,exner,psk,np.int32(n),np.int32(nz)))
    zi=col(atmosphere["z_interface"]); zi=zi-zi[:,0,None]
    zm=0.5*(zi[:,:-1]+zi[:,1:])
    rho=surf(atmosphere["rho"][0])
    rough=surf(fields["znt"]); green=surf(fields["vegfra"])*0.01
    zlo=float(np.float32(0.1)); zup=1.0
    rough_fraction=cp.clip((rough-zlo)/(zup-zlo),0.0,1.0)
    zvfun=cp.sqrt(rough_fraction*cp.maximum(green,0.1))
    zvfun=cp.where(surf(fields["xland"])<1.5,zvfun,0.0)
    sfc=cp.empty((n,16),dtype=cp.float64,order="F")
    sfc[:,0]=psk;sfc[:,1]=surf(fields["br"]);sfc[:,2]=100.0*rough
    sfc[:,3]=surf(fields["tsk"]);sfc[:,4]=surf(fields["hfx"])/(rho*1004.6)
    sfc[:,5]=surf(fields["qfx"])/rho;sfc[:,6]=surf(fields["ust"])**2
    sfc[:,7]=surf(fields["wspd"]);sfc[:,8]=surf(fields["u10"]);sfc[:,9]=surf(fields["v10"])
    sfc[:,10]=surf(fields["fm"]);sfc[:,11]=surf(fields["fh"]);sfc[:,12]=1.0
    sfc[:,13]=float(cfg.dx)*float(cfg.dy)/(surf(state.msft)**2)
    sfc[:,14]=zvfun;sfc[:,15]=green
    return {"u":col(atmosphere["u"]),"v":col(atmosphere["v"]),"t":col(atmosphere["temperature"]),
       "q":q,"swh":col(shortwave)*exner,
       "hlw":col(longwave)*exner,
       "prsi":pi,"prsl":p,"prslk":exner,"phii":zi*9.80665,"phil":zm*9.80665,"sfc":sfc}

def recover_physics_rates(inputs,out,shape,dt,*,scalar_names=()):
    """Convert a finite GFS humidity increment back to WOOF dry-air ratios."""
    import cupy as cp
    unpack=lambda a: cp.ascontiguousarray(a.T.reshape(shape),dtype=cp.float32)
    old=inputs["q"][:,:,:-1];new=old+float(dt)*out["rtg"][:,:,:-1]
    den_old=1.0-old[:,:,0];den_new=1.0-new[:,:,0]
    if bool(cp.any(den_new<=0)):
        raise FloatingPointError("CCPP humidity increment reached unit specific humidity; dry-air mixing ratios would divide by zero")
    rate=(new/den_new[:,:,None]-old/den_old[:,:,None])/float(dt)
    result={"du":unpack(out["du"]),"dv":unpack(out["dv"]),
            "dtheta":unpack(out["tdt"]/inputs["prslk"]),
            "dtke_edmf":unpack(out["rtg"][:,:,-1])}
    for j,key in enumerate(("dqv","dqc","dqr","dqi")):result[key]=unpack(rate[:,:,j])
    for j,name in enumerate(scalar_names,4):result["d_"+name]=unpack(rate[:,:,j])
    return result
