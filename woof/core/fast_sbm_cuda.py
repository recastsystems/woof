"""Device column entry for FAST_SBM, using packaged initialiser tables.

No compiler, oracle fixture or host reference is needed at forecast time.
All mutable workspace is supplied by the state scratch allocator.
"""
from functools import lru_cache
from pathlib import Path
import hashlib,json
import numpy as np
from woof.core.fast_sbm_contract import (CUDA_OPTIONS,FLAG_FIELDS,TABLE_ORDER,
                                        EXTRA_TABLES,STATE_FIELDS,corrected_flags)

KERNEL_DIR=Path(__file__).parent/"kernels"
DATA=Path(__file__).parents[1]/"data/fast_sbm"

@lru_cache(maxsize=None)
def module(device):
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    options=CUDA_OPTIONS+(f"-I{KERNEL_DIR}",)
    with cp.cuda.Device(device):
        ptx,_=compiler.compile_using_nvrtc((KERNEL_DIR/"fast_sbm.cu").read_text(),
                                         options,None,"fast_sbm.cu")
        result=cp.cuda.function.Module()
        result.load(ptx.encode() if isinstance(ptx,str) else ptx)
    return result


@lru_cache(maxsize=None)
def tables(device,precision="float32"):
    import cupy as cp
    from woof.tables import table_path
    path=table_path("fast-sbm", DATA/f"tables-{precision}.npz")
    manifest=json.loads(table_path("fast-sbm", DATA/"packed-manifest.json").read_text())
    expected=manifest["outputs"][path.name]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError(f"FAST_SBM initialiser table hash differs: {path}")
    dtype=np.float32 if precision=="float32" else np.float64
    with cp.cuda.Device(device),np.load(path,allow_pickle=False) as data:
        arrays=[cp.asarray(data[name],dtype=dtype) for name in TABLE_ORDER]
        for name in EXTRA_TABLES:
            key="ecoal_own" if name=="ecoalmassm" else name
            kind=np.int32 if name=="ima" else (dtype if name in ("dropradii","rccn") else np.float64)
            arrays.append(cp.asarray(data[key],dtype=kind))
        pointers=cp.asarray([a.data.ptr for a in arrays],dtype=np.uint64)
        flags=cp.asarray([corrected_flags()[name] for name in FLAG_FIELDS],dtype=np.int32)
        return tuple(arrays),pointers,np.int32(data["ikr_spon_break"][0]),flags


#: Threads per block for the column-local kernels (one column per thread).
#: One thread per block idled 31 of every warp's 32 lanes; a nested 3 km/1 km
#: tree spent most of each step waiting on FAST_SBM at that width. Column
#: arithmetic does not depend on it, so any width gives the same bytes.
COLUMN_BLOCK = 64


def column_launch(columns):
    """(grid, block) for ``columns`` independent columns."""
    columns=int(columns)
    return ((max(1,-(-columns//COLUMN_BLOCK)),),(COLUMN_BLOCK,))


def launch(step,dt,dx,dy,dims,fields,scratch):
    """Advance packed Fortran-layout float32 fields with corrected columns.

    ``scratch(shape,name,dtype)`` owns every work array. The fields stay on
    the device; only the compact failure vector crosses to the host.
    """
    import cupy as cp
    d=np.asarray(dims,dtype=np.int32)
    n=int((d[7]-d[6]+1)*(d[11]-d[10]+1)*(d[9]-d[8]+1))
    nz=int(d[17]-d[16]+1)
    nc=int((d[13]-d[12]+1)*(d[15]-d[14]+1))
    if nz<2 or nc<1: raise ValueError("FAST_SBM needs at least two levels and a nonempty active tile")
    for name in STATE_FIELDS:
        a=fields[name]
        if a.dtype!=np.float32 or not a.flags.f_contiguous:
            raise TypeError(f"FAST_SBM {name} must be Fortran-contiguous float32")
    dd=scratch((18,),"fsbm_dims",np.int32); dd[:]=cp.asarray(d)
    ap=scratch((len(STATE_FIELDS),),"fsbm_pointers",np.uint64)
    ap[:]=cp.asarray([fields[name].data.ptr for name in STATE_FIELDS],dtype=np.uint64)
    work=scratch((7*n+2*nc*(nz+2),),"fsbm_work",np.float32); work.fill(0)
    kz=scratch((nc,),"fsbm_cloud_base",np.int32); kz.fill(0)
    tally=scratch((nc,17),"fsbm_tally",np.float64); tally.fill(0)
    status=scratch((nc,),"fsbm_status",np.int32); status.fill(0)
    _arrays,tp,ikr,flags=tables(cp.cuda.runtime.getDevice())
    args=(np.int32(step),np.float32(dt),np.float32(dx),np.float32(dy),dd,tp,ikr,flags,
          ap,work,kz,tally,status)
    unit=module(cp.cuda.runtime.getDevice())
    halo_columns=(min(int(d[1])-1,int(d[13])+1)-max(int(d[12])-1,1)+1)*(min(int(d[3])-1,int(d[15])+1)-max(int(d[14])-1,1)+1)
    unit.get_function("fsbm_grid_fields_f32")(*column_launch(halo_columns),args)
    unit.get_function("fsbm_prepare_columns_f32")(*column_launch(nc),args)
    unit.get_function("fsbm_columns_f32")(*column_launch(nc),args)
    errors=cp.asnumpy(status)
    bad=np.flatnonzero(errors)
    if bad.size:
        raise RuntimeError(f"FAST_SBM stopped at WRF source line {int(errors[bad[0]])} in column {int(bad[0])}; the partial spectral tile cannot be accepted as a forecast step")
    return tally
