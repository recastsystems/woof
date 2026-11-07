#!/usr/bin/env python3
"""Development-only NWP window replay. No PSD hail or forecast claim."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from replay_levels import analytic


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import cupy as cp
    from woof import wrf_exact
    from woof.core import nwp_windows as nwp
    from woof.core.kernels import module_source
    if not wrf_exact.ENABLED:
        raise ValueError("start with GPUWM_WRF_EXACT=1")
    a=analytic()
    nz,ny,nx=a["t"].shape
    rng=np.random.default_rng(4712)
    arrays=dict(w=rng.uniform(-30,60,(nz+1,ny,nx)).astype("f4"),p8w=a["pw"],
                znw=np.array([1.,.9,.8,.7,.6,.5,.4,.2,0.],dtype="f4"),
                qg=rng.uniform(-1.e-7,5.e-3,(nz,ny,nx)).astype("f4"),
                rho=rng.uniform(.1,1.2,(nz,ny,nx)).astype("f4"),ph=a["ph"],phb=a["phb"],
                u10=rng.uniform(-20,25,(ny,nx)).astype("f4"),v10=rng.uniform(-25,15,(ny,nx)).astype("f4"),
                refl=rng.uniform(-35,60,(nz,ny,nx)).astype("f4"))
    # All upper-level reflectivities are larger than the lowest in this cell.
    arrays["refl"][0,0,0]=-12.
    arrays["refl"][1:,0,0]=70.
    device={k:cp.asarray(v) for k,v in arrays.items()}
    native=np.zeros((nx,ny,12),dtype="f4",order="F")
    actual=cp.zeros((12,ny,nx),dtype="f4")
    library=ctypes.CDLL(str(args.library.resolve()))
    library.nwp_window_ref.restype=None
    ptr=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    results=[]
    for adaptive in (False,True):
        native.fill(0.)
        actual.fill(0.)
        for step in range(1,6):
            dt=np.float32(20.)
            xtime=np.float32(np.float32(step)*dt/np.float32(60.))
            packed=np.zeros((nx,nz+1,ny,7),dtype="f4",order="F")
            for index,name in enumerate(("w","p8w","qg","rho","ph","phb","refl")):
                v=arrays[name]
                packed[:,:v.shape[0],:,index]=v.transpose(2,0,1)
            surface=np.asfortranarray(np.stack([arrays["u10"].T,arrays["v10"].T],axis=-1))
            library.nwp_window_ref(ctypes.c_int(nx),ctypes.c_int(ny),ctypes.c_int(nz),ptr(packed),
                                  ptr(arrays["znw"]),ptr(surface),ptr(native),ctypes.c_float(dt),
                                  ctypes.c_float(xtime),ctypes.c_int(1),ctypes.c_int(step),
                                  ctypes.c_int(adaptive),ctypes.c_int(20))
            nwp.update(*(device[k] for k in ("w","p8w","znw","qg","rho","ph","phb","u10","v10","refl")),
                       actual,dt=dt,xtime=xtime,history_minutes=1,step=step,adaptive=adaptive,max_time_step=20)
            expected=np.ascontiguousarray(native.transpose(2,1,0))
            observed=cp.asnumpy(actual)
            mismatch=expected.view("u4")!=observed.view("u4")
            row={"step":step,"adaptive":adaptive,"words":int(expected.size),"differing_words":int(mismatch.sum())}
            if mismatch.any():
                index=tuple(map(int,np.argwhere(mismatch)[0]))
                row["first_difference"]={"index":index,"native":float(expected[index]),"cuda":float(observed[index])}
            results.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    receipt={"schema":"wrf471-nwp-window-development-v1","qualification":"none; analytic inputs; non-PSD expressions only",
             "native_build":json.loads((args.library.parent/"nwp-build.json").read_text()),"results":results,
             "module_source_sha256":hashlib.sha256(module_source("nwp_windows").encode()).hexdigest()}
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    differing=sum(r["differing_words"] for r in results)
    print(json.dumps({"words":sum(r["words"] for r in results),"differing_words":differing,"receipt":str(args.output)}))
    if differing:
        raise SystemExit(1)


if __name__=="__main__":
    main()
