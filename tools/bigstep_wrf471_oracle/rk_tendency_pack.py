"""Feed a real cropped state through native WRF preparation and RK tendency."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from rk_pack import pad2,pad3,SCALAR_FIELDS

F32=np.float32
FIELDS3=("U","V","W","T","PH","PHB","PB","P","AL","ALB","T_INIT")
FIELDS2=("MU","MUB","MAPFAC_MX","MAPFAC_MY","MAPFAC_UX","MAPFAC_UY",
         "MAPFAC_VX","MF_VX_INV","MAPFAC_VY","XLAT","F","E","SINALPHA","COSALPHA","HGT")
FIELDS1=("C1H","C2H","C1F","C2F","FNM","FNP","DNW","RDN","RDNW",
         "U_BASE","V_BASE","T_BASE")


def generate(executable,state,out,scratch,stored_theta=False):
    scratch.mkdir(parents=True,exist_ok=True)
    p=np.load(state)
    nz,ny,nx=p["T"].shape
    result={"nx":np.array(nx),"ny":np.array(ny),"nz":np.array(nz)}
    x=np.stack([pad3(p[key]+F32(300) if key=="T" and not stored_theta else p[key],nx,ny,nz) for key in FIELDS3])
    original_pressure=x[7].copy()
    # The engine stores total pressure. Feed Fortran the same perturbation
    # word recovered from that stored pressure, and retain the lost words.
    canonical_pressure=(p["P"]+p["PB"])-p["PB"]
    x[7]=pad3(canonical_pressure,nx,ny,nz)
    m=np.stack([pad2(p[key],nx,ny) for key in FIELDS2])
    z=np.stack([np.pad(p[key],(0,nz+1-p[key].size),mode="edge") for key in FIELDS1]
               +[np.zeros(nz+1,np.float32),np.zeros(nz+1,np.float32)])
    q=np.stack([np.zeros_like(x[0])]+[pad3(p[key],nx,ny,nz) for key in SCALAR_FIELDS])
    coeff=np.concatenate([p[key] for key in ("CF1","CF2","CF3","CFN","CFN1")])
    case_meta=[]
    # Reversing the rotation parameters exercises the southern hemisphere
    # without changing any dynamical field or staggering.
    for case,(name,step,hemisphere) in enumerate([
        ("real-specified-rk1",1,1),("real-specified-rk2",2,1),("southern-rotation",2,-1),
    ]):
        mm=m.copy()
        if hemisphere<0: mm[9]*=-1;mm[10]*=-1
        inp=scratch/f"rk-tendency-{case}.input"
        native=scratch/f"rk-tendency-{case}.output"
        with inp.open("wb") as stream:
            stream.write(np.array([nx,ny,nz,step],np.int32).tobytes())
            for a in (x,mm,z,q,coeff):stream.write(a.tobytes())
        process=subprocess.run([str(executable.resolve()),str(inp.resolve()),str(native.resolve())],
                               cwd=scratch,check=True,text=True,capture_output=True)
        (scratch/f"rk-tendency-{case}.log").write_text(process.stdout+process.stderr)
        words=np.fromfile(native,np.float32)
        offset=0
        dims3=(ny+8,nz+1,nx+8);dims2=(ny+8,nx+8)
        for key,shape in (("tend",(5,*dims3)),("held",(5,*dims3)),("saved",(5,*dims3)),
                          ("mtend",(2,*dims2)),("misc",(5,*dims3)),("prep",(9,*dims3)),
                          ("mass",(3,*dims2)),("cfl",(2,)),("prep_before",(9,*dims3))):
            length=int(np.prod(shape));result[f"tendency{case}_{key}_wrf"]=words[offset:offset+length].reshape(shape)
            offset+=length
        if offset!=words.size:raise ValueError("native output size differs from declared WRF argument layout")
        result[f"tendency{case}_x"]=x.copy();result[f"tendency{case}_m"]=mm
        result[f"tendency{case}_z"]=z.copy();result[f"tendency{case}_q"]=q.copy()
        result[f"tendency{case}_coeff"]=coeff.copy()
        result[f"tendency{case}_original_pressure"]=original_pressure.copy()
        case_meta.append({"id":case,"name":name,"step":step,"hemisphere":hemisphere})
    np.savez_compressed(out,**result)
    meta={"schema":"bigstep-rk-tendency-wrf471-v1","cases":case_meta,
          "fixture_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
          "state_sha256":hashlib.sha256(state.read_bytes()).hexdigest(),
          "fields3":FIELDS3,"fields2":FIELDS2,"fields1":[*FIELDS1,"QV_BASE","Z_BASE"],
          "scope":"specified boundaries; order5 horizontal/order3 vertical advection; explicit vertical; no optional damping or diffusion; native rk_step_prep inputs",
          "compiler":"rk_step_prep and rk_tendency O0; actual native WRF O2 child-routine library"}
    meta["pressure_adapter"]={
        "expression":"(WRF_P + WRF_PB) - WRF_PB, each operation float32",
        "changed_words":int(np.count_nonzero(canonical_pressure.view(np.uint32)!=p["P"].view(np.uint32))),
        "words":int(canonical_pressure.size),
        "max_abs_pa":float(np.max(np.abs(canonical_pressure.astype(np.float64)-p["P"]))),
        "original_reference":"rk-tendency-original-pressure.npz",
        "original_pressure_sha256":hashlib.sha256(np.ascontiguousarray(p["P"]).tobytes()).hexdigest(),
        "canonical_pressure_sha256":hashlib.sha256(np.ascontiguousarray(canonical_pressure).tobytes()).hexdigest()}
    meta["theta_representation"]=(
        "literal WRF theta-minus-300 words, matched at the common transport-kernel input"
        if stored_theta else "full theta (WRF T+300), matched at the common transport-kernel input; this is a WOOF-ArWen transport-operator probe, not the literal WRF stored-T call")
    out.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("executable","state","output","scratch"):parser.add_argument(name,type=Path)
    parser.add_argument("--stored-theta",action="store_true")
    args=parser.parse_args()
    generate(args.executable,args.state,args.output,args.scratch,args.stored_theta)
