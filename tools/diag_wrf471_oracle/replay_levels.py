#!/usr/bin/env python3
"""Compare CUDA leaves against the unchanged strict WRF level modules.

--analytic is a development check, never real-input qualification. A captured
native NPZ must contain the eleven raw routine inputs plus terrain, p_levels,
z_levels and a JSON provenance string. No hydrostatic pressure is reconstructed.
Only a compact JSON receipt is written; output arrays are not retained.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
NAMES = ("u", "v", "w", "t", "qv", "ph", "phb", "pp", "pb", "p", "pw")


def analytic():
    """Independent branch coverage, deliberately containing nonmonotone targets."""
    rng = np.random.default_rng(471)
    nz, ny, nx = 8, 3, 4
    terrain = np.array([[-30.,0.,120.,900.],[1100.,2200.,-20.,40.],
                        [300.,1200.,500.,0.]], dtype="f4")
    z = terrain[None] + np.array([0.,80.,230.,600.,1400.,3000.,5500.,9000.,14000.],dtype="f4")[:,None,None]
    ph = rng.uniform(-15,15,z.shape).astype("f4")
    phb = (z*np.float32(9.81)-ph).astype("f4")
    mass_z = np.float32(.5)*(z[1:]+z[:-1])
    p = (100000*np.exp(-mass_z/8000.)).astype("f4")
    pw = (100000*np.exp(-z/8000.)).astype("f4")
    pp = rng.uniform(-125,125,p.shape).astype("f4")
    pb = p.copy()
    qv = rng.uniform(-.0001,.015,p.shape).astype("f4")
    qv[::2,0,0] = 0
    t = rng.uniform(-18.,35.,p.shape).astype("f4")
    u = rng.uniform(-25,35,(nz,ny,nx+1)).astype("f4")
    v = rng.uniform(-30,20,(nz,ny+1,nx)).astype("f4")
    return dict(u=u,v=v,w=np.zeros_like(pw),t=t,qv=qv,ph=ph,phb=phb,pp=pp,pb=pb,p=p,pw=pw,
                terrain=terrain,p_levels=np.array([110000,100000,95000,85000,70000,50000,25000,10000],dtype="f4"),
                z_levels=np.array([0.,.5,-100.,250.,1000.,3000.,7000.,18000.,-20.],dtype="f4"))


def reference(library, data, kind, pressure_kind, extrapolate):
    nz, ny, nx = data["t"].shape
    levels = data["p_levels" if kind == 1 else "z_levels"]
    nl = len(levels)
    packed = np.zeros((nx+1,nz+1,ny+1,11),dtype="f4",order="F")
    for index,name in enumerate(NAMES):
        a = data[name]
        packed[:a.shape[2],:a.shape[0],:a.shape[1],index] = a.transpose(2,0,1)
    terrain = np.zeros((nx+1,ny+1),dtype="f4",order="F")
    terrain[:nx,:ny] = data["terrain"].T
    axis = np.empty(nl,dtype="f4")
    out = np.empty((nx+1,nl,ny+1,9),dtype="f4",order="F")
    ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    library.levels_ref(ctypes.c_int(kind),ctypes.c_int(nx),ctypes.c_int(ny),ctypes.c_int(nz),ctypes.c_int(nl),
                       ptr(packed),ptr(terrain),ptr(levels),ctypes.c_int(pressure_kind),ctypes.c_int(extrapolate),
                       ctypes.c_float(-999.),ptr(axis),ptr(out))
    assert np.array_equal(axis.view("u4"),levels.view("u4"))
    count = 8 if kind == 1 else 9
    return np.ascontiguousarray(out[:nx,:,:ny,:count].transpose(3,1,2,0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library",type=Path,required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--analytic",action="store_true")
    group.add_argument("--capture",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    import cupy as cp
    from woof import wrf_exact
    from woof.core import level_diagnostics as ld
    from woof.core.kernels import module_source, module_options
    if not wrf_exact.ENABLED:
        raise ValueError("replay requires GPUWM_WRF_EXACT=1 before process startup")
    if args.analytic:
        data = analytic()
        provenance = {"kind":"analytic branch coverage","real_native_capture":False}
    else:
        with np.load(args.capture,allow_pickle=False) as capture:
            data = {name:np.ascontiguousarray(capture[name],dtype="f4")
                    for name in (*NAMES,"terrain","p_levels","z_levels")}
            provenance = json.loads(str(capture["provenance"]))
        if not provenance.get("real_native_capture"):
            raise ValueError("capture provenance must explicitly identify a real native capture")
        provenance["capture_sha256"] = hashlib.sha256(args.capture.read_bytes()).hexdigest()
    device = {name:cp.asarray(value) for name,value in data.items()}
    inputs = ld.LevelInputs(*(device[name] for name in ("u","v","t","qv","ph","phb","pp","pb")))
    library = ctypes.CDLL(str(args.library.resolve()))
    library.levels_ref.restype = None
    nz,ny,nx = data["t"].shape
    results = []
    for kind,pressure_kind,extrapolate in ((1,1,1),(1,1,2),(1,2,1),(1,2,2),(2,1,1)):
        expected = reference(library,data,kind,pressure_kind,extrapolate)
        actual = cp.empty(expected.shape,dtype=cp.float32)
        if kind == 1:
            call = lambda: ld.pressure_levels(inputs,device["p"],device["pw"],device["p_levels"],actual,
                                             pressure_kind=pressure_kind,extrapolate=extrapolate)
        else:
            call = lambda: ld.height_levels(inputs,device["terrain"],device["z_levels"],actual)
        call()
        words = cp.asnumpy(actual).view("u4")
        refwords = expected.view("u4")
        mismatches = words != refwords
        start,end = cp.cuda.Event(),cp.cuda.Event()
        start.record()
        call()
        end.record()
        end.synchronize()
        result = {"module":"pld" if kind == 1 else "zld","pressure_kind":pressure_kind,
                  "extrapolate":extrapolate,"words":int(expected.size),"differing_words":int(mismatches.sum()),
                  "stage_ms":float(cp.cuda.get_elapsed_time(start,end))}
        if mismatches.any():
            index = tuple(map(int,np.argwhere(mismatches)[0]))
            result["first_difference"] = {"index":index,"native_word":hex(int(refwords[index])),
                                          "cuda_word":hex(int(words[index])),
                                          "native_value":float(expected[index]),
                                          "cuda_value":float(words.view("f4")[index])}
            result["differing_by_field"] = [int(x.sum()) for x in mismatches]
        results.append(result)
    source = module_source("level_diagnostics")
    receipt = {"schema":"wrf471-level-replay-v1","provenance":provenance,
               "qualification":"development only" if args.analytic else
                   "caller-provided capture comparison; provenance requires independent review; whole forecast still required",
               "dimensions":{"nz":nz,"ny":ny,"nx":nx},"results":results,
               "module_source_sha256":hashlib.sha256(source.encode()).hexdigest(),
               "compiler_options":list(wrf_exact.effective_options(module_options("level_diagnostics"))),
               "gpu":cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
               "cupy":cp.__version__,"native_build":json.loads((args.library.parent/"build.json").read_text())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"receipt":str(args.output),"words":sum(r["words"] for r in results),
                      "differing_words":sum(r["differing_words"] for r in results),
                      "qualification":receipt["qualification"]}))
    if any(r["differing_words"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
