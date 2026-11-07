#!/usr/bin/env python3
"""Compare the surface operator with unmodified, pinned WRF v4.6.1.

--smoke is a constructed arithmetic check. It is never a native-run replay.
Python and the host adapter are test tools, never forecast backends.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "woof/spectral_nudging/surface.cu"
KEYS = ("u", "v", "theta", "temp", "qv", "p", "z", "ht", "regime", "znt",
        "old", "new", "odis_old", "odis_new", "wpbl", "wzfac")
F32 = np.float32
PTR = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")


def run(command, log):
    result = subprocess.run([str(x) for x in command], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    with log.open("a") as stream:
        stream.write(json.dumps([str(x) for x in command]) + "\n" + result.stdout)
    result.check_returncode()


def build_native(source_root, build):
    pins = json.loads((SOURCE.parent / "surface_sources.json").read_text())
    for row in pins["files"]:
        path = source_root / row["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"native source hash mismatch: {path}")
    flags = ["gfortran", "-cpp", "-ffree-form", "-ffree-line-length-none", "-O2",
             "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-fno-tree-vectorize",
             "-J", build, "-I", build]
    log = build / "build.log"
    for src, name in (("share/module_model_constants.F", "constants"),
                      ("phys/module_fdda_psufddagd.F", "surface")):
        run([*flags, "-c", source_root / src, "-o", build / f"{name}.o"], log)
    # Preserve the original PX arithmetic verbatim. This excerpt checks only
    # the observation-target calculation, not the complete PX land model.
    px_source = (source_root / "phys/module_sf_pxlsm.F").read_text()
    start = px_source.index("            T2_OLD = T2_NDG_OLD(I,J)")
    end = px_source.index("\n", px_source.index("SNOBS = CORB * SN_NDG_OLD", start))
    px = build / "native_px.F90"
    px.write_text("""subroutine native_px(nx,ny,core,T2_NDG_OLD,T2_NDG_NEW,Q2_NDG_OLD,Q2_NDG_NEW, &
  SN_NDG_OLD,SN_NDG_NEW,PSFC,T2OBS,Q2OBS,rh,sn) bind(C)
use iso_c_binding
use module_model_constants
implicit none
integer(c_int),value :: nx,ny
real(c_float),value :: core
real(c_float) :: T2_NDG_OLD(nx,ny),T2_NDG_NEW(nx,ny),Q2_NDG_OLD(nx,ny),Q2_NDG_NEW(nx,ny)
real(c_float) :: SN_NDG_OLD(nx,ny),SN_NDG_NEW(nx,ny),PSFC(nx,ny)
real(c_float) :: T2OBS(nx,ny),Q2OBS(nx,ny),rh(nx,ny),sn(nx,ny)
real :: T2_OLD,T2_NEW,VAPPRS,QSBT,RH2_OLD,RH2_NEW,RH2OBS,SNOBS,SFCPRS,CORB
integer :: i,j
CORB=1.0-CORE
do j=1,ny
do i=1,nx
SFCPRS=PSFC(i,j)/1000.0
""" + px_source[start:end] + "\nrh(i,j)=RH2OBS\nsn(i,j)=SNOBS\nend do\nend do\nend subroutine\n")
    run([*flags, "-shared", Path(__file__).with_name("surface_bridge.F90"), px,
         build / "constants.o", build / "surface.o", "-o", build / "native.so"], log)
    lib = ct.CDLL(str(build / "native.so"))
    lib.native_surface.argtypes = [ct.c_int]*4 + [ct.c_float]*7 + [PTR]*19
    lib.native_surface.restype = None
    lib.native_px.argtypes = [ct.c_int]*2 + [ct.c_float] + [PTR]*11
    lib.native_px.restype = None
    return lib


def smoke():
    """One mixed-regime fixture covers the native branches and boundaries."""
    nz, ny, nx = 4, 9, 11
    k, j, i = np.indices((nz, ny, nx), dtype=np.float32)
    a = {
        "u": 2 + i/7 + k, "v": -1 + j/9 - k,
        "theta": 290 + i/5 + k, "temp": 280 + i/5 - 7*k,
        "qv": .004 + i*.0001 + k*.0002, "p": 95000 - 8500*k + 7*j,
        "z": 100 + 40*k + i*2, "ht": (i*2)[0],
        "regime": ((i+j) % 5 + 1)[0], "znt": (.005 + (i % 4)*.12)[0],
        "odis_old": ((i*13+j*7) % 120)[0],
        "odis_new": ((i*11+j*17) % 150)[0],
    }
    a["old"] = np.stack((a["u"][0] + 2, a["v"][0] - 1,
                          a["theta"][0] + 3, a["qv"][0] * 4))
    a["new"] = a["old"] * F32(1.01)
    a["wpbl"] = np.broadcast_to(np.array([0, .1, 1, 1], dtype=F32)[None,:,None,None],
                                  (4,nz,ny,nx)).copy()
    a["wzfac"] = np.array([[1, 1, .1, 0]] * 4, dtype=F32)
    return {key: np.ascontiguousarray(value, dtype=F32) for key,value in a.items()}


def fortran(a):
    # WRF (i,k,j) column-major -> a C-order (j,k,i) allocation.
    if a.ndim == 3:
        return np.ascontiguousarray(a.transpose(1,0,2))
    if a.ndim == 4:
        return np.ascontiguousarray(a.transpose(0,2,1,3))
    return a.copy()


def native(lib, a, controls, fasdas):
    nz,ny,nx = a["u"].shape
    arrays = [fortran(a[key]) if key in (*KEYS[:7],"wpbl") else a[key].copy() for key in KEYS]
    out = fortran(initial_output(a))
    h = np.full((ny,nx), F32(-123.25)); q = h.copy()
    lib.native_surface(nx,ny,nz,fasdas,*controls,*arrays,out,h,q)
    return np.ascontiguousarray(out.transpose(0,2,1,3)),h,q


def initial_output(a):
    if "_initial_output" in a:
        return a["_initial_output"]
    return np.full((4,*a["u"].shape), F32(-123.25))


def capture(path):
    """Decode the one-frame, little-endian stream written by instrument.py."""
    raw = np.memmap(path, mode="r", dtype=np.uint8)
    if raw[:8].tobytes() != b"SFDD471A":
        raise ValueError("unknown native surface frame magic")
    nx,ny,nz,fasdas = map(int,np.frombuffer(raw,dtype="<i4",count=4,offset=8))
    if min(nx,ny) < 3 or nz < 1 or fasdas not in (0,1):
        raise ValueError("invalid native surface frame bounds")
    controls = list(np.frombuffer(raw,dtype="<f4",count=7,offset=24))
    offset = 52

    def take(shape,transpose=None):
        nonlocal offset
        size = int(np.prod(shape))
        array = np.frombuffer(raw,dtype="<f4",count=size,offset=offset).reshape(shape)
        offset += size*4
        if transpose is not None:
            array = array.transpose(transpose)
        return np.ascontiguousarray(array)

    a = {name:take((ny,nz,nx),(1,0,2)) for name in KEYS[:7]}
    a.update({name:take((ny,nx)) for name in ("ht","regime","znt")})
    a.update({name:take((4,ny,nx)) for name in ("old","new")})
    a.update({name:take((ny,nx)) for name in ("odis_old","odis_new")})
    a["wpbl"] = take((4,ny,nz,nx),(0,2,1,3))
    a["wzfac"] = take((4,nz))
    a["_initial_output"] = take((4,ny,nz,nx),(0,2,1,3))
    expected = [take((4,ny,nz,nx),(0,2,1,3))]
    if fasdas:
        expected.extend(take((ny,nx)) for _ in range(2))
    if offset != len(raw):
        raise ValueError(f"native frame has {len(raw)-offset} trailing bytes")
    return a,controls,fasdas,expected


def host(build):
    run(["g++", "-x", "c++", "-DSURFACE_FDDA_HOST_ORACLE", "-O2", "-fPIC",
         "-shared", "-fno-fast-math", "-ffp-contract=off", SOURCE,
         "-o", build / "translated.so"], build / "build.log")
    lib = ct.CDLL(str(build / "translated.so"))
    lib.sfc_host_density.argtypes = [PTR]*4 + [ct.c_int]*2 + [ct.c_float]*2
    lib.sfc_host_density.restype = None
    lib.sfc_host_tendencies.argtypes = [PTR]*18 + [ct.c_int]*4 + [ct.c_float]*5
    lib.sfc_host_tendencies.restype = None
    lib.sfc_host_px_targets.argtypes = [PTR]*11 + [ct.c_int,ct.c_float]
    lib.sfc_host_px_targets.restype = None

    def execute(a, controls, fasdas):
        nz,ny,nx = a["u"].shape
        dx,radius,coef,tfac,guv,gt,gq = controls
        raw = np.empty((2,ny,nx), dtype=F32); density = np.empty_like(raw)
        lib.sfc_host_density(a["odis_old"],a["odis_new"],raw,density,nx,ny,dx,radius)
        out = initial_output(a).copy()
        h = np.full((ny,nx), F32(-123.25)); q = h.copy()
        args = [a[key] for key in KEYS[:12]] + [density,a["wpbl"],a["wzfac"],out,h,q]
        lib.sfc_host_tendencies(*args,nx,ny,nz,fasdas,coef,tfac,guv,gt,gq)
        return out,h,q
    def px(arrays, core):
        out = tuple(np.empty_like(arrays[0]) for _ in range(4))
        lib.sfc_host_px_targets(*arrays,*out,arrays[0].size,core)
        return out
    return execute,px


def cuda():
    # The oracle selects strict before the package installs its compile hook.
    os.environ["GPUWM_NUMERICS"] = "strict"
    sys.path.insert(0,str(ROOT))
    import cupy as cp
    from woof.spectral_nudging.surface_gpu import (SurfaceCoefficients,SurfaceOperator,
                                                   PxObservationTargets)

    def execute(a, controls, fasdas):
        nz,ny,nx = a["u"].shape
        dx,radius,coef,tfac,guv,gt,gq = map(F32, controls)
        stream = cp.cuda.Stream()
        with stream:
            aliases = {"temp":"temperature","p":"pressure","old":"old_analysis","new":"new_analysis"}
            fields = {aliases.get(key,key):cp.asarray(a[key]) for key in KEYS}
            out = cp.asarray(initial_output(a))
            h = cp.full((ny,nx), F32(-123.25)); q = h.copy()
            fields.update(tendencies=out,sda_hfx=h,sda_qfx=q,
                          time_factors=cp.asarray([coef,tfac],dtype=cp.float32))
            operator = SurfaceOperator(SurfaceCoefficients(dx,radius,guv,gt,gq),fields,
                                       fasdas=bool(fasdas),strict=True)
            operator.prepare_density()
            stream.begin_capture()
            operator.record()
            graph = stream.end_capture()
            begin,end = cp.cuda.Event(),cp.cuda.Event()
            begin.record(stream); graph.launch(stream); end.record(stream)
        stream.synchronize()  # Oracle output collection, never forecast code.
        execute.last_milliseconds = cp.cuda.get_elapsed_time(begin,end)
        return tuple(cp.asnumpy(x) for x in (out,h,q))
    def px(arrays, core):
        stream = cp.cuda.Stream()
        with stream:
            names = ("t_old","t_new","q_old","q_new","sn_old","sn_new","psfc")
            fields = {key:cp.asarray(value) for key,value in zip(names,arrays)}
            out = tuple(cp.empty_like(fields["psfc"]) for _ in range(4))
            fields.update(zip(("t2obs","q2obs","rh2obs","snobs"),out))
            fields["time_factors"] = cp.asarray([core,1],dtype=cp.float32)
            targets = PxObservationTargets(fields,strict=True)
            stream.begin_capture(); targets.record(); graph = stream.end_capture()
            graph.launch(stream)
        stream.synchronize()
        return tuple(cp.asnumpy(x) for x in out)
    return execute,px


def compare(expected, actual):
    left,right = expected.view(np.uint32),actual.view(np.uint32)
    mask = left != right
    first = np.argwhere(mask)
    return {"words":int(mask.size), "differing_words":int(mask.sum()),
            "max_abs":float(np.max(np.abs(expected-actual))),
            "first": first[:8].tolist(),
            "first_native_words":[int(left[tuple(i)]) for i in first[:8]],
            "first_translated_words":[int(right[tuple(i)]) for i in first[:8]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("host","cuda"), default="cuda")
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--smoke", action="store_true")
    kind.add_argument("--capture", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.build_dir = args.build_dir.resolve(); args.build_dir.mkdir(parents=True,exist_ok=True)
    lib = build_native(args.source_root.resolve(),args.build_dir)
    execute,px = host(args.build_dir) if args.backend == "host" else cuda()
    provenance = {}
    if args.capture:
        if not args.provenance:
            parser.error("--capture requires --provenance describing the native run and forcing")
        provenance = json.loads(args.provenance.read_text())
        a,controls,fasdas,recorded = capture(args.capture)
        cases = (fasdas,)
    else:
        a = smoke()
        controls = [F32(x) for x in (3000,80,.375,.8,.0003,.00025,.00035)]
        cases = (0,1)
    rows = []
    for fasdas in cases:
        ref = native(lib,a,controls,fasdas)
        got = execute(a,controls,fasdas)
        row = {"fasdas":fasdas,"tendencies":compare(ref[0],got[0])}
        if args.backend == "cuda":
            row["single_graph_launch_ms"] = execute.last_milliseconds
        if fasdas:
            row.update(sda_hfx=compare(ref[1],got[1]),sda_qfx=compare(ref[2],got[2]))
        if args.capture:
            row["captured_native_tendencies"] = compare(recorded[0],got[0])
            row["native_reexecution"] = compare(recorded[0],ref[0])
            if fasdas:
                # INTENT(OUT) SDA words at excluded boundaries are undefined.
                for name,x,y in zip(("captured_sda_hfx","captured_sda_qfx"),recorded[1:],got[1:]):
                    row[name] = compare(x[:-1,:-1].copy(),y[:-1,:-1].copy())
        rows.append(row)
    if args.smoke:
        ny,nx = a["ht"].shape
        snow = np.indices((ny,nx))[1].astype(F32) * F32(20)
        arrays = [a["temp"][0].copy(), a["temp"][0]+F32(2),
                  a["old"][3].copy(),a["new"][3].copy(),snow,F32(210)-snow,a["p"][0].copy()]
        ref = tuple(np.empty((ny,nx), dtype=F32) for _ in range(4))
        lib.native_px(nx,ny,controls[2],*arrays,*ref)
        got = px(arrays,controls[2])
        rows.append({"px_fragment":True,**{key:compare(x,y) for key,x,y in zip(
            ("t2obs","q2obs","rh2obs","snobs"),ref,got)}})
    receipt = {"kind":"native-run-module-capture" if args.capture else "constructed-module-smoke",
               "qualifying_native_run_capture":bool(args.capture and provenance.get("qualifying_native_run_capture",False)),
               "backend":args.backend,"strict_numerics":True,"results":rows,
               "provenance":provenance,
               "kernel_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    if args.capture:
        with args.capture.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda:stream.read(1024*1024),b""):
                digest.update(chunk)
        receipt["capture"] = {"sha256":digest.hexdigest(),"bytes":args.capture.stat().st_size,
                              "shape":list(a["u"].shape)}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
    return int(any(value["differing_words"] for row in rows for value in row.values() if isinstance(value,dict)))


if __name__ == "__main__":
    raise SystemExit(main())
