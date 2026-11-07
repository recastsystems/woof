"""Direct strict MPAS oracle tests (build products are outside the checkout).

Prepare: python tests/test_kessler_v842.py --emit /tmp/kessler-scratch
Build: make -j4 -C /tmp/kessler-scratch -f .../kessler_oracle/Makefile SUPPORT=... MPAS_REF=...
CPU: KESSLER_ORACLE_DIR=/tmp/kessler-scratch python -m pytest tests/test_kessler_v842.py -q
CUDA: the GPU queue run --minutes 10 -- ... python tests/test_kessler_v842.py --gpu /tmp/kessler-scratch --receipt ...
CUDA uses the unchanged shared engine glibc header and hex native_powf.
Their results are qualified against this host's strict native oracle.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from woof.hex.mpasphys.kessler_v842 import CUDA_SOURCE, CUDA_RELHUM_SOURCE, COMPILE_OPTIONS, launch_kessler, run_mpas, shared_math_header

NAMES = ("t", "qv", "qc", "qr", "rho", "pii", "z", "dz8w", "rainnc", "rainncv")
# Values only for this synthetic test; launcher has no default physics constants.
CONSTANTS = np.array([2.5e6, 1004.5, 287.0 / 461.6, .6112, 17.67, 29.65, 273.15, 1000.0], dtype=np.float32)
HOST_HEADER = r"""
#include <cmath>
#include <algorithm>
using std::max;
#define __global__
#define __device__
struct Index { int x; }; Index blockIdx, blockDim, threadIdx;
float __fadd_rn(float a,float b) { return a+b; }
float __fsub_rn(float a,float b) { return a-b; }
float __fmul_rn(float a,float b) { return a*b; }
float __fdiv_rn(float a,float b) { return a/b; }
float mpas_powf(float a,float b) { return powf(a,b); }
float mpas_expf(float a) { return expf(a); }
float mpas_sqrtf(float a) { return sqrtf(a); }
"""
HOST_WRAPPER = r"""
extern "C" void witness(float *t,float *qv,float *qc,float *qr,float *rho,float *pii,float *z,float *dz,
float *rain,float *rainv,float dt,float *v,int nz,int nc) {
 float *work=new float[6*nz*nc];
 blockDim.x=1; threadIdx.x=0;
 for(int c=0;c<nc;++c) { blockIdx.x=c;
 kessler_v842(t,qv,qc,qr,rho,pii,z,dz,rain,rainv,work,dt,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],nz,nc); }
 delete[] work;
}
"""


def synthetic_columns(nz=9, nc=67, profile="mixed"):
    rng = np.random.default_rng(1428)
    shape = (nz, nc)
    fields = {
        "t": rng.uniform(275, 325, shape).astype("f4"),
        "qv": rng.uniform(0, .025, shape).astype("f4"),
        "qc": rng.uniform(0, .004, shape).astype("f4"),
        "qr": rng.uniform(0, .006, shape).astype("f4"),
        "rho": np.broadcast_to(np.linspace(1.2, .25, nz, dtype="f4")[:, None], shape).copy(),
        "pii": np.broadcast_to(np.linspace(1, .7, nz, dtype="f4")[:, None], shape).copy(),
        "z": np.broadcast_to(np.arange(nz, dtype="f4")[:, None] * 180, shape).copy(),
        "dz8w": np.full(shape, 180, dtype="f4"),
        "rainnc": rng.uniform(0, 30, nc).astype("f4"),
        "rainncv": np.full(nc, -7, dtype="f4"),
    }
    if profile == "dry":
        for k in ("qv", "qc", "qr"):
            fields[k].fill(0)
    elif profile == "cap":
        fields["t"].fill(410)
        fields["pii"].fill(1)
    elif profile == "negative":
        fields["qv"].fill(-1e-5)
        fields["qc"].fill(-1e-5)
        fields["qr"].fill(-1e-5)
    elif profile == "thin":
        fields["dz8w"].fill(5)
        fields["z"][:] = np.arange(nz, dtype="f4")[:, None] * 5
    elif profile == "subnormal":
        for k in ("qv", "qc", "qr"):
            fields[k].fill(0)
        # Dry columns preserve cumulative rain via rainnc + 0. FTZ loses it.
        words = np.array([1, 0x00400000, 0x007fffff], dtype="u4")
        fields["rainnc"][:] = np.resize(words, nc).view("f4")
    return fields


def native(directory, fields, dt, symbol="oracle"):
    fields = {k: v.copy() for k, v in fields.items()}
    filename = {"oracle": "liboracle.so", "witness": "libwitness.so", "shared_witness": "libsharedwitness.so"}[symbol]
    lib = ctypes.CDLL(str(Path(directory).resolve() / filename))
    func = getattr(lib, "witness" if symbol == "shared_witness" else symbol)
    ptr = ctypes.POINTER(ctypes.c_float)
    func.argtypes = [ptr] * 10 + [ctypes.c_float, ptr, ctypes.c_int, ctypes.c_int]
    func.restype = None
    nz, nc = fields["t"].shape
    func(*(fields[k].ctypes.data_as(ptr) for k in NAMES), np.float32(dt), CONSTANTS.ctypes.data_as(ptr), nz, nc)
    return fields


def compare(a, b):
    return {k: int(np.count_nonzero(a[k].view("u4") != b[k].view("u4"))) for k in NAMES}


@pytest.mark.parametrize("profile,dt,nz,nc", [
    ("dry", 1, 2, 1), ("mixed", 1, 9, 67), ("mixed", 90, 55, 5),
    ("cap", 60, 9, 3), ("thin", 300, 9, 7), ("negative", 1, 9, 1),
    ("mixed", 0, 9, 7), ("mixed", 60, 257, 1), ("subnormal", 1, 2, 3),
])
@pytest.mark.parametrize("symbol", ["witness", "shared_witness"])
def test_host_glibc_witness(profile, dt, nz, nc, symbol):
    directory = os.environ.get("KESSLER_ORACLE_DIR")
    if not directory:
        pytest.skip("standalone oracle not built: set KESSLER_ORACLE_DIR")
    fields = synthetic_columns(nz, nc, profile)
    differences = compare(native(directory, fields, dt), native(directory, fields, dt, symbol))
    assert all(n == 0 for n in differences.values()), differences


def emit(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parent / "kessler_oracle"))
    from driver_extract import extract
    extract(directory, os.environ["MPAS_REF"])
    (directory / "witness.cpp").write_text(HOST_HEADER + CUDA_SOURCE + CUDA_RELHUM_SOURCE + HOST_WRAPPER + DRIVER_WRAPPER)
    host_intrinsics = HOST_HEADER[:HOST_HEADER.index("float mpas_powf")]
    (directory / "sharedwitness.cpp").write_text(
        POW_HOST_INTRINSICS + host_intrinsics + SHARED_HOST_INTRINSICS + shared_math_header()
        + CUDA_SOURCE + CUDA_RELHUM_SOURCE + HOST_WRAPPER + DRIVER_WRAPPER)
    from woof.hex.native_powf import CUDA_NATIVE_POWF
    (directory / "sharedpow.cpp").write_text(POW_HOST_INTRINSICS + CUDA_NATIVE_POWF + r"""
extern "C" void sharedpow(const float *x,float *out,float y,int n) {
 for(int i=0;i<n;++i) out[i]=mpas_native_powf_positive(x[i],y);
}
extern "C" void glibcpow(const float *x,float *out,float y,int n) {
 for(int i=0;i<n;++i) out[i]=powf(x[i],y);
}
""")


def replay(directory, receipt, gpu=False, shared_host=False):
    rows = []
    if gpu:
        import cupy as cp
        math_header = shared_math_header()
    for profile, dt, nz, nc in [("dry", 1, 2, 1), ("mixed", 1, 9, 67), ("mixed", 90, 55, 5),
                                ("cap", 60, 9, 3), ("thin", 300, 9, 7), ("negative", 1, 9, 1),
                                ("mixed", 0, 9, 7), ("mixed", 60, 257, 1),
                                ("subnormal", 1, 2, 3)]:
        fields = synthetic_columns(nz, nc, profile)
        expected = native(directory, fields, dt)
        if gpu:
            device = {k: cp.asarray(v) for k, v in fields.items()}
            launch_kessler(**device, dt_in=dt, math_header=math_header,
                           **dict(zip(("xlv", "cp", "ep2", "svp1", "svp2", "svp3", "svpt0", "rhowater"), CONSTANTS)))
            actual = {k: cp.asnumpy(v) for k, v in device.items()}
        else:
            actual = native(directory, fields, dt, "shared_witness" if shared_host else "witness")
        rows.append({"profile": profile, "dt": dt, "nz": nz, "nc": nc, "differing_words": compare(expected, actual),
                     "input_sha256": hashlib.sha256(b"".join(fields[k].tobytes() for k in NAMES)).hexdigest()})
    driver_rows = []
    for top, bucket, accumulate, diagnostics, temperature in [
            (9,100,1,True,None),(5,1,1,True,None),(5,1,0,False,None),(9,0,1,False,None),
            (9,1,1,True,175),(9,1,1,True,410)]:
        state = driver_state()
        if temperature is not None:
            set_temperature(state, temperature)
        expected = native_driver(directory, state, 90, top, bucket, accumulate, "driver_oracle", diagnostics=diagnostics)
        if gpu:
            device = {k: cp.asarray(v) for k,v in state.items()}
            constants = dict(zip(("xlv","cp","ep2","svp1","svp2","svp3","svpt0","rhowater"),CONSTANTS))
            constants.update(rvord=np.float32(461.6)/np.float32(287), rd=287, p0=1e5,
                             rcv=np.float32(287)/np.float32(717.5), gravity=np.float32(9.80616))
            run_mpas(state=device, constants=constants, dt_microp=90, dt_dyn=60,
                     mp_top_level=top, config_bucket_rainnc=bucket,
                     l_acrain=bool(accumulate),math_header=math_header,l_diags=diagnostics)
            actual = {k:cp.asnumpy(v) for k,v in device.items()}
        else:
            actual = native_driver(directory,state,90,top,bucket,accumulate,
                                   "driver_shared_witness" if shared_host else "driver_witness", diagnostics=diagnostics)
        differences = {k:int(np.count_nonzero(expected[k].view("u4") != actual[k].view("u4"))) for k in state}
        details = {}
        for k,count in differences.items():
            if count:
                idx=np.argwhere(expected[k].view("u4") != actual[k].view("u4"))[:3]
                details[k]=[{"index":p.tolist(),"native":float(expected[k][tuple(p)]),
                             "actual":float(actual[k][tuple(p)])} for p in idx]
        driver_rows.append({"top":top,"bucket":bucket,"accumulate":accumulate,"diagnostics":diagnostics,"temperature":temperature,
                            "differing_words":differences,"first_differences":details})
    result = {"process_count": 1,
              "backend": "hex-cuda-shared-libm" if gpu else "host-shared-libm-witness" if shared_host else "host-glibc-witness",
              "compile_options": list(COMPILE_OPTIONS), "rows": rows, "driver_rows":driver_rows,
              "kernel_source_sha256":hashlib.sha256((CUDA_SOURCE+CUDA_RELHUM_SOURCE).encode()).hexdigest(),
              "all_zero": all(all(v == 0 for v in row["differing_words"].values()) for row in rows + driver_rows),
              "shared_math_header_sha256":hashlib.sha256(shared_math_header().encode()).hexdigest() if gpu or shared_host else None,
              "acceptance": False, "reason": "synthetic module/driver oracle only; required forecast proof is separate"}
    Path(receipt).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def subnormal_probe(directory, receipt):
    """Test the actual port and a deliberate FTZ mutation, under gpu/the build queue."""
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    fields = synthetic_columns(2, 3, "subnormal")
    expected = native(directory, fields, 1)
    assert np.array_equal(expected["rainnc"].view("u4"), fields["rainnc"].view("u4"))
    device = {k: cp.asarray(v) for k, v in fields.items()}
    workspace = launch_kessler(**device, dt_in=1, math_header=shared_math_header(),
        **dict(zip(("xlv", "cp", "ep2", "svp1", "svp2", "svp3", "svpt0", "rhowater"), CONSTANTS)))
    actual = {k: cp.asnumpy(v) for k, v in device.items()}
    differences = compare(expected, actual)
    assert all(n == 0 for n in differences.values()), differences

    # Counterfactual only: prove that this fixture detects the measured hazard.
    mutation_options = tuple("--ftz=true" if v == "--ftz=false" else v for v in COMPILE_OPTIONS)
    image, _ = compile_using_nvrtc(shared_math_header() + CUDA_SOURCE + CUDA_RELHUM_SOURCE,
                                  mutation_options, None, "kessler_v842_ftz_mutation.cu")
    module = cp.cuda.function.Module()
    module.load(image.encode() if isinstance(image, str) else image)
    mutated = {k: cp.asarray(v) for k, v in fields.items()}
    work = cp.empty(6 * 2 * 3, dtype=cp.float32)
    args = tuple(mutated[k] for k in NAMES) + (work, np.float32(1), *CONSTANTS,
                                               np.int32(2), np.int32(3))
    module.get_function("kessler_v842")((1,), (128,), args)
    mutation_differences = compare(expected, {k: cp.asnumpy(v) for k, v in mutated.items()})
    assert mutation_differences["rainnc"] == 3, mutation_differences
    result = {"process_count": 1, "scope": "subnormal module regression, not forecast proof",
              "input_rainnc_words": fields["rainnc"].view("u4").tolist(),
              "native_rainnc_words": expected["rainnc"].view("u4").tolist(),
              "cuda_rainnc_words": actual["rainnc"].view("u4").tolist(),
              "compile_options": list(COMPILE_OPTIONS), "differing_words": differences,
              "forced_ftz_mutation_differing_words": mutation_differences,
              "passed": True, "forecast_proven": False}
    Path(receipt).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))



DRIVER_WRAPPER = r"""
extern "C" void driver_witness(
float *theta_m,float *qv,float *qc,float *qr,float *rho_zz,float *zz,float *exner,float *exner_b,
float *rtheta_b,float *rtheta_p,float *pressure_p,float *pressure_b,float *rt,float *dtheta,float *relhum,
float *zgrid,float *sp,float *tsp,float *rain,float *rainv,float *precipw,int *buckets,
float *v,float dt,float dtdyn,int mp_top,float bucket,int accumulate,int diags,int nz,int nc) {
 float *mem=new float[8*nz*nc+2*nc];
 float *t=mem,*qvp=t+nz*nc,*qcp=qvp+nz*nc,*qrp=qcp+nz*nc;
 float *rho=qrp+nz*nc,*pii=rho+nz*nc,*z=pii+nz*nc,*dz=z+nz*nc;
 float *rainlocal=dz+nz*nc,*last=rainlocal+nc;
 blockDim.x=1; threadIdx.x=0;
 for(int p=0;p<nz*nc;++p) { blockIdx.x=p;
 kessler_from_mpas(theta_m,qv,qc,qr,rho_zz,zz,exner,zgrid,t,qvp,qcp,qrp,rho,pii,z,dz,v[8],nz,nc); }
 for(int c=0;c<nc;++c) { rainlocal[c]=0; last[c]=0; }
 witness(t,qvp,qcp,qrp,rho,pii,z,dz,rainlocal,last,dt,v,nz,nc);
 for(int c=0;c<nc;++c) { blockIdx.x=c;
 if(diags) for(int k=0;k<nz;++k) { blockIdx.x=k*nc+c;
 kessler_relhum(t,pii,qvp,pressure_b,pressure_p,relhum,nz*nc); }
 blockIdx.x=c;
 kessler_precip_to_mpas(rho,qvp,dz,rainlocal,rain,rainv,precipw,buckets,bucket,accumulate,nz,nc);
 kessler_to_mpas(theta_m,qv,qc,qr,rho_zz,zz,exner,exner_b,rtheta_b,rtheta_p,
 pressure_p,pressure_b,zgrid,sp,tsp,rt,dtheta,t,qvp,qcp,qrp,v[8],v[9],v[10],v[11],v[12],dtdyn,mp_top,nz,nc); }
 delete[] mem;
}
"""


def driver_state(nz=9, nc=7):
    raw = synthetic_columns(nz, nc)
    rvord = np.float32(461.6) / np.float32(287)
    state = {k: raw[k].copy() for k in ("qv", "qc", "qr", "rainnc", "rainncv")}
    state.update(theta_m=raw["t"] * (np.float32(1) + rvord * raw["qv"]),
                 rho_zz=raw["rho"].copy(), zz=np.ones((nz,nc), dtype="f4"),
                 exner=raw["pii"].copy(), exner_b=raw["pii"].copy(),
                 rtheta_b=np.full((nz,nc), 300, dtype="f4"),
                 rtheta_p=np.full((nz,nc), 1, dtype="f4"),
                 pressure_p=np.full((nz,nc), 1, dtype="f4"),
                 pressure_b=np.full((nz,nc), 90000, dtype="f4"),
                 rt_diabatic_tend=np.full((nz,nc), -7, dtype="f4"),
                 dtheta_dt_mp=np.full((nz,nc), -7, dtype="f4"),
                 relhum=np.full((nz,nc), -7, dtype="f4"),
                 zgrid=np.broadcast_to(np.arange(nz+1,dtype="f4")[:,None]*180,(nz+1,nc)).copy(),
                 surface_pressure=np.full(nc, 100000, dtype="f4"),
                 tend_sfc_pressure=np.full(nc, -7, dtype="f4"),
                 precipw=np.full(nc, -7, dtype="f4"), i_rainnc=np.zeros(nc, dtype="i4"))
    return state


def native_driver(directory, state, dt, top, bucket, accumulate, symbol, diagnostics=True):
    sys.path.insert(0, str(Path(__file__).parent / "kessler_oracle"))
    from driver_extract import DRIVER_FIELDS
    state = {k: v.copy() for k,v in state.items()}
    filename = {"driver_oracle":"liboracle.so", "driver_witness":"libwitness.so",
                "driver_shared_witness":"libsharedwitness.so"}[symbol]
    lib = ctypes.CDLL(str(Path(directory).resolve() / filename))
    func = getattr(lib, "driver_witness" if symbol == "driver_shared_witness" else symbol)
    func.argtypes = [ctypes.c_void_p] * len(DRIVER_FIELDS) + [ctypes.c_void_p, ctypes.c_float, ctypes.c_float,
                      ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    func.restype = None
    constants = np.concatenate((CONSTANTS, np.array([np.float32(461.6)/np.float32(287), 287, 1e5,
                              np.float32(287)/np.float32(717.5), 9.80616], dtype="f4")))
    nz,nc=state["theta_m"].shape
    func(*(state[k].ctypes.data for k in DRIVER_FIELDS), constants.ctypes.data, dt, 60, top, bucket, accumulate, int(diagnostics), nz, nc)
    return state


@pytest.mark.parametrize("top,bucket,accumulate", [(9, 100, 1), (5, 1, 1), (5, 1, 0), (9, 0, 1)])
@pytest.mark.parametrize("symbol", ["driver_witness", "driver_shared_witness"])
def test_driver_host_witness(top,bucket,accumulate,symbol):
    directory = os.environ.get("KESSLER_ORACLE_DIR")
    if not directory:
        pytest.skip("standalone oracle not built: set KESSLER_ORACLE_DIR")
    state = driver_state()
    expected = native_driver(directory,state,90,top,bucket,accumulate,"driver_oracle")
    actual = native_driver(directory,state,90,top,bucket,accumulate,symbol)
    differences = {k: int(np.count_nonzero(expected[k].view("u4") != actual[k].view("u4"))) for k in state}
    assert all(n == 0 for n in differences.values()), differences



@pytest.mark.parametrize("temperature", [175,410])
@pytest.mark.parametrize("symbol", ["driver_witness", "driver_shared_witness"])
def test_driver_diagnostics_extremes(temperature,symbol):
    directory=os.environ.get("KESSLER_ORACLE_DIR")
    if not directory:
        pytest.skip("standalone oracle not built: set KESSLER_ORACLE_DIR")
    state=driver_state()
    set_temperature(state, temperature)
    expected=native_driver(directory,state,90,9,1,True,"driver_oracle")
    actual=native_driver(directory,state,90,9,1,True,symbol)
    differences={k:int(np.count_nonzero(expected[k].view("u4")!=actual[k].view("u4"))) for k in state}
    assert all(n==0 for n in differences.values()), differences


def set_temperature(state, temperature):
    rvord=np.float32(461.6)/np.float32(287)
    state["theta_m"][:]=(np.float32(temperature)/state["exner"])*(np.float32(1)+rvord*state["qv"])

POW_HOST_INTRINSICS = r"""
#include <cmath>
#include <cstring>
#define __constant__
#define __device__
unsigned __float_as_uint(float x) { unsigned r; memcpy(&r,&x,4); return r; }
float __uint_as_float(unsigned x) { float r; memcpy(&r,&x,4); return r; }
unsigned long long __double_as_longlong(double x) { unsigned long long r; memcpy(&r,&x,8); return r; }
double __longlong_as_double(unsigned long long x) { double r; memcpy(&r,&x,8); return r; }
float __double2float_rn(double x) { return (float)x; }
double __dmul_rn(double a,double b) { return a*b; }
double __dadd_rn(double a,double b) { return a+b; }
double __dsub_rn(double a,double b) { return a-b; }
double __fma_rn(double a,double b,double c) { return std::fma(a,b,c); }
"""

SHARED_HOST_INTRINSICS = r"""
float __int_as_float(int x) { return __uint_as_float((unsigned)x); }
float __fsqrt_rn(float x) { return sqrtf(x); }
float __fmaf_rn(float a,float b,float c) { return std::fma(a,b,c); }
double __ddiv_rn(double a,double b) { return a/b; }
"""


def pow_probe(directory, receipt):
    rng=np.random.default_rng(49128)
    ptr=ctypes.POINTER(ctypes.c_float)
    rows=[]
    for power in (.1364,.875,1004/287,.2046,.525,287/717.5):
        x=np.exp(rng.uniform(-23,3,100000)).astype("f4")
        x[0]=0
        for filename in ("libpow.so","libpow-fma.so"):
            lib=ctypes.CDLL(str(Path(directory).resolve()/filename))
            expected=np.empty_like(x);actual=np.empty_like(x)
            for symbol,out in (("glibcpow",expected),("sharedpow",actual)):
                f=getattr(lib,symbol)
                f.argtypes=[ptr,ptr,ctypes.c_float,ctypes.c_int]
                f.restype=None
                f(x.ctypes.data_as(ptr),out.ctypes.data_as(ptr),np.float32(power),x.size)
            rows.append({"exponent":float(np.float32(power)),"variant":filename,
                         "words":x.size,"differing_words":int(np.count_nonzero(expected.view("u4")!=actual.view("u4")))})
    result={"scope":"existing CUDA_NATIVE_POWF evaluated on host with round-to-nearest intrinsics, compared to glibc 2.43; not a GPU proof", "rows":rows}
    Path(receipt).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit")
    parser.add_argument("--host")
    parser.add_argument("--gpu")
    parser.add_argument("--receipt")
    parser.add_argument("--pow")
    parser.add_argument("--shared-host")
    parser.add_argument("--subnormal-gpu")
    args = parser.parse_args()
    if args.emit:
        emit(args.emit)
    if args.host or args.gpu:
        replay(args.host or args.gpu, args.receipt, bool(args.gpu))

    if args.shared_host:
        replay(args.shared_host, args.receipt, shared_host=True)

    if args.pow:
        pow_probe(args.pow, args.receipt)

    if args.subnormal_gpu:
        subnormal_probe(args.subnormal_gpu, args.receipt)
