"""MPAS-A v8.4.2 Kessler and its MPAS driver slice.

Source: MPAS-Model-v8.4.2/src/core_atmosphere/physics/physics_wrf/module_mp_kessler.F:9-243.
Arrays use [level, cell], float32, cells contiguous. One thread owns a column;
work is dynamically sized so there is no WRF fixed vertical dimension limit.
Elementary maths is provided by the caller's shared device header, defining
``mpas_powf``, ``mpas_expf`` and ``mpas_sqrtf``. Kernels never choose a libm.
The raw call preserves RAINNC accumulation and last-split RAINNCV; the MPAS
call resets local precipitation and returns the complete-step total instead.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from typing import Mapping

import numpy as np

SCHEME = "mp_kessler"
BACKEND = "hex-cuda"
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")


def shared_math_header() -> str:
    """Assemble existing shared device math without copying or changing it.

    The engine header labels its transcription glibc 2.39; column receipts
    must qualify its results against the strict oracle's actual host libm.
    """
    from pathlib import Path
    from woof.core import kernels
    from woof.hex.native_powf import CUDA_NATIVE_POWF

    header = (Path(kernels.__file__).parent / "glibc_flt32.cuh").read_text(encoding="utf-8")
    return CUDA_NATIVE_POWF + header + r"""
__device__ float mpas_powf(float x,float y) { return mpas_native_powf_positive(x,y); }
__device__ float mpas_expf(float x) { return gfk_exp(x); }
__device__ float mpas_sqrtf(float x) { return __fsqrt_rn(x); }
"""

# Explicit operations retain the GNU strict source evaluation order. Keep this
# source usable by the test-only host-glibc witness, without a second algorithm.
CUDA_SOURCE = r"""
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define MAX(a,b) fmaxf((a),(b))
#define MIN(a,b) fminf((a),(b))
// MPAS-Model-v8.4.2/src/core_atmosphere/physics/physics_wrf/module_mp_kessler.F:9-243
extern "C" __global__ void kessler_v842(
 float *t, float *qv, float *qc, float *qr,
 const float *rho, const float *pii, const float *z, const float *dz8w,
 float *rainnc, float *rainncv, float *work,
 float dt, float xlv, float cp, float ep2, float svp1, float svp2,
 float svp3, float svpt0, float rhowater, int nz, int nc) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c>=nc) return;
 // Each work plane is [level, cell]. No shared column or vertical cap.
 float *prodk=work, *vt=work+nz*nc, *vtden=work+2*nz*nc;
 float *rhok=work+3*nz*nc, *rdzk=work+4*nz*nc, *rdzw=work+5*nz*nc;
 float f5=D(M(M(svp2,S(svpt0,svp3)),xlv),cp);
 float crmax=0.0f;
 for(int k=0;k<nz;++k) {
  int p=k*nc+c;
  prodk[p]=qr[p]; rhok[p]=rho[p];
  float qrr=MAX(0.0f,M(M(qr[p],0.001f),rhok[p]));
  vtden[p]=mpas_sqrtf(D(rhok[c],rhok[p]));
  vt[p]=M(M(36.34f,mpas_powf(qrr,0.1364f)),vtden[p]);
  rdzw[p]=D(1.0f,dz8w[p]);
  crmax=MAX(M(M(vt[p],dt),rdzw[p]),crmax);
 }
 for(int k=0;k<nz-1;++k) rdzk[k*nc+c]=D(1.0f,S(z[(k+1)*nc+c],z[k*nc+c]));
 int top=(nz-1)*nc+c;
 rdzk[top]=D(1.0f,S(z[top],z[top-nc]));
 // NINT is ties away from zero; the argument is positive.
 int nfall=max(1,(int)A(A(0.5f,D(crmax,0.75f)),0.5f));
 float dtfall=D(dt,(float)nfall), time_sediment=dt;
 while(nfall>0) {
  time_sediment=S(time_sediment,dtfall);
  float ppt=D(M(M(M(rhok[c],prodk[c]),vt[c]),dtfall),rhowater);
  rainncv[c]=M(ppt,1000.0f);
  rainnc[c]=A(rainnc[c],M(ppt,1000.0f));
  for(int k=0;k<nz-1;++k) {
   int p=k*nc+c, u=p+nc;
   float factor=D(M(dtfall,rdzk[p]),rhok[p]);
   prodk[p]=S(prodk[p],M(factor,S(M(M(rhok[p],prodk[p]),vt[p]),
                                 M(M(rhok[u],prodk[u]),vt[u]))));
  }
  prodk[top]=S(prodk[top],M(M(M(dtfall,rdzk[top]),prodk[top]),vt[top]));
  if(nfall>1) {
   --nfall; crmax=0.0f;
   for(int k=0;k<nz;++k) {
    int p=k*nc+c;
    float qrr=MAX(0.0f,M(M(prodk[p],0.001f),rhok[p]));
    vt[p]=M(M(36.34f,mpas_powf(qrr,0.1364f)),vtden[p]);
    crmax=MAX(M(M(vt[p],time_sediment),rdzw[p]),crmax);
   }
   int nnew=max(1,(int)A(A(0.5f,D(crmax,0.75f)),0.5f));
   if(nnew!=nfall) { nfall=nnew; dtfall=D(time_sediment,(float)nfall); }
  } else { nfall=0; }
 }
 for(int k=0;k<nz;++k) {
  int p=k*nc+c;
  float factorn=D(1.0f,A(1.0f,M(M(2.2f,dt),mpas_powf(MAX(0.0f,qr[p]),0.875f))));
  float qrprod=A(M(qc[p],S(1.0f,factorn)),
                 M(M(M(factorn,0.001f),dt),MAX(S(qc[p],0.001f),0.0f)));
  float rcgs=M(0.001f,rho[p]);
  qc[p]=MAX(S(qc[p],qrprod),0.0f);
  qr[p]=S(A(qr[p],prodk[p]),qr[p]);
  qr[p]=MAX(A(qr[p],qrprod),0.0f);
  float temp=M(pii[p],t[p]);
  float pressure=M(1.0e5f,mpas_powf(pii[p],(1004.0f/287.0f)));
  float gam=D(2.5e6f,M(1004.0f,pii[p]));
  float es=M(M(1000.0f,svp1),mpas_expf(D(M(svp2,S(temp,svpt0)),S(temp,svp3))));
  es=MIN(es,M(0.99f,pressure));
  float qvs=D(M(ep2,es),S(pressure,es));
  float delta=S(temp,svp3);
  float prod=D(S(qv[p],qvs),A(1.0f,D(M(M(D(pressure,S(pressure,es)),qvs),f5),M(delta,delta))));
  float rq=M(rcgs,qr[p]);
  float ern=MIN(MIN(M(M(dt,D(M(A(1.6f,M(124.9f,mpas_powf(rq,0.2046f))),
                                mpas_powf(rq,0.525f)),
                              A(D(2.55e8f,M(pressure,qvs)),5.4e5f))),
                          D(MAX(S(qvs,qv[p]),0.0f),M(rcgs,qvs))),
                    MAX(S(-prod,qc[p]),0.0f)),qr[p]);
  float product=MAX(prod,-qc[p]);
  t[p]=A(t[p],M(gam,S(product,ern)));
  qv[p]=MAX(A(S(qv[p],product),ern),0.0f);
  qc[p]=A(qc[p],product); qr[p]=S(qr[p],ern);
 }
}

// MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_interface.F:598-823
extern "C" __global__ void kessler_from_mpas(
 const float *theta_m,const float *qv,const float *qc,const float *qr,
 const float *rho_zz,const float *zz,const float *exner,const float *zgrid,
 float *t,float *qvp,float *qcp,float *qrp,float *rho,float *pii,float *z,float *dz,
 float rvord,int nz,int nc) {
 int p=blockIdx.x*blockDim.x+threadIdx.x;
 if(p>=nz*nc) return;
 qvp[p]=qv[p]; qcp[p]=qc[p]; qrp[p]=qr[p];
 rho[p]=M(zz[p],rho_zz[p]);
 t[p]=D(theta_m[p],A(1.0f,M(rvord,MAX(0.0f,qvp[p]))));
 pii[p]=exner[p]; z[p]=zgrid[p]; dz[p]=S(zgrid[p+nc],zgrid[p]);
}

// MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_interface.F:826-964
extern "C" __global__ void kessler_to_mpas(
 float *theta_m,float *qv,float *qc,float *qr,const float *rho_zz,const float *zz,
 float *exner,const float *exner_b,const float *rtheta_b,float *rtheta_p,
 float *pressure_p,const float *pressure_b,const float *zgrid,
 float *surface_pressure,float *tend_sfc_pressure,float *rt_diabatic_tend,float *dtheta_dt_mp,
 const float *t,const float *qvp,const float *qcp,const float *qrp,
 float rvord,float rd,float p0,float rcv,float gravity,float dt_dyn,int mp_top,int nz,int nc) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c>=nc) return;
 for(int k=0;k<mp_top;++k) {
  int p=k*nc+c;
  float old=theta_m[p], olddry=D(old,A(1.0f,M(rvord,qv[p])));
  qv[p]=qvp[p]; qc[p]=qcp[p]; qr[p]=qrp[p];
  theta_m[p]=M(t[p],A(1.0f,M(rvord,qvp[p])));
  rt_diabatic_tend[p]=D(S(theta_m[p],old),dt_dyn);
  dtheta_dt_mp[p]=D(S(D(theta_m[p],A(1.0f,M(rvord,qv[p]))),olddry),dt_dyn);
  rtheta_p[p]=S(M(rho_zz[p],theta_m[p]),rtheta_b[p]);
  exner[p]=mpas_powf(M(M(zz[p],D(rd,p0)),A(rtheta_p[p],rtheta_b[p])),rcv);
  pressure_p[p]=M(M(zz[p],rd),A(M(exner[p],rtheta_p[p]),M(S(exner[p],exner_b[p]),rtheta_b[p])));
 }
 for(int k=mp_top;k<nz;++k) { int p=k*nc+c; qc[p]=0.0f; qr[p]=0.0f; rt_diabatic_tend[p]=0.0f; }
 float tem1=S(zgrid[c+nc],zgrid[c]), tem2=S(zgrid[c+2*nc],zgrid[c+nc]);
 float rho1=M(M(rho_zz[c],zz[c]),A(1.0f,qvp[c]));
 float rho2=M(M(rho_zz[c+nc],zz[c+nc]),A(1.0f,qvp[c+nc]));
 float old=surface_pressure[c];
 surface_pressure[c]=M(M(M(0.5f,gravity),tem1),S(rho1,D(M(M(0.5f,S(rho2,rho1)),tem1),A(tem1,tem2))));
 surface_pressure[c]=A(A(surface_pressure[c],pressure_p[c]),pressure_b[c]);
 tend_sfc_pressure[c]=D(S(surface_pressure[c],old),dt_dyn);
}

// MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:503-653
extern "C" __global__ void kessler_precip_to_mpas(
 const float *rho,const float *qvp,const float *dz,const float *rain_local,
 float *rainnc,float *rainncv,float *precipw,int *i_rainnc,
 float bucket,int accumulate,int nz,int nc) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c>=nc) return;
 precipw[c]=0.0f;
 for(int k=0;k<nz;++k) {
  int p=k*nc+c;
  float rho_a=D(rho[p],A(1.0f,qvp[p]));
  precipw[c]=A(precipw[c],M(M(qvp[p],rho_a),dz[p]));
 }
 rainncv[c]=rain_local[c]; rainnc[c]=A(rainnc[c],rainncv[c]);
 if(accumulate && bucket>0.0f && rainnc[c]>bucket) {
  i_rainnc[c]+=1; rainnc[c]=S(rainnc[c],bucket);
 }
}
"""


# MPAS compute_relhum depends on these two Thompson polynomial helpers.
# They contain only basic arithmetic, no per-scheme elementary maths.
CUDA_RELHUM_SOURCE = r"""
// MPAS-Model-v8.4.2/src/core_atmosphere/physics/physics_wrf/module_mp_thompson.F:4597-4627
__device__ float mpas_rslf(float p,float t) {
 const float coefficients[9]={.611583699E03f,.444606896E02f,.143177157E01f,
 .264224321E-1f,.299291081E-3f,.203154182E-5f,.702620698E-8f,
 .379534310E-11f,-.321582393E-13f};
 float x=MAX(-80.0f,S(t,273.16f)), es=coefficients[8];
 for(int i=7;i>=0;--i) es=A(coefficients[i],M(x,es));
 es=MIN(es,M(p,0.15f));
 return D(M(0.622f,es),S(p,es));
}
// MPAS-Model-v8.4.2/src/core_atmosphere/physics/physics_wrf/module_mp_thompson.F:4632-4658
__device__ float mpas_rsif(float p,float t) {
 const float coefficients[9]={.609868993E03f,.499320233E02f,.184672631E01f,
 .402737184E-1f,.565392987E-3f,.521693933E-5f,.307839583E-7f,
 .105785160E-9f,.161444444E-12f};
 float x=MAX(-80.0f,S(t,273.16f)), es=coefficients[8];
 for(int i=7;i>=0;--i) es=A(coefficients[i],M(x,es));
 es=MIN(es,M(p,0.15f));
 return D(M(0.622f,es),S(p,es));
}
// MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:798-850
extern "C" __global__ void kessler_relhum(
 const float *t,const float *pii,const float *qvp,const float *pressure_b,
 const float *pressure_p,float *relhum,int n) {
 int p=blockIdx.x*blockDim.x+threadIdx.x;
 if(p>=n) return;
 float pressure=__fadd_rn(pressure_b[p],pressure_p[p]);
 float temp=__fmul_rn(t[p],pii[p]);
 float tempc=__fsub_rn(temp,273.16f);
 float qvs=mpas_rslf(pressure,temp);
 if(tempc<=0.0f) qvs=mpas_rsif(pressure,temp);
 relhum[p]=__fmul_rn(__fdiv_rn(qvp[p],qvs),100.0f);
}
"""


#: Correct mode (CORRECTIONS_V842): exact text, corrected text, expected count.  WB3:
#: mpas_atmphys_driver_microphysics.F:614-617 divides the dry rho_p by (1+qv) a second time.
CORRECT_MODE_SOURCE = {
    "WB3": (('float rho_a=D(rho[p],A(1.0f,qvp[p]));', 'float rho_a=rho[p];', 1),),
}


def _module(math_header: str, corrections=()):
    """The Kessler module loaded on the calling card."""
    import cupy as cp
    # A CUDA module belongs to the card it was loaded on. Same-host ranks drive
    # several cards from one process, so a cache keyed without the card hands
    # the second card the first card's handle (CUDA_ERROR_INVALID_HANDLE at step 1).
    return _module_on(int(cp.cuda.Device().id), math_header, corrections)


# Unbounded on purpose (one entry per card and variant): a bounded per-card cache thrashes once ranks x
# variants exceed it, and every rank then rebuilds its module each step (thompson_v842._module, 8 cards,
# 2026-10-28: 1.28 s/step against 0.25 s with the cache fixed).
@lru_cache(maxsize=None)
def _module_on(device: int, math_header: str, corrections=()):
    """Compile using the supplied shared libm (no per-scheme elementary math)."""
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    source = math_header + CUDA_SOURCE + CUDA_RELHUM_SOURCE
    if corrections:
        from .driver_v842 import correct_source
        source = correct_source(source, CORRECT_MODE_SOURCE, corrections, "kessler_v842")
    # RawModule appends -ftz=true after its caller options in CuPy 14.2.
    # Use the same existing direct compiler as the engine's no-FTZ kernels.
    image, _ = compile_using_nvrtc(source, _hex_numerics.options(COMPILE_OPTIONS), None, "kessler_v842.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _arrays(arrays: Mapping[str, object], names, shape):
    """Validate storage; kernels require float32 cells contiguous."""
    import cupy as cp
    out = []
    for name in names:
        a = arrays[name]
        if not isinstance(a, cp.ndarray) or a.dtype != cp.float32 or a.shape != shape or not a.flags.c_contiguous:
            raise ValueError(f"{name} must be a contiguous device float32 array of shape {shape}")
        out.append(a)
    return out


def launch_kessler(*, t, qv, qc, qr, rho, pii, z, dz8w, rainnc, rainncv,
                   dt_in, xlv, cp, ep2, svp1, svp2, svp3, svpt0, rhowater,
                   math_header: str):
    """Update raw scheme arrays in place.

    MPAS-Model-v8.4.2/src/core_atmosphere/physics/physics_wrf/module_mp_kessler.F:9-243.
    """
    import cupy as cupy
    if len(t.shape) != 2 or t.shape[0] < 2 or t.shape[1] < 1:
        raise ValueError("Kessler needs at least two vertical levels and one cell (top spacing reads kte-1)")
    nz, nc = t.shape
    fields = _arrays(locals(), ("t", "qv", "qc", "qr", "rho", "pii", "z", "dz8w"), (nz, nc))
    rain = _arrays(locals(), ("rainnc", "rainncv"), (nc,))
    work = cupy.empty((6, nz, nc), dtype=cupy.float32)
    constants = tuple(np.float32(v) for v in (dt_in, xlv, cp, ep2, svp1, svp2, svp3, svpt0, rhowater))
    _module(math_header).get_function("kessler_v842")(
        ((nc + 63) // 64,), (64,), (*fields, *rain, work, *constants, np.int32(nz), np.int32(nc)))
    return work  # caller may retain workspace through asynchronous consumption


def run_mpas(*, state: Mapping[str, object], constants: Mapping[str, float],
             dt_microp: float, dt_dyn: float, mp_top_level: int,
             config_bucket_rainnc: float, l_acrain: bool, math_header: str,
             l_diags: bool = False, corrections=()):
    """Kessler driver slice; state is updated in place.

    MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:317-500,503-653,798-850.
    MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_interface.F:598-823,826-964.

    Constants are MPAS xlv, cp, ep2, svp1/2/3/t0, rhowater, rvord, rd, p0,
    rcv and gravity; no WRF/default constants are substituted. Diagnostics additionally require
    state["relhum"]; the MPAS Thompson polynomial helpers are included. mp_top_level
    is MPAS's one-based inclusive top. Reflectivity is not defined by Kessler.
    """
    import cupy as cp
    nz, nc = state["theta_m"].shape
    if nz < 2 or nc < 1 or not 1 <= mp_top_level <= nz:
        raise ValueError("invalid Kessler vertical extent or MPAS top level")
    fields = ("theta_m", "qv", "qc", "qr", "rho_zz", "zz", "exner", "exner_b",
              "rtheta_b", "rtheta_p", "pressure_p", "pressure_b", "rt_diabatic_tend", "dtheta_dt_mp")
    _arrays(state, fields, (nz, nc))
    _arrays(state, ("zgrid",), (nz + 1, nc))
    _arrays(state, ("surface_pressure", "tend_sfc_pressure", "rainnc", "rainncv", "precipw"), (nc,))
    if (not isinstance(state["i_rainnc"], cp.ndarray) or state["i_rainnc"].dtype != cp.int32
            or state["i_rainnc"].shape != (nc,) or not state["i_rainnc"].flags.c_contiguous):
        raise ValueError("i_rainnc must be contiguous int32 [cell]")
    if l_diags:
        _arrays(state, ("relhum",), (nz, nc))
    # Gather constants before launching so incomplete seam input cannot partially mutate state.
    scalars = {k: np.float32(constants[k]) for k in
               ("xlv", "cp", "ep2", "svp1", "svp2", "svp3", "svpt0", "rhowater", "rvord", "rd", "p0", "rcv", "gravity")}
    local = {k: cp.empty((nz, nc), dtype=cp.float32) for k in ("t", "qv", "qc", "qr", "rho", "pii", "z", "dz8w")}
    local_rain = cp.zeros(nc, dtype=cp.float32)
    last_split = cp.zeros(nc, dtype=cp.float32)
    module = _module(math_header)
    module.get_function("kessler_from_mpas")(((nz * nc + 63) // 64,), (64,), (
        *(state[k] for k in ("theta_m", "qv", "qc", "qr", "rho_zz", "zz", "exner", "zgrid")),
        *(local[k] for k in ("t", "qv", "qc", "qr", "rho", "pii", "z", "dz8w")),
        scalars["rvord"], np.int32(nz), np.int32(nc)))
    work = launch_kessler(**local, rainnc=local_rain, rainncv=last_split,
                          dt_in=dt_microp, math_header=math_header,
                          **{k: scalars[k] for k in ("xlv", "cp", "ep2", "svp1", "svp2", "svp3", "svpt0", "rhowater")})
    if l_diags:
        module.get_function("kessler_relhum")(((nz * nc + 63) // 64,), (64,), (
            local["t"], local["pii"], local["qv"], state["pressure_b"], state["pressure_p"],
            state["relhum"], np.int32(nz * nc)))
    grid, block = ((nc + 63) // 64,), (64,)
    # Correct mode (CORRECT_MODE_SOURCE): the precipitation writeback comes from the corrected text.
    precip = _module(math_header, tuple(corrections)) if corrections else module
    precip.get_function("kessler_precip_to_mpas")(grid, block, (
        local["rho"], local["qv"], local["dz8w"], local_rain,
        *(state[k] for k in ("rainnc", "rainncv", "precipw", "i_rainnc")),
        np.float32(config_bucket_rainnc), np.int32(l_acrain), np.int32(nz), np.int32(nc)))
    module.get_function("kessler_to_mpas")(grid, block, (
        *(state[k] for k in ("theta_m", "qv", "qc", "qr", "rho_zz", "zz", "exner", "exner_b", "rtheta_b", "rtheta_p",
                             "pressure_p", "pressure_b", "zgrid", "surface_pressure", "tend_sfc_pressure", "rt_diabatic_tend", "dtheta_dt_mp")),
        *(local[k] for k in ("t", "qv", "qc", "qr")),
        *(scalars[k] for k in ("rvord", "rd", "p0", "rcv", "gravity")),
        np.float32(dt_dyn), np.int32(mp_top_level), np.int32(nz), np.int32(nc)))
    return local, local_rain, last_split, work


def register(registry, *, inputs_from_pools):
    """Register with f4's SlotRegistry.

    MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:371-385.

    ``inputs_from_pools(call.pools)`` supplies state, constants, dt_dyn,
    mp_top_level, math_header. Arrays remain owned by the shared MPAS pools;
    no hidden prognostic state or import-time registry mutation is retained.
    The framework adapter supplies this view so time levels/halo extents stay
    under its ownership. SlotCall supplies the actual timestep and rain flag.
    """
    def invoke(call):
        return run_mpas(**inputs_from_pools(call.pools), dt_microp=call.dt,
                        config_bucket_rainnc=call.options["config_bucket_rainnc"],
                        l_acrain=call.flags.rain_bucket, l_diags=call.flags.diagnostics)
    registry.register("microphysics", SCHEME, BACKEND, invoke)
    return invoke


def bind_native_pools(provider, call):
    """Bind solve-cell storage at native microphysics time level two.

    MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_interface.F:598-671,826-922.
    MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:503-653.

    The shared provider owns packing, halo preservation and commit. This binder
    only selects storage; from/to arithmetic remains in run_mpas exactly once.
    """
    state = {}
    for pool, names in (
            ("state", ("theta_m", "rho_zz")),
            ("mesh", ("zz", "zgrid")),
            ("diag", ("exner", "surface_pressure", "dtheta_dt_mp")),
            ("tend", ("tend_sfc_pressure", "rt_diabatic_tend")),
            ("diag_physics", ("rainnc", "rainncv", "precipw", "i_rainnc"))):
        state.update({name: provider[pool][name] for name in names})
    state.update({name: provider["diag"][native_name] for name, native_name in (
        ("exner_b", "exner_base"), ("pressure_b", "pressure_base"),
        ("pressure_p", "pressure_p"), ("rtheta_b", "rtheta_base"),
        ("rtheta_p", "rtheta_p"))})
    scalars = provider["state"]["scalars"]
    indices = provider.session.layout.indices
    for name in ("qv", "qc", "qr"):
        state[name] = scalars[indices["state", "index_" + name] - 1]
    if call.flags.diagnostics:
        state["relhum"] = provider["diag"]["relhum"]
    aliases = {"ep2": "ep_2", "rhowater": "rho_w", "rd": "r_d"}
    constants = {name: provider.native.get_constant(aliases.get(name, name))
                 for name in ("xlv", "cp", "ep2", "svp1", "svp2", "svp3",
                              "svpt0", "rhowater", "rvord", "rd", "p0",
                              "rcv", "gravity")}
    return {"state": state, "constants": constants,
            "dt_dyn": provider["scalar_state"]["dt_dyn"],
            "mp_top_level": int(provider["tend_physics"]["mp_top_level"].item()),
            "math_header": shared_math_header()}


def run_slot(call):
    """Execute the table-selected native-pool driver callback.

    MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:317-500.
    """
    from .driver_v842 import slot_corrections
    corrections = slot_corrections(call, CORRECT_MODE_SOURCE)
    return run_mpas(**call.pools, dt_microp=call.dt,
                    config_bucket_rainnc=call.options["config_bucket_rainnc"],
                    l_acrain=call.flags.rain_bucket,
                    l_diags=call.flags.diagnostics,
                    **({"corrections": corrections} if corrections else {}))
