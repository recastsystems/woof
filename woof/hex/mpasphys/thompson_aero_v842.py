# Modified 2026-10-28: experimental HRRR-like profile and aerosol route support.
"""MPAS-A v8.4.2 aerosol optics and aerosol microphysics driver slice.

``gt_aod`` ports the complete module_mp_thompson_aerosols.F:23-210.
The aerosol state is supplied by MPAS initialization (i6), in number/kg.
These launchers do not synthesize an aerosol profile or register an incomplete
microphysics backend. The shared ``mp_thompson`` core belongs to p3-thompson.

Optics and permutation helpers use C-contiguous (column, level) buffers.
The full microphysics driver follows p3's C-contiguous (level, column)
layout. The explicit MPAS interface accepts (level, cell) buffers and
permutes on the device. Aerosol-only helpers use single-precision basic
arithmetic; the full column requires the shared device maths source.
"""

from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from typing import Any, Callable, Mapping

SCHEME = "mp_thompson_aerosols"
BACKEND = "hex-cuda"
AEROSOL_FIELDS = ("nc", "nifa", "nwfa")
CUDA_OPTIONS = ("--std=c++11", "--fmad=false", "--ftz=false", "--prec-div=true")
# module_mp_thompson.F:1138-1203 has no communication between columns.
# Only the A/B path with the column workspace off (WOOF_HEX_THOMPSON_WORKSPACE=0) still runs the
# per-thread-array kernel in bounded batches: its 46 KB frame exhausted device local memory when a
# large capture launched every column together. The default path keeps the column arrays in
# thompson_v842's pooled workspace and launches like mp_thompson (see launch_driver).
DRIVER_COLUMN_BATCH = 128

# MPAS module_mp_thompson.F:1001-1387 and aerosol call at
# mpas_atmphys_driver_microphysics.F:411-435. The shared full column
# transcription is compiled in the aerosol mode in this private module.
_AERO_DRIVER = r'''
// module_mp_thompson.F:1001-1387; mpas_atmphys_driver_microphysics.F:411-435.
// Named as thompson_v842's driver kernel: this module's own build of the shared column, so the
// shared workspace storage and fast phase split find the same prologue, call and commit.
extern "C" __global__ void mpas_thompson_step(
    float* qv,float* qc,float* qi,float* qr,float* qs,float* qg,
    float* ni,float* nr,float* th,const float* pii,const float* p,
    const float* w,const float* dz,float* rainprod,float* evapprod,
    float* re_cloud,float* re_ice,float* re_snow,
    float* nc,float* nifa,float* nwfa,
    float* rainnc,float* rainncv,float* snownc,float* snowncv,
    float* graupelnc,float* graupelncv,float* sr,
    const float* ntc,const float* muc,const float* nifa2d,const float* nwfa2d,
    int nz,int columns,float dt,int has_reqc,int has_reqi,int has_reqs,
    TABLE_DECLARATIONS) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=columns)return;
    ThompsonColumn m;
    TABLE_ASSIGNMENTS
    m.nt_c=ntc[col];m.mu_c=muc[col];
    float qv1d[THOMPSON_COLUMN_KMAX],qc1d[THOMPSON_COLUMN_KMAX],qi1d[THOMPSON_COLUMN_KMAX],qr1d[THOMPSON_COLUMN_KMAX],qs1d[THOMPSON_COLUMN_KMAX],qg1d[THOMPSON_COLUMN_KMAX];
    float ni1d[THOMPSON_COLUMN_KMAX],nr1d[THOMPSON_COLUMN_KMAX],nc1d[THOMPSON_COLUMN_KMAX],nwfa1d[THOMPSON_COLUMN_KMAX],nifa1d[THOMPSON_COLUMN_KMAX];
    float t1d[THOMPSON_COLUMN_KMAX],p1d[THOMPSON_COLUMN_KMAX],w1d[THOMPSON_COLUMN_KMAX],dz1d[THOMPSON_COLUMN_KMAX];
    float rainprod1d[THOMPSON_COLUMN_KMAX],evapprod1d[THOMPSON_COLUMN_KMAX];
    float re_qc1d[THOMPSON_COLUMN_KMAX],re_qi1d[THOMPSON_COLUMN_KMAX],re_qs1d[THOMPSON_COLUMN_KMAX];
    // MPAS :1142-1174: incoming native number/kg, including negative values.
    for(int k=0;k<nz;k++) {
        int idx=k*columns+col;
        qv1d[k]=qv[idx];qc1d[k]=qc[idx];qi1d[k]=qi[idx];qr1d[k]=qr[idx];qs1d[k]=qs[idx];qg1d[k]=qg[idx];
        ni1d[k]=ni[idx];nr1d[k]=nr[idx];
        nc1d[k]=nc[idx];nwfa1d[k]=nwfa[idx];nifa1d[k]=nifa[idx];
        t1d[k]=th[idx]*pii[idx];p1d[k]=p[idx];w1d[k]=w[idx];dz1d[k]=dz[idx];
        rainprod1d[k]=0.0f;evapprod1d[k]=0.0f;
    }
    float rain=0.0f,snow=0.0f,graupel=0.0f,ice=0.0f;
    m.mp_thompson(qv1d,qc1d,qi1d,qr1d,qs1d,qg1d,ni1d,nr1d,nc1d,nwfa1d,nifa1d,t1d,p1d,w1d,dz1d,
                  rain,snow,graupel,ice,rainprod1d,evapprod1d,1,nz,dt,1,1);
    rainncv[col]=((rain+snow)+graupel)+ice;
    // MPAS :1194-1203: preserve the left-associated accumulation order (both sums above and below).
    rainnc[col]=(((rainnc[col]+rain)+snow)+graupel)+ice;
    snowncv[col]=snow+ice;snownc[col]=(snownc[col]+snow)+ice;
    graupelncv[col]=graupel;graupelnc[col]=graupelnc[col]+graupel;
    sr[col]=((snow+graupel)+ice)/(rainncv[col]+1.0e-12f);
    // MPAS :1208-1222: both emissions occur after the column network, once
    // per subcycle, and are not clipped again after emission.
    nwfa1d[0]=nwfa1d[0]+nwfa2d[col]*dt;
    nifa1d[0]=nifa1d[0]+nifa2d[col]*dt;
    for(int k=0;k<nz;k++) {
        int idx=k*columns+col;
        nc[idx]=nc1d[k];nwfa[idx]=nwfa1d[k];nifa[idx]=nifa1d[k];
        qv[idx]=qv1d[k];qc[idx]=qc1d[k];qi[idx]=qi1d[k];qr[idx]=qr1d[k];qs[idx]=qs1d[k];qg[idx]=qg1d[k];
        ni[idx]=ni1d[k];nr[idx]=nr1d[k];th[idx]=t1d[k]/pii[idx];
        rainprod[idx]=rainprod1d[k];evapprod[idx]=evapprod1d[k];
        // MPAS :1321-1333 operates on the copied qv field; radius diagnostics
        // still use the unchanged local qv1d. Other negatives only log warnings.
        // The upper neighbor retains its incoming qv until its loop iteration.
        if(qv1d[k]<0.0f) {
            if(k+1 < nz-2 && k+1 > 2)
                qv[idx]=tc_max(1.0e-7f,0.5f*(qv[idx-columns]+qv[idx+columns]));
            else qv[idx]=1.0e-7f;
        }
    }
    if(has_reqc && has_reqi && has_reqs) {
        for(int k=0;k<nz;k++) {
            re_qc1d[k]=2.49e-6f;re_qi1d[k]=4.99e-6f;re_qs1d[k]=9.99e-6f;
        }
        m.calc_effectrad(t1d,p1d,qv1d,qc1d,nc1d,qi1d,ni1d,qs1d,re_qc1d,re_qi1d,re_qs1d,1,nz);
        for(int k=0;k<nz;k++) {
            int idx=k*columns+col;
            re_cloud[idx]=tc_max(2.49e-6f,tc_min(re_qc1d[k],50.0e-6f));
            re_ice[idx]=tc_max(4.99e-6f,tc_min(re_qi1d[k],125.0e-6f));
            re_snow[idx]=tc_max(9.99e-6f,tc_min(re_qs1d[k],999.0e-6f));
        }
    }
}
'''

CUDA_SOURCE = r'''
// module_mp_thompson_aerosols.F:23-210; mpas_atmphys_interface.F:1114-1130.
// Same binary32 payload arithmetic as p3's tc_* helpers. The strict SSE
// oracle preserves operand payloads and emits negative quiet NaN on invalid
// arithmetic; device intrinsics alone canonicalize these words.
__device__ __forceinline__ bool aero_nan(float a) { return (__float_as_uint(a)&0x7fffffffU)>0x7f800000U; }
// module_mp_thompson_aerosols.F:23-210; mpas_atmphys_interface.F:1114-1130.
__device__ __forceinline__ float aero_quiet(float a) { return __uint_as_float(__float_as_uint(a)|0x00400000U); }
// module_mp_thompson_aerosols.F:23-210, strict RKIND arithmetic adapter.
__device__ __forceinline__ float add(float a, float b) { if(aero_nan(a))return aero_quiet(a);if(aero_nan(b))return aero_quiet(b);float r=__fadd_rn(a,b);return aero_nan(r)?__uint_as_float(0xffc00000U):r; }
// module_mp_thompson_aerosols.F:23-210, strict RKIND arithmetic adapter.
__device__ __forceinline__ float sub(float a, float b) { if(aero_nan(a))return aero_quiet(a);if(aero_nan(b))return aero_quiet(b);float r=__fsub_rn(a,b);return aero_nan(r)?__uint_as_float(0xffc00000U):r; }
// module_mp_thompson_aerosols.F:23-210, strict RKIND arithmetic adapter.
__device__ __forceinline__ float mul(float a, float b) { if(aero_nan(a))return aero_quiet(a);if(aero_nan(b))return aero_quiet(b);float r=__fmul_rn(a,b);return aero_nan(r)?__uint_as_float(0xffc00000U):r; }
// module_mp_thompson_aerosols.F:23-210, strict RKIND arithmetic adapter.
__device__ __forceinline__ float divrn(float a, float b) { if(aero_nan(a))return aero_quiet(a);if(aero_nan(b))return aero_quiet(b);float r=__fdiv_rn(a,b);return aero_nan(r)?__uint_as_float(0xffc00000U):r; }

// Fortran NINT: ties away from zero, including negative arguments.
// module_mp_thompson_aerosols.F:134,146,162.
__device__ __forceinline__ int nint_mpas(float x) {
    int n = __float2int_rz(x);
    float rem = sub(x, __int2float_rn(n));
    return n + (rem >= 0.5f) - (rem <= -0.5f);
}

// src/core_atmosphere/physics/mpas_atmphys_functions.F:156-186, RKIND=single, strict Horner order.
__device__ __forceinline__ float mpas_rslf(float p, float t) {
    float x = fmaxf(-80.0f, sub(t,273.16f));
    float esl = add(0.379534310E-11f, mul(x,-0.321582393E-13f));
    esl = add(0.702620698E-8f, mul(x,esl));
    esl = add(0.203154182E-5f, mul(x,esl));
    esl = add(0.299291081E-3f, mul(x,esl));
    esl = add(0.264224321E-1f, mul(x,esl));
    esl = add(0.143177157E01f, mul(x,esl));
    esl = add(0.444606896E02f, mul(x,esl));
    esl = add(0.611583699E03f, mul(x,esl));
    esl = fminf(esl,mul(p,0.15f));
    return divrn(mul(0.622f,esl),sub(p,esl));
}

// module_mp_thompson_aerosols.F:45-116; [aerosol][temperature][RH].
__device__ __constant__ float aod_table[64] = {
    5.73936E-15f, 6.93515E-15f, 7.24707E-15f, 8.95130E-15f, 1.06695E-14f, 1.37908E-14f, 2.26019E-14f, 4.41983E-13f,
    5.73936E-15f, 6.93168E-15f, 7.23809E-15f, 9.01582E-15f, 1.06370E-14f, 1.37172E-14f, 2.24435E-14f, 3.93335E-13f,
    5.73936E-15f, 6.92570E-15f, 7.23108E-15f, 9.13216E-15f, 1.05999E-14f, 1.36362E-14f, 2.23254E-14f, 3.45569E-13f,
    5.73936E-15f, 6.91833E-15f, 7.21800E-15f, 9.16219E-15f, 1.05443E-14f, 1.35287E-14f, 2.20496E-14f, 2.96971E-13f,
    2.63577E-12f, 2.72095E-12f, 2.77219E-12f, 2.87263E-12f, 2.96752E-12f, 3.15081E-12f, 3.66798E-12f, 7.50091E-11f,
    2.63577E-12f, 2.72092E-12f, 2.77222E-12f, 2.87252E-12f, 2.96726E-12f, 3.15020E-12f, 3.66540E-12f, 6.79097E-11f,
    2.63577E-12f, 2.72091E-12f, 2.77201E-12f, 2.87241E-12f, 2.96702E-12f, 3.14927E-12f, 3.66173E-12f, 6.07845E-11f,
    2.63577E-12f, 2.72087E-12f, 2.77111E-12f, 2.87211E-12f, 2.96603E-12f, 3.14817E-12f, 3.65796E-12f, 5.36085E-11f
};
__device__ __constant__ float aod_rh[8] = {10.f,60.f,70.f,80.f,85.f,90.f,95.f,99.8f};

// Complete gt_aod, module_mp_thompson_aerosols.F:23-210.
extern "C" __global__ void gt_aod_kernel(
    const float* p, const float* dz, const float* t, const float* qv,
    const float* nwfa, const float* nifa, float* out, int nc, int nz,
    int c0, int c1, int k0, int k1) {
    int h = blockIdx.x*blockDim.x + threadIdx.x;
    if (h >= nc*nz) return;
    int c = h/nz, k = h%nz;
    if (c < c0 || c >= c1 || k < k0 || k >= k1) return;
    float rhoa = divrn(p[h],mul(287.f,t[h]));
    int ti = max(1,min(nint_mpas(sub(10.999f,mul(0.0333f,t[h]))),4))-1;
    float rh = fminf(98.f,fmaxf(10.1f,mul(divrn(qv[h],mpas_rslf(p[h],t[h])),100.f)));
    int i1, i2;
    if (rh < 60.f) {
        i1=0; i2=1;
    } else {
        int ri = rh < 80.f ? nint_mpas(add(mul(0.1f,rh),-4.f))
                          : min(8,nint_mpas(add(mul(0.2f,rh),-12.f)));
        ri -= 1;
        float rd = sub(rh,aod_rh[ri]);
        if (rd < 0.f) { i1=ri-1; i2=ri; }
        else {
            i1=ri; i2=ri+1;
            if (i2 > 7) { i2=7; i1=6; }
        }
    }
    float low = divrn(aod_rh[i1],sub(100.f,aod_rh[i1]));
    float high = divrn(aod_rh[i2],sub(100.f,aod_rh[i2]));
    float frac = fmaxf(0.f,fminf(1.f,divrn(sub(divrn(rh,sub(100.f,rh)),low),sub(high,low))));
    float bw = add(aod_table[ti*8+i1],mul(sub(aod_table[ti*8+i2],aod_table[ti*8+i1]),frac));
    float bi = add(aod_table[32+ti*8+i1],mul(sub(aod_table[32+ti*8+i2],aod_table[32+ti*8+i1]),frac));
    float aw = mul(mul(mul(bw,fmaxf(1.f,fminf(99999.E6f,nwfa[h]))),dz[h]),rhoa);
    float ai = mul(mul(mul(bi,fmaxf(0.01f,fminf(9999.E6f,nifa[h]))),dz[h]),rhoa);
    out[h] = add(aw,ai);
}

// mpas_atmphys_interface.F:730-754, :798-809. No clipping or density scaling.
extern "C" __global__ void aero_from_mpas_kernel(
    const float* nc, const float* ni, const float* nw,
    float* pc, float* pi, float* pw,
    float* tc, float* ti, float* tw, int ncol, int nz, int transpose) {
    int h = blockIdx.x*blockDim.x + threadIdx.x;
    if (h >= ncol*nz) return;
    int native = transpose ? (h%nz)*ncol+h/nz : h;
    pc[h]=nc[native]; pi[h]=ni[native]; pw[h]=nw[native];
    tc[native]=nc[native]; ti[native]=ni[native]; tw[native]=nw[native];
}

// mpas_atmphys_interface.F:1032-1056, :1114-1130.
extern "C" __global__ void aero_to_mpas_kernel(
    const float* pc, const float* pi, const float* pw,
    float* nc, float* ni, float* nw,
    float* tc, float* ti, float* tw, int ncol, int nz, int transpose, float dt_dyn) {
    int h = blockIdx.x*blockDim.x + threadIdx.x;
    if (h >= ncol*nz) return;
    int native = transpose ? (h%nz)*ncol+h/nz : h;
    nc[native]=pc[h]; ni[native]=pi[h]; nw[native]=pw[h];
    tc[native]=divrn(sub(nc[native],tc[native]),dt_dyn);
    ti[native]=divrn(sub(ni[native],ti[native]),dt_dyn);
    tw[native]=divrn(sub(nw[native],tw[native]),dt_dyn);
}

// module_mp_thompson.F:1208-1222. Source is number/kg/s, after mp_thompson.
extern "C" __global__ void aero_surface_emission_kernel(
    float* nw, float* ni, const float* sw, const float* si,
    int ncol, int nz, int kts, float dt_in) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    int h = c*nz+kts;
    nw[h]=add(nw[h],mul(sw[c],dt_in));
    ni[h]=add(ni[h],mul(si[c],dt_in));
}
'''


def _kernel(name: str) -> Any:
    """The aerosol interface kernel loaded on the calling card."""
    import cupy as cp
    # A CUDA module belongs to the card it was loaded on. Same-host ranks drive
    # several cards from one process, so a cache keyed without the card hands
    # the second card the first card's handle (CUDA_ERROR_INVALID_HANDLE at step 1).
    return _kernel_on(int(cp.cuda.Device().id), name)


@lru_cache(maxsize=None)
def _interface_module(device: int) -> Any:
    """Compile module_mp_thompson_aerosols.F:23-210 and its interface kernels."""
    import cupy as cp
    with cp.cuda.Device(device):
        return _compile_module(CUDA_SOURCE,_hex_numerics.options(CUDA_OPTIONS))


@lru_cache(maxsize=None)
def _kernel_on(device: int, name: str) -> Any:
    return _interface_module(device).get_function(name)


def _compile_module(source: str, options: tuple[str, ...]) -> Any:
    """Preserve RKIND subnormals for module_mp_thompson.F:1400-3642.

    CuPy 14.2 RawModule/RawKernel append -ftz=true after caller options.
    p3's log10-shared-cuda-scan.json witnesses the resulting flushed words.
    Direct NVRTC compilation retains --ftz=false without global mutation.
    """
    from cupy.cuda import compiler, function
    image,_=compiler.compile_using_nvrtc(source,options=options,cache_in_memory=True)
    module=function.Module()
    module.load(image)
    return module


def _fields(fields: Mapping[str, Any], *, shape: tuple[int, ...] | None = None) -> tuple[int, ...]:
    """Validate buffers for mpas_atmphys_interface.F:730-754 number copies."""
    import cupy as cp

    if not fields:
        raise ValueError("at least one device field is required")
    if shape is None:
        first = next(iter(fields.values()))
        if not isinstance(first, cp.ndarray):
            raise TypeError("MPAS fields must be CuPy device arrays")
        shape = first.shape
    if not shape or any(n <= 0 for n in shape):
        raise ValueError("MPAS device field dimensions must be positive")
    for name, array in fields.items():
        if not isinstance(array, cp.ndarray):
            raise TypeError(f"{name} must be a CuPy device array")
        if array.shape != shape or array.dtype.name != "float32" or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be float32 C-contiguous with shape {shape}")
        if array.device.id != cp.cuda.Device().id:
            raise ValueError(f"{name} is on a different CUDA device")
    return shape


def gt_aod(*, p_phy: Any, dz8w: Any, t_phy: Any, qvapor: Any,
           nwfa: Any, nifa: Any, taod5503d: Any,
           columns: tuple[int, int] | None = None,
           levels: tuple[int, int] | None = None) -> Any:
    """Complete module_mp_thompson_aerosols.F:23-210, with tile preservation.

    Tile bounds are zero-based half-open; omitted bounds cover the whole buffer.
    The output is supplied by the caller and updated only inside the tile.
    """
    shape = _fields(dict(p=p_phy, dz=dz8w, t=t_phy, qv=qvapor,
                         nwfa=nwfa, nifa=nifa, out=taod5503d))
    if len(shape) != 2:
        raise ValueError("gt_aod fields require (column, level) shape")
    nc, nz = shape
    c0, c1 = columns if columns is not None else (0, nc)
    k0, k1 = levels if levels is not None else (0, nz)
    if not (0 <= c0 <= c1 <= nc and 0 <= k0 <= k1 <= nz):
        raise ValueError("gt_aod tile exceeds the allocated device fields")
    _kernel("gt_aod_kernel")(((nc*nz+255)//256,), (256,),
                            (p_phy, dz8w, t_phy, qvapor, nwfa, nifa, taod5503d,
                             nc, nz, c0, c1, k0, k1))
    return taod5503d


def aerosol_from_mpas(native: Mapping[str, Any], scheme: Mapping[str, Any],
                      before: Mapping[str, Any], *, transpose: bool = False) -> None:
    """Aerosol X_from_MPAS, mpas_atmphys_interface.F:730-754, :798-809.

    ``before`` stores the original values for native tendency bookkeeping.
    The default preserves the seam's (level, cell) layout, as needed by the
    full driver. ``transpose=True`` prepares (column, level) optics buffers.
    The four surface arrays (nifa2d/nwfa2d/nt_c/mu_c) are direct device views
    of their corresponding single MPAS cell axis; no unit conversion applies.
    """
    shape = _fields({n: native[n] for n in AEROSOL_FIELDS})
    if len(shape) != 2:
        raise ValueError("native aerosol state requires (level, cell) shape")
    nz, nc = shape
    _fields({n: before[n] for n in AEROSOL_FIELDS}, shape=shape)
    _fields({n: scheme[n] for n in AEROSOL_FIELDS}, shape=(nc, nz) if transpose else shape)
    args = tuple(native[n] for n in AEROSOL_FIELDS) + tuple(scheme[n] for n in AEROSOL_FIELDS)
    args += tuple(before[n] for n in AEROSOL_FIELDS) + (nc, nz, int(transpose))
    _kernel("aero_from_mpas_kernel")(((nc*nz+255)//256,), (256,), args)


def aerosol_to_mpas(scheme: Mapping[str, Any], native: Mapping[str, Any],
                    before: Mapping[str, Any], *, dt_dyn: float, transpose: bool = False) -> None:
    """Aerosol X_to_MPAS, mpas_atmphys_interface.F:1032-1056, :1114-1130.

    ``before`` becomes rncmpten/rnifampten/rnwfampten, using the dynamical
    timestep even when the scheme is subcycled with a different dt_microp.
    MPAS does not zero aerosol state or these tendencies above mp_top_level.
    """
    import numpy as np

    shape = _fields({n: native[n] for n in AEROSOL_FIELDS})
    if len(shape) != 2:
        raise ValueError("native aerosol state requires (level, cell) shape")
    nz, nc = shape
    _fields({n: before[n] for n in AEROSOL_FIELDS}, shape=shape)
    _fields({n: scheme[n] for n in AEROSOL_FIELDS}, shape=(nc, nz) if transpose else shape)
    dt = np.float32(dt_dyn)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt_dyn must be finite and positive for aerosol tendencies")
    args = tuple(scheme[n] for n in AEROSOL_FIELDS) + tuple(native[n] for n in AEROSOL_FIELDS)
    args += tuple(before[n] for n in AEROSOL_FIELDS) + (nc, nz, int(transpose), dt)
    _kernel("aero_to_mpas_kernel")(((nc*nz+255)//256,), (256,), args)


def surface_emission(*, nwfa: Any, nifa: Any, nwfa2d: Any, nifa2d: Any,
                     dt_in: float, kts: int = 0) -> None:
    """Post-microphysics emission, module_mp_thompson.F:1208-1222."""
    import numpy as np

    shape = _fields(dict(nwfa=nwfa, nifa=nifa))
    if len(shape) != 2:
        raise ValueError("aerosol state requires (column, level) shape")
    nc, nz = shape
    _fields(dict(nwfa2d=nwfa2d, nifa2d=nifa2d), shape=(nc,))
    if not 0 <= kts < nz:
        raise ValueError("emission level exceeds the allocated aerosol state")
    dt = np.float32(dt_in)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt_in must be finite and positive for surface emissions")
    _kernel("aero_surface_emission_kernel")(((nc+255)//256,), (256,),
                                          (nwfa, nifa, nwfa2d, nifa2d, nc, nz, kts, dt))


def _shared_core() -> Any:
    """Resolve p3's complete module_mp_thompson.F:1400-3642 MPAS column."""
    from . import thompson_v842
    return thompson_v842


#: The aerosol auxiliary arrays native initialisation leaves (thompson_init with is_aerosol_aware, then the
#: CCN_ACTIVATE_DATA read of mpas_atmphys_init_microphysics.F:251-310), in TABLE_ARGUMENTS[27:] order.
#: ``woof hex physics-resources build`` generates them and refuses other bytes.
AUXILIARY_SHA256 = "25d8d171fed2641f2bb2350b5d16bac2fa04f10f2283e2fad59283e87078f305"


def load_resources(resources: Mapping[str, Any], device: Any) -> Mapping[str, Any]:
    """Bind the complete initialized aerosol packet without classic CCN defaults.

    MPAS module_mp_thompson.F:492-497,624-960;
    mpas_atmphys_init_microphysics.F:132-138,256-310.
    Rust decodes records; Python authenticates and transfers their original
    words. The auxiliary packet is the strict aerosol initializer's export,
    including CCN_ACTIVATE_DATA, in the declared TABLE_ARGUMENTS order.
    """
    from pathlib import Path
    from .table_resources_v842 import RustLookupRecords, certified_path

    core = _shared_core()
    decoder = RustLookupRecords(resources["rust_decoder"])
    groups = {
        "MP_THOMPSON_QRacrQG_DATA.DBL": core.TABLE_ARGUMENTS[:6],
        "MP_THOMPSON_QRacrQS_DATA.DBL": core.TABLE_ARGUMENTS[6:18],
        "MP_THOMPSON_freezeH2O_DATA.DBL": core.TABLE_ARGUMENTS[18:24],
        "MP_THOMPSON_QIautQS_DATA.DBL": core.TABLE_ARGUMENTS[24:27],
    }
    host = {}
    for filename, names in groups.items():
        path = certified_path(dict(path=Path(resources["table_directory"])/filename,
                                   sha256=core.TABLE_FILE_SHA256[filename]))
        host.update(decoder.arrays(path, names, core.TABLE_SHAPES,
                                   records=True, little=False))
    # MPAS :132-138 loads actual activation data, replacing :837's unit values.
    # This immutable native export uses the aerosol bridge's declared order;
    # the classic packet has another order and deliberately retains unit CCN.
    auxiliary = certified_path(dict(path=resources["auxiliary"], sha256=AUXILIARY_SHA256))
    host.update(decoder.arrays(auxiliary, core.TABLE_ARGUMENTS[27:], core.TABLE_SHAPES,
                               records=False, little=True))
    tables = {name: device.asarray(host[name], order="F") for name in core.TABLE_ARGUMENTS}
    tables["_file_sha256"] = dict(core.TABLE_FILE_SHA256)
    maths = certified_path(resources["device_math_source"]).read_text()
    return dict(tables=tables, device_math_source=maths)


def cuda_source(*, device_math_source: str, max_levels: int, workspace: bool = False,
                numerics: str | None = None) -> str:
    """Compile the whole p3 MPAS column with aerosol branches enabled.

    MPAS module_mp_thompson.F:1400-3642,4176-4212,4391-4466,4662-4795,
    4809-4913. This specialization changes only the MPAS module's public
    is_aerosol_aware flag, set by native init_thompson_aerosols_forMPAS, and
    the driver (aerosol carriers in and out, surface emission). It never uses
    the frozen WRF aerosol kernel family. ``workspace`` takes the shared column
    workspace (and, in fast numerics, the shared phase split) exactly as
    mp_thompson does: storage and launch shape only, every statement unchanged.
    ``numerics`` lets the offline kernel pack compose both modes in one process (as mp_thompson's).
    """
    core = _shared_core()
    source = core.cuda_source(device_math_source=device_math_source, max_levels=max_levels,
                              workspace=workspace, numerics=numerics, driver=_AERO_DRIVER)
    marker = "static constexpr bool is_aerosol_aware = false;"
    if source.count(marker) != 1:
        raise RuntimeError("p3 Thompson source no longer exposes the aerosol mode declaration; "
                           "specialization must be reconciled to avoid compiling the classic branch")
    return source.replace(marker, "static constexpr bool is_aerosol_aware = true;")


@lru_cache(maxsize=1)
def workspace_layout():
    """Workspace slots per kernel of this module's source (levels, maths and numerics do not change them)."""
    core = _shared_core()
    declarations = ",".join(f"const {'float' if n == 'tnccn_act' else 'double'}* input_{n}"
                            for n in core.TABLE_ARGUMENTS)
    assignments = "\n".join(f"m.{n}=input_{n};" for n in core.TABLE_ARGUMENTS)
    driver = _AERO_DRIVER.replace("TABLE_DECLARATIONS", declarations).replace("TABLE_ASSIGNMENTS", assignments)
    return core._workspace_source(f"#define THOMPSON_COLUMN_KMAX {core.MAX_LEVELS}\n" + core._SUPPORT
                                  + core._COLUMN + driver + core._RADAR_DRIVER)[1]


# Unbounded on purpose (one entry per card and variant): a bounded per-card cache thrashes once ranks x
# variants exceed it, and every rank then rebuilds its module each step (thompson_v842._module, 8 cards,
# 2026-10-28: 1.28 s/step against 0.25 s with the cache fixed).
@lru_cache(maxsize=None)
def _aero_module(device: int, device_math_source: str, max_levels: int, workspace: bool = False,
                 numerics: str = "strict") -> Any:
    """Compile module_mp_thompson.F:1001-1387,1400-3642 on the selected device (``numerics`` keys the cache)."""
    import cupy as cp
    core = _shared_core()
    with cp.cuda.Device(device):
        return core.compile_module(cuda_source(device_math_source=device_math_source,
                                               max_levels=max_levels, workspace=workspace))


def launch_driver(fields: Mapping[str, Any], surface: Mapping[str, Any],
                  tables: Mapping[str, Any], *, dt: float, device_math_source: str,
                  has_reqc: int = 1, has_reqi: int = 1, has_reqs: int = 1,
                  workspace: bool | None = None) -> None:
    """MPAS mp_gt_driver aerosol call, module_mp_thompson.F:1001-1387.

    Full driver buffers follow p3: float32 C-contiguous (level, column).
    They include nc/nifa/nwfa and both native surface source tendencies.
    The outer MPAS driver owns n_microp subcalls and accumulated diagnostics.
    THE BREAKAGE THE WORKSPACE LAUNCH PREVENTS: the per-thread-array kernel ran 128 columns per launch
    (21 launches a call on x1.2562, each with 21 slice copies in and out) with a 46 KB frame whose
    local-memory store was reset and regrown at every microphysics call: x1.2562 24 h fast,
    convection_permitting + Thompson-aerosol 6.83 s a step against 0.065 s with mp_thompson (105x).
    """
    import cupy as cp
    import numpy as np
    core = _shared_core()
    if tables.get("_file_sha256") != core.TABLE_FILE_SHA256:
        raise ValueError("Rust table preparation must certify the f0 strict MPAS lookup bytes; WRF table values change collision rates")
    shape = _fields({n:fields[n] for n in (*core.FIELD_ARGUMENTS,*AEROSOL_FIELDS)})
    if len(shape) != 2 or not 1 <= shape[0] <= core.MAX_LEVELS:
        raise ValueError("Thompson aerosol driver requires (level, column) buffers within p3 capacity")
    nz,nc = shape
    surface_names = (*core.SURFACE_ARGUMENTS,"nifa2d","nwfa2d")
    _fields({n:surface[n] for n in surface_names},shape=(nc,))
    for name,expected in core.TABLE_SHAPES.items():
        array = tables[name]
        dtype = cp.float32 if name == "tnccn_act" else cp.float64
        if (not isinstance(array,cp.ndarray) or array.shape != expected or array.dtype != dtype
                or not array.flags.f_contiguous or array.device.id != cp.cuda.Device().id):
            raise ValueError(f"{name} must preserve MPAS table shape {expected}, precision, Fortran order and device")
    dt_in = np.float32(dt)
    if not np.isfinite(dt_in) or dt_in <= 0:
        raise ValueError("positive finite dt required by MPAS reciprocal timestep")
    field_names = (*core.FIELD_ARGUMENTS,*AEROSOL_FIELDS)
    workspace = core.workspace_enabled() if workspace is None else workspace
    module = _aero_module(cp.cuda.Device().id,device_math_source,nz,workspace,_hex_numerics.mode())
    scalars = (np.int32(nz),np.int32(nc),dt_in,np.int32(has_reqc),np.int32(has_reqi),np.int32(has_reqs))
    table_args = tuple(tables[n] for n in core.TABLE_ARGUMENTS)
    if workspace:
        # Every column in one launch (or workspace tiles), as mpas_thompson_step runs. A carrier that is
        # not one contiguous (level, column) block gets one device copy in and out around the call.
        work = {n:(fields[n] if fields[n].flags.c_contiguous else cp.ascontiguousarray(fields[n]))
                for n in field_names}
        args = (tuple(work[n] for n in field_names) + tuple(surface[n] for n in surface_names)
                + scalars + table_args)
        core._launch_columns("mpas_thompson_step",module,args,nz,nc,True,layout=workspace_layout())
        for n in field_names:
            if work[n] is not fields[n]:
                fields[n][...] = work[n]
        return
    kernel = module.get_function("mpas_thompson_step")
    for first in range(0,nc,DRIVER_COLUMN_BATCH):
        last = min(first+DRIVER_COLUMN_BATCH,nc)
        # A level-major slice of a larger column array needs a contiguous
        # device copy. MPAS :1138-1203 reads/writes each column independently;
        # no field arithmetic, host staging or value normalization occurs here.
        batch_fields = {n:cp.ascontiguousarray(fields[n][:,first:last]) for n in field_names}
        batch_surface = {n:surface[n][first:last] for n in surface_names}
        args = tuple(batch_fields[n] for n in field_names)
        args += tuple(batch_surface[n] for n in surface_names)
        args += (np.int32(nz),np.int32(last-first),dt_in,np.int32(has_reqc),np.int32(has_reqi),np.int32(has_reqs))
        args += table_args
        kernel(((last-first+31)//32,),(32,),args)
        for name in field_names:
            fields[name][:,first:last] = batch_fields[name]


def microphysics_substeps(dt_dyn: float) -> tuple[int, Any]:
    """RKIND/NINT timestep rule, mpas_atmphys_manager.F:681-694.

    The ninety-second reference is a native scheme constant, not a maximum:
    NINT rounds the ratio, then MPAS divides the complete dynamical timestep.
    """
    import numpy as np
    dt = np.float32(dt_dyn)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("positive finite dynamical timestep required by MPAS manager")
    ratio = np.float32(dt / np.float32(90.))
    whole = int(ratio)
    substeps = max(whole + int(np.float32(ratio-np.float32(whole)) >= np.float32(.5)),1)
    dt_microp = np.float32(dt/np.float32(substeps))
    if dt <= dt_microp:
        dt_microp = dt
    return substeps,dt_microp


def register_backend(registry: Any, *, arguments: Callable[[Any], Mapping[str, Any]],
                     prepare: Callable[[Any], None], finish: Callable[[Any], None]) -> None:
    """Register the aerosol slot against the f4 immutable SlotCall contract.

    MPAS mpas_atmphys_driver_microphysics.F:362-498. ``prepare`` and ``finish``
    bind the shared p3 conversions/precipitation/diagnostics and this module's
    aerosol_from_mpas/aerosol_to_mpas to resident MPAS pools. ``arguments``
    supplies launch_driver's fields, surface, tables, shared device maths and
    radius flags as device views. These providers must keep persistent state
    in call.pools; this callback retains no mutable numerical state. Their
    full-model bindings are integrator-owned and require native proof.
    """
    if not all(callable(f) for f in (arguments,prepare,finish)):
        raise TypeError("MPAS aerosol backend requires executable pool binding and conversion hooks")

    def slot(call: Any) -> None:
        """Execute mpas_atmphys_driver_microphysics.F:411-435 subcalls."""
        prepare(call)
        inputs = dict(arguments(call))
        n_microp,dt_microp = microphysics_substeps(call.dt)
        for _ in range(n_microp):
            launch_driver(**inputs,dt=dt_microp)
        finish(call)

    registry.register("microphysics",SCHEME,BACKEND,slot)


def bind_native_pools(provider: Any, call: Any) -> Mapping[str, Any]:
    """Bind aerosol views to the real f4 native pool provider.

    MPAS mpas_atmphys_interface.F:730-754,798-809,1032-1056,1114-1130;
    mpas_atmphys_driver_microphysics.F:411-435;
    mpas_atmphys_vars.F:185-218,934-942.

    The common p3 binder only binds storage; it executes no classic physics.
    The owner supplies its certified tables and approved mathematics in
    provider.thompson_v842_resources. Aerosol state and emissions are native
    restart-owned views, with no synthetic profile or host arithmetic.
    """
    from dataclasses import replace

    if call.slot != "microphysics" or call.scheme != SCHEME or str(call.backend) != BACKEND:
        raise ValueError("native aerosol binder requires its registered microphysics slot")
    core = _shared_core()
    common = core.bind_native_pools(provider, replace(call, scheme=core.SCHEME))
    p = common["thompson_v842"]
    try:
        scalars = provider["state"]["scalars"]
        for name, tendency in zip(AEROSOL_FIELDS, ("rncmpten", "rnifampten", "rnwfampten")):
            index = provider.session.layout.indices["state", "index_" + name]
            if not 1 <= index <= scalars.shape[0]:
                raise ValueError(f"native index_{name} must address its 1-based scalar component")
            p["model"][name] = scalars[index - 1]
            # mpas_atmphys_driver_microphysics.F:163-165 allocates kms:kme;
            # mpas_atmphys_interface.F:730-754 uses the active kts:kte tile.
            # The native allocation includes the extra interface level.
            p["fields"][name] = provider["prepared"][name + "_p"][:p["model"][name].shape[0]]
            p["tendencies"][name] = provider["tend_physics"][tendency]
        for name in ("nifa2d", "nwfa2d"):
            p["surface"][name] = provider["diag_physics"][name]
    except BaseException:
        core._finish_native_workspace(p, commit=False)
        raise
    return {"thompson_aero_v842": p}


def register(registry: Any) -> None:
    """Register the published f4 slot, mpas_atmphys_driver_microphysics.F:411-435."""
    registry.register("microphysics", SCHEME, BACKEND, run_slot)


def run_slot(call: Any) -> None:
    """Commit native aerosol words before releasing owned prepared arrays.

    MPAS mpas_atmphys_driver_microphysics.F:359-498;
    mpas_atmphys_interface.F:826-1138. The common binder carries ownership
    through this aerosol mapping; caller-owned and replay storage is retained.
    """
    if call.slot != "microphysics" or call.scheme != SCHEME or str(call.backend) != BACKEND:
        raise ValueError("Thompson aerosol callback requires its exact registered MPAS slot")
    core = _shared_core()
    p = call.pools["thompson_aero_v842"]
    try:
        _run_bound_slot(call)
    except BaseException:
        core._finish_native_workspace(p, commit=False)
        raise
    core._finish_native_workspace(p, commit=True)


def _run_bound_slot(call: Any) -> None:
    """Full aerosol boundary, mpas_atmphys_driver_microphysics.F:317-498.

    The integrator binds restart-owned device views in
    call.pools['thompson_aero_v842']. Its common mappings follow p3; model,
    fields and tendencies additionally contain nc/nifa/nwfa, with tendency
    entries pointing at rncmpten/rnifampten/rnwfampten. surface contains
    nifa2d/nwfa2d from diag_physics. Native manager bindings supply n_microp
    and dt_microp. No field arithmetic runs on the host.
    """
    import cupy as cp
    import numpy as np

    if call.slot != "microphysics" or call.scheme != SCHEME or str(call.backend) != BACKEND:
        raise ValueError("Thompson aerosol callback requires its exact registered MPAS slot")
    core = _shared_core()
    p = call.pools["thompson_aero_v842"]
    if type(p["n_microp"]) is not int or p["n_microp"] < 1:
        raise ValueError("native manager must supply a positive integer microphysics substep count")
    maths, fields = p["device_math_source"], p["fields"]
    conversion = (p["model"], p["mesh"], p["diagnostics"], p["tendencies"], fields, p["constants"])
    core.launch_precip("from", fields, p["surface"], p["precip"], device_math_source=maths,
                       l_acrain=p["l_acrain"], bucket=call.options["config_bucket_rainnc"])
    core.launch_conversion("from", *conversion, device_math_source=maths)
    aerosol_from_mpas(p["model"], fields, p["tendencies"])
    for _ in range(p["n_microp"]):
        launch_driver(fields, p["surface"], p["tables"], dt=p["dt_microp"],
                      device_math_source=maths,
                      **{n:p[n] for n in ("has_reqc", "has_reqi", "has_reqs")})
    if call.flags.diagnostics:
        core.launch_radar(fields, p["diagnostics"], device_math_source=maths)
    if p.get("nest_look") is not None:
        # A trigger look's reflectivity (mpasphys/nest_look_v842): scratch only off a diagnostics step.
        p["nest_look"].form(fields, p["diagnostics"], diagnostics_formed=bool(call.flags.diagnostics),
                            launch_radar=core.launch_radar, device_math_source=maths)
    if call.flags.diagnostics:
        nz, columns = fields["qv"].shape
        relhum = p["diagnostics"]["relhum"]
        _fields(dict(relhum=relhum), shape=(nz, columns))
        args = tuple(fields[n] for n in ("qv", "th", "pii", "p")) + (relhum, np.int32(nz*columns))
        core._module(cp.cuda.Device().id, maths, nz).get_function("mpas_thompson_relhum")(
            ((nz*columns+31)//32,), (32,), args)
    core.launch_precip("to", fields, p["surface"], p["precip"], device_math_source=maths,
                       l_acrain=p["l_acrain"], bucket=call.options["config_bucket_rainnc"])
    core.launch_conversion("to", *conversion, device_math_source=maths,
                           mp_top_level=p["mp_top_level"], dt_dyn=call.dt)
    aerosol_to_mpas(fields, p["model"], p["tendencies"], dt_dyn=call.dt)
