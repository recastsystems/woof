"""Pinned RRFS radiation siblings; importing this module changes no defaults.

RRFSx/MPAS-Model 1d6e370e, TEMPO 9adf9ef, MYNN-EDMF c942418.
Public MPAS profiles are [level, cell]; local arrays are [cell, level].
Pristine LW, spectral aerosol optics and gt_aod match v8.4.2 byte for byte.
Those numerical kernels are supplied by p21/p22/p23. The SW solar delta
runs before the wrapper's day/night selection on every call.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

import numpy as np

CLOUD_SCHEMES = ("cld_incidence", "cld_fraction", "cld_fraction_thompson",
                 "cld_fraction_mynn", "cld_tempo")
RADIUS_SCHEMES = ("mp_thompson", "mp_thompson_aerosols", "mp_wsm6",
                  "mp_nssl2m", "mp_tempo")
CUDA_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                "--prec-div=true", "--prec-sqrt=true")
_JULIAN = ContextVar("rrfs_radiation_julian")

CUDA_SOURCE = r'''
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_sw.F:438-439 @ 1d6e370e
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_lw.F:562 @ 1d6e370e
extern "C" __global__ void degrees(int n,float degrad,const float *radians,float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=FDIV(radians[i],degrad);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_ra_rrtmg_sw.F:10208-10219 @ 1d6e370e
__device__ float rrfs_eot(float julian) {
    float da=FDIV(FMUL(6.2831853071795862f,FSUB(julian,1.f)),365.f);
    float two=FMUL(2.f,da);
    float v=FADD(.000075f,FMUL(.001868f,glibc_cosf(da)));
    v=FSUB(v,FMUL(.032077f,glibc_sinf(da)));
    v=FSUB(v,FMUL(.014615f,glibc_cosf(two)));
    v=FSUB(v,FMUL(.04089f,glibc_sinf(two)));
    return FMUL(v,229.18f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_ra_rrtmg_sw.F:10208-10219 @ 1d6e370e
extern "C" __global__ void solar(int n,float julian,float xtime,float gmt,
    float radt,float degrad,float declin,const float *lat,const float *lon,
    float *coszr,float *eot) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float eq=rrfs_eot(julian);
    float xt24=FADD(fmodf(FADD(xtime,FMUL(radt,.5f)),1440.f),eq);
    float local=FADD(FADD(gmt,FDIV(xt24,60.f)),FDIV(lon[i],15.f));
    float hrang=FMUL(FMUL(15.f,FSUB(local,12.f)),degrad);
    float xxlat=FMUL(lat[i],degrad);
    coszr[i]=FADD(FMUL(glibc_sinf(xxlat),glibc_sinf(declin)),
        FMUL(FMUL(glibc_cosf(xxlat),glibc_cosf(declin)),glibc_cosf(hrang)));
    if(eot)eot[i]=eq;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F:173-257 @ 1d6e370e
extern "C" __global__ void cloud_from(int nc,int nz,float len,int mynn,int step,
    int gfl,const float *density,const float *land,const float *v,const float *c,
    const float *ice,const float *snow,const float *bl,const float *cu,
    float *dx,float *xland,float *qv,float *qc,float *qi,float *qs,float *cf) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nc*nz)return;
    qv[i]=v[i];qc[i]=c[i];qi[i]=ice[i];qs[i]=snow[i];float frac=0.f;
    if(mynn) {
        if(step>0) {if(bl[i]>.001f)frac=bl[i];}
        else if(c[i]>1.e-6f||ice[i]>1.e-9f||snow[i]>1.e-8f)frac=1.f;
    }
    if(gfl&&cu[i]>.001f)frac=fminf(1.f,FADD(frac,cu[i]));
    cf[i]=frac;
    if(i%nz==0) {
        int col=i/nz;
        dx[col]=FMUL(FDIV(len,gfk_pow(density[col],.25f)),.001f);
        xland[col]=land[col];
    }
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F:508-577 @ 1d6e370e
extern "C" __global__ void cloud_condensate(int n,const float *cf,
    const float *added_c,const float *added_i,float *qc,float *qi) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(qc[i]<1.e-6f&&cf[i]>.001f)qc[i]=FADD(qc[i],added_c[i]);
    if(qi[i]<1.e-9f&&cf[i]>.001f)qi[i]=FADD(qi[i],added_i[i]);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_lw.F:459-497,759-784 @ 1d6e370e
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_sw.F:536-571 @ 1d6e370e
extern "C" __global__ void radii(int n,int enabled,const float *a,
    const float *b,const float *c,float *x,float *y,float *z) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    x[i]=enabled?a[i]:0.f;y[i]=enabled?b[i]:0.f;z[i]=enabled?c[i]:0.f;
}
'''


def _cp():
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F:76-90 @ 1d6e370e."""
    import cupy
    return cupy


def _module(eot_day_offset=1.0, eot_second_sine=0.04089):
    """The RRFS cloudiness module loaded on the calling card."""
    # A CUDA module belongs to the card it was loaded on. Same-host ranks drive
    # several cards from one process, so a cache keyed without the card hands
    # the second card the first card's handle (CUDA_ERROR_INVALID_HANDLE at step 1).
    return _module_on(int(_cp().cuda.Device().id), eot_day_offset, eot_second_sine)


@lru_cache(maxsize=None)
def _module_on(device: int, eot_day_offset=1.0, eot_second_sine=0.04089):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F:291-370 @ 1d6e370e."""
    import woof.core.kernels
    from cupy.cuda import compiler, function
    root = Path(woof.core.kernels.__file__).parent
    source = (root / "glibc_flt32.cuh").read_text()
    # RRFS 44c0849 module_ra_rrtmg_sw.F:10207-10210 moves the day origin
    # to noon and corrects the second sine coefficient. Values come from the row.
    solar_source = CUDA_SOURCE.replace('FSUB(julian,1.f)', f'FSUB(julian,{eot_day_offset}f)')
    solar_source = solar_source.replace('.04089f', f'{eot_second_sine}f')
    source += (root / "glibc_trig_flt32.cuh").read_text() + solar_source
    binary, _ = compiler.compile_using_nvrtc(source, options=_hex_numerics.options(CUDA_OPTIONS),
        arch=_cp().cuda.Device(device).compute_capability)
    with _cp().cuda.Device(device):
        module = function.Module()
        module.load(binary)
    return module


def _launch(name, count, args, *, eot_day_offset=1.0, eot_second_sine=0.04089):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F:291-370 @ 1d6e370e."""
    if count:
        _module(eot_day_offset, eot_second_sine).get_function(name)(((int(count) + 127) // 128,), (128,), args)


def _array(value, shape, name):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F:141-158 @ 1d6e370e."""
    cp = _cp()
    if not isinstance(value, cp.ndarray) or value.dtype != cp.float32:
        raise TypeError(f"{name} must be device float32; prevents an RKIND conversion")
    if value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous {shape}; prevents stride/halo corruption")
    return value


def solar_geometry(*, julian, xtime, gmt, radt, degrad, declin, xlat, xlong,
                   coszr=None, with_eot=False, eot_day_offset=1.0, eot_second_sine=0.04089):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_ra_rrtmg_sw.F:10208-10219 @ 1d6e370e."""
    cp = _cp()
    nc = xlat.size
    _array(xlat, (nc,), "xlat"); _array(xlong, (nc,), "xlong")
    out = cp.empty(nc, dtype=cp.float32) if coszr is None else _array(coszr, (nc,), "coszr")
    eot = cp.empty(nc, dtype=cp.float32) if with_eot else np.uint64(0)
    _launch("solar", nc, (np.int32(nc), *(np.float32(v) for v in
        (julian, xtime, gmt, radt, degrad, declin)), xlat, xlong, out, eot),
        eot_day_offset=eot_day_offset, eot_second_sine=eot_second_sine)
    return (out, eot) if with_eot else out


def angles_to_degrees(radians, degrad):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_sw.F:438-439 @ 1d6e370e."""
    cp = _cp()
    radians = cp.ascontiguousarray(radians)
    _array(radians, (radians.size,), "radians")
    out = cp.empty_like(radians)
    _launch("degrees", radians.size, (np.int32(radians.size),np.float32(degrad),radians,out))
    return out


def driver_cloudiness(*, scheme, itimestep, convection_scheme, qv, qc, qi, qs,
        t, pres, pres_hyd, rho, meshDensity, xland, len_disp, f_qc, f_qi,
        cldfrac_bl=None, qc_bl=None, qi_bl=None, cldfrac_cu=None,
        qc_cu=None, qi_cu=None):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F:141-370,508-577 @ 1d6e370e.

    cld_tempo deliberately follows the pinned default arm: cf=0, unchanged
    condensates, then optional GFL handling. No TEMPO cloud law is invented.
    """
    from .. import radshared_v842 as shared
    cp = _cp()
    scheme = scheme.strip()
    convection_scheme = convection_scheme.strip()
    if scheme not in CLOUD_SCHEMES:
        raise ValueError("unregistered cloudiness option: " + str(scheme))
    if qv.ndim != 2 or qv.shape[1] == 0:
        raise ValueError("profiles need a level extent; prevents invalid native bounds")
    nc, nz = qv.shape
    profiles = [_array(a, qv.shape, n) for n, a in
        (("qv",qv),("qc",qc),("qi",qi),("qs",qs),("t",t),
         ("pres",pres),("pres_hyd",pres_hyd),("rho",rho))]
    _array(meshDensity, (nc,), "meshDensity"); _array(xland, (nc,), "xland")
    mynn = scheme == "cld_fraction_mynn"
    gfl = convection_scheme == "cu_grell_freitas_li"
    null = np.uint64(0)
    if mynn:
        for n, a in (("cldfrac_bl",cldfrac_bl),("qc_bl",qc_bl),("qi_bl",qi_bl)):
            _array(a, qv.shape, n)
    if gfl:
        for n, a in (("cldfrac_cu",cldfrac_cu),("qc_cu",qc_cu),("qi_cu",qi_cu)):
            _array(a, qv.shape, n)
    out = {n: cp.empty_like(qv) for n in ("qvrad","qcrad","qirad","qsrad","cldfrac")}
    out.update(dx=cp.empty(nc,cp.float32), xland=cp.empty(nc,cp.float32))
    _launch("cloud_from", nc*nz, (np.int32(nc),np.int32(nz),np.float32(len_disp),
        np.int32(mynn),np.int32(itimestep),np.int32(gfl),meshDensity,xland,
        *profiles[:4],cldfrac_bl if mynn else null,cldfrac_cu if gfl else null,
        out["dx"],out["xland"],out["qvrad"],out["qcrad"],out["qirad"],
        out["qsrad"],out["cldfrac"]))
    if scheme == "cld_fraction_thompson":
        scratch = cp.empty((nc,nz,3),cp.float32)
        shared._launch("cloud_thompson",nc,(np.int32(nc),np.int32(nz),out["qvrad"],
            out["qcrad"],out["qirad"],out["qsrad"],pres_hyd,t,rho,
            out["xland"],out["dx"],out["cldfrac"],scratch))
    elif scheme in ("cld_incidence","cld_fraction"):
        shared._launch("cloud_simple",nc*nz,(np.int32(nc*nz),
            np.int32(CLOUD_SCHEMES.index(scheme)),np.int32(f_qc),np.int32(f_qi),
            out["qvrad"],out["qcrad"],out["qirad"],out["qsrad"],t,pres,out["cldfrac"]))
    elif mynn:
        _launch("cloud_condensate",nc*nz,(np.int32(nc*nz),out["cldfrac"],
            qc_bl,qi_bl,out["qcrad"],out["qirad"]))
    if gfl:
        _launch("cloud_condensate",nc*nz,(np.int32(nc*nz),out["cldfrac"],
            qc_cu,qi_cu,out["qcrad"],out["qirad"]))
    if mynn:
        out.update(qcbl=qc_bl, qibl=qi_bl)
    if gfl:
        out.update(qccu=qc_cu, qicu=qi_cu)
    return out


def radii_from_MPAS(*, microp_scheme, microp_re, re_cloud, re_ice, re_snow):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_lw.F:459-497 @ 1d6e370e.

    Returns input radii and zero-initialized diagnostic radii. The presence
    flags remain manager-owned. Radius units remain meters.
    """
    if microp_scheme.strip() not in RADIUS_SCHEMES:
        return None
    cp = _cp()
    arrays = [_array(a,re_cloud.shape,n) for n,a in
        (("re_cloud",re_cloud),("re_ice",re_ice),("re_snow",re_snow))]
    out = tuple(cp.empty_like(re_cloud) for _ in range(3))
    _launch("radii",re_cloud.size,(np.int32(re_cloud.size),np.int32(microp_re),*arrays,*out))
    return out, tuple(cp.zeros_like(re_cloud) for _ in range(3))


def radii_to_MPAS(*, microp_scheme, microp_re, rre_cloud, rre_ice, rre_snow):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_lw.F:759-784 @ 1d6e370e."""
    result = radii_from_MPAS(microp_scheme=microp_scheme,microp_re=microp_re,
        re_cloud=rre_cloud,re_ice=rre_ice,re_snow=rre_snow)
    return None if result is None else result[0]


@lru_cache(maxsize=None)
def shortwave_class(*, eot_day_offset=1.0, eot_second_sine=0.04089):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_ra_rrtmg_sw.F:10026-10674 @ 1d6e370e.

    Lazy dependency import permits standalone source/control inspection
    before the p23 module lands. No globals in the v8.4.2 port are patched.
    """
    from ..rrtmg_sw_v842 import MPASShortwave, PROFILE_OUTPUTS

    class RRFSShortwave(MPASShortwave):
        """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_ra_rrtmg_sw.F:10026-10674 @ 1d6e370e."""

        def _launch(self, name, count, args):
            """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_ra_rrtmg_sw.F:10208-10219 @ 1d6e370e."""
            super()._launch(name,count,args)
            if name == "mpas_sw_prep" and count:
                # Source-identical sounding prep runs first, then its sole
                # fork delta replaces coszr before any day/night decision.
                solar_geometry(julian=_JULIAN.get(),xtime=args[8],gmt=args[9],
                    radt=args[10],degrad=args[11],declin=args[12],
                    xlat=args[14][0],xlong=args[14][1],coszr=args[-1],
                    eot_day_offset=eot_day_offset,eot_second_sine=eot_second_sine)

        def rrtmg_swrad(self, *args, julian, **kwargs):
            """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_ra_rrtmg_sw.F:10036-10055,10208-10219 @ 1d6e370e."""
            token = _JULIAN.set(np.float32(julian))
            try:
                return super().rrtmg_swrad(*args,**kwargs)
            finally:
                _JULIAN.reset(token)

        def driver_controls(self, configs, *, curr_julday, **kwargs):
            """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_sw.F:950-1026 @ 1d6e370e."""
            result = super().driver_controls(configs,curr_julday=curr_julday,**kwargs)
            result["julian"] = np.float32(curr_julday)
            return result

        def radiation_sw_from_MPAS(self, fields, surface, p8w, t8w, *, lat_cell,
                lon_cell, microp_scheme, config_microp_re,
                config_tempo_aerosolaware, nwfa=None, nifa=None, ht=None,
                aerosol_qv=None, **options):
            """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_sw.F:374-808,1020-1030 @ 1d6e370e."""
            cp = self.cp
            microp_scheme = microp_scheme.strip()
            converted = {n:cp.ascontiguousarray(a.T) for n,a in fields.items()}
            surfaces = dict(surface)
            surfaces.update(xlat=angles_to_degrees(lat_cell, options["degrad"]),
                xlong=angles_to_degrees(lon_cell, options["degrad"]))
            diagnosed = bool(config_microp_re) and microp_scheme in RADIUS_SCHEMES
            if not diagnosed:
                for n in ("re_cloud","re_ice","re_snow"):
                    converted[n] = cp.zeros_like(converted["p3d"])
            aware = microp_scheme == "mp_thompson_aerosols" or (
                microp_scheme == "mp_tempo" and config_tempo_aerosolaware)
            aer = None
            if aware:
                if any(a is None for a in (nwfa,nifa,ht,aerosol_qv)):
                    raise ValueError("native aerosol path needs nwfa/nifa, original qv_p and zgrid surface height")
                # gt_aod and calc_aerosol_rrtmg_sw are source-identical;
                # TEMPO uses original qv, not the radiation carrier.
                aer = self.thompson_aerosols(converted["p3d"],converted["t3d"],
                    cp.ascontiguousarray(aerosol_qv.T),converted["dz8w"],
                    cp.ascontiguousarray(nwfa.T),cp.ascontiguousarray(nifa.T),surfaces["xland"],ht)
            options.update(aer_opt=3 if aware else 0,aerosol=aer)
            if "o3clim" in options:
                options["o3clim"] = cp.ascontiguousarray(options["o3clim"].T)
            out = self.rrtmg_swrad(converted,surfaces,cp.ascontiguousarray(p8w.T),
                cp.ascontiguousarray(t8w.T),**options)
            for n in ("rthratensw",*PROFILE_OUTPUTS):
                out[n] = cp.ascontiguousarray(out[n].T)
            if aer is not None:
                out.update(taod5502d=aer["taod5502d"],
                    taod5503d=cp.ascontiguousarray(aer["taod5503d"].T))
            return out

    return RRFSShortwave


def cloudiness_slot(call):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F:291-370 @ 1d6e370e."""
    from .. import radshared_v842 as shared
    pools = call.pools
    nc = int(pools["nCellsSolve"])
    profiles = {n:shared._from_public(pools["prepared"][n],nc) for n in
        ("qv","qc","qi","qs","t","pres","pres_hyd","rho")}
    diag = pools["diag_physics"]
    convection_scheme = call.options["config_convection_scheme"].strip()
    extra = {}
    if call.scheme == "cld_fraction_mynn":
        extra.update({n:shared._from_public(diag[n],nc) for n in
            ("cldfrac_bl","qc_bl","qi_bl")})
    if convection_scheme == "cu_grell_freitas_li":
        extra.update({n:shared._from_public(diag[n],nc) for n in
            ("cldfrac_cu","qc_cu","qi_cu")})
    out = driver_cloudiness(scheme=call.scheme,itimestep=pools["itimestep"],
        convection_scheme=convection_scheme,**profiles,**extra,
        meshDensity=pools["mesh"]["meshDensity"][:nc],xland=pools["sfc_input"]["xland"][:nc],
        len_disp=pools["len_disp"] if "len_disp" in pools else call.options["config_len_disp"],
        f_qc=pools["f_qc"],f_qi=pools["f_qi"])
    shared._to_public(out["cldfrac"],diag["cldfrac"])
    for name, target in pools.get("radiation_carriers", {}).items():
        if target.ndim == 1:
            target[:nc] = out[name]
        else:
            shared._to_public(out[name], target)
    pools.setdefault("radshared",{}).update(out)
    return out


def shortwave_slot(call):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_sw.F:813-885,1020-1030 @ 1d6e370e."""
    if "_resident_radiation_slot" in call.pools:
        return call.pools["_resident_radiation_slot"].run(call)
    from ..rrtmg_sw_v842 import FIELD_NAMES, SCALAR_OUTPUTS
    pools = call.pools; nc = int(pools["nCellsSolve"])
    engine = pools["rrtmg_sw_engine"]; sounding = pools["radiation_sw_input"]
    diag,sfc,mesh = (pools[n] for n in ("diag_physics","sfc_input","mesh"))
    fields = {n:sounding[n][:,:nc] for n in FIELD_NAMES}
    surface = dict(albedo=diag["sfc_albedo"][:nc],tsk=sfc["skintemp"][:nc])
    surface.update({n:sfc[n][:nc] for n in ("snow","xice","xland")})
    options = dict(sounding["controls"])
    pin = pools["atm_input"]["pin"]
    ozone = diag["o3clim"][:,:nc] if options["o3input"] == 2 else engine.cp.empty((0,nc),engine.cp.float32)
    options.update(pin=pin,o3clim=ozone)
    aerosols = {n:sounding[n][:,:nc] for n in ("nwfa","nifa","aerosol_qv") if n in sounding}
    if "ht" in sounding:
        aerosols["ht"] = sounding["ht"][:nc]
    out = engine.radiation_sw_from_MPAS(fields,surface,sounding["p8w"][:,:nc],
        sounding["t8w"][:,:nc],lat_cell=mesh["latCell"][:nc],lon_cell=mesh["lonCell"][:nc],
        microp_scheme=call.options["config_microp_scheme"],config_microp_re=call.options["config_microp_re"],
        config_tempo_aerosolaware=call.options["config_tempo_aerosolaware"],**aerosols,**options)
    for n in SCALAR_OUTPUTS:
        diag[n][:nc] = out[n]
    pools["tend_physics"]["rthratensw"][:,:nc] = out["rthratensw"]
    for n in ("taod5502d","taod5503d"):
        if n in out:
            if diag[n].ndim == 1:
                diag[n][:nc] = out[n]
            else:
                diag[n][:,:nc] = out[n]
    return out


def load_shortwave_resources(packet, device, *, eot_day_offset=1.0, eot_second_sine=0.04089):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_rrtmg_swinit.F:70-1194 @ 1d6e370e."""
    import hashlib
    import tempfile
    paths = {}
    for name in ("data", "decoder"):
        reference = packet[name]
        path = Path(reference["path"]).resolve()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != reference["sha256"]:
            raise ValueError("shortwave resource pin changed: " + name)
        paths[name] = path
    cache = Path(packet["cache_root"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="rrtmg-sw-", dir=cache)) / "decoded"
    return shortwave_class(eot_day_offset=eot_day_offset, eot_second_sine=eot_second_sine).from_mpas_data(paths["data"], decoder=paths["decoder"],
                                          decoded_directory=directory)


def bind_native_pools(provider, call):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_sw.F:374-808,1020-1030 @ 1d6e370e."""
    from .resident_radiation_rrfs import resident_slot
    slot = resident_slot(provider, call)
    if slot is not None:
        return {"_resident_radiation_slot": slot}
    if call.slot == "cloudiness":
        dims = provider.session.layout.dimensions
        nz, nc = int(dims["nVertLevels"]), int(dims["nCellsSolve"])
        shared = getattr(provider, "radshared_rrfs_state", None)
        if shared is None:
            shared = {}; provider.radshared_rrfs_state = shared
        carriers = {name:provider.prepared(name+"_p")[:nz] for name in
            ("qvrad","qcrad","qirad","qsrad","cldfrac")}
        carriers.update({name:provider.prepared(name+"_p") for name in ("dx","xland")})
        if call.scheme == "cld_fraction_mynn":
            carriers.update({name:provider.prepared(name+"_p")[:nz] for name in ("qcbl","qibl")})
        if call.options["config_convection_scheme"].strip() == "cu_grell_freitas_li":
            carriers.update({name:provider.prepared(name+"_p")[:nz] for name in ("qccu","qicu")})
        return {"prepared": {name:provider.prepared(name+"_p")[:nz] for name in
            ("qv","qc","qi","qs","t","pres","pres_hyd","rho")},
            "mesh":provider["mesh"], "sfc_input":provider["sfc_input"],
            "diag_physics":provider["diag_physics"], "nCellsSolve":nc,
            "itimestep":call.itimestep,
            "f_qc":provider["scalar_state"]["f_qc"],
            "f_qi":provider["scalar_state"]["f_qi"], "radshared":shared,
            "radiation_carriers":carriers,
            # config_len_disp is an nhyd_model option, not a physics one: the fork
            # driver reads it from the configs pool (mpas_atmphys_driver_cloudiness.F:169),
            # where core init has already put nominalMinDc in place of a zero.
            "len_disp":provider["configs"]["config_len_disp"]}
    if call.slot == "radiation_lw":
        from ..radshared_v842 import _from_public
        dims = provider.session.layout.dimensions
        nz, nc = int(dims["nVertLevels"]), int(dims["nCellsSolve"])
        prepared = {name:provider.prepared(name+"_p")[:nz+1 if name in
            ("pres2_hyd","t2") else nz] for name in
            ("pres_hyd","pres2_hyd","t","t2","dz","pi")}
        prepared["xlat"] = angles_to_degrees(
            provider["mesh"]["latCell"][:nc], provider["constants"]["degrad"])
        return {"prepared":prepared, "diag_physics":provider["diag_physics"],
            "sfc_input":provider["sfc_input"], "tend_physics":provider["tend_physics"],
            "atm_input":_ozone_pool(provider, call), "scalar_state":provider["scalar_state"],
            "nCellsSolve":nc, "icloud":1,
            "julday":provider.native.get_table("mpas_atmphys_manager.julday")[0][()],
            "rrtmg_lw_tables":provider.rrtmg_lw_rrfs_resources,
            "radshared":{name:_from_public(provider.prepared(name+"_p")[:nz],nc)
                for name in ("qvrad","qcrad","qirad","qsrad")}}
    engine = provider.rrtmg_sw_rrfs_resources
    prepared, diag = provider["prepared"], provider["diag_physics"]
    nz = int(provider.session.layout.dimensions["nVertLevels"])
    aliases = dict(p3d="pres_hyd_p", t3d="t_p", dz8w="dz_p", pi3d="pi_p",
                   qv3d="qvrad_p", qc3d="qcrad_p", qi3d="qirad_p", qs3d="qsrad_p")
    sounding = {name: prepared[native][:nz] for name, native in aliases.items()}
    sounding["cldfra3d"] = diag["cldfrac"]
    diagnosed = call.options["config_microp_re"] and call.options["config_microp_scheme"] in RADIUS_SCHEMES
    for name in ("re_cloud", "re_ice", "re_snow"):
        sounding[name] = diag[name] if diagnosed else provider.cp.zeros_like(sounding["p3d"])
    sounding.update(p8w=prepared["pres2_hyd_p"][:nz+1], t8w=prepared["t2_p"][:nz+1])
    scalars, constants = provider["scalar_state"], provider["constants"]
    calendar = {name: provider.native.get_table("mpas_atmphys_manager." + name)[0][()]
                for name in ("curr_julday", "gmt", "julday")}
    controls = engine.driver_controls(call.options, xtime_s=call.time_seconds,
        dt_radtsw=scalars["dt_radtsw"], curr_julday=calendar["curr_julday"],
        degrad=constants["degrad"], dpd=constants["dpd"])
    controls.update(gmt=calendar["gmt"], julday=calendar["julday"], icloud=1)
    controls.update({name: scalars[name] for name in ("has_reqc", "has_reqi", "has_reqs")})
    sounding["controls"] = controls
    microp_scheme = call.options["config_microp_scheme"].strip()
    aware = microp_scheme == "mp_thompson_aerosols" or (
        microp_scheme == "mp_tempo" and call.options["config_tempo_aerosolaware"])
    if aware:
        sounding.update(nwfa=prepared["nwfa_p"][:nz], nifa=prepared["nifa_p"][:nz],
                        aerosol_qv=prepared["qv_p"][:nz], ht=provider["mesh"]["zgrid"][0])
    provider.values.update(nCellsSolve=provider.session.layout.dimensions["nCellsSolve"],
                           rrtmg_sw_engine=engine, radiation_sw_input=sounding,
                           atm_input=_ozone_pool(provider, call))
    return provider


def _ozone_pool(provider, call):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_lw.F:514-517,616-619 and
    mpas_atmphys_driver_radiation_sw.F:697-700 @ 1d6e370e.

    radiation_{lw,sw}_from_MPAS set pin_p(k) = pin(k) from the atm_input pool before every call, so the scheme
    sees the pool's ozone pressure levels. The slot reads that pool field itself: the native module array pin_p
    is only ever filled by those native routines, which no resident (or library-route CUDA) slot runs, so reading
    pin_p handed RRTMG all-zero levels and every column took its lowest climatology level (surface ozone) at every
    height (o3vmr, LW and SW heating, glw/gsw off native from step 1). The module replays passed because their
    captured pin_p came from native after its own copy.
    """
    if not call.options["config_o3climatology"]:
        return {"pin":provider.cp.empty(0,provider.cp.float32)}
    nz = int(provider.session.layout.dimensions["nOznLevels"])
    return {"pin":provider["atm_input"]["pin"][:nz]}


def load_longwave_resources(packet, device):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_rrtmg_lwinit.F:59-1593 @ 1d6e370e."""
    from ..rrtmg_lw_v842 import load_tables_device
    from ..table_resources_v842 import certified_path
    # woof hex physics-resources build writes the same pinned RRTMG_LW_DATA
    # (sha256 bcfdee24..., byte-identical in the fork) under "table".
    data = packet["data"] if "data" in packet else packet["table"]
    return load_tables_device(certified_path(data),
                              decoder=certified_path(packet["decoder"]))


def longwave_slot(call):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_lw.F:300-614,616-789,908-928 @ 1d6e370e."""
    if "_resident_radiation_slot" in call.pools:
        return call.pools["_resident_radiation_slot"].run(call)
    from .. import rrtmg_lw_v842 as lw
    from ..radshared_v842 import OVERLAP
    cp = _cp(); pools = call.pools; nc = int(pools["nCellsSolve"])
    prep,diag,rad,sfc = (pools[n] for n in ("prepared","diag_physics","radshared","sfc_input"))
    config = call.options
    overlap = OVERLAP[config["config_radt_cld_overlap"].strip()]
    idcor = int(overlap>=4 and config["config_radt_cld_dcorrlen"].strip()=="latitude_varying")
    o3input = 2 if config["config_o3climatology"] else 0
    state = {n:prep[p][:,:nc] for n,p in (("p3d","pres_hyd"),("p8w","pres2_hyd"),
        ("t3d","t"),("t8w","t2"),("dz8w","dz"),("pi3d","pi"))}
    state.update({n:rad[r][:nc].T for n,r in (("qv3d","qvrad"),("qc3d","qcrad"),
        ("qi3d","qirad"),("qs3d","qsrad"))})
    state["cldfra3d"] = diag["cldfrac"][:,:nc]
    radius_scheme = config["config_microp_scheme"].strip() in RADIUS_SCHEMES
    enabled = bool(config["config_microp_re"]) and radius_scheme
    state.update({n:diag[n][:,:nc] if enabled else cp.zeros_like(state["p3d"])
        for n in ("re_cloud","re_ice","re_snow")})
    state.update({n:sfc[s][:nc] for n,s in (("tsk","skintemp"),("xland","xland"),
        ("xice","xice"),("snow","snow"))})
    state.update(emiss=diag["sfc_emiss"][:nc],xlat=prep["xlat"][:nc])
    pin = pools["atm_input"]["pin"] if o3input==2 else None
    ozone = cp.ascontiguousarray(diag["o3clim"][:,:nc].T) if o3input==2 else None
    if o3input==2:
        p = cp.ascontiguousarray(state["p3d"].T); mb = cp.empty_like(p)
        lw._gpu_kernel("mpas_pressure_mb")(((p.size+127)//128,),(128,),(np.int32(p.size),p,mb))
        diag["o3vmr"][:,:nc] = lw.vinterp_ozn_device(mb,pin,ozone).T
    flags = pools["scalar_state"]
    # retain_profiles=False, as the v8.4.2 slot binds it (rrtmg_lw_v842.bind_native_pools, retain_lw_profiles):
    # the driver writes the pools below and discards the slot result, and this slot writes no lw{up,dn}flx[c]_p
    # profile. Fluxes, tendencies and radii do not depend on it (rrtmg_lwrad_device contract).
    # THE BREAKAGE THIS PREVENTS: the default (True) built a dict of eight device slices for every column on
    # every LW call: about 1.0 s of host Python per call at 40,962 columns (x1.40962: LW 1,084 ms of a
    # 1,205 ms radiation call, its solver about 90 ms), growing with the column count.
    result = lw.rrtmg_lwrad_device(*lw.radiation_lw_from_mpas(state),pools["rrtmg_lw_tables"],
        icloud=int(pools["icloud"]),julday=int(pools["julday"]),cldovrlp=overlap,idcor=idcor,
        has_reqc=int(flags["has_reqc"]),has_reqi=int(flags["has_reqi"]),has_reqs=int(flags["has_reqs"]),
        o3input=o3input,pin=pin,ozone=ozone,retain_profiles=False)
    out = lw.radiation_lw_to_mpas(result)
    for n in lw.FLUX_FIELDS:
        diag["olrtoa" if n=="olr" else n][:nc] = out[n]
    pools["tend_physics"]["rthratenlw"][:,:nc] = out["rthratenlw"]
    if radius_scheme:
        for n in ("rre_cloud","rre_ice","rre_snow"):
            diag[n][:,:nc] = out[n] if enabled else np.float32(0)
    return result


def register_hex_cuda(registry):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:158-168,299-309 @ 1d6e370e.

    The caller supplies the registry selected by its resolved pin table row.
    Register once on that registry; never overwrite the v8.4.2 registry.
    """
    for scheme in CLOUD_SCHEMES:
        registry.register("cloudiness",scheme,"hex-cuda",cloudiness_slot)
    registry.register("radiation_sw","rrtmg_sw","hex-cuda",shortwave_slot)
    registry.register("radiation_lw","rrtmg_lw","hex-cuda",longwave_slot)
