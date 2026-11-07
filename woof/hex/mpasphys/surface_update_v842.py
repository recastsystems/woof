"""MPAS-A v8.4.2 surface services, operating on resident RKIND arrays.

Pool names match MPAS; 2-D arrays use (level/month/lag, cell), Fortran order.
The caller owns stream decoding (Rust), halo exchange, restart and the clock.
No forecast/default suite selection is performed here.
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from importlib import resources
import re

import numpy as np

RADIATION_FIELDS = tuple(
    band + direction + location + clear
    for band in ("sw", "lw") for direction in ("dn", "up")
    for location in ("b", "t") for clear in ("", "c")
)
CUDA_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                "--prec-div=true", "--prec-sqrt=true")

# MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_update_surface.F:54-568
_CUDA = r'''
#include "glibc_flt32.cuh"
// MPAS source: mpas_atmphys_update_surface.F:89-108,373-417,512-561.
__device__ float add(float a,float b){return __fadd_rn(a,b);}
// MPAS source: mpas_atmphys_update_surface.F:89-108,373-417,512-561.
__device__ float sub(float a,float b){return __fadd_rn(a,-b);}
// MPAS source: mpas_atmphys_update_surface.F:89-108,373-417,512-561.
__device__ float mul(float a,float b){return __fmul_rn(a,b);}
// MPAS source: mpas_atmphys_update_surface.F:89-108,373-417,512-561.
__device__ float divr(float a,float b){return __fdiv_rn(a,b);}
// MPAS source: mpas_atmphys_update_surface.F:89-108.
extern "C" __global__ void monthly(int n,int m1,int m2,int w1,int w2,int span,
 const float* green,const float* alb,const int* land,int use_alb,
 float* veg,float* lo,float* hi,float* bg){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 veg[i]=divr(add(mul(green[m2+12*i],float(w2)),mul(green[m1+12*i],float(w1))),float(span));
 float mn=green[12*i],mx=mn;
 for(int k=1;k<12;k++){float v=green[k+12*i];if(v<mn)mn=v;if(v>mx)mx=v;}
 lo[i]=mn;hi[i]=mx;
 if(use_alb){float v=divr(add(mul(alb[m2+12*i],float(w2)),mul(alb[m1+12*i],float(w1))),float(span));
 bg[i]=land[i]==0?0.08f:divr(v,100.f);}
}
// MPAS source: mpas_atmphys_update_surface.F:196-313.
extern "C" __global__ void sst_update(int n,int ns,int frac,float threshold,int isice,int iswater,
 int* ivg,int* isl,float* veg,float* bg,const float* sst,float* tmn,float* tsk,
 float* tslb,float* sh2o,float* smois,float* snow,float* snowc,float* snowh,
 float* seaice,float* xice,float* xland,float* alb,float* emiss,float* emibck,float* old){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float x=xice[i],o=old[i];
 if(frac){if(x<threshold)x=0.f;}else{x=x>=threshold?1.f:0.f;}
 if(frac && x!=o && o>=threshold){
 alb[i]=add(0.08f,divr(mul(sub(alb[i],0.08f),x),o));
 emiss[i]=add(0.98f,divr(mul(sub(emiss[i],0.98f),x),o));}
 if(xland[i]>=1.5f && x>=threshold && o<threshold){
 ivg[i]=isice;isl[i]=16;veg[i]=0.f;xland[i]=1.f;tmn[i]=271.4f;
 for(int k=0;k<ns;k++){tslb[k+ns*i]=tsk[i];smois[k+ns*i]=1.f;sh2o[k+ns*i]=0.f;}
 alb[i]=add(mul(0.80f,x),mul(0.08f,sub(1.f,x)));
 emiss[i]=add(mul(0.98f,x),mul(0.98f,sub(1.f,x)));bg[i]=0.80f;emibck[i]=0.98f;
 }else if(xland[i]<1.5f && x<threshold && o>=threshold){
 ivg[i]=iswater;isl[i]=14;veg[i]=0.f;xland[i]=2.f;tmn[i]=sst[i];
 snowc[i]=0.f;snow[i]=0.f;snowh[i]=0.f;alb[i]=0.08f;bg[i]=0.08f;emiss[i]=0.98f;emibck[i]=0.98f;
 for(int k=0;k<ns;k++){tslb[k+ns*i]=sst[i];smois[k+ns*i]=1.f;sh2o[k+ns*i]=1.f;}
 }else if(x<threshold && o<threshold){x=0.f;}
 if(xland[i]>=1.5f){tsk[i]=sst[i];for(int k=0;k<ns;k++)tslb[k+ns*i]=273.16f;}
 xice[i]=x;old[i]=x;seaice[i]=x>0.f?1.f:0.f;
}
// MPAS source: mpas_atmphys_update_surface.F:369-444.
extern "C" __global__ void skin(int n,float dt,const float* sst,float* tsk,const float* xland,
 float* sstsk,float* dtc1,float* dtw1,const float* emiss,const float* glw,const float* gsw,
 const float* hfx,const float* qfx,const float* ust){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || !(xland[i]>=1.5f))return;
 tsk[i]=sst[i];
 float ss2=mul(sstsk[i],sstsk[i]);
 float qo=sub(sub(sub(glw[i],mul(mul(emiss[i],5.67051e-8f),mul(ss2,ss2))),mul(2.5e6f,qfx[i])),hfx[i]);
 float us=fmaxf(ust[i],0.01f),tb=sub(tsk[i],273.15f),dtwo=dtw1[i];
 float q=divr(qo,4294750.f),sw=divr(gsw[i],4294750.f);
 // gfortran folds both constant exponentials and their surrounding expression.
 float f1=0.63517636060714722f;
 float alw=mul(1.e-5f,fmaxf(tb,1.f));
 float con4=divr(mul(mul(156.8f,alw),9.9999994241612840e-19f),1.9599998504087925e-14f);
 float usw=mul(0.034215956926345825f,us),u2=mul(usw,usw);
 float con5=divr(con4,mul(u2,u2));
 float q2=fmaxf(divr(1.f,4294750.f),-q);
 float zlan=divr(6.f,gfk_pow(add(1.f,gfk_pow(mul(con5,q2),0.75f)),0.333f));
 float dep=divr(mul(zlan,1.e-6f),usw);
 float fs=sub(add(0.065f,mul(11.f,dep)),mul(divr(6.6e-5f,dep),sub(1.f,gfk_exp(divr(-dep,8.e-4f)))));
 fs=fmaxf(fs,0.01f);
 float dtc=fminf(divr(mul(dep,add(q,mul(sw,fs))),1.4e-7f),0.f);
 float con1=__fsqrt_rn(divr(mul(147.f,alw),0.3f)),con2=mul(3.92f,alw);
 float qn=add(q,mul(sw,f1));
 if(dtwo>0.f && qn<0.f){float qn1=divr(mul(__fsqrt_rn(dtwo),u2),con1);qn=fmaxf(qn,qn1);}
 float zeta=divr(mul(mul(3.f,con2),qn),mul(u2,usw));
 float phi=zeta>0.f?add(1.f,mul(5.f,zeta)):divr(1.f,__fsqrt_rn(sub(1.f,mul(16.f,zeta))));
 float con3=divr(mul(0.4f,usw),mul(3.f,phi));
 float dtw=divr(add(dtwo,divr(mul(mul(4.333333015f,add(q,mul(sw,f1))),dt),3.f)),add(1.f,mul(mul(1.3f,con3),dt)));
 dtw=fmaxf(0.f,dtw);
 float ts=add(add(tb,dtw),dtc);sstsk[i]=add(ts,273.15f);dtc1[i]=dtc;dtw1[i]=dtw;tsk[i]=sstsk[i];
}
// MPAS source: mpas_atmphys_update_surface.F:512-561.
extern "C" __global__ void deep(int n,int nl,int leap,float dt,float jul,
 float* steps,float* days,float* day,float* year,float* mean,float* lag,float* tmn,const float* tsk){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 day[i]=add(day[i],mul(tsk[i],dt));steps[i]=add(steps[i],dt);
 float nearest=jul>=0.f?floorf(add(jul,0.5f)):ceilf(sub(jul,0.5f));
 float delta=mul(mul(sub(jul,nearest),24.f),3600.f);
 if(fabsf(delta)<=divr(dt,2.f)){
 float j=add(sub(jul,1.f),divr(dt,86400.f)),prior=0.f;
 for(int k=0;k<nl;k++)prior=add(prior,lag[k+nl*i]);
 prior=divr(prior,float(nl));tmn[i]=add(mul(0.6f,mean[i]),mul(0.3999999761581421f,prior));
 for(int k=0;k<nl-1;k++)lag[k+nl*i]=lag[k+1+nl*i];
 lag[nl-1+nl*i]=divr(day[i],steps[i]);day[i]=0.f;steps[i]=0.f;
 if(sub(leap?366.f:365.f,j)<=1.f){mean[i]=divr(year[i],days[i]);year[i]=0.f;days[i]=0.f;}
 else{year[i]=add(year[i],lag[nl-1+nl*i]);days[i]=add(days[i],1.f);}
 }
}
// MPAS source: mpas_atmphys_update.F:141-247; mpas_atmphys_driver_convection.F:1081-1097; mpas_atmphys_driver_microphysics.F:619-630.
extern "C" __global__ void accumulate(int n,int begin,int end,float dt,float bucket,int ringing,
 const float* rate,float* total,int* count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || i<begin || i>=end)return;
 total[i]=add(total[i],mul(rate[i],dt));
 if(ringing && bucket>0.f && total[i]>bucket){count[i]+=1;total[i]=sub(total[i],bucket);}
}
'''

def _module():
    """The surface-update module loaded on the calling card."""
    import cupy as cp
    # A CUDA module belongs to the card it was loaded on. Same-host ranks drive
    # several cards from one process, so a cache keyed without the card hands
    # the second card the first card's handle (CUDA_ERROR_INVALID_HANDLE at step 1).
    return _module_on(int(cp.cuda.Device().id))


@lru_cache(maxsize=None)
def _module_on(device: int):
    """mpas_atmphys_update_surface.F:373-417; shared pinned GNU float maths."""
    import cupy as cp
    from cupy.cuda import compiler
    directory = resources.files("woof").joinpath("core/kernels")
    # CuPy's RawModule cache compiler appends forced FTZ, losing native
    # subnormal accumulation words. Direct NVRTC preserves --ftz=false.
    binary, _ = compiler.compile_using_nvrtc(
        _CUDA, options=(*_hex_numerics.options(CUDA_OPTIONS, keep_ftz=True), f"-I{directory}"),
        arch=_compilation_target(device, xp=cp))
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(binary)
    return module


def _array(pool, name, shape, dtype="float32", order="C"):
    """mpas_atmphys_update_surface.F:78-87,139-176; validate native pool storage."""
    import cupy as cp
    a = pool[name]
    if not isinstance(a, cp.ndarray) or a.dtype != dtype or a.shape != shape:
        raise ValueError(f"{name} must be a resident {dtype} array with shape {shape}")
    if not (a.flags.f_contiguous if order == "F" else a.flags.c_contiguous):
        raise ValueError(f"{name} must use {order} order; strided pool arrays would corrupt cell indexing")
    return a


def _launch(name, n, args):
    """mpas_atmphys_update_surface.F:89-108,196-313,369-444,512-561."""
    if n:
        _module().get_function(name)(((n + 127) // 128,), (128,), (np.int32(n), *args))


class MonthlyCalendar:
    """Preserve MPAS's saved February length across calls.

    MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_date_time.F:51-84,148-255
    DATA mmd in get_julgmt retains leap February even in a following nonleap
    year. This state is kept per service, not corrected silently.
    """
    def __init__(self, february=28):
        """src/core_atmosphere/physics/mpas_atmphys_date_time.F:51-84."""
        self.february = february

    def weights(self, current_date):
        """src/core_atmosphere/physics/mpas_atmphys_date_time.F:148-255."""
        d = datetime.fromisoformat(current_date.replace("_", "T"))
        y = d.year
        if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0):
            self.february = 29
        lengths = (31, self.february, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        mid = [y * 1000 + sum(lengths[:m]) + 15 for m in range(12)]
        middle = [mid[0] - 31, *mid, mid[-1] + 31]
        target = y * 1000 + sum(lengths[:d.month-1]) + d.day
        for l in range(13):
            if middle[l] < target <= middle[l+1]:
                m1, m2 = (11, 0) if l in (0, 12) else (l-1, l)
                return (m1, m2, middle[l+1]-target, target-middle[l], middle[l+1]-middle[l])
        raise ValueError("date outside MPAS monthly brackets")


def physics_update_surface(current_date, config_sfc_albedo, mesh, sfc_input, *, calendar):
    """mpas_atmphys_update_surface.F:54-111; src/core_atmosphere/physics/mpas_atmphys_date_time.F:148-255."""
    n = mesh["nCellsSolve"]
    weights = tuple(np.int32(v) for v in calendar.weights(current_date))
    green = _array(sfc_input, "greenfrac", (12, mesh.get("nCells", n)), order="F")
    # Halo capacity may exceed nCellsSolve; kernels only touch owned columns.
    capacity = green.shape[1]
    alb = _array(sfc_input, "albedo12m", (12, capacity), order="F")
    land = _array(sfc_input, "landmask", (capacity,), "int32")
    out = tuple(_array(sfc_input, k, (capacity,)) for k in ("vegfra", "shdmin", "shdmax", "sfc_albbck"))
    _launch("monthly", n, (*weights, green, alb, land, np.int32(config_sfc_albedo), *out))


def physics_update_sst(config_frac_seaice, mesh, sfc_input, diag_physics, *, xice_threshold):
    """mpas_atmphys_update_surface.F:114-326 (dminfo has no active operations)."""
    n, ns = mesh["nCellsSolve"], mesh["nSoilLevels"]
    capacity = mesh.get("nCells", n)
    args = [np.int32(ns), np.int32(config_frac_seaice), np.float32(xice_threshold),
            np.int32(sfc_input["isice"]), np.int32(sfc_input["iswater"])]
    for key in ("ivgtyp", "isltyp", "vegfra", "sfc_albbck", "sst", "tmn", "skintemp",
                "tslb", "sh2o", "smois", "snow", "snowc", "snowh", "seaice", "xice", "xland"):
        rank2 = key in ("tslb", "sh2o", "smois")
        args.append(_array(sfc_input, key, (ns, capacity) if rank2 else (capacity,),
                           "int32" if key in ("ivgtyp", "isltyp") else "float32", "F" if rank2 else "C"))
    args.extend(_array(diag_physics, k, (capacity,)) for k in ("sfc_albedo", "sfc_emiss", "sfc_emibck", "xicem"))
    _launch("sst_update", n, tuple(args))


def physics_update_sstskin(dt, mesh, diag_physics, sfc_input):
    """mpas_atmphys_update_surface.F:329-464."""
    n, capacity = mesh["nCellsSolve"], mesh.get("nCells", mesh["nCellsSolve"])
    args = [np.float32(dt)]
    args.extend(_array(sfc_input, k, (capacity,)) for k in ("sst", "skintemp", "xland"))
    args.extend(_array(diag_physics, k, (capacity,)) for k in
                ("sstsk", "sstsk_dtc", "sstsk_dtw", "sfc_emiss", "glw", "gsw", "hfx", "qfx", "ust"))
    _launch("skin", n, tuple(args))


def physics_update_deepsoiltemp(LeapYear, dt, julian_in, mesh, sfc_input, diag_physics):
    """mpas_atmphys_update_surface.F:467-568."""
    n, nl = mesh["nCellsSolve"], mesh["nLags"]
    if nl < 1:
        raise ValueError("nLags must be positive: MPAS indexes tlag(nLags,iCell)")
    capacity = mesh.get("nCells", n)
    args = [np.int32(nl), np.int32(LeapYear), np.float32(dt), np.float32(julian_in)]
    args.extend(_array(diag_physics, k, (capacity,)) for k in
                ("nsteps_accum", "ndays_accum", "tday_accum", "tyear_accum", "tyear_mean"))
    args.append(_array(diag_physics, "tlag", (nl, capacity), order="F"))
    args.extend(_array(sfc_input, k, (capacity,)) for k in ("tmn", "skintemp"))
    _launch("deep", n, tuple(args))


def _accumulate(mesh, rate, total, count, dt, bucket, ringing, its, ite):
    """mpas_atmphys_update.F:141-247; mpas_atmphys_driver_convection.F:1081-1097."""
    n = mesh.get("nCells", mesh["nCellsSolve"])
    if not 1 <= its <= ite <= mesh["nCellsSolve"]:
        raise ValueError("its/ite must be MPAS's inclusive owned-cell bounds")
    p = {"rate": rate, "total": total, "count": count}
    _launch("accumulate", n, (np.int32(its-1), np.int32(ite), np.float32(dt), np.float32(bucket),
        np.int32(ringing), _array(p, "rate", (n,)), _array(p, "total", (n,)), _array(p, "count", (n,), "int32")))


def update_radiation_diagnostics(configs, mesh, diag_physics, its, ite, *, dt_dyn, l_acradt):
    """mpas_atmphys_update.F:72-254; accumulate every step, bucket once on alarm."""
    for k in RADIATION_FIELDS:
        _accumulate(mesh, diag_physics[k], diag_physics["ac"+k], diag_physics["i_ac"+k],
                    dt_dyn, configs["config_bucket_radt"], l_acradt, its, ite)


def update_precipitation_buckets(configs, mesh, diag_physics, its, ite, *, dt_dyn, l_acrain,
                                 convection=True, microphysics=True):
    """mpas_atmphys_driver_convection.F:1060-1100; mpas_atmphys_driver_microphysics.F:619-630.

    Microphysics supplies rainncv in mm/step; convection supplies cuprec in
    mm/s. The microphysics driver remains responsible for its other diagnostics.
    Call exactly once at the corresponding driver boundary, not again after
    an f4/native precipitation routine that already accumulates these fields.
    """
    for enabled, suffix, rate, dt in ((convection, "c", "cuprec", dt_dyn),
                                      (microphysics, "nc", "rainncv", 1.0)):
        if enabled:
            _accumulate(mesh, diag_physics[rate], diag_physics["rain"+suffix], diag_physics["i_rain"+suffix],
                        dt, configs["config_bucket_rain"+suffix], l_acrain, its, ite)


def _interval(value):
    """mpas_timekeeping.F:1347-1591; fixed surface-service interval."""
    match = re.fullmatch(r"(?:(\d+)_)?(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", value)
    if not match:
        raise ValueError(f"surface interval must be days_HH:MM:SS or HH:MM:SS: {value}")
    d, h, m, s = match.groups()
    result = timedelta(days=int(d or 0), hours=int(h), minutes=int(m), seconds=float(s))
    if result <= timedelta(0):
        raise ValueError("surface interval must advance the alarm")
    return result


@dataclass
class SurfaceAlarms:
    """mpas_atmphys_manager.F:231-253,343-363,549-565,593-612.

    Start green/sfcbdy at startTime; buckets at startTime+interval. The
    stream callback must install the Rust-decoded current surface record
    before SST conversion. Save next ringing times with clock restart state.
    """
    next_green: datetime
    green_interval: timedelta
    next_surface: datetime | None
    surface_interval: timedelta | None
    next_rain: datetime | None
    next_radiation: datetime | None
    bucket_interval: timedelta | None

    @classmethod
    def create(cls, start, configs, *, surface_record_interval=None):
        """mpas_atmphys_manager.F:549-565,593-612."""
        start = datetime.fromisoformat(start.replace("_", "T")) if isinstance(start, str) else start
        green = _interval(configs["config_greeness_update"])
        surface = None
        if configs["config_sst_update"]:
            if surface_record_interval is None:
                raise ValueError("SST update needs the surface stream record interval")
            surface = _interval(surface_record_interval)
        bucket = None if configs["config_bucket_update"] == "none" else _interval(configs["config_bucket_update"])
        return cls(start, green, start if surface else None, surface,
                   start+bucket if bucket else None, start+bucket if bucket else None, bucket)

    def _consume(self, name, now, period):
        """mpas_timekeeping.F:925-985; mpas_atmphys_manager.F:231-247."""
        next_time = getattr(self, "next_"+name)
        due = next_time is not None and now >= next_time
        if due:
            # MPAS interval division advances directly past missed rings.
            next_time += ((now-next_time)//period + 1)*period
            setattr(self, "next_"+name, next_time)
        return due

    def surface_rings(self, now):
        """mpas_atmphys_manager.F:231-247; read/reset within each block."""
        return {name: self._consume(name, now, getattr(self, name+"_interval"))
                for name in ("green", "surface")}

    def bucket_rings(self, now, dt_dyn, *, convection_enabled=True, radiation_enabled=True):
        """mpas_atmphys_manager.F:343-363; buckets look ahead by dt.

        Call once after the surface block loop. Precipitation is gated on
        convection scheme != off in the pristine manager, including rainnc.
        Radiation is gated on either radiation scheme != off.
        """
        through = now + timedelta(seconds=dt_dyn)
        return dict(acrain=convection_enabled and self._consume("rain", through, self.bucket_interval),
                    acradt=radiation_enabled and self._consume("radiation", through, self.bucket_interval))

    def rings(self, now, dt_dyn=0.):
        """mpas_atmphys_manager.F:231-253,343-363; one-block alarm slices."""
        return {**self.surface_rings(now), **self.bucket_rings(now, dt_dyn)}


def physics_timetracker_surface(current_date, dt_dyn, curr_julday, LeapYear, configs,
                                mesh, sfc_input, diag_physics, *, flags, calendar,
                                xice_threshold, read_surface_record=None):
    """mpas_atmphys_manager.F:231-253; call per block with its surface alarm flags.

    Read/reset surface alarms inside the block loop, as MPAS does. Subsequent
    blocks see reset alarms; skin/deepsoil still run on every block.
    Skin/deepsoil run each dynamics step, independently of the stream cadence.
    Bucket flags are consumed at precipitation/radiation driver boundaries.
    """
    if flags["green"]:
        physics_update_surface(current_date, configs["config_sfc_albedo"], mesh, sfc_input, calendar=calendar)
    if flags["surface"] and configs["config_sst_update"]:
        if read_surface_record is None:
            raise ValueError("surface stream callback missing: SST would use a stale record")
        read_surface_record(current_date, sfc_input)
        physics_update_sst(configs["config_frac_seaice"], mesh, sfc_input, diag_physics, xice_threshold=xice_threshold)
    if configs["config_sstdiurn_update"]:
        physics_update_sstskin(dt_dyn, mesh, diag_physics, sfc_input)
    if configs["config_deepsoiltemp_update"]:
        physics_update_deepsoiltemp(LeapYear, dt_dyn, curr_julday, mesh, sfc_input, diag_physics)


class NativeSurfaceService:
    """CUDA surface callbacks at the native driver boundary.

    mpas_atmphys_manager.F:231-253; mpas_atmphys_driver.F:209-365;
    mpas_atmphys_update_surface.F:54-568; mpas_atmphys_update.F:72-254;
    mpas_atmphys_driver_convection.F:1060-1100.

    Native storage retains its full Fortran stride, including halos and the
    framework's sentinel cell. Only owned cells change. The session owns clocks
    and pool snapshots; this object owns the saved monthly February length.
    Microphysics retains its native precipitation accumulation, so it is never
    accumulated again here. A supplied Rust stream callback installs SST/ice
    words before the SST update; Python performs only storage transfers.
    """

    OPERATIONS = ('physics_update_surface', 'physics_update_sst',
                  'physics_update_sstskin', 'physics_update_deepsoiltemp',
                  'update_radiation_diagnostics', 'update_convection_step2')

    def __init__(self, session, device, *, read_surface_record=None):
        """mpas_atmphys_manager.F:231-253; src/core_atmosphere/physics/mpas_atmphys_date_time.F:51-84."""
        import hashlib
        from pathlib import Path
        self.session, self.native, self.cp = session, session.native, device
        self.read_surface_record = read_surface_record
        self.calendar = MonthlyCalendar()
        self.executions = {name: 0 for name in self.OPERATIONS}
        self.module_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def callbacks(self):
        """mpas_atmphys_driver.F:209-365; mpas_atmphys_manager.F:231-253."""
        return {name: self.invoke for name in self.OPERATIONS}

    def _pool(self, name, fields):
        """mpas_atmphys_update_surface.F:78-87,139-176,349-360,497-507."""
        result = {}
        for field in fields:
            provider = getattr(self.session, 'device_pools', None)
            if provider is not None and provider.resident:
                if not self.native._fields[name, field][1]:
                    # Registry scalars here are static class IDs, not fields.
                    result[field] = self.native.get_field(name, field)[()]
                    continue
                value = provider.field(name, field)
                # These service kernels use native vertical/soil-first
                # Fortran strides. The common pool keeps level-first C order.
                result[field] = self.cp.asfortranarray(value) if value.ndim == 2 else value
                continue
            host = self.native.get_field(name, field)
            result[field] = (host[()] if host.ndim == 0 else
                             self.cp.array(host, order='F' if host.ndim == 2 else 'C'))
        return result

    def _commit(self, name, pool, changed):
        """mpas_atmphys_interface.F:284-425; mpas_atmphys_update_surface.F:89-108."""
        for field in changed:
            # Preserve native shape, scalar kind, time levels and halo words.
            self.session.put(name, field, pool[field])

    def _scalar(self, name):
        """mpas_atmphys_vars.F:161-166; mpas_atmphys_manager.F:204-213."""
        host, _ = self.native.get_variable(name)
        if host.ndim:
            raise ValueError(f'native surface global {name} must be scalar')
        return host[()]

    def invoke(self, call):
        """mpas_atmphys_update_surface.F:54-568; mpas_atmphys_update.F:72-254;
        mpas_atmphys_driver_convection.F:1060-1100.
        """
        operation = call.slot
        if operation not in self.OPERATIONS:
            raise ValueError(f'unknown surface operation {operation}')
        mesh = dict(self.session.layout.dimensions)
        options = call.options
        metadata = self.session.arguments.calendar
        sfc, diag = {}, {}
        if operation == 'physics_update_surface':
            fields = ('landmask', 'albedo12m', 'greenfrac', 'vegfra',
                      'shdmin', 'shdmax', 'sfc_albbck')
            sfc = self._pool('sfc_input', fields)
            mesh['nCells'] = sfc['landmask'].size
            physics_update_surface(metadata['current_date'], options['config_sfc_albedo'],
                                   mesh, sfc, calendar=self.calendar)
            self._commit('sfc_input', sfc, ('vegfra', 'shdmin', 'shdmax', 'sfc_albbck'))
        elif operation == 'physics_update_sst':
            fields = ('isice', 'iswater', 'ivgtyp', 'isltyp', 'vegfra', 'sfc_albbck',
                      'sst', 'tmn', 'skintemp', 'tslb', 'sh2o', 'smois', 'snow',
                      'snowc', 'snowh', 'seaice', 'xice', 'xland')
            sfc = self._pool('sfc_input', fields)
            diag = self._pool('diag_physics', ('sfc_albedo', 'sfc_emiss', 'sfc_emibck', 'xicem'))
            mesh['nCells'] = sfc['sst'].size
            if self.read_surface_record is None:
                # MPAS manager reads the stream before this routine. Without it
                # a newly ringing SST alarm would silently reuse stale words.
                raise ValueError('SST alarm requires its Rust surface-record installer')
            self.read_surface_record(metadata['current_date'], sfc)
            physics_update_sst(options['config_frac_seaice'], mesh, sfc, diag,
                               xice_threshold=self._scalar('xice_threshold'))
            self._commit('sfc_input', sfc, fields[2:])
            self._commit('diag_physics', diag, tuple(diag))
        elif operation == 'physics_update_sstskin':
            sfc = self._pool('sfc_input', ('sst', 'skintemp', 'xland'))
            diag = self._pool('diag_physics', ('sstsk', 'sstsk_dtc', 'sstsk_dtw',
                              'sfc_emiss', 'glw', 'gsw', 'hfx', 'qfx', 'ust'))
            mesh['nCells'] = sfc['sst'].size
            physics_update_sstskin(call.dt, mesh, diag, sfc)
            self._commit('sfc_input', sfc, ('skintemp',))
            self._commit('diag_physics', diag, ('sstsk', 'sstsk_dtc', 'sstsk_dtw'))
        elif operation == 'physics_update_deepsoiltemp':
            sfc = self._pool('sfc_input', ('tmn', 'skintemp'))
            diag = self._pool('diag_physics', ('nsteps_accum', 'ndays_accum', 'tday_accum',
                              'tyear_accum', 'tyear_mean', 'tlag'))
            mesh['nCells'] = sfc['tmn'].size
            physics_update_deepsoiltemp(metadata['leapyear'], call.dt, metadata['julian_in'],
                                        mesh, sfc, diag)
            self._commit('sfc_input', sfc, ('tmn',))
            self._commit('diag_physics', diag, tuple(diag))
        elif operation == 'update_radiation_diagnostics':
            fields = (*RADIATION_FIELDS, *('ac'+k for k in RADIATION_FIELDS),
                      *('i_ac'+k for k in RADIATION_FIELDS))
            diag = self._pool('diag_physics', fields)
            mesh['nCells'] = diag[RADIATION_FIELDS[0]].size
            update_radiation_diagnostics(options, mesh, diag, self.session.arguments.its,
                                         self.session.arguments.ite, dt_dyn=call.dt,
                                         l_acradt=call.flags.radiation_bucket)
            self._commit('diag_physics', diag, fields[len(RADIATION_FIELDS):])
        else:
            diag = self._pool('diag_physics', ('cuprec', 'rainc', 'i_rainc'))
            mesh['nCells'] = diag['cuprec'].size
            update_precipitation_buckets(options, mesh, diag, self.session.arguments.its,
                                         self.session.arguments.ite, dt_dyn=call.dt,
                                         l_acrain=call.flags.rain_bucket,
                                         convection=True, microphysics=False)
            self._commit('diag_physics', diag, ('rainc', 'i_rainc'))
        self.executions[operation] += 1
        return self.receipt()

    def receipt(self):
        """mpas_atmphys_manager.F:231-253; mpas_atmphys_driver.F:209-365."""
        return dict(unit='p25-surface-update', backend='hex-cuda',
                    module_sha256=self.module_sha256, process_count=1,
                    executed=any(self.executions.values()), operations=dict(self.executions),
                    precipitation_owner=dict(rainc='surface-service', rainnc='native-microphysics'))

    def restart_state(self):
        """src/core_atmosphere/physics/mpas_atmphys_date_time.F:51-84."""
        return dict(february=self.calendar.february)

    def restore_restart_state(self, state):
        """src/core_atmosphere/physics/mpas_atmphys_date_time.F:51-84."""
        february = state['february']
        if february not in (28, 29):
            raise ValueError('invalid native saved February length')
        self.calendar = MonthlyCalendar(february=february)


class RustSurfaceRecordReader:
    """Install timed SST/ice words decoded by the supplied Rust bridge.

    mpas_atm_core.F:779-806; mpas_atmphys_manager.F:231-247;
    Registry.xml:3463-3479.
    Timestamp selection is clock metadata, using MPAS_STREAM_LATEST_BEFORE.
    NetCDF types, dimensions, cell ranges and float words are decoded in Rust.
    The caller binds a resolved input filename and its native cell extent;
    input publication/initialization pins remain owned by the forecast deck.
    """

    def __init__(self, library, path, *, n_cells, first_cell=0):
        """mpas_atm_core.F:779-806; mpas_stream_manager.F:3458-3460."""
        import ctypes as ct
        import os
        self.ct, self.path = ct, os.fsencode(path)
        self.n_cells, self.first_cell = n_cells, first_cell
        self.library = ct.CDLL(str(library))
        self.read_time = self.library.read_surface_time
        self.read_time.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
        self.read_time.restype = ct.c_int
        self.read_field = self.library.read_surface_field
        self.read_field.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_char_p,
                                   ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_size_t,
                                   ct.c_void_p, ct.c_size_t]
        self.read_field.restype = ct.c_int
        self.times = []
        record = 0
        while True:
            stamp = ct.create_string_buffer(512)
            length = self.read_time(self.path, record, stamp, len(stamp))
            if length == -10004:
                break
            if length <= 0:
                raise ValueError(f'Rust surface clock decoder failed: {length}')
            now = datetime.fromisoformat(bytes(stamp[:length]).decode('ascii').rstrip(' \0').replace('_', 'T'))
            if self.times and now <= self.times[-1]:
                raise ValueError('surface record clocks must increase for latest-before selection')
            self.times.append(now)
            record += 1
        if not self.times:
            raise ValueError('surface stream has no timed records')
        self.last_record = None
        self.reads = []
        self._host_record = np.empty((2, n_cells), dtype=np.float32)

    def __call__(self, current_date, pool):
        """mpas_atm_core.F:794-806; mpas_atmphys_update_surface.F:114-326."""
        from bisect import bisect_right
        now = datetime.fromisoformat(current_date.replace('_', 'T'))
        record = bisect_right(self.times, now)-1
        if record < 0:
            raise ValueError('surface stream has no record at or before the model clock')
        from time import perf_counter
        import cupy as cp
        started = perf_counter()
        for index, name in enumerate(('sst', 'xice')):
            target = pool[name]
            host = self._host_record[index]
            if target.dtype != np.float32 or target.ndim != 1 or target.size < self.n_cells:
                raise ValueError(f'surface {name} pool does not match its native cell extent')
            status = self.read_field(self.path, name.encode('ascii'), 5, None, 1,
                                     self.first_cell, self.n_cells, record,
                                     host.ctypes.data, host.size)
            if status:
                raise ValueError(f'Rust surface {name} decoder failed: {status}')
        decoded = perf_counter()
        # One contiguous H2D transfer per record; no existing pool D2H.
        # The device assignments preserve trailing native halo/padding words.
        device = cp.asarray(self._host_record)
        for index, name in enumerate(('sst', 'xice')):
            pool[name][:self.n_cells] = device[index]
        self.last_record = record
        self.reads.append(dict(record=record, time=self.times[record].isoformat(),
                               decode_seconds=decoded-started,
                               enqueue_seconds=perf_counter()-decoded,
                               h2d_bytes=self._host_record.nbytes,
                               h2d_transfers=1, d2h_bytes=0))
