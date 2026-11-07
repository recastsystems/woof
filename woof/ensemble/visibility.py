"""CUDA visibility diagnostic; equations and contract match rustwx-calc visibility.

Only orchestration lives here. A column thread computes all thermodynamics,
extinction and RH, borrowing up to three lowest mass levels of each carrier.
"""
from __future__ import annotations
import numpy as np
from woof.certify.kernel_manifest import record_module

SOURCE = r'''
__device__ bool good(double x) { return isfinite(x) && fabs(x)<1.e30; }
__device__ double rh(double p,double t,double r) {
    const double a[10]={.99999683,-.0090826951,.000078736169,-.00000061117958,
      4.3884187e-9,-2.9883885e-11,2.1874425e-13,-1.7892321e-15,1.1112018e-17,-3.0994571e-20};
    double x=t-273.15,pol=0.; for(int j=9;j>=0;--j) pol=pol*x+a[j];
    double es=6.1078/pow(pol,8.);
    return fmin(100.,fmax(0.,100.*(p/100.*r/(.62197+r))/es));
}
extern "C" __global__ void visibility_column(
    const float *p,const float *pb,const float *theta,const float *base,
    const float *qv,const float *qc,const float *qr,const float *qi,
    const float *qs,const float *qg,float *vis,float *flag,
    unsigned long long cells,int nz,int mode,int base_profile,int mask) {
    unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=cells) return;
    const float *species[5]={qc,qr,qi,qs,qg};
    double mq[5]={0.,0.,0.,0.,0.},maxrh=0.,p0=0.,t0=0.,r0=0.;
    bool ok=true;
    for(int k=0;k<nz;++k) {
        unsigned long long j=k*cells+i;
        double pr=p[j]+(mode==0 ? pb[j] : 0.);
        double th=theta[j]+(mode==0 || mode==2 ? 300. : base[base_profile ? k : j]);
        double t=th*pow(pr/100000.,0.2857142857),r=qv[j];
        if(!good(pr)||!good(t)||!good(r)||pr<=0.||t<150.||t>350.||r<0.) ok=false;
        if(k==0) {p0=pr;t0=t;r0=r;}
        if(k<2) maxrh=fmax(maxrh,rh(pr,t,r));
        for(int s=0;s<5;++s) if(mask&(1<<s)) {
            double q=species[s][j]; if(!good(q)) ok=false;
            mq[s]=fmax(mq[s],fmax(0.,q));
        }
    }
    if(!ok) {vis[i]=__int_as_float(0x7fc00000);flag[i]=__int_as_float(0x7fc00000);return;}
    double tv=t0*(1.+r0/.62197)/(1.+r0),rho=p0/(287.*tv);
    double volume=(1.+r0)/rho+(mq[0]+mq[1])/1000.+(mq[2]+mq[3]+mq[4])/917.;
    double c[5];for(int s=0;s<5;++s)c[s]=1000.*mq[s]/volume;
    double beta=144.7*pow(c[0],.88)+2.24*pow(c[1],.75)+327.8*c[2]
      +10.36*pow(c[3],.7776)+8.*pow(c[4],.75)+1.e-10;
    double haze=90000.*exp(-2.5*fmin(.8,fmax(0.,maxrh/100.-.15)));
    double v=fmin(90000.,fmin(1000.*3.912023005428146/beta,haze));
    vis[i]=(float)v;flag[i]=v<1000. ? 1.f : 0.f;
}
'''
_KERNELS = {}


INPUT_NAMES = ("p", "pb", "theta", "base", "qv", "qc", "qr", "qi", "qs", "qg")


def diagnose(*, state, fields, streamed, visibility, fog_flag, workspace, array_module):
    """Fill preallocated surfaces. Missing required carriers return False.

    Mode 0 consumes WRF T/P/PB, mode 1 native thp+thb/full pressure, mode 2
    the exact-WRF resident theta perturbation/full pressure. No volume math
    or reduction takes place in Python. Absent optional species are zero.
    """
    xp = array_module
    if streamed:
        names = ("P", "PB", "T", "QVAPOR", "QCLOUD")
        if any(name not in fields for name in names):
            return False
        p, pb, theta, vapor, cloud = (fields[name] for name in names)
        base, mode = theta, 0
        species = [cloud] + [fields.get(name) for name in ("QRAIN", "QICE", "QSNOW", "QGRAUP")]
    else:
        p, theta, base, vapor, cloud = (getattr(state, name, None) for name in ("p", "thp", "thb", "qv", "qc"))
        if any(a is None for a in (p, theta, base, vapor, cloud)):
            return False
        pb, mode = p, 2 if getattr(state, "_WRF_EXACT", False) else 1
        if getattr(state, "use_theta_m", 0):
            theta = state.total_theta() - xp.float32(300.0)
            mode = 2
        species = [cloud] + [getattr(state, name, None) for name in ("qr", "qi", "qs", "qg")]
    shape = tuple(visibility.shape)
    if len(p.shape) != 3 or tuple(p.shape[1:]) != shape or p.shape[0] < 1:
        raise ValueError("visibility needs mass-level pressure on the output grid")
    nz = min(3, p.shape[0])
    def copy(a, name, *, profile=False):
        target = workspace[f"headline:visibility:{name}"]
        target = target.reshape(-1)[:nz] if profile else target[:nz]
        source = a[:nz]
        if isinstance(source, np.ndarray):
            target.set(np.ascontiguousarray(source, dtype=np.float32))
        else:
            xp.copyto(target, source)
        return target
    def volume(a, name):
        if tuple(a.shape) != tuple(p.shape):
            raise ValueError("visibility mass-level carriers must share one grid")
        return copy(a, name)
    pressure, perturb, temperature, water = [volume(a, name) for a, name in
        zip((p, pb, theta, vapor), ("p", "pb", "theta", "qv"))]
    profile = len(base.shape) == 1
    if mode != 1:
        background = temperature
        profile = False
    elif profile:
        if base.shape[0] != p.shape[0]:
            raise ValueError("visibility theta base profile has wrong levels")
        background = copy(base, "base", profile=True)
    else:
        background = volume(base, "base")
    mask = sum(1 << s for s, a in enumerate(species) if a is not None)
    hydro = [volume(a, name) if a is not None else water for a, name in
             zip(species, ("qc", "qr", "qi", "qs", "qg"))]
    device = int(xp.cuda.runtime.getDevice())
    if device not in _KERNELS:
        options = ("--std=c++17", "--fmad=false")
        module = xp.RawModule(code=SOURCE, options=options)
        _KERNELS[device] = module.get_function("visibility_column")
        record_module("woof.ensemble.visibility", source=SOURCE, options=options, module=module)
    cells = int(visibility.size)
    _KERNELS[device](((cells + 255) // 256,), (256,),
        (pressure, perturb, temperature, background, water, *hydro, visibility, fog_flag,
         np.uint64(cells), np.int32(nz), np.int32(mode), np.int32(profile), np.int32(mask)))
    return True
