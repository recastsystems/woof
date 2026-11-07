"""GPU upper-ocean heat content, D26 and barrier-layer diagnostics.

OHC is the surface-connected heat integral above 26 C, in kJ cm-2.
References: doi:10.1175/2007WAF2006111.1; doi:10.1029/2004JC002378.
Uses supplied model density or the caller's linear equation of state.
This is explicitly not a TEOS-10 calculation.
No upstream implementation was copied. Numerical work is a CUDA kernel.
"""
from functools import lru_cache
import math

_SOURCE = r'''
__device__ double location(const double* z,int k,int samples) {
 return samples?z[k]:0.5*(z[k]+(k?z[k-1]:0));
}
extern "C" __global__ void tc_ocean_diagnostics(
 const double* t,const double* s,const double* density_profile,const double* bottom,const unsigned char* wet,
 float* ohc,float* d26,float* ild,float* mld,float* barrier,float* censored,
 int n,int nz,int use_density,int samples,double rho,double heat,double alpha,double beta,double ref,double delta,double drho) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 ohc[c]=d26[c]=ild[c]=mld[c]=barrier[c]=censored[c]=nanf("");
 if(!wet[c])return;
 const double *T=t+(long long)c*nz,*S=s+(long long)c*nz,*B=bottom+(long long)c*nz;
 const double *R=density_profile+(long long)c*nz;
 double prev=samples?-1:0;
 for(int k=0;k<nz;k++) {
  if(!isfinite(T[k])||!isfinite(S[k])||!isfinite(B[k])||B[k]<0||B[k]<=prev||
     T[k]<260||T[k]>330||S[k]<0||S[k]>50||
     (use_density&&(!isfinite(R[k])||R[k]<950||R[k]>1100)))return;
  prev=B[k];
 }
 // Piecewise-linear reconstruction at layer centres, constant to surface/bottom.
 // The first downward 26 C crossing defines the surface-connected warm layer.
 double heat_int=0,zlast=0,tlast=T[0],cross=0;
 bool warm=T[0]>299.15,found=!warm;
 if(warm) for(int k=0;k<=nz;k++) {
  double zk=k==nz?B[nz-1]:location(B,k,samples);
  double tk=k==nz?T[nz-1]:T[k];
  if(tk<=299.15) {
   cross=zlast+(zk-zlast)*(tlast-299.15)/(tlast-tk);
   heat_int+=0.5*(tlast-299.15)*(cross-zlast);found=true;break;
  }
  heat_int+=0.5*(tlast+tk-2*299.15)*(zk-zlast);
  zlast=zk;tlast=tk;
 }
 // A warm bottom gives a lower bound; do not label its depth a measured D26.
 ohc[c]=float(rho*heat*heat_int/1.e7);
 d26[c]=found?float(cross):nanf("");censored[c]=found?0.f:1.f;
 if(B[nz-1]<ref)return;
 double tr=T[0],sr=S[0],rr=use_density?R[0]:0;
 for(int k=1;k<nz;k++) {
  double zl=location(B,k-1,samples),zh=location(B,k,samples);
  if(ref>=zl&&ref<=zh) {double a=(ref-zl)/(zh-zl);tr=T[k-1]+a*(T[k]-T[k-1]);sr=S[k-1]+a*(S[k]-S[k-1]);if(use_density)rr=R[k-1]+a*(R[k]-R[k-1]);break;}
  if(ref>zh) {tr=T[k];sr=S[k];if(use_density)rr=R[k];}
 }
 double it=nan(""),mt=nan(""),zp=ref,dtp=0,drp=0;
 for(int k=0;k<=nz;k++) {
  double zz=k==nz?B[nz-1]:location(B,k,samples);if(zz<=ref)continue;
  int j=k==nz?nz-1:k;
  double dt=tr-T[j],dr=use_density?R[j]-rr:rho*(alpha*dt+beta*(S[j]-sr));
  if(!isfinite(it)&&dt>=delta)it=zp+(zz-zp)*(delta-dtp)/(dt-dtp);
  if(!isfinite(mt)&&dr>=drho)mt=zp+(zz-zp)*(drho-drp)/(dr-drp);
  zp=zz;dtp=dt;drp=dr;
 }
 ild[c]=float(it);mld[c]=float(mt);
 barrier[c]=isfinite(it)&&isfinite(mt)?float(fmax(0.,it-mt)):nanf("");
}
'''


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    from woof.certify.kernel_manifest import record_module
    module=cp.RawModule(code=_SOURCE,options=('--std=c++17',))
    record_module('woof.ocean.tc_diagnostics',source=_SOURCE,options=('--std=c++17',),module=module)
    return module.get_function('tc_ocean_diagnostics')


def diagnose(temperature, salinity, layer_bottom_m, wet, *, density=1025.,
             heat_capacity=3985., thermal_expansion=2.e-4, haline_contraction=7.6e-4,
             reference_depth_m=10., temperature_threshold_k=0.2,
             density_profile=None, density_threshold_kg_m3=0.03, coordinate_kind="layer_bottom"):
    """Arrays have (...,level) layout, top to bottom, T in K and S in PSU.

    With coordinate_kind="layer_bottom", values are reconstructed at layer
    centres. With "sample_depth", the supplied coordinate gives the actual
    sample depths and the final sample is the integration bottom. Values
    are constant from the surface to the first sample. No remapping occurs. OHC is a lower
    bound if D26_CENSORED=1; D26 is missing then. MLD and ILD are missing if
    their thresholds are not reached in the available column. No per-step
    work, CPU transforms, or synchronization: call only at output cadence.
    MLD uses a 0.03 kg m-3 density increase from 10 m by default. Supply the
    model's density when available; otherwise its linear EOS constants apply.
    """
    import cupy as cp
    import numpy as np
    if coordinate_kind not in ('layer_bottom','sample_depth'):
        raise ValueError('coordinate_kind must be layer_bottom or sample_depth')
    scalars=(density,heat_capacity,thermal_expansion,haline_contraction,
             reference_depth_m,temperature_threshold_k,density_threshold_kg_m3)
    if not all(math.isfinite(v) and v>0 for v in scalars):
        raise ValueError('ocean diagnostic constants must be finite and positive')
    t,s=(cp.ascontiguousarray(v,dtype=cp.float64) for v in (temperature,salinity))
    if t.shape!=s.shape or t.ndim<2 or t.shape[-1]<2:
        raise ValueError('ocean T/S need matching (...,>=2 levels) arrays')
    z=cp.ascontiguousarray(cp.broadcast_to(cp.asarray(layer_bottom_m),t.shape),dtype=cp.float64)
    r=t if density_profile is None else cp.ascontiguousarray(density_profile,dtype=cp.float64)
    if r.shape!=t.shape:raise ValueError('density profile must match ocean T/S shape')
    mask=cp.ascontiguousarray(wet,dtype=cp.uint8)
    if mask.shape!=t.shape[:-1]:raise ValueError('wet mask must match ocean column dimensions')
    names=('OCEAN_OHC26','OCEAN_D26','OCEAN_ILD','OCEAN_MLD','OCEAN_BARRIER_LAYER','OCEAN_D26_CENSORED')
    out={name:cp.empty(mask.shape,dtype=cp.float32) for name in names}
    _kernel()(((mask.size+127)//128,),(128,),
        (t,s,r,z,mask,*out.values(),np.int32(mask.size),np.int32(t.shape[-1]),np.int32(density_profile is not None),np.int32(coordinate_kind=="sample_depth"),
         *(np.float64(v) for v in scalars)))
    return out


def arwen_fields(fields):
    """Output-cadence views from WOOF's live ocean state."""
    import cupy as cp
    wet=(fields.get('_pwp_land',fields['xland'])>1.5)&(fields.get('lakemask',0)<.5)
    return diagnose(cp.moveaxis(fields['om_tmp'],0,-1),cp.moveaxis(fields['om_s'],0,-1),
                    cp.moveaxis(fields['om_depth'],0,-1),wet,
                    density_profile=cp.moveaxis(fields['om_density'],0,-1))
