"""Spectral stress roughness with a published high-wind drag envelope."""
from __future__ import annotations

CUDA = r"""
// Holthuijsen, Powell and Pietrzak (2012), JGR 117 C09003,
// doi:10.1029/2012JC007983, equation (1), Table 2 and paragraph 38.
// Piecewise-linear interpolation in directional spread is their prescription.
// Constant divisors are spelled __fdiv_rn: gpuwm.ocean.coupling compiles this
// unit as a RawModule, whose appended -ftz=true turns x/C into a rounded
// reciprocal multiply on compute_100/120 (A146).
__device__ float wave_drag_envelope(float wind,float spread){
 float low=spread<=50.f?__fdiv_rn(spread-30.f,20.f):__fdiv_rn(80.f-spread,30.f);
 float high=1.f-__fdiv_rn(fabsf(spread-50.f),5.f);
 low=fminf(1.f,fmaxf(0.f,low));high=fminf(1.f,fmaxf(0.f,high));
 float a=1.05f+(0.7f-1.05f)*low,b=1.25f+(1.1f-1.25f)*low;
 float c=1.4f+(6.f-1.4f)*low,d=2.3f+(8.2f-2.3f)*high;
 float e=10.f+(2.5f-10.f)*high;
 return .001f*fmaxf(.7f,fminf(a+b*powf(__fdiv_rn(wind,27.5f),c),d*(1.f-powf(__fdiv_rn(wind,54.f),e))));
}
extern "C" __global__ void spectral_roughness(const float* wave_z0,
 const float* spread,const float* u,const float* v,const float* ust,
 const float* wet,float* alpha,float* z0,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(wet[i]<.5f){alpha[i]=0.f;z0[i]=0.f;return;}
 float raw=wave_z0[i],w=hypotf(u[i],v[i]),a=fmaxf(ust[i],.001f);
 // Fail visibly at output collection instead of hiding invalid source output.
 if(!isfinite(raw)||raw<0.f||!isfinite(spread[i])){z0[i]=nanf("");alpha[i]=nanf("");return;}
 if(raw==0.f){z0[i]=0.f;alpha[i]=0.f;return;}
 float cd=wave_drag_envelope(w,spread[i]);
 // Neutral logarithmic inversion, z0 = 10 exp(-kappa/sqrt(Cd)).
 // Powell et al. (2003), Nature 422 279-283, doi:10.1038/nature01481;
 // Donelan et al. (2004), GRL 31 L18306, doi:10.1029/2004GL019460.
 // The varying envelope above represents saturation and subsequent decline;
 // it does not extrapolate the old bulk wave-age law into hurricane winds.
 float cap=10.f*expf(-.4f/sqrtf(cd));
 // Numerical blending joins the spectral solution to the high-wind envelope
 // over 20..27.5 m/s. These join endpoints are an explicit coupling choice,
 // not a new fit to observations. The scalar heat/moisture laws are untouched.
 float blend=fminf(1.f,fmaxf(0.f,__fdiv_rn(w-20.f,7.5f)));
 blend=blend*blend*(3.f-2.f*blend);
 float z=raw+blend*(fminf(raw,cap)-raw);
 z0[i]=z;alpha[i]=fmaxf(0.f,z-.11f*1.5e-5f/a)*9.80665f/(a*a);
}
"""
