"""Build-only host-glibc transcription witness; never a forecast backend."""
from pathlib import Path
from woof.hex.mpasphys.rrfs.tempo_v303 import _COLUMN, _DRIVER, _MPAS
math='''#include <cmath>\nusing namespace std;\n#define __device__\n#define __forceinline__ inline\n#define __global__\n#define mpas_expf expf\n#define mpas_exp exp\n#define mpas_logf logf\n#define mpas_log log\n#define mpas_log10f log10f\n#define mpas_log10 log10\n#define mpas_powf powf\n#define mpas_pow pow\n#define mpas_tgammaf tgammaf\n#define mpas_tgamma tgamma\n#define mpas_cosf cosf\n#define mpas_sin sin\n#define mpas_cos cos\n#define mpas_hypot hypot\n#define mpas_atan2 atan2\n'''
math+='\n#define __fmaf_rn fmaf\n'
# Whole-module math tape for diagnosing device/host differences. Numerical
# transport stays in Rust; this tape records the actual transcription calls.
math+=r"""
#ifdef TEMPO_MATH_AUDIT
#include <vector>
static std::vector<double> math_tape;
static double record_math(int fn,double x,double y,double result) {
 math_tape.insert(math_tape.end(),{double(fn),x,y,result});return result;
}
float audit_powf(float x,float y){return record_math(0,x,y,powf(x,y));}
float audit_expf(float x){return record_math(1,x,0,expf(x));}
float audit_logf(float x){return record_math(2,x,0,logf(x));}
float audit_log10f(float x){return record_math(3,x,0,log10f(x));}
double audit_pow(double x,double y){return record_math(4,x,y,pow(x,y));}
double audit_exp(double x){return record_math(5,x,0,exp(x));}
double audit_log(double x){return record_math(6,x,0,log(x));}
double audit_log10(double x){return record_math(7,x,0,log10(x));}
float audit_cosf(float x){return record_math(8,x,0,cosf(x));}
double audit_hypot(double x,double y){return record_math(9,x,y,hypot(x,y));}
double audit_atan2(double x,double y){return record_math(10,x,y,atan2(x,y));}
double audit_sin(double x){return record_math(11,x,0,sin(x));}
double audit_cos(double x){return record_math(12,x,0,cos(x));}
#undef mpas_powf
#undef mpas_expf
#undef mpas_logf
#undef mpas_log10f
#undef mpas_pow
#undef mpas_exp
#undef mpas_log
#undef mpas_log10
#define mpas_powf audit_powf
#define mpas_expf audit_expf
#define mpas_logf audit_logf
#define mpas_log10f audit_log10f
#define mpas_pow audit_pow
#define mpas_exp audit_exp
#define mpas_log audit_log
#define mpas_log10 audit_log10
#undef mpas_cosf
#undef mpas_hypot
#undef mpas_atan2
#undef mpas_sin
#undef mpas_cos
#define mpas_cosf audit_cosf
#define mpas_hypot audit_hypot
#define mpas_atan2 audit_atan2
#define mpas_sin audit_sin
#define mpas_cos audit_cos
extern "C" const double* cpp_math_tape(){return math_tape.data();}
extern "C" unsigned long long cpp_math_count(){return math_tape.size();}
#endif
"""
call=_DRIVER.split('// RRFS src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:74-825.')[0]
export='''extern "C" void cpp_step(int nz,float dt,float* a,const float* bl,float* diag,const double* blob,const int* flags,int first,int presence) {TempoState s={};TempoWork work={};for(int f=0;f<17;f++)for(int k=1;k<=nz;k++)s.f[f][k]=a[f*nz+k-1];tempo_call(nz,dt,s,work,(const TempoBlob*)blob,flags,first,bl,diag,presence);for(int f=0;f<17;f++)for(int k=1;k<=nz;k++)a[f*nz+k-1]=s.f[f][k];}\n'''
Path('scratch/tempo-build/host.cpp').write_text(math+_COLUMN+call+export)

ml_export=r"""
extern "C" void cpp_ml(int nz,const float* a,float* stages) {
 for(int k=0;k<nz;++k) {
  float trans[7],layer[24];
  for(int i=0;i<7;++i)stages[i*nz+k]=trans[i]=(a[i*nz+k]-tempo_nc_ml_trans_mean(i))/sqrt(tempo_nc_ml_trans_var(i));
  for(int n=0;n<24;++n){float value=0.0f;for(int i=0;i<7;++i)value=value+tempo_nc_ml_w00(n+24*i)*trans[i];stages[(n+7)*nz+k]=layer[n]=tmax(value+tempo_nc_ml_b00(n),0.0f);}
  float value=0.0f;for(int n=0;n<24;++n)value=value+tempo_nc_ml_w01(n)*layer[n];stages[31*nz+k]=tmax(value+tempo_nc_ml_b01(0),0.0f);
  stages[32*nz+k]=tempo_cloud_number(a[k],a[nz+k],a[2*nz+k],a[3*nz+k],a[4*nz+k],a[5*nz+k],a[6*nz+k]);
 }
}
"""
with Path('scratch/tempo-build/host.cpp').open('a') as stream:stream.write(ml_export)

glue_export=r"""
struct TempoGrid {int x;};TempoGrid blockIdx={0},blockDim={1},threadIdx={0};
inline float __fadd_rn(float a,float b){return a+b;}
inline float __fmul_rn(float a,float b){return a*b;}
inline float __fdiv_rn(float a,float b){return a/b;}
"""+_MPAS+r"""
extern "C" void cpp_glue_from(float** m,float** mesh,float** d,float** t,float** f,int nz,int nc,int aero,int hail,const float* constants) {
 for(threadIdx.x=0;threadIdx.x<nz*nc;++threadIdx.x)tempo_from(m,mesh,d,t,f,nz,nc,aero,hail,constants[0],constants[1]);
}
extern "C" void cpp_glue_to(float** m,float** mesh,float** d,float** t,float** f,int nz,int nc,int aero,int hail,float dt,const float* c) {
 for(threadIdx.x=0;threadIdx.x<nc;++threadIdx.x)tempo_to(m,mesh,d,t,f,nz,nc,aero,hail,dt,c[0],c[1],c[2],c[3],c[4],c[5]);
}
extern "C" void cpp_init_aerosols(float* nwfa,float* nifa,float* nwfa2d,float* nifa2d,const float* z,const float* area,int nz,int nc,int initw,int initi,const double* blob) {
 for(threadIdx.x=0;threadIdx.x<nc;++threadIdx.x)tempo_init_aerosols(nwfa,nifa,nwfa2d,nifa2d,z,area,nz,nc,initw,initi,(const TempoBlob*)blob);
}
"""
with Path('scratch/tempo-build/host.cpp').open('a') as stream:stream.write(glue_export)

driver_export=r"""
extern "C" void cpp_driver(float** f,float** s,float** d,float** p,float** r,float* rh,const float* out,int* counter,int nz,int nc,int phase,int full,int hourly,int accrue,float bucket) {
 for(threadIdx.x=0;threadIdx.x<nc;++threadIdx.x){
 if(phase==0)tempo_precip(f,s,d,counter,nz,nc,0,accrue,bucket);
 if(phase==1)tempo_step_diags(out,s,p,nz,nc);
 if(phase==2)tempo_precip(f,s,d,counter,nz,nc,1,accrue,bucket);
 if(phase==3)tempo_radar(f,p,r,rh,nz,nc,hourly,full,nullptr);
 }
}
"""
with Path('scratch/tempo-build/host.cpp').open('a') as stream:stream.write(driver_export)
