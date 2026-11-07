// WOOF arithmetic bindings for the clean GFDL statement transcription.
#if defined(GD_HOST_DIAGNOSTIC)
#define GD_NOINLINE
#else
#define GD_NOINLINE __noinline__
#endif
#if defined(GD_HOST_DIAGNOSTIC)
#include <cmath>
#define GD_DEVICE
inline float gd_exp(float x){return std::exp(x);}
inline double gd_exp(double x){return std::exp(x);}
inline float gd_log(float x){return std::log(x);}
inline double gd_log(double x){return std::log(x);}
inline float gd_log10(float x){return std::log10(x);}
inline double gd_log10(double x){return std::log10(x);}
inline float gd_sqrt(float x){return std::sqrt(x);}
inline double gd_sqrt(double x){return std::sqrt(x);}
inline float gd_sin(float x){return std::sin(x);}
inline float gd_cos(float x){return std::cos(x);}
inline float gd_asin(float x){return std::asin(x);}
inline float gd_acos(float x){return std::acos(x);}
inline float gd_tan(float x){return std::tan(x);}
inline float gd_atan(float x){return std::atan(x);}
inline float gd_real_pow(float x,float y){return std::pow(x,y);}
inline double gd_double_pow(double x,double y){return std::pow(x,y);}
#elif defined(GPUWM_GFDLETA_STRICT)
#define GD_DEVICE __device__
__device__ inline float gd_exp(float x){return gfk_exp(x);}
__device__ inline double gd_exp(double x){return plm_exp(x);}
__device__ inline float gd_log(float x){return gfk_log(x);}
__device__ inline double gd_log(double x){return plm_log(x);}
// Keep the measured glibc 2.39 fixture and glibc 2.43 real-input bindings
// separate, selected by GPUWM_WRF_GLIBC through the source loader.
// The newer proof uses portable binary64 log before the REAL rounding.
__device__ inline float gd_log10(float x){
#if GPUWM_GFDLETA_GLIBC243
 return __double2float_rn(__dmul_rn(plm_log((double)x),0x1.bcb7b1526e50ep-2));
#else
 return gfk_log10(x);
#endif
}
__device__ inline double gd_log10(double x){return __dmul_rn(plm_log(x),0x1.bcb7b1526e50ep-2);}
__device__ inline float gd_sqrt(float x){return __fsqrt_rn(x);}
__device__ inline double gd_sqrt(double x){return __dsqrt_rn(x);}
__device__ inline float gd_sin(float x){return glibc_sinf(x);}
__device__ inline float gd_cos(float x){return glibc_cosf(x);}
__device__ inline float gd_asin(float x){return glibc_asinf(x);}
__device__ inline float gd_acos(float x){return glibc_acosf(x);}
__device__ inline float gd_tan(float x){return glibc_tanf(x);}
__device__ inline float gd_atan(float x){return glibc_atanf(x);}
__device__ inline float gd_real_pow(float x,float y){return gfk_pow(x,y);}
__device__ inline double gd_double_pow(double x,double y){return plm_pow(x,y);}
#else
#define GD_DEVICE __device__
__device__ inline float gd_exp(float x){return expf(x);}
__device__ inline double gd_exp(double x){return exp(x);}
__device__ inline float gd_log(float x){return logf(x);}
__device__ inline double gd_log(double x){return log(x);}
__device__ inline float gd_log10(float x){return log10f(x);}
__device__ inline double gd_log10(double x){return log10(x);}
__device__ inline float gd_sqrt(float x){return sqrtf(x);}
__device__ inline double gd_sqrt(double x){return sqrt(x);}
__device__ inline float gd_sin(float x){return sinf(x);}
__device__ inline float gd_cos(float x){return cosf(x);}
__device__ inline float gd_asin(float x){return asinf(x);}
__device__ inline float gd_acos(float x){return acosf(x);}
__device__ inline float gd_tan(float x){return tanf(x);}
__device__ inline float gd_atan(float x){return atanf(x);}
__device__ inline float gd_real_pow(float x,float y){return powf(x,y);}
__device__ inline double gd_double_pow(double x,double y){return pow(x,y);}

#endif

template<class T> GD_DEVICE inline T gd_min(T a,T b){return a<b?a:b;}
template<class T> GD_DEVICE inline T gd_max(T a,T b){return a>b?a:b;}
template<class T> GD_DEVICE inline T gd_abs(T a){return a<0?-a:a;}
GD_DEVICE inline float gd_abs(float a){
#ifdef GD_HOST_DIAGNOSTIC
 return std::fabs(a);
#else
 return __uint_as_float(__float_as_uint(a)&0x7fffffffu);
#endif
}
GD_DEVICE inline double gd_abs(double a){
#ifdef GD_HOST_DIAGNOSTIC
 return std::fabs(a);
#else
 return __longlong_as_double(__double_as_longlong(a)&0x7fffffffffffffffULL);
#endif
}
#define min(a,b) gd_min((a),(b))
#define max(a,b) gd_max((a),(b))
#define dmin(a,b) gd_min((a),(b))
#define dmax(a,b) gd_max((a),(b))
#define abs(a) gd_abs(a)
#define dabs(a) gd_abs(a)
GD_DEVICE inline int i_nint(float *x){return (int)((double)*x+(*x>=0?0.5:-0.5));}
GD_DEVICE inline float r_nint(float *x){return (float)i_nint(x);}
GD_DEVICE inline float r_mod(float *x,float *y){return fmodf(*x,*y);}
GD_DEVICE inline float r_sign(float *x,float *y){return *y>=0?gd_abs(*x):-gd_abs(*x);}
GD_DEVICE inline double r_lg10(float *x){return (double)gd_log10(*x);}
GD_DEVICE inline double gd_pow_double_ptr(double *x,double *y){return gd_double_pow(*x,*y);}
GD_DEVICE inline double pow_dd(double *x,double *y){return (double)gd_real_pow((float)*x,(float)*y);}
GD_DEVICE inline float pow_ri(float *x,int *n){
 int k=*n;float base=*x,result=1.0f;
 if(k<0){base=1.0f/base;k=-k;}
 while(k){if(k&1)result=result*base;k>>=1;if(k)base=base*base;}
 return result;
}

GD_DEVICE inline double r_int(float *x){return (double)(*x>=0?floorf(*x):ceilf(*x));}
