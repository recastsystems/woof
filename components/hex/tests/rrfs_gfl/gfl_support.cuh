
// Precision and column scratch support; no scheme algebra lives here.
#ifdef GF_HOST_WITNESS
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
// Host precision bridge for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
#define __device__
// Host precision bridge for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
#define __global__
inline float __fadd_rn(float a,float b){return a+b;}
inline float __fsub_rn(float a,float b){return a-b;}
inline float __fmul_rn(float a,float b){return a*b;}
inline float __fdiv_rn(float a,float b){return a/b;}
inline float __fsqrt_rn(float a){return sqrtf(a);}
inline float glibc_atanf(float a){return atanf(a);}
inline float glibc_sinf(float a){return sinf(a);}
inline float gfk_tgamma(float a){return tgammaf(a);}
inline float gfk_exp(float a){return expf(a);}
inline float gfk_log(float a){return logf(a);}
inline float ubm_log10f(float a){return log10f(a);}
inline float gfk_pow(float a,float b){return powf(a,b);}
#endif
struct GFReal {
  float value;
  __device__ GFReal() = default;
  template<class T> __device__ GFReal(T a): value((float)a) {}
  // Precision cast for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
  template<class T> __device__ explicit operator T() const {return (T)value;}
};
static_assert(sizeof(GFReal)==4, "RKIND must be binary32");
__device__ inline GFReal operator+(GFReal a,GFReal b){return __fadd_rn(a.value,b.value);}
__device__ inline GFReal operator-(GFReal a,GFReal b){return __fsub_rn(a.value,b.value);}
__device__ inline GFReal operator*(GFReal a,GFReal b){return __fmul_rn(a.value,b.value);}
__device__ inline GFReal operator/(GFReal a,GFReal b){return __fdiv_rn(a.value,b.value);}
__device__ inline GFReal operator-(GFReal a){return -a.value;}
__device__ inline bool operator<(GFReal a,GFReal b){return a.value<b.value;}
__device__ inline bool operator>(GFReal a,GFReal b){return a.value>b.value;}
__device__ inline bool operator<=(GFReal a,GFReal b){return a.value<=b.value;}
__device__ inline bool operator>=(GFReal a,GFReal b){return a.value>=b.value;}
__device__ inline bool operator==(GFReal a,GFReal b){return a.value==b.value;}
__device__ inline bool operator!=(GFReal a,GFReal b){return a.value!=b.value;}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
template<class A,class B> __device__ inline auto gf_max(A a,B b)->decltype(a+b){return b>a?decltype(a+b)(b):decltype(a+b)(a);}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
template<class A,class B> __device__ inline auto gf_min(A a,B b)->decltype(a+b){return b<a?decltype(a+b)(b):decltype(a+b)(a);}
// Native ABS_EXPR semantics, including -0: MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/module_cu_gfl_deep.F90:3389-3585
__device__ inline GFReal gf_abs(GFReal a){return fabsf(a.value);}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
template<class T> __device__ inline T gf_abs(T a){return a<T(0)?-a:a;}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline GFReal gf_exp(GFReal a){return gfk_exp(a.value);}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline GFReal gf_log(GFReal a){return gfk_log(a.value);}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline GFReal gf_log10(GFReal a){return ubm_log10f(a.value);}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline GFReal gf_pow(GFReal a,GFReal b){return gfk_pow(a.value,b.value);}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline GFReal gf_sqrt(GFReal a){return __fsqrt_rn(a.value);}
// GCC powi multiply chains. The tree lowers fixed small exponents into expressions.
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline GFReal gf_powi(GFReal a,int p){
 if(p==0)return GFReal(1);if(p==1)return a;if(p==2)return a*a;
 if(p==3)return (a*a)*a;if(p==4){GFReal b=a*a;return b*b;}
 GFReal result=1;unsigned int n=p<0?-p:p;
 while(n){if(n&1)result=result*a;n>>=1;if(n)a=a*a;}
 return p<0?GFReal(1)/result:result;
}
typedef char GFText[50];
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline void gf_memmove(void* dst,const void* src,unsigned long long count){
  char* d=(char*)dst;const char* s=(const char*)src;
  if(d<s){for(unsigned long long i=0;i<count;i++)d[i]=s[i];}
  else{for(unsigned long long i=count;i>0;i--)d[i-1]=s[i-1];}
}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline void gf_memset(void* dst,int val,unsigned long long n){for(unsigned long long i=0;i<n;i++)((char*)dst)[i]=(char)val;}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline int gf_compare_string(long long na,const char* a,long long nb,const char* b){
  for(long long i=0;i<gf_max(na,nb);i++){int x=i<na?(unsigned char)a[i]:32;int y=i<nb?(unsigned char)b[i]:32;if(x!=y)return x-y;}return 0;
}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline void gf_trap(){
#ifdef GF_HOST_WITNESS
  throw std::runtime_error("GF scratch capacity exceeded or native STOP reached");
#else
  asm("trap;");
#endif
}
struct GFWorkspace {
  char* data;unsigned long long capacity,position,highwater;unsigned int diagnostics;GFReal row_dx_first;
  // Mutable SAVE, including the early return: MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/module_cu_gfl_deep.F90:2641-2705
  bool fct1d3_error;
  // A column whose need passes its share of the launch scratch continues in one device-heap block of spill_bytes
  // (gfl_column frees it): the share is sized to what columns use, the spill keeps a deeper column exact and running.
  char* spill;unsigned long long spill_bytes;
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
  __device__ void* allocate(unsigned long long n){
    n=(n+15)&~15ULL;
    if(position+n>capacity){
#ifdef GF_HOST_WITNESS
      gf_trap();return nullptr;
#else
      if(position<capacity)position=capacity;
      if(spill==nullptr){spill=(char*)malloc(spill_bytes);if(spill==nullptr){gf_trap();return nullptr;}}
      if(position+n>capacity+spill_bytes){gf_trap();return nullptr;}
      void* result=spill+(position-capacity);position+=n;highwater=gf_max(highwater,position);return result;
#endif
    }
    void* result=data+position;position+=n;highwater=gf_max(highwater,position);return result;
  }
};
struct GFFrame {
 GFWorkspace &ws;unsigned long long saved;
 __device__ GFFrame(GFWorkspace &w):ws(w),saved(w.position){}
 __device__ ~GFFrame(){ws.position=saved;}
};
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline GFReal gf_infinity(){
#ifdef GF_HOST_WITNESS
 return INFINITY;
#else
 return __int_as_float(0x7f800000);
#endif
}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline GFReal gf_nan(){
#ifdef GF_HOST_WITNESS
 return NAN;
#else
 return __int_as_float(0x7fc00000);
#endif
}


struct GFDType {unsigned long long elem_len;int version; signed char rank,type;};
struct GFDim {long long stride,lbound,ubound;};
template<int N> struct GFDescriptor {void* data;long long offset;GFDType dtype;long long span;GFDim dim[N];};
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline GFReal gf_atan(GFReal a){return glibc_atanf(a.value);}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline GFReal gf_sin(GFReal a){return glibc_sinf(a.value);}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline GFReal gf_tgamma(GFReal a){return gfk_tgamma(a.value);}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
template<class T> __device__ inline T gf_expect(T value,int,long long){return value;}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline int gf_memcmp(const void* a,const void* b,unsigned long long n){
 const unsigned char* x=(const unsigned char*)a;const unsigned char* y=(const unsigned char*)b;
 for(unsigned long long i=0;i<n;i++)if(x[i]!=y[i])return (int)x[i]-(int)y[i];return 0;
}
// Typed CUDA support for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_common.F90:1-11
__device__ inline void gf_stop_string(const char*,long long,bool){gf_trap();}
