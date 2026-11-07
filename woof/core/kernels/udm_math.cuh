// UDM uses native REAL kinds in both arithmetic modes. Strict replays
// retain the pinned CPU libm words; fast forecasts use CUDA's ordinary
// transcendental functions at the declared precision, as other fast
// radiation and physics units do. Integer powers keep their source tree.
#pragma once
#if GPUWM_WRF_EXACT
__device__ float udm_exp(float x) { return tc_exp(x); }
__device__ double udm_exp(double x) { return tc_exp(x); }
__device__ float udm_log(float x) { return tc_log(x); }
__device__ double udm_log(double x) { return tc_log(x); }
__device__ float udm_log10(float x) { return tc_log10(x); }
__device__ double udm_log10(double x) { return tc_log10(x); }
__device__ float udm_pow(float x,float y) { return tc_pow(x,y); }
__device__ double udm_pow(double x,double y) { return tc_pow(x,y); }
#else
__device__ float udm_exp(float x) { return expf(x); }
__device__ double udm_exp(double x) { return exp(x); }
__device__ float udm_log(float x) { return logf(x); }
__device__ double udm_log(double x) { return log(x); }
__device__ float udm_log10(float x) { return log10f(x); }
__device__ double udm_log10(double x) { return log10(x); }
__device__ float udm_pow(float x,float y) { return powf(x,y); }
__device__ double udm_pow(double x,double y) { return pow(x,y); }
#endif
template<class T> __device__ T udm_pow(T x,int n) { return tc_pow(x,n); }
template<class A,class B> __device__ auto udm_pow(A x,B y)->decltype(x+y) {
    using T=decltype(x+y);return udm_pow(T(x),T(y));
}
