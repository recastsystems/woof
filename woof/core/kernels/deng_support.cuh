// Coalesced per-column storage and Fortran REAL32 intrinsics.

template<class T> struct DengArray {
 T* data;int stride,l1,h1,l2=1,h2=1,l3=1,h3=1;
 __device__ T& operator()(int i) const{return data[(i-l1)*stride];}
 __device__ T& operator()(int i,int j) const{return data[((i-l1)+(h1-l1+1)*(j-l2))*stride];}
 __device__ T& operator()(int i,int j,int k) const{return data[((i-l1)+(h1-l1+1)*((j-l2)+(h2-l2+1)*(k-l3)))*stride];}
};
struct DengArena {
 float* data;int stride,used,limit;int* status;
 template<class T> __device__ DengArray<T> make(int n,int l1,int h1,int l2=1,int h2=1,int l3=1,int h3=1){
  int offset=used;used+=n;
  if(used>limit){atomicMax(status,99999);asm("trap;");}
  return {reinterpret_cast<T*>(data+offset*stride),stride,l1,h1,l2,h2,l3,h3};
 }
};
struct DengFrame {
 DengArena& arena;int saved;
 __device__ DengFrame(DengArena& a):arena(a),saved(a.used){}
 __device__ ~DengFrame(){arena.used=saved;}
};
template<class A,class B> __device__ auto lake_min(A a,B b)->decltype(a+b){return a<b?a:b;}
template<class A,class B> __device__ auto lake_max(A a,B b)->decltype(a+b){return a>b?a:b;}
template<class A,class B,class... C> __device__ auto lake_min(A a,B b,C... c)->decltype(a+b){return lake_min(lake_min(a,b),c...);}
template<class A,class B,class... C> __device__ auto lake_max(A a,B b,C... c)->decltype(a+b){return lake_max(lake_max(a,b),c...);}
template<class T> __device__ T lake_abs(T x){return x<0?-x:x;}
__device__ int lake_div(int a,int b){return a/b;}
__device__ float lake_div(float a,float b){return __fdiv_rn(a,b);}
__device__ double lake_div(double a,double b){return __ddiv_rn(a,b);}
template<class A,class B> __device__ auto lake_div(A a,B b)->decltype(a+b){using T=decltype(a+b);return lake_div(T(a),T(b));}
template<class T> __device__ T lake_pow(T x,int n){
 bool neg=n<0;unsigned int k=neg?-n:n;T y=1;
 while(k){if(k&1)y=y*x;k>>=1;if(k)x=x*x;}return neg?lake_div(T(1),y):y;
}
// Native libm emulation is confined to strict replay. Forecasts use CUDA
// REAL32 intrinsics, so replay's internal double arithmetic is not a GPU cost.
#ifdef GPUWM_DENG_FAST
__device__ float lake_pow(float x,float y){return powf(x,y);}
__device__ float lake_exp(float x){return expf(x);}
__device__ float lake_log(float x){return logf(x);}
#else
__device__ float lake_pow(float x,float y){return gfk_pow(x,y);}
__device__ float lake_exp(float x){return gfk_exp(x);}
__device__ float lake_log(float x){return gfk_log(x);}
#endif
__device__ float deng_real(int x){return float(x);}
__device__ float lake_sin(float x){return sinf(x);}
__device__ float alog(float x){return lake_log(x);}
__device__ int ifix(float x){return int(x);}
