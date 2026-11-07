// Public WRF REAL/DOUBLE arithmetic adapters; see tools/nssl2_wrf461_variant_oracle/SOURCE_PIN.json.

namespace nssl1 {
__device__ __forceinline__ float nssl1_log10f(float x){
#ifdef GPUWM_WRF_EXACT
return __double2float_rn(log10((double)x));
#else
return log10f(x);
#endif
}

template<class T,int N> struct Storage;
template<class T,int N,int... D> struct FixedStorage;
template<class T> struct Array {
 T* data; int lo[4],size[4];
 template<class U> __device__ __forceinline__ Array(const Array<U>& o):data(o.data){for(int d=0;d<4;++d){lo[d]=o.lo[d];size[d]=o.size[d];}}
 template<class U,int N> __device__ __forceinline__ Array(Storage<U,N>& o):data(o.data){for(int d=0;d<4;++d){lo[d]=o.lo[d];size[d]=o.size[d];}}
 template<class U,int N,int... D> __device__ __forceinline__ Array(FixedStorage<U,N,D...>& o):data(o.data){for(int d=0;d<4;++d){lo[d]=o.lo[d];size[d]=o.size[d];}}
 template<class... D> __device__ __forceinline__ Array(T* p,D... ds):data(p) {int dims[]={int(ds)...};for(int i=0;i<sizeof...(D)/2;++i){lo[i]=dims[2*i];size[i]=dims[2*i+1]-lo[i]+1;}}
 template<class... I> __device__ __forceinline__ T& operator()(I... ix) const {int ind[]={int(ix)...};int off=0,mul=1;for(int d=0;d<sizeof...(I);++d){off+=(ind[d]-lo[d])*mul;mul*=size[d];}return data[off];}
};
// Owned storage has no self-pointer: unused native work arrays and metadata
// can die, and fixed point values can stay in registers after inlining.
template<class T,int N> struct Storage {
 T data[N>0?N:1]; int lo[4],size[4];
 template<class... D> __device__ __forceinline__ Storage(D... ds) {int dims[]={int(ds)...};for(int i=0;i<sizeof...(D)/2;++i){lo[i]=dims[2*i];size[i]=dims[2*i+1]-lo[i]+1;}}
 template<class... I> __device__ __forceinline__ T& operator()(I... ix) {int ind[]={int(ix)...};int off=0,mul=1;for(int d=0;d<sizeof...(I);++d){off+=(ind[d]-lo[d])*mul;mul*=size[d];}return data[off];}
};
template<class T,int N,int L0,int H0,int L1=0,int H1=0,int L2=0,int H2=0,int L3=0,int H3=0> struct StaticStorage {
 T data[N>0?N:1];
 static constexpr int lo[4]={L0,L1,L2,L3};
 static constexpr int size[4]={H0-L0+1,H1-L1+1,H2-L2+1,H3-L3+1};
 template<class... I> __device__ __forceinline__ T& operator()(I... ix) {int ind[]={int(ix)...};int off=0,mul=1;for(int d=0;d<sizeof...(I);++d){off+=(ind[d]-lo[d])*mul;mul*=size[d];}return data[off];}
};
template<class T,int N,int... D> struct FixedStorage:StaticStorage<T,N,D...> {};
__device__ __forceinline__ float lake_div(float a,float b){
#ifdef GPUWM_WRF_EXACT
return __fdiv_rn(a,b);
#else
return __fdividef(a,b);
#endif
}
__device__ __forceinline__ double lake_div(double a,double b){return __ddiv_rn(a,b);}
__device__ __forceinline__ double lake_div(double a,float b){return __ddiv_rn(a,(double)b);}
__device__ __forceinline__ double lake_div(float a,double b){return __ddiv_rn((double)a,b);}
__device__ __forceinline__ float lake_div(int a,float b){return __fdiv_rn((float)a,b);}
__device__ __forceinline__ float lake_div(float a,int b){return __fdiv_rn(a,(float)b);}
__device__ __forceinline__ double lake_div(int a,double b){return __ddiv_rn((double)a,b);}
__device__ __forceinline__ double lake_div(double a,int b){return __ddiv_rn(a,(double)b);}
__device__ __forceinline__ int lake_div(int a,int b){return a/b;}
__device__ __forceinline__ float lake_pow(float a,float b){return gfk_pow(a,b);}
__device__ __forceinline__ double lake_pow(double a,double b){return pow(a,b);}
__device__ __forceinline__ double lake_pow(float a,double b){return pow((double)a,b);}
__device__ __forceinline__ double lake_pow(double a,float b){return pow(a,(double)b);}
template<class T> __device__ __forceinline__ T lake_pow(T a,int b){if(b==6){T x3=(a*a)*a;return x3*x3;}if(b<0)return T(1)/lake_pow(a,-b);T out=T(1);while(b){if(b&1)out*=a;b>>=1;if(b)a*=a;}return out;}
template<class T> __device__ __forceinline__ T lake_abs(T a){return a<T(0)?-a:a;}
template<class T> __device__ __forceinline__ T lake_min(T a,T b){return a<b?a:b;}
template<class T,class... V> __device__ __forceinline__ T lake_min(T a,T b,V... v){return lake_min(lake_min(a,b),v...);}
template<class T> __device__ __forceinline__ T lake_max(T a,T b){return a>b?a:b;}
template<class T,class... V> __device__ __forceinline__ T lake_max(T a,T b,V... v){return lake_max(lake_max(a,b),v...);}
__device__ __forceinline__ double lake_min(float a,double b){return double(a)<b?double(a):b;}
__device__ __forceinline__ double lake_min(double a,float b){return a<double(b)?a:double(b);}
__device__ __forceinline__ double lake_max(float a,double b){return double(a)>b?double(a):b;}
__device__ __forceinline__ double lake_max(double a,float b){return a>double(b)?a:double(b);}
__device__ __forceinline__ float lake_atan(float a){return atanf(a);}
__device__ __forceinline__ float gamma_sp(float a){return nssl2_gamma_sp(a);}
__device__ __forceinline__ float gamxinf(float a,float b){return float(::gamxinfdp(a,b));}
__device__ __forceinline__ float gaminterp(float r,float a,int i,int h){if(h!=1){asm("trap;");return 0.f;}return ::gaminterp(r,a,i);}
}
