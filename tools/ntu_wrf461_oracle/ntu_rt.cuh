// ---------------------------------------------------------------------------
// NTU multi-moment microphysics (mp_physics = 56): device vocabulary.
//
// Every arithmetic operation in the transcription below is one IEEE binary32
// (or binary64 where WRF declares DOUBLE PRECISION) operation in WRF's own
// order.  The strict arm is compiled with --fmad=false --ftz=false so that a
// written a*b+c stays two roundings, exactly as gfortran -ffp-contract=off.
// The transcendental words are WOOF's own math routines (expf/logf/powf and
// the binary64 exp/log); gfortran lowers WRF's REAL**REAL to powf except for
// the exponents 2.0, 1.0 and -1.0, which it folds (ntu_sq, identity, 1/x).
// ---------------------------------------------------------------------------
#ifndef NTU_DEV
#define NTU_DEV __device__
#endif
// The four long WRF routines stay out of line so NVRTC compiles them once.
#ifndef NTU_DEV_BIG
#define NTU_DEV_BIG __device__ __noinline__
#endif
#ifndef NTU_CONST_TABLE
#define NTU_CONST_TABLE __device__ const
#endif
#ifndef NTU_KMAX
#define NTU_KMAX 64
#endif

// Status word bits (one per launch, OR-reduced): the column hit a WRF
// fatal-error branch (STOP / wrf_error_fatal) and its result is not WRF's.
#define NTU_ERR_STOP  1u
#define NTU_ERR_FATAL 2u
#define NTU_ERR_KMAX  4u

// WOOF's float32 log10 arithmetic, also used by the land-surface path.
// Keep each binary32 rounding explicit. A binary64 log rounded once is a
// different function at the oracle's float32 boundaries.
#ifndef NTU_HOST
// Native glibc 2.43 LOG10F rounds the logarithm once to binary32.
// The recovered split-float route produced 17 wrong real-capture words.
NTU_DEV float ntu_log10(float x) {
    return (float)__dmul_rn(glibc_log((double)x),0.43429448190325182765);
}
#endif

// WOOF ERF: exp(-x*x) times the positive integral recurrence.
// Two-word arithmetic preserves the recurrence at binary64 boundaries.
// The 2/sqrt(pi) split is computed from the Machin identity with
// Decimal precision 90; no external math source or runtime asset is used.
#ifndef NTU_HOST
struct NtuDD {double hi,lo;};
NTU_DEV NtuDD ntu_dd_norm(double a,double b){double h=__dadd_rn(a,b);return {h,__dsub_rn(b,__dsub_rn(h,a))};}
NTU_DEV NtuDD ntu_dd_add(NtuDD a,NtuDD b){double h=__dadd_rn(a.hi,b.hi),v=__dsub_rn(h,a.hi);double e=__dadd_rn(__dsub_rn(a.hi,__dsub_rn(h,v)),__dsub_rn(b.hi,v));e=__dadd_rn(e,__dadd_rn(a.lo,b.lo));return ntu_dd_norm(h,e);}
NTU_DEV NtuDD ntu_dd_mul(NtuDD a,NtuDD b){double h=__dmul_rn(a.hi,b.hi);double e=__fma_rn(a.hi,b.hi,-h);e=__dadd_rn(e,__dadd_rn(__dmul_rn(a.hi,b.lo),__dmul_rn(a.lo,b.hi)));return ntu_dd_norm(h,e);}
NTU_DEV NtuDD ntu_dd_div(NtuDD a,double b){double q=__ddiv_rn(a.hi,b);double r=__dadd_rn(__fma_rn(-q,b,a.hi),a.lo);return ntu_dd_norm(q,__ddiv_rn(r,b));}
NTU_DEV_BIG NtuDD ntu_dd_exp(NtuDD x){
 // ln(2) split and a short reduced Taylor series, retaining both words.
 const NtuDD ln2={0.6931471805599453,2.3190468138462996e-17};
 int k=(int)round(x.hi*1.4426950408889634);
 NtuDD r=ntu_dd_add(x,ntu_dd_mul({-(double)k,0},ln2));
 NtuDD term={1,0},sum=term;
 for(int n=1;n<64;++n){term=ntu_dd_div(ntu_dd_mul(term,r),(double)n);sum=ntu_dd_add(sum,term);if(fabs(term.hi)<fabs(sum.hi)*0x1p-110)break;}
 return {ldexp(sum.hi,k),ldexp(sum.lo,k)};
}
NTU_DEV_BIG double ntu_erf(double x){
 if(x!=x)return x+x;
 double a=x<0?-x:x;
 if(a>=6)return x<0?-1:1;
 if(a==0)return x;
 NtuDD y=ntu_dd_mul({a,0},{a,0}),z=ntu_dd_mul(y,{2,0}),term={a,0},sum=term;
 for(int n=1;n<240;++n){term=ntu_dd_div(ntu_dd_mul(term,z),(double)(2*n+1));sum=ntu_dd_add(sum,term);if(term.hi<sum.hi*0x1p-105)break;}
 NtuDD expx=ntu_dd_exp({-y.hi,-y.lo});
 NtuDD answer=ntu_dd_mul(ntu_dd_mul({1.1283791670955126,1.533545961316588e-17},expx),sum);
 double result=__dadd_rn(answer.hi,answer.lo);if(result>1)result=1;return x<0?-result:result;
}
NTU_DEV_BIG float ntu_erff(float x){return (float)ntu_erf((double)x);}
#endif

#ifndef NTU_HOST
__device__ unsigned int* ntu_status_word;
#define NTU_FATAL(code) do { if (ntu_status_word) atomicOr(ntu_status_word, (code)); } while (0)
#define NTU_EXPF(x)  gfk_exp(x)
#define NTU_LOGF(x)  gfk_log(x)
#define NTU_POWF(x, y) gfk_pow((x), (y))
#define NTU_EXP(x)   glibc_exp(x)
#define NTU_LOG(x)   glibc_log(x)
// ERF of a REAL retains the binary64-to-binary32 route. The oracle
// host float routine can differ by one ULP; the diagnostic records it.
#define NTU_LOG10F(x) ntu_log10(x)
#define NTU_ERFF(x)   ntu_erff(x)
#define NTU_ERF(x)    ntu_erf(x)
#define NTU_COSF(x)   glibc_cosf(x)
#define NTU_SINF(x)   glibc_sinf(x)
#define NTU_TANHF(x)  tanhf(x)
#endif

NTU_DEV unsigned int ntu_f2u(float f) {
    union { float f; unsigned int u; } v; v.f = f; return v.u;
}
NTU_DEV float ntu_bits_f(unsigned int u) {
    union { unsigned int u; float f; } v; v.u = u; return v.f;
}
NTU_DEV double ntu_bits_d(unsigned long long u) {
    union { unsigned long long u; double f; } v; v.u = u; return v.f;
}
// gfortran folds REAL**2.0 to one multiplication.
NTU_DEV float ntu_sq(float x) { return x * x; }
NTU_DEV double ntu_sqd(double x) { return x * x; }

// Fortran MIN/MAX: a NaN operand propagates; on a tie the larger zero wins
// for MAX and the smaller for MIN (gfortran -O2's IEEE minimum/maximum).
NTU_DEV float ntu_fmax(float a, float b) {
    if (a != a) return a;
    if (b != b) return b;
    if (a > b) return a;
    if (b > a) return b;
    return (ntu_f2u(a) & 0x80000000u) ? b : a;
}
NTU_DEV float ntu_fmin(float a, float b) {
    if (a != a) return a;
    if (b != b) return b;
    if (a < b) return a;
    if (b < a) return b;
    return (ntu_f2u(a) & 0x80000000u) ? a : b;
}
NTU_DEV double ntu_dmax(double a, double b) {
    if (a != a) return a;
    if (b != b) return b;
    return (a > b) ? a : ((b > a) ? b : (a == 0.0 && b == 0.0 && a < 1.0 / a ? b : a));
}
NTU_DEV double ntu_dmin(double a, double b) {
    if (a != a) return a;
    if (b != b) return b;
    return (a < b) ? a : ((b < a) ? b : a);
}
NTU_DEV int ntu_imax(int a, int b) { return a > b ? a : b; }
NTU_DEV int ntu_imin(int a, int b) { return a < b ? a : b; }
// NINT: round half away from zero.
NTU_DEV int ntu_nint(float x) { return (int)roundf(x); }
